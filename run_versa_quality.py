#!/usr/bin/env python3
"""
run_versa_quality.py
--------------------
Run VERSA audio-quality metrics over one ASR dataset (one language) and
produce both per-utterance scores and a dataset-level summary.

Input:
    A CSV file (Kaldi-scp-like) where each row has at least
        <full_path_to_wav>,<transcript>
    Audio is assumed to be mono 16 kHz, 16-bit WAV.

Output (under --out_dir):
    wav.scp                        VERSA-format SCP (utt_id <tab> path)
    text                           VERSA-format transcripts (utt_id <tab> text)
    versa_raw.jsonl                Per-utterance metric output from VERSA
    per_utterance.csv              Per-utterance scores merged with original
                                   CSV (audio_path, transcript, all metrics)
    dataset_summary.json           Per-metric mean/median/std/min/max/count
                                   for the whole dataset
    versa_stdout.log / versa_stderr.log

Example:
    python run_versa_quality.py \\
        --csv  /data/quechua_train.csv \\
        --dataset_name quechua_train \\
        --out_dir   ./quality_results/quechua_train \\
        --versa_repo /opt/versa \\
        --score_config ./audio_quality_config.yaml \\
        --use_gpu

Run once per dataset; loop externally over your ~50 datasets.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("run_versa_quality")


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_csv(csv_path, delimiter, has_header, audio_col, text_col):
    """
    Read the user's CSV. Returns a list of (audio_path, transcript) tuples
    in original order.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for i, row in enumerate(reader, start=1):
            if not row or all(c.strip() == "" for c in row):
                continue
            if max(audio_col, text_col) >= len(row):
                raise ValueError(
                    f"Row {i} has {len(row)} fields; needs at least "
                    f"{max(audio_col, text_col) + 1}. Row: {row!r}"
                )
            audio = row[audio_col].strip()
            text = row[text_col].strip()
            rows.append((audio, text))
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")
    return rows


def make_utt_id(audio_path, used_ids):
    """
    Build a unique, whitespace-free utterance ID from the file path.
    VERSA's SCP parser splits on whitespace with maxsplit=1, so utt IDs
    must contain no whitespace. We prefer a human-readable basename, and
    fall back to a hash suffix on collision.
    """
    stem = Path(audio_path).stem
    # Replace anything that isn't alnum / dash / underscore / dot
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)
    if not safe:
        safe = "utt"
    candidate = safe
    if candidate in used_ids:
        h = hashlib.md5(audio_path.encode("utf-8")).hexdigest()[:8]
        candidate = f"{safe}_{h}"
        # Extremely unlikely, but bulletproof against the bulletproof case.
        n = 1
        while candidate in used_ids:
            candidate = f"{safe}_{h}_{n}"
            n += 1
    used_ids.add(candidate)
    return candidate


def write_versa_inputs(rows, out_dir):
    """
    Write wav.scp and text files in VERSA format. Returns a list of
    (utt_id, audio_path, transcript) parallel to `rows`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used = set()
    indexed = []
    bad = []
    for audio, text in rows:
        if not audio:
            bad.append(("empty path", audio, text))
            continue
        if not Path(audio).is_file():
            bad.append(("missing file", audio, text))
            continue
        utt = make_utt_id(audio, used)
        # text in versa is "key value" with a single split, so newlines/tabs
        # inside the transcript would corrupt the file. Flatten whitespace.
        flat_text = " ".join(text.split()) if text else ""
        indexed.append((utt, audio, flat_text))

    if bad:
        LOG.warning("Skipped %d row(s) with missing/empty audio paths.", len(bad))
        # Persist the skip list for transparency.
        with open(out_dir / "skipped_rows.tsv", "w", encoding="utf-8") as fh:
            fh.write("reason\taudio_path\ttranscript\n")
            for reason, a, t in bad:
                fh.write(f"{reason}\t{a}\t{t}\n")

    if not indexed:
        raise RuntimeError("No usable audio files after filtering.")

    with open(out_dir / "wav.scp", "w", encoding="utf-8") as fh:
        for utt, audio, _ in indexed:
            fh.write(f"{utt}\t{audio}\n")

    with open(out_dir / "text", "w", encoding="utf-8") as fh:
        for utt, _, txt in indexed:
            fh.write(f"{utt}\t{txt}\n")

    return indexed


# ---------------------------------------------------------------------------
# VERSA invocation
# ---------------------------------------------------------------------------

def run_versa(versa_repo, score_config, out_dir, use_gpu, pass_text):
    """
    Invoke versa/bin/scorer.py as a subprocess. We use --io soundfile
    because our SCP entries point to plain WAV files on disk.
    """
    versa_repo = Path(versa_repo).resolve()
    scorer = versa_repo / "versa" / "bin" / "scorer.py"
    if not scorer.is_file():
        raise FileNotFoundError(f"scorer.py not found at {scorer}")

    out_dir = Path(out_dir).resolve()
    raw_out = out_dir / "versa_raw.jsonl"
    cmd = [
        sys.executable, str(scorer),
        "--score_config", str(Path(score_config).resolve()),
        "--pred", str(out_dir / "wav.scp"),
        "--gt", "None",
        "--output_file", str(raw_out),
        "--io", "soundfile",
        "--use_gpu", "True" if use_gpu else "False",
    ]
    if pass_text:
        cmd += ["--text", str(out_dir / "text")]

    LOG.info("Running VERSA: %s", " ".join(cmd))
    stdout_log = out_dir / "versa_stdout.log"
    stderr_log = out_dir / "versa_stderr.log"
    with open(stdout_log, "w") as so, open(stderr_log, "w") as se:
        # cwd=versa_repo so that relative paths inside the YAML (e.g.
        # ./tools/NISQA/weights/nisqa.tar) resolve correctly.
        rc = subprocess.call(cmd, stdout=so, stderr=se, cwd=str(versa_repo))
    if rc != 0:
        raise RuntimeError(
            f"VERSA scorer.py exited with code {rc}. "
            f"See {stderr_log} for details."
        )
    if not raw_out.is_file():
        raise RuntimeError(f"VERSA finished but no output at {raw_out}")
    return raw_out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_versa_results(jsonl_path):
    """Read VERSA's JSONL output. One dict per utterance keyed by 'key'."""
    results = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key")
            if key is None:
                continue
            results[key] = obj
    return results


def is_numeric(v):
    return isinstance(v, (int, float)) and not (
        isinstance(v, float) and (math.isnan(v) or math.isinf(v))
    )


def write_per_utterance_csv(indexed, versa_results, out_path):
    """
    Merge the original (utt_id, audio_path, transcript) with VERSA's numeric
    metrics into one wide CSV, one row per utterance.
    """
    # Determine the union of numeric metric keys across all utterances.
    metric_keys = set()
    for k, obj in versa_results.items():
        for mk, mv in obj.items():
            if mk == "key":
                continue
            if is_numeric(mv):
                metric_keys.add(mk)
    metric_keys = sorted(metric_keys)

    header = ["utt_id", "audio_path", "transcript"] + metric_keys
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for utt, audio, txt in indexed:
            obj = versa_results.get(utt, {})
            row = [utt, audio, txt]
            for mk in metric_keys:
                v = obj.get(mk)
                row.append("" if v is None else v)
            w.writerow(row)
    return metric_keys


def summarize(versa_results, metric_keys, dataset_name, n_total, n_scored):
    """Compute mean/median/std/min/max/count for each numeric metric."""
    summary = {
        "dataset": dataset_name,
        "n_utterances_input": n_total,
        "n_utterances_scored": n_scored,
        "metrics": {},
    }
    for mk in metric_keys:
        vals = [
            obj[mk] for obj in versa_results.values()
            if mk in obj and is_numeric(obj[mk])
        ]
        if not vals:
            continue
        summary["metrics"][mk] = {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run VERSA audio-quality metrics on one ASR dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", required=True,
                    help="Input CSV/TSV with audio path and transcript columns.")
    ap.add_argument("--dataset_name", required=True,
                    help="Short label for this dataset (used in summary).")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write all outputs.")
    ap.add_argument("--versa_repo", required=True,
                    help="Path to a checked-out wavlab-speech/versa repo.")
    ap.add_argument("--score_config", required=True,
                    help="VERSA YAML config (e.g. audio_quality_config.yaml).")
    ap.add_argument("--delimiter", default=",",
                    help="CSV delimiter. Use $'\\t' for TSV.")
    ap.add_argument("--has_header", action="store_true",
                    help="Set if the first row of the CSV is a header.")
    ap.add_argument("--audio_col", type=int, default=0,
                    help="0-based column index for the audio path.")
    ap.add_argument("--text_col", type=int, default=1,
                    help="0-based column index for the transcript.")
    ap.add_argument("--use_gpu", action="store_true",
                    help="Pass --use_gpu True to VERSA scorer.")
    ap.add_argument("--no_text", action="store_true",
                    help="Don't pass transcripts to VERSA. Set this if your "
                         "config doesn't include any text-based metric "
                         "(speeds startup, avoids spurious warnings).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Reading CSV: %s", args.csv)
    rows = parse_csv(
        args.csv,
        delimiter=args.delimiter,
        has_header=args.has_header,
        audio_col=args.audio_col,
        text_col=args.text_col,
    )
    LOG.info("Read %d rows from CSV", len(rows))

    LOG.info("Writing VERSA inputs (wav.scp, text) to %s", out_dir)
    indexed = write_versa_inputs(rows, out_dir)
    LOG.info("Prepared %d utterances for scoring", len(indexed))

    LOG.info("Running VERSA scorer...")
    raw_out = run_versa(
        versa_repo=args.versa_repo,
        score_config=args.score_config,
        out_dir=out_dir,
        use_gpu=args.use_gpu,
        pass_text=not args.no_text,
    )

    LOG.info("Loading VERSA results from %s", raw_out)
    versa_results = load_versa_results(raw_out)
    LOG.info("Got results for %d/%d utterances",
             len(versa_results), len(indexed))

    per_utt_csv = out_dir / "per_utterance.csv"
    LOG.info("Writing per-utterance CSV: %s", per_utt_csv)
    metric_keys = write_per_utterance_csv(indexed, versa_results, per_utt_csv)

    LOG.info("Computing dataset summary over %d metrics", len(metric_keys))
    summary = summarize(
        versa_results, metric_keys,
        dataset_name=args.dataset_name,
        n_total=len(rows),
        n_scored=len(versa_results),
    )
    summary_path = out_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    LOG.info("Wrote dataset summary: %s", summary_path)

    print(f"Done. Per-utterance scores: {per_utt_csv}")
    print(f"      Dataset summary:     {summary_path}")


if __name__ == "__main__":
    main()
