import re
import pathlib

ROOT = pathlib.Path("~/.conda/envs/versa/lib/python3.10/site-packages/s3prl")

PATTERN = re.compile(
    r'^(?P<indent>[ \t]*)torchaudio\.set_audio_backend\((?P<arg>[^)]*)\)[ \t]*$',
    re.MULTILINE,
)

def replace(m):
    indent = m.group("indent")
    arg = m.group("arg")
    return (
        f"{indent}try:\n"
        f"{indent}    torchaudio.set_audio_backend({arg})\n"
        f"{indent}except AttributeError:\n"
        f"{indent}    pass  # removed in torchaudio >= 2.1"
    )

patched = 0
skipped = 0
for f in ROOT.rglob("*.py"):
    text = f.read_text()
    if "set_audio_backend" not in text:
        continue
    new = PATTERN.sub(replace, text)
    if new == text:
        skipped += 1
        print(f"  no change: {f}")
        continue
    bak = f.with_suffix(f.suffix + ".bak")
    if not bak.exists():
        bak.write_text(text)
    f.write_text(new)
    patched += 1
    print(f"  patched:   {f}")

print(f"\nDone. Patched {patched} file(s); {skipped} contained the string but no patch was needed.")
