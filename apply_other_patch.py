import pathlib, re
p = pathlib.Path("~/.conda/envs/versa/lib/python3.10/site-packages/s3prl/hub.py")
text = p.read_text()
bak = p.with_suffix(p.suffix + ".bak")
if not bak.exists():
    bak.write_text(text)

# Wrap each `from s3prl.upstream.X.hubconf import *` line in try/except
new = re.sub(
    r'^(from s3prl\.upstream\..*\.hubconf import \*)$',
    r'try:\n    \1\nexcept (ImportError, ModuleNotFoundError, AttributeError) as _e:\n    import warnings; warnings.warn(f"skipping s3prl upstream: {_e}")',
    text,
    flags=re.MULTILINE,
)
p.write_text(new)
print("patched")
