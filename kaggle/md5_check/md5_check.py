"""Compute md5 of every file in the attached LLVIP mirror, keyed by path relative to the LLVIP/ folder."""
import hashlib, json, os
from pathlib import Path

roots = [p for p in Path("/kaggle/input").rglob("LLVIP") if p.is_dir() and (p / "Annotations").exists()]
print("LLVIP roots found:", roots)
root = roots[0]
out = {}
for p in sorted(root.rglob("*")):
    if p.is_file():
        out[p.relative_to(root).as_posix()] = hashlib.md5(p.read_bytes()).hexdigest()
print(len(out), "files hashed")
Path("/kaggle/working/mirror_md5.json").write_text(json.dumps({"root": str(root), "md5": out}))
