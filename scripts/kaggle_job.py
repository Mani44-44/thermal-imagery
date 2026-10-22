"""Generate (and optionally push) a Kaggle script kernel that trains one run.

The kernel clones this repo at a pinned commit, installs the transformers version from uv.lock,
attaches the verified LLVIP mirror plus our private annotations dataset, trains with torchrun on
all visible GPUs, then writes predictions for val/test/golden with the best checkpoint.
`--resume-from <kernel slug>` attaches an earlier kernel's output and continues from its last.pt.

    uv run python scripts/kaggle_job.py --slug tir-rgb-s1 --modality rgb --seed 1 --push
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "yashmanisingh"
REPO = "https://github.com/Mani44-44/thermal-imagery"
LLVIP_MIRROR = "afradhossain/llvip-dataset"  # verified identical to the official release (46,465 md5s)
ANNOTATIONS = f"{USER}/llvip-thermal-imagery-annotations"

RUNNER = r'''
import glob, json, os, shutil, subprocess, sys, time
CFG = json.loads(r"""__CFG__""")
def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"pip install -q transformers=={CFG['transformers']} pycocotools")
sh(f"git clone -q {CFG['repo']} /kaggle/working/repo && cd /kaggle/working/repo && git checkout -q {CFG['commit']}")
llvip = [p for p in glob.glob("/kaggle/input/**/LLVIP", recursive=True) if os.path.isdir(p + "/visible")][0]
ann = os.path.dirname(glob.glob("/kaggle/input/**/golden_ci.json", recursive=True)[0])
out = "/kaggle/working/run"
if CFG.get("resume_from"):
    prev = glob.glob(f"/kaggle/input/**/{CFG['resume_from']}/run", recursive=True) or \
           glob.glob("/kaggle/input/**/run/last.pt", recursive=True)
    src = prev[0] if prev[0].endswith("run") else os.path.dirname(prev[0])
    shutil.copytree(src, out, dirs_exist_ok=True)
    print("resuming from", src, os.listdir(out), flush=True)
ngpu = int(subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
os.chdir("/kaggle/working/repo")
env = f"GIT_COMMIT={CFG['commit']} PYTHONUNBUFFERED=1"
sh(f"{env} torchrun --standalone --nproc_per_node={ngpu} -m src.train --root {llvip} --ann-dir {ann} "
   f"--out {out} --batch-size {CFG['batch_per_gpu']} --workers 4 {CFG['train_args']}")
if os.path.exists(f"{out}/DONE"):
    for split in ("val", "test", "golden"):
        sh(f"python -m src.predict --gt {ann}/{split}.json --modality {CFG['modality']} --checkpoint {out}/best "
           f"--root {llvip} --out {out}/pred_{split}.json --batch-size 16 --workers 4")
    if not CFG.get("keep_last"):
        os.remove(f"{out}/last.pt")  # keep the output small once training is finished
shutil.rmtree("/kaggle/working/repo", ignore_errors=True)
print("FINISHED", os.listdir(out), flush=True)
'''


def locked_version(package: str) -> str:
    text = (ROOT / "uv.lock").read_text()
    return re.search(rf'name = "{package}"\nversion = "([^"]+)"', text).group(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True)
    p.add_argument("--modality", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--checkpoint", default="PekingU/rtdetr_v2_r18vd")
    p.add_argument("--batch-per-gpu", type=int, default=8)
    p.add_argument("--extra", default="", help="extra src.train args")
    p.add_argument("--resume-from", help="slug of a previous kernel whose output to continue")
    p.add_argument("--time-limit-hours", type=float, default=11.0)
    p.add_argument("--keep-last", action="store_true")
    p.add_argument("--push", action="store_true")
    a = p.parse_args()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, text=True)
    assert not dirty.strip(), f"src/ has uncommitted changes; commit + push first:\n{dirty}"
    train_args = (f"--modality {a.modality} --seed {a.seed} --epochs {a.epochs} --checkpoint {a.checkpoint} "
                  f"--time-limit-hours {a.time_limit_hours} --name {a.slug} {a.extra}").strip()
    cfg = {"repo": REPO, "commit": commit, "transformers": locked_version("transformers"),
           "modality": a.modality, "batch_per_gpu": a.batch_per_gpu, "train_args": train_args,
           "resume_from": a.resume_from, "keep_last": a.keep_last}
    folder = ROOT / "runs" / "kaggle" / "jobs" / a.slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(RUNNER.replace("__CFG__", json.dumps(cfg)))
    meta = {"id": f"{USER}/{a.slug}", "title": a.slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [LLVIP_MIRROR, ANNOTATIONS], "competition_sources": [],
            "kernel_sources": [f"{USER}/{a.resume_from}"] if a.resume_from else []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"kernel folder: {folder}\nconfig: {json.dumps(cfg)}")
    if a.push:
        subprocess.run(["uv", "run", "kaggle", "kernels", "push", "-p", ".", "--accelerator", "NvidiaTeslaT4"],
                       cwd=folder, check=True)


if __name__ == "__main__":
    main()
