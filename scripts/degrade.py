"""Make a deliberately degraded copy of a model, to show the regression gate goes red (Phase 5 exit).

Adds Gaussian noise to every conv/linear weight matrix, scaled per tensor by that tensor's std.

    uv run python scripts/degrade.py --model models/thermal_s1 --sigma 0.05 --out runs/scratch/thermal_noisy
    CANDIDATE_MODEL=runs/scratch/thermal_noisy uv run pytest -m regression
"""

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    g = torch.Generator().manual_seed(a.seed)
    sd = load_file(a.model / "model.safetensors")
    for k, t in sd.items():
        if t.is_floating_point() and t.dim() >= 2 and k.endswith(".weight"):  # conv/linear weights, not BN stats or anchors
            sd[k] = t + torch.randn(t.shape, generator=g, dtype=t.dtype) * t.std() * a.sigma
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.model / "config.json", a.out / "config.json")
    save_file(sd, a.out / "model.safetensors", metadata={"format": "pt"})
    print(f"wrote {a.out} (sigma {a.sigma})")


if __name__ == "__main__":
    main()
