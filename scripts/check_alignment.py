"""Phase 0 alignment check: are LLVIP visible/infrared pairs registered pixel for pixel?

For N random pairs, saves a 3-panel image (visible | infrared | edge overlay) to docs/alignment/.
In the overlay, visible-image edges are red and infrared edges are cyan, so aligned structure shows
as white or overlapping lines. Also prints a phase-correlation shift estimate per pair.

    uv run python scripts/check_alignment.py --root data/raw/LLVIP --n 20
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def edges(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/raw/LLVIP")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    visible = sorted((args.root / "visible").rglob("*.jpg"))
    assert visible, f"no visible images under {args.root / 'visible'}"
    pairs = [(v, args.root / "infrared" / v.relative_to(args.root / "visible")) for v in visible]
    missing = [ir for _, ir in pairs if not ir.exists()]
    print(f"{len(pairs)} visible images, {len(missing)} without an infrared partner")

    out_dir = ROOT / "docs" / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    shifts = []
    for vis_path, ir_path in random.sample([p for p in pairs if p[1].exists()], args.n):
        vis = cv2.imread(str(vis_path))
        ir = cv2.imread(str(ir_path))
        assert vis.shape == ir.shape, f"size mismatch {vis_path.name}: {vis.shape} vs {ir.shape}"
        vis_g = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        ir_g = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

        # Phase correlation on edge maps: modalities differ in intensity but share edges.
        (dx, dy), _ = cv2.phaseCorrelate(edges(vis_g).astype(np.float32),
                                         edges(ir_g).astype(np.float32))
        shifts.append((dx, dy))

        overlay = np.zeros_like(vis)
        overlay[..., 2] = edges(vis_g)        # red: visible
        overlay[..., 0] = overlay[..., 1] = edges(ir_g)  # cyan: infrared
        panel = np.hstack([vis, ir, overlay])
        scale = 1600 / panel.shape[1]
        cv2.imwrite(str(out_dir / f"{vis_path.stem}.jpg"), cv2.resize(panel, None, fx=scale, fy=scale))
        print(f"{vis_path.name}: {vis.shape[1]}x{vis.shape[0]}  shift dx={dx:+.1f}px dy={dy:+.1f}px")

    mags = np.hypot(*np.array(shifts).T)
    print(f"\nshift magnitude: median {np.median(mags):.1f}px, max {mags.max():.1f}px "
          f"(< ~3px means effectively registered)")
    print(f"panels saved to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
