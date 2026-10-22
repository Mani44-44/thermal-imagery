"""Phase 0 alignment check: are LLVIP visible/infrared pairs registered pixel for pixel?

For N random pairs, estimates the residual translation between visible and infrared with ECC
(cv2.findTransformECC) on blurred gradient-magnitude images. Gradients are comparable across
modalities where raw intensities are not, and ECC starts from zero shift, so it refines locally
instead of jumping to a spurious far-away match. (Global phase correlation and an edge chamfer
search were both tried first and were unreliable on dark frames and dense foliage.)

Also saves a 3-panel image (visible | infrared | edge overlay; red = visible edges, cyan =
infrared edges) per pair to docs/alignment/ for visual inspection. Those images are dataset
content and are git-ignored.

    uv run python scripts/check_alignment.py --root data/raw/LLVIP --n 20
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCALE = 0.5  # run ECC at half resolution; shifts are reported in full-resolution pixels


def gradient_image(gray: np.ndarray) -> np.ndarray:
    g = cv2.equalizeHist(gray)  # lifts dark visible frames so they have usable gradients
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return cv2.GaussianBlur(mag, (0, 0), 3)


def ecc_shift(vis_gray: np.ndarray, ir_gray: np.ndarray):
    """Return (dx, dy, corr_at_zero, corr_after) or None if ECC does not converge."""
    a, b = gradient_image(vis_gray), gradient_image(ir_gray)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    try:
        corr, warp = cv2.findTransformECC(b, a, warp, cv2.MOTION_TRANSLATION, criteria, None, 5)
    except cv2.error:
        return None
    corr0 = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return warp[0, 2] / SCALE, warp[1, 2] / SCALE, corr0, corr


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

        result = ecc_shift(vis_g, ir_g)
        if result is None:
            print(f"{vis_path.name}: ECC did not converge")
        else:
            dx, dy, corr0, corr = result
            shifts.append((dx, dy))
            print(f"{vis_path.name}: {vis.shape[1]}x{vis.shape[0]}  shift dx={dx:+6.1f}px "
                  f"dy={dy:+6.1f}px  gradient corr {corr0:.2f} at 0 -> {corr:.2f}")

        overlay = np.zeros_like(vis)
        overlay[..., 2] = edges(vis_g)                   # red: visible
        overlay[..., 0] = overlay[..., 1] = edges(ir_g)  # cyan: infrared
        panel = np.hstack([vis, ir, overlay])
        scale = 1600 / panel.shape[1]
        cv2.imwrite(str(out_dir / f"{vis_path.stem}.jpg"), cv2.resize(panel, None, fx=scale, fy=scale))

    mags = np.hypot(*np.array(shifts).T)
    print(f"\n{len(shifts)}/{args.n} converged | shift median {np.median(mags):.1f}px, "
          f"90th pct {np.percentile(mags, 90):.1f}px, max {mags.max():.1f}px")
    print(f"panels saved to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
