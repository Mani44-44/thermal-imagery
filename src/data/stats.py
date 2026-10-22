"""Per-image and per-box photometric statistics for LLVIP.

Outputs (data/meta/):
- image_stats.csv: stem, split, scene, visible mean/p95 luminance, infrared mean/std
- box_stats.csv: one row per box with its size and how much it stands out in the visible and
  infrared images (used for the "RGB-visible" flag and EDA)

    uv run python -m src.data.stats
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.llvip import LLVIP, ROOT, Pair, list_pairs, read_boxes

RING = 0.5  # context ring around a box: box expanded by this fraction of its size on each side


def box_contrast(gray: np.ndarray, xyxy: list[float]) -> tuple[float, float, float]:
    """(mean inside, mean of surrounding ring, std inside) for a box in a grayscale image."""
    h, w = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return float("nan"), float("nan"), float("nan")
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = max(int(x1 - RING * bw), 0), max(int(y1 - RING * bh), 0)
    X2, Y2 = min(int(x2 + RING * bw), w), min(int(y2 + RING * bh), h)
    outer = gray[Y1:Y2, X1:X2].astype(np.float64)
    inner = gray[y1:y2, x1:x2].astype(np.float64)
    ring_sum, ring_n = outer.sum() - inner.sum(), outer.size - inner.size
    ring_mean = ring_sum / ring_n if ring_n > 0 else float("nan")
    return float(inner.mean()), float(ring_mean), float(inner.std())


def process(pair: Pair, root: Path) -> tuple[dict, list[dict]]:
    vis = cv2.imread(str(pair.visible(root)), cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(str(pair.infrared(root)), cv2.IMREAD_GRAYSCALE)
    image = {
        "stem": pair.stem, "split": pair.official_split, "scene": pair.scene,
        "vis_mean": float(vis.mean()), "vis_p95": float(np.percentile(vis, 95)),
        "ir_mean": float(ir.mean()), "ir_std": float(ir.std()),
    }
    boxes = []
    for i, b in enumerate(read_boxes(pair.annotation(root))):
        x1, y1, x2, y2 = b["xyxy"]
        v_in, v_ring, v_std = box_contrast(vis, b["xyxy"])
        i_in, i_ring, i_std = box_contrast(ir, b["xyxy"])
        boxes.append({
            "stem": pair.stem, "box": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": x2 - x1, "h": y2 - y1, "truncated": b["truncated"],
            "vis_in": v_in, "vis_ring": v_ring, "vis_std": v_std,
            "ir_in": i_in, "ir_ring": i_ring, "ir_std": i_std,
        })
    return image, boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=LLVIP)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "meta")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pairs = list_pairs(args.root)
    images, boxes = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        results = pool.map(process, pairs, [args.root] * len(pairs), chunksize=64)
        for image, bxs in tqdm(results, total=len(pairs), desc="stats"):
            images.append(image)
            boxes.extend(bxs)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(images).to_csv(args.out / "image_stats.csv", index=False, float_format="%.3f")
    pd.DataFrame(boxes).to_csv(args.out / "box_stats.csv", index=False, float_format="%.3f")
    print(f"{len(images)} images, {len(boxes)} boxes -> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
