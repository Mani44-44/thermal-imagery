"""Evaluate COCO-format predictions against an annotation file.

Reports mAP@0.5, mAP@0.5:0.95, precision/recall at an operating confidence, and breakdowns
by lighting condition, box-height size bin, location, and the RGB-visible box subset. Includes
location-level bootstrap 95% CIs and paired bootstrap CIs for the difference between two models.

Predictions: list of {"image_id", "category_id", "bbox": [x, y, w, h] in original pixels, "score"}.

    uv run python -m src.eval.metrics --gt data/annotations/test.json --pred runs/x/pred_test.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eval.cocoish import IOU_THRS, average_precision, match_image, precision_recall_at
from src.data.split import load_params

N_BOOT = 1000


def _by_image(items, key="image_id"):
    out = defaultdict(list)
    for x in items:
        out[x[key]].append(x)
    return out


def match_all(gt: dict, preds: list[dict], view: str = "all") -> dict[int, object]:
    """Match predictions per image under a 'view': all boxes, a size bin, or RGB-visible only."""
    size_bins = load_params()["data"]["size_bins"]
    gts, dts = _by_image(gt["annotations"]), _by_image(preds)
    matches = {}
    for im in gt["images"]:
        g = gts.get(im["id"], [])
        d = dts.get(im["id"], [])
        gb = np.array([a["bbox"] for a in g], float).reshape(-1, 4)
        db = np.array([p["bbox"] for p in d], float).reshape(-1, 4)
        sc = np.array([p["score"] for p in d], float)
        ignore = np.zeros(len(g), bool)
        det_oor = None
        if view in size_bins:
            lo, hi = size_bins[view]
            ignore = (gb[:, 3] < lo) | (gb[:, 3] >= hi)
            det_oor = (db[:, 3] < lo) | (db[:, 3] >= hi)
        elif view == "rgb_visible":
            ignore = np.array([not a["rgb_visible"] for a in g], bool)
        elif view != "all":
            raise ValueError(view)
        matches[im["id"]] = match_image(gb, ignore, db, sc, det_out_of_range=det_oor)
    return matches


def summarize(matches: list, conf: float) -> dict:
    ap = average_precision(matches)
    p, r = precision_recall_at(matches, conf)
    return {"mAP50": float(ap[0]), "mAP50_95": float(np.nanmean(ap)) if not np.isnan(ap).all() else float("nan"),
            "precision": p, "recall": r, "n_images": len(matches), "n_gt": sum(m.n_gt for m in matches)}


def bootstrap_ci(matches: dict, clusters: dict[int, list[int]], n_boot=N_BOOT, seed=0) -> dict:
    """95% percentile CI of mAP50 and mAP50_95, resampling whole clusters (locations)."""
    rng = np.random.default_rng(seed)
    keys = sorted(clusters)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        ms = [matches[i] for k in pick for i in clusters[k]]
        ap = average_precision(ms)
        stats.append((ap[0], np.mean(ap)))
    lo, hi = np.percentile(np.array(stats), [2.5, 97.5], axis=0)
    return {"mAP50": [float(lo[0]), float(hi[0])], "mAP50_95": [float(lo[1]), float(hi[1])]}


def evaluate(gt: dict, preds: list[dict], conf: float = 0.5, n_boot: int = N_BOOT) -> dict:
    images = {im["id"]: im for im in gt["images"]}
    clusters = defaultdict(list)
    for im in gt["images"]:
        clusters[im["location"]].append(im["id"])

    m_all = match_all(gt, preds, "all")
    result = {"overall": summarize(list(m_all.values()), conf),
              "ci95": bootstrap_ci(m_all, clusters, n_boot) if n_boot else None,
              "conf": conf, "by_condition": {}, "by_location": {}, "by_size": {}}
    for cond in ("dark", "dim", "lit"):
        ids = [i for i, im in images.items() if im["condition"] == cond]
        if ids:
            sub_clusters = {k: [i for i in v if images[i]["condition"] == cond] for k, v in clusters.items()}
            sub_clusters = {k: v for k, v in sub_clusters.items() if v}
            result["by_condition"][cond] = summarize([m_all[i] for i in ids], conf)
            if n_boot:
                result["by_condition"][cond]["ci95"] = bootstrap_ci(m_all, sub_clusters, n_boot)
    for loc, ids in sorted(clusters.items()):
        result["by_location"][str(loc)] = summarize([m_all[i] for i in ids], conf)
    for size in load_params()["data"]["size_bins"]:
        result["by_size"][size] = summarize(list(match_all(gt, preds, size).values()), conf)
    result["rgb_visible"] = summarize(list(match_all(gt, preds, "rgb_visible").values()), conf)
    return result


def paired_delta(gt: dict, preds_a: list[dict], preds_b: list[dict], n_boot=N_BOOT, seed=0,
                 condition: str | None = None) -> dict:
    """Bootstrap CI of (b - a) for mAP50 and mAP50_95, same location resamples for both models."""
    ids_ok = {im["id"] for im in gt["images"] if condition is None or im["condition"] == condition}
    clusters = defaultdict(list)
    for im in gt["images"]:
        if im["id"] in ids_ok:
            clusters[im["location"]].append(im["id"])
    ma, mb = match_all(gt, preds_a), match_all(gt, preds_b)
    rng = np.random.default_rng(seed)
    keys = sorted(clusters)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        ids = [i for k in pick for i in clusters[k]]
        a, b = average_precision([ma[i] for i in ids]), average_precision([mb[i] for i in ids])
        deltas.append((b[0] - a[0], np.mean(b) - np.mean(a)))
    deltas = np.array(deltas)
    ids = [i for v in clusters.values() for i in v]
    a, b = average_precision([ma[i] for i in ids]), average_precision([mb[i] for i in ids])
    lo, hi = np.percentile(deltas, [2.5, 97.5], axis=0)
    return {"mAP50": {"delta": float(b[0] - a[0]), "ci95": [float(lo[0]), float(hi[0])]},
            "mAP50_95": {"delta": float(np.mean(b) - np.mean(a)), "ci95": [float(lo[1]), float(hi[1])]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    res = evaluate(json.loads(args.gt.read_text()), json.loads(args.pred.read_text()), args.conf)
    text = json.dumps(res, indent=1)
    if args.out:
        args.out.write_text(text)
    o = res["overall"]
    print(f"mAP50 {o['mAP50']:.4f}  mAP50-95 {o['mAP50_95']:.4f}  P {o['precision']:.3f}  R {o['recall']:.3f}")


if __name__ == "__main__":
    main()
