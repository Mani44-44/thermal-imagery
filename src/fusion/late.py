"""Late fusion: merge per-image boxes from an RGB model and a thermal model.

Methods:
- wbf: Weighted Boxes Fusion (ensemble-boxes). A box found by only one model has its score
  scaled down (conf_type="avg"), so agreement between modalities is rewarded.
- nms: concatenate both models' boxes and run NMS (the simple baseline).

Weights and IoU threshold are tuned on **val** only (`tune`), then applied once to test/golden.

    uv run python -m src.fusion.late --rgb runs/rgb_s1 --thermal runs/thermal_s1 --out runs/late_s1
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from torchvision.ops import nms

from src.eval.metrics import evaluate, match_all, summarize

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "annotations"
W, H = 1280, 1024


def _group(preds):
    g = defaultdict(list)
    for p in preds:
        g[p["image_id"]].append(p)
    return g


def _to_xyxy_norm(ps):
    b = np.array([p["bbox"] for p in ps], float).reshape(-1, 4)
    xyxy = np.stack([b[:, 0] / W, b[:, 1] / H, (b[:, 0] + b[:, 2]) / W, (b[:, 1] + b[:, 3]) / H], 1)
    return np.clip(xyxy, 0, 1), np.array([p["score"] for p in ps], float)


def fuse_image(rgb: list[dict], thermal: list[dict], method: str, weights=(1, 1), iou_thr=0.55,
               skip_thr=0.001, top_k=100) -> list[tuple[list[float], float]]:
    (b1, s1), (b2, s2) = _to_xyxy_norm(rgb), _to_xyxy_norm(thermal)
    if method == "wbf":
        boxes, scores, _ = weighted_boxes_fusion([b1.tolist(), b2.tolist()], [s1.tolist(), s2.tolist()],
                                                 [[0] * len(s1), [0] * len(s2)], weights=list(weights),
                                                 iou_thr=iou_thr, skip_box_thr=skip_thr)
    elif method == "nms":
        boxes, scores = np.concatenate([b1, b2]), np.concatenate([s1, s2])
        keep = nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32), iou_thr).numpy()
        boxes, scores = boxes[keep], scores[keep]
    else:
        raise ValueError(method)
    order = np.argsort(-np.asarray(scores))[:top_k]
    return [([float(boxes[i][0] * W), float(boxes[i][1] * H), float((boxes[i][2] - boxes[i][0]) * W),
              float((boxes[i][3] - boxes[i][1]) * H)], float(scores[i])) for i in order]


def fuse(rgb_preds, thermal_preds, image_ids, **kw) -> list[dict]:
    gr, gt_ = _group(rgb_preds), _group(thermal_preds)
    out = []
    for i in image_ids:
        for bbox, score in fuse_image(gr.get(i, []), gt_.get(i, []), **kw):
            out.append({"image_id": i, "category_id": 1, "bbox": [round(v, 2) for v in bbox],
                        "score": round(score, 5)})
    return out


def tune(rgb_val, thermal_val, gt_val) -> tuple[dict, list[dict]]:
    ids = [im["id"] for im in gt_val["images"]]
    grid = [{"method": "nms", "iou_thr": t} for t in (0.5, 0.6, 0.7)]
    # RT-DETR emits 100 boxes per image, mostly near-zero scores; WBF averages them into the real
    # boxes (WBF of thermal with itself drops val mAP@0.5 0.956 -> 0.830 at skip 0.001), so the
    # score cut-off is tuned too.
    grid += [{"method": "wbf", "weights": w, "iou_thr": t, "skip_thr": k}
             for w, t, k in itertools.product([(1, 1), (1, 2), (1, 3), (1, 5)], (0.6, 0.7, 0.8, 0.9), (0.05, 0.1, 0.2))]
    results = []
    for cfg in grid:
        s = summarize(list(match_all(gt_val, fuse(rgb_val, thermal_val, ids, **cfg)).values()), 0.5)
        results.append({**cfg, "val_mAP50": s["mAP50"], "val_mAP50_95": s["mAP50_95"]})
    best = max(results, key=lambda r: r["val_mAP50_95"])
    return {k: best[k] for k in ("method", "weights", "iou_thr", "skip_thr") if k in best}, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", type=Path, required=True, help="run dir with pred_{val,test,golden}.json")
    p.add_argument("--thermal", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default=None)
    a = p.parse_args()
    load = lambda run, split: json.loads((run / f"pred_{split}.json").read_text())
    gt = {s: json.loads((ANN / f"{s}.json").read_text()) for s in ("val", "test", "golden")}

    cfg, grid = tune(load(a.rgb, "val"), load(a.thermal, "val"), gt["val"])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tuning_val.json").write_text(json.dumps(grid, indent=1))
    print("best on val:", cfg)
    for split in ("val", "test", "golden"):
        preds = fuse(load(a.rgb, split), load(a.thermal, split), [im["id"] for im in gt[split]["images"]], **cfg)
        (a.out / f"pred_{split}.json").write_text(json.dumps(preds))
    (a.out / "run.json").write_text(json.dumps({"name": a.name or a.out.name, "fusion": cfg,
                                                "rgb_run": str(a.rgb), "thermal_run": str(a.thermal)}, indent=1))
    res = evaluate(gt["test"], json.loads((a.out / "pred_test.json").read_text()))
    (a.out / "metrics_test.json").write_text(json.dumps(res, indent=1))
    print("test mAP50 %.4f mAP50-95 %.4f" % (res["overall"]["mAP50"], res["overall"]["mAP50_95"]))


if __name__ == "__main__":
    main()
