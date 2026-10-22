"""Phase 6, step 1: flag likely label errors in the train split from out-of-fold predictions.

Predictions come from models that never saw the images they predict (scripts/oof.sh), so a
confident disagreement with the label is evidence about the label, not memorisation.

Signals per image:
- FiftyOne `compute_mistakenness` (sample `mistakenness`, `possible_missing`, `possible_spurious`)
- missing:  a prediction with score >= MISS_CONF overlapping no GT box (IoU < 0.3)
- spurious: a GT box with no prediction of score >= 0.1 near it (IoU < 0.3)
- loose:    a GT box whose best confident prediction (score >= 0.8) has IoU in [0.3, 0.6)

Writes data/quality/flags.json (ranked review queue) and renders review panels to
runs/quality/panels/<stem>.jpg (visible | infrared; GT green + index, flagged preds red + score).
Saved FiftyOne views: "missing", "spurious", "loose" on dataset "llvip-train-oof".

    uv run python -m src.quality.flag --top 250
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.data.llvip import LLVIP, ROOT
from src.eval.cocoish import iou_xywh

OUT = ROOT / "data" / "quality"
PANELS = ROOT / "runs" / "quality" / "panels"
MISS_CONF, SPUR_CONF, LOOSE_CONF = 0.7, 0.1, 0.8
W, H = 1280, 1024


def load_oof() -> list[dict]:
    preds = []
    for p in sorted((ROOT / "runs" / "oof").glob("*-fold*/pred_heldout.json")):
        preds += json.loads(p.read_text())
    assert preds, "no out-of-fold predictions; run scripts/oof.sh"
    return preds


def rule_flags(gt_boxes: np.ndarray, pred: list[dict]) -> dict:
    pb = np.array([p["bbox"] for p in pred], float).reshape(-1, 4)
    ps = np.array([p["score"] for p in pred], float)
    ious = iou_xywh(pb, gt_boxes) if len(pb) and len(gt_boxes) else np.zeros((len(pb), len(gt_boxes)))
    missing = [int(i) for i in np.flatnonzero(ps >= MISS_CONF)
               if not len(gt_boxes) or ious[i].max() < 0.3]
    spurious, loose = [], []
    for g in range(len(gt_boxes)):
        near = ious[:, g] if len(pb) else np.zeros(0)
        if not (near[ps >= SPUR_CONF] >= 0.3).any():
            spurious.append(g)
        conf = np.flatnonzero(ps >= LOOSE_CONF)
        if len(conf):
            best = conf[near[conf].argmax()]
            if 0.3 <= near[best] < 0.6:
                loose.append({"gt": g, "pred": int(best), "iou": round(float(near[best]), 3)})
    return {"missing": missing, "spurious": spurious, "loose": loose}


def fiftyone_mistakenness(coco: dict, preds_by_image: dict) -> dict:
    import fiftyone as fo
    import fiftyone.brain as fob

    name = "llvip-train-oof"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)
    gts = defaultdict(list)
    for a in coco["annotations"]:
        gts[a["image_id"]].append(a)
    samples = []
    for im in coco["images"]:
        if im["id"] not in preds_by_image:
            continue
        det = lambda b, **kw: fo.Detection(label="person", bounding_box=[b[0] / W, b[1] / H, b[2] / W, b[3] / H], **kw)
        samples.append(fo.Sample(
            filepath=str(LLVIP / "infrared" / im["file_name"]), stem=im["stem"], location=im["location"],
            condition=im["condition"], visible_path=str(LLVIP / "visible" / im["file_name"]),
            ground_truth=fo.Detections(detections=[det(a["bbox"]) for a in gts[im["id"]]]),
            predictions=fo.Detections(detections=[det(p["bbox"], confidence=p["score"])
                                                  for p in preds_by_image[im["id"]] if p["score"] >= 0.05])))
    ds.add_samples(samples)
    fob.compute_mistakenness(ds, "predictions", label_field="ground_truth")
    out = {}
    for s in ds.select_fields(["stem", "mistakenness", "possible_missing", "possible_spurious"]):
        out[s.stem] = {"mistakenness": s["mistakenness"], "possible_missing": s["possible_missing"],
                       "possible_spurious": s["possible_spurious"]}
    return out, ds


def render(stem: str, file_name: str, gt_boxes, pred: list[dict], flags: dict, out: Path):
    vis = cv2.imread(str(LLVIP / "visible" / file_name))
    vis = cv2.convertScaleAbs(vis, alpha=2.2, beta=10)  # brighten dark frames for review
    ir = cv2.imread(str(LLVIP / "infrared" / file_name))
    shown = set(flags["missing"]) | {l["pred"] for l in flags["loose"]}
    for img in (vis, ir):
        for g, b in enumerate(gt_boxes):
            x, y, w, h = map(int, b)
            col = (0, 255, 255) if g in flags["spurious"] else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            cv2.putText(img, f"G{g}", (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        for i in sorted(shown):
            x, y, w, h = map(int, pred[i]["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, f"P{i} {pred[i]['score']:.2f}", (x, min(y + h + 22, H - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    panel = cv2.resize(np.hstack([vis, ir]), (1600, 640))
    cv2.putText(panel, stem, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(out / f"{stem}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=250)
    p.add_argument("--no-fiftyone", action="store_true")
    a = p.parse_args()

    coco = json.loads((ROOT / "data" / "annotations" / "train.json").read_text())
    preds = defaultdict(list)
    for x in load_oof():
        preds[x["image_id"]].append(x)
    for v in preds.values():
        v.sort(key=lambda d: -d["score"])
    gts = defaultdict(list)
    for ann in coco["annotations"]:
        gts[ann["image_id"]].append(ann["bbox"])

    fo_scores, ds = ({}, None) if a.no_fiftyone else fiftyone_mistakenness(coco, preds)
    rows = []
    for im in coco["images"]:
        if im["id"] not in preds:
            continue
        g = np.array(gts[im["id"]], float).reshape(-1, 4)
        f = rule_flags(g, preds[im["id"]])
        fs = fo_scores.get(im["stem"], {})
        score = (sum(preds[im["id"]][i]["score"] for i in f["missing"]) + len(f["spurious"])
                 + 0.5 * len(f["loose"]) + (fs.get("mistakenness") or 0))
        if f["missing"] or f["spurious"] or f["loose"]:
            rows.append({"stem": im["stem"], "image_id": im["id"], "file_name": im["file_name"],
                         "location": im["location"], "condition": im["condition"], "score": round(score, 3),
                         **f, "fiftyone": fs})
    rows.sort(key=lambda r: -r["score"])
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"images_with_oof_preds": len(preds), "flagged_images": len(rows),
               "missing": sum(len(r["missing"]) for r in rows), "spurious": sum(len(r["spurious"]) for r in rows),
               "loose": sum(len(r["loose"]) for r in rows), "review_queue": args_top(rows, a.top)}
    (OUT / "flags.json").write_text(json.dumps({"summary": summary, "queue": rows[:a.top], "all": rows}, indent=1))
    PANELS.mkdir(parents=True, exist_ok=True)
    for r in rows[:a.top]:
        render(r["stem"], r["file_name"], np.array(gts[r["image_id"]], float).reshape(-1, 4),
               preds[r["image_id"]], r, PANELS)
    if ds is not None:
        from fiftyone import ViewField as F
        ds.save_view("high_mistakenness", ds.sort_by("mistakenness", reverse=True).limit(a.top))
        ds.save_view("possible_missing", ds.match(F("possible_missing") > 0))
        ds.save_view("possible_spurious", ds.match(F("possible_spurious") > 0))
    print(json.dumps(summary, indent=1))


def args_top(rows, top):
    return min(top, len(rows))


if __name__ == "__main__":
    main()
