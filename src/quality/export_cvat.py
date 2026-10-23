"""Phase 6, step 2 (tooling): export the review queue for CVAT.

Writes two COCO 1.0 files over the same image list (the flagged queue from src.quality.flag), which
CVAT imports as annotations into a task created from the infrared images:
- data/quality/cvat/ground_truth.json   current (v1) labels, to be corrected in CVAT
- data/quality/cvat/oof_predictions.json   out-of-fold predictions (score >= 0.3) as pre-annotations

The first pass of this project was reviewed by the model directly from the rendered panels
(data/quality/review.jsonl, reviewer "claude-model"); a human pass would load these files in CVAT and
export corrected COCO back, which src.quality.apply-style merging can consume.

    uv run python -m src.quality.export_cvat
"""

import json
from collections import defaultdict

from src.data.llvip import ROOT
from src.quality.flag import load_oof

OUT = ROOT / "data" / "quality" / "cvat"
MIN_SCORE = 0.3


def main():
    queue = json.loads((ROOT / "data" / "quality" / "flags.json").read_text())["queue"]
    ids = {r["image_id"] for r in queue}
    coco = json.loads((ROOT / "data" / "annotations" / "train.json").read_text())
    images = [{"id": im["id"], "file_name": im["file_name"].split("/")[-1], "width": im["width"],
               "height": im["height"]} for im in coco["images"] if im["id"] in ids]
    cats = [{"id": 1, "name": "person", "supercategory": ""}]

    gt = [{"id": a["id"], "image_id": a["image_id"], "category_id": 1, "bbox": a["bbox"], "area": a["area"],
           "iscrowd": 0, "segmentation": []} for a in coco["annotations"] if a["image_id"] in ids]
    preds = defaultdict(list)
    for p in load_oof():
        if p["image_id"] in ids and p["score"] >= MIN_SCORE:
            preds[p["image_id"]].append(p)
    pr = []
    for image_id, ps in preds.items():
        for p in sorted(ps, key=lambda d: -d["score"]):
            x, y, w, h = p["bbox"]
            pr.append({"id": len(pr) + 1, "image_id": image_id, "category_id": 1, "bbox": [x, y, w, h],
                       "area": w * h, "iscrowd": 0, "segmentation": [],
                       "attributes": {"score": round(p["score"], 3)}})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, anns in (("ground_truth", gt), ("oof_predictions", pr)):
        (OUT / f"{name}.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    print(f"{len(images)} images, {len(gt)} GT boxes, {len(pr)} predictions -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
