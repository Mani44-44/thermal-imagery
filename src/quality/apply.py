"""Phase 6, step 3: apply reviewed corrections -> data/annotations/train_v2.json (dataset v2).

Input: data/quality/review.jsonl, one reviewed image per line:
  {"stem": "080123", "verdict": "fixed" | "no_error" | "ambiguous",
   "add": [[x, y, w, h], ...],        # boxes to add (e.g. a flagged prediction P3 -> its bbox)
   "remove": [gt_index, ...],         # GT boxes to delete (index as shown on the panel, G<i>)
   "replace": {"gt_index": [x, y, w, h]},  # tighten/fix a GT box
   "reviewer": "claude-model" | "<human>", "note": "..."}
Indices refer to GT boxes in train.json order for that image (as rendered by src.quality.flag).

Only the train split is corrected; val and test keep the original labels (v1), so results stay
comparable and the test set is never cleaned with a model.

    uv run python -m src.quality.apply
"""

import json
from collections import defaultdict

from src.data.llvip import ROOT
from src.data.split import load_params

ANN = ROOT / "data" / "annotations"
REVIEW = ROOT / "data" / "quality" / "review.jsonl"


def _bin(h, bins):
    return next(n for n, (lo, hi) in bins.items() if lo <= h < hi)


def main():
    coco = json.loads((ANN / "train.json").read_text())
    reviews = [json.loads(l) for l in REVIEW.read_text().splitlines() if l.strip()]
    by_stem = {r["stem"]: r for r in reviews}
    size_bins = load_params()["data"]["size_bins"]
    stems = {im["id"]: im["stem"] for im in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    out, stats = [], defaultdict(int)
    for image_id, boxes in anns.items():
        r = by_stem.get(stems[image_id])
        if not r or r["verdict"] != "fixed":
            out += boxes
            continue
        remove = set(r.get("remove", []))
        replace = {int(k): v for k, v in r.get("replace", {}).items()}
        for g, a in enumerate(boxes):
            if g in remove:
                stats["removed"] += 1
                continue
            if g in replace:
                x, y, w, h = replace[g]
                a = {**a, "bbox": [x, y, w, h], "area": w * h, "size_bin": _bin(h, size_bins), "corrected": True}
                stats["replaced"] += 1
            out.append(a)
    for r in reviews:  # additions (also for images that had no GT at all)
        if r["verdict"] != "fixed":
            continue
        image_id = int(r["stem"])
        for k, (x, y, w, h) in enumerate(r.get("add", [])):
            out.append({"id": image_id * 100 + 90 + k, "image_id": image_id, "category_id": 1,
                        "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0, "size_bin": _bin(h, size_bins),
                        "rgb_visible": -1, "truncated": 0, "corrected": True})
            stats["added"] += 1
    stats.update({f"verdict_{v}": sum(r["verdict"] == v for r in reviews) for v in ("fixed", "no_error", "ambiguous")})
    v2 = {**coco, "annotations": sorted(out, key=lambda a: (a["image_id"], a["id"]))}
    ids = [a["id"] for a in v2["annotations"]]
    assert len(ids) == len(set(ids)), "duplicate annotation ids"
    (ANN / "train_v2.json").write_text(json.dumps(v2))
    summary = {"reviewed_images": len(reviews), **stats,
               "boxes_v1": len(coco["annotations"]), "boxes_v2": len(v2["annotations"])}
    (ROOT / "data" / "quality" / "apply_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
