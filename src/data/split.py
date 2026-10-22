"""Build train/val/test COCO annotation files plus scene and normalisation metadata.

Inputs: data/raw/LLVIP, data/meta/{image,box}_stats.csv (src.data.stats), data/meta/locations.csv
(src.data.locations), params.yaml.
Outputs:
- data/annotations/{train,val,test}.json: COCO detection format, one category (person, id 1).
  Image entries carry `stem`, `scene`, `location`, `condition`, `vis_mean`; annotations carry `size_bin`,
  `rgb_visible` and `truncated`. `file_name` is "<official_split>/<stem>.jpg", relative to the
  modality folder (data/raw/LLVIP/visible or .../infrared).
- data/meta/scenes.csv: location -> split, image count, median brightness.
- data/meta/norm.json: per-channel mean/std (0-1 scale) of the training images, for the thermal
  channel of the fused model.

    uv run python -m src.data.split
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.llvip import HEIGHT, ROOT, WIDTH

META = ROOT / "data" / "meta"
OUT = ROOT / "data" / "annotations"
TEST_SCENES = {19, 20, 21, 22, 23, 24, 26}


def load_params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text())


def bin_of(value: float, bins: dict[str, list[float]]) -> str:
    for name, (lo, hi) in bins.items():
        if lo <= value < hi:
            return name
    raise ValueError(f"{value} outside bins {bins}")


def assign_split(scene: int, location: int, val_locations: set[int]) -> str:
    # Test is the official split (by filename prefix); val holds out whole train *locations*, so
    # frames of the mixed scene 25 shot at a val location go to val too.
    if scene in TEST_SCENES:
        return "test"
    return "val" if location in val_locations else "train"


def build(images: pd.DataFrame, boxes: pd.DataFrame, params: dict) -> dict[str, dict]:
    p = params["data"]
    val_locations = set(p["val_locations"])
    assert not val_locations & TEST_SCENES, "val locations must come from the official train scenes"
    images = images.assign(
        split=[assign_split(s, l, val_locations) for s, l in zip(images.scene, images.location)],
        condition=[bin_of(v, p["condition_bins"]) for v in images.vis_mean])
    # LLVIP has 5 degenerate boxes (0 px wide, 1-7 px tall: labelling slips). Drop them.
    degenerate = (boxes.w < 2) | (boxes.h < 2)
    if degenerate.any():
        print(f"dropping {degenerate.sum()} degenerate boxes: "
              f"{boxes.loc[degenerate, 'stem'].tolist()}")
    boxes = boxes[~degenerate]
    contrast = np.maximum((boxes.vis_in - boxes.vis_ring).abs(), boxes.vis_std)
    boxes = boxes.assign(
        size_bin=[bin_of(h, p["size_bins"]) for h in boxes.h],
        rgb_visible=(contrast >= p["rgb_visible_min_contrast"]).astype(int))

    by_stem = {stem: grp for stem, grp in boxes.groupby("stem")}
    cocos = {}
    for split, imgs in images.groupby("split"):
        coco = {"images": [], "annotations": [],
                "categories": [{"id": 1, "name": "person", "supercategory": "person"}]}
        for im in imgs.sort_values("stem").itertuples():
            image_id = int(im.stem)
            official = "test" if im.scene in TEST_SCENES else "train"
            coco["images"].append({
                "id": image_id, "file_name": f"{official}/{im.stem}.jpg",
                "width": WIDTH, "height": HEIGHT, "stem": im.stem, "scene": int(im.scene),
                "location": int(im.location),
                "condition": im.condition, "vis_mean": round(float(im.vis_mean), 2)})
            for b in by_stem.get(im.stem, pd.DataFrame()).itertuples():
                coco["annotations"].append({
                    "id": image_id * 100 + int(b.box), "image_id": image_id, "category_id": 1,
                    "bbox": [b.x1, b.y1, b.w, b.h], "area": float(b.w * b.h), "iscrowd": 0,
                    "size_bin": b.size_bin, "rgb_visible": int(b.rgb_visible),
                    "truncated": int(b.truncated)})
        cocos[split] = coco
    return cocos, images


def norm_stats(images: pd.DataFrame) -> dict:
    """Pooled mean/std over training images, from per-image mean/std (0-1 scale)."""
    tr = images[images.split == "train"]
    out = {}
    for ch in ("ir",):
        mean = tr[f"{ch}_mean"].mean()
        var = (tr[f"{ch}_std"] ** 2).mean() + tr[f"{ch}_mean"].var(ddof=0)
        out[ch] = {"mean": round(mean / 255, 4), "std": round(float(np.sqrt(var)) / 255, 4)}
    return out


def main():
    params = load_params()
    images = pd.read_csv(META / "image_stats.csv", dtype={"stem": str})
    boxes = pd.read_csv(META / "box_stats.csv", dtype={"stem": str})
    locations = pd.read_csv(META / "locations.csv", dtype={"stem": str})
    images = images.merge(locations[["stem", "location"]], on="stem", validate="1:1")
    cocos, images = build(images, boxes, params)

    OUT.mkdir(parents=True, exist_ok=True)
    for split, coco in cocos.items():
        (OUT / f"{split}.json").write_text(json.dumps(coco))
        n_vis = sum(a["rgb_visible"] for a in coco["annotations"])
        print(f"{split:5}: {len(coco['images']):5} images, {len(coco['annotations']):5} boxes "
              f"({n_vis / len(coco['annotations']):.1%} RGB-visible), locations "
              f"{sorted({i['location'] for i in coco['images']})}")

    scenes = images.groupby("location").agg(split=("split", "first"), images=("stem", "size"),
                                         vis_mean_median=("vis_mean", "median"))
    scenes.round(1).to_csv(META / "scenes.csv")
    (META / "norm.json").write_text(json.dumps(norm_stats(images), indent=2))


if __name__ == "__main__":
    main()
