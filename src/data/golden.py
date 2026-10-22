"""Carve the frozen golden set out of the test split.

Stratified by location x condition. Writes:
- golden/manifest.json (git): stems, CI subset, and SHA-256 of every image and annotation file, so
  any drift in the golden data fails tests/test_golden.py.
- data/annotations/golden.json and golden_ci.json (DVC): COCO subsets of test.json.

The manifest is generated once. Re-running refuses to overwrite it unless --force is given: the
golden set must not change silently.

    uv run python -m src.data.golden
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from src.data.llvip import LLVIP, ROOT, Pair
from src.data.split import OUT, load_params

MANIFEST = ROOT / "golden" / "manifest.json"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Proportional allocation over location x condition strata (at least 1 per stratum)."""
    rng = np.random.default_rng(seed)
    groups = df.groupby(["location", "condition"])
    quota = (groups.size() / len(df) * n).round().clip(lower=1).astype(int)
    picked = []
    for key, grp in groups:
        stems = sorted(grp.stem)
        picked += list(rng.choice(stems, size=min(quota[key], len(stems)), replace=False))
    return sorted(picked)


def subset(coco: dict, stems: set[str]) -> dict:
    ids = {int(s) for s in stems}
    return {"images": [i for i in coco["images"] if i["id"] in ids],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
            "categories": coco["categories"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate the manifest")
    args = parser.parse_args()
    params = load_params()["golden"]
    test = json.loads((OUT / "test.json").read_text())

    if MANIFEST.exists() and not args.force:
        manifest = json.loads(MANIFEST.read_text())
        print(f"Using existing {MANIFEST.relative_to(ROOT)} ({len(manifest['stems'])} images)")
    else:
        df = pd.DataFrame([{"stem": i["stem"], "location": i["location"], "condition": i["condition"]}
                           for i in test["images"]])
        stems = stratified_sample(df, params["size"], params["seed"])
        ci = stratified_sample(df[df.stem.isin(stems)], params["ci_size"], params["seed"])
        files = {}
        for s in stems:
            pair = Pair(s, "test")
            files[s] = {"visible": sha256(pair.visible()), "infrared": sha256(pair.infrared()),
                        "annotation": sha256(pair.annotation())}
        manifest = {"description": "Frozen golden set: LLVIP test split, stratified by "
                                   "location x condition. Do not edit; see src/data/golden.py.",
                    "seed": params["seed"], "stems": stems, "ci_stems": ci, "sha256": files}
        MANIFEST.parent.mkdir(exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=1))
        print(f"Wrote {MANIFEST.relative_to(ROOT)}: {len(stems)} images, CI subset {len(ci)}")

    for name, stems in (("golden", manifest["stems"]), ("golden_ci", manifest["ci_stems"])):
        coco = subset(test, set(stems))
        assert len(coco["images"]) == len(stems), f"{name}: stems missing from test.json"
        (OUT / f"{name}.json").write_text(json.dumps(coco))
        print(f"{name}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")


if __name__ == "__main__":
    main()
