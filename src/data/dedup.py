"""Near-duplicate scan across splits (leakage check).

Computes a 256-bit difference hash of every infrared image (thermal is used because visible
night frames are too dark to hash reliably), then lists every pair from *different* splits whose
hashes differ by at most --max-dist bits. Output: data/meta/cross_split_near_dups.csv (may be empty).

    uv run python -m src.data.dedup
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from src.data.llvip import LLVIP, ROOT
from src.data.split import META, OUT


def dhash(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-dist", type=int, default=12)
    args = parser.parse_args()

    rows = []
    for split in ("train", "val", "test"):
        coco = json.loads((OUT / f"{split}.json").read_text())
        rows += [(i["stem"], split, str(LLVIP / "infrared" / i["file_name"])) for i in coco["images"]]
    with ProcessPoolExecutor(8) as pool:
        hashes = np.stack(list(pool.map(dhash, [r[2] for r in rows], chunksize=128)))
    bits = np.unpackbits(hashes, axis=1).astype(np.uint8)
    splits = np.array([r[1] for r in rows])

    found = []
    for i in range(len(rows)):
        other = splits != splits[i]
        other[:i + 1] = False  # each unordered pair once
        dist = (bits[other] != bits[i]).sum(axis=1)
        for j, d in zip(np.flatnonzero(other)[dist <= args.max_dist], dist[dist <= args.max_dist]):
            found.append({"stem_a": rows[i][0], "split_a": rows[i][1],
                          "stem_b": rows[j][0], "split_b": rows[j][1], "hamming": int(d)})
    out = META / "cross_split_near_dups.csv"
    pd.DataFrame(found, columns=["stem_a", "split_a", "stem_b", "split_b", "hamming"]).to_csv(out, index=False)
    print(f"{len(found)} cross-split near-duplicate pairs (hamming <= {args.max_dist}) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
