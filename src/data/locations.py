"""Assign every LLVIP frame a camera *location*, the unit for splitting and for bootstrap CIs.

Filename prefixes 01-24 are single locations. 25 (train) and 26 (test) are mixed: they hold
frames from the other train (01-18) and test (19-24) locations, plus daytime footage. Each 25/26
frame takes the location of its nearest frame (infrared 256-bit difference hash) among the pure
scenes of the same official split if the distance is at most --max-dist. Otherwise it keeps 25
or 26 as its own location.

Output: data/meta/locations.csv (stem, scene, location, nn_dist).

    uv run python -m src.data.locations
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.data.dedup import dhash
from src.data.llvip import LLVIP, ROOT, list_pairs

MIXED = {25: range(1, 19), 26: range(19, 25)}  # mixed scene -> pure scenes it draws from


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-dist", type=int, default=20)
    args = parser.parse_args()

    pairs = list_pairs()
    with ProcessPoolExecutor(8) as pool:
        hashes = np.stack(list(pool.map(dhash, [str(p.infrared()) for p in pairs], chunksize=128)))
    bits = np.unpackbits(hashes, axis=1).astype(np.uint8)
    scenes = np.array([p.scene for p in pairs])

    location, nn_dist = scenes.copy(), np.full(len(pairs), -1)
    for mixed, pure in MIXED.items():
        cand = np.flatnonzero(np.isin(scenes, list(pure)))
        for i in np.flatnonzero(scenes == mixed):
            d = (bits[cand] != bits[i]).sum(axis=1)
            j = d.argmin()
            nn_dist[i] = d[j]
            if d[j] <= args.max_dist:
                location[i] = scenes[cand[j]]

    df = pd.DataFrame({"stem": [p.stem for p in pairs], "scene": scenes,
                       "location": location, "nn_dist": nn_dist})
    out = ROOT / "data" / "meta" / "locations.csv"
    df.to_csv(out, index=False)
    for mixed in MIXED:
        sub = df[df.scene == mixed]
        moved = sub[sub.location != mixed]
        print(f"scene {mixed}: {len(moved)}/{len(sub)} frames reassigned -> "
              f"{moved.location.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
