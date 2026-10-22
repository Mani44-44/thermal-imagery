# Phase 0 data checks: LLVIP

Checked 2026-10-23 on the official Google Drive release.

## Download integrity

- `LLVIP.zip`: 4,003,858,172 bytes, SHA-256
  `b3b55475c093ac54663c467859eb31a82b840b53a7196ad1887cb547e4518b45` (also in `data/raw/LLVIP.zip.sha256`)
- Contents match the bundled README: **15,488 pairs** (12,025 train / 3,463 test), 30,976 images, and
  15,488 VOC XML annotations shared by each visible/infrared pair. Every visible image has an infrared
  partner. All images are 1280×1024.

## Scenes and the official split

- Filenames start with a two-digit prefix `01`–`26`, which gives exactly 26 distinct values, matching the 26
  scenes in the dataset description. **We treat the prefix as the scene ID** (to be confirmed visually
  in Phase 1).
- **The official split is scene-disjoint:** train = scenes 01–18 and 25; test = scenes 19–24 and 26.
  Phase 1 only needs to carve a validation set out of the train scenes.
- Scene sizes range from 319 to 1,518 pairs (08, 09 and 10 are the largest).
- Which two scenes are daytime is not documented. Phase 1 brightness binning will show it.

## Visible/infrared registration

`scripts/check_alignment.py`, 20 random pairs, ECC translation on gradient images:

- Converged on 20/20 pairs. Residual shift is **2.9 px median**, 8.5 px at the 90th percentile and 29.4 px max
  (at 1280×1024).
- The max (29 px, `020183`) had very weak gradient correlation (0.16 → 0.22), so the estimate is
  unreliable.
- All three scene `08` samples show a consistent ~9 px vertical offset. That points to a small fixed offset
  in that capture session. It is about 4–5 px at the 640 px training size, small next to a pedestrian box.
  Re-measure per scene in Phase 1.
- The visual overlays agree: kerbs, poles, manholes and road markings coincide.

**Conclusion:** the pairs are registered well enough for early (channel-stacking) fusion. No registration
step is needed.

Method note: global phase correlation and an edge chamfer search were tried first and gave spurious
large shifts on dark frames and dense foliage. ECC refines locally from zero shift and is robust here.

## Observation for the RGB vs thermal comparison

In dark scenes, some labelled pedestrians are clearly visible in infrared but nearly invisible in the
visible image (e.g. `081485`). Since labels are shared, the RGB-only model is scored against targets
it may physically be unable to see. This is the effect the project is meant to measure, but report it
explicitly (e.g. an "RGB-visible" subset) so RGB-only recall is interpreted fairly.

# Phase 1 findings (2026-10-23)

## Scenes are not quite locations: 25 and 26 are mixed

A near-duplicate scan (256-bit infrared difference hash) found that filename prefixes **25 (train)** and
**26 (test)** are grab-bags, not single scenes:
- 25 holds frames shot at train locations 01–18 plus the daytime footage.
- 26 holds frames shot at test locations 19–24.

`src.data.locations` gives each 25/26 frame the location of its nearest pure-scene frame (hash
distance ≤ 20): 179/423 of 25 and 307/533 of 26 were reassigned. The remaining frames, mostly
daytime, keep 25/26 as their own location. **Location is the unit for the val split and for
bootstrap CIs.**

The first val choice (scene 12) leaked: 188 of its frames nearly duplicated scene-25 frames in
train. After switching to location-based splits, the scan finds **0 cross-split near-duplicates**
(threshold 12 bits), and there are no close train-test matches at all: the official split is clean.

## Splits

| Split | Locations | Images | Boxes | RGB-visible boxes |
|---|---|---|---|---|
| train | 1–10, 12, 13, 16–18, 25 | 10,377 | 30,238 | 95.6% |
| val | 11 (dark), 14 (lit), 15 (dim) | 1,648 | 3,892 | 92.6% |
| test (official) | 19–24, 26 | 3,463 | 8,302 | 92.0% |
| golden (⊂ test) | all test locations | 402 | 979 | |
| golden CI (⊂ golden) | all test locations | 151 | 371 | |

## Labels

- One class (person); 42,437 boxes, 7,727 marked truncated, none marked difficult.
- 5 degenerate boxes (0 px wide) dropped: 020118, 020236, 090437, 091180, 170350.

## Lighting conditions

Measured by mean visible-image luminance: dark < 25, dim 25–60, lit ≥ 60.
- Test: 1,144 dark / 1,224 dim / 1,095 lit.
- "Lit" here means mostly street-lit night. Only location 25 (train) has real daytime footage, so the
  project measures dark-vs-lit night, not night-vs-day.

## Box sizes

Pedestrians are large in LLVIP: median height 225 px, 5th percentile 124 px. COCO area bins put 93% of
boxes in "large" (only 18 "small"), so size breakdowns use box height:
small < 150 px, medium 150–250 px, large ≥ 250 px (native 1280×1024).

## RGB-visible flag

Contrast = max(|mean inside box − mean of surrounding ring|, std inside box) on the grayscale visible image.
Crop montages showed boxes below about 8 are mostly unrecognisable silhouettes, and 9–12 are mostly
recognisable. Threshold: 8, which marks about 4–8% of boxes per split as not RGB-visible. This is a
heuristic, so RGB-only recall is reported on both all boxes and RGB-visible boxes.

## Thermal normalisation

Pooled over train: infrared mean 0.287, std 0.185 (0–1 scale), in `data/meta/norm.json`.
