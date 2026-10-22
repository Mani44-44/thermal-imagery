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
