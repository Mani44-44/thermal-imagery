# Phase 6: label-error mining and its effect

## Pipeline

1. **Out-of-fold predictions** (`scripts/oof.sh`): two thermal RT-DETRv2-R18 models, each trained on half of
   the train locations (A = 1,2,3,4,8,16,17,18; B = 5,6,7,9,10,12,13,25), predict the other half. Every
   train image gets predictions from a model that never saw it. Fold A stopped early (best checkpoint from
   epoch 4, val mAP@0.5 0.926) after its process hung; fold B ran 16 epochs. Enough for flagging.
2. **Flagging** (`src.quality.flag`): FiftyOne `compute_mistakenness` plus rules (confident unmatched
   prediction = possible missing box, GT with no prediction nearby = possible spurious box, GT matched only
   at IoU 0.3–0.6 = possibly loose box). 1,295 of 10,377 train images flagged; the top 250 by score were
   rendered as visible|infrared panels.
3. **Review**: **the reviewer was the model (Claude), not a human**, working from the panels and
   native-resolution crops for close calls (`data/quality/review.jsonl`, reviewer `claude-model`, DVC-tracked).
   Rules: add a box only when a person is clearly visible in infrared; remove one only when clearly no
   person is there; replace a box only when it is visibly wrong (not merely different from the model's box).
   A CVAT-importable export of the same queue (GT and OOF pre-annotations, COCO 1.0) is in
   `data/quality/cvat/` (`src.quality.export_cvat`) for a human pass.
4. **Apply** (`src.quality.apply`) → `data/annotations/train_v2.json` (DVC, git tag `data-v2`). Only train
   is corrected; val/test keep the original labels.

## What the review found (250 images)

| Verdict | Images |
|---|---|
| fixed | 43 |
| no error | 109 |
| ambiguous (left unchanged) | 98 |

Box changes: **29 added, 25 replaced, 7 removed** (30,238 → 30,260 boxes, 0.2% of boxes touched).

Recurring patterns:
- **Missing**: pedestrians walking behind foliage at the top of scene 09/18 frames, clear in infrared,
  never labelled; people entering at frame edges.
- **Wrong boxes**: thin "sliver" boxes covering a fraction of a person (scene 09), boxes stopping at the
  waist of scooter riders, one person split into two boxes (e.g. 120006, 080947).
- **Spurious**: a decorative wall lamp labelled as a person (scene 01), stray duplicate boxes.
- **Not errors** (the largest flagged group): thermal **reflections of people on the polished monument in
  scene 01** (the model detects them; correctly unlabelled), riders with passengers, people walking in
  pairs that the model merges.
- **Ambiguous**: boxes in the scene-07 tree canopy and thin edge slivers that recur across consecutive
  frames, likely occluded people labelled from video context. Left as is.

## Effect on the model (thermal, 24 epochs, same recipe)

LLVIP test (3,463 images), paired location bootstrap vs `tir-thermal-s1`:

| Run | Labels | mAP@0.5 | mAP@0.5:0.95 | Δ mAP@0.5 [95% CI] | Δ mAP@0.5:0.95 [95% CI] |
|---|---|---|---|---|---|
| tir-thermal-s1 | v1 | 0.960 | 0.685 | ref | ref |
| tir-thermal-s2 | v1 | 0.964 | 0.679 | +0.004 [-0.004, 0.007] | -0.006 [-0.021, 0.002] |
| tir-thermal-v2-s1 | v2 | 0.958 | 0.673 | -0.002 [-0.005, 0.004] | -0.012 [-0.017, 0.005] |
| tir-thermal-v2-s2 | v2 | 0.960 | 0.681 | -0.000 [-0.014, 0.008] | -0.004 [-0.026, 0.009] |

Golden set (402 images) mAP@0.5 / mAP@0.5:0.95: v1 0.953 / 0.677 and 0.963 / 0.673; v2 0.957 / 0.671 and
0.961 / 0.677.

**Result: no measurable effect.** The v1→v2 differences are the same size as the seed-to-seed spread of
the v1 runs (about 0.004–0.006), and every paired CI except one dark-subset delta includes zero.

Why, and what it does and doesn't show:
- 61 box changes out of 30,238 is too small a dose to move a model trained on 10k images.
- The test labels were not corrected and contain the same error types (unlabelled people behind
  foliage). A model that learns to find them is penalised for "false positives" there, so test mAP cannot
  reward the fix. Measuring it would need a corrected (human-reviewed) slice of the test set.
- The loop's value here is the audit itself: it characterised LLVIP's label noise (above) and showed the
  benchmark ranking is not an artefact of it. A human pass in CVAT over the 98 ambiguous images and the
  remaining 1,045 flagged images is the natural next step.
