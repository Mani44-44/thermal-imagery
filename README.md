# Thermal Imagery: RGB vs thermal vs fusion pedestrian detection

RT-DETR fine-tuned on LLVIP three ways (RGB only, thermal only, RGB+thermal fusion), with an evaluation
harness, a golden-set regression test in CI, and a DVC-versioned data-quality loop.
See [PLAN.md](PLAN.md) for the phased plan.

## Results (LLVIP test, 3,463 images, 7 held-out locations)

| Arm | mAP@0.5 | mAP@0.5:0.95 | dark mAP@0.5 | small mAP@0.5 | p50 latency |
|---|---|---|---|---|---|
| zero-shot RT-DETR (COCO), thermal input | 0.746 | 0.465 | | | |
| RGB only | 0.880 / 0.875 | 0.514 / 0.515 | 0.840 | 0.767 | 33.3 ms |
| **thermal only** | **0.960 / 0.964** | **0.685 / 0.679** | **0.957** | 0.800 | 31.8 ms |
| early fusion (4-channel) | 0.949 / 0.947 | 0.617 / 0.621 | 0.943 | 0.756 | 32.6 ms |
| late fusion (RGB+thermal, WBF) | 0.957 | 0.681 | 0.956 | 0.800 | 68.0 ms |
| thermal, corrected labels (v2) | 0.958 / 0.960 | 0.673 / 0.681 | 0.960 | | 31.8 ms |

Two seeds per arm (s1 / s2); RT-DETRv2-R18, 24 epochs, 2×T4 on Kaggle; latency is local RTX 4050, fp32,
batch 1. CIs (location-level bootstrap) and per-location tables are in
[reports/comparison.md](reports/comparison.md).

- **Thermal alone wins.** It is 8 mAP@0.5 points better than RGB overall and 12 points better in the dark,
  with tight CIs. LLVIP is a night-time dataset and pedestrians are hot.
- **Fusion does not help here.** Early fusion is 1.2 points below thermal (CI excludes zero) and loses more
  at mAP@0.5:0.95. Tuned late fusion (WBF, weighted 3:1 toward thermal) matches thermal alone within
  0.003 mAP@0.5 but costs two forward passes (68 ms vs 32 ms).
- **The label-error loop found real errors but they didn't move the metric.** 17% of the 250 flagged images
  had a confirmed error (47% for "missing box" flags). Retraining on the corrected labels changed mAP by
  less than the seed spread. See [reports/data_quality.md](reports/data_quality.md). The review was done by
  the model (Claude), not a human; a CVAT export is provided for a human pass.
- **Regression gate:** `pytest -m regression` passes the promoted thermal model and fails the fusion model
  on all 4 golden-set checks ([reports/regression_gate.md](reports/regression_gate.md)).

## Layout

| Path | What |
|---|---|
| `src/data/` | LLVIP conversion, location-disjoint splits, de-duplication, golden-set manifest |
| `src/eval/` | mAP, P/R, size/condition/location breakdowns, bootstrap CIs, latency, reports |
| `src/models/`, `src/train.py`, `src/predict.py` | RT-DETR (3- or 4-channel), training loop, inference |
| `src/fusion/late.py` | box-level late fusion (WBF / NMS), tuned on val |
| `src/quality/` | out-of-fold flagging (FiftyOne mistakenness + rules), review application, CVAT export |
| `tests/` | unit tests (metrics, splits, leakage, pipeline) and the golden-set regression test |
| `scripts/` | Kaggle job runner, OOF training, model promotion |
| `dvc.yaml`, `*.dvc` | data pipeline; raw data, labels v2, models and all run results are DVC-tracked |
| `reports/`, `docs/` | results, data checks, licences, Kaggle GPU ledger |

## Reproduce

```bash
uv sync && dvc pull                         # needs access to the DVC remote (local storage in this project)
dvc repro                                   # splits + golden manifest from raw LLVIP
uv run pytest -m "not regression"           # unit tests
uv run pytest -m regression                 # golden-set gate against golden/baseline.json
CANDIDATE_MODEL=models/fused4_s1 CANDIDATE_MODALITY=fused4 uv run pytest -m regression   # fails (worse model)
python scripts/kaggle_job.py --help         # training runs on Kaggle (see docs/kaggle-ledger.md)
```

## Setup

```bash
uv sync                     # Python 3.11 env with CUDA PyTorch, from uv.lock
uv run python scripts/gpu_check.py
```

## Dataset

This project uses **LLVIP: A Visible-infrared Paired Dataset for Low-light Vision**
([website](https://bupt-ai-cz.github.io/LLVIP/), [paper](https://arxiv.org/abs/2108.10831)).
LLVIP is licensed for non-commercial use only; see [docs/licences.md](docs/licences.md). The images are
not redistributed in this repository.

```bibtex
@inproceedings{jia2021llvip,
  title={LLVIP: A visible-infrared paired dataset for low-light vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3496--3504},
  year={2021}
}
```
