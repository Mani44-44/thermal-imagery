# Compute measurements

Training probe from `scripts/gpu_check.py` (random data, AMP, AdamW).

| Date | Device | Model | Size | AMP | Max batch | it/s | img/s | Peak GB |
|---|---|---|---|---|---|---|---|---|
| 2026-10-23 | NVIDIA GeForce RTX 4050 Laptop GPU (Olympus) | PekingU/rtdetr_v2_r18vd | 640 | bf16 | 8 | 2.39 | 19.1 | 3.58 |
| 2026-10-23 | NVIDIA GeForce RTX 4050 Laptop GPU (Olympus) | PekingU/rtdetr_v2_r50vd | 640 | bf16 | 4 | 2.08 | 8.3 | 4.17 |
| 2026-10-23 | Tesla T4, 1 of 2 (Kaggle; torch 2.11.0+cu128) | PekingU/rtdetr_v2_r18vd | 640 | fp16 | 32 | 0.78 | 25.0 | 13.74 |
| 2026-10-23 | Tesla T4, 1 of 2 (Kaggle; torch 2.11.0+cu128) | PekingU/rtdetr_v2_r50vd | 640 | fp16 | 16 | 0.74 | 11.9 | 14.76 |

## Local vs Kaggle

Throughput is the best images/s across the batch sizes that fit (random data, so real data loading will
lower it). One epoch is about 12,000 training images.

| Model | RTX 4050 Laptop (6 GB) | One Kaggle T4 (16 GB) | T4 speed-up | Epoch: local / 1×T4 / 2×T4 (ideal) |
|---|---|---|---|---|
| R18 | 19.1 img/s @ batch 8 | 26.7 img/s @ batch 16 | ~1.4× | ~10.5 / ~7.5 / ~4 min |
| R50 | 8.3 img/s @ batch 4 | 11.9 img/s @ batch 8–16 | ~1.4× | ~24 / ~17 / ~9 min |

Takeaways:
- One T4 is about 1.4× faster than the laptop GPU, which is held back by its 75 W power cap. The T4 also fits a
  4× larger batch. Throughput flattens at batch 8–16 on the T4, so bigger batches mainly help with
  matching the effective batch size, not speed.
- Kaggle's PyTorch (2.11, CUDA 12.8) differs from the local env (2.14, CUDA 13.0). Fine for training,
  but record versions per run, and run all latency benchmarks locally.
- Kaggle's 30 GPU-h/week at ~4 min/epoch on 2×T4 is roughly 450 R18 epochs/week, enough for the
  3 arms × 3 seeds sweep within a week or two.
