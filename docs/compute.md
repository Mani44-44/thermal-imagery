# Compute measurements

Training probe from `scripts/gpu_check.py` (random data, AMP, AdamW).

| Date | Device | Model | Size | AMP | Max batch | it/s | img/s | Peak GB |
|---|---|---|---|---|---|---|---|---|
| 2026-10-23 | NVIDIA GeForce RTX 4050 Laptop GPU (Olympus) | PekingU/rtdetr_v2_r18vd | 640 | bf16 | 8 | 2.39 | 19.1 | 3.58 |
| 2026-10-23 | NVIDIA GeForce RTX 4050 Laptop GPU (Olympus) | PekingU/rtdetr_v2_r50vd | 640 | bf16 | 4 | 2.08 | 8.3 | 4.17 |
