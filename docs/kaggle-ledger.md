# Kaggle GPU ledger

Weekly quota: 30 GPU-h, max 2 concurrent GPU sessions, 12 h per session. Check with `uv run kaggle quota`.
Accelerator: `NvidiaTeslaT4` (T4 ×2, DDP). Data: `afradhossain/llvip-dataset` (verified identical to the official
LLVIP release) + private `yashmanisingh/llvip-thermal-imagery-annotations`.

| Date (UTC) | Kernel | Purpose | Epochs | Wall time | Quota after |
|---|---|---|---|---|---|
| 2026-10-23 | llvip-mirror-md5 | hash the mirror (CPU only) | – | – | 0.71 h used |
| 2026-10-23 | tir-smoke-fused4 | end-to-end smoke test, stops after epoch 1 | 1 | ~15 min | |
| 2026-10-23 | tir-smoke-fused4-resume | resume from smoke output, epoch 2 + predictions | 1 | ~12 min | 1.35 h used |
| 2026-10-23 | tir-rgb-s1 | Phase 3 RGB, seed 1 | 24 | 1.8 h training (38.5 img/s) | |
| 2026-10-23 | tir-thermal-s1 | Phase 3 thermal, seed 1 | 24 | 2.0 h training (34.3 img/s) | |
| 2026-10-23 | tir-fused4-s1 | Phase 4 early fusion, seed 1 | 24 | 2.65 h training (28.6 img/s) | |
| 2026-10-23 | tir-thermal-s2 | seed 2 | 24 | | |
| 2026-10-23 | tir-rgb-s2 | seed 2 | 24 | | 12.3 h used (while s2 running) |
| 2026-10-23 | tir-fused4-s2 | seed 2 | 24 | | |
| 2026-10-23 | tir-thermal-s3 | seed 3 | – | failed on Kaggle (empty log), not retried | |
| 2026-10-23 | tir-thermal-v2-s1 | Phase 6 retrain on train_v2 (first attempt errored, re-pushed) | 24 | | |
| 2026-10-23 | tir-thermal-v2-s2 | Phase 6 retrain on train_v2, seed 2 | 24 | | 21.4 h used, 8.6 h left |

Measured on Kaggle (fused4, 2×T4, fp16): 13 img/s with the first data loader, 22 img/s after moving
colour jitter after the resize (~8 min/epoch). The 4 Kaggle vCPUs limit data loading, not the GPUs.

Not run: rgb-s3 and fused4-s3 (~4.5 GPU-h). Two seeds per arm plus location-bootstrap CIs were judged enough;
the seed-1/seed-2 spread (≤0.005 mAP@0.5) is far below the gaps between arms.
Out-of-fold training for Phase 6 ran locally on the RTX 4050 (scripts/oof.sh).
