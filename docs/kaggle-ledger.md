# Kaggle GPU ledger

Weekly quota: 30 GPU-h, max 2 concurrent GPU sessions, 12 h per session. Check with `uv run kaggle quota`.
Accelerator: `NvidiaTeslaT4` (T4 ×2, DDP). Data: `afradhossain/llvip-dataset` (verified identical to the official
LLVIP release) + private `yashmanisingh/llvip-thermal-imagery-annotations`.

| Date (UTC) | Kernel | Purpose | Epochs | Wall time | Quota after |
|---|---|---|---|---|---|
| 2026-10-23 | llvip-mirror-md5 | hash the mirror (CPU only) | – | – | 0.71 h used |
| 2026-10-23 | tir-smoke-fused4 | end-to-end smoke test, stops after epoch 1 | 1 | ~15 min | |
| 2026-10-23 | tir-smoke-fused4-resume | resume from smoke output, epoch 2 + predictions | 1 | ~12 min | 1.35 h used |
| 2026-10-23 | tir-rgb-s1 | Phase 3 RGB, seed 1 | 24 | | |
| 2026-10-23 | tir-thermal-s1 | Phase 3 thermal, seed 1 | 24 | | |
| 2026-10-23 | tir-fused4-s1 | Phase 4 early fusion, seed 1 (queued for a free slot) | 24 | | |

Measured on Kaggle (fused4, 2×T4, fp16): 13 img/s with the first data loader, 22 img/s after moving
colour jitter after the resize (~8 min/epoch). The 4 Kaggle vCPUs limit data loading, not the GPUs.
