# Zero-shot COCO RT-DETRv2-R18 (no fine-tuning)

## Results on `test`

| Model | mAP@0.5 | 95% CI | mAP@0.5:0.95 | 95% CI | P | R | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| zero-shot R18 rgb | 0.775 | [0.706, 0.838] | 0.428 | [0.373, 0.484] | 0.885 | 0.686 | 35.4 | 41.1 |
| zero-shot R18 thermal | 0.746 | [0.630, 0.825] | 0.465 | [0.356, 0.566] | 0.798 | 0.651 | 35.1 | 39.2 |

P/R at confidence ≥ 0.5, IoU 0.5. CIs: location-level bootstrap (1000 resamples).

### By lighting condition (mAP@0.5 / mAP@0.5:0.95)

| Model | dark | dim | lit |
|---|---|---|---|
| zero-shot R18 rgb | 0.694 / 0.359 | 0.808 / 0.438 | 0.837 / 0.486 |
| zero-shot R18 thermal | 0.613 / 0.368 | 0.776 / 0.459 | 0.819 / 0.552 |

### By box height (mAP@0.5): small <150 px, medium 150–250, large ≥250

| Model | small | medium | large | RGB-visible boxes only |
|---|---|---|---|---|
| zero-shot R18 rgb | 0.474 | 0.745 | 0.901 | 0.792 |
| zero-shot R18 thermal | 0.484 | 0.620 | 0.899 | 0.752 |

### By location (mAP@0.5)

| Model | 19 | 20 | 21 | 22 | 23 | 24 | 26 |
|---|---|---|---|---|---|---|---|
| zero-shot R18 rgb | 0.817 | 0.826 | 0.787 | 0.863 | 0.760 | 0.565 | 0.959 |
| zero-shot R18 thermal | 0.844 | 0.863 | 0.678 | 0.822 | 0.661 | 0.467 | 0.799 |

### Paired differences vs zero-shot R18 rgb (location bootstrap, 95% CI)

| Model | Δ mAP@0.5 | CI | Δ mAP@0.5:0.95 | CI | Δ mAP@0.5 dark | CI |
|---|---|---|---|---|---|---|
| zero-shot R18 thermal | -0.030 | [-0.094, 0.009] | +0.037 | [-0.051, 0.108] | -0.081 | [-0.108, 0.049] |
