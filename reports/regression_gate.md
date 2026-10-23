# Golden-set regression gate: evidence

Baseline: `models/thermal_s1` (RT-DETRv2-R18, thermal, seed 1), scored on the 151-image golden CI subset
(`golden/baseline.json`, written by `scripts/promote.py`, on the local RTX 4050).

| Metric | Baseline | Tolerance | Floor |
|---|---|---|---|
| mAP@0.5 | 0.9558 | 0.01 | 0.9458 |
| mAP@0.5:0.95 | 0.6595 | 0.01 | 0.6495 |
| Recall (conf ≥ 0.5) | 0.9003 | 0.02 | 0.8803 |
| mAP@0.5, dark images | 0.9482 | 0.02 | 0.9282 |

## The gate passes the baseline and fails a worse model

```
$ uv run pytest -m regression
4 passed

$ CANDIDATE_MODEL=runs/kaggle/out/tir-fused4-s1/run/best CANDIDATE_MODALITY=fused4 uv run pytest -m regression
AssertionError: mAP50_95 regressed: 0.5885 < baseline 0.6595 - tol 0.01
AssertionError: recall regressed: 0.8571 < baseline 0.9003 - tol 0.02
AssertionError: dark_mAP50 regressed: 0.9102 < baseline 0.9482 - tol 0.02
4 failed
```

The degraded candidate is the seed-1 early-fusion model, which is genuinely worse than thermal-only
on test (mAP@0.5:0.95 0.617 vs 0.685, `reports/comparison.md`).
