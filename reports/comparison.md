# RGB vs thermal vs fusion (RT-DETRv2-R18, seeds 1-2)

## Results on `test`

| Model | mAP@0.5 | 95% CI | mAP@0.5:0.95 | 95% CI | P | R | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| tir-rgb-s1 | 0.880 | [0.828, 0.933] | 0.514 | [0.449, 0.571] | 0.755 | 0.869 | 33.3 | 39.5 |
| tir-rgb-s2 | 0.875 | [0.826, 0.922] | 0.515 | [0.455, 0.566] | 0.742 | 0.861 | 33.3 | 39.5 |
| tir-thermal-s1 | 0.960 | [0.951, 0.968] | 0.685 | [0.608, 0.744] | 0.930 | 0.917 | 31.8 | 37.1 |
| tir-thermal-s2 | 0.964 | [0.950, 0.973] | 0.679 | [0.602, 0.738] | 0.944 | 0.917 | 31.8 | 37.1 |
| tir-fused4-s1 | 0.949 | [0.931, 0.962] | 0.617 | [0.553, 0.676] | 0.902 | 0.889 | 32.6 | 38.8 |
| tir-fused4-s2 | 0.947 | [0.930, 0.963] | 0.621 | [0.564, 0.673] | 0.881 | 0.887 | 32.6 | 38.8 |
| late-nms-s1 | 0.935 | [0.919, 0.951] | 0.669 | [0.591, 0.729] | 0.687 | 0.943 | – | – |

P/R at confidence ≥ 0.5, IoU 0.5. CIs: location-level bootstrap (1000 resamples).

### By lighting condition (mAP@0.5 / mAP@0.5:0.95)

| Model | dark | dim | lit |
|---|---|---|---|
| tir-rgb-s1 | 0.840 / 0.455 | 0.896 / 0.525 | 0.903 / 0.557 |
| tir-rgb-s2 | 0.840 / 0.469 | 0.890 / 0.522 | 0.893 / 0.549 |
| tir-thermal-s1 | 0.957 / 0.649 | 0.962 / 0.668 | 0.963 / 0.732 |
| tir-thermal-s2 | 0.955 / 0.642 | 0.966 / 0.662 | 0.969 / 0.726 |
| tir-fused4-s1 | 0.943 / 0.600 | 0.953 / 0.607 | 0.953 / 0.649 |
| tir-fused4-s2 | 0.944 / 0.610 | 0.952 / 0.616 | 0.950 / 0.645 |
| late-nms-s1 | 0.923 / 0.629 | 0.936 / 0.650 | 0.945 / 0.721 |

### By box height (mAP@0.5): small <150 px, medium 150–250, large ≥250

| Model | small | medium | large | RGB-visible boxes only |
|---|---|---|---|---|
| tir-rgb-s1 | 0.767 | 0.850 | 0.945 | 0.886 |
| tir-rgb-s2 | 0.705 | 0.850 | 0.945 | 0.880 |
| tir-thermal-s1 | 0.800 | 0.954 | 0.986 | 0.960 |
| tir-thermal-s2 | 0.826 | 0.955 | 0.988 | 0.964 |
| tir-fused4-s1 | 0.756 | 0.926 | 0.986 | 0.950 |
| tir-fused4-s2 | 0.774 | 0.919 | 0.986 | 0.948 |
| late-nms-s1 | 0.814 | 0.934 | 0.973 | 0.935 |

### By location (mAP@0.5)

| Model | 19 | 20 | 21 | 22 | 23 | 24 | 26 |
|---|---|---|---|---|---|---|---|
| tir-rgb-s1 | 0.878 | 0.865 | 0.934 | 0.930 | 0.945 | 0.714 | 0.976 |
| tir-rgb-s2 | 0.875 | 0.861 | 0.915 | 0.922 | 0.936 | 0.713 | 0.967 |
| tir-thermal-s1 | 0.966 | 0.980 | 0.967 | 0.955 | 0.971 | 0.931 | 0.971 |
| tir-thermal-s2 | 0.973 | 0.977 | 0.964 | 0.957 | 0.975 | 0.920 | 0.977 |
| tir-fused4-s1 | 0.952 | 0.969 | 0.967 | 0.937 | 0.973 | 0.895 | 0.971 |
| tir-fused4-s2 | 0.944 | 0.973 | 0.965 | 0.939 | 0.970 | 0.895 | 0.980 |
| late-nms-s1 | 0.945 | 0.941 | 0.949 | 0.947 | 0.961 | 0.885 | 0.971 |

### Paired differences vs tir-thermal-s1 (location bootstrap, 95% CI)

| Model | Δ mAP@0.5 | CI | Δ mAP@0.5:0.95 | CI | Δ mAP@0.5 dark | CI |
|---|---|---|---|---|---|---|
| tir-rgb-s1 | -0.080 | [-0.124, -0.031] | -0.171 | [-0.220, -0.087] | -0.117 | [-0.218, -0.019] |
| tir-rgb-s2 | -0.085 | [-0.126, -0.044] | -0.171 | [-0.216, -0.098] | -0.118 | [-0.214, -0.028] |
| tir-thermal-s2 | +0.004 | [-0.004, 0.007] | -0.006 | [-0.021, 0.002] | -0.003 | [-0.012, 0.003] |
| tir-fused4-s1 | -0.012 | [-0.021, -0.003] | -0.068 | [-0.089, -0.031] | -0.014 | [-0.036, 0.005] |
| tir-fused4-s2 | -0.014 | [-0.022, -0.003] | -0.064 | [-0.086, -0.026] | -0.013 | [-0.036, 0.004] |
| late-nms-s1 | -0.025 | [-0.037, -0.013] | -0.016 | [-0.022, -0.009] | -0.034 | [-0.050, -0.011] |

Notes:
- Latency: measured once per architecture on the local RTX 4050 (batch 1, fp32, end to end incl.
  pre/post-processing); seed-2 models share the seed-1 measurement (identical architecture). Late fusion
  runs both single-modality models, so its latency is roughly the sum (~65 ms) and is not listed.
- Late fusion method and IoU threshold were tuned on val only (NMS, IoU 0.7 won over WBF).
