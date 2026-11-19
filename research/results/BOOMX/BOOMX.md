# BOOMX results (exploratory, research/prereg/BOOMX.md)

## direction: up-first vs down-first (8%)

Rows 18,752; base rate 0.500.

| model | 2023 | 2024 | 2025 |
|---|---|---|---|
| CatBoost | 0.534 (conf 0.53/0.57) | 0.527 (conf 0.54/0.55) | 0.492 (conf 0.47/0.55) |
| LightGBM | 0.541 (conf 0.56/0.57) | 0.518 (conf 0.50/0.54) | 0.495 (conf 0.46/0.54) |
| XGBoost | 0.544 (conf 0.55/0.60) | 0.522 (conf 0.53/0.53) | 0.492 (conf 0.48/0.52) |
| ExtraTrees | 0.549 (conf 0.60/0.59) | 0.529 (conf 0.54/0.50) | 0.499 (conf 0.49/0.56) |
| Logistic | 0.520 (conf 0.52/0.54) | 0.505 (conf 0.48/0.52) | 0.476 (conf 0.39/0.51) |
| Tree depth 3 | 0.502 (conf 0.52/0.48) | 0.513 (conf 0.51/0.51) | 0.498 (conf 0.48/0.53) |
| CatBoost top 10 | 0.498 (conf 0.47/0.51) | 0.514 (conf 0.54/0.53) | 0.497 (conf 0.48/0.54) |

Strongest single inputs (AUC on its own; 0.5 = nothing; low/high tenth label rate):

- score: AUC 0.476, lowest tenth 0.521, highest tenth 0.459
- rng_atr: AUC 0.482, lowest tenth 0.518, highest tenth 0.484
- mom12_1: AUC 0.483, lowest tenth 0.509, highest tenth 0.501
- rz_2: AUC 0.484, lowest tenth 0.513, highest tenth 0.484
- rsi14: AUC 0.484, lowest tenth 0.515, highest tenth 0.454
- hi52: AUC 0.484, lowest tenth 0.510, highest tenth 0.466
- pm_dv: AUC 0.484, lowest tenth 0.517, highest tenth 0.473
- vrel20: AUC 0.485, lowest tenth 0.523, highest tenth 0.478
- vr_0: AUC 0.485, lowest tenth 0.523, highest tenth 0.478
- lo52: AUC 0.485, lowest tenth 0.533, highest tenth 0.493
- rg_0: AUC 0.486, lowest tenth 0.504, highest tenth 0.487
- r60: AUC 0.486, lowest tenth 0.523, highest tenth 0.476
- r120: AUC 0.486, lowest tenth 0.529, highest tenth 0.494
- gap: AUC 0.486, lowest tenth 0.494, highest tenth 0.451
- d_sma200: AUC 0.486, lowest tenth 0.521, highest tenth 0.474

SHAP (CatBoost, 2023-2025/09), top 15, with the sign of the effect:

- gap: 0.0795, higher input -> less (-0.04)
- rz_7: 0.0542, higher input -> more (+0.77)
- pm_dv: 0.0536, higher input -> less (-0.09)
- rz_5: 0.0414, higher input -> less (-0.90)
- log_dv: 0.0376, higher input -> more (+0.76)
- spy_vol20: 0.0365, higher input -> more (+0.16)
- spy_r5: 0.0343, higher input -> more (+0.06)
- spy_r1: 0.0343, higher input -> more (+0.58)
- vr_9: 0.0336, higher input -> less (-0.47)
- rz_4: 0.0329, higher input -> less (-0.75)
- pm_gap: 0.0320, higher input -> less (-0.48)
- pre_fb_max_neg: 0.0312, higher input -> less (-0.84)
- rz_2: 0.0283, higher input -> less (-0.37)
- vr_2: 0.0275, higher input -> more (+0.38)
- mom12_1: 0.0261, higher input -> more (+0.09)

Input groups' share of SHAP: price trend 37%, volume and size 21%, volatility and range 16%, gap and pre-market 11%, market 8%, news 5%, BOOM1 score 1%

Depth-3 tree (2019-2022):

```
|--- gap <= 0.048
|   |--- rz_0 <= -2.311
|   |   |--- class: 0
|   |--- rz_0 >  -2.311
|   |   |--- spy_r1 <= 0.012
|   |   |   |--- class: 1
|   |   |--- spy_r1 >  0.012
|   |   |   |--- class: 1
|--- gap >  0.048
|   |--- spy_d200 <= 0.071
|   |   |--- class: 0
|   |--- spy_d200 >  0.071
|   |   |--- class: 0

```

## direction: close above open

Rows 33,770; base rate 0.452.

| model | 2023 | 2024 | 2025 |
|---|---|---|---|
| CatBoost | 0.537 (conf 0.55/0.60) | 0.513 (conf 0.46/0.61) | 0.538 (conf 0.44/0.65) |
| LightGBM | 0.537 (conf 0.53/0.58) | 0.512 (conf 0.46/0.56) | 0.535 (conf 0.46/0.63) |
| XGBoost | 0.537 (conf 0.55/0.60) | 0.520 (conf 0.50/0.58) | 0.536 (conf 0.47/0.65) |
| ExtraTrees | 0.527 (conf 0.52/0.57) | 0.523 (conf 0.48/0.60) | 0.538 (conf 0.47/0.64) |
| Logistic | 0.524 (conf 0.48/0.57) | 0.505 (conf 0.46/0.58) | 0.533 (conf 0.47/0.64) |
| Tree depth 3 | 0.511 (conf 0.48/0.54) | 0.506 (conf 0.47/0.55) | 0.530 (conf 0.50/0.60) |
| CatBoost top 10 | 0.517 (conf 0.46/0.55) | 0.509 (conf 0.47/0.56) | 0.527 (conf 0.48/0.63) |

Strongest single inputs (AUC on its own; 0.5 = nothing; low/high tenth label rate):

- maxret20: AUC 0.470, lowest tenth 0.479, highest tenth 0.386
- vol60: AUC 0.473, lowest tenth 0.493, highest tenth 0.384
- vol20: AUC 0.474, lowest tenth 0.482, highest tenth 0.380
- bb_width: AUC 0.474, lowest tenth 0.489, highest tenth 0.399
- rg_0: AUC 0.476, lowest tenth 0.491, highest tenth 0.409
- rg_5: AUC 0.476, lowest tenth 0.479, highest tenth 0.397
- rg_2: AUC 0.477, lowest tenth 0.485, highest tenth 0.417
- atr14: AUC 0.477, lowest tenth 0.493, highest tenth 0.394
- rg_3: AUC 0.477, lowest tenth 0.485, highest tenth 0.419
- pm_dv: AUC 0.478, lowest tenth 0.479, highest tenth 0.408
- rg_6: AUC 0.478, lowest tenth 0.484, highest tenth 0.397
- rg_8: AUC 0.478, lowest tenth 0.491, highest tenth 0.405
- rg_9: AUC 0.478, lowest tenth 0.472, highest tenth 0.406
- rg_7: AUC 0.478, lowest tenth 0.484, highest tenth 0.406
- score: AUC 0.478, lowest tenth 0.467, highest tenth 0.408

SHAP (CatBoost, 2023-2025/09), top 15, with the sign of the effect:

- pm_gap: 0.0644, higher input -> less (-0.57)
- pm_dv: 0.0599, higher input -> less (-0.18)
- spy_d200: 0.0404, higher input -> less (-0.84)
- spy_r1: 0.0325, higher input -> less (-0.24)
- vol60: 0.0309, higher input -> less (-0.84)
- spy_r5: 0.0288, higher input -> more (+0.27)
- rz_7: 0.0280, higher input -> more (+0.75)
- spy_vol20: 0.0260, higher input -> less (-0.18)
- vrel5_20: 0.0255, higher input -> more (+0.71)
- vr_8: 0.0242, higher input -> more (+0.36)
- vr_9: 0.0240, higher input -> less (-0.27)
- oc: 0.0229, higher input -> less (-0.74)
- rz_4: 0.0223, higher input -> less (-0.87)
- log_dv: 0.0215, higher input -> more (+0.56)
- vr_4: 0.0195, higher input -> less (-0.54)

Input groups' share of SHAP: price trend 33%, volume and size 20%, volatility and range 18%, gap and pre-market 12%, market 11%, news 3%, BOOM1 score 1%, other 0%

Depth-3 tree (2019-2022):

```
|--- vol60 <= 0.109
|   |--- spy_d200 <= 0.074
|   |   |--- pm_gap <= -0.011
|   |   |   |--- class: 1
|   |   |--- pm_gap >  -0.011
|   |   |   |--- class: 0
|   |--- spy_d200 >  0.074
|   |   |--- spy_vol20 <= 0.013
|   |   |   |--- class: 0
|   |   |--- spy_vol20 >  0.013
|   |   |   |--- class: 1
|--- vol60 >  0.109
|   |--- maxret20 <= 0.238
|   |   |--- vr_8 <= 0.659
|   |   |   |--- class: 0
|   |   |--- vr_8 >  0.659
|   |   |   |--- class: 1
|   |--- maxret20 >  0.238
|   |   |--- r1 <= -0.062
|   |   |   |--- class: 0
|   |   |--- r1 >  -0.062
|   |   |   |--- class: 0

```

## boom: 8%+ move next session

Rows 3,876,280; base rate 0.039.

| model | 2023 | 2024 | 2025 |
|---|---|---|---|
| CatBoost | 0.922 (top20 0.50) | 0.936 (top20 0.61) | 0.924 (top20 0.68) |
| LightGBM | 0.927 (top20 0.50) | 0.939 (top20 0.61) | 0.923 (top20 0.68) |
| XGBoost | 0.926 (top20 0.50) | 0.938 (top20 0.61) | 0.927 (top20 0.67) |
| ExtraTrees | 0.921 (top20 0.47) | 0.935 (top20 0.58) | 0.917 (top20 0.65) |
| Logistic | 0.918 (top20 0.47) | 0.931 (top20 0.58) | 0.919 (top20 0.65) |
| Tree depth 3 | 0.841 (top20 0.40) | 0.870 (top20 0.49) | 0.849 (top20 0.57) |
| CatBoost top 10 | 0.923 (top20 0.46) | 0.934 (top20 0.57) | 0.918 (top20 0.64) |

Strongest single inputs (AUC on its own; 0.5 = nothing; low/high tenth label rate):

- atr14: AUC 0.889, lowest tenth 0.001, highest tenth 0.236
- rg_0: AUC 0.878, lowest tenth 0.001, highest tenth 0.227
- vol20: AUC 0.862, lowest tenth 0.002, highest tenth 0.210
- rg_1: AUC 0.861, lowest tenth 0.002, highest tenth 0.212
- vol60: AUC 0.852, lowest tenth 0.002, highest tenth 0.191
- rg_2: AUC 0.849, lowest tenth 0.002, highest tenth 0.200
- rg_3: AUC 0.843, lowest tenth 0.002, highest tenth 0.194
- rg_4: AUC 0.836, lowest tenth 0.002, highest tenth 0.188
- bb_width: AUC 0.832, lowest tenth 0.002, highest tenth 0.191
- rg_5: AUC 0.830, lowest tenth 0.002, highest tenth 0.183
- minret20: AUC 0.171, lowest tenth 0.179, highest tenth 0.003
- rg_6: AUC 0.826, lowest tenth 0.003, highest tenth 0.180
- rg_7: AUC 0.822, lowest tenth 0.003, highest tenth 0.176
- rg_8: AUC 0.817, lowest tenth 0.004, highest tenth 0.174
- rg_9: AUC 0.813, lowest tenth 0.004, highest tenth 0.170

SHAP (CatBoost, 2023-2025/09), top 15, with the sign of the effect:

- vol20: 0.6864, higher input -> more (+0.49)
- atr14: 0.3484, higher input -> more (+0.48)
- log_dv: 0.3161, higher input -> less (-0.95)
- rg_0: 0.2306, higher input -> more (+0.94)
- rg_1: 0.1137, higher input -> more (+0.89)
- vr_0: 0.0711, higher input -> more (+0.83)
- vrel20: 0.0685, higher input -> more (+0.79)
- vol60: 0.0601, higher input -> more (+0.66)
- pre_fb_max_neg: 0.0593, higher input -> more (+0.74)
- log_px: 0.0560, higher input -> less (-0.84)
- rg_5: 0.0542, higher input -> more (+0.65)
- spy_vol20: 0.0522, higher input -> more (+0.20)
- pre_articles: 0.0521, higher input -> more (+0.87)
- minret20: 0.0516, higher input -> less (-0.34)
- rg_4: 0.0514, higher input -> more (+0.73)

Input groups' share of SHAP: volatility and range 59%, volume and size 18%, price trend 11%, news 7%, market 5%, gap and pre-market 0%

Depth-3 tree (2019-2022):

```
|--- atr14 <= 0.075
|   |--- atr14 <= 0.054
|   |   |--- pre_fb_max_neg <= 0.075
|   |   |   |--- class: 0
|   |   |--- pre_fb_max_neg >  0.075
|   |   |   |--- class: 0
|   |--- atr14 >  0.054
|   |   |--- rg_0 <= 0.077
|   |   |   |--- class: 0
|   |   |--- rg_0 >  0.077
|   |   |   |--- class: 0
|--- atr14 >  0.075
|   |--- rg_0 <= 0.115
|   |   |--- atr14 <= 0.101
|   |   |   |--- class: 0
|   |   |--- atr14 >  0.101
|   |   |   |--- class: 0
|   |--- rg_0 >  0.115
|   |   |--- bb_width <= 0.605
|   |   |   |--- class: 0
|   |   |--- bb_width >  0.605
|   |   |   |--- class: 1

```
