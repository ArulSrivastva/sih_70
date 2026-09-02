# P4 Feature Engineering — Final Summary

## Experiment: P3_P4_FEATURE_ENGINEERING

## PART 1: Data Inspection
- 15 columns: 6 features + cyclone_id, season, name, subbasin, timestamp, wind_speed, pressure, category, pre_genesis_favorable
- wind_speed and pressure NOT used as features (available at inference but not in original model)
- 104 train cyclones, 21 val, 24 test — zero cross-split ID overlap
- Timestamps are ordered within storms (3-hourly)

## PART 2: Feature Engineering
16 engineered features created, all physically motivated:

| Feature | Formula | Reason | Inference OK |
|---------|---------|--------|-------------|
| wind_speed_mag | sqrt(wu²+wv²) | Total wind magnitude | Yes |
| wind_dir_sin | wv/|w| | Direction component | Yes |
| wind_dir_cos | wu/|w| | Direction component | Yes |
| abs_lat | |lat| | Distance from equator | Yes |
| lat_sq | lat² | Coriolis nonlinear | Yes |
| lon_sin/cos | sin/cos(lon·π/180) | Cyclic geographic encoding | Yes |
| pressure_deficit | 1013.25-msl | Direct intensity proxy | Yes |
| pressure_sq | deficit² | Nonlinear pressure response | Yes |
| sst_x_pressure | sst·msl | Warm+low pressure interaction | Yes |
| sst_x_lat | sst·|lat| | Warm water + low latitude | Yes |
| pressure_x_lat | msl·|lat| | Pressure-latitude structure | Yes |
| sst_above_26 | sst>26.5 | Cyclogenesis threshold | Yes |
| sst_anomaly | sst-28.77 | Relative warmth | Yes |
| wind_u/v_x_pressure | wind·msl | Wind-pressure interaction | Yes |

## PART 3: SST Missingness
| Method | Val Acc | Val F1 | Test Acc | Test F1 |
|--------|---------|--------|----------|---------|
| Native NaN | 41.12% | 0.2036 | 46.24% | 0.3444 |
| Median impute | 40.73% | 0.1958 | 47.31% | 0.3747 |
| Median+indicator | 38.61% | 0.1799 | 47.77% | 0.3779 |
| KNN impute | 35.71% | 0.1741 | 47.16% | 0.3118 |

**Finding**: Median+indicator gives best test F1 (0.3779) but worst val F1. Val metrics unreliable due to missing classes.

## PART 4: Feature Ablation
| Set | Feats | Val Acc | Val F1 |
|-----|-------|---------|--------|
| E1 original | 6 | 41.12% | 0.2036 |
| E2 +nonlinear | 13 | 40.54% | 0.1969 |
| E3 +interactions | 11 | 39.19% | 0.1856 |
| E4 +both | 18 | 39.19% | 0.1870 |
| E5 +all | 22 | 38.80% | 0.1767 |
| E6 no-SST | 21 | 37.64% | 0.1718 |

**All engineered features hurt. Original 6 features are optimal.**

## PART 5: Hyperparameter Tuning (80 trials)
Best: n_estimators=500, lr=0.02, depth=5, leaves=7, min_child=20,
      bagging=0.85, lambda_l1=5, lambda_l2=0
Val F1: 0.207 | Val Acc: 43.24%

## PART 6: Cross-Validation
| Method | Mean Acc | ± Std | Mean F1 | ± Std |
|--------|----------|-------|---------|-------|
| GroupKFold (storm) | 41.96% | 3.06% | 0.2772 | 0.0602 |
| Stratified | 62.54% | 0.78% | 0.5935 | 0.0194 |

**Critical gap**: StratifiedCV = 62.5%, GroupKFold = 42%. Within-storm correlation inflates naive estimates by ~20%.

## PART 7: Error Analysis
Top adjacent confusions:
- Deep Depression→Depression: 63.7% (65/102)
- Cyclonic Storm→Depression: 39.1% (43/110)
- Very Severe→Severe: 26.4% (24/91)

## PART 8: Champion Selection
Original LightGBM (47.00% / 0.3744) RETAINED.

Final test (tuned model): 46.24% / 0.3458 — worse than champion.

## Decision: NO_IMPROVEMENT
