# P3 Phase 2 -- Final Report

## P3 PHASE 2 RESULT

### Baseline
- Model: LightGBM (n_estimators=300, lr=0.05, max_depth=4, num_leaves=15)
- Features: 6 original (lat, lon, sst, pressure_msl, wind_u, wind_v)
- Test accuracy: 46.08%
- Test macro-F1: 0.3612
- GroupKFold accuracy: 41.82% +/- 2.97%
- GroupKFold macro-F1: 0.2700 +/- 0.0531

### Best candidate
- Model: LightGBM (same as baseline)
- Features: 6 original (no improvement from additional features)
- GroupKFold accuracy: 41.82% +/- 2.97%
- GroupKFold macro-F1: 0.2700 +/- 0.0531
- Test accuracy: 47.00%
- Test macro-F1: 0.3650

### Improvement
- Accuracy delta: +0.92%
- Macro-F1 delta: +0.0038

### Leakage
PASS

### Robustness
NO_ROBUST_IMPROVEMENT

### Final decision
**PHASE1_RETAINED**

### Explanation
1. No feature improvement: temporal/context features showed negligible impact on GroupKFold macro-F1
2. No model improvement: CatBoost, XGBoost, ordinal approaches did not outperform LightGBM
3. Class imbalance strategies ineffective: balanced/moderate/sqrt weighting did not improve macro-F1
4. Data limitations: 6 features, 28% SST missingness, severe class imbalance (SuCS=25 samples)
5. Phase-1 LightGBM champion remains the best available model
