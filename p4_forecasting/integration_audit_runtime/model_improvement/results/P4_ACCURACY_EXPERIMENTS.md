# P4 Accuracy Experiments Report

## Executive Summary

Comprehensive accuracy improvement experiments were conducted on the cyclone track forecasting system. The **LightGBM displacement model** (26-feature tabular representation, per-horizon lat/lon regressors) remains the champion model with test track errors of **35.13 / 73.07 / 159.82 km** at 6h / 12h / 24h horizons.

## Baselines (Test Set: 198 samples)

| Model | 6h (km) | 12h (km) | 24h (km) |
|-------|---------|----------|----------|
| Persistence | 64.69 | 123.94 | 227.33 |
| Movement Vector | 38.05 | 80.50 | 180.66 |
| **LightGBM (Champion)** | **35.13** | **73.07** | **159.82** |

## Experiment Results

### E06: Movement-Vector + LightGBM Blend

Blends movement-vector and LightGBM predictions with per-horizon weights chosen on validation:
- Weights: w=0.55 (6h), w=0.50 (12h), w=0.45 (24h)
- **Validation**: IMPROVED over both components at all horizons
- **Test**: 35.75 / 74.66 / 162.70 km (beats MV by 6-10%, trails LightGBM by ~2%)

### E07: Boosting Algorithm Comparison

| Algorithm | 6h | 12h | 24h | vs LightGBM |
|-----------|-----|-----|-----|-------------|
| LightGBM | 35.13 | 73.07 | 159.82 | baseline |
| CatBoost | 35.07 | 71.72 | 159.45 | -0.06 / -1.35 / -0.38 |
| XGBoost | 36.36 | 71.14 | 153.28 | +1.23 / -1.93 / -6.54 |
| RandomForest | 35.52 | 72.69 | 166.35 | +0.39 / -0.38 / +6.53 |

CatBoost shows marginal improvement at 6h/12h. XGBoost improves at 24h. All algorithms beat movement-vector baseline.

### E08: LightGBM Hyperparameter Search

Best config from 200 random samples: depth=3, lr=0.1, subsample=0.7, colsample=0.7, lambda=0.001
- Test: 35.24 / 71.99 / 161.45 km
- **Finding**: Baseline config (depth=5, lr=0.05) was already near-optimal. Shallower trees with faster learning rate marginally better on validation but similar on test.

### E09: Feature Ablation Study

| Feature Group | # Features | Test Delta (km) | Importance |
|---------------|-----------|-----------------|------------|
| state0 (current state) | 7 | +5.98 | CRITICAL |
| velocity_recent | 4 | +3.79 | HIGH |
| velocity_old | 2 | +0.89 | MODERATE |
| rolling_mean | 2 | +0.47 | LOW |
| acceleration | 2 | +0.37 | LOW |
| tendencies | 2 | +0.22 | LOW |
| environment | 3 | +0.04 | NEGLIGIBLE |
| trend_24h | 2 | -0.24 | HARMFUL |
| turning | 2 | -0.98 | HARMFUL |

**Key finding**: Current state and recent velocity are the dominant predictors. Environmental wind and SST features add negligible value with current engineering.

### E10: Error Analysis

**Error Distribution**:
- 6h: mean=35.1, median=29.6, p90=64.7, max=198.6 km
- 12h: mean=73.1, median=64.6, p90=131.0, max=306.7 km
- 24h: mean=159.8, median=142.3, p90=297.7, max=562.2 km

**By Storm Intensity**:
- Weak storms (wind < 1st quartile): 45.7 / 100.8 / 230.7 km (2x error of strong storms)
- Moderate storms: 29.1 / 61.3 / 131.3 km
- Strong storms: 31.6 / 56.1 / 113.7 km

**Key insight**: Weak storms (low wind speed) are hardest to track, with 2x the error of strong storms at 24h.

## Conclusions

1. **LightGBM displacement model is well-tuned** - alternative configurations and algorithms show marginal or no improvement
2. **Feature engineering matters more than algorithm choice** - state0 + velocity features account for most predictive power
3. **Weak storms remain challenging** - environmental features may need better engineering to help here
4. **Ensemble blending helps on validation** but shows limited generalization to test

## Files Produced

- `P4_ACCURACY_EXPERIMENTS.json` - Aggregated results (machine-readable)
- `P4_E06_BLEND.json` - Blend experiment details
- `P4_E07_BOOSTING_COMPARE.json` - Algorithm comparison
- `P4_E08_LGBM_SEARCH.json` - Hyperparameter search results
- `P4_E09_FEATURE_ABLATION.json` - Feature importance analysis
- `P4_E10_ERROR_ANALYSIS.json` - Error distribution and segmentation
