# P3 E07 — Ordinal Classification Experiment

## Objective
Test whether ordinal classification improves cyclone intensity categorization.

## Class Ordering (confirmed from project code)

| Index | Class | Severity |
|-------|-------|----------|
| 0 | Depression | Weakest |
| 1 | Deep Depression |  |
| 2 | Cyclonic Storm |  |
| 3 | Severe Cyclonic Storm |  |
| 4 | Very Severe Cyclonic Storm |  |
| 5 | Extremely Severe Cyclonic Storm |  |
| 6 | Super Cyclonic Storm | Strongest |

## Results

| Approach | Val F1 | Test Acc | Test F1 | Test BalAcc | MAE | ±1 | ±2 |
|----------|--------|----------|---------|-------------|-----|----|----|
| LightGBM Classification (champion baseline) | 0.1994 | 47.00% | 0.3744 | 38.26% | 0.8341 | 78.03% | 94.01% |
| LightGBM Classification (original params) | 0.1847 | 45.93% | 0.3541 | 36.03% | 0.8510 | 78.49% | 93.86% |
| Ordinal Regression (LightGBM) | 0.2783 | 39.78% | 0.3442 | 34.50% | 0.8095 | 84.18% | 95.24% |
| Cumulative Binary (LightGBM) | 0.2015 | 44.24% | 0.3507 | 34.82% | 0.7819 | 81.57% | 96.77% |
| Ordinal Ridge Regression | 0.1177 | 44.39% | 0.2285 | 28.63% | 1.0538 | 68.36% | 87.71% |
| CatBoost Ordinal Regression | 0.2692 | 41.78% | 0.3609 | 36.36% | 0.7773 | 84.49% | 96.31% |
| Ordinal Probability Rounding (LightGBM) | 0.2162 | 38.86% | 0.3367 | 34.56% | 0.7849 | 85.25% | 97.70% |

## Decision: NO_IMPROVEMENT

**Final P3 Champion**: LightGBM Classification (champion baseline)

### Champion Metrics
- Accuracy: 47.00%
- Macro-F1: 0.3744
- Weighted-F1: 0.4525
- Balanced Accuracy: 38.26%
- MAE: 0.8341
- ±1: 78.03%
- ±2: 94.01%

### Caveats
- Validation set missing ESCS and SuCS classes - val F1 is unreliable for model selection
- SST has 28% missing values - StandardScaler propagates NaN; tree models handle natively
- SuCS has only 25 training samples in train, 0 in test - cannot be evaluated
- Class imbalance: Depression=40% of training data
- Ordinal approaches tested: cumulative binary, ordinal regression, probability rounding, ridge
- Ordinal modeling did not improve macro-F1 or accuracy over standard classification
- The ordinal structure is already captured by standard classifiers via feature relationships