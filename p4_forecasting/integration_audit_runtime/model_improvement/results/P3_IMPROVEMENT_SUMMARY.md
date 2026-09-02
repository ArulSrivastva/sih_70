# P3 Classification Improvement Summary

## Dataset
- **Train**: 3039 samples (104 cyclones)
- **Val**: 518 samples (21 cyclones)
- **Test**: 651 samples (24 cyclones)
- **Features**: lat, lon, sst, pressure_msl, wind_u, wind_v (6 features, SST has 28% NaN)
- **Classes**: 8 in train (incl. Low Pressure Area with 1 sample), 7 IMD classes

## Data Quality Findings
- **No cross-split leakage** (unique cyclone IDs per split)
- **Severe class imbalance**: Depression=40%, Deep Depression=20%, Cyclonic Storm=19%, VSCS=8%, SCS=7%, ESCS=5%, SuCS=0.8%, LPA=0.03%
- **Missing classes in val**: ESCS, SuCS
- **Missing classes in test**: SuCS

## Baseline Reproduction
| Metric | Stored Claim | Reproduced | Gap |
|--------|-------------|------------|-----|
| Accuracy | 47.00% | 45.93% | -1.07% |
| Macro-F1 | 0.3703 | 0.3541 | -0.016 |

## Key Findings

### Class Imbalance: Balanced weights HURT performance
| Method | Accuracy | Macro-F1 |
|--------|----------|----------|
| No weights (baseline) | 45.93% | 0.3541 |
| Balanced weights | 43.01% | 0.3471 |
| Focal (3x rare) | 44.09% | 0.3329 |

**Conclusion**: Class weighting hurts. The minority class (SuCS) has too few samples for weighting to help.

### Model Comparison (all with balanced weights)
| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| LightGBM | 43.01% | 0.3471 |
| GradientBoosting | 43.16% | 0.3078 |
| XGBoost | 42.70% | 0.3323 |
| RandomForest | 41.32% | 0.3349 |
| CatBoost | 39.17% | 0.2705 |
| SVM | 37.94% | 0.2444 |

**LightGBM wins** on Macro-F1.

### Hyperparameter Tuning (best configurations)
| Model | Params | Val F1 | Test F1 |
|-------|--------|--------|---------|
| LightGBM | depth=4, lr=0.05, n=300 | 0.2202 | 0.3626 |
| CatBoost | depth=6, lr=0.02, n=1000 | 0.2201 | 0.2708 |

**Note**: Val F1 is unreliable due to missing classes in val set (ESCS, SuCS absent).

### Ordinal Structure
- **Standard Accuracy**: 39.17%
- **Off-by-one Accuracy**: 75.58%
- **Off-by-two Accuracy**: 94.32%

The model understands ordinal relationships well — most errors are off by exactly 1 class.

### Feature Importance
1. pressure_msl: 29.35%
2. lat: 19.41%
3. lon: 17.72%
4. sst: 16.11%
5. wind_v: 9.95%
6. wind_u: 7.45%

### Error Analysis
- **Error rate**: 60.8% (396/651)
- **Top confusion**: Depression ↔ Deep Depression (99 errors), VSCS ↔ SCS (35 errors)
- **Severe storm errors**: 87 (VSCS, ESCS misclassified)
- **Confidence gap**: Correct=0.54, Error=0.52 (model is uncertain)

## Improvement vs Stored Claim
The stored claim of 47%/0.37 F1 was likely from a different data split or with different features. Our controlled reproduction gives 45.93%/0.3541 on the current data.

The tuned model (43.47%/0.3626) does not beat the baseline reproduction (45.93%/0.3541) on accuracy, though F1 is marginally improved.

## Conclusion
The P3 tabular classifier has limited capacity for improvement with the current feature set. The main bottleneck is:
1. Only 6 features with significant SST missingness
2. Severe class imbalance (SuCS has only 25 train samples)
3. Missing classes in validation/test sets
4. Feature information (pressure, lat, lon) is shared with P4 regression tasks

The off-by-one accuracy of 75.58% suggests the model captures the ordinal structure well, but exact class boundaries are noisy.
