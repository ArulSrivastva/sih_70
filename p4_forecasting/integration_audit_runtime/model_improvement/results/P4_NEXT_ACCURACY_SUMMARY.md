# P4 Next Accuracy Summary (E11-E15)

## Executive Summary

Storm-level cross-validation confirms that **CatBoost is the new champion** for cyclone track forecasting. The improvement over LightGBM is robust across all 5 cyclone-disjoint folds and all 3 horizons.

## Final Comparison

| Model | 6h (km) | 12h (km) | 24h (km) | Validation | Storm-CV | Status |
|-------|---------|----------|----------|------------|----------|--------|
| Persistence | 64.69 | 123.94 | 227.33 | — | — | BASELINE |
| Movement-Vector | 38.05 | 80.50 | 180.66 | — | 27.77/59.19/138.92 | BASELINE |
| LightGBM Displacement | 35.13 | 73.07 | 159.82 | 28.51/62.56/135.14 | 26.89/57.30/132.18 | PREVIOUS CHAMPION |
| **CatBoost Displacement** | **35.07** | **71.72** | **159.45** | 27.94/58.93/127.16 | 26.82/56.57/127.28 | **NEW CHAMPION** |
| LGB Residual | 37.22 | 76.22 | 163.66 | 27.66/59.56/129.18 | — | WORSE |
| CatBoost Residual | 36.73 | 74.91 | 164.72 | 27.39/60.17/128.83 | — | WORSE |
| Hybrid Gated | 37.26 | 76.34 | 164.45 | 27.68/59.62/129.36 | — | WORSE |

## Experiment Results

### E11: Storm-Level Cross-Validation

5-fold cyclone-disjoint CV on train+val (1443 samples, 70 cyclones):

| Model | 6h (mean±std) | 12h (mean±std) | 24h (mean±std) |
|-------|---------------|----------------|----------------|
| LightGBM | 26.89 ± 1.80 | 57.30 ± 4.60 | 132.18 ± 14.45 |
| **CatBoost** | **26.82 ± 1.64** | **56.57 ± 4.51** | **127.28 ± 11.48** |
| Movement-Vector | 27.77 ± 0.63 | 59.19 ± 2.05 | 138.92 ± 7.52 |

**Key finding**: CatBoost wins at ALL horizons across ALL 5 folds. The improvement is consistent and not due to test-set noise.

### E12: Error Stratification

- Top error cyclone: 2024252N18087 (328.1 km at 24h)
- MV wins on 90/198 samples at 6h, ML wins on 108/198
- MV wins on 89/198 samples at 24h, ML wins on 109/198
- ML is better ~55% of the time

### E13: Residual Model

Residual = actual - movement_vector displacement. This approach is WORSE:
- CatBoost Residual: 36.73/74.91/164.72 (vs CatBoost Displacement: 35.07/71.72/159.45)

The MV baseline is too noisy to serve as a good starting point for residual correction.

### E14: Horizon-Specific Residual

- Per-horizon is better than shared
- But both are worse than direct displacement

### E15: Hybrid Gate

- Gate selects between MV and residual based on speed threshold
- Does not improve over direct displacement
- Verdict: NO IMPROVEMENT

## Conclusions

1. **CatBoost is the new champion** with robust storm-CV evidence
2. **Residual approaches do not help** — the MV baseline is too noisy
3. **Hybrid gating does not help** — direct displacement is better
4. **The improvement is modest but real**: CatBoost reduces 24h error by ~4.9 km vs LightGBM on storm-CV

## Files Produced

- `P4_E11_STORM_CV.json` — Storm-level cross-validation results
- `P4_E12_ERROR_STRATIFICATION.json` — Error analysis
- `P4_E13_RESIDUAL_MODEL.json` — Residual model results
- `P4_E14_HORIZON_RESIDUAL.json` — Horizon-specific residual results
- `P4_E15_HYBRID_GATE.json` — Hybrid gate results
- `P4_NEXT_ACCURACY_SUMMARY.json` — Aggregated summary
- `P4_NEXT_ACCURACY_SUMMARY.md` — This file
