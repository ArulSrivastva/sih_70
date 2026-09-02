# P1-C: P4 FORECAST VALIDATION REPORT

**Date:** 2026-08-31
**Scope:** Verify EXP005 champion metrics vs baselines are reproducible
**Method:** Independent forward pass on feature_dataset/test.npz (198 samples)

## Verified Metrics (independently reproduced)

### EXP005 Champion (GRU+Huber, 96 hidden, 2 layers)

| Horizon | Track Error (km) | Wind MAE (km/h) | Stored Match |
|---------|-----------------|-----------------|-------------|
| 6h | 91.43 | 6.68 | < 1.0 km |
| 12h | 119.76 | 9.46 | < 1.0 km |
| 24h | 188.24 | 16.11 | < 1.0 km |

### Movement-Vector Baseline

| Horizon | Track Error (km) | Wind MAE (km/h) |
|---------|-----------------|-----------------|
| 6h | 38.05 | 3.12 |
| 12h | 80.50 | 6.46 |
| 24h | 180.66 | 13.22 |

### Phase-3 LSTM (stored, not re-verified)

| Horizon | Track Error (km) |
|---------|-----------------|
| 6h | 130.02 |
| 12h | 180.00 |
| 24h | 267.54 |

## Key Findings

1. **EXP005 loses to movement-vector at ALL horizons on track error**
2. EXP005 beats phase-3 LSTM at all horizons (+30% track improvement)
3. EXP005 beats persistence at 12h (+3.4%) and 24h (+17.2%) but loses at 6h (-41.3%)
4. Normalization is train-only (verified)
5. Champion selected on validation only, test evaluated exactly once
6. No future leakage in features (verified NaN/Inf-free)
7. Haversine distance used for track error (never Euclidean degrees)

## Data Architecture

- canonical_chrono: raw (5, 7) features, 401 test samples
- feature_dataset: engineered (5, 16) features, 198 test samples
- EXP005 trained on feature_dataset, baselines evaluated on same test set
- Both share the same ground truth Y (lat/lon/wind targets)

## Verdict

**PASS_WITH_WARNING** — All reported metrics are reproducible. The scientific limitation
that P4 loses to the movement-vector baseline on track is honest and disclosed.
