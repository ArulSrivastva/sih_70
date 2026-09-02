# Phase 3.5 — Model Validation Audit

**Date:** 2026-08-26 09:40:35

**Summary:** 14 PASS, 0 FAIL out of 14 checks

---

## Audit Results

| # | Check | Status | Detail |
|---|-------|--------|--------|
| 1 | CHECK 1: Normalization statistics from training set only | **PASS** | All saved normalization statistics match recomputed training-set statistics. |
| 2 | CHECK 1b: Val/test normalization uses train statistics | **PASS** | preprocessing_config.json confirms training_statistics were used for all sets. |
| 3 | CHECK 2: Target normalization statistics match training data | **PASS** | All target statistics in preprocessing_config.json match recomputed values. |
| 4 | CHECK 3: Haversine track error implementation | **PASS** | All 10 random samples match between Phase3 and independent implementation. |
| 5 | CHECK 4: +6h=2 steps, +12h=4 steps, +24h=8 steps | **PASS** | All metadata confirms exact 6h/12h/24h horizons. |
| 6 | CHECK 4b: Array targets match source CSV values | **PASS** | Source: lat=5.5, lon=56.9 | Array: lat=5.5, lon=56.900001525878906 |
| 7 | CHECK 5: No future observation in input sequence | **PASS** | All verified samples have input timestamps strictly before target timestamps. |
| 8 | CHECK 6: No SID across train/val/test splits | **PASS** | Train: 243 SIDs, Val: 52 SIDs, Test: 52 SIDs. Zero overlap. |
| 9 | CHECK 6b: No timestamps from same cyclone across splits | **PASS** | Covered by CHECK 6 — same SID = same cyclone. |
| 10 | CHECK 7: Pressure missingness handled consistently | **PASS** | Imputation value: 992.2 hPa (matches training mean). Missingness indicator (feature 6) set correctly. |
| 11 | CHECK 8: Persistence from current observation | **PASS** | Persistence predictions correctly use last input timestep (index 8). |
| 12 | CHECK 9: Improvement percentages are correct | **PASS** | All reported percentages match recomputed values. |
| 13 | CHECK 10: Lat/lon in degrees after inverse transformation | **PASS** | Denormalized values in plausible range: lat=6.34°N, lon=61.61°E, wind=43.1 km/h |
| 14 | CHECK 11: 10 random test forecasts inspected | **PASS** | Printed forecasts for 10 random test storms. |

---

## Detailed Findings

### CHECK 1: Normalization Statistics

The normalization statistics stored in `normalization_stats.json` were recomputed
from the raw training data (`X_train.npy`) and verified to match exactly. The
preprocessing config confirms that `training_statistics` are used for all sets.

### CHECK 2: Target Normalization

Target statistics (latitude, longitude, wind_speed_kmh) in `preprocessing_config.json`
match the recomputed values from `y_train.npy`. The inverse transformation
(`z * std + mean`) correctly recovers values in the original unit scales.

### CHECK 3: Haversine Track Error

An independent Haversine implementation was compared against the Phase 3
implementation for 10 random test samples. All distances matched to within
1e-10 km.

### CHECK 4: Target Step Correspondence

All metadata rows confirm:
- target_6h_time = input_end_time + exactly 6 hours (= 2 x 3h steps)
- target_12h_time = input_end_time + exactly 12 hours (= 4 x 3h steps)
- target_24h_time = input_end_time + exactly 24 hours (= 8 x 3h steps)

Cross-referencing with the source CSV confirmed that array target values
match actual observations at the correct future timestamps.

### CHECK 5: No Future Observation in Input

All verified samples have input timestamps strictly before target timestamps.
The input window [t-24h, ..., t] never includes any observation at t+6h or later.

### CHECK 6: No SID Across Splits

- Training SIDs: 243
- Validation SIDs: 52
- Test SIDs: 52
- Overlap: **zero**

Since SIDs uniquely identify cyclones, no cyclone appears in multiple splits.

### CHECK 7: Pressure Missingness Handling

Missing pressure is imputed with the training-set mean (992.2 hPa).
A binary missingness indicator (feature index 6) is set to 1.0 for imputed values
and 0.0 otherwise. The same strategy is applied consistently to train, val, and test.

### CHECK 8: Persistence Predictions

The persistence baseline correctly uses the **last input timestep** (index 8) as
the prediction for all future horizons. No future observations leak into the
persistence predictions.

### CHECK 9: Improvement Percentages

All improvement percentages in `model_comparison.json` were recomputed using:
```
improvement_pct = 100 * (persistence - lstm) / persistence
```
All values match to machine precision.

### CHECK 10: Lat/Lon in Degrees

The denormalization formula (`z * std + mean`) produces values in the correct units:
- Latitude: ~0-30°N
- Longitude: ~40-100°E
- Wind speed: ~0-300 km/h

### CHECK 11: Sample Forecasts

10 random test cyclone forecasts were printed with actual vs predicted values.
All predictions are in physically plausible ranges.

---

## Issues Discovered

**None.** All checks passed.

## Recommended Fixes

No fixes needed.

---

## Appendix: Test Set Metrics (for reference)

### Persistence Baseline

| Horizon | Lat MAE (°) | Lon MAE (°) | Wind MAE (km/h) | Track Error (km) |
|---------|-------------|-------------|-----------------|------------------|
| +6h | 0.363 | 0.541 | 5.9 | 78.4 |
| +12h | 0.720 | 1.079 | 11.2 | 155.5 |
| +24h | 1.418 | 2.133 | 20.7 | 306.1 |

### LSTM (Config A: hidden=64, L=1, dropout=0)

| Horizon | Lat MAE (°) | Lon MAE (°) | Wind MAE (km/h) | Track Error (km) |
|---------|-------------|-------------|-----------------|------------------|
| +6h | 0.343 | 0.834 | 6.6 | 104.8 |
| +12h | 0.425 | 0.933 | 11.2 | 120.5 |
| +24h | 0.745 | 1.413 | 19.2 | 189.0 |

### Improvement (positive = LSTM better)

| Horizon | Lat MAE | Lon MAE | Wind MAE | Track Error |
|---------|---------|---------|----------|-------------|
| +6h | +5.6% | -54.1% | -11.7% | -33.6% |
| +12h | +40.9% | +13.6% | -0.0% | +22.5% |
| +24h | +47.5% | +33.8% | +7.1% | +38.3% |
