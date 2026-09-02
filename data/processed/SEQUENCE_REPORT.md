# Cyclone Forecasting Sequence Report

**Phase 2: Supervised Temporal Sequences**

Generated: 2026-08-26 09:20:28

---

## 1. Input Dataset

| Property | Value |
|----------|-------|
| Source | `ibtracs_forecasting_base.csv` |
| Total observations | 17,778 |
| Total storms | 471 |
| Date range | 1980-10-10 to 2025-12-02 |

---

## 2. Sequence Construction

### 2.1 Regular 3-Hour Timesteps

| Property | Value |
|----------|-------|
| Timestep | 3 hours |
| Input window | 9 observations (27h) |
| Gap handling | **Terminate sequence at gaps** (no interpolation) |

**Rationale:** Interpolating missing observations would fabricate data points that
don't exist in the real observational record. Terminating sequences at gaps ensures
the model only trains on real, consecutive observations.

### 2.2 Valid Segments

| Property | Value |
|----------|-------|
| Total valid segments | 453 |
| Min segment length | 9 observations |
| Max segment length | 168 observations |
| Mean segment length | 37.1 observations |
| Median segment length | 30 observations |

### 2.3 Input Window Structure

Each input window contains 9 observations at 3h intervals:

```
t-24h, t-21h, t-18h, t-15h, t-12h, t-9h, t-6h, t-3h, t
```

**Features per timestep:**
1. `latitude` (degrees N)
2. `longitude` (degrees E)
3. `wind_speed_kmh` (km/h)
4. `pressure_hpa` (hPa)
5. `storm_speed` (kts)
6. `storm_direction` (degrees)
7. `pressure_hpa_missing` (binary indicator)

**Total input dimension:** 9 timesteps x 7 features = **63 values**

### 2.4 Missingness Indicators

| Indicator | Source |
|-----------|--------|
| `pressure_hpa_missing` | 1 if `pressure_hpa` is NaN, 0 otherwise |

**Decision:** Missing pressure is NOT imputed or set to zero. The binary indicator
preserves the information that pressure was unavailable, while the NaN value
signals to downstream code that imputation should be applied during normalization.

---

## 3. Forecast Targets

| Horizon | Steps ahead | Usable samples | % of total |
|---------|-------------|----------------|------------|
| +6h | 2 | 9,514 | 95.8% |
| +12h | 4 | 9,453 | 95.2% |
| +24h | 8 | 9,423 | 94.9% |

**Target variables:** `latitude`, `longitude`, `wind_speed_kmh`

**Decision:** Only samples with valid (non-NaN) targets for ALL horizons are included
in the final arrays. This simplifies training but may reduce sample count.

---

## 4. Data Splitting

### 4.1 Split Strategy

| Property | Value |
|----------|-------|
| Method | **Chronological** (by cyclone start date) |
| Random seed | 42 |
| Train | 243 storms (70%) |
| Validation | 52 storms (15%) |
| Test | 52 storms (15%) |

### 4.2 Why Chronological > Random

| Aspect | Random Split | Chronological Split |
|--------|-------------|-------------------|
| Realism | Mixes past and future storms | Test set = most recent storms |
| Generalization | Tests on similar storms | Tests on unseen time periods |
| Forecasting proxy | Overestimates performance | Better estimates real-world skill |
| Leakage risk | Low (by SID) | None (temporal separation) |

**Recommendation:** Use **chronological split** for all model evaluation. This
mimics the real forecasting scenario where we train on historical data and predict
future storms.

### 4.3 Alternative Random Split (for reference)

| Split | Storms |
|-------|--------|
| Train | 242 |
| Validation | 52 |
| Test | 53 |

---

## 5. Data Leakage Check

| Check | Result |
|-------|--------|
| SID in multiple splits | PASSED |
| Future targets in inputs | PASSED (by construction) |
| Normalization from val/test | PASSED (train-only stats) |

**Overall:** PASSED

---

## 6. Normalization Statistics (Training Set Only)

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| latitude | 13.971 | 5.089 | 0.70 | 29.20 |
| longitude | 88.707 | 19.669 | 49.00 | 163.70 |
| wind_speed_kmh | 70.000 | 39.172 | 5.56 | 277.80 |
| pressure_hpa | 992.196 | 15.797 | 890.00 | 1012.00 |
| storm_speed | 7.467 | 4.181 | 0.00 | 54.00 |
| storm_direction | 239.255 | 104.927 | 0.00 | 360.00 |

**Saved to:** `normalization_stats.json`

---

## 7. Output Files

### 7.1 Numpy Arrays

| File | Shape | Description |
|------|-------|-------------|
| `X_train.npy` | (7174, 9, 7) | Training input windows |
| `y_train.npy` | (7174, 3, 3) | Training targets (3 horizons x 3 features) |
| `X_val.npy` | (1563, 9, 7) | Validation input windows |
| `y_val.npy` | (1563, 3, 3) | Validation targets |
| `X_test.npy` | (1191, 9, 7) | Test input windows |
| `y_test.npy` | (1191, 3, 3) | Test targets |

### 7.2 Metadata

| File | Rows | Columns |
|------|------|---------|
| `train_metadata.csv` | 7,174 | sample_id, SID, input_start_time, input_end_time, target_*_time |
| `val_metadata.csv` | 1,563 | same |
| `test_metadata.csv` | 1,191 | same |

### 7.3 Reports

| File | Description |
|------|-------------|
| `SEQUENCE_REPORT.md` | This report |
| `normalization_stats.json` | Training-set statistics for Phase 3 |

---

## 8. Temporal Gap Statistics

| Interval | Count | Percentage |
|----------|-------|------------|
| 3-hour | 16,993 | 98.2% |
| 6-hour | 20 | 0.1% |
| Other (>6h) | 294 | 1.7% |
| **Total** | **17,307** | **100%** |

- Storms with major gaps (>12h): 35/471 (7.4%)

---

## 9. Summary

| Metric | Value |
|--------|-------|
| Total valid sequences | 9,928 |
| Training samples | 7,174 |
| Validation samples | 1,563 |
| Test samples | 1,191 |
| Input shape | (9, 7) = 9 timesteps x 7 features |
| Target horizons | +6h, +12h, +24h |
| Targets per horizon | 3 (lat, lon, wind) |
| Missing pressure samples | 4,016 (40.5%) |
| Split method | Chronological (recommended) |

---

## 10. Limitations

1. **Sequence termination at gaps:** Cyclones with many gaps produce short segments,
   reducing usable samples. 7.4% of storms have major gaps (>12h).

2. **Missing pressure:** 34.0% of input timesteps have missing pressure. The
   missingness indicator preserves this information, but downstream normalization
   must decide on imputation strategy (e.g., fill with training mean).

3. **Single-source targets:** Targets use `wind_speed_kmh` from the coalesced
   hierarchy (IMD -> WMO -> JTWC). Different sources may estimate different values.

4. **No ERA5/INSAT features:** Current input features are limited to cyclone track
   parameters. Adding atmospheric context (ERA5) and satellite imagery (INSAT) in
   later phases should improve forecasting skill.

5. **3-hour resolution:** Rapid intensification events between observations are
   not captured.
