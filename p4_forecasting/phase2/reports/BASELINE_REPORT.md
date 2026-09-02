# P4 Phase 2 - Baseline Report (PS 26070 / SIH 2026)

- Generated: 2026-08-29T08:24:08.803450+00:00
- Policy: CLEAN-only chronological dataset; no value transforms, no imputation, no interpolation, no normalization; SST kept in degrees Celsius.
- Source: `C:\Users\aruls\Desktop\SIH26\ps70\cyclone-project\p4_forecasting\canonical_chrono` (read-only) + `canonical/sample_quality.csv`.

## Dataset (canonical_chronological_clean)

| Split | Cyclones | Sequences | Notes |
|---|---|---|---|
| Train | 57 | 1212 | CLEAN only |
| Val   | 13 | 231 | CLEAN only |
| Test  | 10 | 198 | CLEAN only |

> Deviation from spec placeholder counts (68/15/14 cyclones, 2259/416/401 seq): those match the **full chronological** set; Phase-2 uses the **CLEAN-only** subset per project decision.

## Baselines

- **Persistence**: forecast = most recent observed (lat, lon, wind) for all horizons (+6/+12/+24 h).
- **Movement vector**: 6-hour vector = last two history steps (t-6h -> t); extrapolated x1/x2/x4 (lon wrapped 0..360); wind = persistence. TARGETS never consumed. Constant-velocity, linear-displacement approximation.

All errors measured with Haversine (km).

## Primary results (TEST split)

| Horizon | Model | Track Err km (mean) | (median) | (std) | Wind MAE km/h | Wind RMSE km/h |
|---|---|---:|---:|---:|---:|---:|
| 6h | persistence | 64.7 | 63.8 | 34.8 | 3.1 | 5.9 |
| 12h | persistence | 123.9 | 125.0 | 62.3 | 6.5 | 10.3 |
| 24h | persistence | 227.3 | 220.2 | 115.5 | 13.2 | 19.7 |
| 6h | movement_vector | 38.1 | 32.8 | 28.6 | 3.1 | 5.9 |
| 12h | movement_vector | 80.5 | 64.5 | 56.9 | 6.5 | 10.3 |
| 24h | movement_vector | 180.7 | 165.0 | 120.3 | 13.2 | 19.7 |

## Validation results

| Horizon | Model | Track Err km (mean) | (median) | (std) | Wind MAE km/h | Wind RMSE km/h |
|---|---|---:|---:|---:|---:|---:|
| 6h | persistence | 68.5 | 64.0 | 33.0 | 6.6 | 10.3 |
| 12h | persistence | 136.5 | 128.0 | 66.2 | 13.6 | 19.2 |
| 24h | persistence | 282.1 | 262.7 | 144.9 | 27.5 | 37.9 |
| 6h | movement_vector | 28.2 | 24.1 | 19.6 | 6.6 | 10.3 |
| 12h | movement_vector | 62.6 | 55.0 | 42.9 | 13.6 | 19.2 |
| 24h | movement_vector | 139.7 | 112.4 | 103.2 | 27.5 | 37.9 |

## Summary

Movement-vector shows no better 24h track error than persistence at the sample level in this clean delivery; persistence remains competitive and is the recommended lower-bound for later ML models.
