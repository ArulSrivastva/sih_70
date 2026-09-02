# P4 Phase-4 Report (Tropical Cyclone Forecasting)

- Generated: 2026-08-29T09:45:57.892732+00:00
- Overall status: **PASS**


## 1. Objective

Train and honestly benchmark three Phase-4 model families (improved LSTM, GRU, multi-task LSTM) on an engineered 16-feature causal contract, select one champion using VALIDATION only, and evaluate that champion exactly once on TEST against the Phase-2 persistence and movement-vector baselines and the Phase-3 LSTM.

## 2. Source datasets

- `p4_forecasting/canonical/` (canonical raw arrays + quality flags)
- `p4_forecasting/canonical_chrono/` (chronological 70/15/15 split)
- `p4_forecasting/phase2/results/canonical_chronological_clean/` (authoritative CLEAN-only source; all inputs strictly read-only)

## 3. Input contract

X = (N, 5, 7) raw: `[lat, lon, wind_speed, pressure, sst, wind_u, wind_v]`; Y = (N, 3, 3) `[lat, lon, wind_speed]` at +6/+12/+24 h; SST in degrees Celsius; history cadence 6-hourly.

## 4. Nine engineered features

See `results/FEATURE_CONTRACT.md` (formulas/units). Table:

| feature | units | formula |
|---|---|---|
| delta_lat | deg | lat(i) - lat(i-1); i=0 -> 0 |
| delta_lon | deg | wrapped (lon(i) - lon(i-1)) in (-180, 180]; i=0 -> 0 |
| movement_speed | km/h | Haversine(prev,cur) / 6h; i=0 -> 0 |
| movement_direction | deg | initial bearing from prev to cur, [0,360); i=0 -> 0 |
| wind_change | km/h | wind_speed(i) - wind_speed(i-1); i=0 -> 0 |
| pressure_change | hPa | pressure(i) - pressure(i-1); i=0 -> 0 |
| sst_change | deg C | sst(i) - sst(i-1); i=0 -> 0 |
| environmental_wind_speed | m/s | sqrt(wind_u^2 + wind_v^2) (no predecessor) |
| environmental_wind_direction | deg | FROM-direction atan2(-u,-v) mod 360 (no predecessor) |


## 5. First-step zero-fill policy

> For difference/trend features requiring a predecessor, the first available history timestep is zero-filled because no predecessor exists within the model's causal input window.

## 6. Causality / leakage validation

- Input audit overall PASS: True
- Future-leak mutation tests + full unit suite passed: True
- Engineered features depend only on X[:, :i+1, :]; target Y mutation cannot change any engineered feature (verified by tests).

## 7. Dataset counts

| split | sequences | cyclones | X | Y |
|---|---:|---:|---|---|
| train | 1212 | 57 | [1212, 5, 7] | [1212, 3, 3] |
| val | 231 | 13 | [231, 5, 7] | [231, 3, 3] |
| test | 198 | 10 | [198, 5, 7] | [198, 3, 3] |

## 8. Normalization policy

- Train-only statistics: TRAIN only; never val/test/combined.
- Zero-std features handled as: std<=0 replaced by scale 1.0 (identity), deterministically ([]).
- Directional features documented in stats `directional_features`.

## 9. Model architectures

- ImprovedLSTM: `16 -> LSTM -> last hidden -> Linear -> 9 -> (B,3,3)`
- GRUCyclone: same contract with GRU encoder
- MultiTaskLSTM: shared LSTM encoder + separate track (lat/lon) and intensity (wind) heads composed to `(B,3,3)` = [lat, lon, wind]

## 10. Loss functions

- `ForecastMSELoss` - plain MSE over (B,3,3)
- `ForecastHuberLoss` - smooth-L1 with configurable delta
- `WeightedMultiTaskLoss` - configurable track_weight / intensity_weight / horizon_weights, normalized by weight sums

## 11. EXP001-EXP006 configurations

| exp | model | loss | hidden | layers | dropout | lr | batch | seed |
|---|---|---|---:|---:|---:|---|---|---:|
| EXP001 | improved_lstm | mse | 64 | 1 | 0.0 | 0.001 | 64 | 42 |
| EXP002 | improved_lstm | mse | 96 | 2 | 0.1 | 0.001 | 64 | 42 |
| EXP003 | gru | mse | 96 | 2 | 0.1 | 0.001 | 64 | 42 |
| EXP004 | multitask_lstm | mse | 96 | 2 | 0.1 | 0.001 | 64 | 42 |
| EXP005 | gru | huber | 96 | 2 | 0.1 | 0.001 | 64 | 42 |
| EXP006 | gru | weighted | 96 | 2 | 0.1 | 0.001 | 64 | 42 |

## 12. Validation results

| exp | model | loss | best_epoch | best_val_loss | track 6h | track 12h | track 24h | wind MAE 6/12/24h | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| EXP001 | improved_lstm | mse | 38 | 0.0733 | 141.6952 | 161.5702 | 197.2024 | 8.4206/11.7113/20.6099 | PASS |
| EXP002 | improved_lstm | mse | 36 | 0.0685 | 114.0205 | 137.6266 | 190.0393 | 9.2155/11.6193/19.6165 | PASS |
| EXP003 | gru | mse | 31 | 0.0696 | 98.7229 | 111.0280 | 171.7082 | 7.8484/11.7924/21.4329 | PASS |
| EXP004 | multitask_lstm | mse | 30 | 0.0701 | 107.7232 | 136.9302 | 176.7731 | 9.2073/12.4684/20.7445 | PASS |
| EXP005 | gru | huber | 16 | 0.0338 | 79.8431 | 101.1676 | 158.2118 | 7.5560/11.9086/22.5536 | PASS |
| EXP006 | gru | weighted | 31 | 0.0696 | 98.7229 | 111.0280 | 171.7082 | 7.8484/11.7924/21.4329 | PASS |

## 13. Champion-selection rule

Locked: lowest equal-weight mean of VALIDATION track errors `mean(val_track_6h, val_track_12h, val_track_24h)`; tie-break wind MAE then RMSE. TEST is never used for selection.

## 14. Champion

- Champion experiment: **EXP005**
- Primary (validation) score: 113.0741
- Components: {'6h': 79.8430765890122, '12h': 101.16757375743805, '24h': 158.21177219925482}
- Selection rule: lowest equal-weight mean of validation track errors mean(val_track_6h, val_track_12h, val_track_24h); tie-break wind MAE then RMSE; TEST IS NEVER USED FOR SELECTION

## 15. Final test results (champion, evaluated exactly once)

- +6h: track=91.43 km (median 74.02), wind MAE=6.68, RMSE=8.28 km/h
- +12h: track=119.76 km (median 94.90), wind MAE=9.46, RMSE=11.67 km/h
- +24h: track=188.24 km (median 152.67), wind MAE=16.11, RMSE=19.65 km/h

## 16. Comparison against persistence

| horizon | track % | wind MAE % | wind RMSE % |
|---|---:|---:|---:|
| 6h | -41.3 | -114.1 | -41.3 |
| 12h | +3.4 | -46.4 | -13.2 |
| 24h | +17.2 | -21.9 | +0.0 |

## 17. Comparison against movement-vector

| horizon | track % | wind MAE % | wind RMSE % |
|---|---:|---:|---:|
| 6h | -140.3 | -114.1 | -41.3 |
| 12h | -48.8 | -46.4 | -13.2 |
| 24h | -4.2 | -21.9 | +0.0 |

## 18. Comparison against phase-3 LSTM

| horizon | track % | wind MAE % | wind RMSE % |
|---|---:|---:|---:|
| 6h | +29.7 | +0.4 | +1.3 |
| 12h | +33.5 | +12.5 | +13.0 |
| 24h | +29.6 | +4.9 | +9.3 |

> improvement % = (baseline - challenger)/|baseline| * 100; positive = challenger better; track error uses mean km; wind uses MAE/RMSE. Negative values mean the baseline beats the Phase-4 champion and are reported honestly, consistent with Phase-3 reporting.

## 19. Honest limitations

- If the champion underperforms the baselines that is reported verbatim (negative improvement percentages), never fabricated.
- Failed experiments remain in the registry with status=FAILED.
- First-step zero-fill applies to all predecessor-dependent features (documented).
- Directional features are z-scored linearly; the 0/360 wrap discontinuity is a documented limitation of the locked 16-column contract (no sine/cosine embedding).

## 20. Reproducibility

- Fixed seed (42), CPU-only deterministic execution, `PYTHONDONTWRITEBYTECODE=1`, `python -X utf8`.
- Idempotent orchestrator: completed experiments are reused unless `--force-retrain`; feature dataset and normalization are deterministic and content-equal.

## 21. Source immutability

| bucket | changed files |
|---|---|
| p1_modified | 0 |
| phase1_modified | 0 |
| phase2_modified | 0 |
| phase3_modified | 0 |
| outside_phase4_modified | 0 |
- Verdict: PASS

## 22. Final status

**PASS**
