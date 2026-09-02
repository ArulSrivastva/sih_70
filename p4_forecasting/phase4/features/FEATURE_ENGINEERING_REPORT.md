# P4 Phase-4 FEATURE ENGINEERING REPORT

- Generated: 2026-08-29T09:45:57.969650+00:00
- Source dataset: `p4_forecasting/phase2/results/canonical_chronological_clean` (1212 train / 231 val / 198 test sequences; CLEAN-only policy authoritative).

## The 9 derived features

| # | feature | units | formula / convention |
|---|---|---|---|
| delta_lat | deg | lat(i) - lat(i-1); i=0 -> 0 |
| delta_lon | deg | wrapped (lon(i) - lon(i-1)) in (-180, 180]; i=0 -> 0 |
| movement_speed | km/h | Haversine(prev,cur) / 6h; i=0 -> 0 |
| movement_direction | deg | initial bearing from prev to cur, [0,360); i=0 -> 0 |
| wind_change | km/h | wind_speed(i) - wind_speed(i-1); i=0 -> 0 |
| pressure_change | hPa | pressure(i) - pressure(i-1); i=0 -> 0 |
| sst_change | deg C | sst(i) - sst(i-1); i=0 -> 0 |
| environmental_wind_speed | m/s | sqrt(wind_u^2 + wind_v^2) (no predecessor) |
| environmental_wind_direction | deg | FROM-direction atan2(-u,-v) mod 360 (no predecessor) |

## Longitude wrapping

Canonical longitudes are stored in [0, 360). delta_lon uses the wrapped difference `(lon_i - lon_{i-1} + 180) mod 360 - 180`, so no artificial jump across the 0/360 boundary is created. Predicted longitudes are wrapped back into [0, 360) at inference.

## First-step zero-fill policy

> For difference/trend features requiring a predecessor, the first available history timestep is zero-filled because no predecessor exists within the model's causal input window.

## Causal rule

At each history step i the engineered features depend only on steps <= i (for i=0 only on step 0). None of the nine features uses a future observation, the forecast targets, the i=-1 step, or any statistical summary of the whole window (no global mean/std enters feature values).

## Why no future data is used

A forecast issued at t_zero may legitimately only know information up to t_zero. Using t_zero+k or the +6/+12/+24 h targets would leak the answer into the input. The leakage tests (test_features.py, test_no_future_leak.py) mutate future steps and targets and assert the engineered features do not change.

## Source dataset

- Clean dir: `C:\Users\aruls\Desktop\SIH26\ps70\cyclone-project\p4_forecasting\phase2\results\canonical_chronological_clean`
- Chrono provenance: `C:\Users\aruls\Desktop\SIH26\ps70\cyclone-project\p4_forecasting\canonical_chrono`
- Quality: `C:\Users\aruls\Desktop\SIH26\ps70\cyclone-project\p4_forecasting\canonical\sample_quality.csv`

## Resulting shapes

| split | X | Y |
|---|---|---|
| train | (N,5,16) | (N,3,3) |
| val | (N,5,16) | (N,3,3) |
| test | (N,5,16) | (N,3,3) |

## Quality filtering

Only rows whose Phase-1 quality status == CLEAN are retained (Phase-2 authoritative policy). Counts equal the Phase-2 clean set exactly.

## Missing-value handling

The CLEAN source contains no NaN/Inf. No imputation is introduced. The only deterministic handling is the documented first-step zero-fill for predecessor-dependent features.

## Verification results

- Raw 7 columns byte-identical to the source (tested).
- First-step predecessor-dependent features exactly zero (tested).
- Future-step and target mutation do not alter engineered features (tested).
- Longitude wrap and movement numerics validated (tested).
- No NaN/Inf introduced (tested).
- Counts verified against Phase-2: True (cyclones) / True (sequences).
