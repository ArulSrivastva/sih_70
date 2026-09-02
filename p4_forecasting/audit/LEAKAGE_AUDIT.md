# LEAKAGE AUDIT (P4 PHASE 1)

## A. Static pipeline review (build_datasets.py)

- Line 148: `grp_interp = grp[feature_cols].interpolate(method="time").ffill().bfill()` (whole-storm feature fill)
- Line 164: `interp_full = grp_interp.reindex(combined_idx).interpolate(method="time").ffill().bfill()` (history + target sampling re-interpolation)
- Non-causal fill operators: `interpolate(method="time")` (uses past AND future), `bfill()` (uses future).
- Affected columns: lat, lon, wind_speed, pressure, sst, wind_u, wind_v (input) and lat, lon, wind_speed (targets).
- ANY cell that was not present as a real observation in master_dataset.csv is treated as potentially non-causal.

## B. Per-sample contamination counts (vs master_dataset.csv)

| split | n | sequences with >=1 synthetic INPUT cell | sequences with >=1 synthetic TARGET cell |
|---|---|---|---|
| train | 2275 | 932 | 474 |
| val | 378 | 200 | 92 |
| test | 423 | 175 | 93 |

Total: 1307 / 3076 sequences carry non-causal INPUT risk; 659 / 3076 carry synthetic TARGET cells.

Per-feature synthetic input cells:

| feature | train filled cells |
|---|---|
| lat | 0 |
| lon | 0 |
| wind_speed | 2290 |
| pressure | 2202 |
| sst | 1494 |
| wind_u | 0 |
| wind_v | 0 |

## C. Feature/target temporal separation

Every sample: input_end = t_zero < target_start = t+6h (6h strict separation).
Confirmed leakage samples: 0 of 3076.

## D. Severity

- Non-causal interpolation risk: **MEDIUM** (no confirmed leakage; contamination isolated to missing-value fills via `interpolate`/`bfill`).
- Cyclone split leakage: NONE (67/14/16 disjoint).

Severity label used in the final report: MEDIUM (non-causal), NONE (confirmed).
