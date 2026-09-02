# P4 DATA CONTRACT (CANONICAL FORECASTING DATASET)

Version 1 | PS 26070 (SIH2026) | created by P4 PHASE 1 audit

## 1. Arrays

- `X` : history tensor, shape (N, 5, 7), dtype float32
- `Y` : forecast targets tensor, shape (N, 3, 3), dtype float32
- `features` : length-7 array of feature names, in column order of `X`
- `targets`  : length-3 array of target names, in column order of `Y`

Expected sample counts (original P1 split): train=2275, val=378, test=423 (total 3076).

## 2. History time steps (X)

| index | step | feature columns |
|---|---|---|
| 0 | t-24h | lat, lon, wind_speed, pressure, sst, wind_u, wind_v |
| 1 | t-18h | lat, lon, wind_speed, pressure, sst, wind_u, wind_v |
| 2 | t-12h | lat, lon, wind_speed, pressure, sst, wind_u, wind_v |
| 3 | t-6h  | lat, lon, wind_speed, pressure, sst, wind_u, wind_v |
| 4 | t-0h  | lat, lon, wind_speed, pressure, sst, wind_u, wind_v |

## 3. Target time steps (Y)

| index | lead time | target columns |
|---|---|---|
| 0 | t+6h  | lat, lon, wind_speed |
| 1 | t+12h | lat, lon, wind_speed |
| 2 | t+24h | lat, lon, wind_speed |

## 4. Units

- lat:  degrees, -90..90
- lon:  degrees, -180..360 (P1 stores 0..360 for North Indian Ocean basin)
- wind_speed: km/h
- pressure: hPa
- sst:  DEGREES CELSIUS (P1 delivery; upstream ERA5 raw is Kelvin and P1 converted K -> deg C as `sst_celsius`). NOTE: The Phase-1 spec assumed `sst` is Kelvin. It is NOT. Raw values ~25..31 are decimal degrees Celsius. Canonical arrays preserve P1 values unmodified (no silent conversion). Any Kelvin conversion is a deliberate Phase-2 preprocessing step.
- wind_u / wind_v: m/s

## 5. Time conventions

- `t_zero` : forecast issue time (reference time). History covers 24h up to and including `t_zero`. Targets are strictly AFTER `t_zero`.
- Timestamps in metadata: ISO UTC strings (e.g. `2013-11-02 06:00:00`).
- Expected history cadence: 6-hourly (t-24, t-18, t-12, t-6, t-0).

## 6. Quality / exclusion policy

| status | meaning | kept in canonical? |
|---|---|---|
| CLEAN | no interpolated/filled input or target cells | yes |
| MISSING_ENVIRONMENTAL_DATA | only sst/wind_u/wind_v/pressure cells were filled (non-causal risk) | yes (flagged) |
| NON_CAUSAL_RISK | >=1 input or target cell fabricated by interpolate/ffill/bfill | yes (flagged) |
| LEAKAGE | confirmed future data in inputs (none found) | excluded |
| INVALID | physically invalid sample (none found) | excluded |

Confirmed exclusions applied: 0 (no LEAKAGE / INVALID samples exist).

## 7. Non-normalization

- Canonical arrays are RAW (physical units). `NPZ_NORMALIZATION = NONE`.
- Normalization / scaling is Phase-2 modelling preprocessing, fit on training stats only.

## 8. File layout

- `p4_forecasting/canonical/{train,val,test}.npz` and `{train,val,test}_metadata.csv`
- `p4_forecasting/canonical_chrono/...` secondary chronological split (70/15/15 by cyclone start date) + `split_manifest.csv`
- `p4_forecasting/canonical/sample_quality.csv` per-sample quality flags

## 9. Source of truth

- P1 files (PS70-main.zip) are byte-for-byte unmodified. Inventoried in `p4_forecasting/audit/P4_FILE_INVENTORY.md`.
