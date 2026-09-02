# P4 Phase-4 FEATURE CONTRACT (LOCKED)

- Generated: 2026-08-29T09:45:57.969650+00:00
- First-step policy: For difference/trend features requiring a predecessor, the first available history timestep is zero-filled because no predecessor exists within the model's causal input window.

## Input contract

| item | value |
|---|---|
| raw history | (N, 5, 7) float32 |
| featured history | (N, 5, 16) float32 |
| raw feature order | `lat, lon, wind_speed, pressure, sst, wind_u, wind_v` |
| derived features | `delta_lat, delta_lon, movement_speed, movement_direction, wind_change, pressure_change, sst_change, environmental_wind_speed, environmental_wind_direction` |
| target order | `lat, lon, wind_speed` |
| horizons | +6 / +12 / +24 hours |
| SST units | degrees Celsius (untouched) |

## Dataset splits (from `phase2/results/canonical_chronological_clean` - CLEAN only)

| split | sequences | cyclones | X shape | Y shape | nan/inf |
|---|---|---:|---:|---|---|---|
| train | 1212 | 57 | (1212, 5, 16) | (1212, 3, 3) | False/False |
| val | 231 | 13 | (231, 5, 16) | (231, 3, 3) | False/False |
| test | 198 | 10 | (198, 5, 16) | (198, 3, 3) | False/False |

## Causality

- No target Y cell is ever used to construct an input feature.
- No future history timestep is used.
- No interpolation is introduced (the CLEAN source has none).
- OHz delta_lon uses wrapped difference in (-180, 180]; no 0/360 jump.
- movement_speed = Haversine(km) / 6 h expressed in km/h.
- movement_direction: initial bearing clockwise from true north, [0,360).
- environmental_wind_speed = sqrt(u^2+v^2) m/s; direction = meteorological FROM-convention, degrees clockwise from north, [0,360).
