# TEMPORAL AUDIT (P4 PHASE 1)

## Split: train (n=2275)

- window t-24h -> t0  : min window 24.0 h, max deviation 0.000 h
- horizon t0 -> +24h : min span 24.0 h, max deviation 0.000 h
- t_zero off the 3h grid : 0 samples
- irregular window/span   : 0 samples
- t_zero range            : 2013-05-10 18:00:00 .. 2025-11-29 06:00:00

### History-row availability in master (rows missing -> that step required interpolation, i.e. non-causal fill)

| step | lag | rows present | rows missing |
|---|---|---|---|
| 0 | t-24h | 2275 | 0 |
| 1 | t-18h | 2275 | 0 |
| 2 | t-12h | 2275 | 0 |
| 3 | t-6h | 2275 | 0 |
| 4 | t-0h | 2275 | 0 |

### Target-row availability in master (rows missing per horizon)

| horizon | rows missing |
|---|---|
| t+6h | 0 |
| t+12h | 0 |
| t+24h | 0 |

## Split: val (n=378)

- window t-24h -> t0  : min window 24.0 h, max deviation 0.000 h
- horizon t0 -> +24h : min span 24.0 h, max deviation 0.000 h
- t_zero off the 3h grid : 0 samples
- irregular window/span   : 0 samples
- t_zero range            : 2013-11-07 00:00:00 .. 2025-10-28 18:00:00

### History-row availability in master (rows missing -> that step required interpolation, i.e. non-causal fill)

| step | lag | rows present | rows missing |
|---|---|---|---|
| 0 | t-24h | 378 | 0 |
| 1 | t-18h | 378 | 0 |
| 2 | t-12h | 378 | 0 |
| 3 | t-6h | 378 | 0 |
| 4 | t-0h | 378 | 0 |

### Target-row availability in master (rows missing per horizon)

| horizon | rows missing |
|---|---|
| t+6h | 0 |
| t+12h | 0 |
| t+24h | 0 |

## Split: test (n=423)

- window t-24h -> t0  : min window 24.0 h, max deviation 0.000 h
- horizon t0 -> +24h : min span 24.0 h, max deviation 0.000 h
- t_zero off the 3h grid : 1 samples
- irregular window/span   : 0 samples
- t_zero range            : 2013-05-30 03:00:00 .. 2025-12-01 18:00:00

### History-row availability in master (rows missing -> that step required interpolation, i.e. non-causal fill)

| step | lag | rows present | rows missing |
|---|---|---|---|
| 0 | t-24h | 422 | 1 |
| 1 | t-18h | 422 | 1 |
| 2 | t-12h | 422 | 1 |
| 3 | t-6h | 422 | 1 |
| 4 | t-0h | 423 | 0 |

### Target-row availability in master (rows missing per horizon)

| horizon | rows missing |
|---|---|
| t+6h | 1 |
| t+12h | 1 |
| t+24h | 1 |

Interpretation: history/target step *timestamps* are generated exactly at the documented 6h cadence; availability of real observations per step is reported.
