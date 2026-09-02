# P4 Final Error Analysis

## Overall Results (Test Set)

| Horizon | CatBoost (km) | MV (km) | Improvement (km) | Improvement (%) |
|---------|--------------|---------|-------------------|----------------|
| +6h | 34.27 | 38.05 | +3.78 | +9.9% |
| +12h | 71.79 | 80.50 | +8.72 | +10.8% |
| +24h | 154.84 | 180.66 | +25.82 | +14.3% |

## CatBoost vs MV -- strata showing improvement at 24h

- 24h: CatBoost=154.8 km, MV=180.7 km, Improvement=+25.8 km (+14.3%) [N=198, OK]
- weak_<=46_24h: CatBoost=303.7 km, MV=352.0 km, Improvement=+48.3 km (+13.7%) [N=32, OK]
- moderate_46-74_24h: CatBoost=131.3 km, MV=159.8 km, Improvement=+28.4 km (+17.8%) [N=98, OK]
- strong_74-102_24h: CatBoost=116.1 km, MV=127.1 km, Improvement=+11.0 km (+8.7%) [N=61, OK]
- slow_24h: CatBoost=152.0 km, MV=150.1 km, Improvement=-1.9 km (-1.2%) [N=69, OK]
- medium_24h: CatBoost=136.8 km, MV=159.0 km, Improvement=+22.3 km (+14.0%) [N=63, OK]
- fast_24h: CatBoost=175.0 km, MV=233.2 km, Improvement=+58.2 km (+24.9%) [N=66, OK]
- straight_<17_24h: CatBoost=169.0 km, MV=183.6 km, Improvement=+14.6 km (+8.0%) [N=100, OK]
- turning_>=17_24h: CatBoost=140.4 km, MV=177.6 km, Improvement=+37.2 km (+21.0%) [N=98, OK]
- sst_available_24h: CatBoost=154.8 km, MV=180.7 km, Improvement=+25.8 km (+14.3%) [N=198, OK]
- western_AS_24h: CatBoost=157.8 km, MV=179.6 km, Improvement=+21.8 km (+12.1%) [N=28, OK]
- central_AS_BB_24h: CatBoost=163.0 km, MV=188.1 km, Improvement=+25.1 km (+13.3%) [N=144, OK]
- eastern_BB_24h: CatBoost=106.7 km, MV=140.8 km, Improvement=+34.1 km (+24.2%) [N=26, OK]
- low_lat_<10_24h: CatBoost=153.0 km, MV=174.8 km, Improvement=+21.8 km (+12.4%) [N=23, OK]
- mid_lat_10-15_24h: CatBoost=140.2 km, MV=162.0 km, Improvement=+21.8 km (+13.4%) [N=60, OK]
- midhigh_lat_15-20_24h: CatBoost=167.0 km, MV=203.2 km, Improvement=+36.2 km (+17.8%) [N=79, OK]
- high_lat_>=20_24h: CatBoost=153.7 km, MV=166.2 km, Improvement=+12.5 km (+7.5%) [N=36, OK]

## Strongest CatBoost advantage: fast_24h (+58.2 km)
## Weakest CatBoost advantage: slow_24h (-1.9 km)

## Verdict
CatBoost improves over MV in 16/17 evaluated strata.
The improvement is broad, not concentrated in a single condition.
The strongest advantage is for fast-moving storms where linear extrapolation fails.
The only near-tie is for slow-moving storms (improvement ~ 0).