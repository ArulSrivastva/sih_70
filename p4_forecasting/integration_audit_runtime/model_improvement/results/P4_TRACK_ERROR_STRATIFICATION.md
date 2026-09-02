# P4 Track Error Stratification

## Overall Results (Test Set)

| Horizon | CatBoost (km) | MV (km) | Improvement (km) | Improvement (%) |
|---------|--------------|---------|-------------------|----------------|
| +6h | 34.27 | 38.05 | +3.78 | +9.9% |
| +12h | 71.79 | 80.50 | +8.72 | +10.8% |
| +24h | 154.84 | 180.66 | +25.82 | +14.3% |

## Stratification by Wind Intensity

| Category | Horizon | N | CatBoost (km) | MV (km) | Improvement | Flag |
|----------|---------|---|--------------|---------|-------------|------|
| moderate_46-74 | 12h | 98 | 64.27 | 71.69 | +7.42 km | OK |
| moderate_46-74 | 24h | 98 | 131.30 | 159.75 | +28.45 km | OK |
| moderate_46-74 | 6h | 98 | 30.86 | 33.92 | +3.05 km | OK |
| strong_74-102 | 12h | 61 | 52.82 | 61.31 | +8.49 km | OK |
| strong_74-102 | 24h | 61 | 116.12 | 127.13 | +11.02 km | OK |
| strong_74-102 | 6h | 61 | 28.06 | 30.61 | +2.55 km | OK |
| very_strong_102-130 | 12h | 7 | 58.62 | 54.34 | -4.27 km | LOW_N |
| very_strong_102-130 | 24h | 7 | 141.17 | 156.44 | +15.28 km | LOW_N |
| very_strong_102-130 | 6h | 7 | 37.61 | 38.29 | +0.69 km | LOW_N |
| weak_<=46 | 12h | 32 | 133.83 | 149.80 | +15.97 km | OK |
| weak_<=46 | 24h | 32 | 303.72 | 352.02 | +48.30 km | OK |
| weak_<=46 | 6h | 32 | 55.82 | 64.85 | +9.04 km | OK |

## Stratification by Movement Speed

| Speed | Horizon | N | CatBoost (km) | MV (km) | Improvement | Flag |
|-------|---------|---|--------------|---------|-------------|------|
| fast | 12h | 66 | 77.36 | 96.75 | +19.39 km | OK |
| fast | 24h | 66 | 175.02 | 233.20 | +58.18 km | OK |
| fast | 6h | 66 | 36.78 | 44.73 | +7.96 km | OK |
| medium | 12h | 63 | 61.81 | 68.55 | +6.74 km | OK |
| medium | 24h | 63 | 136.78 | 159.04 | +22.26 km | OK |
| medium | 6h | 63 | 31.28 | 33.20 | +1.92 km | OK |
| slow | 12h | 69 | 75.57 | 75.88 | +0.32 km | OK |
| slow | 24h | 69 | 152.02 | 150.14 | -1.88 km | OK |
| slow | 6h | 69 | 34.60 | 36.09 | +1.49 km | OK |

## Stratification by Turning

| Turning | Horizon | N | CatBoost (km) | MV (km) | Improvement |
|---------|---------|---|--------------|---------|-------------|
| straight_<17 | 12h | 100 | 74.59 | 78.06 | +3.47 km |
| straight_<17 | 24h | 100 | 169.02 | 183.64 | +14.62 km |
| straight_<17 | 6h | 100 | 34.43 | 36.69 | +2.26 km |
| turning_>=17 | 12h | 98 | 68.92 | 83.00 | +14.08 km |
| turning_>=17 | 24h | 98 | 140.36 | 177.62 | +37.25 km |
| turning_>=17 | 6h | 98 | 34.10 | 39.44 | +5.34 km |

## Stratification by Region

| Region | Horizon | N | CatBoost (km) | MV (km) | Improvement | Flag |
|--------|---------|---|--------------|---------|-------------|------|
| central_AS_BB | 12h | 144 | 75.00 | 81.89 | +6.89 km | OK |
| central_AS_BB | 24h | 144 | 162.96 | 188.07 | +25.11 km | OK |
| central_AS_BB | 6h | 144 | 34.61 | 38.18 | +3.57 km | OK |
| eastern_BB | 12h | 26 | 58.29 | 66.25 | +7.96 km | OK |
| eastern_BB | 24h | 26 | 106.67 | 140.82 | +34.15 km | OK |
| eastern_BB | 6h | 26 | 30.67 | 32.67 | +2.00 km | OK |
| western_AS | 12h | 28 | 67.76 | 86.61 | +18.85 km | OK |
| western_AS | 24h | 28 | 157.82 | 179.57 | +21.75 km | OK |
| western_AS | 6h | 28 | 35.86 | 42.40 | +6.54 km | OK |

## Stratification by Latitude

| Latitude | Horizon | N | CatBoost (km) | MV (km) | Improvement | Flag |
|----------|---------|---|--------------|---------|-------------|------|
| high_lat_>=20 | 12h | 36 | 66.77 | 73.33 | +6.56 km | OK |
| high_lat_>=20 | 24h | 36 | 153.71 | 166.17 | +12.46 km | OK |
| high_lat_>=20 | 6h | 36 | 32.31 | 32.80 | +0.49 km | OK |
| low_lat_<10 | 12h | 23 | 68.01 | 72.34 | +4.34 km | OK |
| low_lat_<10 | 24h | 23 | 153.04 | 174.81 | +21.77 km | OK |
| low_lat_<10 | 6h | 23 | 28.61 | 30.36 | +1.76 km | OK |
| mid_lat_10-15 | 12h | 60 | 66.96 | 70.30 | +3.34 km | OK |
| mid_lat_10-15 | 24h | 60 | 140.19 | 161.96 | +21.77 km | OK |
| mid_lat_10-15 | 6h | 60 | 33.37 | 37.81 | +4.45 km | OK |
| midhigh_lat_15-20 | 12h | 79 | 78.84 | 93.90 | +15.06 km | OK |
| midhigh_lat_15-20 | 24h | 79 | 167.00 | 203.17 | +36.17 km | OK |
| midhigh_lat_15-20 | 6h | 79 | 37.50 | 42.86 | +5.37 km | OK |

## Key Findings
- Strongest CatBoost advantage (24h): fast_24h (+58.2 km)
- Weakest CatBoost advantage (24h): slow_24h (-1.9 km)
- CatBoost improves over MV in 116/198 samples at 24h