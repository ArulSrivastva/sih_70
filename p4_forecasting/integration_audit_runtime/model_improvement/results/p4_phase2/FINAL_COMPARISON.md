# P4 Phase 2 -- Final Comparison

## Test Set Results

| Horizon | Phase-2 CatBoost | Phase-1 CatBoost | MV Baseline |
|---------|-----------------|-----------------|-------------|
| +6h | 33.75 km | 33.30 km | 38.05 km |
| +12h | 70.33 km | 68.88 km | 80.50 km |
| +24h | 155.82 km | 148.69 km | 180.66 km |

## Improvement over Phase-1

| Horizon | Improvement (km) | Improvement (%) |
|---------|-----------------|----------------|
| +6h | -0.46 | -1.4% |
| +12h | -1.45 | -2.1% |
| +24h | -7.13 | -4.8% |

## Robustness

- Cyclones improved (24h): **4/10**

## Regime-Level 24h Performance

| Regime | Phase-2 | Phase-1 | MV | Phase-2 n |
|--------|---------|---------|-----|----------|
| slow | 189.2 | 182.0 | 170.8 | 37 |
| fast | 165.0 | 149.8 | 215.3 | 32 |
| turning | 145.4 | 139.9 | 178.7 | 104 |
| steady | 138.1 | 134.6 | 159.2 | 25 |
