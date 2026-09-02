# P4 Phase 2 -- Experiment Summary

## Objective
Test whether environmental context and regime-aware selection can improve cyclone track forecasts beyond the Phase-1 CatBoost model.

## Data Availability
- **Available**: SST, MSL pressure, 10m U/V wind (single-level ERA5)
- **Unavailable**: Multi-level atmospheric data (500hPa, 850hPa, 200hPa), vertical wind shear, humidity, vorticity

## Feature Ablation Results
| Experiment | Features | 6h (km) | 12h (km) | 24h (km) |
|-----------|----------|---------|----------|----------|
| E0_baseline | 26 | 27.40 | 57.38 | 130.04 |
| E1_motion | 42 | 27.40 | 57.64 | 128.92 |
| E2_env | 37 | 27.45 | 57.73 | 129.39 |
| E3_all | 53 | 27.49 | 57.68 | 128.55 |

## Hybrid Analysis (CV)
| Model | 6h (km) | 12h (km) | 24h (km) |
|-------|---------|----------|----------|
| MV_baseline | 27.77 | 59.19 | 138.92 |
| CatBoost_baseline | 26.82 | 56.57 | 127.27 |
| CatBoost_phase2 | 27.49 | 57.68 | 128.55 |
| Hybrid_rule | 27.44 | 57.75 | 129.42 |
| Hybrid_blend_25 | 26.81 | 56.37 | 130.06 |
| Hybrid_blend_50 | 26.39 | 55.22 | 125.30 |
| Hybrid_blend_75 | 26.62 | 55.70 | 124.82 |

## Along-Track / Cross-Track
| Representation | 6h (km) | 12h (km) | 24h (km) |
|---------------|---------|----------|----------|
| direct_displacement | 27.49 | 57.68 | 128.55 |
| along_cross_track | 26.59 | 56.02 | 125.54 |

## Final Test Results
| Horizon | Phase-2 | Phase-1 | MV | Improvement vs Phase-1 |
|---------|---------|---------|-----|----------------------|
| +6h | 33.75 km | 33.30 km | 38.05 km | -1.4% |
| +12h | 70.33 km | 68.88 km | 80.50 km | -2.1% |
| +24h | 155.82 km | 148.69 km | 180.66 km | -4.8% |

## Robustness
- Cyclones improved (24h): **4/10**
- Regime performance: see FINAL_COMPARISON.md

## Champion Decision
**PHASE1_RETAINED**

## Time elapsed
249 seconds
