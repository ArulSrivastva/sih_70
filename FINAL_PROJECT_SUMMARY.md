# SIH 2026 — Final Project Summary
## Cyclone Track Forecasting and Intensity Classification

## Problem

Tropical cyclones in the North Indian Ocean cause significant loss of life and property.
Accurate forecasting of cyclone tracks and intensity is critical for disaster preparedness.
This project develops ML-based models for:
1. **P3**: Cyclone intensity classification (IMD category from satellite/environmental data)
2. **P4**: Cyclone track forecasting (future position at +6h, +12h, +24h)

## P3 — Intensity Classification

**Model**: LightGBM gradient-boosted decision tree classifier

**Features**: 6 original features — latitude, longitude, sea surface temperature (SST),
mean sea-level pressure (MSLP), and two wind components (u, v)

**Results**:
- Test accuracy: **47.00%** (7-class IMD classification)
- Test macro-F1: **0.3744**
- GroupKFold by storm: **41.95% accuracy / 0.2786 macro-F1**

**Why P3 was locked**:
The GroupKFold result reveals that observations within the same cyclone are highly
correlated. Random/stratified validation overestimates true generalization by ~5%.
All attempted improvements (feature engineering, ordinal classification, class weighting,
storm-aggregated features) failed to improve macro-F1 under storm-aware validation.
The original 6 features are near-optimal for this dataset size.

**Key limitation**: Severe class imbalance (Depression = 40% of training data;
Super Cyclonic Storm = 25 samples). SST has 28% missing values.

## P4 — Track Forecasting

**Champion**: CatBoost gradient-boosted regression (displacement-based)

**Baseline**: Movement-Vector (linear extrapolation of recent storm motion)

**Results on held-out test set**:

| Horizon | CatBoost (km) | MV (km) | Improvement |
|---------|--------------|---------|-------------|
| +6h     | 34.27        | 38.05   | 9.9%        |
| +12h    | 71.79        | 80.50   | 10.8%       |
| +24h    | 154.84       | 180.66  | 14.3%       |

**Storm-disjoint 5-fold CV** (1443 samples, 70 cyclones):
- CatBoost: 26.82 / 56.57 / 127.28 km
- LightGBM: 26.89 / 57.30 / 132.18 km
- MV baseline: 27.77 / 59.19 / 138.92 km

**Error stratification** (24h):
- **Strongest improvement**: Fast storms (>speed p66) — 175.0 vs 233.2 km (24.9%)
- **Turning storms** (>=17 heading change): 140.4 vs 177.6 km (21.0%)
- **Weakest improvement**: Slow storms (<speed p33) — 152.0 vs 150.1 km (-1.2%)
- CatBoost improves in **16/17 evaluated strata**

## Physics + AI Integration

**What we implemented**:
- Physics-based baseline: Movement-Vector (deterministic kinematic extrapolation)
- AI model: CatBoost (gradient-boosted trees on engineered features)
- Production model: GRU neural network (served via phase5/phase6 API)

**What we do NOT claim**:
- No coupled atmosphere-ocean numerical simulation
- No data assimilation or NWP
- No physics-informed neural network with embedded conservation laws
- No direct landfall prediction (landfall is a post-hoc geometric heuristic)

**Honest interpretation**: CatBoost improves over the physics-based MV baseline by
learning nonlinear relationships in storm dynamics. The improvement is strongest
where linear extrapolation fails — fast-moving and turning storms.

## Limitations

1. **Small dataset**: 1443 training sequences, 70 cyclones for P4; 3039 samples for P3
2. **Class imbalance**: Depression dominates P3 training (40%); Super Cyclonic Storm
   has only 25 samples
3. **SST missingness**: ~28% of SST observations are missing
4. **Storm-level correlation**: Observations within the same cyclone are correlated,
   making naive validation optimistic
5. **Slow-moving storms**: CatBoost does not improve over MV for very slow storms
6. **No full physics simulation**: Our models learn statistical patterns, not
   atmosphere-ocean dynamics
7. **Test set size**: Only 10 cyclones (198 samples) in the P4 test set

## Key Contribution

**AI-based nonlinear forecasting consistently improves cyclone track prediction
over a movement-based baseline, particularly under fast and turning storm conditions.**

The validated CatBoost model reduces 24h track error by 14.3% (154.84 vs 180.66 km)
compared to the physics-based movement-vector baseline. The improvement is:
- Broad (16/17 strata improved)
- Physically interpretable (greatest where linear extrapolation fails)
- Robust (consistent across 5 storm-disjoint CV folds)

## Files and Artifacts

- `results/FINAL_CHAMPIONS.json` — Machine-readable champion records
- `results/P4_FINAL_ERROR_ANALYSIS.json` — Complete error stratification
- `results/P4_TRACK_ERROR_STRATIFICATION.json` — Detailed stratum-level analysis
- `docs/PHYSICS_AI_INTEGRATION.md` — Architecture documentation
- `results/FINAL_REGRESSION_REPORT.md` — Regression/immutability checks
- `FINAL_PROJECT_SUMMARY.md` — This document
