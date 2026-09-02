# Physics + AI Integration — Architecture Documentation

## Overview

The cyclone forecasting system combines physics-based baselines with machine learning
models for track and intensity prediction in the North Indian Ocean basin.

## What We Have Implemented

### 1. Physics-Based Baseline: Movement-Vector (MV)

The Movement-Vector baseline is a **deterministic, physics-based extrapolation** method:
- Uses the storm's displacement over the most recent 6 hours
- Linearly extrapolates this motion to +6h, +12h, and +24h
- No training required; purely kinematic

**Formula**: position(t+h) = position(t) + h/6 * [position(t) - position(t-6h)]

This represents the simplest physically motivated forecast: storms continue moving
in the direction and at the speed they were recently traveling.

### 2. AI Forecasting: CatBoost Displacement Model

CatBoost is a gradient-boosted decision tree model that:
- Takes 23 engineered tabular features as input (storm state, velocity, acceleration,
  turn rate, environmental conditions)
- Predicts displacement in km at +6h, +12h, and +24h horizons
- Trained on historical cyclone tracks from the IBTrACS/ERA5 dataset
- Evaluated using storm-disjoint cross-validation and a held-out test set

**Key input features**:
- Current storm position (lat, lon) and intensity (wind, pressure, SST)
- Velocity components at multiple time steps
- Acceleration and turn rate
- 24-hour trend
- Environmental wind components

### 3. GRU Neural Network (Phase-4/5 Production Champion)

The production inference pipeline uses a GRU (Gated Recurrent Unit) neural network:
- Input: 5-step temporal sequence of 16 engineered features
- Output: Predicted lat, lon, and wind speed at +6h, +12h, +24h
- Loss: Huber loss for robust regression
- Experiment ID: EXP005

This model is served through the phase5/phase6 API architecture.

## What We Do NOT Claim

1. **No coupled atmosphere-ocean simulation**: We do not solve primitive equations
   or simulate atmosphere-ocean dynamics. Our models learn statistical relationships
   from historical data.

2. **No data assimilation**: We do not implement 4D-Var, EnKF, or any operational
   data assimilation system.

3. **No explicit numerical weather prediction**: We do not run any NWP model (WRF,
   GFS, ECMWF, etc.).

4. **No physics-informed neural network**: While our input features include
   physically meaningful quantities (pressure deficit, wind components), the model
   itself does not embed physical constraints or conservation laws.

5. **No direct landfall prediction**: Landfall distance is computed as a post-hoc
   geometric calculation from the predicted track to the nearest coastline point.

## Honest Architecture Summary

```
Input: 5-timestep history of (lat, lon, wind, pressure, SST, wind_u, wind_v)
  |
  v
Feature Engineering: 16 tabular features (velocity, acceleration, trends)
  |
  +---> Movement-Vector baseline (physics: linear extrapolation)
  |
  +---> CatBoost displacement model (ML: gradient-boosted trees)
  |
  +---> GRU neural network (ML: recurrent neural network)
  |
  v
Output: Predicted position at +6h, +12h, +24h
  |
  v
Post-processing: Haversine distance to coastline (landfall heuristic)
```

## Validation Methodology

- **Storm-disjoint splits**: Train/val/test contain different cyclones
- **GroupKFold CV**: 5-fold cross-validation by cyclone ID
- **No test-set tuning**: All model selection uses validation only
- **Physical metrics**: Track error in km (Haversine distance)

## References

- Movement-Vector: Standard kinematic extrapolation used in operational forecasting
- CatBoost: Prokudin et al., "CatBoost: unbiased boosting with categorical features", NeurIPS 2018
- GRU: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder", EMNLP 2014
