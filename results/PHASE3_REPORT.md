# Phase 3 Report: Baseline + LSTM Forecaster

Generated: 2026-08-26 09:40:35

---

## 1. Preprocessing

### 1.1 Input Data

| Property | Value |
|----------|-------|
| Source | Phase 2 sequences |
| X_train shape | (7174, 9, 7) |
| X_val shape | (1563, 9, 7) |
| X_test shape | (1191, 9, 7) |
| Features | 7 (6 continuous + 1 missingness indicator) |
| Sequence length | 9 timesteps (24h at 3h resolution) |

### 1.2 Feature Descriptions

| Index | Feature | Unit | Normalized |
|-------|---------|------|------------|
| 0 | latitude | °N | Yes (z-score) |
| 1 | longitude | °E | Yes (z-score) |
| 2 | wind_speed_kmh | km/h | Yes (z-score) |
| 3 | pressure_hpa | hPa | Yes (z-score) |
| 4 | storm_speed | kts | Yes (z-score) |
| 5 | storm_direction | degrees | Yes (z-score) |
| 6 | pressure_hpa_missing | binary | No (stays 0/1) |

---

## 2. Missing Pressure Handling

### Strategy

1. **Imputation:** Missing pressure values replaced with training-set mean (992.2 hPa)
2. **Indicator:** Binary feature (index 6) set to 1.0 wherever pressure was imputed
3. **Normalization:** Continuous features normalized using z-score; indicator NOT normalized

### Rationale

- The missingness indicator preserves the information that pressure was unavailable
- Using training-set mean avoids information leakage from val/test sets
- The model can learn to weight pressure-derived features less when the indicator is 1.0

### Statistics

| Set | Pressure missing | Indicator=1 |
|-----|-----------------|-------------|
| Train | 28,782 | 28,782 |
| Val | 13 | 13 |
| Test | 96 | 96 |

---

## 3. Persistence Baseline

The persistence baseline predicts that the current cyclone state persists unchanged.

**Rule:** For each horizon (+6h, +12h, +24h), predict:
- future latitude = current latitude (last input timestep)
- future longitude = current longitude
- future wind = current wind speed

### Results

| Horizon | Lat MAE (°) | Lon MAE (°) | Wind MAE (km/h) | Track Error (km) |
|---------|-------------|-------------|-----------------|------------------|
| +6h | 0.363 | 0.541 | 5.9 | 78.4 |
| +12h | 0.720 | 1.079 | 11.2 | 155.5 |
| +24h | 1.418 | 2.133 | 20.7 | 306.1 |

---

## 4. LSTM Architecture

### Model: CycloneLSTM

```
Input (batch, 9, 7)
  → LSTM (hidden=64, layers=1, dropout=0.0)
  → Linear (hidden=64 → 9)
  → Reshape (batch, 3, 3)
Output
```

| Property | Value |
|----------|-------|
| Hidden size | 64 |
| Number of layers | 1 |
| Dropout | 0.0 |
| Parameters | 19,273 |
| Output | (batch, 3 horizons × 3 targets) |

### Output Structure

| Index | Horizon | Target |
|-------|---------|--------|
| [0] | +6h | latitude, longitude, wind_speed |
| [1] | +12h | latitude, longitude, wind_speed |
| [2] | +24h | latitude, longitude, wind_speed |

---

## 5. Training Process

| Property | Value |
|----------|-------|
| Optimizer | Adam |
| Loss function | Huber Loss |
| Learning rate | 0.001 (initial) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |
| Early stopping patience | 15 epochs |
| Gradient clipping | max_norm=1.0 |
| Batch size | 64 |
| Random seed | 42 |

### Configuration Search Results

| Config | Hidden | Layers | Dropout | Best Val Loss | Training Time |
|--------|--------|--------|---------|---------------|---------------|
| A | 64 | 1 | 0.0 | 0.041945 | 10.3s **BEST** |
| B | 128 | 1 | 0.0 | 0.042124 | 11.3s |
| C | 128 | 2 | 0.1 | 0.043242 | 12.2s |

---

## 6. Evaluation Metrics

### Metrics Explained

1. **Latitude MAE:** Mean absolute error in degrees north
2. **Longitude MAE:** Mean absolute error in degrees east
3. **Wind MAE:** Mean absolute error in km/h
4. **Track Error:** Great-circle (haversine) distance in km between predicted and actual position

---

## 7. Baseline vs LSTM Comparison

| Horizon | Metric | Persistence | LSTM | Improvement |
|---------|--------|-------------|------|-------------|
| +6h | Lat MAE (°) | 0.363 | 0.343 | +5.6% |
| | Lon MAE (°) | 0.541 | 0.834 | -54.1% |
| | Wind MAE (km/h) | 5.9 | 6.6 | -11.7% |
| | Track Error (km) | 78.4 | 104.8 | -33.6% |
| +12h | Lat MAE (°) | 0.720 | 0.425 | +40.9% |
| | Lon MAE (°) | 1.079 | 0.933 | +13.6% |
| | Wind MAE (km/h) | 11.2 | 11.2 | -0.0% |
| | Track Error (km) | 155.5 | 120.5 | +22.5% |
| +24h | Lat MAE (°) | 1.418 | 0.745 | +47.5% |
| | Lon MAE (°) | 2.133 | 1.413 | +33.8% |
| | Wind MAE (km/h) | 20.7 | 19.2 | +7.1% |
| | Track Error (km) | 306.1 | 189.0 | +38.3% |

**Interpretation:** Positive improvement means LSTM outperforms persistence.
Negative values mean persistence is better.

---

## 8. Best Model Configuration

| Property | Value |
|----------|-------|
| Configuration | A |
| Hidden size | 64 |
| Number of layers | 1 |
| Dropout | 0.0 |
| Best validation loss | 0.041945 |
| Total training time | 10.3s |
| Epochs trained | 22 |

---

## 9. Limitations

1. **Track-only features:** The model uses only cyclone track parameters (lat, lon, wind, pressure, speed, direction). No atmospheric context (ERA5) or satellite imagery (INSAT) is included.

2. **Single data source:** Wind and pressure come from a coalesced hierarchy (IMD → WMO → JTWC). Agency transitions may introduce value jumps.

3. **34% missing pressure:** The imputation strategy (training mean + indicator) is simple but may not capture the true relationship between pressure and storm intensity.

4. **Fixed architecture:** Only 3 configurations were tested. A more thorough hyperparameter search might find better settings.

5. **No temporal features:** The model does not explicitly encode time-of-year, which could help capture seasonal patterns in cyclone activity.

6. **No physics constraints:** Predictions may not satisfy physical conservation laws. Future work could add physics-informed loss terms.

7. **Short evaluation period:** The test set covers ~2019-2025. Performance on earlier decades is not evaluated.

---

## 10. Generated Files

| File | Description |
|------|-------------|
| `models/lstm_forecaster.pt` | Best model weights |
| `models/model_config.json` | Model configuration |
| `models/preprocessing_config.json` | Preprocessing parameters |
| `results/persistence_baseline.json` | Persistence baseline metrics |
| `results/lstm_results.json` | LSTM test metrics |
| `results/model_comparison.json` | Head-to-head comparison |
| `results/plots/training_loss_curves.png` | Training curves for all configs |
| `results/plots/example_forecast.png` | Example predicted vs actual |
| `results/plots/track_error_comparison.png` | Track error by horizon |
| `results/plots/wind_prediction_examples.png` | Wind prediction examples |
| `src/forecasting/inference.py` | Reusable inference module |
| `results/PHASE3_REPORT.md` | This report |
