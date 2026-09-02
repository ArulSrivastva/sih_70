# Model Report - LSTM Cyclone Forecast (P4 Phase 3, PS 26070 / SIH 2026)

- Generated: 2026-08-29T08:50:36.097006+00:00

## Dataset

- **Split**: chronological, cyclone-disjoint (train/val/test by cyclone, no cyclone appears in more than one split).
- **Policy**: CLEAN-only (Phase-1 `sample_quality.csv` filter). Excludes every sequence with any interpolated/filled (non-causal) cell.
- **Sequence dimensions**: history (5, 7); targets (3, 3).
- **Train**: 1212 sequences / 57 cyclones
- **Validation**: 231 sequences / 13 cyclones
- **Test**: 198 sequences / 10 cyclones
- **Input features (exact order)**: lat, lon, wind_speed, pressure, sst, wind_u, wind_v.
- **Targets**: lat, lon, wind_speed (per horizon).
- **SST unit**: degrees Celsius (as delivered; not converted to Kelvin).

## Model

```
history (batch,5,7)
  -> LSTM(7 -> 64, num_layers=1, batch_first=True, dropout=0)
  -> last hidden state
  -> Linear(64 -> 9)
  -> reshape (batch, 3 horizons x 3 targets)
```

- Trainable parameters: **19273**
- Optimizer: Adam, lr=0.001, batch_size=64, seed=42

## Normalization

Normalization statistics were calculated exclusively from the training split and then applied unchanged to validation and test data.

- z = (x - mean) / std for all 7 features; targets normalized the same way.
- Zero-variance or NaN/Inf statistics are a hard failure (never silently replaced).
- Source NPZ files are never modified.

## Results (TEST split, primary)

| Model           | Horizon | Track Error km (mean) | Wind MAE km/h | Wind RMSE km/h |
| --------------- | ------: | --------------------: | ------------: | -------------: |
| persistence     |    6h | 64.69 | 3.12 | 5.86 |
| persistence     |   12h | 123.94 | 6.46 | 10.30 |
| persistence     |   24h | 227.33 | 13.22 | 19.66 |
| movement_vector |    6h | 38.05 | 3.12 | 5.86 |
| movement_vector |   12h | 80.50 | 6.46 | 10.30 |
| movement_vector |   24h | 180.66 | 13.22 | 19.66 |
| lstm            |    6h | 130.02 | 6.71 | 8.39 |
| lstm            |   12h | 180.00 | 10.81 | 13.41 |
| lstm            |   24h | 267.54 | 16.95 | 21.65 |

## Baseline improvement (LSTM vs baselines, test split)

Improvement % = (baseline - lstm) / |baseline| * 100; positive = LSTM better.

| Horizon | vs Persistence track err % | vs Persistence wind MAE % | vs Movement-vector track err % | vs Movement-vector wind MAE % |
| ------: | -------------------------: | ------------------------: | -----------------------------: | ----------------------------: |
|     6h | -101.00 | -114.95 | -241.69 | -114.95 |
|    12h | -45.24 | -67.38 | -123.59 | -67.38 |
|    24h | -17.69 | -28.21 | -48.09 | -28.21 |

- **+6h**: persistence baseline beats the LSTM on mean track error (-101.00%); movement-vector beats the LSTM on mean track error (-241.69%).
- **+12h**: persistence baseline beats the LSTM on mean track error (-45.24%); movement-vector beats the LSTM on mean track error (-123.59%).
- **+24h**: persistence baseline beats the LSTM on mean track error (-17.69%); movement-vector beats the LSTM on mean track error (-48.09%).

## Validation results

| Model           | Horizon | Track Error km (mean) | Wind MAE km/h | Wind RMSE km/h |
| --------------- | ------: | --------------------: | ------------: | -------------: |
| persistence     |    6h | 68.48 | 6.64 | 10.33 |
| persistence     |   12h | 136.50 | 13.63 | 19.18 |
| persistence     |   24h | 282.13 | 27.50 | 37.86 |
| movement_vector |    6h | 28.23 | 6.64 | 10.33 |
| movement_vector |   12h | 62.59 | 13.63 | 19.18 |
| movement_vector |   24h | 139.75 | 27.50 | 37.86 |
| lstm            |    6h | 142.62 | 8.27 | 10.38 |
| lstm            |   12h | 193.00 | 13.44 | 16.87 |
| lstm            |   24h | 300.66 | 25.61 | 31.13 |

## Training / overfitting check

- Best epoch: 46  (patience 12)
- Best validation loss: 0.1028912728612041
- Final train loss: 0.029184759904940922
- Final validation loss: 0.10690838524273463
- Training time: 16.38 s
- Device: cpu


## Config / reproducibility

```json
{
  "input_size": 7,
  "hidden_size": 64,
  "num_layers": 1,
  "output_size": 9,
  "learning_rate": 0.001,
  "batch_size": 64,
  "seed": 42,
  "max_epochs": 100,
  "patience": 12,
  "horizons_hours": [
    6,
    12,
    24
  ],
  "parameters": 19273,
  "architecture": "LSTM(7->64) -> Linear(64->9) -> reshape(3,3)"
}
```
