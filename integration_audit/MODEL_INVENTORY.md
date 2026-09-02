# Model Inventory — SIH 2026 PS 26070

Philosophy: **no model/weights/checkpoint is a deletion candidate.** All of the following are KEEP.

## P2 Detection
| File | Framework | Architecture | Input | Output | Purpose | Used by | Duplicated? | Integration needs? |
|---|---|---|---|---|---|---|---|---|
| `PS70-main/models/detection/model_weights.pt` (4,739,155 B, in zip) | PyTorch | `CycloneDetector` — MobileNetV3-small backbone + 3 heads (presence/pattern/category) | satellite crop resized 224×224 (B,3,224,224) | presence logit, pattern logits (3), category logits (7) | detect cyclone + tag structural pattern + category | `src/detection/inference.py::CycloneInference` | no | YES — `/api/analyze.detection` and/or `/api/detect` |
| `models/detection/placeholder.txt` (1 B, in zip) | — | — | — | — | naive init marker | repo layout | no | no |
| `models/detection/sample_prediction.png` (526,730 B) | — | — | — | — | eval visual | none (reference) | no | no |

Metrics (`metrics/detection_metrics.json`): pattern acc 0.714 / F1 0.631; category acc 0.333 / F1 0.231 (weak category head — flag for integration review, not a fix).

## P3 Classification
| File | Framework | Architecture | Input | Output | Purpose | Used by | Duplicated? | Integration needs? |
|---|---|---|---|---|---|---|---|---|
| `models/classification/image_only_model.pt` (45,186,485 B, in zip) | PyTorch | `ImageOnlyIntensityModel` = ResNet18 (ImageNet init) + dropout MLP classifier (7) + wind regressor (1) | 256×256 RGB (B,3,256,256) | 7-class logits + wind (km/h) | image-only intensity | `src/classification/inference.py::get_image_model` | no | YES `/api/analyze.classification` |
| `models/classification/tabular_multisource_model.pkl` (4,816,632 B) | scikit-Learn/LightGBM (pickle) | `MultisourceTabularModel`: scaler + LGBMClassifier + LGBMRegressor(wind) + LGBMRegressor(pressure) | (1,6) lat,lon,sst,pressure_msl,wind_u,wind_v | cat idx (7), wind km/h, pressure hPa, confidence | tabular multi-source (recommended) | `get_tabular_model` | no | YES |
| `models/classification/metrics_comparison.json` (661 B) | — | — | — | — | eval: image acc 38.1% vs multi 47.0%; multi wind MAE 18.84 vs image 109.81 | eval | no | no |
| `models/classification/confusion_matrix.png` (243,059 B) | — | — | — | — | eval visual | — | no | no |

## P4 Forecasting (workspace `p4_forecasting/`)
| File | Framework | Arch | Input | Output | Purpose | Used by | Notes |
|---|---|---|---|---|---|---|---|
| `phase4/results/experiments/EXP005/checkpoint.pt` (362,018 B) | PyTorch | **EXP005 GRU** (input_size 16, hidden 96, layers 2, dropout 0.1, Huber) | normalized (1,5,16) | (1,3,9) → (3,3) lat/lon/wind @ +6/12/24h | **CHAMPION** (validation primary score ~113.07) | `phase5/inference/predictor.py`, `phase6` API | hashes differ from EXP003/EXP006 same-size files → not duplicates |
| `phase4/results/experiments/EXP00{1..6}/checkpoint.pt` | PyTorch | LSTM/GRU/MTLSTM variants | (1,5,16) | (1,3,9) | ablation registry | phase4 | EXP001 88,934 B; EXP002/004 480,290/480,866 B; EXP003/005/006 362,018 B (all distinct hashes) |
| `phase3/checkpoints/best_lstm.pt` (79,708 B) | PyTorch | Phase-3 baseline LSTM | (1,5,7) | (1,3,9) | historical baseline | phase3 eval/report | historical, KEEP |
| `models/lstm_forecaster.pt` (79,768 B, workspace `models/`) | PyTorch | Phase-3 trained LSTM copy | (1,5,7) | (1,3,9) | legacy artifact | legacy references | historical, KEEP |
| `phase4/results/experiments/EXP005/metrics.json`, `training_history.json`, `test_results.json`, `validation_results.json` | json | — | — | — | experiment provenance | audit | KEEP |
| `phase4/results/champion_model.json`, `champion_rationale.json`, `FINAL_COMPARISON.json`, `experiments_summary.json` | json | — | — | — | champion selection evidence | audit | KEEP |

P4 model support files: `phase4/configs/EXP*.json` (== copies under each `results/experiments/EXP*/config.json`, all hash-identical — intentional experiment registry snapshots, KEEP both), `phase4/results/normalization_stats.json`.

## Pre-existing workspace models not covered above
`models/model_config.json` (679 B), `models/preprocessing_config.json` (3,263 B) — legacy Phase-3 configs, KEEP.

Summary: **9 trained artifacts** (P2 1 pt · P3 1 pt + 1 pkl · P4 6 experiment checkpoints + 1 phase3 pt). All verified present in the newly downloaded archive/workspace; none duplicated byte-for-byte; none automatically deletable.