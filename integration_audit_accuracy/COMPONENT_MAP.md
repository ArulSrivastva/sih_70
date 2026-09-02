# Component Map

Mapping from deliverables to the code that produces them, verified by reading every referenced source file in `PS70-main.zip` and `p4_forecasting`.

## P1 — Data pipeline

| Component | File (archive path unless noted) | Produces |
|---|---|---|
| Raw archive merge | `src/data/build_datasets.py` | `data/metadata/ibtracs_with_era5.csv` from `ibtracs_clean.csv` + ERA5 `.nc` extraction; renames columns |
| Master | `data/metadata/master_dataset.csv` (+ `data/processed/master_dataset.csv`, byte-identical) | 5,481×15; wind/pressure/category/sst with missingness; `pre_genesis_favorable` |
| Splits | `data/metadata/{train,validation,test}.csv` | 105/22/24 cyclones, disjoint |
| Dataset A (detection) | `build_datasets.py` | `data/processed/detection/*_detection.csv` + `detection_all.csv` — **labels synthesized from wind**: `cyclone_detected=True` always, `mock_bbox` constant `[420,190,600,370]` always, `structural_pattern` by wind-threshold rule, `category` by IMD wind mapping |
| Dataset B (classification) | `src/data/clean_kaggle_intensity.py` + `build_datasets.py` | `data/processed/classification/multisource_*.csv` (ERA5 features + IBTrACS targets); `image_only_kaggle/{labels,train,val,test}_labels.csv` (labels parsed from Kaggle INSAT-3D sheet, wind kt → kmh ×1.852, IMD categories) |
| Dataset C (forecasting) | `build_datasets.py` | `data/processed/forecasting/*_sequences.npz` X(5,7)/Y(3,3); windows built with `interpolate(method='time').ffill().bfill()` (bfill can reach forward in time within a storm) |
| ERA5 reanalysis | `data/raw/era5/*.nc` (94 files) | `u_wind/v_wind/sst/pressure_msl` joined onto all 5,481 rows |

## P2 — Detection

| Component | File | Notes |
|---|---|---|
| Model | `src/detection/detector.py` — `CycloneDetector` (MobileNetV3-small backbone, presence/pattern/category heads) | ImageNet-pretrained backbone |
| Training | `src/detection/train.py` | 10 epochs, Adam lr 1e-4, batch 16; label maps built from combined train+val+test frames; saves best on val |
| Weights | `models/detection/model_weights.pt` | checkpoint keys: `model_state_dict`, `pattern_to_idx` (3 classes), `category_to_idx` (7 classes) |
| Evaluation | `src/detection/evaluate.py` | Resize(224,224)+ToTensor; weighted precision/recall/F1; **presence head is never scored** |
| Consumed runtime contract | `src/detection/inference.py` | `load_cyclone_detector` + `detect_cyclone(image)` |

## P3 — Classification

| Component | File | Notes |
|---|---|---|
| Image model | `src/classification/classifier.py` — `ImageOnlyIntensityModel` (ResNet18 backbone + 7-class head + wind regressor head) | ImageNet-pretrained |
| Tabular model | `classifier.py` — `MultisourceTabularModel` (StandardScaler + LightGBM classifier/regressors; RandomForest fallback) | **Requires lightgbm** |
| Fusion stub | `MultimodalFusionModel` (defined, not trained) | No weights shipped |
| Weights | `models/classification/image_only_model.pt`, `tabular_multisource_model.pkl` | pkl contains unpickled LGBM + scaler |
| Evaluation | `src/classification/evaluate.py` | Image model on 21 test images; tabular model on 651 test records, then folds them into one `performance_delta` (cross-population comparison) |
| Inference | `src/classification/inference.py` | `classify_cyclone(image_input=…, environmental_data=…)` |

## P4 — Forecasting

| Component | File | Notes |
|---|---|---|
| Clean chrono data | `p4_forecasting/canonical_chrono/*` (npz + meta, prefix-hashed) | 3 h cadence windows; SHA256 prefixes train `df70303e…`, val `48cf065d…`, test `89e9c2e2…` |
| Feature engineering | `p4_forecasting/phase4/features/feature_engineering.py` | (5,7)→(5,16): 9 causal derived features; 0-fill at step 0; no future consulted |
| Normalization | `p4_forecasting/phase4/training/normalization.py` → `results/normalization_stats.json` | train-only z-score; zero-std→scale 1 |
| Model | `p4_forecasting/phase4/models/gru.py` — `GRUCyclone` (16→GRU→Linear→(3,3)) | |
| Experiments | `phase4/results/experiment_registry.csv`, `phase4/results/experiments/EXP00X/{config,metrics,validation_results,test_results,checkpoint.pt,source_hashes}` | EXP001–EXP006; only EXP005 has a `test_results.json` (test evaluated once) |
| Selection | `phase4/results/champion_model.json` | EXP005 (val-only rule) |
| Baselines | `phase2/baselines/*.py`, `phase2/results/baseline_results.json` | persistence + movement-vector on clean 1,212/231/198 |
| Output contract | `phase5/inference/predictor.py`, `phase5/config.py` | lon→[0,360), lat∈[-90,90], wind≥0; NaN/Inf refusal |
| API | `phase6/api/{app,routes}.py`, `phase6/schemas/*` | GET /health, GET /model, POST /forecast, POST /forecast/compare |

## P5 — Frontend

| Component | File | Notes |
|---|---|---|
| API client | `cyclone-dashboard/src/api/client.js` | `USE_MOCK=true`; targets `/api/detect…` and merged `POST /api/analyze` shape |
| Mock payload | `cyclone-dashboard/src/api/mockData.js` | `{meta, detection, classification, forecast[{hour,label,lat,lon,windSpeedKmh,pressureHpa,confidence}], landfall, risk, satellite}` |
| App bootstrap | `cyclone-dashboard/src/App.jsx` | calls `fetchAnalyze()` once on mount → mock only |
| Pristine copy | `SIH26/cyclone-dashboard/cyclone-dashboard` (29 files) | older snapshot, same `USE_MOCK=true` behavior |

## Runtime wiring map (frontend ↔ backend endpoints)

| Frontend call (when real mode on) | Delivered backend | Resolution |
|---|---|---|
| GET `{API_BASE}/analyze` (`/api/analyze`) | not exposed | **404** (verified live) |
| GET `{API_BASE}/detect` (`/api/detect`) | not exposed | **404** |
| GET `{API_BASE}/classify` (`/api/classify`) | not exposed | **404** |
| GET `{API_BASE}/forecast` (`/api/forecast`) | `POST /forecast` (no prefix; POST-only) | not compatible (method + path + shape) |
| (none) | `GET /health`, `GET /model` | not consumed by the frontend |
| (none) | `POST /forecast/compare` | not consumed by the frontend |