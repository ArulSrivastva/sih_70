# Dataset Inventory — SIH 2026 PS 26070

Everything below is read from `PS70-main.zip` (not extracted) unless noted as workspace.

---

## 1. IBTrACS

| Dataset | Path (in zip) | Format | Size | Rows | Date range | Coverage | Columns | Labels | Raw/Proc | Dup? | Used by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Raw | `data/raw/ibtracs/ibtracs_NI_raw.csv` | csv | 27,875,881 | ~40k+ obs | 1980–2025 | North Indian Ocean (3 basins) | IBTrACS v04r01 native | wind/pressure | raw | no | P1 pipeline |
| Clean | `data/metadata/ibtracs_clean.csv` | csv | 1,487,310 | 18,168 | 1980-10-10 → 2025-12-02 | NIO, 471 cyclones | cyclone_id..category + era5 | IMD category | processed | identical copy → `p4_forecasting/_source_p1` | P1/P4 |
| ERA5-joined | `data/metadata/ibtracs_with_era5.csv` | csv | 709,456 | 5,481 | 2013→2025 | NIO, 151 cyclones | + sst, wind_u, wind_v, pressure_msl | IMD | processed | no | P1 |
| Master | `data/metadata/master_dataset.csv` | csv | 738,948 | 5,481 | 2013-05-09 → 2025-12-02 | 3 basins, 151 cyclones | `cyclone_id,season,name,subbasin,timestamp,lat,lon,wind_speed,pressure,category,sst,wind_u,wind_v,pressure_msl,pre_genesis_favorable` | IMD category | processed | mirror `data/processed/master_dataset.csv` (same bytes) | P1/P3/P4 |
| Split manifests | `train.csv` 3,911 / `validation.csv` 752 / `test.csv` 818 (+ `*_cyclones.csv` ID indexes) | csv | — | — | 2013→2025 | cyclone-level (no leakage) | same as master | — | processed | no | P3/P4 |

## 2. ERA5

| Item | Value |
|---|---|
| Files | 94 × NetCDF `.nc` (`raw/era5/era5_YYYY_MM.nc`) |
| Total size | 1,103,681,364 B (~1.10 GB) |
| Years / months | 2013–2025 (13 y; 5–10 monthly files per year) |
| Variables | sea_surface_temperature (K→°C), mean_sea_level_pressure (Pa→hPa), 10m u/v wind (m/s) |
| Spatial resolution | 0.25° × 0.25° (~28 km) |
| Bounding box | N 30°, S −5°, W 50°E, E 105°E (covers BoB + Arabian Sea) |
| Temporal | 3-hourly (8 samples/day) |
| Extraction | nearest-neighbour via xarray; 5,481/5,481 observations matched (100%); QA report notes sst has 1,538 missing in the master while README claims 0 — **the repo is internally inconsistent on sst coverage; do not "fix" it in this audit** |
| Used by | P3 (tabular features), P4 (per-timestep env features) |

## 3. INSAT / Satellite

| Dataset | Path | Format | Count | Size | Labels | Raw/Proc | Dup? |
|---|---|---|---|---|---|---|---|
| Reference visual | `raw/insat/insat3d_for_reference_ds/CYCLONE_DATASET/` | jpeg/png | 143 | part of 47.5 MB | none (reference archive) | raw | **whole tree duplicated in `raw/insat_kaggle/`** |
| Infrared | `raw/insat/insat3d_ir_cyclone_ds/CYCLONE_DATASET_INFRARED/` | jpg | 136 | same | none | raw | duplicated |
| Raw/RGB | `raw/insat/insat3d_raw_cyclone_ds/CYCLONE_DATASET_FINAL/` | jpg/JPEG | 140 | same | none | raw | duplicated |
| Label sheet | `raw/insat/insat_3d_ds - Sheet.csv` | csv | 1 | — | sheet metadata | raw | duplicated |
| Kaggle image-only set | `data/processed/classification/image_only_kaggle/images/` | jpg/jpeg | 133 (67 "plain" + 66 "variant(1..k)" frames) | ~5 MB | `labels.csv`, `train/val/test_labels.csv` (93/19/21) | processed | **no byte-identical duplicates** (hash-checked all 133) |
| Detection manifests | `data/processed/detection/{train,val,test}_detection.csv` + `detection_all.csv` | csv | 133 rows | — | `cyclone_detected`, `structural_pattern` (eye_visible/curved_band/shear_pattern), `category`, `mock_bbox` | processed | image_path points to the 133 shared Kaggle images |

## 4. Forecasting datasets

| Dataset | Path | Format | Shape | Counts | Features | Targets | Used by | Notes |
|---|---|---|---|---|---|---|---|---|
| Dataset C (P1) | `data/processed/forecasting/*_sequences.npz` | npz | X (N,5,7), Y (N,3,3) | 2275/378/423 = 3,076 | lat, lon, wind_speed, pressure, sst, wind_u, wind_v | lat/lon/wind at +6/+12/+24h | P4 entry | value-identical to p4 `canonical/` npz (array-equality confirmed) |
| P4 canonical | `p4_forecasting/canonical/*.npz` | npz | X(N,5,7) Y(N,3,3) | same | 7 | 3 | P4 phase2 | byte-differs from P1 (compression) but array-equal |
| P4 chrono | `p4_forecasting/canonical_chrono/*.npz` | npz | (5,7)/(3,3) | 2259/416/401 | 7 | 3 | P4 phase2 alt | chronological split |
| P4 clean | `p4_forecasting/phase2/results/canonical_chronological_clean/*` | npz/csv | (5,7)/(3,3) | 1212/231/198 | 7 | 3 | P4 phase3-4 | clean chrono split used downstream |
| **P4 FINAL features** | `p4_forecasting/phase4/results/feature_dataset/{train,val,test}.npz` | npz | **X (N,5,16)**, Y (N,3,3) | 1212/231/198 | 16 (7 raw + 9 derived) | lat/lon/wind | **EXP005 champion** | final flow input |
| P4 normalization | `p4_forecasting/phase4/results/normalization_stats.json` | json | — | train-only | 16-feature order + mean/std + Y stats | — | phase5/6 | authoritative contract |
| Legacy 9-step | workspace `data/processed/sequences/*.npy` | npy | X (N,9,7), y (N,3,3) | 7174/1563/1191 | 7 | 3 | early P4 Phase-2 attempts | superseded, keep (dataset, not deletion candidate) |

## 5. P2 inputs/outputs
- Inputs: `data/processed/detection/*.csv` (133 rows) → images at `processed/classification/image_only_kaggle/images/`.
- Outputs: `metrics/detection_metrics.json`; inference result dict `{detected, confidence, structural_pattern, pattern_confidence, category, category_confidence}`.

## 6. P3 inputs/labels/models/outputs
- Inputs: tabular `processed/classification/multisource_*.csv` (4,208 rows, 6 features) and images (133, 256×256).
- Labels: 7-tier IMD category + wind (km/h) + pressure (hPa); Label scheme = IMD scale (D/DD/CS/SCS/VSCS/ESCS/SuCS).
- Models: `models/classification/image_only_model.pt` (ResNet18 dual-head), `tabular_multisource_model.pkl` (LightGBM).
- Outputs: `/api/analyze`-shaped dict `{category, imd_code, wind_speed, pressure, confidence}`.

Dataset category groups: **IBTrACS** (raw→clean→joined→master→splits) · **ERA5** (94 nc) · **INSAT** (419 image tree + 133 kaggle images) · **Forecasting** (5,7→3,3 npz) · **Detection** (133 manifests) · **Classification** (4,208 tabular + 133 image).

Total per tree: 5,481 master obs · 18,168 clean obs · 3,076 sequences (P1) · 4,208 tabular · 133 image · 419 raw INSAT images (in a duplicated tree).