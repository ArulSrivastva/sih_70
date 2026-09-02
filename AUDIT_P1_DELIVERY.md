# AUDIT_P1_DELIVERY.md — P1 → P4 Data Delivery Audit

**Project:** SIH 2026 PS 26070 — Tropical Cyclone AI/ML System
**Auditor:** Person 4 (Cyclone Forecasting) · **Date:** 2026-08-29
**Mode:** READ-ONLY. Nothing was modified, deleted, retrained, or downloaded. All claims below were verified against actual file contents, sizes, headers, and sampled values (not documentation).

**Critical framing finding:** the files listed as "Previously reported P1 outputs" in the task brief (`ibtracs_forecasting_base.csv`, `sequences/X_*.npy`, `normalization_stats.json`, `SEQUENCE_REPORT.md`, `DATA_QUALITY_REPORT.md`, 9×7 / 9,928 sequences, CycloneLSTM) are **NOT in `PS70-main.zip`**. They live only in the local working tree (`C:\Users\aruls\Desktop\SIH26\ps70\cyclone-project\`) and belong to a **different, earlier in-repo pipeline** (chronological split, 3-h grid, NO ERA5 features). The ZIP contains a **different forecasting dataset**: 5-step × 7-feature npz arrays with ERA5 integrated (random split, 2,275/378/423 samples). Both are audited below.

---

## 1. Project / Directory Inventory

### 1.1 Local project tree (`cyclone-project\`)

```
cyclone-project\
├── PS70-main.zip                  (1,182,652,292 B — P1 delivery, UNEXTRACTED in project)
├── data\
│   ├── raw\
│   │   ├── era5\                  (EMPTY)
│   │   └── ibtracs\ibtracs.NI.list.v04r01.csv   (26.59 MB — raw IBTrACS v04r01 NI export)
│   └── processed\
│       ├── sequences\             X_train 1.72MB, X_val 0.38MB, X_test 0.29MB, y_train 0.25MB, y_val 0.05MB, y_test 0.04MB, *_metadata.csv
│       ├── sequence_visualizations\ (2 png)
│       ├── ibtracs_forecasting_base.csv   (1.3 MB — local phase-1 clean CSV)
│       ├── normalization_stats.json
│       ├── DATA_QUALITY_REPORT.md, SEQUENCE_REPORT.md  (local phase docs)
│       ├── P4_DATA_HANDOFF_REPORT.md, P4_HANDOFF_AUDIT.md  (prior audit outputs)
├── models\   lstm_forecaster.pt (0.08MB), model_config.json, preprocessing_config.json
├── scripts\  phase1_clean_dataset.py, phase2_build_sequences.py, phase3_baseline_lstm.py, phase3_5_audit.py
├── src\forecasting\inference.py, __init__.py
├── results\  plots\, lstm_results.json, model_comparison.json, persistence_baseline.json, PHASE3_REPORT.md, PHASE3_AUDIT.md
├── notebooks\  (empty)
└── DATASET_REPORT.md
```

### 1.2 ZIP contents (`PS70-main.zip`; inspected via read-only extraction)

| Path (inside ZIP) | Type | Size | Purpose |
|---|---|---|---|
| `data/raw/era5/era5_YYYY_MM.nc` (94) | NetCDF4 | 1,103,681,364 B | ERA5 SST/MSLP/U10/V10, 2013–2025 |
| `data/raw/ibtracs/ibtracs_NI_raw.csv` | CSV | 26.58 MB | raw IBTrACS NIO export (62,849 rows) |
| `data/raw/insat/` + `data/raw/insat_kaggle/` | images+CSV | ~47 MB | INSAT-3D imagery (duplicated dirs) |
| `data/metadata/ibtracs_clean.csv` | CSV | 1.42 MB | cleaned IBTrACS (18,168×12) |
| `data/metadata/ibtracs_with_era5.csv` | CSV | 0.68 MB | IBTrACS + ERA5 per-row (5,481×14) |
| `data/metadata/master_dataset.csv` | CSV | 0.70 MB | aligned master (5,481×15) |
| `data/metadata/{train,validation,test}.csv` + `*_cyclones.csv` | CSV | — | cyclone-level split tables/manifests |
| `data/processed/master_dataset.csv` | CSV | 0.70 MB | identical copy (md5 `6b3e93e8…`) |
| `data/processed/forecasting/{train,val,test}_sequences.npz` + `*_metadata.csv` + `README.md` | NPZ/CSV/MD | — | **P4 sequence arrays (5×7 → 3×3)** |
| `data/processed/classification/` (multisource CSVs, image_only_kaggle/) | CSV/img | — | P3 classification streams |
| `data/processed/detection/` | CSV | — | P2 detection manifests (mock bbox) |
| `data/qa_reports/QA_REPORT.md` + `figures/` | MD/PNG | — | QA report + 4 figures |
| `docs/ERA5_ALIGNMENT.md` | MD | — | ERA5 alignment spec |
| `future_task/tasks*.md` | MD | — | phase guidance |
| `get_ibtracs.py`, `download_era5.py`, `extract_era5_at_points.py`, `build_datasets.py`, `clean_kaggle_intensity.py`, `check_*.py`, `find_near_matches.py`, `prioritize_gap.py`, `split_coverage_gap.py` | PY | — | P1 preprocessing scripts |
| `src/data/{forecasting,classification,detection}_dataset.py`, `preprocess_satellite.py`, `dataloader_example.py` | PY | — | dataset loaders |
| `README.md`, `PS70_Person1_Status.md`, `ps70_team_plan_edited.pdf`, `LICENSE` | — | — | docs |

**The ZIP does NOT contain:** `ibtracs_forecasting_base.csv`, `sequences/*.npy`, `normalization_stats.json`, `SEQUENCE_REPORT.md`, `DATA_QUALITY_REPORT.md`, any model, or the phase1–3 scripts. Those are local-repo artifacts only.

---

## 2. IBTrACS Forecasting Dataset

### 2.1 `data/processed/ibtracs_forecasting_base.csv` (LOCAL repo — not in P1 ZIP)

| Property | Verified value |
|---|---|
| Rows / cols | 17,778 / 10 |
| Columns (exact) | `SID, timestamp, latitude, longitude, wind_speed_kmh, pressure_hpa, storm_speed, storm_direction, nature, subbasin` |
| dtypes | SID str · timestamp str · lat/lon/wind/pressure/storm_speed/storm_direction float · nature/subbasin str |
| Date range | 1980-10-10 06:00 → 2025-12-02 18:00 UTC |
| Unique SIDs | 471 |
| Missing | wind_speed_kmh 1,224 (6.9%) · pressure_hpa 6,050 (34.0%) · storm_speed 5 · storm_direction 5 · others 0 |
| Latitude | 0.70 – 31.00 °N |
| Longitude | 41.80 – 163.70 °E |
| Wind range | 5.56 – 277.8 km/h |
| Pressure range | 890 – 1,014 hPa |
| Inter-obs intervals | 3 h: 16,994 (≈97%) · 1 h: 116 · 2 h: 112 · 6 h: 22 · others (9 h…48 h): <100 |
| Chronological per SID | YES (verified monotonic within all 471) |
| Duplicates | 0 (also 0 for (SID, timestamp)) |

**ERA5-derived fields present in this CSV? NO — `sst`, `u_wind`, `v_wind`, `wind_u`, `wind_v`, `pressure_msl`, `category` are ALL ABSENT.** This is the local phase-1 base (IBTrACS-only, pre-ERA5). If P4 plans to use it as input, SST/u/v would have to be joined externally.

> Note: this file was NOT produced by the ZIP pipeline. P1's ZIP IBTrACS artifacts are `ibtracs_clean.csv` (18,168 rows, 471 cyclones, `cyclone_id` schema, IMD→WMO fallback, `wind_speed_kmh`, `pressure_hpa`, `category`) and the aligned subset (see §4).

---

## 3. ERA5 Audit (ZIP: `data/raw/era5/`, 94 files)

Per-file metadata verified with h5py (headers + ≤5 evenly spaced slices per variable per file; full-file scan done for `era5_2013_05.nc`):

| Property | Verified value |
|---|---|
| Format | NetCDF4 (HDF5; magic `89 48 44 46`), GRIB-derived attrs (`GRIB_centre: ecmf`) |
| Files | 94 (`era5_YYYY_MM.nc`), total 1,103,681,364 B |
| Months covered | May 2013 → Dec 2025 (only cyclone-active months; e.g., 2013 has 05,07,08,10,11,12) |
| Spatial bbox (from file coords) | `latitude` 30.0 → −5.0 °N (141 pts, 0.25°, descending); `longitude` 50.0 → 105.0 °E (221 pts, 0.25°) |
| Temporal resolution | 3-hourly, exactly {00,03,06,09,12,15,18,21} UTC |
| Timesteps total | 5,728 (range 2013-05-09 00:00 → 2025-12-03 21:00) |
| Dimensions | `valid_time` (≤744) × `latitude` (141) × `longitude` (221); `number` scalar 0; `expver` field = '01' |
| Variable names (exact, in-file) | **`sst`, `msl`, `u10`, `v10`** |
| Units (exact, in-file) | `sst` K · `msl` Pa · `u10` m s**-1 · `v10` m s**-1 |
| Missing values | `sst` **36.25 %** (present in all 94 files); `msl`/`u10`/`v10` **0 %** |

Requested-variable confirmation:

| Requested | In-file variable | Present? |
|---|---|---|
| sea_surface_temperature | `sst` | YES (K) |
| mean_sea_level_pressure | `msl` | YES (Pa) |
| 10m_u_component_of_wind | `u10` | YES (m/s) |
| 10m_v_component_of_wind | `v10` | YES (m/s) |

**Bounding box vs NIO forecast region:** file coordinates are 30→−5°N, 50→105°E — matches and slightly exceeds the intended 0–30°N / 40–100°E region (covers the full Bay of Bengal / Arabian Sea + 5°S / 105°E buffer). At its western edge it starts at 50°E (not 40°E), which does not truncate the NIO cyclone domain (longitude range of the aligned set is 41.8–141°E; rows <50°E are few and edge-clipped). **ERA5 spatial coverage: PASS.**

---

## 4. ERA5 ↔ IBTrACS Alignment

**Classification: A) Fully aligned** (with caveats listed below).

Verified evidence:
- `ibtracs_with_era5.csv` (5,481 rows) has per-observation values for `sst_celsius, u_wind, v_wind, pressure_msl_hpa` — i.e., ERA5 extracted **at cyclone locations**.
- `master_dataset.csv` = renamed aligned master (`lat, lon, wind_speed, pressure, category, sst, wind_u, wind_v, pressure_msl, pre_genesis_favorable`).
- Method (`extract_era5_at_points.py`): `xarray .sel(method="nearest")` on **time, latitude, longitude** — i.e., **nearest-neighbour interpolation (spatial + temporal)**; **bilinear was NOT used**. `number` dropped; `expver` merged by fillna; longitude re-normalized to 0–360 before lookup.
- Timestamp tolerance: temporal `.sel(method="nearest")` on a 3-hourly grid — effectively ≤3 h (1 row was 1 h off-grid at 14:00 → nearest 15:00). Spatial tolerance: ≤½ grid cell (0.125°).
- When ERA5 missing: per-row `None` → `sst` kept as NaN (raw ERA5 SST is ~36% NaN); `msl/u10/v10` complete. Missing SST never silently filled at this stage.
- Cross-validation: **12/12 sampled cyclone observations** re-read from the raw NetCDF match `master_dataset` exactly (ΔSST < 0.005 °C, Δu/v = 0.0 m/s, ΔMSLP < 0.05 hPa).
- ERA5 still stored separately (raw NetCDF in `data/raw/era5/`) AND already fused into `master_dataset.csv` and the forecasting npz.

Caveats: SST incomplete (28.1% of aligned rows missing upstream); 396 aligned rows (7.2%) lie outside the ERA5 bbox and therefore carry **edge-clipped** nearest-cell values (not flagged).

---

## 5. Satellite / INSAT Audit

| Question | Answer (verified) |
|---|---|
| Actual INSAT-3D/3DR data present? | Yes, INSAT-3D-style crops (IR imagery) — lineage is the **Kaggle "INSAT-3D cyclone detection" dataset** (label CSV `insat_3d_ds - Sheet.csv`; folder names `insat3d_*`), **not MOSDAC/official**. |
| Raw or preprocessed? | Raw copies under `data/raw/insat/` and `data/raw/insat_kaggle/` (byte-identical duplicates); a cleaned 133-image subset under `data/processed/classification/image_only_kaggle/images/`. |
| IR / visible? | IR (infrared) crops; no visible band. |
| File formats | .jpg/.jpeg (+2 .png in reference set) |
| Image counts | 140 (raw cyclone dataset, ~357×357 RGB) · 136 (IR subset) · 143 (reference) · 133 (processed subset) |
| Timestamps | **NONE** — no datetime metadata anywhere |
| Cyclone IDs | **NONE** |
| Geographic coordinates / georeferencing | **NONE** |
| Paired with IBTrACS observations | **NO** — can't be (no timestamps/geo). |
| Detection labels / bounding boxes | `data/processed/detection/*.csv` carry `cyclone_detected` (=True), `structural_pattern`, and **`mock_bbox`** (documented as integration contract placeholders) — i.e., **mock labels**, not real annotations. |
| Classification labels | 133 rows with `wind_speed_kt/ kmh` + `category`, but **label semantics are suspect**: `clean_kaggle_intensity.py` maps the Kaggle image-number column to knots (see §11). Affects P3, not P4 inputs. |

**INSAT is not usable for P4 forecasting** (no time/space join possible with the current annotations).

---

## 6. Sequence Audit

### 6.1 ZIP pipeline — `data/processed/forecasting/*_sequences.npz` (P1 deliverable, ERA5-integrated)

Verified (float32):

| Set | X shape | Y shape | Unique cyclones | t_zero range |
|---|---|---|---|---|
| train | (2,275, 5, 7) | (2,275, 3, 3) | 67 | 2013-05-10 → 2025-11-29 |
| val | (378, 5, 7) | (378, 3, 3) | 14 | 2013-11-07 → 2025-10-28 |
| test | (423, 5, 7) | (423, 3, 3) | 16 | 2013-05-30 → 2025-12-01 |

- **7 input features (exact order, from `features` array + `build_datasets.py`):** `[0] lat, [1] lon, [2] wind_speed, [3] pressure, [4] sst, [5] wind_u, [6] wind_v` — matches P4's intended input list **exactly**.
- **Timesteps:** 5 steps at **6-h spacing** (t−24, t−18, t−12, t−6, t). Window = 24 h.
- **Targets:** horizons **+6, +12, +24 h** × outputs **`[lat, lon, wind_speed]`** (order from `targets` array). Verified Y-row-vs-metadata max diff ≤ 6e-6.
- `X[:,-1,:3]` == metadata `origin_lat/lon/wind` (≤6e-6) → last step is "now".
- Metadata CSV columns: `cyclone_id, t_zero, t_minus_24h, t_plus_24h, origin_{lat,lon,wind}, target_{6,12,24}h_{lat,lon,wind}`.

### 6.2 Local repo pipeline — `sequences/*.npy` (earlier in-repo pipeline, NO ERA5)

Verified:

| Set | X shape | Y shape | Unique SIDs | Start dates |
|---|---|---|---|---|
| train | (7,174, 9, 7) | (7,174, 3, 3) | 243 | oldest era |
| val | (1,563, 9, 7) | (1,563, 3, 3) | 52 | middle era |
| test | (1,191, 9, 7) | (1,191, 3, 3) | 52 | most recent era |

- Total sequences 9,928 = 7,174+1,563+1,191 ✓.
- **7 input features (traced to `phase2_build_sequences.py` FEATURE_COLS + `preprocessing_config.json`):** `[0] latitude, [1] longitude, [2] wind_speed_kmh, [3] pressure_hpa, [4] storm_speed, [5] storm_direction, [6] pressure_hpa_missing` (binary). **No SST, no u/v wind** → does NOT match the intended 7-feature environmental spec.
- **Timesteps:** 9 steps at **3-h spacing** (t−24, t−21, …, t) → 27-h window (t−24…t).
- **Targets:** 3 horizons (+6 h = 2 steps, +12 h = 4 steps, +24 h = 8 steps) × `[latitude, longitude, wind_speed_kmh]`.
- Arrays contain NaNs (X_train 32,182 NaN, mostly wind/pressure; y_train 1,354 NaN) — pressure left NaN with indicator; targets NaN-masked at training.
- Loader `src/data/forecasting_dataset.py` (ZIP) is for the 5×7 npz; the local pipeline is loaded directly in `phase3_baseline_lstm.py`.

---

## 7. Normalization Audit (local pipeline)

| Item | Verified value |
|---|---|
| File | `data/processed/normalization_stats.json` + `models/preprocessing_config.json` |
| Method | **z-score** (per-feature mean/std) |
| Which features | 6 continuous: `latitude, longitude, wind_speed_kmh, pressure_hpa, storm_speed, storm_direction` |
| Feature 6 (missingness) | binary 0/1, **NOT normalized** |
| Target normalization | yes: `target_statistics` (latitude/longitude/wind) — denormalize `z*std+mean` at inference |
| Stats source | **training set only** (`n_valid`/`n_total` keys; documented; re-verified by `PHASE3_AUDIT.md` CHECK 1/2 as exact) |
| Val/test normalization | use **training** statistics (config `normalization.training_statistics`) |
| Pressure imputation | missing pressure → **training mean 992.196 hPa**; indicator feature idx 6 = 1 |
| Missing-value treatment | NaNs preserved in arrays; imputed at training (mean + indicator); NaN targets masked in loss |
| P1 npz normalization | **NONE** — raw float32 arrays; no scalers shipped; `CycloneForecastingDataset.get_feature_stats()` offers train-only stats if needed |

---

## 8. Train/Val/Test Leakage

| Check | Local pipeline | ZIP npz |
|---|---|---|
| Unique cyclones per split | 243 / 52 / 52 | 67 / 14 / 16 |
| Intersection (train∩val / train∩test / val∩test) | **0 / 0 / 0** | **0 / 0 / 0** |
| Split method | **Chronological** by cyclone start date (test = most recent) | **Random** 70/15/15 by cyclone_id (all splits span 2013–2025) |
| Cross-split temporal window overlap | None — each cyclone entirely in one split, windows belong to single storms | None by cyclone; but eras interleave (a 2013 test cyclone coexists with 2013 train cyclones) |
| Normalization using test stats | No (train-only) | N/A (no normalization) |
| Same cyclone in 2 splits | No (verified) | No (verified) |

**Conclusion:** no same-cyclone leakage in either dataset; the local pipeline adds chronological separation; the ZIP's random split is cyclone-disjoint but temporally interleaved (weaker for "forecast the future").

---

## 9. Forecasting Baseline + Model Files

| Item | Verified value |
|---|---|
| Persistence baseline | Predict = **last input timestep** (index 8) for all horizons (`phase3_baseline_lstm.py`; audit CHECK 8) |
| LSTM architecture (`CycloneLSTM`) | Input `(batch, 9, 7)` → `nn.LSTM(input_size=7, hidden_size=64, num_layers=1, batch_first=True, dropout=0)` → take last hidden `(batch,64)` → `nn.Linear(64, 9)` → `.view(-1, 3, 3)` |
| Output | `(batch, 3 horizons, 3 targets)` = +6/+12/+24 h × lat/lon/wind |
| Training config | Adam lr 1e-3, batch 64, max 150 epochs, ReduceLROnPlateau(0.5/7, min 1e-6), early-stop patience 15, seed 42; best config A → best_val_loss 0.041945, 10.3 s |
| Saved model | `models/lstm_forecaster.pt` (0.08 MB) + `model_config.json` + `preprocessing_config.json` |
| Metrics | `results/*.json` — LSTM beats persistence at +12 h (+22.5% track error) and +24 h (+38.3%) but **worse at +6 h** (104.8 vs 78.4 km) |
| Inference | `src/forecasting/inference.py::CycloneForecaster.forecast(seq)` accepts `(9,7)` or `(1,9,7)`; normalizes with stored train stats (features 0–5), runs LSTM, denormalizes targets, returns `{"forecast":[{hours:6|12|24, latitude, longitude, wind_speed_kmh}, …]}` — **verified logically consistent; can return +6/+12/+24 lat/lon/wind**. Note it loads `from scripts.phase3_baseline_lstm import CycloneLSTM` (requires that module importable). Does NOT accept the ZIP 5×7 npz natively (input width differs) — inference is tied to the local 9×7 pipeline. |

---

## 10. Reproducibility Check

Commands (from repo root; inspected, NOT run):

| Step | Script (local / ZIP) | Command |
|---|---|---|
| IBTrACS raw download | (manual) / `get_ibtracs.py` | `python get_ibtracs.py` (local: `python scripts/phase1_clean_dataset.py`) |
| ERA5 download | `download_era5.py` (ZIP) | `python download_era5.py` |
| ERA5 alignment | `extract_era5_at_points.py` (ZIP) | `python extract_era5_at_points.py` |
| Master + sequences + splits | `build_datasets.py` (ZIP) | `python build_datasets.py` |
| INSAT labels | `clean_kaggle_intensity.py` (ZIP) | `python clean_kaggle_intensity.py` |
| Local sequences (9×7) | `phase2_build_sequences.py` | `python scripts/phase2_build_sequences.py` (writes `sequences/*.npy`, `normalization_stats.json`) |
| Local norm config | phase2 helper | config written during phase2 run |
| Local LSTM train | `phase3_baseline_lstm.py` | `python scripts/phase3_baseline_lstm.py` (config A default) |
| Local audit | `phase3_5_audit.py` | `python scripts/phase3_5_audit.py` |
| Inference | `src/forecasting/inference.py` | `python -c "from src.forecasting.inference import CycloneForecaster; ..."` |

All examined scripts are self-contained (path constants at top); no destructive commands were run.

---

## 11. Critical Gaps

| Component | Present? | Validated? | Problem / GAP | Severity |
|---|---|---|---|---|
| IBTrACS raw (NIO v04r01) | YES | YES | — | OK |
| IBTrACS clean | YES | YES | 44.7% wind / 47.4% pressure missing (full era); 22–23% in aligned set | MEDIUM |
| ERA5 raw (94 files) | YES | YES | ~1.1 GB; only cyclone-active months | OK |
| ERA5 variables (sst/msl/u10/v10) | YES | YES | sst 36% missing; msl/u/v complete | MEDIUM |
| ERA5 alignment | YES (Fully aligned) | YES | nearest-neighbour only; edge-clipped for 396 rows (7.2%) | LOW |
| INSAT | YES | YES | no timestamps/geo/ID; Kaggle lineage | HIGH (P2/P3) |
| Satellite timestamps | NO | — | absent entirely | CRITICAL (P2/P3) |
| Satellite labels | PARTIAL | NO | `mock_bbox`; Kaggle label misread as knots | HIGH (P2/P3) |
| Master dataset | YES | YES | 2 identical copies; SST 28.1% missing; QA "100%" claim wrong | LOW |
| Forecast sequences (ZIP 5×7) | YES | YES | non-causal interpolation fills signals; random split | HIGH |
| Forecast sequences (local 9×7) | YES | YES | **no ERA5 (no sst/u/v)**; NaNs in arrays; masks targets | HIGH |
| Normalization | YES | YES | local only; train-only stats correct | OK |
| Train/Val/Test split | YES | YES | ZIP random (interleaved eras); local chronological | MEDIUM |
| Persistence baseline | YES | YES | last-step persistence | OK |
| LSTM | YES | YES | trained + saved; modest skill at +6 h | OK |
| Inference | YES | YES | local 9×7 only; not wired to ZIP npz | MEDIUM |
| Documentation | YES | PARTIAL | QA count bugs; two pipelines' docs are mutually inconsistent | MEDIUM |

---

## 12. Final Person-4 Readiness

**VERDICT: FORECASTING PARTIALLY READY**

**Explanation:** P4's required data chain exists and is verified genuine — the ERA5-synchronized `master_dataset.csv` (nearest-cell extraction, 12/12 cross-checked against the raw NetCDF), plus the ZIP's 5×7 → 3×3 sequence npz whose feature order `[lat, lon, wind_speed, pressure, sst, wind_u, wind_v]` matches the intended input spec and whose +6/+12/+24 targets are exact. So forecasting can start. However, the working tree also contains the older 9×7 pipeline whose inputs contain **no environmental (ERA5) fields** — inconsistent with the intended feature list; the ZIP sequences were built with **non-causal interpolation** (unflagged fabricated history/targets) and a **random (interleaved) split**; SST is 28% missing with no flag; and a few docs claim completeness that the files don't have. These must be resolved (decide the canonical dataset, cull/cause-mark samples, choose chronological split, document SST policy) before trustworthy results.

### Q&A

1. **Can I start Person-4 forecasting work immediately?** Yes for exploration, data QA, and baseline experiments — but formally decide which one canonical dataset you will commit to before tuning. I recommend the ZIP npz (it matches the intended feature contract).
2. **What files should I use as inputs?**
   - Primary: `PS70-main.zip` → `data/processed/forecasting/{train,val,test}_sequences.npz` + `*_sequences_metadata.csv` (5×7 → 3×3, ERA5 included).
   - Aligned source (for your own sequence rebuilds): `data/metadata/master_dataset.csv` (5,481×15; all ERA5 columns present).
   - Fallback/reference only: local `data/processed/sequences/*.npy` (9×7, chronological, but no SST/u/v).
3. **Do I need to download anything else?** No. All required raw data (IBTrACS, ERA5, and even the original CSV-level Kaggle labels) is already present locally or in the ZIP.
4. **Do I need P1 to change anything?** Only if you choose to hold them to the issues: (a) rebuild sequences with causal interpolation + add a synthetic-origin flag, (b) apply a chronological cyclone split to the npz, (c) fix the QA report completeness figure and add an SST-flag column, (d) confirm the intended handling of the 7.2% edge-clipped rows. None of these block a first experiment.
5. **Is ERA5 actually integrated or do we still need to integrate it?** **Integrated** for the P1 master/ZIP data (nearest-neighbour spatial + temporal extraction, verified). It is **not** integrated into the older local 9×7 sequences (you would have to fuse it yourself if you choose that path).
6. **Are the existing 9×7 sequences safe to use?** Safe from same-cyclone leakage and chronologically split, but **not safe as-is for the intended spec**: input features are `[lat, lon, wind_kmh, pressure, storm_speed, storm_direction, pressure_missing]` — **SST/u/v are missing**, and arrays contain NaNs that phase-3 imputes (mean + indicator). Use only if you accept that narrower feature set, or rebuild from master first.
7. **What should be my FIRST next step?** Extract the ZIP into the project, promote `data/processed/forecasting/*_sequences.npz` + `master_dataset.csv` + ERA5 as the canonical P4 inputs, then write a small read-only sanity script that (i) confirms the 3,076 sequences and feature order, (ii) reports synthetic/interpolated target fractions from `build_datasets.py` semantics, and (iii) re-splits chronologically by `cyclone_id` — before any model training. Keep the local 9×7 pipeline as a documented reference.

*No files were created or altered in this project except this report (`AUDIT_P1_DELIVERY.md`).*