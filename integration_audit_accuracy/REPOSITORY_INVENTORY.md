# Repository Inventory

**Audit:** Pre-Integration Accuracy + Readiness Audit (read-only)
**Root:** `C:\Users\aruls\Desktop\SIH26\ps70\cyclone-project`
**Date:** 2026-08-30
**Method:** All P1/P2/P3 artifacts were read in-place from `PS70-main.zip` (never extracted into the workspace); P4/P5 artifacts read in place. No file outside `integration_audit_accuracy/` was created/modified/deleted. Environment: Python 3.13.7, numpy 2.3.2, pandas 3.0.0, torch 2.6.0+cu124, torchvision present, scikit-learn 1.7.1, scipy 1.16.1, pytest 9.1.1, fastapi/uvicorn/httpx present. **lightgbm is NOT installed.**

## 1. Delivered artifacts (top level)

| Path | Kind | Notes |
|---|---|---|
| `PS70-main.zip` | Archive (1,231,540,094 B) | Contains the entire P1–P3 sub-system (metadata, processed datasets, models, `src/`). 1,165 zip entries, of which 94 are ERA5 `.nc` reanalysis files (1,103,681,364 B). |
| `p4_forecasting/` | P4+ sub-system (294 files) | phase2 (baselines), phase3 (legacy LSTM), phase4 (experiments EXP001–EXP006 + champion), phase5 (predictor), phase6 (FastAPI), canonical + canonical_chrono npz, reports, tests. |
| `cyclone-dashboard/` | P5 frontend (33 files + node_modules excluded) | React/Vite working copy. `USE_MOCK=true`, no backend wiring. |
| `phase4/`, `results/`, `data/`, `models/`, `src/`, `scripts/`, `notebooks/` | Workspace dirs (mostly empty at root) | `src/` 3 scripts, `data/` 19 files (spreadsheets/CSVs), `models/` 3 files, `results/` 10 files — low-level workspace scratch, **not** part of the delivered P1–P3 archive. |
| `AUDIT_P1_DELIVERY.md`, `DATASET_REPORT.md` | Docs | Delivery/QA notes from P1. |
| `integration_audit/` | Prior audit (completed earlier) | Separate task; not modified by this audit. |
| `integration_audit_accuracy/` | **This audit** | The only directory created by this audit. |

## 2. Delivery scopes (who/build steps)

| Scope | Canonical location | Build script (in `PS70-main.zip`) | Content |
|---|---|---|---|
| P1 data | `data/metadata/master_dataset.csv` (+ splits), `data/processed/forecasting/*sequences*` | `src/data/build_datasets.py`, `src/data/clean_kaggle_intensity.py` | ERA5+IBTrACS master (5,481 rows / 151 cyclones), 70/15/15 cyclone-level splits, three task datasets (A detection / B classification / C forecasting) |
| P2 detection | `models/detection/model_weights.pt`, `src/detection/*` | `build_datasets.py` (Dataset A) | MobileNetV3-small CycloneDetector (presence/pattern/category heads) |
| P3 classification | `models/classification/*` (image_only_model.pt, tabular_multisource_model.pkl, metrics_comparison.json) | `clean_kaggle_intensity.py`, `build_datasets.py` (Dataset B) | ImageOnly ResNet18 + Multisource LightGBM tabular |
| P4 forecasting | `p4_forecasting/canonical_chrono/*`, `p4_forecasting/phase4/*`, phase2/phase3 results | `phase4` training/eval pipeline (EXP001–EXP006 → champion EXP005) | GRU+Huber 16→GRU→9 forecaster on 6-hourly track windows |
| P5 frontend | `cyclone-dashboard/` (working copy) + SIH26 pristine copy | n/a | React dashboard, mock-only API client |

## 3. Counts (independent verification)

- Master dataset: **5,481 rows, 151 cyclones**, 2013-05-09 … 2025-12-02, sub-basins {AS, BB, MM}; 0 exact-duplicate rows; 0 duplicate (cyclone_id, timestamp); splits disjoint: train 105 cyclones / 3,911 rows, val 22 / 752, test 24 / 818 (union = all 151, no intersection).
- Clean IBTrACS: **18,168 rows / 471 cyclones**, 1980–2025, 3 h median cadence; wind/pressure/category missing in ~45% of rows (clean = dedup'd IBTrACS, NOT imputed).
- ERA5 join: **5,481 / 5,481 rows matched** (100%); 94 `.nc` files, 1.1 GB.
- Classification: multisource train 3,039 / val 518 / test 651; image-only Kaggle 133 images (train 93 / val 19 / test 21).
- Detection: 133 rows, 3/3/1 pattern classes; **cyclone_detected=True in 133/133**, mock_bbox identical in 133/133.
- Forecasting sequences (P1, 6 h): train 2,275 / val 378 / test 423 (67/14/16 cyclones). Note: only 97 of 151 master cyclones appear in Dataset C.
- P4 view of the same data: canonical_chrono npz train 2,259 / val 416 / test 401 (3 h cadence); phase4 feature_dataset train 1,212 / val 231 / test 198 (6 h subset). The experiment/test/baseline evaluations were performed on **feature_dataset** (198 test samples / 10 cyclones), which is a different population than the delivered `canonical_chrono/test.npz` (401). See `LEAKAGE_AUDIT.md` (population note) and `P4_FORECASTING_AUDIT.md`.

## 4. Verification artifacts produced

- `p1_metrics.json`, `p2_metrics.json`, `p3_metrics.json`, `p4_metrics.json`, `p5_metrics.json`
- `METRIC_VERIFICATION.csv` (every reported metric vs. independent recomputation)
- `SOURCE_HASHES_BEFORE.json` / `SOURCE_HASHES_AFTER.json` / `SOURCE_IMMUTABILITY_REPORT.json`
- `tests/` (independently runnable pytest suite), `AUDIT_MANIFEST.json`