# Directory Inventory — SIH 2026 PS 26070 Integrated Workspace

Audit date: 2026-08-30 (local) · Workspace root: `C:\Users\aruls\Desktop\SIH26\ps70\cyclone-project`

Legend: **P1** Person 1 (data) · **P2** Person 2 (detection) · **P3** Person 3 (classification) · **P4** Person 4 (forecasting) · **P5** Person 5 (frontend).

Top-level layout:

```
C:\Users\aruls\Desktop\SIH26\ps70\cyclone-project\     <- integrated workspace root (NO git repo)
├── PS70-main.zip           1,231,540,094 B  (30-08-2026)  STILL ZIPPED — not extracted
├── PS70-main\              (INSIDE the zip: 1,207 entries = 42 dirs + 1,165 files)
├── cyclone-dashboard\       P5 frontend WORKING copy (Vite/React) + node_modules + dist
├── p4_forecasting\          P4 FINAL package (phases 1-6, 294 files) incl. _source_p1 snapshot
├── data\  models\  src\  scripts\  results\  notebooks\  phase4\   (pre-existing P4-side workspace)
├── AUDIT_P1_DELIVERY.md  DATASET_REPORT.md
└── integration_audit\       (THIS audit — the only new files)

C:\Users\aruls\Desktop\SIH26\cyclone-dashboard\        <- P5 frontend DOWNLOAD original (extract of cyclone-dashboard.zip)
C:\Users\aruls\Desktop\SIH26\cyclone-dashboard.zip     <- P5 source zip (download, 62 KB)
```

Git status: **no `.git`** anywhere (workspace, both dashboard copies, zip contents). Branches/remotes: N/A.

---

## P1 DATA (inside `PS70-main.zip` unless noted)

| Path (in zip) | Type | Ext | Size | Modified* | Purpose | Owner | Verdict | Reason |
|---|---|---|---|---|---|---|---|---|
| `data/raw/ibtracs/ibtracs_NI_raw.csv` | F | csv | 27,875,881 | zip | Raw NOAA IBTrACS North Indian basin v04r01 | P1 | KEEP | canonical raw source |
| `data/raw/era5/era5_YYYY_MM.nc` × 94 | F | nc | 1,103,681,364 total | zip | Monthly ERA5 reanalysis NetCDF, 2013-2025, 0.25°, bbox 30N/5S/50E/105E, 3-hourly | P1 | KEEP | canonical environmental source |
| `data/raw/insat/` (143+136+140 imgs + 1 csv) | F | jpg/jpeg/png/csv | 47,459,738 | zip | INSAT-3D/3DR reference+IR+raw cyclone crops, `insat_3d_ds - Sheet.csv` labels | P1 | KEEP | canonical satellite imagery |
| `data/raw/insat_kaggle/` (identical tree) | F | jpg/jpeg/png/csv | 47,459,738 | zip | **Byte-identical duplicate** of `data/raw/insat/` (420/420 files, 0 diffs) | P1 | DUPLICATE | full tree duplicate — see DUPLICATE_REPORT.md |
| `data/metadata/ibtracs_clean.csv` | F | csv | 1,487,310 | zip | Cleaned IBTrACS NIO, 18,168 rows, 1980-10 → 2025-12, 471 cyclones | P1 | KEEP | clean ground truth |
| `data/metadata/ibtracs_with_era5.csv` | F | csv | 709,456 | zip | 5,481 obs joined with ERA5 | P1 | KEEP | aligned dataset |
| `data/metadata/master_dataset.csv` | F | csv | 738,948 | zip | Aligned master, 5,481 rows, 2013→2025, 151 cyclones, 3 basins | P1 | KEEP | canonical master |
| `data/metadata/train.csv` | F | csv | 527,509 | zip | 3,911 rows, cyclone-level split | P1 | KEEP | train manifest |
| `data/metadata/validation.csv` | F | csv | 100,863 | zip | 752 rows | P1 | KEEP | val manifest |
| `data/metadata/test.csv` | F | csv | 110,840 | zip | 818 rows | P1 | KEEP | test manifest |
| `data/metadata/*_cyclones.csv` ×3 | F | csv | 347-1,481 | zip | Cyclone-ID partition indexes | P1 | KEEP | split provenance |
| `data/metadata/mosdac_*.csv` ×3 | F | csv | 1,392-3,697 | zip | MOSDAC satellite acquisition priority lists | P1 | KEEP | acquisition plan |
| `data/processed/forecasting/*sequences.npz` ×3 | F | npz | 13,677-69,267 | zip | Dataset C: X (N,5,7) / Y (N,3,3); 2275/378/423 seqs | P1 | KEEP | forecasting base arrays |
| `data/processed/forecasting/*sequences_metadata.csv` ×3 | F | csv | 52,265-306,770 | zip | Sequence row metadata | P1 | KEEP | provenance |
| `data/processed/forecasting/README.md` | F | md | 2,283 | zip | Dataset C spec | P1 | KEEP | docs |
| `data/processed/master_dataset.csv` | F | csv | 738,948 | zip | Mirror of metadata/master_dataset.csv | P1 | DUPLICATE-BY-VALUE | same bytes as `metadata/master_dataset.csv` — flagged, not auto-delete |
| `data/raw/ibtracs/` dir | D | — | — | zip | — | P1 | KEEP | — |
| `data/qa_reports/QA_REPORT.md` + figures ×4 png | F | md/png | 1,325-601,835 | zip | QA validation report | P1 | KEEP | QA evidence |

`*` Files inside the zip carry the archive's embedded timestamps; the archive itself was downloaded 30-08-2026. Per-file mtimes are preserved in the archive but not duplicated here for brevity.

## P1 CODE

| Path (in zip) | Type | Ext | Size | Purpose | Owner | Verdict | Reason |
|---|---|---|---|---|---|---|---|
| `build_datasets.py` | F | py | 13,662 | Master dataset builder + sliding-window generator | P1 | KEEP | pipeline entry point |
| `clean_kaggle_intensity.py` | F | py | — | INSAT IR processor + metadata joiner | P1 | KEEP | — |
| `download_era5.py` | F | py | 4,374 | CDS API ERA5 downloader | P1 | KEEP | — |
| `extract_era5_at_points.py` | F | py | 5,722 | Nearest-neighbour ERA5 extractor | P1 | KEEP | — |
| `get_ibtracs.py` | F | py | 9,006 | IBTrACS downloader/filter/IMD converter | P1 | KEEP | — |
| `check_coverage.py`, `check_missing_files.py`, `find_near_matches.py`, `prioritize_gap.py`, `split_coverage_gap.py` | F | py | 1-15,000 | Coverage/QA utilities | P1 | KEEP | — |
| `src/data/` (5 modules + `__init__.py`) | F | py | 1,713-4,405 | PyTorch/NumPy dataset loaders (classification, detection, forecasting, preprocess_satellite, dataloader_example) | P1 | KEEP | shared loaders |
| `src/data/__pycache__/` + `src/__pycache__/` (13 .pyc) | F | pyc | 134-7,497 | Python 3.12/3.14 bytecode caches bundled in the archive | P1(shipping) | CACHE (inside zip) | generated; purge on extraction, NOT deleted here |
| `requirements.txt`, `README.md`, `.gitattributes` | F | txt/md | — | Env + repo spec | P1 | KEEP | — |

## P2 DATA

| Path | Type | Ext | Size | Purpose | Owner | Verdict |
|---|---|---|---|---|---|---|
| `data/processed/detection/train/val/test_detection.csv` | F | csv | 2,818-13,302 | Dataset A manifests: 93/19/21 rows (133 total) | P1→P2 | KEEP |
| `data/processed/detection/detection_all.csv` | F | csv | 19,007 | All 133 rows | P1→P2 | KEEP |
| `data/processed/detection/README.md` | F | md | 1,055 | schema doc | P1 | KEEP |
| (images referenced) `data/processed/classification/image_only_kaggle/images/` | F | jpg/jpeg | 133 | shared image set | P1→P2/P3 | KEEP |

P2 input = 133 multi-source rows (image manifests); P2 output = detection CSVs (presence + structural_pattern + category + mock_bbox).

## P2 CODE

| Path | Ext | Size | Purpose | Owner | Verdict |
|---|---|---|---|---|---|
| `src/detection/detector.py` | py | 1,360 | MobileNetV3-small backbone + presence/pattern/category heads | P2 | KEEP |
| `src/detection/inference.py` | py | 3,298 | `CycloneInference.detect_cyclone(image_path)` loader/predictor | P2 | KEEP |
| `src/detection/train.py`, `evaluate.py` | py | 3,901/6,269 | training + evaluation | P2 | KEEP |
| `metrics/detection_metrics.json` | json | 346 | pattern acc 0.714, category acc 0.333 | P2 | KEEP |

## P3 DATA

| Path | Ext | Size | Purpose | Owner | Verdict |
|---|---|---|---|---|---|
| `data/processed/classification/multisource_{train,val,test}.csv` | csv | 73,178-426,068 | Dataset B tabular: 3,039/518/651 (4,208) rows, 6 features | P1→P3 | KEEP |
| `data/processed/classification/image_only_kaggle/{train,val,test,labels}.csv` | csv | 790-5,148 | image labels (93/19/21) | P1→P3 | KEEP |
| `data/processed/classification/image_only_kaggle/images/*` | jpg/jpeg | 133 (~5 MB) | INSAT IR crops (256×256) | P1→P3 | KEEP |

## P3 CODE + MODELS

| Path | Ext | Size | Purpose | Owner | Verdict |
|---|---|---|---|---|---|
| `src/classification/classifier.py` | py | 7,707 | ResNet18/MobileNetV3 image model, LightGBM tabular, fusion model | P3 | KEEP |
| `src/classification/inference.py` | py | 8,244 | `classify_cyclone(...)` → unified contract | P3 | KEEP |
| `src/classification/train.py`, `evaluate.py` | py | 5,743/8,244 | training/eval | P3 | KEEP |
| `src/classification/defs/README.md` | md | 2,994 | subsystem doc | P3 | KEEP |
| `models/classification/image_only_model.pt` | pt | 45,186,485 | ResNet18 dual-head CNN | P3 | KEEP (model) |
| `models/classification/tabular_multisource_model.pkl` | pkl | 4,816,632 | LightGBM tabular | P3 | KEEP (model) |
| `models/classification/confusion_matrix.png`, `metrics_comparison.json` | png/json | 243,059/661 | eval artifacts | P3 | KEEP |

## P4 DATA (workspace `p4_forecasting/`)

| Path | Ext | Size | Purpose | Owner | Verdict |
|---|---|---|---|---|---|
| `canonical/{train,val,test}.npz` + csvs | npz/csv | ~1.2 MB | (5,7)→(3,3); 2275/378/423 — value-identical to P1 Dataset C | P4 | KEEP (reference) |
| `canonical_chrono/{train,val,test}.npz` + csvs + split_manifest.csv | npz/csv | ~1.1 MB | chronological split 2259/416/401 | P4 | KEEP |
| `phase2/results/canonical_chronological_clean/*` | npz/csv | ~1.5 MB | clean chrono split 1212/231/198 | P4 | KEEP |
| `phase4/results/feature_dataset/*` | npz/csv | ~1 MB | FINAL engineered features X (N,5,16), Y (N,3,3) | P4 | KEEP |
| `phase4/results/normalization_stats.json` | json | 2,508 | train-only Z-score stats (authoritative 16-feature order) | P4 | KEEP |
| (older) `data/processed/sequences/*.npy` 12 files (workspace `data\processed\sequences`) | npy | 9-step legacy 9×7 arrays 1191/7174/1563 | early Phase-2 experiments (superseded) | P4 | KEEP (don't delete datasets) |
| `_source_p1/PS70-main/` 23 files | mixed | 4,234,627 | P1 repo reference snapshot (older build; 4 build scripts + old metadata + Dataset C) | P4 | KEEP (immutability reference) |

## P4 CODE
`p4_forecasting/phase1..phase6`, `src/forecasting`, `scripts/phase*.py`, `models/lstm_forecaster.pt` etc. — full P4 pipeline; FINAL reference is phase4 (EXP005) + phase5 + phase6. See API_FRONTEND_INVENTORY.md and INTEGRATION_MAP.md. All KEEP. Note `models/lstm_forecaster.pt` and `phase3/checkpoints/best_lstm.pt` are Phase-3 historical checkpoints (KEEP — not deletion candidates).

## P5 FRONTEND

Working copy `cyclone-project/cyclone-dashboard/` (Vite 8 / React 19 / Tailwind 4 / Leaflet / Recharts / oxlint):
- `src/` 24 files: `App.jsx`, `main.jsx`, `index.css`, `api/{client.js,mockData.js}`, `lib/format.js`, `assets/{hero.png,vite.svg}`, `components/` (CardShell, detection-classification-forecast cards, Header, LandfallPanel, LiveCycloneStatus, ModelIntelligence, PipelineStatus, BaselineComparison, RiskIndicator, SatelliteViewer), `components/charts/{ChartsPanel,MiniLineChart}.jsx`, `components/map/{MapView.jsx,uncertaintyCone.js}`.
- `public/` 2 files (favicon.svg, icons.svg). `index.html`, `package.json`, `package-lock.json`, `vite.config.js`, `README.md`, `.oxlintrc.json`, `.gitignore`.
- Generated: `node_modules/` (8,045 files, 114,612,029 B), `dist/` (5 files, 830,806 B).
- Empty leftover dir: `src/components/satellite/` (no content, no references).

Original download `C:\Users\aruls\Desktop\SIH26\cyclone-dashboard\cyclone-dashboard\` — same project WITHOUT the 4 extra components and WITHOUT dist; identical `package.json` (hash same); `vite.config.js` differs (no fs.allow); node_modules identical bytes to working copy. Source subset of the working copy. See DUPLICATE_REPORT.md.

## SHARED / INTEGRATION
- P5 `src/api/client.js` + `mockData.js` define the `/api/analyze` contract (primary) and `/api/detect` `/api/classify` `/api/forecast` fallback endpoints. No backend implements them yet.
- P4 `phase6/` FastAPI implements `/health`, `/model`, `/forecast`, `/forecast/compare` (no `/api` prefix). See API_FRONTEND_INVENTORY.md.

## DOCUMENTATION / CONFIG / TESTS / BUILD OUTPUT / CACHE / TEMP / UNKNOWN

| Category | Items | Verdict |
|---|---|---|
| DOCUMENTATION | `PS70-main/README.md`, `docs/ERA5_ALIGNMENT.md`, `future_task/tasks*.md` ×4, `data/*/README.md`, `AUDIT_P1_DELIVERY.md`, `DATASET_REPORT.md`, p4 `reports/`, `canonical/P4_DATA_CONTRACT.md`, `audit/*.md` | KEEP |
| CONFIGURATION | `requirements.txt`, `package.json`, `package-lock.json`, `vite.config.js`, `.oxlintrc.json`, `.gitignore`, phase4 configs + experiment registry, `models/*.json`, `src/detection/placeholder.txt`, `phase6/config.py` | KEEP |
| TESTS | p4 phases 2-6 test suites (pytest) | KEEP |
| BUILD OUTPUT | `cyclone-dashboard/dist/` (5 files, 830,806 B) | DELETE_CANDIDATE (reproducible via `npm run build`) |
| CACHE | `scripts/__pycache__/` (1 pyc), inside-zip `src/**/__pycache__` (15 pyc), `node_modules/` (×2, reproducible, one a full duplicate) | DELETE_CANDIDATE (reproducible) |
| TEMP | `.vite-temp` inside node_modules (empty) | DELETE_CANDIDATE |
| DUPLICATES | `raw/insat` = `raw/insat_kaggle`; `data/processed/master_dataset.csv` = `data/metadata/master_dataset.csv`; P5 node_modules ×2; `_source_p1` vs zip metadata (mostly different versions, only `ibtracs_clean.csv` + 4 scripts identical) | see DUPLICATE_REPORT |
| UNKNOWN / PLACEHOLDER | Empty dirs: `notebooks/`, `phase4/audit_post_completion/`, `data/raw/era5/`, `src/components/satellite/` | satellite/ DELETE (empty, unreferenced); others KEEP_UNCERTAIN (inert placeholders) |
| P4 `logs/` run+runners logs (6 run_*.log + p1_source_hashes.json) | generated run evidence | KEEP_UNCERTAIN (evidence trail, tiny) |

Counts: workspace (cyclone-project) = 8,420 files / 1,391,670,542 B; SIH26 dashboard = 8,074 files / 114,776,017 B; PS70-main.zip = 1,207 entries / 1,294,258,377 B uncompressed.