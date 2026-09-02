# Integration Map — SIH 2026 PS 26070

High-level architecture of the target system (SIH 2026 PS 26070, Tropical Cyclone AI/ML):

```
┌─────────────────── P5 FRONTEND (cyclone-dashboard, Vite/React/Tailwind/Leaflet) ───────────────────┐
│  src/api/client.js  USE_MOCK=true  →  mockData.js  (UI runs on mock today)                          │
│  expects: POST /api/analyze                                   (primary, per tasks.md)                │
│            GET  /api/detect | classify | forecast             (fallbacks coded)                     │
└───────────────┬──────────────────────────────────────────────────────────────────────────────────────┘
                │ HTTP calls (today: /api/* has NO backend)
                ▼
       ┌────────────────  P4 BACKEND (phase6 FastAPI, running 127.0.0.1:8000)  ────────────────┐
       │  GET /health · GET /model · POST /forecast · POST /forecast/compare                     │
       │  pipeline: parse_history → engineer_history (5×7 → 5×16) → Normalizer → GRU EXP005      │
       │         → denormalize → (3,3) lat/lon/wind @ +6/+12/+24h                                │
       └──────▲───────────────────────────▲───────────────────────────────────────────────────────┘
              │  (used directly)          │  (NOT exposed as endpoints yet — no /api/detect, no /api/classify)
┌─────────────┴──────────┐       ┌─────────┴──────────────┐
│  P2 DETECTION          │       │  P3 CLASSIFICATION     │
│  src/detection/        │       │  src/classification/   │
│  MobileNetV3 (224×224) │       │  ResNet18 (256×256) +  │
│  → presence/pattern/   │       │  LightGBM (6 features) │
│    category dict       │       │  → IMD category + wind │
│  input: satellite crop │       │    + pressure dict     │
└──────────┬─────────────┘       └──────────┬──────────────┘
           │  images (133 kaggle set)        │  features: ERA5 + IBTrACS
───────────┴───────────────────────────────┴──────────────────
   P1 DATA (PS70-main.zip — 1.16k files, STILL ZIPPED)
   IBTrACS: raw→clean(18,168)→era5-joined→master(5,481, 151 cyclones)
   ERA5  : 94 × NetCDF 0.25° (sst/mslp/u10/v10), 2013–2025
   INSAT : 419 images (raw trees ×2 duplicate) → 133 kaggle crops (256×256)
   Datasets A (detection 133) / B (classification 4,208 tabular + 133 img) / C (forecast 5×7→3×3)
```

## Component → source tree map

| Component | Source root | Status in workspace | HTTP | Integrated? |
|---|---|---|---|---|
| Data foundation (P1) | `PS70-main.zip` `data/` + scripts + `src/data/` | zipped (unpack: pending) | none | blocked by unzip |
| Detection (P2) | `data/processed/detection/*` + `src/detection/` + `models/detection/model_weights.pt` | in zip | none | function-only |
| Classification (P3) | `processed/classification/*` + `src/classification/` + `models/classification/*` | in zip | none | function-only |
| Forecasting (P4) | `p4_forecasting/` phase4(EXP005)→phase5→phase6 API | extracted + verified | FastAPI `:8000` | API works standalone |
| Dashboard (P5) | `cyclone-dashboard/` (working copy) + `SIH26/…` (pristine original) | extracted + running | dev server | mock-only |

## Endpoint reconciliation (what a bridge API must expose; NOT built in this audit)

| Frontend needs | Exists? | P4 phase6 | Notes |
|---|---|---|---|
| `POST /api/analyze` unified | **NO** | — | future_task/tasks.md target; composite of detect+classify+forecast+landfall+risk+history+satellite |
| `GET /api/detect` | NO | — | P2 infer() reusable; needs a backend wrapper + image upload/lookup |
| `GET /api/classify` | NO | — | P3 infer() reusable; needs 6-field env input |
| `GET /api/forecast` | only `POST /forecast` | YES | add `/api` prefix + GET + pressure/confidence fields to match frontend |
| `/api/analyze.forecast[]` shape | differs | — | frontend wants {hour,label,lat,lon,windSpeedKmh,pressureHpa,confidence}; P4 returns {hours,latitude,longitude,wind_speed_kmh}×3 |

## Integration blockers / follow-ups (DECIDED BY THIS AUDIT — recorded, not fixed)

1. **P1/P2/P3 still zipped** — `PS70-main.zip` must be extracted into `cyclone-project/` (next step, with `__pycache__`+duplicate-D1/D2 purged at repack time). Audit only — no extraction performed.
2. **No unified `/api/analyze` backend** and **no `/api/detect` or `/api/classify`** — those live as Python functions only. P2/P3 must be wrapped (serve image + env fields).
3. **Contract mismatch forecast**: P4 outputs `{hours, latitude, longitude, wind_speed_kmh}`; frontend expects pressure+confidence + camelCase hour/label + `minutes`-style labels (48h/72h/96h...). Bridge must map.
4. **Frontend is mock-only** (`USE_MOCK=true`) — swap to real API after bridge exists.
5. **Model quality flags** recorded for review: P2 category acc 0.333; P3 multi-source 47% vs image 38.1%, image wind MAE 109.8 km/h. Dashboard must decide how/whether to surface these.
6. **ERA5 `sst` coverage inconsistency inside P1 repo** (QA_REPORT lists 1,538 missing sst vs README's "100% completeness") — flag for P1 clarification, not this audit.
7. **`_source_p1` is an older P1 snapshot** inside p4_forecasting — after real P1 extraction, re-baseline P4's immutability reference to the fresh archive.

## Data flows (verified)
- P1 → A/B/C datasets → P2 (images+manifests), P3 (tabular+images), P4 (sequences 5×7 / feature 5×16).
- P4 canonical npz == P1 Dataset C (value-equal); P4 feature_dataset = the actual GRU input (5×16).
- P2/P3 models read `image_only_kaggle/images` (133 crops); the 419 raw INSAT images are the parent archive (still zipped).
- P5 mock mirrors the tasks.md unified contract → the bridge API's single source of truth is `future_task/tasks.md` + `mockData.js`.

## Guardrail summary
- Sources are immutable: verified 357/357 bytes unchanged across the cleanup (SOURCE_HASHES_BEFORE/AFTER.json).
- Deleted only reproducible/cache artifacts (C1–C4, 115.5 MB; see CLEANUP_CANDIDATES.md).
- The only new files this deliverable introduces live under `integration_audit/`.