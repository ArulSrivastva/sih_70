# API / Backend / Frontend Inventory — SIH 2026 PS 26070

## 1. How requests are meant to flow

```
React dashboard (src/api/client.js)
   │  USE_MOCK=true  → mockData.js  (currently the UI runs on mock data ONLY)
   ▼
   fetch(API_BASE + path)        API_BASE = '/api'  (Vite dev server proxy target: NOT CONFIGURED)
   │
   ├─ POST  → /api/analyze        (primary, per future_task/tasks.md + mockData.js)
   └─ GET   → /api/detect, /api/classify, /api/forecast   (fallbacks coded in client.js)
```

## 2. GPS of every backend/API/inference module

| Location | Framework | What it exposes | HTTP server? |
|---|---|---|---|
| `PS70-main/src/detection/inference.py` | PyTorch | `CycloneInference().detect_cyclone(image_path)` → dict | **No** (plain function) |
| `PS70-main/src/classification/inference.py` | torch/sklearn | `classify_cyclone(image_input, environmental_data)` → contract dict | **No** (plain function) |
| `PS70-main/src/data/*.py` | PyTorch/NumPy | dataset loaders | No |
| `p4_forecasting/phase6/` | **FastAPI** | **GET /health, GET /model, POST /forecast, POST /forecast/compare** (uvicorn, port 8000) | **YES** |
| `p4_forecasting/phase5/service/forecasting_service.py` | Python | `ForecastingService.forecast(history)` / `.compare_baselines(history)` | No (service used by phase6) |

`FastAPI|Flask|uvicorn` grep across PS70-main `src/`: **zero matches** → P1/P2/P3 contain NO HTTP API layer.

## 3. Frontend expected endpoints (from actual code — `src/api/client.js`, `mockData.js`)

| Expected endpoint | HTTP method (as coded) | Request JSON | Response JSON (shape) | Current status | Backend present? |
|---|---|---|---|---|---|
| `/api/analyze` | GET via `getJSON` (intended POST) | none sent | `{ meta, detection, classification, forecast[], landfall, risk, historicalTrack[], windHistory[], pressureHistory[], confidenceHistory[], sstHistory[], envWindHistory[], satellite }` | **MOCK ONLY** (USE_MOCK=true) | **NO** |
| `/api/detect` | GET (fetch) | none | `detection` block `{detected, confidence, location{lat,lon}, movementDirection, movementSpeedKmh}` | MOCK ONLY | **NO** (P2 inference exists as a function) |
| `/api/classify` | GET (fetch) | none | `classification` block `{category, scale, windSpeedKmh, pressureHpa, confidence, structuralPattern}` | MOCK ONLY | **NO** (P3 function exists) |
| `/api/forecast` | GET (fetch) | none | `forecast` array `[{hour,label,lat,lon,windSpeedKmh,pressureHpa,confidence}]` | MOCK ONLY | **PARTIAL**: P4 phase6 has `POST /forecast` (no `/api` prefix; different shape: `{hours, latitude, longitude, wind_speed_kmh}` ×3, no pressure/confidence) |

Unified contract reference (`future_task/tasks.md` + `mockData.js`) — `POST /api/analyze` returning `{timestamp, detection{detected,confidence,center,bbox,structural_pattern}, classification{category,wind_speed,pressure,confidence}, forecast[{hours,latitude,longitude,wind_speed}×3], landfall{...}, risk{score,level}}`.

## 4. P4 FastAPI (existing, verified)

| Route | Method | Request | Response | Status |
|---|---|---|---|---|
| `/health` | GET | — | `{status,service,phase,offline,model_ready}` | 200 verified |
| `/model` | GET | — | `{experiment_id,model,loss,hidden_size,layers,...,validation_primary_score}` | 200 verified |
| `/forecast` | POST | 5 obs × `{timestamp,latitude,longitude,wind_speed_kmh,pressure_hpa,sst,wind_u,wind_v}` (6-hourly) | `{status,model,input,forecast[{hours,latitude,longitude,wind_speed_kmh}×3]}` | 200/422 verified |
| `/forecast/compare` | POST | same | model + persistence + movement-vector forecasts | 200/422 verified |

Serving command (from `p4_forecasting/`): `python -X utf8 -m uvicorn phase6.api.app:app --host 127.0.0.1 --port 8000`.

## 5. Gaps / notes for integration (NOT created here)
- No `/api` prefix and no GET `forecast` — frontend fallback expects `GET /api/forecast`, backend exposes `POST /forecast`.
- No unified `/api/analyze` and no `/api/detect` or `/api/classify` endpoints anywhere.
- Frontend currently serves purely from `mockData.js`; wiring the real P2/P3 outputs requires the P2/P3 inference being **invokable with the same inputs the dashboard will send** (a satellite image + env fields) — P2/P3 only accept `image_path` / optional env dict.
- P2/P3 model metrics are weak (category acc 0.33 / 0.38-0.47); integration must decide whether to expose them (architectural decision — NOT this task).

**API NOT PRESENT** for `/api/analyze`, `/api/detect`, `/api/classify`. Partial: `POST /forecast` (P4 phase6).