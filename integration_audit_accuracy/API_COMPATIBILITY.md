# API Compatibility Audit

**Status: BLOCKED_FOR_DIRECT_WIRING** (both sides exist and are individually verified; there is no conforming path between them).

## Method

Read-only: (a) parsed the P5 mock/API contract in `cyclone-dashboard/src/api/{client.js,mockData.js}` and `App.jsx`; (b) built the phase6 FastAPI app in-process with `TestClient` (no server, no writes) and exercised endpoints live; (c) compared OpenAPI surface with the frontend’s calls.

## Live verification of the delivered backend (phase6)

| Check | Result |
|---|---|
| GET /health | 200 `{status:ok, service:cyclone-forecasting, phase:phase6, offline:true, model_ready:true}` |
| GET /model | 200 `{experiment_id:EXP005, model:GRU, loss:Huber, … validation_primary_score:113.074}` |
| POST /forecast (valid 5×6h) | 200 → `forecast:[{hours:6, latitude, longitude, wind_speed_kmh}, …]` (3 rows) |
| POST /forecast/compare | 200 → `{model_forecast, persistence_forecast, movement_vector_forecast}` |
| POST /forecast, 4 observations | 422 |
| POST /forecast, missing field | 422 |
| POST /forecast, lat=95 | 422 |
| GET /openapi.json paths | `['/forecast','/forecast/compare','/health','/model']` |
| GET /api/analyze (frontend target) | **404** |

The backend is a real, working, offline FastAPI layer over the EXP005 champion. That layer is technically sound.

## Frontend calls (when switched off mock)

| Call (client.js) | Backend reality |
|---|---|
| `GET /api/analyze` | nothing at this path (404, verified) |
| `GET /api/detect` | no detection endpoint exists |
| `GET /api/classify` | no classification endpoint exists |
| `GET /api/forecast` | `POST /forecast` exists (wrong method + path prefix) |

## Shape mismatches on the only overlapping concept (forecast)

| Frontend mock item | Backend ForecastItem |
|---|---|
| `hour` | `hours` |
| `lat`, `lon` | `latitude`, `longitude` |
| `windSpeedKmh` | `wind_speed_kmh` |
| `label` (+6h/+12h/+24h) | absent |
| `pressureHpa` | absent |
| `confidence` | absent |

The frontend would also render `landfall`, `risk`, `satellite`, detection and classification blocks from fields the backend never returns; and in the client’s own fallback path (`fetchAnalyze` catch) `landfall` and `risk` are hard-coded to `null`.

## Verdicts

| Item | Verdict |
|---|---|
| Delivered backend (phase6) itself works and is verified live | YES |
| Delivered frontend logic is internally consistent mock | YES |
| Direct frontend↔backend wiring today | **IMPOSSIBLE without change** (paths, methods, and schemas all differ) |
| Minimal fix (recommended, not performed — audit is read-only) | Add an `/api/analyze` facade; forward to phase6 `/forecast(+compare)`; map fields; add trivial detection/classification passthrough or document them as mock-only |