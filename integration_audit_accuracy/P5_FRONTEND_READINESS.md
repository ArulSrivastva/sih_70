# P5 Frontend Readiness Audit

**Status: BUILDABLE_UI / NOT_INTEGRATED (mock-only; no live backend connection and several contract gaps).**

## 1. What exists

- **Working copy** `cyclone-project/cyclone-dashboard` (33 tracked files, Vite + React). **Pristine copy** `SIH26/cyclone-dashboard/cyclone-dashboard` (29 files) — older snapshot, same `USE_MOCK=true` behavior; working copy is a superset.
- `src/api/client.js`: `USE_MOCK = true`, `API_BASE = '/api'`. Functions `fetchDetect/fetchClassify/fetchForecast/fetchAnalyze` all return mock payloads after a synthetic delay.
- `src/api/mockData.js`: `mockAnalyzeResponse` = `{ meta, detection: { detected, …, location:{lat,lon} }, classification: { category, … }, forecast: [{hour, label, lat, lon, windSpeedKmh, pressureHpa, confidence}], landfall, risk: {score, level}, satellite }`.
- `src/App.jsx`: on mount calls `fetchAnalyze()` once; renders `data?.meta/…` with loading/error states.
- Components consume only the return values of `fetch*` (never mockData directly) — a genuinely clean seam for swapping the data source.

## 2. Readiness facts (independent inspection)

- **UI code is coherent and self-consistent** against its own mock contract; renders requirement-relevant panels (detection, classification, forecast, landfall, risk, satellite, baseline comparison).
- **USE_MOCK is still `true` in both copies** → the page today shows demonstration data, not any model output.
- **No deployment/wrapper exists** to give the P5 app a `GET /api/analyze` (or `/api/*`) — pointing the current client at the delivered backend produces 404s and shape mismatches.

## 3. Contract gaps vs delivered backend (detail in API_COMPATIBILITY.md)

| Gap | Detail |
|---|---|
| Path | client calls under `/api/`; phase6 serves `/health`, `/model`, `/forecast`, `/forecast/compare` (no prefix) |
| Endpoints | `analyze`, `detect`, `classify` do not exist; `forecast` is POST-only vs client GET |
| Response shape | client expects `forecast[].{hour,label,lat,lon,windSpeedKmh,pressureHpa,confidence}`; API returns `forecast[].{hours,latitude,longitude,wind_speed_kmh}` |
| Coverage | P5 needs detection + classification + landfall + risk + satellite data; the delivered backend only forecasts tracks |

## Verdicts

| Item | Verdict |
|---|---|
| Frontend builds / renders with mock data | PASS (verifiable by inspection; node_modules removed so no local build was re-run here) |
| Uses live model output today | NO (mock-only) |
| Wireable to delivered phase6 API as-is | NO (needs a `/api/analyze`-style adapter or conforming endpoint) |
| Fulfils the demo requirement end-to-end | NOT YET — pending an integration adapter between `/api/analyze` contract and phase6 `/forecast` (+ new detection/classification surface) |

## p5_metrics.json

Dashboard file census, `USE_MOCK` flags in both copies, API paths invoked, endpoint-path diff against phase6 OpenAPI, and contract field diff.