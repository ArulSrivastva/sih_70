# Integration Architecture — SIH 2026 PS70/260 (VARTHA dashboard ↔ integration_api ↔ P2/P3/P4/P5)

This document describes the **end-to-end integration** between the React dashboard
(`cyclone-dashboard/`), the FastAPI integration layer (`p4_forecasting/integration_api/`),
and the audited learning components P2 (satellite detection), P3 (intensity
classification), P4/P5 (track+wind forecasting), plus the deterministic
landfall/risk heuristics.

It is written from a read of the actual serving code
(`integration_api/*`, `phase6/*`, `phase5/*`, `cyclone-dashboard/src/*`) and a
live `/api/analyze` trace (see `results/e2e_trace.json`). It records every
field, transform, assumption and failure mode that links the layers together so
that future changes cannot silently break the contract.

---

## 1. Two-process deployment model

```
Browser (Vite dev :5173)                 FastAPI (uvicorn :8000)
┌──────────────────────────────┐         ┌──────────────────────────────────────────────┐
│ cyclone-dashboard/src/App.jsx │  POST /api/analyze   │ integration_api/app.py           │
│   fetchAnalyze()  ────────────────────►              │   phase6 create_app + /api router│
│   (client.js, USE_MOCK=false)│  JSON {history[5],    │   app.state.analyzer             │
└──────────────────────────────┘ meta}                  │      └ CycloneAnalyzer          │
        ▲                             │                 │           ├ p2_detector          │
        │  response                    │ 200 JSON       │           ├ p3_classifier        │
        │  /api/analyze contract       │                │           ├ forecasting_adapter │
        └──────────────────────────────┘                │              └ ForecastingService│
                                                        │                 └ CyclonePredictor│
  vite.config.js  /api -> 127.0.0.1:8000 (proxy)        │                                   │
                                                        └──────────────────────────────────┘
```

* The dashboard is served by Vite; its dev server **proxies** every `/api/*`
  request to `http://127.0.0.1:8000` (`cyclone-dashboard/vite.config.js`, target
  `127.0.0.1:8000`, `changeOrigin: true`).
* The backend is the phase-6 app augmented by the integration router
  (`integration_api/app.py:create_app` reuses `phase6.api.app.create_app`, adds
  `app.state.analyzer`, and mounts `/api`).
* Both processes are local and **fully offline** for the scientific path; the only
  online dependency is the optional Leaflet/OpenStreetMap basemap in the browser
  (see §8).

---

## 2. Endpoints exposed

| Method | Path            | Handler                          | Purpose |
|--------|-----------------|----------------------------------|---------|
| GET    | `/api/health`   | `routes.health`                  | service + ML readiness |
| POST   | `/api/analyze`  | `routes.analyze`                 | full dashboard payload |
| POST   | `/api/detect`   | `routes.detect`                  | detection block only |
| POST   | `/api/classify` | `routes.classify`                | classification block only |
| POST   | `/api/forecast` | `routes.forecast`                | forecast block only |

Phase-6 endpoints (`/health`, `/model`, `/forecast`, `/forecast/compare`, `/docs`)
remain mounted from the base app (composable, not shadowed).

---

## 3. Request contract (`POST /api/{analyze,detect,classify,forecast}`)

The request schema **is the audited phase-6 `ForecastRequest`** plus an optional
`meta` block (`integration_api/schemas.py:AnalyzeRequest`). Validation is
centralised in `phase6/schemas/requests.py` and happens before any inference.

```jsonc
{
  "history": [                                    // EXACTLY 5 observations
    {
      "timestamp": "2026-08-25T00:00:00Z",        // ISO-8601, strictly 6h apart
      "latitude": 13.80,                          // [0? no] [-90, 90]
      "longitude": 80.10,                         // [0, 360) degrees-east
      "wind_speed_kmh": 105,                      // [0, 400]
      "pressure_hpa": 986,                        // [850, 1100]
      "sst": 28.2,                                // [-5, 45]
      "wind_u": 10.0, "wind_v": 7.0               // any finite float
    }, /* ... t-18h, t-12h, t-6h, t ... */
  ],
  "meta": {                                       // OPTIONAL identity block
    "systemId": "BOB07",
    "systemName": "Cyclonic Storm ANIKA",
    "basin": "Bay of Bengal",
    "lastPass": "2026-08-26T05:30:00Z",
    "source": "…"                                 // optional free-form
  }
}
```

Validation rules that must never be relaxed:
* exactly 5 observations (`INVALID_HISTORY_LENGTH`),
* strictly-monsoon 6-hour spacing, strictly increasing (`INVALID_HISTORY_SPACING`,
  `NON_MONOTONIC_HISTORY`),
* physical range checks (`INVALID_LATITUDE/LONGITUDE/WIND`), pressure/sst guards
  (`INVALID_REQUEST`),
* finite values (`NON_FINITE_VALUE`), no extra fields (`extra="forbid"`).

---

## 4. Data flow through the analyzer

`CycloneAnalyzer.analyze` (with the reference frame) drives every block:

### 4.1 Common inputs
* `history` → validated `AnalyzeRequest`. Last observation `t` is the "Now" anchor;
  `history[-1]` feeds the displayed wind/pressure, movement vector, and is the seed
  for the forecast.
* reference frame → `zip_store.reference_image()` selects a deterministic
  INSAT-3D IR frame from `PS70-main.zip`
  (`…/CYCLONE_DATASET_FINAL/45(1).jpg`), returned as in-memory PIL RGB.

### 4.2 Detection (P2) — `p2_detector.py`
* Architecture replica of `PS70-main/src/detection/detector.py`
  (`mobilenet_v3_small(weights=None)`, presence/pattern/category heads).
* Weights loaded from zip in memory, `weights_only=True`.
* `detect(image)` → Resize(224,224)+ToTensor → forward → sigmoid presence≥0.5,
  argmax pattern/category. Returns `detected`, `confidence` (0–1 → ×100 int),
  `structural_pattern`.
* Dashboard `location` is **not** from the model — it is the last validated
  observation's lat/lon (the P2 model has no localizer).

### 4.3 Classification (P3) — `p3_classifier.py`
* Image-only ResNet18 replica (`resnet18(weights=None)`), weights from zip.
* `classify_image(image)` → Resize(256,256)/255 → forward → softmax category +
  confidence; IMD category from `config.IMD_CLASSES`.
* **The P3 wind regressor is degenerate and is never displayed.** The shown
  `windSpeedKmh`/`pressureHpa` are the *latest observed* history values
  (`classify` returns `last.wind_speed_kmh` / `last.pressure_hpa`).
* `tabular_status()` reports the `.pkl` (LightGBM) as `available:false / NOT_RUN`
  because `lightgbm` is not installed — never silently substituted.

### 4.4 Forecast (P4/P5) — `phase6/integration/forecasting_adapter.py`
* `ForecastingAdapter.forecast(request)` → `request.phase5_document()` (dict
  timeline) → `ForecastingService.forecast` → `CyclonePredictor.predict_features`
  returns a **(3,3)** array = rows `+6h/+12h/+24h`, cols `[lat, lon, wind_kmh]`
  (de-normalised, lon wrapped to `[0,360)`, wind clipped ≥0).
* `analyzer._forecast_items` maps each row to the dashboard shape:
  `hour, label, lat, lon, windSpeedKmh, pressureHpa=null, confidence=null`
  (no calibrated pressure/uncertainty — honestly null).
* Champion identity is read from the audited Phase-4 artifacts
  (`champion_model.json`, `config.json`, `normalization_stats.json`), never hard-coded.

### 4.5 Heuristics (not ML) — `analyzer.py`
* **Movement**: `_haversine_km` + `_bearing_deg` between the last two observations
  → `movementDirection` (16-point compass), `movementSpeedKmh` (= dist/6h).
* **Landfall**: for each forecast point find nearest of the embedded
  `NIO_COAST_POINTS`; if `dist ≤ 200 km` → `estimated:true` with coast point,
  `estimated_time = t + best.hours`, `predictedWindKmh` (forecast wind), and
  `distanceToLandKm`. Else `estimated:false` (still returns distance).
* **Risk**: `score = clamp(5 + (maxWind − 30)·0.6, 5, 99)`; if
  `landfall.distanceToLandKm < 300` add `(300 − d)/20`; label HIGH/MODERATE/LOW.
* **History views**: `windHistory/pressureHistory/sstHistory/envWindHistory` replay
  the observed history with labels `-24h…Now`; `envWind = hypot(wind_u, wind_v)`;
  `confidenceHistory` is a **placeholder constant** = current P2 detection
  confidence (no calibrated per-step confidence exists).
* **Satellite block**: `label` + `source` (frame path); `boundingBox` is **null**
  (no audited localizer in the bundle).

### 4.6 Provenance block
`analyze` always emits `provenance.{pipeline, sources, notes[], reference_image,
tabular}` — a machine+human readable record of exactly which outputs are model
outputs vs observations vs heuristics, and which model is NOT loaded.

---

## 5. Response contract (`/api/analyze` top-level)

```
status: "success"
meta:            {systemId, systemName, basin, lastPass, source}
detection:       {detected, confidence, location:{lat,lon}, movementDirection, movementSpeedKmh}
classification:  {category, scale, windSpeedKmh, pressureHpa, confidence, structuralPattern}
forecast[3]:     {hour, label, lat, lon, windSpeedKmh, pressureHpa, confidence}
landfall:        {estimated, latitude, longitude, estimated_time, predictedWindKmh, distanceToLandKm}
risk:            {score, level}
historicalTrack[5]:   {lat, lon, timestamp}
wind/pressure/confidence/sst/envWind History[5]: {t, value}
satellite:       {label, boundingBox|null, source}
provenance:      {pipeline, sources{}, notes[], reference_image, tabular{}}
```

This mirrors `cyclone-dashboard/src/api/mockData.js` field-for-field except where
honesty requires: the backend supplies `null` for `forecast[].pressureHpa/confidence`
and `satellite.boundingBox`, and it adds the `provenance` block (which the frontend
now renders via `ProvenancePanel`).

---

## 6. Formats & transforms (single source of truth)

| Item             | Input contract              | Transform                                | Output contract |
|------------------|-----------------------------|------------------------------------------|-----------------|
| lon              | [0, 360)                       | model wraps `lon % 360`                    | [0, 360)        |
| forecast weights | raw features (5,16)           | train-only normaliser + GRU + denormalise  | (3,3) phys      |
| wind             | km/h [0,400]                  | clip ≥0 after denormalise                  | float ≥ 0       |
| P3 image         | 256×256 RGB                   | /255, permute → CHW                          | category+confid |
| history labels   | timestamps                    | `-24h -18h -12h -6h Now`                  | label            |

---

## 7. Failure modes & guards

| Failure                              | Guard / response                                         |
|--------------------------------------|----------------------------------------------------------|
| Malformed/insufficient history       | pydantic validation → `422` with coded message            |
| Out-of-range / NaN / Inf values      | `422` `INVALID_*`/`NON_FINITE_VALUE`                      |
| Missing artifact bundle              | `ModelNotReady` → `health.ml` flag / `503`                |
| Bad/corrupt zip                      | `ModelNotReady` (never a traceback)                       |
| Zip member not found                 | `ModelNotReady` → `503`                                   |
| Forecast service internal error      | `InfrastructureError` → `500 INFERENCE_ERROR` (no stack)  |
| Oversized request body               | body-size guard → `413 INVALID_REQUEST`                   |
| Network attempt during inference     | none needed — code is offline (verified, see `offline_proof.json`) |

---

## 8. Offline vs online dependency split

* **Offline (core, required):** zip read, P2/P3 torch inference (`weights=None` →
  no ImageNet download), P4/P5 GRU inference, all heuristics, all validation.
* **Online (optional, display-only):** OpenStreetMap tile basemap in
  `MapView` (`TileLayer`). If offline, the map shows no basemap but every model
  block still renders. Disclosed in the map caption.

---

## 9. Assumptions (documented, not silently baked in)

1. Reference frame is a **deterministic sample**, never a live feed (disclosed in
   `meta.source`, `satellite.source`, `provenance`).
2. Displayed wind/pressure are **observations**, not P3 regression output.
3. `forecast[].pressureHpa/confidence` are `null` because no calibrated outputs exist.
4. `confidenceHistory` is a documented placeholder constant.
5. Landfall/risk are **server-side heuristics**, never labelled ML.
6. `satellite.boundingBox` is `null` (no audited localizer).
7. Tabular (LightGBM) classifier is `NOT_RUN` in this environment and must never
   be presented as available.
