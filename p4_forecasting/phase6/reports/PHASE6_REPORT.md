# PHASE 6 REPORT - FORECASTING API & FRONTEND INTEGRATION

**Recorded at:** 2026-08-29T12:27:34.427701+00:00  
**Final status:** `PASS`  

> **Phase 6 exposes the validated Phase-5 forecasting system through a local API. It does not retrain the model and does not establish new forecasting-performance claims.**

## 1. Objective
Expose the audited Phase-5 inference engine (EXP005, GRU + Huber) through a clean, local, offline FastAPI so a future React/Leaflet frontend can request +6h/+12h/+24h forecasts with zero knowledge of PyTorch or feature engineering.

## 2. Phase-5 integration

- Public interface reused (inspected before coding): `phase5/service/forecasting_service.py` -> `ForecastingService` with `forecast(history)` and `compare_baselines(history)`.
- `phase6/integration/forecasting_adapter.py` maps the HTTP request schema onto the Phase-5 input contract and maps Phase-5 error codes onto the public Phase-6 vocabulary.
- No scientific component (engineering, normalisation, model, baselines) was duplicated, modified or re-trained.

## 3. API architecture

```
HTTP request
  -> Pydantic validation (422 before inference)
  -> ForecastingAdapter (schema mapping + error mapping)
  -> Phase-5 ForecastingService
  -> EXP005 (CPU, deterministic, offline)
  -> validated output -> JSON response
```

## 4. Endpoints

- `GET /health` -> **PASS**
- `GET /model` -> **PASS**
- `POST /forecast` -> **PASS**
- `POST /forecast/compare` -> **PASS**

## 5. Request schema

```json
{"history": [
  {"timestamp": "2025-11-29T00:00:00Z", "latitude": 12.1,
   "longitude": 85.2, "wind_speed_kmh": 100.0,
   "pressure_hpa": 980.0, "sst": 28.4, "wind_u": 3.2,
   "wind_v": -1.4},
   ... exactly 5 observations, 6-hourly, t-24h ... t ...]}
```

## 6. Response schema

```json
{"status": "success",
 "model": {"experiment_id": "EXP005", "family": "GRU",
            "loss": "Huber"},
 "input": {"history_hours": 24, "history_steps": 5,
            "feature_count": 16},
 "forecast": [{"hours": 6, "latitude": ..., "longitude": ...,
                 "wind_speed_kmh": ...} x3]}
```

## 7. Input validation

Pydantic-first: length (exactly 5), ISO timestamps, strict 6-hour cadence, monotonicity, physical bounds, finite values, mandatory fields.  Rejected before the service is ever called with HTTP 422; nothing is repaired or interpolated.
- battery: 11 invalid cases -> all 422 with expected codes -> **PASS**

## 8. Error handling

Structured `{status, error:{code, message}}` on every failure; stack traces and filesystem paths are never exposed.  Error codes: `INVALID_REQUEST, INVALID_HISTORY_LENGTH, INVALID_TIMESTAMP, INVALID_HISTORY_SPACING, NON_MONOTONIC_HISTORY, INVALID_LATITUDE, INVALID_LONGITUDE, INVALID_WIND, MISSING_FEATURE, NON_FINITE_VALUE, MODEL_NOT_READY, INFERENCE_ERROR`.

## 9. Causality

- **PASS**
- Future (t+6/+12/+24) values are never accepted (rejected 422); changing the latest observed value DOES change the forecast.
- The request schema has no target/future slots at all; the adapter forwards exactly the 5 validated observations.

## 10. Determinism

- **PASS** - identical requests produce byte-identical responses (single-thread CPU inference).

## 11. Offline operation

- **PASS** - no network imports in the inference path; a forecast still succeeds with sockets disabled; all artifacts are local.

## 12. Baseline integration

- **PASS** - persistence and movement-vector outputs match the audited Phase-2 definitions (parity-checked against `phase2/baselines`).  The compare endpoint exists for debugging / demonstration, never to claim the model 'wins'.

## 13. CORS

- origins: `http://localhost:3000, http://localhost:5173` (local dev only; no wildcard).

## 14. OpenAPI

- **PASS** - `/docs` and `/openapi.json` are served locally by FastAPI.

## 15. Latency (software only; no accuracy claim)

- cold model load: **0.001 ms**
- warm inference: mean 1.915 ms | median 1.88 ms | min 1.649 ms | max 2.28 ms (15 calls)

## 16. Tests

- **67 passed / 0 failed** (pytest, cache + bytecode disabled).
- Files: health, model_endpoint, forecast_endpoint, compare_endpoint, validation_errors, response_schema, causality, determinism, offline, source_immutability + conftest.

## 17. Source immutability

Checked against `phase6/results/source_hashes_before.json`.  Only files under `p4_forecasting/phase6/` were created.
- P1: 0 file(s) changed
- Phase 1: 0 file(s) changed
- Phase 2: 0 file(s) changed
- Phase 3: 0 file(s) changed
- Phase 4: 0 file(s) changed
- Phase 5: 0 file(s) changed
- outside phase6: 0 file(s) changed

## 18. Frontend integration instructions

1. Start the API (from `p4_forecasting/`):
```
python -X utf8 -m uvicorn phase6.api.app:app --host 127.0.0.1 --port 8000
```
2. Ask the frontend to POST `history` (5 observations) and read `data.forecast`.
```javascript
fetch("http://localhost:8000/forecast", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(request)
})
  .then(r => r.json())
  .then(data => { /* data.forecast = [{hours, latitude, longitude, wind_speed_kmh}x3] */ });
```
3. Interactive docs: http://localhost:8000/docs

## 19. Limitations

- Input requires exactly 24h of 6-hourly history ending at `t`.
- Forecasts are 6h/12h/24h ahead; no longer lead times in this phase.
- Latency numbers are software timings on this machine, not forecasting-accuracy claims.
- CORS is limited to local dev origins; a cross-origin deployment would need an explicit configuration change.

## 20. Scientific statement

> **Phase 6 exposes the validated Phase-5 forecasting system through a local API. It does not retrain the model and does not establish new forecasting-performance claims.**  

The project retains the audited Phase-4 results unchanged: `EXP005` is the validation-selected ML champion, while the movement-vector baseline remains superior overall on the audited test comparison and persistence remains stronger at +6h.  No 'AI beats traditional methods', 'best predictor', 'state-of-the-art', '100% accurate' or real-time-accuracy claim is made.

