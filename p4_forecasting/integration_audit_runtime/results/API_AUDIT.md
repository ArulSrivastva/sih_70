# API Audit — `integration_api` `/api/*` endpoints

Source of truth: `results/api_audit_battery.json` (23-case validation battery),
`results/e2e_trace.json` (live `POST /api/analyze`), `results/performance.json`,
`results/security_audit.json`, `results/model_identity.json`, `results/offline_proof.json`,
`results/determinism.json`.

> Note: this audit is a **read-only continuation** of the previously completed
> integration QA. Every value below is transcribed from the authoritative
> evidence files under `p4_forecasting/integration_api/results/`; the application
> was NOT re-run for this continuation.

## 1. Endpoint matrix

| Endpoint | HTTP | Notes |
|----------|------|-------|
| `GET /api/health` | 200 | service + `ml.{p2,p3_image,p3_tabular,reference_frame}` + `forecasting`; tabular `available:false` |
| `POST /api/analyze` | 200 | full dashboard contract (14 top-level blocks) |
| `POST /api/detect` | 200 | detection block |
| `POST /api/classify` | 200 | classification block |
| `POST /api/forecast` | 200 | `{status, model, forecast[3]}` |

## 2. Validation battery — 22 rejection cases + 1 control (source: api_audit_battery.json)

| # | Case | HTTP | Error code | Expectation met |
|---|------|------|------------|-----------------|
| 1 | `missing_history` | 422 | `MISSING_FEATURE` | yes |
| 2 | `history_4_items` | 422 | `INVALID_HISTORY_LENGTH` | yes |
| 3 | `history_6_items` | 422 | `INVALID_HISTORY_LENGTH` | yes |
| 4 | `spacing_30h` | 422 | `INVALID_HISTORY_SPACING` | yes |
| 5 | `non_monotonic` | 422 | `NON_MONOTONIC_HISTORY` | yes |
| 6 | `bad_timestamp_full5` | 422 | `INVALID_TIMESTAMP` | yes |
| 7 | `lat_95` | 422 | `INVALID_LATITUDE` | yes |
| 8 | `lat_minus40_full5` | **200** | — | **by design (see §3)** |
| 9 | `lon_negative` | 422 | `INVALID_LONGITUDE` | yes |
| 10 | `lon_360` | 422 | `INVALID_LONGITUDE` | yes |
| 11 | `wind_negative` | 422 | `INVALID_WIND` | yes |
| 12 | `wind_9999` | 422 | `INVALID_WIND` | yes |
| 13 | `pressure_800` | 422 | `INVALID_REQUEST` | yes |
| 14 | `nested_pressure_1200` | 422 | `INVALID_REQUEST` | yes |
| 15 | `sst_nan_str` | 422 | `NON_FINITE_VALUE` | yes |
| 16 | `obs_extra_field` | 422 | `INVALID_REQUEST` (`extra=forbid`) | yes |
| 17 | `meta_extra_field` | 422 | `INVALID_REQUEST` (`extra=forbid`) | yes |
| 18 | `body_extra_field` | 422 | `INVALID_REQUEST` (`extra=forbid`) | yes |
| 19 | `empty_body` | 422 | `MISSING_FEATURE` | yes |
| 20 | `not_found_endpoint` (`/api/nope`) | 404 | — | yes |
| 21 | `battery_good_control` | 200 | — | yes (control) |
| 22 | `oversized_body` (~300 KB) | 413 | `INVALID_REQUEST` | yes |

Summary: **21 of 22 malformed-input cases rejected with the expected structured
error code; 1 valid control accepted; 1 by-design acceptance (lat −40).**
No invalid input is silently repaired; every rejection uses the phase-6 unified
`{status:"error", error:{code,message}}` shape.

## 3. Correct interpretation of `lat_minus40_full5 → HTTP 200`

`lat_minus40_full5` returns **200 (accepted)** at the **phase-6 physical-bounds
layer**, and this is a **contract-design property, not a bug**:

* `phase6/config.py` defines `LAT_RANGE = (-90.0, 90.0)`, i.e. phase-6 validates
  against the **full physical latitude range**.
* `lat = -40` lies inside `(-90, 90)`, so it passes `INVALID_LATITUDE` bounds
  check by design at that layer.
* The **North Indian Ocean starts near 0°N**; `-40` is physically not NIO, so an
  NIO-only deployment needs a **strict policy guard upstream** (a policy
  decision, not a defect in the phase-6 contract).

### Implemented (P1, 2026-08-30): upstream NIO policy guard
The integration layer (`AnalyzeRequest`) now guards the NIO box **upstream** of
phase-6, so through the `/api/*` routes:

* `lat = -40` and `lon = 120` are rejected `422 OUT_OF_DOMAIN` (never clamped).
* Domain: lat `[0, 30]` deg N, lon `[40, 100]` deg E (degrees-East).
* Only `allowed_integration` code changed; the phase-6 physical-bounds layer is
  left intact and its own `/forecast` behaviour is unchanged.

`lat_minus40_full5` remains a documented property of the *phase-6 contract*; the
integration layer simply refuses to forward out-of-NIO input to it.

## 4. Error vocabulary / consistency
* Unified `{status, error:{code,message}}` on all rejects; no stack traces or
  filesystem paths leak.
* Error codes: `INVALID_*`, `NON_FINITE_VALUE`, `MISSING_FEATURE`,
  `INVALID_REQUEST`, plus `MODEL_NOT_READY` / `INFERENCE_ERROR` for model-path
  failures.
* Strict schemas (`extra=forbid`) reject unexpected fields rather than dropping
  them.

## 5. Determinism (source: determinism.json)
Two identical `POST /api/analyze` calls → **byte-identical** response (4,212 bytes).
Deterministic inference confirmed.

## 6. Offline behaviour (source: offline_proof.json)
`/api/analyze` completes `200 / success` (64.33 ms) with `socket.connect` /
`socket.sendall` patched to raise → **PASS**. The only network use is the
dashboard's optional OSM basemap (display-only, not part of the forecast
computation).

## 7. Security (source: security_audit.json)
* Zip opened read-only, **never extracted to disk** (1,207 members).
* All path-traversal probes (`../../etc/passwd`, `PS70-main/../x`, `/etc/passwd`,
  backslash variants) → rejected with `ModelNotReady`.
* No user-supplied filesystem/image path is accepted by any endpoint (the
  dashboard upload is client-side display only — see USER_WORKFLOW_AUDIT.md).
* Backend binds local and CORS is restricted to local dev origins.

## 8. Model-backed inference identity (source: model_identity.json)
All three components load and run on CPU from the artifact bundle:
* P2: `mobilenet_v3_small (weights=None)`, 3 patterns / 7 categories, runtime
  params 1,149,995.
* P3 image: `resnet18 (weights=None)`, 7 classes, runtime params 11,275,976.
* P4 EXP005: GRU + Huber, horizons [6,12,24], targets [lat,lon,wind_speed_kmh],
  runtime params 89,577, validation primary score 113.07414084856835.

## Verdict
**PASS** — the API honours the phase-6 validation contract, is offline-clean,
deterministic, model-backed, and security-sound for a local demo deployment.
`lat_minus40 → 200` is correctly established as a contract-design property, not a bug.

---

## 9. Post-audit addition: `POST /api/image` (Option A) — 2026-08-30

A subsequent implementation added a secure, offline image-ingestion endpoint.
Verified by `tests/test_image_upload.py` (19 tests) + a live HTTP call. It does
**not** alter the audited `/api/analyze` contract (that endpoint is unchanged and
re-verified below).

### Endpoint matrix (updated)
| Endpoint | HTTP | Notes |
|----------|------|-------|
| `POST /api/image` | 200 | multipart JPEG/PNG → P2 + P3 on the uploaded bytes (`ImageAnalyzeResponse`) |

### Request handling
* Multipart `file` field; optional param → `MISSING_FEATURE` when absent.
* Extension (`{.jpg,.jpeg,.png}`) **and** MIME (`{image/jpeg,image/png}`)
  allow-lists; then actual in-memory decode via `PIL` — real bytes must decode,
  so a ZIP masquerading as `.png` is rejected.
* Size cap: 5 MiB (beyond the phase-6 `MAX_REQUEST_BYTES` 413 guard).
* Filename used **only** as a display label (`Path(...).name`); never a path.

### Validation / security battery (all PASS in tests)
| Case | Result |
|------|--------|
| valid PNG/JPEG | 200 success; P2+P3 run |
| missing file | 422 `MISSING_FEATURE` |
| bad MIME (`text/plain`) | 422 `INVALID_REQUEST` |
| corrupt bytes | 422 `INVALID_REQUEST` (decode fail) |
| oversized (>5 MiB) | 413/422 `INVALID_REQUEST` |
| archive masquerading as `.png` | 422 `INVALID_REQUEST` (not a decodable image) |
| path-traversal filename `..\..\etc\passwd.png` | 200; `source="passwd.png"` (basename, no path) |
| absolute/relative path filename | 200; basename only |
| socket-blocked (offline) inference | 200 success |
| no stack trace / no path leak | structured error only |

### Response honesty
* `detection.location` / `movement*` = null (image alone gives no fix/motion).
* `satellite.boundingBox` = null (no audited localizer).
* `provenance.image_source = "USER-UPLOADED IMAGE"` (vs the empty/absent label
  on the `/api/analyze` reference-frame path).
* Exposes only P3 category + softmax confidence; wind regressor + hardcoded
  pressure NOT exposed; tabular `NOT_RUN` preserved.
* No forecast / landfall / risk for an image alone — clearly stated in notes.

### Regression
`/api/analyze` still returns `200 success` with the demo history (forecast
`[6,12,24]`, `image_source` empty) over live HTTP; the full integration_api suite
(58 incl. 13 NIO) and phase-6 suite (67) pass; backend total 125.

## 10. P1 hardening verification (2026-08-30) — live HTTP confirmation

A further live-HTTP E2E pass (real uvicorn) re-confirmed the full contract on a
fresh port and additively verified the image guards that a strict read of the
battery table had not yet exercised over HTTP:

* **P1-A tabular reproduction.** Health (`GET /api/health`) returns 200 and
  `p3_tabular.available=false` with the precise no-extraction reason; the stored
  tabular metrics were **independently reproduced** (acc 47.0%, macro-F1 0.3703,
  wind MAE 18.84, wind RMSE 27.28, pressure MAE 5.11, pressure RMSE 8.33;
  0 SID overlap) — see `reports/P1_HARDENING_REPORT.md`. The API deliberately
  does not over-claim in-process servability.
* **NIO guard over HTTP.** `lat = -40` and `lon = 120` both → `422
  OUT_OF_DOMAIN`; NaN lat → `422 NON_FINITE_VALUE`; wind < 0 → `422
  INVALID_REQUEST`; bad timestamp → `422 INVALID_TIMESTAMP`; short history →
  `422 INVALID_HISTORY_LENGTH`.
* **Image ingestion over HTTP.** bad MIME → `422 INVALID_REQUEST`; missing file
  → `422 MISSING_FEATURE`; **oversized (>5 MiB)** → rejected via the size cap /
  middleware 413 path; **a 5000×5000 image (84 KB, compressible) → `422
  INVALID_REQUEST` (≥4096 px / 16.8 MP dimension cap)**, while a 4096×4096
  boundary image is accepted (200) and an 8×8 image is accepted (200).
* **Provenance honesty over HTTP.** A valid PNG returns 200 with real inference,
  `image_source = "USER-UPLOADED IMAGE"`, and `location` / `movement*` /
  `boundingBox` all honest `null` (no fabricated geography for a single frame).

All checks PASS; no application defect was revealed and no source change was
needed as a result of this E2E pass.
