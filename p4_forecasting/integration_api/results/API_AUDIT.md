# API Audit — `integration_api` `/api/*` endpoints

Scope: contract correctness, validation strictness, error vocabulary, offline
behaviour, determinism, model identity, security. Full numeric evidence under
`results/` (all JSON files named below).

## 1. Endpoint matrix (live `TestClient` trace)

| Endpoint        | Valid demo body → status | Notes |
|-----------------|--------------------------|-------|
| `GET /api/health` | 200 | returns service + `ml.{p2,p3_image,p3_tabular,reference_frame}` + `forecasting`; tabular `available:false` |
| `POST /api/analyze` | 200 | full dashboard contract, 14 top-level keys present |
| `POST /api/detect` | 200 | detection block |
| `POST /api/classify` | 200 | classification block |
| `POST /api/forecast` | 200 | `{status, model, forecast[3]}` |

## 2. Validation battery (`results/api_audit_battery.json`, 23 cases)

Every invalid input is rejected with the *correct* pydantic error code and never
silently repaired:

| Case | status | code |
|------|--------|------|
| missing `history` | 422 | `MISSING_FEATURE` |
| 4 / 6 observations | 422 | `INVALID_HISTORY_LENGTH` |
| 30h (non-6h) spacing | 422 | `INVALID_HISTORY_SPACING` |
| reversed (non-monotonic) timestamps | 422 | `NON_MONOTONIC_HISTORY` |
| unparseable timestamp | 422 | `INVALID_TIMESTAMP` |
| `lat=95` | 422 | `INVALID_LATITUDE` |
| `lat=-40` | **200** | within declared `[-90,90]` physical range — *accepted as designed* (contract is global latitude, not NIO-restricted) |
| `lon=-5`, `lon=360` | 422 | `INVALID_LONGITUDE` |
| `wind=-1`, `wind=9999` | 422 | `INVALID_WIND` |
| `pressure=800`, `pressure=1200` | 422 | `INVALID_REQUEST` |
| `sst:"nan"` | 422 | `NON_FINITE_VALUE` |
| extra field on observation / meta / body | 422 | `INVALID_REQUEST` (`extra=forbid`) |
| empty body | 422 | `MISSING_FEATURE` |
| unknown route `/api/nope` | 404 | — |
| oversized body (~300KB) | 413 | `INVALID_REQUEST` |
| good control | 200 | `success` |

Note: the single `lat=-40 → 200` row is a **contract-design property, not a
defect**: the phase-6 schema validates latitude to the full physical `[-90,90]`
range. A North-Indian-Ocean-only guard would require an upstream policy decision;
it is flagged as a recommendation, not a bug.

## 3. Error vocabulary
Errors use the unified shape `{status:"error", error:{code,message}}` via the
phase-6 handlers; no stack traces or filesystem paths leak (codes:
`INVALID_*`, `NON_FINITE_VALUE`, `MISSING_FEATURE`, `MODEL_NOT_READY`,
`INFERENCE_ERROR`).

## 4. Determinism (`results/determinism.json`)
Two consecutive `/api/analyze` calls with the same body → **byte-identical**
(4212 bytes).

## 5. Offline proof (`results/offline_proof.json`)
`/api/analyze` completes 200/success with `socket.connect`/`sendall` patched to
raise → **PASS**. The only network dependency is the browser's optional OSM
basemap (display-only, disclosed in `MapView`).

## 6. Model identity (`results/model_identity.json`)
All three model backends load and run on CPU from the zip/artifacts:

| Backend | backbone | classes | runtime parameters |
|---------|----------|---------|--------------------|
| P2 detector | `mobilenet_v3_small (weights=None)` | 3 patterns / 7 categories | 1,149,995 |
| P3 image | `resnet18 (weights=None)` | 7 | 11,275,976 |
| P4 EXP005 | GRU (Huber) | horizons 6/12/24, targets lat/lon/wind | 89,577 |

(Reconciliation: earlier context quoted P2 1,162,141 / P3 11,285,596 — these
differ slightly from the *live, re-instantiated-model* counts above because the
originals came from a different measurement; the numbers here are measured from
the exact running models.)

## 7. Security (`results/security_audit.json`)
* Zip is opened read-only, never extracted (`zip_never_extracted_to_disk:true`).
* All path-traversal probes (`../`, `/`, backslash) rejected with `ModelNotReady`.
* No user-supplied filesystem path or image path is accepted by any endpoint
  (upload is client-side display only; see USER_WORKFLOW_AUDIT.md).
* Backend binds `127.0.0.1`; CORS restricted to local dev origins.

## Verdict
**PASS** — every endpoint honours the phase-6 validation + error contract, is
offline-clean, deterministic, model-backed, and security-sound for a local
demo deployment.
