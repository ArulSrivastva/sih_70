# P0 FIX REPORT — SIH 2026 PS 26070

**System:** SIH 2026 PS 26070 Tropical Cyclone AI System · VARTHA forecaster UI
**Scope:** Close the two verified P0 gaps from the integration QA verdict
`READY_WITH_MINOR_LIMITATIONS`:
  * **P0-A** — real user-uploaded satellite images never reached P2/P3 backend
    inference (client-side display only, no ingest endpoint).
  * **P0-B** — the dashboard was hard-wired to `DEMO_HISTORY` (no history editor).
**Date:** 2026-08-30
**Method:** Everything below was **actually run and verified** — backend test
suites, frontend lint/build, and live HTTP end-to-end requests against a running
uvicorn server. No model retraining; no P1–P5 / phase6 / dataset / model source
changed; no fabricated metrics; honesty labels preserved.

---

## 1. P0-A — real satellite image ingestion

### 1.1 Backend: `POST /api/image`
* `p4_forecasting/integration_api/routes.py` — `analyze_image()`:
  * Accepts `multipart/form-data` field `file`.
  * **Validation** (in order, each returning a clean structured error):
    1. file present → else `422 MISSING_FEATURE`
    2. extension allow-list `{.jpg,.jpeg,.png}` → else `422 INVALID_REQUEST`
    3. MIME allow-list `{image/jpeg,image/png}` → else `422 INVALID_REQUEST`
    4. size cap **5 MiB** → else `413/422 INVALID_REQUEST`
    5. **actual in-memory decode** via `PIL.Image.open(bytes).convert("RGB")`
       (a ZIP / corrupt / non-image masquerading as `.png` is rejected) →
       else `422 INVALID_REQUEST`
    6. **dimension guard** (≤ 4096 px per side / 16.8 MP — anti
       decompression-bomb) → else `422 INVALID_REQUEST`
  * Filename is used **only as a display label** (`Path(file.filename).name`);
    never a filesystem path; nothing is written to disk; no archive extraction;
    no network.
  * Reuses the **same** P2 detector (`get_detector().detect`) and P3 image
    classifier (`get_image_classifier().classify_image`) as the reference-frame
    path via `CycloneAnalyzer.analyze_image()` — **no second inference
    implementation.**
* `p4_forecasting/integration_api/schemas.py` — `ImageAnalyzeResponse`:
  `meta`, `detection`, `classification`, `satellite`, `provenance`. Deliberately
  **no** forecast/landfall/risk (those require the validated history).
  * `detection.location` / `movement*` = **null** (a single frame is not a
    localizer and has no motion without history — honest, never invented).
  * `classification` exposes only P3 category + softmax confidence (the P3 wind
    regressor and hardcoded image pressure are **not** exposed).
  * `satellite.boundingBox` = **null** (no audited localizer).
  * `provenance.image_source` = **`USER-UPLOADED IMAGE`** (distinct from the
    reference-frame `/api/analyze` path, which carries no such label).

### 1.2 Frontend: `SatelliteViewer`
* `cyclone-dashboard/src/components/SatelliteViewer.jsx`:
  * "Upload Satellite Image (backend analyzed)" → actually POSTs the file to
    `/api/image` via `uploadSatelliteImage()`.
  * Shows the selected image locally (preview) and explicit state machine:
    **selected → uploading → done / failed** (with visible error on failure).
  * Honest labels: "User-uploaded satellite image" /
    `USER IMAGE · BACKEND ANALYZED` / `REFERENCE FRAME · NOT LIVE`. The
    reference frame remains the zero-upload default; a user image is **never**
    presented as a live stream.
  * `onImageResult` surfaces the returned `provenance` to a second
    `ProvenancePanel` so the honesty notes are visible.
  * **No fake/static result** is shown as if it came from the upload; on
    failure the error is explicit.
* `cyclone-dashboard/src/api/client.js` — `uploadSatelliteImage(file)`.

### 1.3 Offline / backend-unavailable behaviour
* The backend inference path is **offline-clean** (socket-blocked test passes:
  `/api/image` still returns 200).
* If the backend is unreachable, `uploadSatelliteImage` throws and
  `SatelliteViewer` moves to **failed** state showing the explicit
  backend-unavailable/HTTP error — it does **not** silently claim inference
  succeeded. The existing reference-frame `/api/analyze` path continues to work
  independently.

---

## 2. P0-B — remove `DEMO_HISTORY` hard-wiring

### 2.1 `HistoryEditor`
* `cyclone-dashboard/src/components/HistoryEditor.jsx`:
  * Editable 5-row table with the exact feature schema the P4 `POST /api/analyze`
    contract expects: timestamp, latitude, longitude, wind_speed_kmh,
    pressure_hpa, sst, wind_u, wind_v.
  * Controls: **+ Add observation**, per-row **delete (✕)**, **Clear**,
    **Reset to demo**, and **Run forecast**.
  * **Client-side validation** (exactly 5 obs, 6h spacing, strictly
    chronological, physical bounds, NaN/non-finite rejection) — the model
    requires **exactly 5** timesteps, so that constraint is enforced explicitly
    and shown as an error if the count is anything else.
  * Header badge switches from **`DEMO DATA`** to **`USER-EDITED HISTORY`** as
    soon as any cell is edited, so demo vs. user input is always explicit.
  * "Run forecast" is disabled while the history is invalid. Demo results are
    never presented as user/model inference.
* `cyclone-dashboard/src/api/client.js` — `fetchDetect/fetchClassify/
  fetchForecast/fetchAnalyze` accept optional `(history, meta)`; `DEMO_HISTORY`
  remains only as the **default/in-reset** value, never silently substituted for
  a user-supplied valid history.
* `cyclone-dashboard/src/App.jsx` — `handleHistoryAnalyze(history)` sends the
  user history through `fetchAnalyze(history)` to the real backend.

### 2.2 Server-side authority
* The `POST /api/analyze` JSON contract is **unchanged** (still the audited
  phase-6 ForecastRequest). The backend re-validates with the audited rules; an
  invalid history returns a structured 422 and is **never silently repaired** —
  so a user cannot slip an invalid history into P4.

---

## 3. Files changed (this continuation + the established P0 work)

All changes are in **approved scope** (frontend + `allowed_integration`). No
P1–P5 / phase6 / dataset / model file was modified.

* `p4_forecasting/integration_api/routes.py` — `POST /api/image` + validation
  (dimension guard added in this continuation).
* `p4_forecasting/integration_api/schemas.py` — image response blocks.
* `p4_forecasting/integration_api/analyzer.py` — `analyze_image()` reuse of P2/P3.
* `p4_forecasting/integration_api/tests/test_image_upload.py` — image + history
  tests (dimension test added in this continuation).
* `cyclone-dashboard/src/api/client.js` — `uploadSatelliteImage`, optional
  history/meta args.
* `cyclone-dashboard/src/components/SatelliteViewer.jsx` — real backend upload.
* `cyclone-dashboard/src/components/HistoryEditor.jsx` — history editor
  (add/delete/clear added in this continuation).
* `cyclone-dashboard/src/App.jsx` — HistoryEditor wiring, `onImageResult`,
  footer.
* `cyclone-dashboard/src/components/ProvenancePanel.jsx` — optional title +
  `image_source` rendering (from the established P0 work).

---

## 4. API contract (updated)

### `POST /api/image`
| Item | Value |
|------|-------|
| HTTP method | `POST` |
| Content-Type | `multipart/form-data`, field `file` |
| Supported formats | `.jpg` / `.jpeg` / `.png` (MIME `image/jpeg` / `image/png`) |
| Max upload size | **5 MiB** (plus phase-6 body guard → 413) |
| Max dimensions | 4096 px/side, 16.8 MP total |
| Validation rejects | missing file (`MISSING_FEATURE`); bad ext/MIME/corrupt/overflow/Dimension (`INVALID_REQUEST`) |
| Success response | `ImageAnalyzeResponse` (meta, detection, classification, satellite, provenance); `status:"success"` |
| Honest values | `detection.location/movement=null`, `satellite.boundingBox=null`, `provenance.image_source="USER-UPLOADED IMAGE"`; **no** forecast/landfall/risk |
| Error response | `{status:"error", error:{code,message}}`; no stack trace / path leak |
| Offline | inference no network dependency; if backend down, frontend shows explicit failure (not fake success) |
| Storage | none — in-memory decode, no disk writes |

### `POST /api/analyze` (unchanged, now history-driven)
| Item | Value |
|------|-------|
| HTTP method | `POST`, JSON body |
| Request | `{history:[5x{timestamp,latitude,longitude,wind_speed_kmh,pressure_hpa,sst,wind_u,wind_v}], meta?}` |
| History requirement | **exactly 5** observations, 6h spacing, strictly chronological, physical bounds, finite |
| Validation rejects | `INVALID_HISTORY_LENGTH`, `INVALID_HISTORY_SPACING`, `NON_MONOTONIC_HISTORY`, `INVALID_TIMESTAMP`, `INVALID_LATITUDE`, `INVALID_LONGITUDE`, `INVALID_WIND`, `NON_FINITE_VALUE`, `INVALID_REQUEST` |
| Success response | full `AnalyzeResponse` (14 blocks incl. forecast `[6,12,24]`, landfall, risk, provenance) |
| Provenance distinction | `image_source` **absent** on the reference-frame path (vs `USER-UPLOADED IMAGE` on `/api/image`) |

---

## 5. Validation behaviour (verified)

Live HTTP + 45 backend tests confirmed:
* valid PNG upload → 200, real P2/P3 output, honest nulls, `image_source` set
* missing file → 422 `MISSING_FEATURE`
* bad MIME / corrupt / archive-masquerade → 422 `INVALID_REQUEST`
* oversized (>5 MiB) → 413 `INVALID_REQUEST`
* unreasonable dimensions (>4096 px) → 422 `INVALID_REQUEST`
* path-traversal / absolute filename → 200, `source` = basename only (never a path)
* historical: wrong count, non-monotonic, bad lat, bad wind, bad timestamp,
  NaN → all 422 with the documented codes

---

## 6. Test results (all actually run)

| Suite | Result |
|-------|--------|
| `integration_api` (whole) | **45 passed, 0 failed** (44 baseline + 1 new dimension guard test) |
| `phase6` regression | **67 passed, 0 failed** |
| Frontend `npm run lint` (oxlint) | **PASS** (exit 0; only pre-existing unused-var warnings) |
| Frontend `npm run build` (vite) | **PASS** (653 modules; pre-existing chunk-size warning only) |
| Live HTTP E2E (uvicorn, real server) | **PASS** — `/api/image` valid+invalid; `/api/analyze` valid+invalid; health; all as documented |

Live HTTP observed: `/api/image` on a real artifact frame → 200,
`detection.confidence=86`, `classification.category="Cyclonic Storm"`,
`satellite.source="user_sat.png"` (basename), `image_source="USER-UPLOADED
IMAGE"`, `location=null`.

---

## 7. Offline behaviour

* `tests/test_image_upload.py::test_offline_image_inference` patches
  `socket.connect`/`socket.sendall` to raise → `/api/image` still returns 200.
* The only network use in the dashboard is the optional OSM basemap
  (display-only, not part of inference).
* If `/api/image` is unavailable at runtime, the frontend moves to an explicit
  `failed` state with the real error — it does not claim success.

---

## 8. Honesty safeguards (preserved)

* `USER UPLOAD → BACKEND INFERENCE → REAL MODEL OUTPUT`: the uploaded bytes
  genuinely run through P2/P3; labelled as such.
* `location`/`movement`/`boundingBox` are honest `null` for a user image — no
  fabricated geography.
* No forecast/landfall/risk is invented for an image alone; notes state these
  require the validated 5-observation history.
* P3 wind regressor + hardcoded pressure not exposed; `tabular` stays `NOT_RUN`;
  confidence is softmax (not calibrated).
* Landfall/risk remain `HEURISTIC (NOT ML)`; cone stays `ILLUSTRATIVE`;
  pressure/confidence stay honest null; dashboard stays a research prototype.
* `DEMO DATA` vs `USER-EDITED HISTORY` is explicit in the HistoryEditor; demo is
  never shown as user/model inference.
* Movement-vector baseline remains visible wherever track comparisons are shown.

---

## 9. Immutability result

Verified against `p4_forecasting/integration_api/results/pre_audit_snapshot.json`
(authoritative snapshot, 426 files):

* **Snapshot files:** 426
* **Missing:** 0
* **Changed:** 12
* **Changed by bucket:**
  * `allowed_integration`: 3  (`analyzer.py`, `routes.py`, `schemas.py`)
  * `frontend`: 9  (`App.jsx`, `api/client.js`, `components/Header.jsx`,
    `LandfallPanel.jsx`, `LiveCycloneStatus.jsx`, `PipelineStatus.jsx`,
    `RiskIndicator.jsx`, `SatelliteViewer.jsx`, `map/MapView.jsx`)
* **Forbidden scope (p1/phase1–5/phase6/outside):** 0
* **PS70-main.zip:** unchanged (SHA256 matches, 1207 members, reference frame +
  model artifacts byte-identical)

All 12 changes are the approved frontend/backend integration work. New files
(`HistoryEditor.jsx`, `test_image_upload.py`) are additions in approved scope,
not modifications of audited source. Newly generated QA artifacts
(`integration_audit_runtime/`) are not counted as application-source changes.

---

## 10. Remaining P1/P2 limitations (unchanged by this work, kept disclosed)

* **P1** — P3 LightGBM / tabular classifier is `NOT_RUN` (lightgbm not
  installed); honestly reported, never faked.
* **P1** — P2/P3 are weak image classifiers on synthetic/leak-prone labels
  (pattern 0.714, acc 38.1%); numbers keep caveats.
* **P1** — P4 EXP005 track forecast is below the movement-vector baseline at
  every horizon; stays disclosed.
* **P1** — heuristic landfall/risk and the illustrative cone must never be
  presented as ML; currently correctly labelled.
* **P2** — no calibrated forecast pressure/uncertainty (honest nulls); cold
  bootstrap ~360 ms; P3 dominates analyze latency (~56 of 63 ms); Vite
  chunk-size warning.
* **P2 (policy)** — optional North-Indian-Ocean-only latitude guard upstream
  (the `lat = −40 → 200` contract-design property).

---

## 11. Verdict

**P0_FIX_COMPLETE.** Both P0 gaps are closed, verified end-to-end over live
HTTP, and immutability holds (only approved frontend/backend integration files
changed). See `INTEGRATION_QA_REPORT.md` for the full 18-step audit context.

This does **not** and must not claim production readiness: the remaining
P1/P2 limitations above (weak models, `NOT_RUN` tabular, below-baseline track,
heuristic landfall/risk, uncalibrated outputs) keep this a verified, honest
research prototype.
