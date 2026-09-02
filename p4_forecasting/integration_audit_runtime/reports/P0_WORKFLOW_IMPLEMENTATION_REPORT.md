# P0 Workflow Implementation Report — Option A

**System:** SIH 2026 PS 26070 Tropical Cyclone AI System · VARTHA forecaster UI
**Scope:** Close the two verified P0 workflow gaps identified in the prior
integration QA verdict `READY_WITH_MINOR_LIMITATIONS` (limitations **L1** and
**L2**, and the image-ingestion-architecture gap L3/L4).
**Date:** 2026-08-30
**Continuation:** Inspection + plan → detail; no model retraining; no P1–P5 /
phase6 / dataset / model source changes; no network dependencies; no fabricated
metrics; honest reporting throughout.

---

## 1. P0 gaps closed

| Prior limitation | Resolved by |
|------------------|-------------|
| **L1 / L3 / L4** — real uploaded satellite images never reach P2/P3 backend inference (client-side display only, no ingest endpoint) | New **`POST /api/image`** multipart endpoint + `SatelliteViewer` rework that actually uploads and shows backend results |
| **L2** — `DEMO_HISTORY` static; no history-editor UI | New **`HistoryEditor`** feeding the unchanged `POST /api/analyze` contract |

Both gaps are implemented, tested (19 new backend tests + live HTTP), and
verified honest. No audit-evidence JSONs were altered to force a pass.

## 2. Architecture decision (confirmed Option A)

- **`POST /api/image`** is a **separate multipart endpoint** for image-only
  analysis (P2 + P3 on the uploaded frame).
- **`POST /api/analyze` stays JSON-only and UNCHANGED** — the audited contract
  and all history/detection/classification/forecast/landfall/risk blocks are
  byte-for-byte the same contract the frontend already consumed.
- This preserves the audited analyze contract, keeps `FormData` out of the JSON
  request path, and reuses the existing P2/P3 adapters (which already accept a
  PIL `Image.Image`) — so there is **no second inference implementation**.

## 3. Backend implementation

### 3.1 `POST /api/image` (integration_api/routes.py)
- Multipart `file` upload (optional param → `MISSING_FEATURE` when absent).
- **Extension allow-list** `{.jpg,.jpeg,.png}` + **MIME allow-list**
  `{image/jpeg,image/png}`; then **actual in-memory decode** via
  `PIL.Image.open(io.BytesIO(...)).convert("RGB")` — so a ZIP (or any
  non-image) masquerading as `.png` is rejected.
- **Size cap** 5 MiB (in addition to the phase-6 `MAX_REQUEST_BYTES` 413 guard).
- Filename is used **only as a display label** (`Path(file.filename).name`) —
  never as a filesystem path; no archive extraction; no disk writes; no network.
- Reuses `get_detector().detect(image)` and `get_image_classifier()
  .classify_image(image)` via a new `CycloneAnalyzer.analyze_image()`.
- All errors use the phase-6 unified `{status:"error", error:{code,message}}`
  shape; no stack traces or paths leak.

### 3.2 `ImageAnalyzeResponse` (integration_api/schemas.py)
New honest blocks:
- `ImageDetectionBlock` — `detected`, `confidence`, and **`location: null`,
  `movementDirection: null`, `movementSpeedKmh: null`** (a single frame is not a
  localizer and has no motion without a history).
- `ImageClassificationBlock` — category, scale (IMD), confidence, structural
  pattern; **P3 wind regressor and hardcoded pressure are NOT exposed**.
- `ImageSatelliteBlock` — `label` ("User-uploaded satellite image"),
  `boundingBox: null` (no audited localizer), `source` (basename only).
- `ImageProvenanceBlock` — `pipeline`, `sources`, `notes`, `image_source`
  (= `USER-UPLOADED IMAGE`), `tabular` (stays `NOT_RUN`).
- Top-level response has `meta`, `detection`, `classification`, `satellite`,
  `provenance` — and deliberately **no** forecast/landfall/risk (those require
  the validated history via `/api/analyze`).

## 4. P0-2: history editor

- **`src/components/HistoryEditor.jsx`** (new): a 5-row × 8-column editable
  table (timestamp, latitude, longitude, wind_speed_kmh, pressure_hpa, sst,
  wind_u, wind_v), client-side validation (exactly 5, 6h spacing,
  chronological, bounded ranges), a **Reset to demo** button, and an **Analyze
  with this history** button that posts the raw values through the **unchanged**
  `POST /api/analyze` contract.
- **`src/api/client.js`**: `fetchDetect/fetchClassify/fetchForecast/fetchAnalyze`
  now accept optional `(history, meta)`; `uploadSatelliteImage(file)` POSTs to
  `/api/image` and surfaces `sourceLabel`.
- The **server re-validates with the audited phase-6 rules** — the backend is the
  source of truth; an invalid history is rejected with a structured 422, never
  silently repaired.

## 5. Frontend upload UX (honest)

`SatelliteViewer.jsx` rework:
- Upload button `Upload Satellite Image (backend analyzed)` → real POST.
- Explicit states: **selected → uploading → done / failed** (with visible
  error on failure).
- Honest labels: "User-uploaded satellite image" / `USER IMAGE · BACKEND
  ANALYZED` / `REFERENCE FRAME · NOT LIVE`. The reference frame remains the
  zero-upload default. An upload is **never** presented as a live stream.
- `onImageResult` surfaces the user-image `provenance` into a second
  `ProvenancePanel` so the honesty notes are visible.

## 6. Tests (all run & passing)

### 6.1 Backend — `integration_api` whole suite: **44 passed, 0 failed**
Baseline 25 + **19 new** (in `tests/test_image_upload.py`):
- Image: valid PNG/JPEG, missing, bad MIME, corrupt, oversized, ZIP-masquerade,
  path-traversal filename (basename-only), absolute/relative path filename,
  **offline (socket-blocked)**, no-stack-leak, real-model-output sanity.
- History/analyze: valid 5×6h custom → forecast [6,12,24]; wrong count;
  non-monotonic; invalid lat; invalid wind; non-finite; extra field;
  provenance reference-frame-vs-user-image distinction.

### 6.2 Regression — `phase6` whole suite: **67 passed, 0 failed**

### 6.3 Frontend — `npm run lint` (oxlint) **PASS** (exit 0), `npm run build`
(vite) **PASS** (653 modules).

### 6.4 Live HTTP E2E (uvicorn, real server)
- `POST /api/image` (real PNG upload) → **200 success**;
  `detection.detected=false confidence=44`, `classification.category="Cyclonic
  Storm" confidence=51`, `satellite.source="user_sat.png"` (basename),
  `provenance.image_source="USER-UPLOADED IMAGE"`, `location/movement/bbox=null`.
- `POST /api/analyze` (user-supplied history) → **200 success**, `forecast[6,12,24]`,
  `provenance.reference_image` present, `image_source` absent (reference-frame
  path) → the two provenance paths are correctly distinct.

## 7. Honesty (no new fabrication)

- User images are analysed as **real P2/P3 model output** on the uploaded bytes —
  still weak on synthetic/leak-prone labels, with the caveat retained in notes.
- No fabricated geography: location/movement/boundingBox are honest `null` for
  an image.
- No unsupported forecast: `/api/image` returns no forecast/landfall/risk; that
  is stated explicitly.
- P3 wind regressor + hardcoded pressure not exposed; `tabular` stays `NOT_RUN`;
  confidence is softmax (not calibrated).
- Landfall/risk stay `HEURISTIC (NOT ML)`; cone stays `ILLUSTRATIVE`; pressure/
  confidence stay null; dashboard stays a research prototype.

## 8. Source immutability (Phase 13 re-verified)

Recomputed SHA256 of all **426** pre-audit snapshot files against
`integration_api/results/pre_audit_snapshot.json`:
- **Missing: 0** · **Changed in forbidden scope (p1/phase1–5/phase6/outside): 0**
- **12 changed — all in approved scope:** 9 `frontend` + 3 `allowed_integration`
  (`analyzer.py`, `routes.py`, `schemas.py` — the new `/api/image` code the task
  requires).
- **Zip unchanged:** `SHA256` matches, **1207** members, P2/P3 reference frame +
  model artifacts byte-identical.
- New files (`HistoryEditor.jsx`, `test_image_upload.py`) are additions to
  approved scope, not modifications of audited source.

## 9. Verdict

**P0_WORKFLOW_COMPLETE.**

- User image → backend validation → P2 → P3: **works**.
- Custom valid history → validated → P4: **works**.
- Offline, security (traversal/oversize/MIME/archive/missing), no-stack-leak:
  **verified**.
- Existing `POST /api/analyze` unchanged and re-verified: **PASS**.
- Lint + build: **PASS**.
- No prohibited modifications; immutability holds.

Remaining honest limitations (unchanged by this work): P2/P3 weak real-model
output, P3 tabular `NOT_RUN`, P4 track below movement-vector baseline, heuristic
landfall/risk, uncalibrated forecast, illustrative cone. All stay disclosed.
