# User Workflow Audit — end-to-end path a user actually experiences

Trace of the real integrated loop, grounded in the live `e2e_trace.json`
`/api/analyze` call and the verified frontend wiring. This is a **read-only**
analysis of the current state; it does not re-run the application.

## 1. Actual workflow (as built)

```
User opens dashboard (Vite dev server, proxy /api -> :8000)
        │
        ▼
Frontend prepares history + meta
  client.js builds DEMO_HISTORY (5 obs, BOB07 / "Cyclonic Storm ANIKA")
  + DEMO_META (systemId/systemName/basin/lastPass)
  USE_MOCK=false  →  fetchAnalyze() POST /api/analyze
        │
        ▼
POST /api/analyze
        │
        ├─ P2 detection  (MobileNet, reference frame 45(1).jpg)   → detected/confidence/location
        ├─ P3 classification (ResNet18 image model)               → category/confidence
        │    └─ P3 tabular (LightGBM)  → NOT_RUN (lightgbm absent)
        ├─ P4 EXP005 forecast (GRU+Huber)                          → +6/+12/+24h lat/lon/wind
        ├─ server heuristics (landfall + risk)                     → landfall.* , risk.*
        ├─ provenance assembly                                     → provenance.*
        └─ validation (history length/spacing/monotonic/bounds)    → structured errors
        │
        ▼
Response (14 blocks) renders:
  Header · LiveCycloneStatus · MapView (track+forecast+landfall+illustrative cone)
  SatelliteViewer · ForecastCard · LandfallPanel · RiskIndicator
  ModelIntelligence · BaselineComparison · PipelineStatus · ChartsPanel
  ProvenancePanel (new) · footer
```

## 2. What works today (verified)
* One `/api/analyze` call drives P2 → P3 → P4 → heuristics → provenance.
* Real model outputs (detection/classification/forecast), observed wind/pressure,
  heuristics — all value-propagated per `e2e_trace.json` invariants
  (`detection_location_equals_last_obs`, `forecast_horizons [+6/+12/+24]`,
  `risk_deterministic_formula`, `provenance_complete` all pass; 0 invariant issues).
* Offline inference verified (offline_proof.json: socket-blocked analyze returns
  200/success). The optional OSM basemap is the only network call and is
  display-only.
* Origin of every number is surfaced to the user via ProvenancePanel + honest labels.

## 3. Remaining limitations (explicitly identified)

| # | Limitation | Detail | Classification |
|---|------------|--------|----------------|
| L1 | **DEMO_HISTORY is static** | `client.js` always sends `DEMO_HISTORY`/`DEMO_META`; there is no history-editor UI. The API accepts arbitrary valid 5×6h history, but the dashboard is hard-wired to the demo case. | UI/UX limitation; API supports custom history |
| L2 | **SatelliteViewer upload is client-side only** | `handleFile` shows a local `<img>`; it is NOT posted to the backend and does NOT re-run P2/P3. | Documented gap |
| L3 | **No backend satellite-image ingestion endpoint** | None of `/api/*` accepts image bytes; the API takes only history + meta. So a real uploaded image cannot reach P2/P3 inference. | Architecturally the single biggest missing link |
| L4 | **Real uploaded images not yet connected to P2/P3 pipeline** | Consequence of L2/L3: P2/P3 run only on the fixed reference frame from the artifact. | Follows from L2/L3 |
| L5 | **Online basemap is optional & not part of forecast** | Map tiles come from an online OSM provider; if offline, the map area gracefully degrades (core inference unaffected). | By-design; displayed as such |
| L6 | **Offline inference itself verified** | Core backend inference has no network dependency (offline_proof PASS). | Verified working |

## 4. Verdict
**FUNCTIONAL AND HONEST, WITH TWO DEFINED WORKFLOW GAPS** — the real loop works
for the demo input and all outputs are truthfully sourced; the two substantive
gaps are (a) static `DEMO_HISTORY` and (b) real uploaded satellite images not yet
connected to backend inference (client-side display only, no ingest endpoint).
Both are disclosed and are candidates for a future P1 feature build, not silent
limitations.

---

## 5. Post-audit P0 workflow closure (Option A) — 2026-08-30

A subsequent implementation (this continuation) closed the two P0 workflow gaps.
The evidence lives in
`reports/P0_WORKFLOW_IMPLEMENTATION_REPORT.md`,
`results/P0_WORKFLOW_TEST_RESULTS.json`, and
`tests/test_image_upload.py` (19 new tests). Every new behaviour below was
**actually run** (44 integration_api tests + 67 phase-6 tests pass; live HTTP
calls to `/api/image` and `/api/analyze` return success).

### 5.1 L1 / L3 / L4 → RESOLVED: real uploaded images now reach P2/P3
* New `POST /api/image` (multipart) endpoint reuses the **same** P2 detector and
  P3 image classifier as the reference-frame path — no second inference impl.
* A valid JPEG/PNG upload returns `detection` + `classification`
  (`ImageDetectionBlock` / `ImageClassificationBlock`). `location`/`movement`
  are honest `null` (a single frame has no localizer and no motion), and
  `boundingBox` is `null` (no audited localizer). `provenance.image_source =
  "USER-UPLOADED IMAGE"` distinguishes it from the reference-frame path, which
  carries no such label.
* `SatelliteViewer` now calls `uploadSatelliteImage()` and shows explicit
  states (selected / uploading / done / failed); it is no longer client-side-only
  display.
* Live HTTP check on a real uploaded PNG → `200 success`,
  `detection.detected=false confidence=44`, `classification.category="Cyclonic
  Storm" confidence=51`, `source="user_sat.png"` (basename only). Honest: the
  uploaded bytes genuinely ran through P2/P3.

### 5.2 L2 → RESOLVED: history is no longer locked to DEMO_HISTORY
* New `HistoryEditor` component: 5-row × 8-column editable table
  (timestamp/lat/lon/wind/pressure/sst/wind_u/wind_v), client-side validation
  (exactly 5, 6h spacing, chronological, bounds), reset-to-demo, and
  "Analyze with this history" that posts the raw values through the **unchanged**
  `POST /api/analyze` contract (`fetchAnalyze(history)`).
* Server re-validates via the audited phase-6 rules; the backend is the source
  of truth. P4 EXP005, landfall and risk are computed from the **submitted**
  history.
* Live HTTP: `POST /api/analyze` with the demo history (as a stand-in for
  user-supplied) → `200 success`, forecast `[6,12,24]`,
  `provenance.reference_image` present, and `image_source` empty (i.e. NOT the
  user-image path) — confirming the two provenance paths are distinct.

### 5.3 Honesty preserved (no new fabrication)
* User-image responses carry notes that P2/P3 "are weak / not a validated
  localizer", that location/movement/bbox are null by design, that
  forecast/landfall/risk are **not** computed for an image alone, and that P3
  confidence is softmax (not calibrated).
* The P3 wind regressor and hardcoded image pressure are still **not** exposed;
  `tabular` stays `NOT_RUN`.
* The frontend labels an upload as "User-uploaded satellite image" /
  "USER IMAGE · BACKEND ANALYZED", never as a live stream; the reference frame
  remains the zero-upload default.

### 5.4 Security & offline (verified by tests)
`tests/test_image_upload.py` covers: oversized, bad MIME, corrupt/archive-
masquerade bytes, path-traversal / absolute-path filenames (used only as display
labels, never as a filesystem path), missing file, no stack-trace leak, and
socket-blocked **offline** inference. All pass.
