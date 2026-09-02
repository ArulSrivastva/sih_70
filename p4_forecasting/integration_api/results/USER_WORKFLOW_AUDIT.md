# User Workflow Audit

Traces the end-to-end path a user experiences against the real integrated
system, and records where the workflow is live vs. where it is a stated demo/
client-side limitation.

## 1. Happy path (what actually works today)

1. User starts backend: `uvicorn integration_api.app:app` (fastapi on :8000).
2. User starts dashboard: `npm run dev` (Vite :5173, proxies `/api` → :8000).
3. On load, `App.jsx` calls `fetchAnalyze()` → `POST /api/analyze` with the demo
   history (5 obs, BOB07) + demo meta.
4. Backend runs P2 detection + P3 classification on the fixed reference frame,
   P4 EXP005 forecast on the history, and the landfall/risk heuristics; returns
   the full dashboard contract **including provenance**.
5. Dashboard renders: header, status banner, map (current + historical + forecast
   + landfall + illustrative cone), satellite viewer, forecast timeline, landfall,
   risk gauge, model-intelligence, baseline-vs-model benchmark, pipeline status,
   charts, and (new) the **provenance panel**.
6. Every number shown is either a real model output, an observed value, or a
   clearly-labelled heuristic; naive users see "INTEGRATED · DEMO HISTORY".

## 2. What the user CANNOT do (honest gaps)

| Action user might try | Reality today | Classification |
|----------------------|---------------|----------------|
| Upload a real satellite image and have it analysed | **Upload is client-side only.** `SatelliteViewer.handleFile` shows a local `<img>`; it is NOT sent to the backend and does NOT re-run P2/P3. The live `/api/*` endpoints accept only the validated history + meta, never an image. | Documented limitation (no image-ingest endpoint) |
| Supply a different cyclone history (other than the demo) | The backend *does* accept any valid 5×6h history via `/api/analyze`, but the dashboard's `client.js` always sends `DEMO_HISTORY`/`DEMO_META` — there is no UI to enter/edit history. | Partial: API supports it, UI fixed to demo |
| See a "live" satellite feed update | Not possible; P2/P3 run on one deterministic reference frame, disclosed everywhere. | By-design honesty |
| Get an official IMD-grade decision | Not possible; explicitly "not an official warning". | By-design honesty |
| Use the system fully offline | Core inference is offline; only the OSM basemap needs internet (and degrades gracefully). | Works with note |

## 3. Recommendation (outside the allowed Step-16 scope, for a future iteration)
* Ship a **user-feed image endpoint** (`POST /api/image` accepting an image bytes +
  history, mirroring `AnalyzeRequest`) and have `SatelliteViewer` POST the
  selected file to it, re-running P2/P3 on the user's frame. This is the single
  biggest remaining workflow gap and is *not* implemented because it is a
  feature addition, not a correctness fix.
* Add a minimal history editor so users can provide their own 5-step track.

## Verdict
**HONEST & FUNCTIONAL with documented workflow limitations** — the live loop is
real and closed for the demo input; user supplied images are knowingly display-only
at present (flagged, not hidden).
