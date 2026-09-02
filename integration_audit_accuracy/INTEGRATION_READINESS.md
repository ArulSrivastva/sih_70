# Integration Readiness

**Overall: NOT READY for live end-to-end integration without change.** Every block has a verified, well-defined status; the blocker is the frontend↔backend glue layer plus honest rebranding of the detection/classification claims.

## Status per component

| Component | Status | Evidence |
|---|---|---|
| P1 Data | **PASS_WITH_WARNINGS** | Structure perfect; synthetic detection labels & Kaggle provenance limits; “100% completeness” wording wrong; Dataset-C bfill risk documented & cleaned in P4 |
| P2 Detection | **NOT_VERIFIABLE_AS_DETECTION** (reported metrics PASS) | metrics reproduce exactly; presence/bbox never scored; labels synthetic |
| P3 Classification | **PARTIAL** (image PASS; tabular NOT_RUN) | image metrics reproduce; lightgbm not installed; comparison invalid |
| P4 Forecasting | **PASS_AS_PIPELINE / WEAK_AS_FORECASTER** | methodology+numbers reproduced; track accuracy below baselines; EXP006 registry anomaly |
| P5 Frontend | **UNWIRED (mock-only)** | USE_MOCK=true both copies; no live data path |
| API layer | **WORKS, WRONG CONTRACT** | phase6 verified live; no /api/* paths; shape mismatches |
| Leakage | **CLEAN for P4; WARNING for P2/P3 images** | 22 same-storm cross-split frame pairs; split/time stats clean |

## Blocking gaps (ordered by significance for a live demo)

1. **No `/api/analyze`-style integration on the frontend path.** Dashboard calls `GET /api/*`; phase6 serves `/health`, `/model`, `POST /forecast`, `/forecast/compare`. Add an adapter endpoint (e.g., FastAPI or a small gateway) exposing the dashboard contract and forwarding to phase6. (Small, well-scoped.)
2. **No detection / classification output anywhere in the serving surface.** The dashboard shows Detection/Classification panels fed by the model; the delivered backend forecasts only. Either build trivial endpoints (P2/P3 models exist and load) or gate the dashboards to forecast-only + mark others “mock”.
3. **Field-shape bridge** for forecast items (`hour/hour…`, `lat/lon` vs `latitude/longitude`, `windSpeedKmh` vs `wind_speed_kmh`, missing `confidence`/`pressureHpa`). Map on the adapter. (Cheap.)
4. **Model-claim honesty**: present track forecasts WITH the movement-vector reference; do not call 91.4 km @ 6h “accurate track” while it is worse than the baseline.
5. **Dependency completeness for reproducible eval**: install `lightgbm` if tabular claims must be re-verified; ship phase3 weights if its numbers are to be cited.

## What makes it READY (smallest honest path, ranked by effort/trust)

- Forecast-only dashboard + adapter to phase6 (`/forecast`) → honest “machine-track demo” with baseline comparison panel (compare endpoint already returns persistence/movement – i.e., exactly the panel the dashboard already renders).
- Detection/classification panels: either (a) mark explicitly as `mock`, or (b) add endpoints backed by the delivered checkpoints (they load and run on CPU) with the documented label caveats.
- Keep USE_MOCK default but add an env toggle; ship docker/vite wrapper none currently exist.
- Add sst-availability note on the map when numbers come from partial ERA5 (28% rows missing sst at master level).

## Not required for integration but strongly advised before any scientific claim

- Rebuild the 21-frame test sets to be storm-disjoint (remove same-storm cross-split frames) if classification is featured.
- Re-run EXP006 claim (registry anomaly) or drop it from the report.
- Re-evaluate the champion against movement-vector explicitly in all marketing/demo copy.