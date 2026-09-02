# Integration QA Report — SIH 2026 PS70/260

**Audit:** End-to-end integration QA of the live product
(`cyclone-dashboard` ↔ `integration_api` ↔ P2/P3/P4/P5).
**Date:** 2026-08-30 · **Env:** Python 3.13.7, torch 2.6.0+cu124 (CPU inference),
FastAPI 0.111.0, Node 24.12 / Vite 8.2.2.
**Refusal to fake:** No retraining, no dataset edits, no prepared accuracy, no
mock result in the live path. Tabular (LightGBM) classifier is reported NOT_RUN,
never faked.

---

## Final verdict

### READY_WITH_MINOR_LIMITATIONS

The product is **genuinely integrated off the audited artifacts**: one
`/api/analyze` call drives P2 detection + P3 classification on the reference frame,
the audited P4 EXP005 GRU forecast, and the deterministic landfall/risk heuristics,
all offline, deterministic, and security-sound. Everything shown is traceable to a
real model output, an observation, or a disclosed heuristic. It is **not** "READY"
for claims it cannot back (no live image feed, no calibrated uncertainty/pressure,
track below the movement-vector baseline, weak P2/P3 image classifiers) — hence the
"with minor limitations" qualifier.

---

## PASS / FAIL table

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Pre-audit SHA256 snapshot recorded | **PASS** | `results/pre_audit_snapshot.json` (426 files + zip + per-member) |
| 2 | Source immutability (P1–P5 + zip untouched) | **PASS** | `results/source_immutability_report.json` — only allowed-scope files changed |
| 3 | Architecture documented end-to-end | **PASS** | `INTEGRATION_ARCHITECTURE.md` |
| 4 | Live `/api/analyze` trace (all invariants) | **PASS** | `results/e2e_trace.json` (200, 0 invariant issues) |
| 5 | Endpoint + validation matrix | **PASS** | `results/api_audit_battery.json`, `API_AUDIT.md` (22 invalid cases, 1 by-design 200) |
| 6 | Frontend↔backend contract consistent | **PASS** | `results/frontend_backend_contract.json` |
| 7 | No mock in live path (`USE_MOCK=false`) | **PASS** | grep: only dead `USE_MOCK=true` branches + static colour scale |
| 8 | Offline (socket-blocked analyze) | **PASS** | `results/offline_proof.json` (200/success under blocked sockets) |
| 9 | Runtime model identity + param counts | **PASS** | `results/model_identity.json` (P2, P3, EXP005 GRU all load) |
| 10 | Honesty audit (every field labelled) | **PASS** | `results/HONESTNESS_AUDIT.md` + fixes below |
| 11 | Forecast horizon mapping (+6/+12/+24, units) | **PASS** | e2e forecast `[6,12,24]` km/h; pressure/confid honest null |
| 12 | Risk/landfall labelled heuristic (not ML) | **PASS** | UI → "HEURISTIC (NOT ML)"; provenance states it |
| 13 | User workflow audit | **PASS (with gap)** | `USER_WORKFLOW_AUDIT.md` — user upload is display-only |
| 14 | Failure/edge cases + security (zip traversal) | **PASS** | `results/security_audit.json`; `API_AUDIT.md §7` |
| 15 | Performance | **PASS** | `results/performance.json` — ~79ms warm analyze avg; ~172ms cold |
| 16 | Fixes applied (honesty UI + provenance panel) | **PASS** | frontend edits (all in allowed scope) |
| 17 | Regression: tests + build + lint + offline | **PASS** | integration_api 25 + phase6 67 = 92 passed; `vite build` ✓; oxlint clean |
| 18 | caches cleaned / no stray artifacts | **PASS** | `__pycache__`, `.pytest_cache`, `dist` removed |

---

## What was verified (highlights)

- **One-call integration:** `POST /api/analyze` returns all 14 top-level blocks;
  detection/location, classification (category+confidence from P3; wind/pressure
  from latest observation), EXP005 GRU forecast (`+6/12/24h`, pressure/confid
  null), deterministic landfall + risk, full histories, satellite (bbox null), and
  provenance.
- **Determinism:** two identical calls → byte-identical (4212 bytes).
- **Offline:** core inference runs with sockets disabled.
- **Honesty chain:** forecast pressure/confidence are null (no calibrated outputs),
  landfall/risk are explicitly "HEURISTIC (NOT ML)", tabular NOT_RUN, weak P2/P3
  and the track-below-baseline result are surfaced (BaselineComparison), and the
  backend's provenance block is now rendered in the UI.

## Fixes applied (Step 16 — all within `cyclone-dashboard/src/`)
Added `ProvenancePanel` (rendered in `App`); replaced misleading
`FORECAST MODEL OUTPUT`/`AI Risk`/`LIVE STREAM`/`OPERATIONAL` labels with honest
ones; made `RiskIndicator` contributors dynamic; used `meta.lastPass` for the
satellite acquisition time; corrected the map cone legend + offline/online note;
rewrote the footer.

## Known limitations (why not an unqualified READY)
1. **No user-upload → backend image ingest** (upload is client-side display only).
2. Dashboard always sends the **demo history** (no history editor UI), though the
   API accepts arbitrary valid history.
3. P2/P3 run on a single **fixed reference frame**, not a live feed.
4. Forecast `pressureHpa`/`confidence` **null** (no calibrated outputs).
5. P4 **track ≤ movement-vector baseline**; P2/P3 weak image classifiers;
   tabular classifier **NOT_RUN** (lightgbm absent).

## Next steps (prioritised)
- **P0:** (none blocking) — the system is honest and functional as-is.
- **P1:** add a backend image-ingest endpoint + wire `SatelliteViewer` to POST the
  user's image (closes the biggest workflow gap); add a history editor UI.
- **P2:** install `lightgbm` and re-verify the tabular classifier if its claims are
  to be shown; consider NIO-only latitude guard upstream; address the Vite
  chunk-size warning; add a non-flaky rate-limit around cold model loads.
