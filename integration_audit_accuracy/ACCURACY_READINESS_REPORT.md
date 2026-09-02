# Accuracy & Readiness Report (HEADLINE)

**Date:** 2026-08-30 · **Method:** read-only, independent recomputation of every reported supervised metric; live API smoke test; source-hash immutability before/after.

## Final verdict table

| Component | Verdict | One-line basis |
|---|---|---|
| P1 Data | **PASS_WITH_WARNINGS** | counts/splits/ERA5 all reproduce; synthetic Kaggler labels; “100% completeness” wording inconsistent (sst missing 1,538) |
| P2 Detection | **PASS_ON_REPORTED_METRICS / NOT_VERIFIABLE_AS_DETECTION** | 8/8 pattern/category metrics reproduce exactly; presence/bbox never scored; labels synthetic |
| P3 Classification | **PARTIAL** | image metrics reproduce exactly; tabular NOT_RUN (no lightgbm); multi-source “advantage” invalid (different populations) |
| P4 Forecasting | **PASS_AS_PIPELINE / WEAK_AS_FORECASTER** | prefixes, test metrics, baselines, selection all reproduced; champion LOSES to movement-vector on track at every horizon; EXP006 registry anomaly |
| P5 Frontend | **NOT_INTEGRATED (mock-only)** | USE_MOCK=true; no live data path; 404 on dashboard target |
| API | **BLOCKED_FOR_DIRECT_WIRING** | phase6 verified live but serves no `/api/*`; schema mismatch |
| Leakage | **P4 CLEAN / IMAGE-SPLIT WARNING** | 22 same-storm cross-split frame pairs between train/val/test |
| **OVERALL** | **NOT READY AS-IS** | real forecasting engine + verified numbers exist, but (a) track skill below baselines, (b) detection/classification claims overstated, (c) frontend↔backend unconnected |

## Highlights (independently reproduced, exact)

- P2 detection metrics: reproduced **0.0 Δ** on all 8 (pattern acc 0.7143, F1 0.6307; category acc 0.3333, F1 0.2305).
- P3 image metrics: reproduced **0.0 Δ** (acc 38.1 %, macro-F1 0.2076, wind MAE 109.81 / RMSE 118.39).
- P4 EXP005 test: reproduced to ≤1e-4 (track 91.43 / 119.76 / 188.24 km @ 6/12/24h; wind MAE 6.68 / 9.46 / 16.11 km/h).
- Baselines: reproduced to machine precision — **movement-vector 38.05 / 80.50 / 180.66 km beats the champion everywhere**; persistence 64.69 / 123.94 / 227.33 beats it at 6h.
- canonical_chrono prefixes: `df70303e…` / `48cf065d…` / `89e9c2e2…` **match**.
- Selection: champion chosen on validation only; test scored once.
- Phase6 live: /health, /model, /forecast, /forecast/compare all 200; malformed input → 422.

## Strongest component

**P4 forecasting pipeline (methodology + churn-resistant numbers)** — and its *wind* forecasts are its only claim that beats a trivial baseline on average. EDGE: honestly reproduced end-to-end; causality, normalization, selection, test-once all clean.

## Weakest component (by claim strength)

**P2 “detection”** — shipped as CycloneDetector + metrics, but with synthetic labels (constant bbox, all-positive), no negatives, no presence score; followed closely by **P3 image (21-frame, leak-prone test)** and the **champion’s track accuracy** (below the movement-vector baseline at all horizons).

## Main blocker to a demo-ready system

**No `/api/analyze` (or equivalent) surface for the frontend**, compounded by a field-shape mismatch and missing detection/classification endpoints; the dashboard is mock-only and the (working, verified) phase6 API is not reachable from it.

## Reported-metric verification summary (all would need independent flag)

| Independent verification | COUNT |
|---|---|
| PASS (difference ≤ tolerance) | 37 |
| NOT_RUN (env dependency) | 4 (P3 tabular) |
| FAIL | 0 |
| Informational (QA counts) | 5 |

## Confidence-scoped limits (what to print next to dashboards)

1. Track forecast panel: append “Reference: movement-vector baseline = 38/81/181 km — the model does not beat it on track.”
2. Detection/Classification panels: append “demonstration labels; real-time detection not validated.”
3. If sst used live: “~28% of master rows lack ERA5 sst; interpolated.”