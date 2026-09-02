# P1 HARDENING REPORT — SIH 2026 PS 26070 (Continuation)

**System:** SIH 2026 PS 26070 Tropical Cyclone AI System · VARTHA forecaster UI
**Task:** P1 HARDENING + SCIENTIFIC VALIDATION (continuation from P0)
**Date:** 2026-08-31
**Scope:** All changes confined to approved integration scope. Protected source unchanged.

---

## Executive Summary

All P1 hardening tasks are **complete and independently verified**:

| Task | Status | Evidence |
|------|--------|----------|
| P1-A: LightGBM / P3 Tabular | PASS_WITH_WARNING | lightgbm 4.7.0 installed; metrics reproduced (weak: acc 0.47, macro-F1 0.37) |
| P1-B: NIO Latitude Policy | PASS | 13 tests + live HTTP; OUT_OF_DOMAIN, never clamped |
| P1-C: P4 Forecasting Validation | PASS | 11 tests; EXP005 metrics reproduced; loses to movement-vector baseline |
| P1-D: API Contract Hardening | PASS | Strict validation, deterministic error codes |
| P1-E: Offline Mode Audit | PASS | Offline honest; online tiles degrade gracefully |
| P1-F: Scientific-Honesty Audit | PASS | Every output classified; no fabrication |
| P1-G: Frontend UX Integration | PASS | Complete user journey verified |
| P1-H: Performance Check | PASS | No blocking issues for research prototype |

**Net verdict: READY_WITH_MINOR_LIMITATIONS** — verified, honest research prototype.

---

## P1-A: LightGBM / P3 Tabular Classifier

**Status: PASS_WITH_WARNING**

- lightgbm 4.7.0 installed (env change only)
- P3 tabular metrics independently reproduced: acc 47.0%, macro-F1 0.3703
- Train overfit: 97.2% vs test 47.0%
- 0 SID overlap between splits (no track leakage)
- In-process NOT_RUN preserved (defining module in no-extraction zip)
- See P1_LIGHTGBM_AUDIT.md

## P1-B: North Indian Ocean Policy

**Status: PASS**

- NIO box: lat [0,30], lon [40,100] degrees East
- Implemented upstream on AnalyzeRequest
- 13 tests + live HTTP verified
- Rejected never clamped
- See GEO_SCOPE_AUDIT.md

## P1-C: P4 Forecasting Validation

**Status: PASS**

- EXP005 metrics independently reproduced on feature_dataset/test (198 samples)
- 6h: 91.4 km track, 6.68 km/h wind MAE
- 12h: 119.8 km track, 9.46 km/h wind MAE
- 24h: 188.2 km track, 16.11 km/h wind MAE
- **P4 LOSES to movement-vector baseline at ALL horizons on track**
- P4 beats phase-3 LSTM at all horizons (+30% track)
- Normalization is train-only (verified)
- Champion selected on validation only, test evaluated once
- See P4_FORECAST_VALIDATION.md

## P1-D: API Contract Hardening

**Status: PASS**

- All endpoints validated with strict schemas
- Deterministic error codes for every failure mode
- NaN/Inf/missing/extra fields all handled
- Image upload: type/size/dimension guards
- Path traversal prevented (basename only)
- See API_HARDENING_AUDIT.md

## P1-E: Offline Mode Audit

**Status: PASS**

- All inference runs offline (P2/P3/P4 from local files)
- Online map tiles degrade gracefully
- No false "live" or "real-time" claims
- Reference frame labeled as from zip
- See OFFLINE_AUDIT.md

## P1-F: Scientific-Honesty Audit

**Status: PASS**

- Every output classified: REAL MODEL OUTPUT, DERIVED, OBSERVED, HEURISTIC, STATIC, HONEST NULL, NOT_RUN, ILLUSTRATIVE
- No fabricated metrics, confidence, or forecasts
- Movement-vector baseline superiority disclosed
- "Research prototype" label present
- See SCIENTIFIC_HONESTY_RECHECK.md

## P1-G: Frontend UX Integration

**Status: PASS**

- Complete 18-step user journey verified
- Lint: 0 errors (3 warnings pre-existing)
- Build: PASS
- No runtime errors, no broken state transitions
- Error handling graceful for all failure modes
- See FRONTEND_E2E_REPORT.md

## P1-H: Performance Check

**Status: PASS**

- Cold startup: ~360ms (acceptable)
- /api/analyze: ~63ms (P3 dominates)
- No blocking issues for research prototype
- See PERFORMANCE_AUDIT.md

---

## Regression Results

| Suite | Result |
|-------|--------|
| integration_api tests | **58 passed / 0 failed** |
| phase6 tests | **67 passed / 0 failed** |
| P4 validation tests | **11 passed / 0 failed** |
| Backend total | **136 passed / 0 failed** |
| Frontend lint | **PASS** (0 errors) |
| Frontend build | **PASS** |

---

## Source Immutability

- Snapshot files: 426
- Missing: 0
- Changed: within approved scope only
- Forbidden scope changes: 0
- PS70-main.zip: byte-identical

---

## Remaining Limitations (kept visible)

1. P3 tabular weak (acc 0.47) and not served in-process
2. P2/P3 image models weak on synthetic/leak-prone labels
3. P4 track below movement-vector baseline at all horizons
4. Heuristic landfall/risk — not ML
5. No calibrated forecast uncertainty
6. Forecast cone illustrative, not calibrated
7. Cold bootstrap ~360ms; P3 dominates latency

---

## Verdict

**READY_WITH_MINOR_LIMITATIONS** — all P1 hardening actions verified; integrity
and honesty constraints hold; no fabricated or inflated results.
