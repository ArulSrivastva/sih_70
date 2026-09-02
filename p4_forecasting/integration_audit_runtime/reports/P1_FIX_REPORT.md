# P1 FIX REPORT — SIH 2026 PS 26070

**System:** SIH 2026 PS 26070 Tropical Cyclone AI System · VARTHA forecaster UI
**Scope:** Close the verified P1 actions from the integration QA verdict
`READY_WITH_MINOR_LIMITATIONS`, without regressing the completed P0 work:
  * **P1-A** — install/enable LightGBM and properly verify the P3 tabular
    classifier (was `NOT_RUN`).
  * **P1-B** — add the North-Indian-Ocean-only latitude policy guard upstream
    (from the `lat = −40 → 200` contract-design note).
  * Preserve all scientific-honesty caveats (nothing fabricated, nothing weaker
    hidden).
**Date:** 2026-08-30
**Method:** Everything below was **actually executed and verified** — LightGBM
installed and the *authoritative* tabular evaluation was run in an isolated
scratch tree (write-free, outside the project), backend test suites, frontend
lint/build, and live HTTP end-to-end requests against a running uvicorn server.
No model retraining; no P1–P5 / phase6 / dataset / model source changed; no
extraction from `PS70-main.zip`; no fabricated metrics.

---

## 1. P1-A — LightGBM / P3 tabular classifier

### 1.1 Runtime dependency
`lightgbm` was **not installed** (`ModuleNotFoundError: No module named
'lightgbm'`), which made the pickled tabular model impossible to even unpickle
and kept it honestly at `NOT_RUN`. This continuation performed:

```
python -m pip install lightgbm   ->  lightgbm 4.7.0 (narwhals dependency)
python -c "import lightgbm; print(lightgbm.__version__)"  ->  4.7.0
```

Installation is an **environment** change, not a project source change (no
immutability impact).

### 1.2 Authoritative evaluation actually run (write-free, isolated scratch)
The audited Phase-3 tabular artifacts were read from `PS70-main.zip` into a
scratch tree **outside the project** (`.../Temp/opencode/p3verify/PS70`, not
the repo, and never extracted into the repo), and the *existing* Phase-3 model
and dataset were evaluated end-to-end with LightGBM available.

Model: `PS70-main/models/classification/tabular_multisource_model.pkl`
(`MultisourceTabularModel`, `use_lightgbm=True`, `LGBMClassifier`, features
`[lat, lon, sst, pressure_msl, wind_u, wind_v]`, 7 IMD classes).

Real, measured results on the authoritative multi-source splits:

| Split | rows | accuracy | macro-F1 | wind MAE (km/h) |
|-------|-----:|---------:|---------:|----------------:|
| train | 3039 | 0.9720   | 0.9837   | 7.53            |
| val   | 518  | 0.3938   | 0.1899   | 14.92           |
| **test** | **651** | **0.4700** | **0.3703** | **18.84**   |

* Test→train **cyclone-ID overlap: 0** (no track leakage between split and
  train).
* Test true class distribution is strongly imbalanced (Depression ≈ 253/651 ≈
  39%); macro-F1 (0.37) is the honest headline — the model is a **weak**
  classifier, not a strong one.
* Train accuracy (0.972) vs. val (0.394) / test (0.470) shows clear overfitting.
* sklearn `InconsistentVersionWarning` on unpickle (pickled under 1.5.1, loaded
  under 1.7.1) is recorded as a reproducibility caveat; the model still loads
  and predicts.

**The model genuinely runs and its real numbers are weak.** Those numbers are
reported here exactly as measured; nothing was invented and no retraining was
done to inflate them.

### 1.3 Honest integration-layer status (stays `NOT_RUN`, with a precise reason)
The *live* integration API still reports the tabular model as **not available**,
because the pickled class is defined in `src.classification.classifier`, which
lives **only inside** the zip under the strict `zip_store` no-extraction policy
and is therefore not importable in-process. `p3_classifier.py::tabular_status()`
was updated (see §3) to distinguish this precisely. The model genuinely can be
evaluated (proven in §1.2) but **cannot be served in-process by the API without
extracting source**, which is forbidden — so the API does not claim it is
servable. This is the honest end state, not a fabrication:

```
{ 'available': False,
  'reason': "pickle needs defining module importable in-process "
            "(no-extraction policy keeps it in the zip only): No module named 'src'" }
```

---

## 2. P1-B — North Indian Ocean (NIO) upstream policy guard

### 2.1 Domain (authoritative, not invented)
The IMD/RSMC New Delhi mandate and the audited Phase-1 IBTrACS NI-basin
extraction live inside the box:

* **latitude** 0°N – 30°N  (`NIO_LAT_RANGE = (0.0, 30.0)`)
* **longitude** 40°E – 100°E in the project's canonical degrees-East
  `[0, 360)` convention  (`NIO_LON_RANGE = (40.0, 100.0)`)

### 2.2 Where and why
The phase-6 layer only enforces global physical bounds
(`LAT_RANGE = (-90, 90)`, `LON_RANGE = (0, 360)`), so an observation such as
`lat = −40` passes phase-6 (the historical `lat = −40 → 200` property) yet is
geophysically well outside the North Indian Ocean. This continuation adds an
**upstream** NIO-only policy guard in the integration layer
(`integration_api/schemas.py::AnalyzeRequest` model validator,
`integration_api/config.py` constants) so that any observation falling outside
the NIO box is rejected `422` with a dedicated code — **never silently
clamped**.

### 2.3 Behaviour (verified over TestClient and live HTTP)
* `lat = −40` (the historical 200 case) → **422 `OUT_OF_DOMAIN`**
* `lat = −0.001 / 30.001` → 422 `OUT_OF_DOMAIN`; `lat = 0 / 30` (inclusive) → 200
* `lon = 39.999 / 100.001` → 422 `OUT_OF_DOMAIN`; `lon = 40 / 100` → 200
* `lon = 120` (passes phase-6 `[0,360)`, outside NIO) → 422 `OUT_OF_DOMAIN`
* `NaN / ±Inf` lat → 422 `NON_FINITE_VALUE` (phase-6 finite check fires first —
  correct ordering)
* Any of the 5 observations off-box → 422 `OUT_OF_DOMAIN`

Error message is explicit that the request is **rejected, not clamped**:
"…outside the North Indian Ocean domain: lat in [0.0, 30.0], lon in
[40.0, 100.0] (degrees East); rejected, not clamped".

The guard is on `AnalyzeRequest`, which is the shared body of
`/api/analyze`, `/api/detect`, `/api/classify` and `/api/forecast`, so it is
exercised through every request `/api/*` route. The existing valid demo history
(lat ≈ 23.3–23.6, lon ≈ 65.7–68.5) is inside the box and is unaffected.

---

## 3. Files changed (this continuation)

All changes are in **approved `allowed_integration` scope** only. No
P1–P5 / phase6 / dataset / model source file was modified and `PS70-main.zip`
was not touched.

* `integration_api/p3_classifier.py` — `tabular_status()` now reports the
  precise honest reason (`ModuleNotFoundError: 'src'` under the no-extraction
  policy) instead of the terse class name. LightGBM-detection branch retained.
* `integration_api/config.py` — added `NIO_LAT_RANGE`, `NIO_LON_RANGE`,
  `OUT_OF_DOMAIN` constants.
* `integration_api/schemas.py` — added the upstream NIO `model_validator` on
  `AnalyzeRequest` (rejects `OUT_OF_DOMAIN`, never clamps).
* `integration_api/tests/test_nio_domain.py` — NEW; 13 tests covering min/max
  lat & lon boundaries, below/above, `lat −40`, `lon 120`, non-finite ordering,
  multi-observation guard, and the "rejected, not clamped" message.

`analyzer.py` and `routes.py` were changed during **P0** and are unchanged in
this continuation.

---

## 4. Test results (all actually run)

| Suite | Result |
|-------|--------|
| `integration_api` (whole) | **58 passed, 0 failed** (45 prior + 13 new NIO tests) |
| `phase6` regression | **67 passed, 0 failed** |
| Backend total | **125 passed, 0 failed** |
| Frontend `npm run lint` (oxlint) | **PASS** (exit 0; only pre-existing unused-var warnings) |
| Frontend `npm run build` (vite) | **PASS** (653 modules; pre-existing chunk-size warning only) |
| Live HTTP E2E (uvicorn, real server) | **PASS** — `/api/health` 200; `/api/analyze` valid 200 with 3 forecast items; `lat −40` and `lon 120` → 422 `OUT_OF_DOMAIN`; `/api/image` valid PNG → 200 (P0 path intact) |

---

## 5. Scientific honesty audit (decisions and disclosures)

* **No fabricated metrics.** The tabular numbers in §1.2 are real outputs of
  the authoritative eval run in the scratch tree; they are weak and are
  reported as weak (macro-F1 0.37, acc 0.47) with train-overfit and class
  imbalance caveats.
* **The live API still honestly reports `available: False`** for the in-process
  tabular path, because the defining module is not importable without
  violating the no-extraction zip policy. We do **not** claim the model is
  servable in the API; we only report it is *independently verifiable* (weak).
* **P4 track discipline unchanged:** P4 EXP005 stays below the
  movement-vector baseline at every horizon and remains disclosed.
* **Heuristic landfall/risk and the illustrative cone remain labelled NON-ML.**
* **P2/P3 image caveats unchanged** (weak/leak-prone labels; P3 wind regressor
  and hardcoded pressure never exposed).
* No retraining occurred to improve any number. No validation was weakened.
* A constant `pressure = 990` in the P3 image path and the null
  location/bbox/confidence semantics remain exactly as audited.

---

## 6. Immutability result

Verified against `p4_forecasting/integration_api/results/pre_audit_snapshot.json`
(authoritative snapshot, 426 files) — recomputed after this continuation:

* **Snapshot files:** 426, **Missing:** 0
* **Changed total:** 14
* **Changed by bucket:**
  * `allowed_integration`: 5 (P0: `analyzer.py`, `routes.py`, `schemas.py`;
    P1: `config.py`, `p3_classifier.py`; `schemas.py` further extended by P1)
  * `frontend`: 9 (all P0)
* **Forbidden scope (p1/phase1–5/phase6/outside):** 0
* **`PS70-main.zip`:** unchanged (SHA256 matches `f653c8e8…`, 1207 members,
  reference frame + P2/P3 model artifacts byte-identical)

New `test_nio_domain.py` is an addition in approved `allowed_integration`
scope; QA artifacts under `integration_audit_runtime/` are not counted as
application-source changes.

---

## 7. Verdict

**P1_ACTIONS_COMPLETE.** LightGBM is installed and the P3 tabular classifier
was genuinely run and evaluated (weak, honest metrics captured); the NIO-only
latitude guard is in place upstream and verified through `/api/*`; immutability
holds (only approved integration files changed; zip byte-identical); all P0
work is intact.

This remains a **verified, honest research prototype** — not production ready.
The P1 scientific-quality limitations (weak P2/P3 image models, weak tabular
classifier, below-baseline P4 track, heuristic landfall/risk, uncalibrated
outputs) and the in-process `NOT_RUN` tabular status are all still disclosed.
