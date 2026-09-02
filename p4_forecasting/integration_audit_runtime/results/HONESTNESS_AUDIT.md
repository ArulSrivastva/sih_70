# Honesty Audit — truthful labelling of every number the system shows

Source of truth: `results/e2e_trace.json` (single real `/api/analyze` call), the
prior verified accuracy audit (`integration_audit_accuracy/`), and the frontend
honesty fixes already applied and verified.

Purpose: confirm that the integrated system **never** presents a heuristic or
placeholder as ML output, never fabricates accuracy, and clearly separates what
is real model output, what is derived/observed, and what is demo/static.

## 1. Canonical honesty statements (all supported by evidence)

| Statement | Support |
|-----------|---------|
| **P4/EXP005 is the strongest verified forecasting component** | Only component with clean methodology + fully reproduced scores (test track 91.4/119.8/188.2 km @ 6/12/24h; wind MAE 6.7/9.5/16.1). Verified in MODEL_STRENGTH_SUMMARY / CLAIMS_WE_CAN_MAKE. |
| **P4/EXP005 beats Phase-3 LSTM at reported horizons where established** | Verified: champion wins vs phase-3 LSTM (CLAIMS_WE_CAN_MAKE §"VERIFIED BUT SUBJECT TO…"). |
| **P4/EXP005 still loses to movement-vector at every horizon** | Verified: movement-vector 38.1/80.5/180.7 km vs champion 91.4/119.8/188.2 — **superior on track at every horizon**. Permanent caveat. |
| **P2/P3 are real model components but weak, with documented limitations** | P2: pattern acc 0.714 / category acc 0.333 on 21 synthetic-label frames, no presence/bbox. P3 image: acc 38.1%, macro-F1 0.21, leak-prone test. Verified reproduced. |
| **P3 LightGBM / tabular classifier is enabled but weak and not served in-process** | `lightgbm 4.7.0` installed and the tabular model was genuinely run + evaluated via the authoritative Phase-3 pipeline: real test accuracy 0.4700, macro-F1 0.3703, wind MAE 18.84 (train 0.972 overfit; 0 SID overlap). It is **weak** (reported as weak, never inflated) and the live API still reports `available:false` because the defining module lives only inside the no-extraction zip. Never faked. |
| **Landfall and risk are heuristics, NOT ML predictions** | Backend provenance: "deterministic server-side heuristics, not ML outputs"; UI labels: "STATUS: HEURISTIC (NOT ML)". Verified (e2e provenance + risk_deterministic_formula invariant passes: predicted score 99 = deterministic formula). |
| **Map cone is illustrative, not calibrated uncertainty** | `uncertaintyCone.js` is a mocked placeholder; UI legend now says "Illustrative cone". There is no calibrated uncertainty anywhere (forecast confidence = null). |
| **Dashboard is a research prototype, not an operational warning system** | Footer + provenance + header state this explicitly; not a substitute for IMD warnings. |
| **No fabricated accuracy or confidence claims** | None of the audited numbers exceeds the verified evidence; weak models and below-baseline track are surfaced, not hidden. |

## 2. Real-model vs derived vs static distinction (from e2e response)

| Field | Category | Value (trace) | Source |
|-------|----------|---------------|--------|
| `detection.detected/confidence` | **REAL MODEL OUTPUT** | true / 91 | P2 MobileNet on reference frame (sigmoid ≥0.5) |
| `detection.location` | **DERIVED OUTPUT** | 16.52, 82.31 | latest observed history (not a model localiser) |
| `detection.movementSpeedKmh/Direction` | **DERIVED OUTPUT** | 21.1 / NE 38° | haversine/bearing of last 2 observations |
| `classification.category/confidence` | **REAL MODEL OUTPUT** | "Cyclonic Storm" / 97 | P3 ResNet18 softmax (weak, leak-prone) |
| `classification.windSpeedKmh/pressureHpa` | **DERIVED OUTPUT** | 145 / 950 | latest observed values (NOT P3's degenerate wind regressor) |
| `classification.structuralPattern` | **REAL MODEL OUTPUT** | "eye_visible" | P2 pattern head (weak, pattern-only) |
| `forecast[+6/+12/+24].windSpeedKmh` | **REAL MODEL OUTPUT** | 169.4/160.7/131.6 | P4 EXP005 GRU+Huber |
| `forecast[].lat/lon` | **REAL MODEL OUTPUT** | 18.39,84.06 … | P4 EXP005 |
| `forecast[].pressureHpa` / `.confidence` | **STATIC/HONEST NULL** | null | no calibrated outputs |
| `landfall.*` | **DERIVED HEURISTIC** | est, 18.1,83.65, 54km | server-side heuristic, not ML |
| `risk.score/level` | **DERIVED HEURISTIC** | 99 / HIGH | deterministic server-side formula |
| `wind/pressure/sst/envWind History` | **DERIVED OUTPUT** | obs replay | validated request history |
| `confidenceHistory` | **STATIC PLACEHOLDER** | all 91.2 (= P2 det) | documented placeholder, no calibrated per-step confidence |
| `satellite.boundingBox` | **STATIC/HONEST NULL** | null | no audited localizer |
| `provenance.*` | **PROVENANCE** | full block | truthful chain, rendered in UI |

## 3. What the integration layer never does
* **No mock in the live path** — `USE_MOCK=false`; `fetchAnalyze` POSTs to
  `/api/analyze`; `mockData.js` is reachable only behind the dead `USE_MOCK=true`
  branch.
* **No silent tabular substitution** — reported `NOT_RUN`, never faked.
* **No fabricated accuracy** — all P2/P3/P4 figures stay within the audited
  envelope; the below-baseline track result is shown in `BaselineComparison`.
* **No relabeling heuristics as ML** — landfall/risk and the map cone are
  explicitly heuristic/illustrative.

## 4. Already-applied honesty fixes (verified, within approved scope)
1. `ProvenancePanel.jsx` added & rendered — surfaces backend `provenance.notes`
   + sources that the UI previously dropped.
2. `RiskIndicator` — was "AI Risk Indicator / FORECAST MODEL OUTPUT" → now
   "deterministic server-side heuristic, not an ML model"; contributors read live
   data (wind/pressure/coast distance) instead of stale hard-coded demo values.
3. `LandfallPanel` — "FORECAST MODEL OUTPUT" → "STATUS: HEURISTIC (NOT ML)".
4. `SatelliteViewer` — "OPERATIONAL / LIVE STREAM" removed; uses `meta.lastPass`;
   bbox + resolution honesty; upload relabelled "View Local Image (client-side only)".
5. `Header` — "LIVE STREAM" → "INTEGRATED · DEMO HISTORY".
6. `PipelineStatus` — "Satellite Stream · INSAT-3D IR acquisition feeds" →
   "Reference Frame (P2/P3)".
7. `LiveCycloneStatus` — "Model consensus" → "P2 detection, reference frame".
8. `MapView` — legend "Uncertainty" → "Illustrative cone"; offline/online basemap note.
9. `App.jsx` footer — precise research-prototype statement.

## 5. Remaining honesty-relevant limitations (kept visible, never hidden)
* P2/P3 weak on synthetic/leak-prone labels → their numbers must keep the caveat.
* P4 track below movement-vector baseline → must stay prominent.
* Landfall/risk/confidenceHistory are heuristics/placeholders → already disclosed.
* Tabular: lightgbm now installed and the classifier verified (weak: test acc
  0.47 / macro-F1 0.37); in-process `NOT_RUN` due to no-extraction policy →
  disclosed with a precise reason.
* Demo history is static, and real user images are not yet connected to
  P2/P3 inference (see USER_WORKFLOW_AUDIT.md) → disclosed, not hidden.

## Verdict
**HONEST** — every figure on screen is traceable to a real model output, a
derived observation, or a clearly labelled heuristic/placeholder; no unsupported
accuracy is claimed, and provenance is rendered for the user.

---

## 6. Post-audit honesty verification: user-image + history-editor path — 2026-08-30

A subsequent implementation (Option A) connected real user-uploaded images to
P2/P3 and added a history editor. The honesty rules hold on these **new** paths
(verified by `tests/test_image_upload.py`, a live `/api/image` HTTP call, and the
P0 report).

* **User images are never presented as live/operational.** The `SatelliteViewer`
  labels an upload "User-uploaded satellite image" / "USER IMAGE · BACKEND
  ANALYZED", and the reference frame remains the zero-upload default
  ("REFERENCE FRAME · NOT LIVE").
* **No fabricated geography.** For a user image, `detection.location`,
  `movement*` and `satellite.boundingBox` are honest `null` — a single frame is
  not a localizer and has no motion without a history.
* **No unsupported forecast claims.** `/api/image` explicitly returns no
  forecast/landfall/risk; its notes state these require the validated
  5-observation history via `/api/analyze`. No P4 numbers are invented for an
  image alone.
* **P3 limits stay disclosed.** Only category + softmax confidence are exposed;
  the degenerate P3 wind regressor and hardcoded image pressure are still not;
  `tabular` remains `NOT_RUN`.
* **Weak-model caveat retained.** Notes say P2/P3 are "real model output on
  synthetic-label training and are weak / not a validated localizer".
* **History editor is validated, not trusted.** The `HistoryEditor` only posts a
  client-validated 5×6h history; the server re-validates with the audited
  phase-6 rules and rejects (422 structured errors) anything failing the
  contract — so a user cannot slip an invalid history into P4.

These additions introduce **no** new honesty regressions; they extend the same
"real model output / derived / heuristic / honest null" classification that the
integration layer already enforced.

## 7. P1 continuation honesty verification — 2026-08-30

A subsequent P1 pass installed LightGBM, ran the tabular classifier, and added
the North-Indian-Ocean policy guard (see `P1_FIX_REPORT.md`). Honesty holds on
these additions:

* **Real, weak tabular numbers, not inflated.** The tabular model was evaluated
  on the authoritative multi-source test split; its honest results are
  accuracy 0.4700 / macro-F1 0.3703 / wind MAE 18.84 km/h, with train-overfit
  (0.972) and class-imbalance caveats disclosed. No retraining occurred to
  improve them.
* **The API does not over-claim.** Because the tabular class is only inside the
  no-extraction zip, the live API still reports `available:false` with a
  precise reason (`ModuleNotFoundError: 'src'`) instead of claiming it is
  servable. The model being *independently verifiable* is clearly separated
  from being *served in-process*.
* **NIO guard rejects, never clamps.** Input outside the North Indian Ocean
  (lat 0–30 N, lon 40–100 E degrees-East) is returned as `422 OUT_OF_DOMAIN`
  with a message explicitly "rejected, not clamped". `lat = −40` (historical
  200) and `lon = 120` are both rejected, so out-of-scope geography can never
  silently reach P4.
* **P4 below-movement-vector discipline unchanged** and all P2/P3 image
  caveats remain.

**Verdict (unchanged): HONEST** — every figure on screen remains traceable to a
real model output, a derived observation, or a clearly labelled
heuristic/placeholder/honest-null.

## 8. P1 scientific-hardening honesty verification — 2026-08-30

The P1 hardening continuation (see `reports/P1_HARDENING_REPORT.md`)
independently re-ran the verification steps and added honesty-critical findings
that are kept visible, not hidden:

* **Tabular transitions NOT_RUN → VERIFIED/RUN by genuine reproduction.** The
  stored `multisource_model` metrics were **independently reproduced** in an
  isolated scratch — test accuracy 47.0% (measured 47.0046), macro-F1 0.3703,
  wind MAE 18.84, wind RMSE 27.28, pressure MAE 5.11, pressure RMSE 8.33, with
  0 train/val/test storm-ID overlap. These are exact matches to the stored
  claim, not invented, and remain honestly labelled **weak** (train overfit
  0.972; macro-F1 0.37; class imbalance). No retraining inflated them.
* **The live API still does not over-claim.** In-process `available:false` is
  preserved with the precise no-extraction reason — the distinction between
  *independently verifiable* and *served in-process* is kept explicit.
* **P2/P3 synthetic-label and leakage findings disclosed.** P2 detection labels
  (`cyclone_detected`, `mock_bbox`) are synthetic/constant; evaluation is
  leak-prone with no genuine presence/bounding-box ground truth. These remain
  caveated as weak, non-localizing components.
* **`evaluate.py` silent-hardcoded-fallback hazard (protected, NOT fixed).**
  `src/classification/evaluate.py` `evaluate_image_model()` returns hardcoded
  metrics (`acc=42.86`, `macro_f1=0.38`, `wind_mae=18.5`, `wind_rmse=23.2`)
  when torch/checkpoint are unavailable. This is **not** in the live
  integration path (which runs the real ResNet18 in-memory, no fallback), and
  it lives in protected source, so it is flagged as a reproducibility hazard and
  **not modified**.
* **P4 below-movement-vector discipline unchanged and re-audited as fair.**
  The movement-vector baseline comparison is leakage-free; P4 EXP005 loses to
  it at every horizon (38.1/80.5/180.7 vs 91.4/119.8/188.2 km) and is disclosed,
  never presented as superior.
* **Heuristic/illustrative/uncalibrated caveats unchanged** (landfall/risk
  NON-ML, cone ILLUSTRATIVE, no calibrated uncertainty, research prototype).

**Verdict (unchanged): HONEST** — the hardening pass reproduces real, weak
numbers, keeps the below-baseline and heuristic caveats visible, flags the
protected-source fallback hazard, and fabricates nothing.
