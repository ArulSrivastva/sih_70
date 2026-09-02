# Honesty Audit — every number the dashboard shows, labelled truthfully

Goal: confirm nothing in the integrated product is presented as machine-learning
output when it is not, nothing is silently substituted, and weak/known-limits
components are flagged. Reference: `results/e2e_trace.json` (single real
`/api/analyze` call) + the audited strength evidence in
`../integration_audit_accuracy/`.

## 1. Field-by-field honesty table (live trace)

| Dashboard field | Value (trace) | Real origin | Honest? |
|-----------------|---------------|-------------|---------|
| `detection.detected` | true | P2 model (MobileNet) on reference frame | Yes — real model output on a fixed frame |
| `detection.confidence` | 91 | P2 sigmoid ≥ 0.5 | Yes |
| `detection.location` | 16.52, 82.31 | **latest observed** history (last obs) | Yes (displayed as current position) |
| `detection.movementDirection/Speed` | NE 38° / 21.1 km/h | haversine/bearing of last 2 obs | Yes (geometric, not ML) |
| `classification.category` | "Cyclonic Storm" | P3 ResNet18 softmax | Yes — weak model (see §4) |
| `classification.windSpeedKmh` | 145 | **latest observed** history | Yes; not P3's degenerate regressor |
| `classification.pressureHpa` | 950 | **latest observed** history | Yes |
| `classification.confidence` | 97 | P3 softmax | Yes |
| `classification.structuralPattern` | "eye_visible" | P2 pattern head | Yes (weak, pattern-only) |
| `forecast[+6/+12/+24].windSpeedKmh` | 169.4/160.7/131.6 | **P4 EXP005 GRU** | Yes |
| `forecast[].lat/lon` | 18.39,84.06 … | P4 EXP005 | Yes |
| `forecast[].pressureHpa` | null | no calibrated pressure | Yes (honest null) |
| `forecast[].confidence` | null | no calibrated uncertainty | Yes (honest null) |
| `landfall.*` | est, 18.1,83.65, 54km | **server-side heuristic** | Yes (labelled heuristic) |
| `risk.score/level` | 99 / HIGH | **server-side heuristic** | Yes (labelled heuristic) |
| `wind/pressure/sst/envWind History` | obs replay | validated request history | Yes |
| `confidenceHistory` | all 91.2 (`= P2 det`) | **placeholder constant** | Yes (documented placeholder) |
| `satellite.boundingBox` | null | no audited localizer | Yes (honest null) |
| `satellite.source/label` | frame path | reference frame | Yes |
| `provenance.*` | full block | truthful chain | Yes — now rendered in UI |

## 2. What the integration layer never does
* **No mock results in the live path** — `USE_MOCK=false`; `fetchAnalyze` POSTs
  to `/api/*`; `mockData.js` is reachable only behind the dead `USE_MOCK=true`
  branch.
* **No silent substitution** of the tabular (LightGBM) classifier — reported
  `NOT_RUN/pending`, never faked.
* **No fabricated accuracy** — none of P2/P3/P4's reported numbers are claimed
  to be better than the audited evidence (see BaselineComparison which honestly
  shows EXP005 trailing the movement-vector baseline on track).
* **No relabelling heuristics as ML** — landfall/risk are explicitly
  "HEURISTIC (NOT ML)" in the UI and "deterministic server-side heuristics" in
  provenance.

## 3. Fixes applied for honesty (Step 16, frontend wording)
1. **ProvenancePanel** added and rendered — the backend already emitted a rich
   `provenance` block (sources + honesty notes) but the UI dropped it; now visible.
2. `RiskIndicator`: was labelled "AI Risk Indicator / FORECAST MODEL OUTPUT" →
   now "Deterministic server-side heuristic, not an ML model or an official
   warning"; contributors read live `data` (wind/pressure/coast distance) instead
   of stale hard-coded demo values; "AI-derived risk" removed.
3. `LandfallPanel`: "FORECAST MODEL OUTPUT" → "STATUS: HEURISTIC (NOT ML)".
4. `SatelliteViewer`: "OPERATIONAL" → "REFERENCE FRAME · NOT LIVE"; stale
   "2026-08-26 05:30 UTC" now uses `meta.lastPass`; fake resolution line removed;
   bbox claim corrected ("Not available (no audited localizer)"); upload button
   relabelled "View Local Image (client-side only)".
5. `Header`: "LIVE STREAM" → "INTEGRATED · DEMO HISTORY" (real models, demo input).
6. `PipelineStatus`: "Satellite Stream · INSAT-3D IR acquisition feeds" →
   "Reference Frame (P2/P3)"; removes "live feed" implication.
7. `LiveCycloneStatus`: "Model consensus" → "P2 detection, reference frame".
8. `MapView`: legend "Uncertainty" (a mocked placeholder cone) → "Illustrative
   cone"; added offline-vs-online basemap note.
9. `App.jsx` footer: removed "All figures represent demonstration data" → precise
   statement (reference frame + demo history + EXP005 forecast + heuristic
   landfall/risk; not an IMD warning).

## 4. Known limits (honest, carried forward — do not paper over)
* **P2/P3 are weak classifiers on synthetic/proxy labels, single reference frame.**
  Audited: P3 image acc 38.1%, macro-F1 0.21 (21-frame, leak-prone); P2
  pattern-only. The dashboard displays them because they are real model outputs,
  but ModelIntelligence / provenance do not overstate strength.
* **P4 track accuracy is below the movement-vector baseline** at every horizon —
  shown plainly in BaselineComparison + §2 of MODEL_STRENGTH_SUMMARY.md.
* **Landfall/risk/confidenceHistory are heuristics/placeholders** — disclosed.
* **Tabular classifier NOT_RUN** — disclosed.

## Verdict
**HONEST** — every figure is traceable to a real model output, an observation,
or a disclosed heuristic; nothing is fabricated, and the honest-null + provenance
design was extended so the user can see the truth.
