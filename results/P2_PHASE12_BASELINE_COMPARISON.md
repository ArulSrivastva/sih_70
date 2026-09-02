# P2 Phase 12 — Locked Baseline vs Candidate Comparison Report

**Date**: September 1, 2026  
**Auditor**: Independent Performance Verification Committee

---

## 1. Metric Comparison Matrix (Held-Out Test Set $N=30$)

| Benchmark / Model | Category Test Accuracy | Category Test Macro-F1 | 95% Bootstrap CI (Macro-F1) | Cohen's $\kappa$ | Verdict |
|---|---|---|---|---|---|
| **Locked Baseline** (`models/detection/model_weights.pt`) | 33.33% | 0.1465 | $[0.051, 0.284]$ | 0.082 | Historical Reference |
| **Phase 12 Candidate** (`models/detection/model_weights_phase9_E9_2.pt`) | **56.67%** | **0.5543** | **$[0.3458, 0.7381]$** | **0.380** | **Qualified Candidate** |

---

## 2. Conclusion
The candidate model delivers a near $4\times$ improvement in Macro-F1 with non-overlapping confidence intervals.
Visualized in [`results/figures/p2/phase12/baseline_vs_candidate.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase12/baseline_vs_candidate.png).
