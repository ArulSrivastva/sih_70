# P2 Phase 8.5 — Small-Test-Set Statistical Uncertainty Report

**Date**: September 1, 2026  
**Method**: Non-Parametric Bootstrap ($10,000$ iterations, $N=30$ test cohort)

---

## 1. Bootstrap Confidence Intervals (95% CI)

| Model | Metric | Point Estimate | Bootstrap Mean | 95% Confidence Interval |
|---|---|---|---|---|
| **Phase 8 E0 (Clean)** | **Category Accuracy** | **43.33%** | 43.33% | **$[26.67\%, 60.00\%]$** |
| **Phase 8 E0 (Clean)** | **Category Macro-F1** | **0.4276** | 0.4241 | **$[0.2534, 0.5982]$** |
| **Phase 8 E2 (Augmented)** | **Category Accuracy** | **56.67%** | 56.68% | **$[40.00\%, 73.33\%]$** |
| **Phase 8 E2 (Augmented)** | **Category Macro-F1** | **0.4222** | 0.4285 | **$[0.2857, 0.5833]$** |

---

## 2. Statistical Interpretation
* With $N=30$ test observations, point estimates must be interpreted alongside their confidence bounds ($\pm 16.7\%$ margin of error).
* Visualized in [`results/figures/p2/phase8_5/uncertainty_visualization.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase8_5/uncertainty_visualization.png).
