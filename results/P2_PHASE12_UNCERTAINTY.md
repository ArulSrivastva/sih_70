# P2 Phase 12 — Bootstrap Uncertainty Report

**Date**: September 1, 2026  
**Auditor**: Non-Parametric Bootstrap Engine ($10,000$ Iterations, $N=30$ Test Set)

---

## 1. 95% Confidence Intervals for Phase 12 Candidate (E9-2)

| Metric | Point Estimate | Bootstrap Mean | 95% Confidence Interval |
|---|---|---|---|
| **Category Accuracy** | **56.67%** | 56.68% | **$[40.00\%, 73.33\%]$** |
| **Category Macro-F1** | **0.5543** | 0.5489 | **$[0.3458, 0.7381]$** |
| **Pattern Accuracy** | **30.00%** | 30.01% | **$[13.33\%, 46.67\%]$** |
| **Pattern Macro-F1** | **0.2308** | 0.2305 | **$[0.0833, 0.3636]$** |
| **Cohen's Kappa ($\kappa$)** | **0.3802** | 0.3785 | **$[0.1429, 0.6122]$** |

---

## 2. Statistical Robustness Verdict
$$\mathbf{P2\_PHASE12\_STATISTICAL\_ROBUSTNESS = ROBUST\_GAIN\_CONFIRMED}$$
The lower bound of the candidate's 95% confidence interval for category Macro-F1 ($0.3458$) is substantially greater than the legacy baseline point estimate ($0.1465$).
Visualized in [`results/figures/p2/phase12/uncertainty.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase12/uncertainty.png).
