# P2 Phase 10 — Statistical Uncertainty & Bootstrap Report

**Date**: September 1, 2026  
**Auditor**: Non-Parametric Bootstrap Engine ($10,000$ Resamples, $N=30$)

---

## 1. 95% Confidence Intervals for Phase 9 E9-2

| Metric | Point Estimate | Bootstrap Mean | 95% Confidence Interval |
|---|---|---|---|
| **Category Accuracy** | **56.67%** | 56.68% | **$[40.00\%, 73.33\%]$** |
| **Category Macro-F1** | **0.5543** | 0.5489 | **$[0.3458, 0.7381]$** |
| **Pattern Accuracy** | **30.00%** | 30.01% | **$[13.33\%, 46.67\%]$** |
| **Pattern Macro-F1** | **0.2308** | 0.2305 | **$[0.0833, 0.3636]$** |
| **Cohen's Kappa ($\kappa$)** | **0.3802** | 0.3785 | **$[0.1429, 0.6122]$** |

---

## 2. Statistical Robustness Verdict
$$\mathbf{STATISTICAL\_ROBUSTNESS = ROBUST\_GAIN\_WITH\_EXPECTED\_SMALL\_N\_CI}$$
The lower bound of E9-2's Macro-F1 ($0.3458$) is more than double the legacy baseline ($0.1465$), establishing statistically defensible improvement.
