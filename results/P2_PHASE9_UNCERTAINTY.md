# P2 Phase 9 — Statistical Uncertainty & Bootstrap Report

**Date**: September 1, 2026  
**Method**: Non-Parametric Bootstrap ($10,000$ iterations, $N=30$ test cohort)

---

## 1. 95% Bootstrap Confidence Intervals

| Model | Category Accuracy (95% CI) | Category Macro-F1 (95% CI) |
|---|---|---|
| **E9-0 (Clean)** | **$43.33\%$** ($[26.67\%, 60.00\%]$) | **$0.4276$** ($[0.2534, 0.5982]$) |
| **E9-1 (Refined)** | **$40.00\%$** ($[23.33\%, 56.67\%]$) | **$0.3973$** ($[0.2198, 0.5694]$) |
| **E9-2 (Augmented)** | **$56.67\%$** ($[40.00\%, 73.33\%]$) | **$0.5543$** ($[0.3458, 0.7381]$) |

---

## 2. Scientific Conclusion
The lower bound of E9-2's category Macro-F1 95% CI ($0.3458$) is more than double the legacy baseline's point estimate ($0.1465$), demonstrating statistically significant improvement even under small test cohort constraints.
Visualized in [`results/figures/p2/phase9/uncertainty.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase9/uncertainty.png).
