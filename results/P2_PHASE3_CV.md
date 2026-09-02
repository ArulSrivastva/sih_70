# P2 Phase 3 — 5-Fold Cross-Validation Evaluation Report

**Date**: September 1, 2026  
**Protocol**: 5-Fold Stratified Cross-Validation on Development Set ($N=112$)  
**Random Seed**: 42

---

## 1. Cross-Validation Results by Regularization Strength

| Metric | E0: Baseline ($\lambda=0$) | E1: Weak Phys ($\lambda=0.01$) | E2: Mod Phys ($\lambda=0.05$) | E2 vs E0 Delta | Verdict |
|---|---|---|---|---|---|
| **Pattern Accuracy** | **58.97% ± 2.60%** | **58.97% ± 2.60%** | **58.97% ± 2.60%** | 0.00% | Identical |
| **Pattern Macro-F1** | **0.2767 ± 0.0651** | **0.2767 ± 0.0651** | **0.2767 ± 0.0651** | 0.0000 | Identical |
| **Pattern Weighted-F1**| **0.4552 ± 0.0647** | **0.4552 ± 0.0647** | **0.4552 ± 0.0647** | 0.0000 | Identical |
| **Category Accuracy** | 25.73% ± 9.36% | 26.64% ± 8.79% | **27.55% ± 8.07%** | +1.82% | Within Noise Floor ($\pm 8\%$) |
| **Category Macro-F1** | 0.1422 ± 0.0627 | 0.1473 ± 0.0606 | **0.1502 ± 0.0572** | +0.0080 | Within Noise Floor ($\pm 0.06$) |
| **Category Weighted-F1**| 0.1984 ± 0.1011 | 0.2065 ± 0.0978 | **0.2103 ± 0.0938** | +0.0119 | Within Noise Floor ($\pm 0.09$) |
| **Physical Consistency** | 82.41% ± 20.37% | 83.32% ± 21.02% | **84.19% ± 19.38%** | +1.78% | Slight Improvement |
| **Incompatible Pair Rate**| 14.86% ± 21.01% | 13.95% ± 21.53% | **13.08% ± 19.81%** | -1.78% | Slight Reduction |
| **Mean Physics Loss** | 0.2288 ± 0.0162 | 0.2279 ± 0.0155 | **0.2253 ± 0.0158** | -0.0035 | Monotonic Reduction |

---

## 2. Fold-by-Fold Analysis

| Fold | E0 Cat Macro-F1 | E1 Cat Macro-F1 | E2 Cat Macro-F1 | E0 Phys Rate | E1 Phys Rate | E2 Phys Rate |
|---|---|---|---|---|---|---|
| **Fold 1** | 0.2185 | 0.2185 | 0.2185 | 86.96% | 86.96% | 86.96% |
| **Fold 2** | 0.1429 | 0.1429 | 0.1429 | 95.65% | 95.65% | 95.65% |
| **Fold 3** | 0.0667 | 0.0667 | 0.0667 | 100.0% | 100.0% | 100.0% |
| **Fold 4** | 0.0667 | 0.0897 | 0.1047 | 81.82% | 86.36% | 90.91% |
| **Fold 5** | 0.2163 | 0.2185 | 0.2185 | 47.62% | 47.62% | 47.62% |

### Key Observations:
1. Across 4 of the 5 folds (Folds 1, 2, 3, 5), the predictions between E0, E1, and E2 are almost completely identical.
2. In Fold 4, the physics loss gently guided 2 borderline predictions away from incompatible states, shifting category macro-F1 from 0.0667 to 0.1047 and consistency from 81.8% to 90.9%.
3. However, aggregate CV category macro-F1 changed by only **+0.0080** (from $0.1422 \to 0.1502$), which is well below the cross-validation standard error margin ($\sigma = \pm 0.057$).
