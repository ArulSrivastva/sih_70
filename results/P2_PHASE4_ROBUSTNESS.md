# P2 Phase 4 — Robustness Evaluation Report

**Date**: September 1, 2026  
**Scope**: 6-Point Robustness Audit across all candidates.

---

## 1. Robustness Checklist

| Criterion | Evaluation Requirement | Result | Status |
|---|---|---|---|
| **1. Pattern Macro-F1** | Must not regress materially | Dropped from $0.2767 \to 0.2439$ (A1) and $0.2586$ (F0) | **FAIL** |
| **2. Category Macro-F1** | Genuine improvement over baseline | Moved from $0.1422 \to 0.1502$ (+0.008, within $\pm 0.055$ noise) | **INCONCLUSIVE** |
| **3. Minority Class Integrity** | Not driven purely by majority classes | All rare classes (`shear_pattern`, `Deep Dep`, `Extremely Severe`) stay at $0.0$ F1 | **FAIL** |
| **4. Minority Recovery** | No minority collapse | `curved_band` recall collapsed from $16.7\%$ to $0.0\%$ | **FAIL** |
| **5. Cross-Fold Stability** | Consistent gain across all 5 folds | Inconsistent across individual splits | **FAIL** |
| **6. Zero Leakage** | Strictly fold-isolated data | No test or val data accessed during training | **PASS** |

---

## 2. Verdict
$$\mathbf{ROBUSTNESS\_VERDICT = NO\_ROBUST\_IMPROVEMENT}$$
Neither data augmentation nor backbone freezing strategies provide statistically defensible evidence of outperforming the baseline under cross-validation.
