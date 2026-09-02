# P2 Phase 3 — Final Decision and Experiment Report

**Date**: September 1, 2026  
**Final Decision**: **PHASE1_RETAINED**  
**Official Champion**: **Locked P2 Baseline (`models/detection/model_weights.pt`)**

---

## 1. Summary of Experimental Results

| Metric | E0: Locked Baseline ($\lambda=0$) | E1: Weak Phys ($\lambda=0.01$) | E2: Mod Phys ($\lambda=0.05$) | Best Candidate vs Baseline (Test Delta) |
|---|---|---|---|---|
| **CV Pattern Macro-F1** | 0.2767 ± 0.0651 | 0.2767 ± 0.0651 | 0.2767 ± 0.0651 | 0.0000 |
| **CV Category Macro-F1** | 0.1422 ± 0.0627 | 0.1473 ± 0.0606 | **0.1502 ± 0.0572** | +0.0080 (Within Noise) |
| **CV Physical Consistency**| 82.41% ± 20.37% | 83.32% ± 21.02% | **84.19% ± 19.38%** | +1.78% |
| **Test Pattern Accuracy** | **71.43%** | 66.67% | 66.67% | -4.76% |
| **Test Pattern Macro-F1** | **0.3697** | 0.2667 | 0.2667 | **-0.1030 (Regression)** |
| **Test Category Accuracy**| **33.33%** | 19.05% | 19.05% | **-14.28% (Regression)**|
| **Test Category Macro-F1**| **0.1465** | 0.0533 | 0.0533 | **-0.0932 (Regression)**|
| **Test Phys Consistency** | **19.05%** | 4.76% | 4.76% | -14.29% |

---

## 2. Decision Hierarchy and Scientific Rationale

1. **CV Gate Evaluation**:
   - The physics regularization term ($\lambda_{\text{phys}} = 0.05$) yielded a slight reduction in incompatible prediction rate ($14.86\% \to 13.08\%$) and a nominal increase in category macro-F1 ($0.1422 \to 0.1502$, delta $+0.0080$).
   - However, this $+0.0080$ increase is well within the cross-validation standard deviation noise floor ($\sigma = \pm 0.057$).
2. **Held-Out Test Generalization**:
   - On the single-pass evaluation on the untouched 21 test frames, the physics-regularized models collapsed on `curved_band` and `Severe Cyclonic Storm`, causing test category macro-F1 to decline by **-63.6% relative** ($0.1465 \to 0.0533$).
3. **Formal Verdict**:
   Per Rule 12, because the physics loss did not demonstrate reproducible held-out improvement, **the locked Phase-1 P2 baseline is retained**.

---

## 3. Scientific Takeaway
Explicit Dvorak consistency regularization during multi-task training did not provide sufficient predictive benefit on the available 133-image dataset, demonstrating that data volume and severe class imbalance—rather than lack of physical loss penalties—are the primary constraints on this vision task.
