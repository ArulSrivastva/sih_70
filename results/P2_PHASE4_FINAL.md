# P2 Phase 4 — Final Experiment & Champion Decision Report

**Date**: September 1, 2026  
**Final Decision**: **PHASE1_RETAINED**  
**Official Champion**: **Locked P2 Baseline (`models/detection/model_weights.pt`)**

---

## 1. Final Test Evaluation Summary ($N=21$ Untouched Test Frames)

| Metric | Locked Baseline | Best Candidate (F0 / A1) | Delta (Absolute) | Delta (Relative) | Verdict |
|---|---|---|---|---|---|
| **Pattern Accuracy** | **71.43%** | 66.67% | -4.76% | -6.66% | Regression |
| **Pattern Macro-F1** | **0.3697** | 0.2667 | -0.1030 | -27.86% | **Regression** |
| **Pattern Weighted-F1**| **0.6307** | 0.5333 | -0.0974 | -15.44% | Regression |
| **Category Accuracy** | **33.33%** | 23.81% | -9.52% | -28.56% | **Severe Regression** |
| **Category Macro-F1** | **0.1465** | 0.0641 | -0.0824 | -56.25% | **Severe Regression** |
| **Category Weighted-F1**| **0.2305** | 0.0916 | -0.1389 | -60.26% | **Severe Regression** |

---

## 2. Decision Rationale & Conclusion

1. **Failure of Augmentation and Fine-Tuning on Test Generalization**:
   - Mild spatial translations and color jittering disrupted subtle cloud boundary features.
   - Rotations degraded classification performance because environmental wind shear has directional physical meaning in the North Indian Ocean basin.
   - Freezing or unfreezing backbone blocks did not resolve the fundamental sample size bottleneck of $N=93$ training frames.
2. **Definitive Completion of P2**:
   - Across Phase 1 (Baseline), Phase 2 (Input Normalization), Phase 2B (Dvorak Soft Gating), Phase 3 (Physics Multi-Task Loss), and Phase 4 (Augmentation & Fine-Tuning), the Phase-1 baseline has decisively proven to be the most robust and generalizable model.
   - **P2 is now fully complete, locked, and preserved.**
