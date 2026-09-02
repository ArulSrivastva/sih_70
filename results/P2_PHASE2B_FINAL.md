# P2 Phase 2B — Final Experiment Evaluation Report

**Date**: September 1, 2026  
**Status**: **EXPERIMENT_CONCLUDED / PHASE1_RETAINED**

---

## 1. Final Test Set Evaluation ($N=21$ Untouched Test Frames)

| Metric | Baseline | Candidate (Dvorak Soft Gating) | Absolute Delta | Relative Delta | Verdict |
|---|---|---|---|---|---|
| **Pattern Accuracy** | **71.43%** | 66.67% | -4.76% | -6.66% | Regression |
| **Pattern Weighted-F1**| **0.6307** | 0.5333 | -0.0974 | -15.44% | Regression |
| **Pattern Macro-F1** | **0.3697** | 0.2667 | -0.1030 | -27.86% | **Severe Regression** |
| **Category Accuracy** | **33.33%** | 23.81% | -9.52% | -28.56% | **Severe Regression** |
| **Category Weighted-F1**| **0.2305** | 0.0952 | -0.1353 | -58.70% | **Severe Regression** |
| **Category Macro-F1** | **0.1465** | 0.0667 | -0.0798 | -54.47% | **Severe Regression** |
| **Physical Consistency** | **19.05%** | 4.76% | -14.29% | -75.01% | **Severe Regression** |

---

## 2. Scientific Findings and Conclusions

1. **Failure of Post-Hoc Domain Consistency Gating**:
   While integrating domain constraints is theoretically appealing, post-hoc soft-gating on top of uncalibrated neural probability distributions failed decisively. The majority-class overconfidence of the pattern head cascaded into the category head, suppressing valid detections and amplifying majority-class mode collapse.
2. **Immutability of Baseline**:
   Per project rules, because the candidate demonstrated clear regression on both 5-fold cross-validation and the held-out test set, the candidate is rejected.
3. **Official Retained Champion**:
   - Baseline `CycloneDetector` (`models/detection/model_weights.pt`).
   - Pattern Macro-F1: **0.3697** (Test).
   - Category Accuracy: **33.33%** (Test).
