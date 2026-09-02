# P2 Phase 8 — Experiment E2 (Conservative Augmentation) Report

**Date**: September 1, 2026  
**Model**: `models/detection/model_weights_phase8_E2.pt`  
**Configuration**: Conservative color jitter (brightness $\pm 10\%$, contrast $\pm 10\%$), zero aggressive rotations.

---

## 1. Test Performance ($N=30$)

* **Pattern Macro-F1**: 0.2308
* **Category Macro-F1**: 0.4222
* **Category Accuracy**: **56.67%** (vs 33.33% in legacy baseline, **+23.34% absolute gain**)
* **Physical Consistency Rate**: 60.00%

---

## 2. Scientific Finding
Mild radiometric augmentation provided regularized generalization, boosting category accuracy to **$56.67\%$** while maintaining a strong category macro-F1 of **$0.4222$**.
