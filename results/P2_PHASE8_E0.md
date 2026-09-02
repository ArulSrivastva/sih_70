# P2 Phase 8 — Experiment E0 (Clean Genuine MOSDAC Retraining) Report

**Date**: September 1, 2026  
**Model**: `models/detection/model_weights_phase8_E0.pt`  
**Configuration**: MobileNetV3-small backbone, standard cross-entropy loss, Adam optimizer ($10^{-4}$), batch size 16, deterministic seed 42.

---

## 1. Validation Performance ($N=24$, Storm-Disjoint)

* **Pattern Macro-F1**: 0.3333
* **Category Macro-F1**: 0.2659
* **Category Accuracy**: 37.50%

---

## 2. Held-Out Test Performance ($N=30$, Storm-Disjoint across 4 Cyclones)

* **Pattern Accuracy**: 30.00%
* **Pattern Macro-F1**: 0.2308
* **Pattern Weighted-F1**: 0.3000
* **Category Accuracy**: **43.33%** (vs 33.33% in legacy baseline, **+10.00% absolute gain**)
* **Category Macro-F1**: **0.4276** (vs 0.1465 in legacy baseline, **+0.2811 absolute gain / nearly $3\times$ improvement**)
* **Physical Consistency Rate**: 53.33%

---

## 3. Scientific Finding
When trained strictly on genuine, non-duplicated MOSDAC observations without synthetic template recycling, the model demonstrates genuine feature learning for cyclone intensity classification, achieving a substantial **$+0.2811$ macro-F1 improvement** over the legacy baseline.
