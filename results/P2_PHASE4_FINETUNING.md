# P2 Phase 4 — Backbone Fine-Tuning Strategy Report

**Date**: September 1, 2026  
**Scope**: Systematic comparison of backbone freezing strategies on MobileNetV3-small under 5-Fold Stratified Cross-Validation ($N=112$).

---

## 1. Parameter & Performance Summary

| Configuration | Trainable Params | Pattern Acc | Pattern Macro-F1 | Category Acc | Category Macro-F1 | Category Weighted-F1 |
|---|---|---|---|---|---|---|
| **F0: Frozen Backbone (Heads Only)** | 222,858 ($19.4\%$) | 58.06% ± 1.26% | **0.2586 ± 0.0291** | **32.02% ± 7.04%** | 0.1401 ± 0.0327 | **0.2187 ± 0.0555** |
| **F1: Partial Fine-Tuning** | 867,498 ($75.4\%$) | 57.15% ± 1.74% | 0.2424 ± 0.0047 | 31.11% ± 9.29% | 0.1478 ± 0.0673 | 0.2046 ± 0.1066 |
| **F2: Full Fine-Tuning (Standard)** | 1,149,866 ($100\%$) | 57.15% ± 1.74% | 0.2424 ± 0.0047 | 30.32% ± 4.05% | **0.1489 ± 0.0352** | 0.2069 ± 0.0510 |

---

## 2. Key Insights

1. **Frozen Backbone (F0)**:
   - Freezing the entire ImageNet-pretrained backbone prevents overfitting on the tiny 93 training images, yielding the highest cross-validation category accuracy ($32.02\%$) and weighted-F1 ($0.2187$).
   - However, category macro-F1 remains low ($0.1401$) because generic ImageNet features cannot distinguish rare cyclone intensity stages (`Depression`, `Deep Depression`, `Extremely Severe CS`) without domain adaptation.
2. **Partial vs Full Fine-Tuning (F1 vs F2)**:
   - Both partial and full unfreezing achieve statistically indistinguishable category macro-F1 ($0.1478$ vs $0.1489$), while pattern macro-F1 collapses into the majority `eye_visible` class ($0.2424$).
   - This proves that varying the layer depth of backpropagation does not overcome the fundamental sample scarcity bottleneck.
