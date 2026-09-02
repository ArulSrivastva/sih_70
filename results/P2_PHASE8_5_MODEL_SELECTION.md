# P2 Phase 8.5 — Model Selection Audit Report

**Date**: September 1, 2026  
**Auditor**: Model Selection Committee

---

## 1. Candidate Comparison: E0 (Clean Baseline) vs E2 (Augmented)

| Metric / Dimension | Phase 8 E0 (Clean) | Phase 8 E2 (Augmented) | Superior Model |
|---|---|---|---|
| **Category Macro-F1** | **0.4276** | 0.4222 | **E0 (+0.0054)** |
| **Category Accuracy** | 43.33% | **56.67%** | **E2 (+13.34%)** |
| **Minority Class Recall** | **85.7% on Weak Systems** | 14.3% on Weak Systems | **E0 (Far superior balance)** |
| **Per-Class F1 Balance** | **Balanced across classes** | Concentrated in majority | **E0** |
| **Methodological Cleanliness**| **Pure genuine satellite data** | Uses jitter augmentation | **E0** |

---

## 2. Selection Recommendation

* **Scientifically Preferred Model**: **`Phase 8 E0 (Clean Baseline)`**.
* **Reasoning**: In meteorological disaster forecasting, detecting weak emerging depressions is vital. E2 gains accuracy primarily by over-predicting the majority class (`Very Severe CS`), whereas E0 maintains balanced sensitivity across all intensity regimes.
* **Promotion Status**: **`DO_NOT_PROMOTE_YET`** (Retain E0 as candidate; keep locked baseline untouched).
