# P2 Phase 8 — Experiment E1 (Class-Balanced Training) Report

**Date**: September 1, 2026  
**Model**: `models/detection/model_weights_phase8_E1.pt`  
**Configuration**: Inverse class frequency weighting applied to intensity category cross-entropy loss.

---

## 1. Class Loss Weights
* `Cyclonic Storm`: $3.344$
* `Deep Depression`: $0.478$
* `Depression`: $0.098$
* `Very Severe CS`: $0.080$

---

## 2. Test Performance ($N=30$)

* **Pattern Macro-F1**: 0.2308
* **Category Macro-F1**: 0.3302
* **Category Accuracy**: 36.67%
* **Physical Consistency Rate**: 51.67%

---

## 3. Verdict
Class re-weighting over-penalized majority transitions on small genuine samples, yielding a slightly lower category macro-F1 ($0.3302$) than the unweighted clean baseline E0 ($0.4276$).
