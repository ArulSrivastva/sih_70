# P2 Phase 11 — Detailed Error Analysis Report

**Date**: September 1, 2026  
**Auditor**: Model Error Diagnostic Team  
**Model**: `models/detection/model_weights_phase11_E11_1.pt`

---

## 1. Systemic Error Breakdown (Total Errors: 13 / 30, 43.3%)

1. **Adjacent Intensity Transitions ($8 / 13 = 61.5\%$)**:
   - `Deep Depression` misclassified as `Depression` ($N=2$).
   - `Cyclonic Storm` misclassified as `Very Severe CS` ($N=1$).
   - `Very Severe CS` misclassified as `Cyclonic Storm` ($N=4$).
2. **Dissipating Sheared Convection ($5 / 13 = 38.5\%$)**:
   - Cyclone systems undergoing rapid environmental dry-air intrusion or vertical wind shear where cloud organization lags surface wind deceleration.

---

## 2. Conclusion
Zero extreme inversions (e.g. `Depression` confused with `Super Cyclone`) were observed across the entire test set.
