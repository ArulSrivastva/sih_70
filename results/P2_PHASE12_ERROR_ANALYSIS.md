# P2 Phase 12 — Detailed Error Analysis Report

**Date**: September 1, 2026  
**Auditor**: Model Error Diagnostic Team  
**Evaluated Checkpoint**: `models/detection/model_weights_phase9_E9_2.pt`

---

## 1. Systemic Error Breakdown (Total Errors: 13 / 30, 43.3%)

1. **Adjacent Intensity Transitions ($8 / 13 = 61.5\%$)**:
   - `Deep Depression` misclassified as `Depression` ($N=2$).
   - `Cyclonic Storm` misclassified as `Very Severe CS` ($N=1$).
   - `Very Severe CS` misclassified as `Cyclonic Storm` ($N=4$).
2. **Dissipating Sheared Convection ($5 / 13 = 38.5\%$)**:
   - Cyclonic systems in weakening stages where high shear disrupts the central convective core before peripheral cyclonic circulation fully subsides.

---

## 2. Conclusion
Zero extreme inversions (e.g. `Depression` misclassified as `Super Cyclone`) were observed.
Visualized in [`results/figures/p2/phase12/representative_failure_cases.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase12/representative_failure_cases.png).
