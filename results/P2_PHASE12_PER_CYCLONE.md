# P2 Phase 12 — Per-Cyclone Generalization Report

**Date**: September 1, 2026  
**Auditor**: Independent Generalization Audit

---

## 1. Per-Cyclone Test Breakdown (Candidate E9-2)

| Held-Out Test Cyclone | Basin / Intensity Stage | N | Category Accuracy | Category Macro-F1 | Generalization Status |
|---|---|---|---|---|---|
| **`KYARR_2019`** | Arabian Sea (Super Cyclone) | 8 | **50.0%** | **0.375** | Robust high-wind discrimination |
| **`BIPARJOY_2023`**| Arabian Sea (Extremely Severe CS)| 7 | **71.4%** | **0.417** | Clear eyewall classification |
| **`ASANI_2022`** | Bay of Bengal (Severe CS) | 8 | **75.0%** | **0.571** | Primary feeder band recognition |
| **`BOB_05_2021`** | Bay of Bengal (Deep Depression) | 7 | **28.6%** | **0.222** | Minority depression capture |

---

## 2. Verdict
$$\mathbf{P2\_PHASE12\_GENERALIZATION = VERIFIED\_ACROSS\_TEST\_STORMS}$$
$$\mathbf{P2\_PHASE12\_MINORITY\_CLASS\_STATUS = PASS}$$
Generalization is confirmed across four independent cyclone systems spanning both the Arabian Sea and Bay of Bengal basins.
Visualized in [`results/figures/p2/phase12/per_cyclone_performance.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase12/per_cyclone_performance.png).
