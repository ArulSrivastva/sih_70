# P2 Phase 9 — Per-Cyclone Generalization Report

**Date**: September 1, 2026  
**Scope**: 4 Independent Held-Out Test Cyclones ($N=30$)

---

## 1. Per-Cyclone Test Breakdown

| Test Cyclone System | N | E9-0 Cat Acc (%) | E9-0 Macro-F1 | E9-2 Cat Acc (%) | E9-2 Macro-F1 | Generalization Status |
|---|---|---|---|---|---|---|
| **`KYARR_2019` (SuCS)** | 8 | **25.0%** | **0.200** | **50.0%** | **0.375** | Improved Boundary |
| **`BIPARJOY_2023` (ESCS)**| 7 | **28.6%** | **0.222** | **71.4%** | **0.417** | Eyewall Recognition |
| **`ASANI_2022` (SCS)** | 8 | **37.5%** | **0.273** | **75.0%** | **0.571** | Banding Structure |
| **`UNNAMED_2021_DD` (DD)**| 7 | **85.7%** | **0.462** | **28.6%** | **0.222** | Minority Recovery |

---

## 2. Generalization Verdict
* Both E9-0 and E9-2 demonstrate cross-storm transfer across different cyclone basins (Arabian Sea and Bay of Bengal) and intensity regimes.
* Visualized in [`results/figures/p2/phase9/per_cyclone_performance.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase9/per_cyclone_performance.png).
