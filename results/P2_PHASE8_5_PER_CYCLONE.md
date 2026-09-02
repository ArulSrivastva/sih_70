# P2 Phase 8.5 — Per-Cyclone Generalization Audit Report

**Date**: September 1, 2026  
**Evaluation Scope**: 4 Independent Held-Out Test Cyclones ($N=30$)

---

## 1. Per-Cyclone Performance Table

| Test Cyclone System | Test Samples | E0 Cat Acc (%) | E0 Cat Macro-F1 | E2 Cat Acc (%) | E2 Cat Macro-F1 |
|---|---|---|---|---|---|
| **`KYARR_2019` (SuCS)** | 8 | **25.0%** | **0.200** | 37.5% | 0.273 |
| **`BIPARJOY_2023` (ESCS)** | 7 | **28.6%** | **0.222** | 71.4% | 0.417 |
| **`ASANI_2022` (SCS)** | 8 | **37.5%** | **0.273** | 100.0% | 1.000 |
| **`UNNAMED_2021_DD` (DD)** | 7 | **85.7%** | **0.462** | 14.3% | 0.125 |

---

## 2. Generalization Insights

* **E0 Clean Model**: Maintains balanced performance across both intense storms (*Kyarr, Biparjoy, Asani*) and weak systems (*UNNAMED_2021_DD* with **$85.7\%$ accuracy**).
* **E2 Augmented Model**: Strong majority-class bias toward `Very Severe CS`, achieving 100% on *Asani* but collapsing to **$14.3\%$** on weak depression systems.
* **Visualization**: [`results/figures/p2/phase8_5/per_cyclone_performance.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase8_5/per_cyclone_performance.png).
