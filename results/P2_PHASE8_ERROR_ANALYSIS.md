# P2 Phase 8 — Error Analysis & Minority-Class Report

**Date**: September 1, 2026  
**Scope**: Per-class granular performance breakdown of model E0 on genuine held-out test data ($N=30$).

---

## 1. Per-Class Category Metrics (E0 Clean Baseline)

| Intensity Category | Test Support | Precision | Recall | F1 Score | Status |
|---|---|---|---|---|---|
| **`Very Severe CS`** | 16 | **0.625** | **0.625** | **0.625** | **Strong Primary Recovery** |
| **`Depression`** | 11 | **0.500** | **0.455** | **0.476** | **Genuine Recovery** |
| **`Deep Depression`** | 2 | 0.000 | 0.000 | 0.000 | Severe Small-Sample Support ($N=2$) |
| **`Cyclonic Storm`** | 1 | 0.000 | 0.000 | 0.000 | Single Sample Support ($N=1$) |

---

## 2. Per-Cyclone Test Generalization (E0)

| Cyclone System | Test Crops | Pattern Acc | Category Acc | Category Macro-F1 |
|---|---|---|---|---|
| **`KYARR_2019`** | 8 | 100.0% | **75.0%** | **0.533** |
| **`BIPARJOY_2023`**| 8 | 100.0% | **50.0%** | **0.444** |
| **`ASANI_2022`** | 8 | 100.0% | **37.5%** | **0.364** |
| **`UNNAMED_2021`**| 6 | 0.0% | **33.3%** | **0.333** |

---

## 3. Confusion Matrix Diagnostic
Visualized in [`results/figures/p2/phase8/p2_phase8_confusion_matrices.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase8/p2_phase8_confusion_matrices.png).
