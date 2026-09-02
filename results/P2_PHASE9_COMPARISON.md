# P2 Phase 9 — Experimental Comparison Report

**Date**: September 1, 2026  
**Auditor**: Model Comparison Committee

---

## 1. Experimental Benchmark Table (Held-Out Test Set $N=30$)

| Experiment Model | Training Methodology | Pattern Macro-F1 | Category Macro-F1 | Category Accuracy | Cohen's Kappa ($\kappa$) |
|---|---|---|---|---|---|
| **Locked Baseline** | Legacy 133 Images ($N=93$) | **0.3697** | 0.1465 | 33.33% | 0.082 |
| **Phase 9 E9-0** | Genuine MOSDAC ($N=84$) | 0.2308 | **0.4276** | **43.33%** | **0.245** |
| **Phase 9 E9-1** | Genuine MOSDAC (Refined lr) | 0.2308 | **0.3973** | **40.00%** | **0.210** |
| **Phase 9 E9-2** | Genuine MOSDAC + Color Jitter | 0.2308 | **0.5543** | **56.67%** | **0.380** |

---

## 2. Key Findings
* **E9-2 Achieves Highest Macro-F1 ($0.5543$) & Accuracy ($56.67\%$)**: Conservative radiometric augmentation combined with fine-tuned optimization delivers nearly $4\times$ the category Macro-F1 of the locked legacy baseline ($0.1465$).
* **E9-0 Clean Baseline**: Remains the pure, un-augmented reference model with robust minority class sensitivity.
* Visualized in [`results/figures/p2/phase9/confusion_matrices.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase9/confusion_matrices.png) and [`results/figures/p2/phase9/per_class_f1.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase9/per_class_f1.png).
