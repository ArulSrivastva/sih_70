# P2 Phase 2B — 5-Fold Cross-Validation Evaluation Report

**Date**: September 1, 2026  
**Dataset**: 112 development samples (`train_detection.csv` + `val_detection.csv`)  
**Protocol**: 5-Fold Stratified Cross-Validation (`seed=42`, strictly fold-isolated compatibility matrices)

---

## 1. 5-Fold Cross-Validation Summary Table

| Metric | Baseline | Candidate (Dvorak Consistency) | Absolute Delta | Relative Delta | Verdict |
|---|---|---|---|---|---|
| **Pattern Accuracy** | **58.97% ± 2.60%** | 57.15% ± 1.74% | -1.82% | -3.09% | Regression |
| **Pattern Weighted-F1**| **0.4552 ± 0.0647** | 0.4159 ± 0.0207 | -0.0393 | -8.63% | Regression |
| **Pattern Macro-F1** | **0.2767 ± 0.0651** | 0.2424 ± 0.0047 | -0.0343 | -12.40% | **Regression** |
| **Category Accuracy** | 25.73% ± 9.36% | **28.50% ± 9.50%** | +2.77% | +10.77% | Noise margin |
| **Category Weighted-F1**| **0.1984 ± 0.1011** | 0.1751 ± 0.0708 | -0.0233 | -11.74% | Regression |
| **Category Macro-F1** | **0.1422 ± 0.0627** | 0.1179 ± 0.0438 | -0.0243 | -17.09% | **Regression** |
| **Physical Consistency Rate**| **82.41% ± 20.37%**| 44.31% ± 41.97% | -38.10% | -46.23% | **Regression** |

---

## 2. Fold-by-Fold Breakdown

| Fold | Baseline Pat Macro-F1 | Candidate Pat Macro-F1 | Baseline Cat Macro-F1 | Candidate Cat Macro-F1 | Baseline Phys Consistency | Candidate Phys Consistency |
|---|---|---|---|---|---|---|
| **Fold 1** | **0.3697** | 0.2424 | **0.2185** | 0.1706 | **86.96%** | 4.35% |
| **Fold 2** | 0.2424 | 0.2424 | **0.1429** | 0.1282 | 95.65% | 95.65% |
| **Fold 3** | 0.2424 | 0.2424 | 0.0667 | 0.0667 | 100.0% | 100.0% |
| **Fold 4** | **0.2863** | 0.2424 | 0.0667 | 0.0667 | **81.82%** | 18.18% |
| **Fold 5** | 0.2424 | 0.2424 | **0.2163** | 0.1576 | **47.62%** | 3.37% |
| **Mean ± Std** | **0.2767 ± 0.0651** | 0.2424 ± 0.0047 | **0.1422 ± 0.0627** | 0.1179 ± 0.0438 | **82.41% ± 20.37%** | 44.31% ± 41.97% |

---

## 3. CV Gate Assessment
Under the conservative decision framework:
1. Category macro-F1 regressed from 0.1422 to 0.1179 (-17.1%).
2. Pattern macro-F1 regressed from 0.2767 to 0.2424 (-12.4%).
3. Physical consistency declined from 82.41% to 44.31%.

The candidate fails the Cross-Validation gate.
