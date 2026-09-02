# P2 Phase 3 — Error Analysis & Minority-Class Diagnostic Report

**Date**: September 1, 2026  
**Scope**: Per-class error distributions, minority-class sensitivity, and failure modes across $\lambda_{\text{phys}} \in \{0.0, 0.01, 0.05\}$.

---

## 1. Structural Pattern Per-Class Breakdown (Test Set, $N=21$)

| Class | Support | E0 Baseline F1 | E1 ($\lambda=0.01$) F1 | E2 ($\lambda=0.05$) F1 | Delta (E2 vs E0) |
|---|---|---|---|---|---|
| **`curved_band`** | 6 | **0.286** | 0.000 | 0.000 | -0.286 |
| **`eye_visible`** | 14 | **0.824** | 0.800 | 0.800 | -0.024 |
| **`shear_pattern`**| 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| **Macro-F1** | 21 | **0.3697** | 0.2667 | 0.2667 | **-0.1030** |

---

## 2. Intensity Category Per-Class Breakdown (Test Set, $N=21$)

| Class | Support | E0 Baseline F1 | E1 ($\lambda=0.01$) F1 | E2 ($\lambda=0.05$) F1 | Delta (E2 vs E0) |
|---|---|---|---|---|---|
| **`Cyclonic Storm`** | 5 | **0.435** | 0.320 | 0.320 | -0.115 |
| **`Severe Cyclonic Storm`** | 6 | **0.444** | 0.000 | 0.000 | -0.444 |
| **`Very Severe CS`** | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| **`Extremely Severe CS`** | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| **`Deep Depression`** | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| **`Depression`** | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| **`Super Cyclonic Storm`** | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| **Macro-F1** | 21 | **0.1465** | 0.0533 | 0.0533 | **-0.0932** |

---

## 3. Key Diagnostic Findings

1. **Failure to Revive Minority Classes**:
   - The minority classes (`shear_pattern`, `Deep Depression`, `Depression`, `Extremely Severe Cyclonic Storm`, `Super Cyclonic Storm`) maintained zero recall and zero F1 across all lambda values.
   - The physics penalty acts on softmax product probabilities, but because the underlying feature representations from the convolutional backbone have no discriminative signal for rare classes (e.g. `Depression` has $N=1$ sample in training), a small training penalty cannot synthesize new structural feature representations.
2. **Slight Benefit on Cross-Validation vs Test Degradation**:
   - On Cross-Validation, the physics loss gently reduced incompatible predictions (from $14.86\% \to 13.08\%$) with a tiny increase in category macro-F1 ($0.1422 \to 0.1502$).
   - However, when trained on the full 93 training images and tested on the 21 held-out test frames, both E1 and E2 collapsed `curved_band` recall and `Severe Cyclonic Storm` recall to 0, reducing test macro-F1 by over 50%.
3. **Scientific Conclusion**:
   - Adding a weak physics consistency regularizer does not solve the fundamental data bottleneck of the 133-image Kaggle dataset.
