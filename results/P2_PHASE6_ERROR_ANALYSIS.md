# P2 Phase 6 — Error Analysis & Minority-Class Recovery Report

**Date**: September 1, 2026  
**Scope**: Per-class granular performance and recovery analysis on held-out test cyclones ($N=340$).

---

## 1. Structural Pattern Per-Class Recovery

| Structural Pattern | Test Support | Baseline F1 | Expanded Candidate F1 | Delta F1 | Recovery Status |
|---|---|---|---|---|---|
| **`shear_pattern`** | 82 | 0.000 | **0.994** | **+0.994** | **Zero-to-Hero Breakthrough** |
| **`curved_band`** | 121 | 0.286 | **0.980** | **+0.694** | **Massive Recovery** |
| **`eye_visible`** | 137 | 0.824 | **0.985** | **+0.161** | **Near-Perfect Discrimination** |

---

## 2. Intensity Category Per-Class Recovery

| Category Class | Test Support | Baseline F1 | Expanded Candidate F1 | Delta F1 | Recovery Status |
|---|---|---|---|---|---|
| **`Depression`** | 52 | 0.000 | **0.782** | **+0.782** | **Zero-to-Hero Breakthrough** |
| **`Cyclonic Storm`** | 50 | 0.435 | **0.642** | **+0.207** | **Substantial Gain** |
| **`Very Severe CS`** | 75 | 0.000 | **0.603** | **+0.603** | **Zero-to-Hero Breakthrough** |
| **`Severe CS`** | 71 | 0.444 | **0.564** | **+0.120** | **Substantial Gain** |
| **`Extremely Severe CS`**| 30 | 0.000 | **0.305** | **+0.305** | **Zero-to-Hero Breakthrough** |
| **`Deep Depression`** | 30 | 0.000 | 0.000 | 0.000 | Boundary overlap with Depression |
| **`Super Cyclonic Storm`**| 32 | 0.000 | 0.000 | 0.000 | Boundary overlap with VSCS/ESCS |

---

## 3. Key Diagnostic Findings

1. **Recovery of Previously Invisible Regimes**:
   - `shear_pattern` was completely unrecognized ($F_1 = 0.000$) in all previous phases due to having only 11 training samples. With 637 training frames, the model now achieves **$0.994\ F_1$** (Precision: $1.000$, Recall: $0.988$).
   - `Depression` jumped from $0.000 \to 0.782\ F_1$.
   - `Very Severe Cyclonic Storm` jumped from $0.000 \to 0.603\ F_1$.
2. **Remaining Challenges**:
   - Boundary overlap between adjacent intensity classes (e.g. `Deep Depression` vs `Depression` where wind speeds differ by only $5\text{ knots}$) remains the primary source of classification errors, reflecting the physical continuity of cyclone intensification.
