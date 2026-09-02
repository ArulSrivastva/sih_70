# P2 Phase 3 — Physics-Informed Multi-Task Loss Documentation

## Overview
This document specifies the scientific methodology, loss formulations, experimental results, and conclusions of **P2 Phase 3**, evaluating physics-informed multi-task loss regularization for satellite cyclone classification.

---

## 1. Scientific Motivation
In Phase 2B, post-hoc soft-gating using empirical compatibility matrices proved susceptible to cascading errors when upstream classifiers are uncalibrated. Phase 3 investigated whether incorporating an explicit **differentiable physics-consistency loss penalty during backpropagation** could act as a gentle regularizer to align latent representations with Dvorak meteorological constraints without post-hoc probability manipulation.

---

## 2. Model Architecture and Multi-Task Loss

### Backbone and Prediction Heads
The model uses a MobileNetV3-small backbone with three task-specific classification heads:
1. `presence_head`: Binary presence logit ($z_{\text{pres}}$)
2. `pattern_head`: 3-class structural pattern logits ($z_{\text{pat}}$ for `curved_band`, `eye_visible`, `shear_pattern`)
3. `category_head`: 7-class IMD intensity category logits ($z_{\text{cat}}$)

### Total Loss Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{presence}} + \mathcal{L}_{\text{pattern}} + \mathcal{L}_{\text{category}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}}$$

where $\mathcal{L}_{\text{presence}}$ is BCEWithLogitsLoss, $\mathcal{L}_{\text{pattern}}$ and $\mathcal{L}_{\text{category}}$ are CrossEntropyLoss, and $\mathcal{L}_{\text{phys}}$ is the mean over the batch of joint probabilities on physically incompatible Dvorak pairs:
$$\mathcal{L}_{\text{phys}} = \frac{1}{B} \sum_{b=1}^B \sum_{(i, j) \in \text{Incompatible}} P(\text{pattern}=i \mid x_b) \cdot P(\text{category}=j \mid x_b)$$

---

## 3. Incompatible Dvorak Meteorological Combinations

The four incompatible pairs penalized by $\mathcal{L}_{\text{phys}}$ are:
1. `eye_visible` + `Cyclonic Storm` (An eye requires sustained winds $\ge 89$ km/h, physically impossible at 62–88 km/h).
2. `eye_visible` + `Deep Depression` (Physically impossible).
3. `eye_visible` + `Depression` (Physically impossible).
4. `shear_pattern` + `Extremely Severe Cyclonic Storm` (Strong vertical shear physically precludes core intensification $> 166$ km/h).

---

## 4. Evaluated Regularization Strengths
- **E0 (Baseline)**: $\lambda_{\text{phys}} = 0.0$
- **E1 (Weak Regularization)**: $\lambda_{\text{phys}} = 0.01$
- **E2 (Moderate Regularization)**: $\lambda_{\text{phys}} = 0.05$

---

## 5. Experimental Protocol & Leakage Protection
- **Dataset**: 112 development samples (93 train + 19 val) evaluated under Stratified 5-Fold Cross-Validation (`seed=42`).
- **Controlled Setup**: Identical Adam optimizer ($\text{lr}=10^{-4}$), batch size 16, 10 epochs, and transforms (`Resize((224, 224)) + ToTensor()`).
- **Leakage Protection**: All gradients and validation metrics were calculated strictly per-fold without consulting test data. **Leakage: PASS**.

---

## 6. Results and Scientific Findings

1. **Cross-Validation**:
   - Pattern Macro-F1 remained identical at $0.2767 \pm 0.0651$ across all $\lambda$.
   - Category Macro-F1 shifted nominally from $0.1422 \pm 0.0627$ ($\lambda=0$) to $0.1502 \pm 0.0572$ ($\lambda=0.05$). This $+0.0080$ delta is within the cross-validation noise floor ($\sigma = \pm 0.057$).
   - Physical consistency increased slightly from $82.41\% \to 84.19\%$.
2. **Held-Out Test Generalization ($N=21$)**:
   - Pattern Macro-F1 regressed from **0.3697** (baseline) to **0.2667** ($\lambda=0.01, 0.05$).
   - Category Macro-F1 regressed from **0.1465** (baseline) to **0.0533** ($\lambda=0.01, 0.05$).
   - Category Accuracy dropped from **33.33%** to **19.05%**.
3. **Minority Class Impact**:
   - Zero-recall minority classes (`shear_pattern`, `Deep Depression`, `Depression`, `Extremely Severe CS`, `Super CS`) were not improved by the loss penalty.

---

## 7. What We Explicitly Clarify

1. **Physics-Informed Regularization Only**: This method introduces a soft penalty on joint classification outputs; it is **NOT Numerical Weather Prediction (NWP)** and does not solve primitive atmospheric equations.
2. **No Conservation Equations**: No vorticity, mass, or thermodynamic conservation PDEs were numerically integrated.
3. **No Synthetic / Fabricated Data**: No SMOTE, GANs, synthetic oversampling, or fabricated atmospheric grids were used.

---

## 8. Final Decision
$$\mathbf{FINAL\_DECISION = PHASE1\_RETAINED}$$
The Phase-1 baseline model (`models/detection/model_weights.pt`) is retained as the official locked champion.
