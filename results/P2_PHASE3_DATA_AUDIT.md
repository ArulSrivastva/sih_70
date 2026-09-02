# P2 Phase 3 — Data and Physics Regularization Audit Report

**Date**: September 1, 2026  
**Scope**: Physics-Informed Multi-Task Loss Regularization Formulation and Incompatibility Rules.

---

## 1. Audit Context & Architectural Rationale

Following the Phase-2B diagnostic finding (where post-hoc soft-gating cascaded majority-class bias and caused severe regression), **Phase 3** evaluates whether adding a **differentiable, end-to-end physics consistency penalty** during training provides weak regularization without post-hoc gating.

### Total Loss Objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{presence}} + \mathcal{L}_{\text{pattern}} + \mathcal{L}_{\text{category}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}}$$

where:
- $\mathcal{L}_{\text{presence}} = \text{BCEWithLogitsLoss}(z_{\text{pres}}, y_{\text{pres}})$
- $\mathcal{L}_{\text{pattern}} = \text{CrossEntropyLoss}(z_{\text{pat}}, y_{\text{pat}})$
- $\mathcal{L}_{\text{category}} = \text{CrossEntropyLoss}(z_{\text{cat}}, y_{\text{cat}})$
- $\mathcal{L}_{\text{phys}} = \frac{1}{B} \sum_{b=1}^B \sum_{(i, j) \in \text{Incompatible}} \text{Softmax}(z_{\text{pat}})_{b, i} \cdot \text{Softmax}(z_{\text{cat}})_{b, j}$

---

## 2. Documented Incompatible Dvorak Combinations

The physics penalty $\mathcal{L}_{\text{phys}}$ penalizes joint probability mass assigned to combinations that violate fundamental Dvorak meteorological definitions:

1. **`eye_visible` + `Cyclonic Storm`**: An eye feature requires a mature eye wall convective core with sustained winds $\ge 89$ km/h (T-number $\ge 3.5$). At Cyclonic Storm intensity (62–88 km/h), true eye formation is physically impossible.
2. **`eye_visible` + `Deep Depression`**: Physically impossible (winds 50–61 km/h).
3. **`eye_visible` + `Depression`**: Physically impossible (winds 31–49 km/h).
4. **`shear_pattern` + `Extremely Severe Cyclonic Storm`**: A shear pattern involves displaced cloud mass with an exposed low-level center due to strong vertical wind shear ($\ge 30$ kt), physically precluding the maintenance of an intense core ($> 166$ km/h).

---

## 3. Regularization Strength Schedule
To prevent the physics objective from overwhelming supervised gradients, two small regularizing coefficients were evaluated against the baseline:
- **E0**: $\lambda_{\text{phys}} = 0.0$ (Locked Baseline)
- **E1**: $\lambda_{\text{phys}} = 0.01$ (Weak Regularization)
- **E2**: $\lambda_{\text{phys}} = 0.05$ (Moderate Regularization)
