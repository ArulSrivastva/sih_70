# P2 Phase 2 — Controlled Improvement Experiment Report 001

**Experiment ID**: `P2_PHASE2_IMPROVEMENT_001`  
**Date**: September 1, 2026  
**Type**: Single Controlled Ablation  
**Decision**: **NO_IMPROVEMENT / P2_BASELINE_RETAINED**

---

## 1. Objective and Hypothesis

The MobileNetV3-small backbone in the P2 baseline `CycloneDetector` is initialized from ImageNet-1k pre-trained weights, but the baseline input pipeline applied raw $[0, 1]$ floating-point tensors without standard zero-centering or variance scaling:
$$\text{Baseline}: \quad x_{\text{raw}} = \text{ToTensor}(\text{Resize}(I)) \in [0, 1]$$

**Hypothesis**: Applying standard ImageNet normalization:
$$\text{Candidate}: \quad x_{\text{norm}} = \frac{x_{\text{raw}} - \mu}{\sigma}, \quad \mu = [0.485, 0.456, 0.406], \; \sigma = [0.229, 0.224, 0.225]$$
will better match the expected activation distribution of the pre-trained convolutional filters, improving representation quality and downstream multi-head classification.

---

## 2. Experimental Protocol

- **Dataset**: 112 development samples (93 train + 19 val) from `image_only_kaggle`.
- **Validation**: 5-Fold Stratified Cross-Validation (`seed=42`).
- **Controlled Variables**:
  - Architecture: `CycloneDetector` (MobileNetV3-small + 3 heads) identical.
  - Optimizer: Adam ($\text{lr}=10^{-4}$), batch size 16, 10 epochs.
  - Loss: $\mathcal{L} = \mathcal{L}_{\text{presence}} + \mathcal{L}_{\text{pattern}} + \mathcal{L}_{\text{category}}$ identical.
  - Random seed: 42 across all data loading and weight initialization.
- **Ablated Variable**: Input transform (`transforms.Normalize` vs unnormalized).
- **Leakage Protection**: $\mu$ and $\sigma$ are standard ImageNet constants, not computed from validation or test data. **Leakage: PASS**.

---

## 3. Stratified 5-Fold Cross-Validation Results

| Metric | Baseline (Unnormalized) | Candidate (ImageNet Norm) | Absolute Diff | % Diff | Verdict |
|---|---|---|---|---|---|
| **Pattern Accuracy** | **58.97% ± 2.91%** | 57.15% ± 1.94% | -1.82% | -3.09% | Slight Decrease |
| **Pattern Macro-F1** | **0.2767 ± 0.0728** | 0.2424 ± 0.0053 | -0.0343 | -12.4% | Slight Decrease |
| **Pattern Weighted-F1**| **0.4552 ± 0.0723** | 0.4159 ± 0.0231 | -0.0393 | -8.63% | Slight Decrease |
| **Category Accuracy** | 25.73% ± 10.47% | **29.37% ± 5.28%** | +3.64% | +14.1% | Within Noise Floor |
| **Category Macro-F1** | 0.1422 ± 0.0701 | **0.1524 ± 0.0393** | +0.0102 | +7.17% | Within Noise Floor |
| **Category Weighted-F1**| **0.1984 ± 0.1130** | 0.1891 ± 0.0260 | -0.0093 | -4.69% | Within Noise Floor |
| **Physical Consistency** | **82.41% ± 22.77%** | 73.64% ± 42.08% | -8.77% | -10.6% | Within Noise Floor |

---

## 4. Scientific Analysis & Decision

1. **Lack of Robust Improvement**: Normalization improves category accuracy by +3.64% on 5-fold CV, but simultaneously reduces pattern accuracy by -1.82% and pattern macro-F1 by -0.0343. All differences are well within the $\pm 5-10\%$ standard deviation error margin.
2. **Root Cause**: With only 112 development images and severe class imbalance, the network is fundamentally constrained by sample diversity rather than input scale calibration. Mode collapse to majority classes dominates the error profile regardless of input scaling.
3. **Decision**:
   $$\mathbf{DECISION = NO\_ROBUST\_IMPROVEMENT}$$
   $$\mathbf{CHAMPION = P2\_BASELINE\_RETAINED}$$

Per Part 9 of the project rules, the held-out final test set is not repeatedly evaluated or tuned against. The original P2 baseline model and metrics are preserved intact.
