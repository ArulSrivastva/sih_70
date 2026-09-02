# P2 Phase 2B — Error Analysis & Diagnostic Report

**Date**: September 1, 2026  
**Scope**: Granular per-class failure modes of Dvorak consistency soft-gating mechanism.

---

## 1. Per-Class Structural Pattern Performance

| Pattern Class | Support | Baseline Precision | Baseline Recall | Baseline F1 | Candidate Precision | Candidate Recall | Candidate F1 | Delta F1 |
|---|---|---|---|---|---|---|---|---|
| **`curved_band`** | 38 | **0.6000** | **0.0789** | **0.1395** | 0.0000 | 0.0000 | 0.0000 | **-0.1395** |
| **`eye_visible`** | 64 | 0.5888 | 0.9844 | **0.7368** | 0.5714 | 1.0000 | 0.7273 | -0.0095 |
| **`shear_pattern`**| 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

*Diagnosis*: In the baseline, the model predicted `curved_band` 5 times (3 correct). Under the candidate consistency mechanism, `curved_band` was **never predicted** (0 predictions across all 112 validation samples). The pattern classifier fully collapsed into predicting only `eye_visible`.

---

## 2. Per-Class Intensity Category Performance

| Category Class | Support | Baseline Precision | Baseline Recall | Baseline F1 | Candidate Precision | Candidate Recall | Candidate F1 | Delta F1 |
|---|---|---|---|---|---|---|---|---|
| **`Cyclonic Storm`** | 38 | 0.3889 | 0.1842 | 0.2500 | **0.3492** | **0.5789** | **0.4356** | **+0.1856** |
| **`Deep Depression`** | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **`Depression`** | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **`Extremely Severe CS`**| 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **`Severe CS`** | 30 | **0.2750** | **0.3667** | **0.3143** | 0.2188 | 0.2333 | 0.2258 | **-0.0885** |
| **`Very Severe CS`** | 25 | **0.2037** | **0.4400** | **0.2785** | 0.1765 | 0.1200 | 0.1429 | **-0.1356** |

*Diagnosis*: The candidate mechanism shifted category predictions towards `Cyclonic Storm` (predictions increased from 18 to 63), raising `Cyclonic Storm` recall at the heavy expense of `Severe Cyclonic Storm` (F1 down -28%) and `Very Severe Cyclonic Storm` (F1 down -49%).

---

## 3. Root Cause Analysis

1. **Error Propagation & Overconfidence Cascades**:
   Post-hoc probability gating assumes calibrated, unbiased primary predictors:
   $$\tilde{q}_j \propto q_j \cdot \sum_i p_i C[i, j]$$
   When $p_i$ is severely uncalibrated and biased towards `eye_visible` ($p_{\text{eye}} \approx 1.0$), multiplying by $C[\text{eye}, j]$ forces the category distribution into Severe categories and zeroes out intermediate categories.
2. **Minority Class Invariance**:
   Zero-shot post-hoc gating cannot extract new signal for unlearned minority classes (`shear_pattern`, `Deep Depression`, `Depression`) when the upstream feature extractor produces near-zero logits for those regimes.
