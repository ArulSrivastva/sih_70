# P2 — Physics, Domain Knowledge, and AI Integration Documentation

## Overview
This document specifies the architectural distinction between physics-based methods, meteorological domain knowledge, statistical machine learning models, and experimental consistency mechanisms in **P2 (Cyclone Detection & Structural Pattern Recognition)**.

---

## 1. What is Actually Implemented

### Machine Learning Components (Implemented & Locked)
- **Architecture**: `CycloneDetector` with MobileNetV3-small convolutional backbone.
- **Task Heads**:
  1. `presence_head`: Binary presence classification logit ($1$ output).
  2. `pattern_head`: 3-class Dvorak structural pattern prediction (`curved_band`, `eye_visible`, `shear_pattern`).
  3. `category_head`: 7-class IMD cyclone intensity category prediction.
- **Dataset**: 133 INSAT-3D cropped infrared satellite frames.
- **Operational Integration**: Delivered via `p4_forecasting/integration_api/p2_detector.py` for runtime image inference.

---

## 2. Distinction: Physics vs Domain Knowledge vs AI

| Component | Nature | Status | Description |
|---|---|---|---|
| **Image Feature Extraction** | `AI / ML-based` | `IMPLEMENTED` | 576-dimensional latent feature vector learned by convolutional filters from ImageNet pre-training. |
| **Structural Tagging** | `AI / ML-based` | `IMPLEMENTED` | Multi-class cross-entropy classification on top of latent convolutional representations. |
| **Dvorak Structural Hierarchy** | `Domain Knowledge` | `AUDITED` | Meteorological definition relating cloud organization (sheared $\to$ curved band $\to$ eye) to thermodynamic intensity. |
| **Post-Hoc Consistency Gating** | `Domain-Informed AI` | `EVALUATED / REJECTED` | Post-hoc compatibility soft-gating matrix $C[\text{pattern}, \text{category}]$ tested in Phase 2B; rejected due to error cascading on small imbalanced datasets. |
| **Bounding Box** | `Mock Constant` | `PLACEHOLDER` | Fixed coordinates `[420, 190, 600, 370]` conforming to interface schema; **no active physics or learned object detector**. |
| **Presence Confidence** | `AI / ML-based` | `IMPLEMENTED` | Sigmoid probability of presence; trained on all-positive frames ($100\%$ positive). |

---

## 3. What We Do NOT Claim

1. **No Numerical Weather Prediction (NWP)**: P2 does not solve hydrodynamic, thermodynamic, or primitive atmospheric equations.
2. **No Automated Cyclone Localization**: Bounding boxes are static interface placeholders; no YOLO, Faster R-CNN, or physical centroid localization model is active.
3. **No Embedded Physical Conservation Laws**: The neural network loss function ($\mathcal{L} = \mathcal{L}_{\text{presence}} + \mathcal{L}_{\text{pattern}} + \mathcal{L}_{\text{category}}$) contains no vorticity, mass, or thermal energy conservation penalty terms.
4. **No Multi-Level Upper Air Integration**: Delivered ERA5 files contain surface-only single-level data; no 850/200 hPa vertical wind shear or 500 hPa steering vorticity is consumed by P2.

---

## 4. Phase 2B Experimental Conclusion
In Phase 2B, an explicit Dvorak domain-consistency mechanism ($P_{\text{adj}}(\text{cat}_j) \propto P(\text{cat}_j) \cdot \sum_i P(\text{pat}_i) C[i, j]$) was tested under strict 5-fold cross-validation. The experiment demonstrated that post-hoc domain coupling without end-to-end calibrated feature learning amplifies majority-class bias and degrades macro-F1 (regressing from $0.1422 \to 0.1179$ on CV and $0.1465 \to 0.0667$ on test). Consequently, the Phase-1 baseline champion remains locked and retained.
