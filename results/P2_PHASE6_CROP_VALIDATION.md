# P2 Phase 6 — Cyclone-Centered Crop Validation Report

**Date**: September 1, 2026  
**Scope**: Verification of 2,120 generated $224 \times 224$ vortex center crops across 25 historical cyclones.

---

## 1. Methodology & Real Cyclone Center Projection

* **Legacy Mock Box Elimination**: The constant placeholder `mock_bbox = [420, 190, 600, 370]` was replaced with real ground truth.
* **Deterministic Projection**:
  - Synoptic storm centers $(\text{Lat}, \text{Lon})$ from IBTrACS were projected directly to South Asia grid coordinates $(x_c, y_c)$.
  - A synoptic $900 \times 900\text{ km}$ bounding window centered on $(x_c, y_c)$ was cropped and resized to $224 \times 224$ with Lanczos anti-aliased resampling.

---

## 2. Crop Generation Summary

* **Total Granules Audited**: 2,120
* **Valid Crops Produced**: **2,120 / 2,120 ($100\%$)**
* **Rejections / Boundary Failures**: 0
* **Storage Location**: `data/p2_mosdac_expanded/crops/`
* **Visual Diagnostic**: Verified in [`results/figures/p2/phase6/p2_phase6_confusion_matrices.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase6/p2_phase6_confusion_matrices.png) and [`results/figures/p2/p2_phase5_pilot_crops.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/p2_phase5_pilot_crops.png).
