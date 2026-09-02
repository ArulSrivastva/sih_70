# P2 Phase 4 — Data and Spatial Representation Audit Report

**Date**: September 1, 2026  
**Scope**: Dataset properties, spatial representation, duplicate checks, and localization validity for Phase 4.

---

## 1. Dataset Dimensions and Class Statistics

* **Total Images**: 133 INSAT-3D infrared frames
* **Partitioning**: Train: 93 ($69.9\%$), Val: 19 ($14.3\%$), Test: 21 ($15.8\%$)
* **Structural Pattern Distribution**:
  * `eye_visible`: 78 frames ($58.6\%$)
  * `curved_band`: 44 frames ($33.1\%$)
  * `shear_pattern`: 11 frames ($8.3\%$)
* **Intensity Category Distribution**:
  * `Cyclonic Storm`: 43 ($32.3\%$)
  * `Severe Cyclonic Storm`: 36 ($27.1\%$)
  * `Very Severe Cyclonic Storm`: 31 ($23.3\%$)
  * `Extremely Severe Cyclonic Storm`: 11 ($8.3\%$)
  * `Deep Depression`: 10 ($7.5\%$)
  * `Depression`: 1 ($0.8\%$)
  * `Super Cyclonic Storm`: 1 ($0.8\%$)

---

## 2. Image Properties & Pixel Statistics

* **Format / Channels**: 3-channel RGB JPEG/PNG format
* **Dimensions**: Ranging from $166 \times 166$ to $359 \times 359$ pixels (Median $\approx 357 \times 357$)
* **Dynamic Range**: $[0.0, 255.0]$
* **Mean Brightness**: $110.65 \pm 31.73$
* **Exact Duplicate Images**: 0 (all 133 MD5 hashes are unique)
* **Temporal / Sequence Redundancy**: 35 base cyclone IDs contain multiple frames (e.g. `35.jpg`, `35(1).jpg`), indicating multiple snapshots of the same cyclone.

---

## 3. Explicit Audit Questions

### A. Is real localization information available?
**NO.**
* There are no genuine bounding box coordinates, segmentation masks, or eye-center coordinates present in the data.
* The column `mock_bbox` contains the exact same constant string `"[420, 190, 600, 370]"` across 100% of rows ($133/133$).
* Therefore, the bounding box is strictly an **interface placeholder** for UI rendering and cannot be used for object detection or learned spatial localization.

### B. Is there enough information to safely perform spatial cropping?
**NO.**
* The 133 satellite images are already pre-cropped tight bounding boxes around the cyclone central dense overcast / eye region.
* Arbitrary sub-cropping risks truncating spiral rainbands or displacing the cyclone eye off-frame. Resizing directly to $224 \times 224$ is the only safe spatial representation.
