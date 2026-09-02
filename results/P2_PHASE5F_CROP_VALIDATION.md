# P2 Phase 5F — Cyclone-Centered Cropping & Coordinate Projection Report

**Date**: September 1, 2026  
**Scope**: Geographic-to-pixel coordinate projection and $224 \times 224$ synoptic vortex center cropping.

---

## 1. Elimination of Mock Bounding Box

* **Critical Audit Rule**: The legacy column `mock_bbox = [420, 190, 600, 370]` was a static interface placeholder.
* **New Ground-Truth Provenance**:
  * Real geographic coordinates $(\text{Latitude}, \text{Longitude})$ are extracted directly from official **IBTrACS / IMD Best Track** records for every satellite frame timestamp.
  * In the Mercator South Asia grid ($44.5^\circ\text{E}–105.5^\circ\text{E}, 10^\circ\text{S}–45.5^\circ\text{N}$ at $4\text{ km/pixel}$), the cyclone center coordinates $(x_c, y_c)$ map deterministically to:
    $$x_c = \frac{\text{Longitude} - 44.5^\circ}{105.5^\circ - 44.5^\circ} \times W_{\text{grid}}$$
    $$y_c = \frac{45.5^\circ - \text{Latitude}}{45.5^\circ - (-10.0^\circ)} \times H_{\text{grid}}$$
  * A synoptic bounding box spanning $900 \times 900\text{ km}$ ($\approx 225 \times 225$ native pixels) is extracted around the eye center and resized with high-quality Lanczos interpolation to $224 \times 224$ pixels.

---

## 2. Crop Validation Quality & Visual Evidence

* **Processed Crops**: All 45 pilot frames successfully cropped and saved to `data/p2_mosdac_pilot/processed_crops/`.
* **Diagnostic Figure**: [`results/figures/p2/p2_phase5_pilot_crops.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/p2_phase5_pilot_crops.png) visualizes representative vortex crops across *Fani, Gulab, and BOB 05*, confirming:
  1. Complete capture of the central eye and eyewall convection in mature stages.
  2. Complete inclusion of primary curved spiral rainbands in moderate stages.
  3. Proper framing of displaced low-level circulation centers in sheared depression stages.
  4. Zero edge clipping or coordinate dislocation.

---

## 3. Crop Validation Status
$$\mathbf{CROP\_VALIDATION\_STATUS = PASS}$$
