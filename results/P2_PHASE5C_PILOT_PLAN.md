# P2 Phase 5C — Small Representative Pilot Plan

**Date**: September 1, 2026  
**Scope**: Selection and design of a 45-frame validation pilot.

---

## 1. Selected Pilot Cyclone Systems

Three cyclone systems were selected to span the complete range of operational intensities and Dvorak structural regimes:

1. **Extremely Severe Cyclonic Storm FANI (2019)**:
   - *Target Classes*: `eye_visible`, `Extremely Severe CS`, `Very Severe CS`.
   - *Sample Size*: 15 chronological frames covering intensification and mature eye wall stages.
2. **Cyclonic Storm GULAB (2021)**:
   - *Target Classes*: `curved_band`, `Cyclonic Storm`, `Severe CS`.
   - *Sample Size*: 15 chronological frames covering asymmetric spiral banding across the Bay of Bengal.
3. **Deep Depression BOB 05 (2021)**:
   - *Target Classes*: `shear_pattern`, `Deep Depression`, `Depression`.
   - *Sample Size*: 15 chronological frames capturing exposed low-level circulation centers under vertical wind shear.

---

## 2. Pilot Objectives & Isolation
* **Total Pilot Scope**: 45 granules ($\approx 144\text{ MB}$ raw volume).
* **Storage Location**: `data/p2_mosdac_pilot/raw/` (completely isolated from the existing 133-image dataset in `data/processed/detection/`).
* **Validation Goal**: Validate raw HDF5/TIR-1 decoding, geographic-to-pixel projection, $224 \times 224$ vortex center cropping, and authoritative IBTrACS label assignment.
