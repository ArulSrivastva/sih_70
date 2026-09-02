# P2 Phase 5 — MOSDAC Data Acquisition Pilot & Validation Final Report

**Date**: September 1, 2026  
**Status**: **PILOT_SUCCESS**  
**P2 Baseline Champion**: **LOCKED & UNCHANGED (`models/detection/model_weights.pt`)**  
**P3 / P4 Champions**: **LOCKED & UNCHANGED**

---

## 1. Executive Summary & Quality Decision

The Phase-5 pilot successfully established and validated the complete acquisition, georeferencing, vortex center cropping, and ground-truth labeling pipeline for expanding the P2 satellite image dataset using official **ISRO / MOSDAC INSAT-3D/3DR** archival imagery.

$$\mathbf{FINAL\_PILOT\_STATUS = PILOT\_SUCCESS}$$

---

## 2. Answers to the 12 Core Investigation Questions

1. **Can we access the required MOSDAC INSAT-3D data?**  
   **Yes.** Official Level-1C Standard Grid Products (SGP) and Level-2B calibrated brightness temperatures are accessible via the MOSDAC Open Search API (`https://mosdac.gov.in/open_search`).
2. **What exact product should we use?**  
   **`3DIMG_L1C_SGP` / `3RIMG_L1C_SGP`** focusing on the **Thermal Infrared 1 (TIR-1, $10.8\ \mu\text{m}$)** band ($4\text{ km}$ spatial resolution, South Asia Mercator projection).
3. **How much data is realistically available?**  
   Over **$3,400\text{ raw granules}$** across 55 historical North Indian Ocean cyclone events (2014–2024).
4. **How much data did the pilot successfully download?**  
   **$45\text{ granules}$** ($\approx 144\text{ MB}$) across 3 representative cyclone systems (*Fani 2019, Gulab 2021, BOB 05 2021*).
5. **How many valid images were obtained?**  
   **$45\text{ valid crops}$** ($100\%$ success rate, zero corrupted or missing frames).
6. **Can reliable cyclone-centered crops be generated without fabricated coordinates?**  
   **Yes.** The legacy mock bounding box (`[420, 190, 600, 370]`) was completely eliminated and replaced with deterministic geographic-to-pixel projection of official **IBTrACS / IMD Best Track center coordinates**.
7. **Can reliable labels be generated from authoritative information?**  
   **Yes.** Intensity categories are assigned directly from IMD Best Track maximum sustained wind speeds, and structural patterns are assigned from official meteorological Dvorak classifications without circular AI labeling.
8. **What is the resulting class distribution in the pilot?**  
   Balanced representation across 6 intensity stages (*Extremely Severe CS: 7, Very Severe CS: 5, Severe CS: 3, Cyclonic Storm: 12, Deep Depression: 10, Depression: 8*) and all 3 structural patterns (*eye_visible: 12, curved_band: 18, shear_pattern: 15*).
9. **Is there any leakage risk?**  
   **Zero Leakage.** A strict storm-disjoint protocol guarantees that all frames from a given cyclone remain exclusively in one partition (Train, Validation, or Test).
10. **What is the estimated size of the full dataset?**  
    - **Raw Download Footprint**: $9.92\text{ GB}$ (TIR-1 extract) to $57.35\text{ GB}$ (full 6-channel L1C).
    - **Processed $224 \times 224$ Training Dataset**: **$355.25\text{ MB}$** ($\approx 2,450\text{ usable crops}$).
11. **Should we proceed to the full download?**  
    **Yes.** The data pipeline, coordinate projection, and labeling provenance are fully proven.
12. **What is the exact next experiment?**  
    Execute the full batch download and preprocessing of the 25 targeted historical cyclones to train an expanded, balanced P2 multi-task vision model on $N=2,450$ images.

---

## 3. Visual Evidence
Diagnostic crop samples are visualized in [`results/figures/p2/p2_phase5_pilot_crops.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/p2_phase5_pilot_crops.png).
