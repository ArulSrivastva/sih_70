# P2 Phase 7 — Genuine MOSDAC Data Acquisition & Clean Dataset Final Report

**Date**: September 1, 2026  
**Final Status**: **`GENUINE_DATASET_VALIDATED`**  
**Dataset Quality**: **`PASS`**  
**Pattern Label Status**: **`DERIVED_NOT_INDEPENDENT`**  
**P2 Baseline Champion**: **`UNCHANGED` (`models/detection/model_weights.pt`)**  
**P2 Candidate Champion**: **`UNCHANGED` (`models/detection/model_weights_phase6_candidate.pt`)**

---

## 1. Executive Summary & Forensic Findings

Phase 7 successfully established a completely clean, forensically validated dataset of **138 genuine, 100% unique INSAT-3D satellite observations** across 18 discrete historical cyclone systems.

Every single sample in the Phase-7 dataset satisfies the following strict scientific criteria:
1. **$100\%$ Unique Image Hashes**: Exactly 138 unique SHA-256 hashes for 138 crops (zero synthetic templates, zero parametric replications, zero duplicate scenes).
2. **Zero Mock Coordinates**: Legacy mock bounding box `[420, 190, 600, 370]` is **$0\%$ present**. Center coordinates originate from authoritative **IBTrACS / IMD Best Track records**.
3. **Authoritative Intensity Ground Truth**: Intensity categories are mapped directly from official WMO/IMD maximum sustained wind speeds ($V_{\text{max}}$). Zero circular AI labeling.
4. **Transparent Structural Labeling**: Structural patterns are explicitly documented as `pattern_label_source = DERIVED_RULE` and `pattern_ground_truth_status = NOT_INDEPENDENTLY_ANNOTATED`.
5. **Strict Storm-Disjoint Splitting**:
   - $\text{Train} \cap \text{Val} = \emptyset$
   - $\text{Train} \cap \text{Test} = \emptyset$
   - $\text{Val} \cap \text{Test} = \emptyset$

---

## 2. Dataset Quality Gates (9 / 9 PASS)

| Quality Gate | Requirement | Result | Status |
|---|---|---|---|
| **1. Provenance** | 100% genuine raw sensor observations | 138 / 138 | **PASS** |
| **2. Synthetic Contamination** | 0 synthetic/parametrized samples | 0 instances | **PASS** |
| **3. Mock Coordinates** | 0 uses of mock constant bbox | 0 instances | **PASS** |
| **4. Raw Provenance** | 100% tracked raw file hashes | 138 / 138 | **PASS** |
| **5. Geolocation** | Reproducible Mercator projection | 138 / 138 | **PASS** |
| **6. Intensity Labels** | Authoritative IMD Best Track Vmax | 100% verified | **PASS** |
| **7. Storm Disjointness** | Zero cross-partition storm overlap | 0 overlaps | **PASS** |
| **8. Duplicate Contamination** | Zero cross-split duplicate hashes | 0 duplicates | **PASS** |
| **9. Test Isolation** | Zero test information in training | Fully isolated | **PASS** |

---

## 3. Dataset Comparison (Phase 6 vs Phase 7)

| Metric | Phase 6 (Parametrized Pilot) | Phase 7 (Clean Genuine Dataset) |
|---|---|---|
| **Total Processed Crops** | 2,120 | **138** |
| **Unique Image Hashes** | 414 | **138 (100% Unique)** |
| **Synthetic / Reused Scenes**| 1,706 ($80.5\%$) | **0 (0.0%)** |
| **Unique Cyclones** | 24 | **18** |
| **Storm Disjointness** | Verified | **Verified** |
| **Dataset Quality Status** | Caveats Identified | **PHASE7_DATASET_PASS** |

---

## 4. Immutability & Safety Verification

* **Baseline Weights (`models/detection/model_weights.pt`)**:
  - Start SHA-256: `c296aa21f3e105847878a67abe69390b4a0c566ff31011c08abf78154c2e1971`
  - End SHA-256: `c296aa21f3e105847878a67abe69390b4a0c566ff31011c08abf78154c2e1971`
  - **Verdict**: **BYTE-IDENTICAL & UNCHANGED**
* **Candidate Weights (`models/detection/model_weights_phase6_candidate.pt`)**:
  - **Verdict**: **PRESERVED & UNCHANGED**
* **Model Training in Phase 7**: **STRICTLY PROHIBITED & OMITTED** (Dataset build only).
