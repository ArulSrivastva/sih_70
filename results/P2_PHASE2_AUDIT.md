# P2 Phase 2 — Comprehensive Audit Report

**Namespace**: `P2_PHASE2_AUDIT`  
**Date**: September 1, 2026  
**Status**: **AUDIT_COMPLETE / P2_BASELINE_RETAINED**

---

## 1. Executive Summary & Inventory

This document provides the full, evidence-based audit of **Person 2 (P2: Cyclone Detection, Presence, and Structural Pattern Classification)** within the VARTHA forecasting system.

### Inventory Summary

1. **Intended Scope**: Automated cyclone detection, presence verification, and Dvorak structural pattern tagging from INSAT-3D infrared satellite imagery.
2. **Actual Implementation**:
   - Model: `CycloneDetector` (MobileNetV3-small backbone pretrained on ImageNet-1k with 3 independent MLP heads for binary presence, 3-class pattern, and 7-class IMD category).
   - Preprocessing: `transforms.Resize((224, 224))` + `transforms.ToTensor()`. Inputs are raw $[0, 1]$ floating point values with no ImageNet zero-centering or standard deviation normalization.
   - Code locations: `PS70-main/src/detection/detector.py`, `train.py`, `evaluate.py`, `inference.py`, `src/data/detection_dataset.py`.
3. **Data Source**:
   - Kaggle INSAT-3D cropped imagery (`image_only_kaggle`, 133 total images).
   - Splits: 93 training, 19 validation, 21 testing (`seed=42`, 70/15/15 random split).
   - Ground truth origins:
     - `cyclone_detected`: 100% True (all 133 samples are positive; 0 negative ocean frames). Presence head is degenerate and unscored.
     - `mock_bbox`: Constant `[420, 190, 600, 370]` across all 133 rows.
     - `structural_pattern`: Synthesized by rule (`eye_visible` if "Severe" in category, `curved_band` if "Cyclonic", `shear_pattern` otherwise).
     - `category`: Parsed from Kaggle image filename index (e.g. `25.jpg` $\to$ 25 kt $\to$ `Depression`).
4. **Current Verified Baseline Performance (Test Set $N=21$)**:
   - Pattern Accuracy: **71.43%** (15/21), Weighted F1: **0.6307**, Macro F1: **0.3697**
   - Category Accuracy: **33.33%** (7/21), Weighted F1: **0.2305**, Macro F1: **0.1465**
   - Presence Accuracy: **Unscored / Degenerate (100% positive)**
   - Physical Consistency: **19.05%** (4/21 physically plausible joint predictions)

---

## 2. Data and Physics Audit (22 Key Physical Variables)

Every candidate physical and environmental variable across the workspace was inspected against raw ERA5 NetCDF files (`era5_*.nc`), IBTrACS archives, and dataset files:

| Variable | Classification | Location / Details in Codebase |
|---|---|---|
| **Latitude** | `AVAILABLE` | `master_dataset.csv`, `ibtracs_clean.csv`, `forecasting_sequences.npz` |
| **Longitude** | `AVAILABLE` | `master_dataset.csv`, `ibtracs_clean.csv`, `forecasting_sequences.npz` |
| **Wind Speed** | `AVAILABLE` | 10m maximum sustained winds from IBTrACS/IMD in `master_dataset.csv` |
| **Wind Direction** | `DERIVABLE` | Computed via `atan2(-u, -v)` from 10m ERA5 wind or track displacement |
| **U/V Wind (10m)** | `AVAILABLE` | Single-level 10m surface winds in ERA5 NetCDFs (`u10`, `v10`) |
| **Sea Surface Temp (SST)** | `AVAILABLE` | Surface SST in ERA5 NetCDFs (`sst`); ~28% missing in raw joins |
| **Mean Sea Level Pressure (MSLP)** | `AVAILABLE` | Single-level surface MSLP in ERA5 NetCDFs (`msl` in Pa $\to$ hPa) |
| **Pressure Tendency** | `DERIVABLE` | $\Delta P / \Delta t$ along track from consecutive 3h/6h intervals |
| **Storm Motion (Speed/Heading)** | `DERIVABLE` | Great-circle Haversine displacement and initial bearing along track |
| **Acceleration** | `DERIVABLE` | $\Delta v / \Delta t$ between consecutive displacement steps |
| **Turning Rate** | `DERIVABLE` | Rate of heading angle change $\Delta \theta / \Delta t$ along track |
| **Intensity Tendency** | `DERIVABLE` | $\Delta \text{wind} / \Delta t$ along track |
| **Land / Sea Information** | `DERIVABLE` | Coarse bounding box and ERA5 land mask NaN indicators |
| **Coastline Information** | `DERIVABLE` | Heuristic distance-to-coast vector calculations |
| **Bathymetry / Topography** | `UNAVAILABLE` | No DEM / ocean bathymetry grids present in repository |
| **Ocean Variables (Sub-surface/OHC)** | `UNAVAILABLE` | Only surface SST present; no ocean heat content or thermocline data |
| **Atmospheric Variables (Surface)** | `AVAILABLE` | Surface MSLP, 10m surface u/v wind |
| **Multi-level Pressure Data** | `UNAVAILABLE` | ERA5 NetCDFs contain ONLY surface single-level data (no 850, 500, 200 hPa) |
| **Vertical Wind Shear (200-850 hPa)** | `UNAVAILABLE` | Requires upper-air winds at 200 hPa and 850 hPa; not present in shipped ERA5 |
| **Humidity (Multi-level/RH)** | `UNAVAILABLE` | No atmospheric humidity fields in shipped NetCDFs |
| **Vorticity (850 hPa)** | `UNAVAILABLE` | No upper-air vorticity fields present in shipped NetCDFs |
| **Divergence (200 hPa)** | `UNAVAILABLE` | No upper-air divergence fields present in shipped NetCDFs |
| **Geopotential Height (500 hPa)** | `UNAVAILABLE` | No geopotential height grids present in repository |

---

## 3. Physics Consistency Audit

### Structural Consistency Analysis
In tropical cyclone meteorology (Dvorak Technique):
- An **`eye_visible`** pattern is physically associated with intense systems ($\ge \text{Severe Cyclonic Storm}$, wind $\ge 89$ km/h).
- A **`curved_band`** pattern is physically associated with organized cyclonic systems ($\text{Deep Depression}$ to $\text{Cyclonic Storm}$, wind $62-88$ km/h).
- A **`shear_pattern`** is physically associated with disorganized/developing systems ($\text{Depression}$, wind $< 62$ km/h).

### Baseline Physics Inconsistency Findings
Because the baseline `CycloneDetector` trains the pattern head and category head independently with standard unweighted cross-entropy on 93 imbalanced samples:
1. **Validation Consistency Rate**: **5.26%** (1 / 19 samples). 18 of 19 predictions simultaneously output `(eye_visible, Cyclonic Storm)`—a direct physical contradiction where an eye is predicted for a system too weak to support one.
2. **Test Consistency Rate**: **19.05%** (4 / 21 samples). 17 of 21 test predictions output physically contradictory combinations.
3. **Training Consistency Rate**: **18.28%** (17 / 93 samples).

---

## 4. Ranked List of Improvement Opportunities

1. **Input Transform ImageNet Normalization** (Controlled ablation selected for `IMPROVEMENT_001`):
   - *Change*: Apply standard ImageNet normalization (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
   - *Benefit*: Aligns input distribution with pretrained MobileNetV3 convolutional filters.
2. **Class-Balanced Loss Weighting**:
   - *Change*: Weight cross-entropy loss inversely proportional to class frequency.
   - *Benefit*: Penalizes majority-class mode collapse (`eye_visible` and `Cyclonic Storm`).
3. **Meteorological Hierarchy Coupling Loss**:
   - *Change*: Add a penalty loss $\mathcal{L}_{\text{phys}}$ for physically invalid joint probability mass.
   - *Benefit*: Enforces Dvorak consistency between structural tags and intensity categories.
4. **Domain-Appropriate Image Augmentation**:
   - *Change*: Random horizontal flip and small angle rotations during training.
   - *Benefit*: Reduces memorization on 93 training images.
5. **Unified Multi-Source Environmental Genesis Model**:
   - *Change*: Train tabular pre-genesis model on 5,481 ERA5/IBTrACS rows.
   - *Benefit*: Provides genuine detection capabilities beyond 133 Kaggle image crops.

---

## 5. Audit Conclusion
The delivered P2 baseline was verified, reproduced, and thoroughly audited. Baseline weights, configurations, and evaluation metrics are fully preserved.
