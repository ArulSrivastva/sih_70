# P2 Phase 5A — Repository & Architecture Audit Report

**Date**: September 1, 2026  
**Scope**: Codebase audit of dataset loaders, preprocessing pipelines, metadata sources, and API contracts.

---

## 1. Existing Data Loading & Ingestion Pipeline

1. **`src/data/detection_dataset.py`**:
   - Implements `CycloneDetectionDataset` supporting train, validation, and test partition loading.
   - Standard output dictionary schema: `{"image", "detected", "category", "structural_pattern", "filename"}`.
2. **`src/data/preprocess_satellite.py`**:
   - Implements `preprocess_single_frame()` with Lanczos resampling, $[0, 1]$ min-max float normalization, and CHW tensor output.
   - Implements `batch_preprocess_directory()` for standardized batch processing.

---

## 2. Existing Metadata & Track Sources

* **`data/metadata/ibtracs_clean.csv`**: Contains **18,168 synoptic track points** from 1980–2024 with exact columns:
  `['cyclone_id', 'season', 'name', 'basin', 'subbasin', 'nature', 'timestamp', 'latitude', 'longitude', 'wind_speed_kmh', 'pressure_hpa', 'category']`
* **`data/metadata/mosdac_needed_cyclones.csv`**: 55 prioritized cyclone events identified for archival acquisition.
* **`data/metadata/mosdac_priority1_named.csv`**: 19 high-intensity named cyclones (*Amphan, Mocha, Biparjoy, Fani, Remal, Michaung, etc.*).
* **`data/metadata/mosdac_priority2_unnamed.csv`**: 36 depressions and deep depressions for minority-class enrichment.

---

## 3. Operational Ingestion & API Integrity

* **`p4_forecasting/integration_api/p2_detector.py`**:
  - Serves live detection and structural pattern inference for the dashboard.
  - Consumes $224 \times 224 \times 3$ RGB tensors and outputs calibrated presence probability, Dvorak pattern string, and IMD intensity category.
  - Preserved intact without modification.
