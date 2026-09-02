# P2 Phase 5D — Pilot Data Download & Manifest Report

**Date**: September 1, 2026  
**Status**: **PILOT_MANIFEST_VALIDATED**  
**Storage Directory**: `data/p2_mosdac_pilot/raw/`

---

## 1. Download Manifest Summary

* **Total Granules in Pilot Manifest**: 45 granules
* **Storms Covered**: FANI (15 granules), GULAB (15 granules), BOB 05 2021 (15 granules)
* **Average Granule Size**: $3.2\text{ MB}$ (TIR-1 GeoTIFF / L1C subset extract)
* **Total Pilot Storage Footprint**: $144.0\text{ MB}$
* **Download Integrity Check**: All 45 entries mapped with valid ISO 8601 UTC timestamps, product identifiers (`3DIMG_L1C_SGP`), and SHA-256 verification hashes in [`results/P2_PHASE5D_DOWNLOAD_MANIFEST.json`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/P2_PHASE5D_DOWNLOAD_MANIFEST.json).

---

## 2. Safety Rules Enforced
1. The locked 133-image P2 baseline in `data/processed/detection/` was **not overwritten**.
2. Model weights `models/detection/model_weights.pt` were **not modified**.
3. Pilot data resides strictly inside isolated sandbox directories (`data/p2_mosdac_pilot/`).
