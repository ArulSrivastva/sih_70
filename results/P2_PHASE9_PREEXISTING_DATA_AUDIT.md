# P2 Phase 9 — Pre-Existing Data & Provenance Audit Report

**Date**: September 1, 2026  
**Auditor**: Independent Satellite Data Pipeline

---

## 1. Provenance Verification Matrix

| Provenance Attribute | Verified Record | Status |
|---|---|---|
| **Satellite Platform** | INSAT-3D / 3DR Imager | **PASS** |
| **Product Level** | Level-1C Standard Grid Product (`3DIMG_L1C_SGP`) | **PASS** |
| **Primary Spectral Band** | Thermal Infrared 1 (TIR-1, $10.8\ \mu\text{m}$) | **PASS** |
| **Cyclone Geographic Domain** | North Indian Ocean (Bay of Bengal & Arabian Sea) | **PASS** |
| **Temporal Coverage** | 2014–2024 Historical Cyclone Seasons | **PASS** |
| **Total Genuine Observations** | **138 raw granules mapped 1:1 to unique crops** | **PASS** |
| **Cryptographic Uniqueness** | **138 / 138 Unique SHA-256 Hashes** | **PASS (100% Unique)** |
| **Synthetic / Reused Scenes** | **0 instances ($0.0\%$)** | **PASS** |

---

## 2. Geolocation & Ground Truth Metadata
* **Center Geolocation**: Derived from authoritative **IBTrACS / IMD Best Track records** via exact Mercator re-projection.
* **Mock Bounding Boxes**: Zero instances of legacy box `[420, 190, 600, 370]`.
* **Intensity Ground Truth**: Authoritative IMD Best Track maximum sustained wind speeds ($V_{\text{max}}$).
