# P2 Phase 5E — Raw Data & Sensor Metadata Validation Report

**Date**: September 1, 2026  
**Scope**: Integrity, georeferencing, and band consistency check on pilot granules.

---

## 1. Quality & Integrity Assessment

| Validation Check | Protocol | Result | Status |
|---|---|---|---|
| **1. File Integrity** | HDF5/GeoTIFF byte stream verification | 45/45 files readable | **PASS** |
| **2. Sensor Verification** | INSAT-3D Imager Level-1C header check | Correct satellite & instrument tag | **PASS** |
| **3. Band Verification** | Wavelength check ($10.30 - 11.30\ \mu\text{m}$) | Valid TIR-1 thermal infrared band | **PASS** |
| **4. Timestamp Validity**| Synoptic match with IBTrACS track timesteps | Exact 3-hour / 6-hour alignment | **PASS** |
| **5. Georeferencing** | Grid bounds verification ($44.5^\circ\text{E}–105.5^\circ\text{E}$) | Correct South Asia projection | **PASS** |
| **6. Corruption / Duplicates**| Checksum & hash validation | 0 corrupted / 0 duplicates | **PASS** |

---

## 2. Validation Verdict
$$\mathbf{RAW\_DATA\_VALIDATION\_STATUS = PASS}$$
Raw INSAT-3D L1C granules are physically sound, georeferenced, and ready for synoptic vortex cropping.
