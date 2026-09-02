# P2 Phase 6 — Raw Data Validation Report

**Date**: September 1, 2026  
**Scope**: Integrity, georeferencing, and spectral validation for 2,120 INSAT-3D/3DR granules.

---

## 1. Validation Matrix

| Check | Protocol | Pass Count | Failure Count | Status |
|---|---|---|---|---|
| **1. File Integrity** | Byte stream & header decoding | 2,120 | 0 | **PASS** |
| **2. Satellite Metadata** | INSAT-3D/3DR Imager verification | 2,120 | 0 | **PASS** |
| **3. Band Verification** | $10.8\ \mu\text{m}$ TIR-1 spectral verification | 2,120 | 0 | **PASS** |
| **4. Timestamp Verification** | Synoptic match with IBTrACS records | 2,120 | 0 | **PASS** |
| **5. Georeferencing** | Grid bounds verification ($44.5^\circ\text{E}–105.5^\circ\text{E}$) | 2,120 | 0 | **PASS** |
| **6. Corruption / Duplication** | SHA-256 hash & timestamp uniqueness | 2,120 | 0 | **PASS** |

---

## 2. Verdict
$$\mathbf{RAW\_DATA\_VALIDATION = PASS}$$
Zero corrupted frames; 100% of raw sensor inputs satisfy operational standards.
