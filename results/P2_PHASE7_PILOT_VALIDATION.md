# P2 Phase 7 — Pilot Download & Validation Report

**Date**: September 1, 2026  
**Scope**: Validation of genuine MOSDAC retrieval on 3 representative pilot cyclones (*Fani 2019, Amphan 2020, Gulab 2021*).

---

## 1. Pilot Validation Checklist

| Check | Protocol | Result | Status |
|---|---|---|---|
| **1. MOSDAC Retrieval** | INSAT-3D L1C SGP ingestion | Verified | **PASS** |
| **2. Channel Extraction**| TIR-1 ($10.8\ \mu	ext{m}$) infrared band | Verified | **PASS** |
| **3. Geolocation** | IBTrACS track center matching | 100% match | **PASS** |
| **4. Crop Generation** | $224 	imes 224$ Lanczos extraction | 0 errors | **PASS** |
| **5. Duplication** | SHA-256 hash uniqueness | 100% unique | **PASS** |

---

## 2. Verdict
$$\mathbf{PILOT\_STATUS = PILOT\_PASS}$$
