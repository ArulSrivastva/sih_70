# P2 Phase 8.5 — Dataset Integrity & Split Forensics Audit Report

**Date**: September 1, 2026  
**Auditor**: Independent Machine Learning Audit Pipeline

---

## 1. Dataset Verification Matrix

| Forensic Check | Protocol | Audit Result | Status |
|---|---|---|---|
| **1. Total Genuine Images** | Disk file count | **138 PNG crops** | **PASS** |
| **2. Unique SHA-256 Hashes** | Cryptographic hash audit | **138 / 138 (100.0% Unique)** | **PASS** |
| **3. Synthetic Images** | Image synthesis scan | **0 instances ($0.0\%$)** | **PASS** |
| **4. Duplicated Scenes** | Byte hash duplication | **0 duplicate files** | **PASS** |
| **5. Cross-Split Hash Overlap**| Hash set intersection | **0 shared hashes** | **PASS** |
| **6. Cross-Split Storm Overlap**| Storm ID set intersection | **0 shared storms** | **PASS** |
| **7. Cyclone ID Traceability** | Manifest metadata trace | **100% mapped to IBTrACS** | **PASS** |
| **8. Raw Source Traceability** | MOSDAC source file link | **100% linked to L1C granules** | **PASS** |
| **9. Intensity Category Metadata**| Best Track $V_{\text{max}}$ audit | **100% verified** | **PASS** |
| **10. Structural Pattern Metadata**| Rule derivation disclosure | **100% documented (`DERIVED_RULE`)** | **PASS** |

---

## 2. Partition Distribution Tables

### TRAIN PARTITION ($N=84$ crops, 11 cyclones)
* **Cyclones ($11$)**: *Amphan (2020), Bulbul (2019), Fani (2019), Fengal (2024), Gaja (2018), Gulab (2021), Hudhud (2014), Jawad (2021), Titli (2018), BOB 01 (2018), BOB 01 (2019)*.
* **Category Distribution**: *Very Severe CS: 45, Depression: 31, Deep Depression: 7, Cyclonic Storm: 1*.
* **Pattern Distribution**: *eye_visible: 45, shear_pattern: 38, curved_band: 1*.

### VALIDATION PARTITION ($N=24$ crops, 3 cyclones)
* **Cyclones ($3$)**: *Sitrang (2022), Tauktae (2021), Yaas (2021)*.
* **Category Distribution**: *Very Severe CS: 14, Depression: 10*.
* **Pattern Distribution**: *eye_visible: 14, shear_pattern: 10*.

### TEST PARTITION ($N=30$ crops, 4 cyclones)
* **Cyclones ($4$)**: *Asani (2022), Biparjoy (2023), Kyarr (2019), BOB 05 (2021)*.
* **Category Distribution**: *Very Severe CS: 16, Depression: 11, Deep Depression: 2, Cyclonic Storm: 1*.
* **Pattern Distribution**: *eye_visible: 16, shear_pattern: 14*.

---

## 3. Mathematical Disjointness Proof
$$\text{Train} \cap \text{Val} = \emptyset \quad (0\text{ storm overlaps})$$
$$\text{Train} \cap \text{Test} = \emptyset \quad (0\text{ storm overlaps})$$
$$\text{Val} \cap \text{Test} = \emptyset \quad (0\text{ storm overlaps})$$
$$\text{Hash}(\text{Train}) \cap \text{Hash}(\text{Test}) = \emptyset \quad (0\text{ hash overlaps})$$
