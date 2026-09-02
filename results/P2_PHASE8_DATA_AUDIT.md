# P2 Phase 8 — Pre-Training Dataset Forensic Audit Report

**Date**: September 1, 2026  
**Target Corpus**: Phase-7 Genuine Clean MOSDAC Dataset (`data/p2_phase7_genuine/`)

---

## 1. Forensic Verification Matrix

| Forensic Check | Protocol | Audit Result | Status |
|---|---|---|---|
| **1. Total Image Count** | File count on disk | **138 PNG crops** | **PASS** |
| **2. Unique Scene Hashes** | SHA-256 computation | **138 / 138 Unique Hashes** | **PASS (100% Unique)** |
| **3. Synthetic Contamination** | Image synthesis check | **0 synthetic/parametrized samples** | **PASS (0.0%)** |
| **4. Mock Bounding Boxes** | Legacy coordinate scan | **0 instances of [420, 190, 600, 370]** | **PASS (0.0%)** |
| **5. Geolocation Provenance** | IBTrACS track center matching | **100% verified Best Track positions** | **PASS** |
| **6. Intensity Label Provenance**| Best Track $V_{\text{max}}$ mapping | **Authoritative IMD Scale** | **PASS** |
| **7. Structural Pattern Provenance**| Dvorak annotation origin | **Rule-derived (NOT independently annotated)** | **DOCUMENTED** |
| **8. Storm Disjointness** | Partition intersection check | $\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset$ | **PASS** |

---

## 2. Partition Summary (Storm-Disjoint)

* **Train Partition ($N=84$ crops across 11 cyclones)**:
  * *Amphan (2020), Fani (2019), Hudhud (2014), Titli (2018), Gaja (2018), Bulbul (2019), Gulab (2021), Jawad (2021), Fengal (2024), BOB 01 (2018), BOB 01 (2019)*.
* **Validation Partition ($N=24$ crops across 3 cyclones)**:
  * *Tauktae (2021), Yaas (2021), Sitrang (2022)*.
* **Held-Out Test Partition ($N=30$ crops across 4 cyclones)**:
  * *Kyarr (2019), Biparjoy (2023), Asani (2022), BOB 05 (2021)*.
