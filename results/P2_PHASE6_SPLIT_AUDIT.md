# P2 Phase 6 — Storm-Disjoint Split & Leakage Audit Report

**Date**: September 1, 2026  
**Protocol**: Cyclone-Level Grouping (Zero Intra-Cyclone Leakage).

---

## 1. Cyclone Split Partitioning

* **Training Partition (18 Cyclones, $N=1,500$ frames)**:
  * *Amphan, Fani, Mocha, Hudhud, Titli, Bulbul, Gaja, Michaung, Remal, Dana, Fengal, Gulab, Jawad, Midhili, BOB 01 (2018), BOB 02 (2019), BOB 01 (2019), BOB 02 (2020)*.
* **Validation Partition (3 Cyclones, $N=280$ frames)**:
  * *Tauktae (2021), Yaas (2021), Sitrang (2022)*.
* **Held-Out Test Partition (4 Cyclones, $N=340$ frames)**:
  * *Kyarr (2019 - SuCS), Biparjoy (2023 - ESCS), Asani (2022 - SCS), BOB 05 (2021 - DD)*.

---

## 2. Leakage Protection Verification

1. **Zero Storm Overlap**: No cyclone in the test partition has a single frame present in train or validation.
2. **Untouched Test Quarantine**: The held-out test cyclones (*Kyarr, Biparjoy, Asani, BOB 05*) were evaluated **only once** after model training was complete.
3. **Verdict**: **PASS (Zero Data Leakage)**.
