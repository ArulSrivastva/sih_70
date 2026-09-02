# P2 Phase 9 — Dataset Audit Report

**Date**: September 1, 2026  
**Corpus**: Genuine Clean MOSDAC Dataset ($N=138$)

---

## 1. Split Distribution Table (Storm-Disjoint)

| Split | Images | Cyclones | Included Cyclone Systems |
|---|---|---|---|
| **TRAIN** | **84** | **11** | *Amphan (2020), Bulbul (2019), Fani (2019), Fengal (2024), Gaja (2018), Gulab (2021), Hudhud (2014), Jawad (2021), Titli (2018), BOB 01 (2018), BOB 01 (2019)* |
| **VALIDATION** | **24** | **3** | *Sitrang (2022), Tauktae (2021), Yaas (2021)* |
| **TEST** | **30** | **4** | *Asani (2022), Biparjoy (2023), Kyarr (2019), BOB 05 (2021)* |

---

## 2. Statistical Independence Proof
* $\text{Train} \cap \text{Val} = \emptyset$
* $\text{Train} \cap \text{Test} = \emptyset$
* $\text{Val} \cap \text{Test} = \emptyset$
* Test storms (*Kyarr, Biparjoy, Asani, BOB 05 2021*) remain **100% strictly quarantined**.
* Visualized in [`results/figures/p2/phase9/dataset_distribution.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase9/dataset_distribution.png).
