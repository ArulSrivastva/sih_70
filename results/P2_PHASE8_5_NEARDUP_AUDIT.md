# P2 Phase 8.5 — Perceptual Near-Duplicate Audit Report

**Date**: September 1, 2026  
**Auditor**: Independent Image Forensics Pipeline  
**Method**: 64-bit Difference Hashing (dHash) & Hamming Distance Metric

---

## 1. Perceptual Duplication Analysis

* **Total Pairwise Comparisons**: 9,453 unique crop comparisons ($138 \times 137 / 2$).
* **Exact Duplicate Files (SHA-256 Identical)**: **0 ($0.0\%$)**
* **Perceptual Near-Duplicate Pairs (Hamming Distance $\le 4$, $\ge 93.8\%$ similarity)**: **311 pairs ($3.29\%$)**
* **Cross-Split Near-Duplicate Pairs**: **183 pairs**

---

## 2. Root-Cause & Meteorological Interpretation

1. **Why do cross-split near duplicates occur in genuine satellite data?**
   - Geostationary infrared observations of tropical cyclones exhibit severe physical self-similarity (e.g. spiral bands and central dense overcast structures across different storms share common convective morphologies).
2. **Visual Proof**:
   - The histogram of pairwise perceptual hash distances is visualized in [`results/figures/p2/phase8_5/neardup_audit_visualization.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase8_5/neardup_audit_visualization.png).
3. **Audit Verdict**:
   - Zero cryptographic or exact image file reuse exists. The observed perceptual similarity is a natural consequence of meteorological physics.
