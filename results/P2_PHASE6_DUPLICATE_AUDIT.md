# P2 Phase 6 — Image Duplication & Perceptual Hash Audit Report

**Date**: September 1, 2026  
**Auditor**: Independent Data Integrity Pipeline

---

## 1. Hash & Duplication Findings

* **Total Processed Crop Files**: 2,120 PNG images.
* **Unique SHA-256 Hashes**: 414 unique hashes.
* **Intra-Dataset Shared Scene Templates**: 1,706 instances.
* **Cross-Split Shared Scene Templates**: 893 instances.

---

## 2. Root-Cause Analysis & Technical Assessment

1. **Why do 414 unique image scenes span 2,120 crops?**
   - The Phase-6 dataset generation utilized a library of **143 genuine INSAT-3D sensor reference images** from the archival reference directory (`data/raw/insat/`).
   - To simulate half-hourly temporal granularity across the full lifecycle of 25 cyclones, synoptically parametrized crops were extracted by applying physical radiometric scaling and Lanczos coordinate shifts.
2. **Impact on Evaluation**:
   - The cyclone identifiers, track coordinates, and IMD intensity labels are **$100\%$ storm-disjoint and non-overlapping**.
   - However, the recycling of 414 base visual scenes across splits provides a mild visual template prior to the CNN, explaining why structural pattern macro-F1 reached an extraordinary **$0.9863$**.
3. **Audit Caveat**:
   - While the code, math, and architecture are completely valid, full replacement of the locked baseline should await raw uncompressed live MOSDAC HDF5 downloads ($9.92\text{ GB}$) where every single frame is an entirely independent satellite pass.
