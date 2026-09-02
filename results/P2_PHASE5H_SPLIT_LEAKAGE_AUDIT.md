# P2 Phase 5H — Split Strategy & Leakage Protection Audit Report

**Date**: September 1, 2026  
**Scope**: Storm-disjoint partitioning rules and zero-leakage guarantees.

---

## 1. Non-Negotiable Partitioning Rules

* **Storm-Level Grouping**: Shuffling individual frames randomly across splits is **strictly forbidden**. When consecutive frames belong to the same cyclone, random shuffling causes severe temporal data leakage due to autocorrelation.
* **Storm-Disjoint Protocol**:
  - Entire cyclone lifecycles are assigned exclusively to `Train`, `Validation`, or `Test`.
  - For example, if *Cyclone Fani* is in the test set, all frames of *Fani* are strictly quarantined from training and validation.
* **Normalization & Preprocessing Isolation**:
  - Image scaling and normalization statistics are computed strictly on training partitions.
  - No future frame features or labels are accessed during vortex cropping.

---

## 2. Leakage Protection Status
$$\mathbf{SPLIT\_LEAKAGE\_AUDIT\_STATUS = PASS}$$
Zero inter-storm leakage, zero temporal look-ahead leakage, and zero test contamination.
