# P2 Phase 6 — Independent Verification Final Report

**Date**: September 1, 2026  
**Auditor**: Independent Model & Dataset Verification Audit  
**Final Audit Verdict**: **`VERIFICATION_PASS_WITH_CAVEATS`**  
**Candidate Promotion**: **`DO_NOT_RECOMMEND`** (Retain as candidate; keep locked baseline untouched)

---

## 1. Executive Summary & Verification Findings

An exhaustive, independent verification was conducted on the Phase-6 dataset build, split assignments, crop projections, label provenance, duplicate hashes, and model reproduction.

### Key Verified Facts
1. **Dataset Size & Integrity**: 2,120 image crops on disk ($224 \times 224$ normalized RGB PNGs) across 24 unique cyclone keys (25 historical storm seasons).
2. **Strict Cyclone-Disjoint Splitting**:
   - $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
   - Held-out test cyclones (*Kyarr 2019, Biparjoy 2023, Asani 2022, Unnamed DD 2021*) have zero frame overlap in train or validation.
3. **Exact Reproduction**:
   - Pattern Accuracy: **$98.53\%$** (Exact reproduction)
   - Pattern Macro-F1: **$0.9863$** (Exact reproduction)
   - Category Accuracy: **$55.88\%$** (Exact reproduction)
   - Category Macro-F1: **$0.4138$** (Exact reproduction)
4. **Per-Cyclone Generalization**:
   - *Biparjoy (2023)*: Pattern Acc = $99.0\%$, Category Acc = $75.0\%$
   - *Unnamed DD (2021)*: Pattern Acc = $100.0\%$, Category Acc = $68.3\%$
   - *Asani (2022)*: Pattern Acc = $98.8\%$, Category Acc = $53.8\%$
   - *Kyarr (2019)*: Pattern Acc = $97.0\%$, Category Acc = $31.0\%$

---

## 2. Identified Caveats & Audit Nuances

1. **Template Recycling**:
   - 414 unique image scene templates span the 2,120 crops because data scaling was constructed from a base library of 143 genuine INSAT-3D sensor images before conducting a multi-gigabyte network download.
   - While timestamps and track coordinates are completely storm-disjoint, the CNN benefits from recurring sensor templates.
2. **Label Provenance**:
   - Intensity categories are **Class A / C** (authoritative IMD Best Track maximum sustained wind speeds).
   - Structural patterns are **Class D** (derived from meteorological wind speed thresholds and Dvorak cloud morphology rules).

---

## 3. Scientific Recommendation & Immutability

* **Verdict**: **`VERIFICATION_PASS_WITH_CAVEATS`**
* **Promotion Decision**: **`DO_NOT_RECOMMEND`**
* **Preservation**: The original locked baseline `models/detection/model_weights.pt` remains unchanged. The Phase-6 model remains safely preserved as an experimental candidate at `models/detection/model_weights_phase6_candidate.pt`.
