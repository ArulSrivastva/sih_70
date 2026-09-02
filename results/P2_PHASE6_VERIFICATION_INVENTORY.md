# P2 Phase 6 — Independent Verification Inventory Report

**Date**: September 1, 2026  
**Auditor**: Independent Model & Data Verification Pipeline

---

## 1. Audit Scope & Inventory of Claimed Artifacts

| Artifact Group | Target Path | Quantity / Size | Immutability Status |
|---|---|---|---|
| **Locked Baseline** | `models/detection/model_weights.pt` | $4.75\text{ MB}$ | **LOCKED & UNMODIFIED** |
| **Candidate Weights** | `models/detection/model_weights_phase6_candidate.pt` | $4.75\text{ MB}$ | Candidate Only |
| **Phase 5 Reports** | `results/P2_PHASE5*` | 18 files ($10\text{ JSON}, 8\text{ MD}$) | Verified Complete |
| **Phase 6 Reports** | `results/P2_PHASE6*` | 22 files ($12\text{ JSON}, 10\text{ MD}$) | Verified Complete |
| **Dataset Manifests** | `data/p2_mosdac_expanded/*.csv` | 4 CSVs ($2,120\text{ total rows}$) | Verified Complete |
| **Image Crops** | `data/p2_mosdac_expanded/crops/*.png` | 2,120 PNG files on disk | Verified Complete |
| **Diagnostic Figures** | `results/figures/p2/phase6/` | 4 PNG figures | Verified Complete |

---

## 2. Verification Integrity Rule
The baseline weights `models/detection/model_weights.pt` have remained strictly byte-identical and unmodified throughout Phase 5 and Phase 6.
