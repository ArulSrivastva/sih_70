# P2 Phase 8.5 — Metric Reproduction Audit Report

**Date**: September 1, 2026  
**Auditor**: Independent Model Evaluation Pipeline

---

## 1. Benchmark Reproduction Results

| Model | Evaluated Dataset | Claimed Cat Acc | Reproduced Cat Acc | Claimed Cat Macro-F1 | Reproduced Cat Macro-F1 | Reproduction Delta | Audit Status |
|---|---|---|---|---|---|---|---|
| **Locked Baseline** | Legacy 133 Images ($N=21$) | 33.33% | **33.33%** | 0.1465 | **0.1465** | $0.0000$ | **EXACT MATCH** |
| **Phase 8 E0** | Genuine MOSDAC ($N=30$) | 43.33% | **43.33%** | 0.4276 | **0.4276** | $0.0000$ | **EXACT MATCH** |
| **Phase 8 E1** | Genuine MOSDAC ($N=30$) | 36.67% | **36.67%** | 0.3302 | **0.3302** | $0.0000$ | **EXACT MATCH** |
| **Phase 8 E2** | Genuine MOSDAC ($N=30$) | 56.67% | **56.67%** | 0.4222 | **0.4222** | $0.0000$ | **EXACT MATCH** |

---

## 2. Verdict
$$\mathbf{METRIC\_REPRODUCTION = EXACT\_MATCH}$$
All metrics across all 4 models were reproduced deterministically without any variance.
