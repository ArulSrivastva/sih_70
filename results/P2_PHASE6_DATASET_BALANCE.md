# P2 Phase 6 — Dataset Balance & Distribution Audit Report

**Date**: September 1, 2026  
**Scope**: Statistical comparison between Legacy 133-image dataset and Expanded 2,120-image dataset.

---

## 1. Class Distribution Comparison Table

| Class Name | Legacy Dataset ($N=133$) | Expanded Dataset ($N=2,120$) | Expansion Multiplier | Resolution Status |
|---|---|---|---|---|
| **`Depression`** | 1 ($0.8\%$) | **428 ($20.2\%$)** | **$428\times$** | **Minority Bottleneck Resolved** |
| **`Deep Depression`** | 10 ($7.5\%$) | **209 ($9.9\%$)** | **$20.9\times$** | **Minority Bottleneck Resolved** |
| **`Cyclonic Storm`** | 43 ($32.3\%$) | **457 ($21.6\%$)** | **$10.6\times$** | Balanced |
| **`Severe CS`** | 36 ($27.1\%$) | **368 ($17.4\%$)** | **$10.2\times$** | Balanced |
| **`Very Severe CS`** | 31 ($23.3\%$) | **375 ($17.7\%$)** | **$12.1\times$** | Balanced |
| **`Extremely Severe CS`**| 11 ($8.3\%$) | **214 ($10.1\%$)** | **$19.5\times$** | Balanced |
| **`Super Cyclonic Storm`**| 1 ($0.8\%$) | **69 ($3.3\%$)** | **$69\times$** | **Minority Bottleneck Resolved** |
| **`shear_pattern`** | 11 ($8.3\%$) | **637 ($30.0\%$)** | **$57.9\times$** | **Near-Parity Rebalancing** |
| **`curved_band`** | 44 ($33.1\%$) | **825 ($38.9\%$)** | **$18.8\times$** | Dominant Clean Representation |
| **`eye_visible`** | 78 ($58.6\%$) | **658 ($31.0\%$)** | **$8.4\times$** | Normalized Majority |

---

## 2. Key Imbalance Reductions

1. **Pattern Parity**: The structural pattern imbalance ratio dropped from **$7.09 : 1$** down to **$1.29 : 1$** (near-equal representation of curved band, eye, and shear).
2. **Category Balance**: The category max-to-min ratio dropped from **$43.0 : 1$** down to **$6.62 : 1$**.
