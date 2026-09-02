# P2 Phase 2B — Label Relationship & Dvorak Contingency Audit

**Date**: September 1, 2026  
**Scope**: Training Data ($N=93$ samples) Relationship between `structural_pattern` and `category`.

---

## 1. Training Set Contingency Table

The training dataset ($N=93$ samples from `train_detection.csv`) displays the following exact empirical co-occurrences:

| Structural Pattern | Cyclonic Storm | Deep Depression | Depression | Extremely Severe CS | Severe CS | Very Severe CS | Total Count (%) |
|---|---|---|---|---|---|---|---|
| **`curved_band`** | 33 | 0 | 0 | 0 | 0 | 0 | **33 (35.48%)** |
| **`eye_visible`** | 0 | 0 | 0 | 8 | 24 | 19 | **51 (54.84%)** |
| **`shear_pattern`**| 0 | 8 | 1 | 0 | 0 | 0 | **9 (9.68%)** |
| **Total (%)** | **33 (35.48%)** | **8 (8.60%)** | **1 (1.08%)** | **8 (8.60%)** | **24 (25.81%)** | **19 (20.43%)** | **93 (100.0%)** |

---

## 2. Conditional Probability Distributions $P(\text{category} \mid \text{pattern})$

From the training data:
- $P(\text{Cyclonic Storm} \mid \text{curved\_band}) = 1.000$ (100.0%)
- $P(\text{Severe Cyclonic Storm} \mid \text{eye\_visible}) = 0.4706$ (47.06%)
- $P(\text{Very Severe Cyclonic Storm} \mid \text{eye\_visible}) = 0.3725$ (37.25%)
- $P(\text{Extremely Severe Cyclonic Storm} \mid \text{eye\_visible}) = 0.1569$ (15.69%)
- $P(\text{Deep Depression} \mid \text{shear\_pattern}) = 0.8889$ (88.89%)
- $P(\text{Depression} \mid \text{shear\_pattern}) = 0.1111$ (11.11%)

---

## 3. Analysis of Domain Inconsistency vs Rare Combinations

1. **Deterministic Coupling Artifact**: In `build_datasets.py`, Dataset A labels were generated via deterministic threshold logic (`eye_visible` for Severe categories, `curved_band` for Cyclonic Storm, `shear_pattern` for Depressions). Consequently, each structural pattern in the training dataset only maps to a disjoint subset of intensity categories.
2. **Meteorological Inconsistencies**:
   - `(eye_visible, Cyclonic Storm)` is physically impossible because an eye requires central core organization associated with sustained winds $\ge 89$ km/h (T-number $\ge 3.5$).
   - `(eye_visible, Depression)` is physically impossible.
   - `(shear_pattern, Extremely Severe Cyclonic Storm)` is physically impossible under standard tropical cyclone dynamics.
3. **Rare Combinations**:
   - `(shear_pattern, Depression)` has only 1 sample ($1.08\%$) but is meteorologically valid for developing low-intensity systems.
