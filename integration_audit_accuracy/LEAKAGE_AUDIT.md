# Leakage Audit

Four leakage classes were examined: (1) split disjointness, (2) temporal/causal integrity of forecasting windows, (3) cross-split near-duplicate **image** frames, and (4) normalization/statistics hygiene.

## 1. Cyclone-level split integrity — CLEAN

- Master 5,481 rows / 151 cyclones; splits disjoint (0 cyclones in ≥2 splits), union = all 151; rows sum to 5,481. Verified.
- Splits are **random**, not chronological: the earliest test timestamp (2013-05-29) is inside the train period (train extends to 2025-11-30). So “test” is random held-out storms **overlapping the train era** — legitimate for i.i.d. evaluation, but **not** an out-of-sample "predict the future" benchmark. Consumers must not read test scores as future-VALIDATION.

## 2. Forecasting-window causality — CLEAN after the P4 clean pass (with a documented P1 risk)

- P1 Dataset C windows are built from master tracks with `interpolate(method='time').ffill().bfill()`; a NaN `sst` can, in principle, be **back-filled from a later timestamp within the same storm**. This is a **P1 build risk**, documented in `P1_DATA_AUDIT.md`.
- Mitigation chain verified in P4: `canonical_chrono` (the clean dataset, prefix-hash pinned) removed non-causal cells; `feature_engineering.py` derives 9 features with predecessor-dependence and **zero-fills step 0** (no forward reference); `normalization_stats.json` is `computed_from: train`. Feature windows used for EXP005 inference are NaN/Inf-free and causal by construction. No forward-target leakage was found in `feature_dataset` X/Y.
- Baseline functions (persistence / movement-vector) use history only; their `(5,7)` contract refuses non-`(5,7)` input and TARGETS are never consumed (source-checked).

## 3. Cross-split near-duplicate IMAGE frames — FLAGGED (real contention)

Source: Kaggle INSAT-3D image set, filenames `NN.jpg` / `NN(k).jpg` = **variant frames of the same storm** (base `NN`). Independent frame-hash check: no byte-identical duplicates, so variants are different images (different times) of the same cyclone. But the **same storm appears in multiple splits**:

| Split pair | Same-storm base counts | Example frames |
|---|---|---|
| train ↔ test | 10 | `36.jpg`(train) ↔ `36(3).jpg`(test); `48.jpg/48(1)/48(3)`(train) ↔ `48(2).jpg`(test); `60(1).jpg`(train) ↔ `60(2).jpg`(test); `35.jpg…` ↔ `35(1).jpg` |
| train ↔ val | 10 | `42(3).jpg` ↔ `42.jpg`; `52.jpg` ↔ `52(1).jpg`; `47/47(1)/47(2)/47(4)` ↔ `47(3).jpg` |
| val ↔ test | 2 | `63(1).jpg` (val) ↔ `63.jpg` (test); `64.jpg` (val) … |

Effect: **same-cyclone frames straddle the test boundary**, so image-only models (P3, and the 21-frame P3/P2 test) can memorize storm geography/features seen in train. Reported P3 image accuracy (38.1%) and P2 pattern/category scores are therefore **optimistic lower bounds of overfitting**; treat them with reserve. Byte-duplicate check rules out exact-frame copying, but frame-level near-duplicates remain.

## 4. Statistics / normalization hygiene — CLEAN

- Phase-4 z-score statistics: train-only, verified from `normalization_stats.json` (`computed_from: train`, `n_train_samples: 1212`); zero-std handled deterministically; directional wrap-limitation documented (not hidden).
- P2 label maps: minor — badge maps for the combined split set (train+val+test) are built at train time (`src/detection/train.py:32-67`); the **category/pattern index space is defined using test-set labels**. No target values flow into training, but the practice is a (low-severity) leak of test-label vocabulary.

## 5. Cross-task data consistency

- P2 and P3 images/labels are the **same 133 images with identical split files** (verified `det_{split}_same_files_as_kaggle_{split} = true`). Detection and classification test sets therefore share frames — consistent, not a path for score inflation between P2/P3 (they don’t share model weights).

## 6. Population reproducibility note (not leakage, but score portability)

The delivered sets disagree on test population — P1 sequences 423 rows/16 cyclones; `canonical_chrono` 401/14; phase4 `feature_dataset` 198/10. All P4 reported numbers (EXP005 test, baselines) are defined **on feature_dataset (198)**. A fresh scoring run on `canonical_chrono` will yield different numbers (verified for baselines). Reproducing the report requires the phase4 feature pipeline, not just `canonical_chrono`.

## Verdict

| Class | Verdict |
|---|---|
| Cyclone split disjointness | PASS |
| Forecast-window causality (P4 feature dataset) | PASS |
| P1 bfill forward-fill risk | PASS_WITH_WARNINGS (documented; cleaned downstream) |
| Cross-split storm-frame near-duplicates (P2/P3 images) | **FAIL-LEVEL WARNING** — inflates image-model test scores |
| Train/test statistics isolation | PASS |
| Label-vocabulary leak (P2 label maps) | LOW (flagged) |
| Chronological validity to call test “future” | NOT CLAIMABLE (random split) |