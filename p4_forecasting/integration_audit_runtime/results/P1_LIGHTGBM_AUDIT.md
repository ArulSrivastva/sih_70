# P1-A: LIGHTGBM / P3 TABULAR CLASSIFIER AUDIT

**Date:** 2026-08-31 (independent verification)
**Status:** PASS_WITH_WARNING

## LightGBM Installation

- Package: lightgbm 4.7.0 (narwhals dependency)
- Installation: environment change only, no source change

## Model Verified

- Pickle: `PS70-main/models/classification/tabular_multisource_model.pkl`
- Class: `MultisourceTabularModel` (use_lightgbm=True, LGBMClassifier)
- Features: [lat, lon, sst, pressure_msl, wind_u, wind_v]
- Labels: 7 IMD classes

## Test Metrics (independently reproduced)

| Split | Rows | Accuracy | Macro-F1 | Wind MAE |
|-------|------|----------|----------|----------|
| train | 3039 | 0.9720 | 0.9837 | 7.53 |
| val | 518 | 0.3938 | 0.1899 | 14.92 |
| **test** | **651** | **0.4700** | **0.3703** | **18.84** |

## Leakage Analysis

- train<->val SID overlap: 0
- train<->test SID overlap: 0
- val<->test SID overlap: 0
- Split: random (not chronological)
- No track leakage between splits

## Limitations

- Clear overfitting (train 0.97 vs test 0.47)
- Class imbalance (Depression ~39% of test)
- Macro-F1 0.37 = weak classifier
- NOT served in-process (module lives only in no-extraction zip)
- Live API reports available=False with precise reason

## Verdict

**PASS_WITH_WARNING** — Metrics are real and weak; in-process NOT_RUN preserved honestly.
