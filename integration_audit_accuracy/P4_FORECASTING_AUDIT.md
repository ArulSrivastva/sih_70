# P4 Forecasting Audit

**Status: PASS_WHERE_VERIFIABLE_WITH_SIGNIFICANT_CAVEATS (reported numbers are honest and reproducible; headline capability is weaker than the baselines; one registry anomaly).**

## 1. Canonical chronology contract — VERIFIED

SHA256 of delivered `canonical_chrono/{train,val,test}.npz` match the expected prefixes recorded in `phase4/common.py` exactly:

| File | SHA256 prefix | Expected | Match |
|---|---|---|---|
| train.npz | `df70303e…` | `df70303e` | ✓ |
| val.npz | `48cf065d…` | `48cf065d` | ✓ |
| test.npz | `89e9c2e2…` | `89e9c2e2` | ✓ |

## 2. Champion (EXP005) test metrics — INDEPENDENTLY REPRODUCED

Re-implemented the GRU (`16→GRU(96,2,dropout .1)→Linear→(3,3)`), loaded `EXP005/checkpoint.pt`, reapplied train-only z-score normalization, and scored the 198-sample test set:

| Horizon | Metric | Stored | Independent | Δ |
|---|---|---|---|---|
| 6h | track mean km | 91.429 | 91.429 | 2e-8 |
| 6h | track median km | 74.025 | 74.025 | 7e-6 |
| 6h | track std km | 60.840 | 60.840 | 2e-6 |
| 6h | wind MAE / RMSE km/h | 6.685 / 8.282 | 6.685 / 8.282 | 0 / 0 |
| 12h | track mean km | 119.765 | 119.765 | 7e-6 |
| 12h | track median / std km | 94.898 / 86.749 | … | <1e-5 |
| 12h | wind MAE / RMSE | 9.459 / 11.667 | same | 0 / 0 |
| 24h | track mean km | 188.239 | 188.239 | 8e-6 |
| 24h | track median / std km | 152.667 / 128.672 | … | <1e-5 |
| 24h | wind MAE / RMSE | 16.108 / 19.650 | same | 0 / 0 |

All values match to numerical-noise level. **Reported champion test metrics: reproduced.**

## 3. Baselines — INDEPENDENTLY REPRODUCED (with a population correction)

Running the delivered `phase2` persistence & movement-vector baselines on the *feature_dataset* test set reproduces `phase2/results/baseline_results.json` and `FINAL_COMPARISON.json` to machine precision:

| Model | 6h | 12h | 24h |
|---|---|---|---|
| persistence (comp) | 64.686 | 123.936 | 227.327 |
| persistence (stored) | 64.686 | 123.936 | 227.327 |
| movement_vector (comp) | 38.051 | 80.505 | 180.661 |
| movement_vector (stored) | 38.051 | 80.505 | 180.661 |

**Population note (important):** the experiment/baseline evaluations used the phase4 `feature_dataset` (test = 198 rows / 10 cyclones, 6 h windows). The delivered `canonical_chrono/test.npz` contains **401 rows / 14 cyclones** (3 h cadence). Running the same baselines on canonical_chrono gives different numbers (e.g., movement-vector 24h 167.4). The prefix-hash contract pins `canonical_chrono`; the reported scores pin `feature_dataset`. Anyone re-scoring from `canonical_chrono` alone will **not** reproduce the reported numbers.

## 4. Champion-vs-baseline truth (already disclosed by the team, independently confirmed)

| Horizon | EXP005 GRU | movement_vector | persistence | phase3 LSTM* |
|---|---|---|---|---|
| 6h | 91.43 | **38.05** | **64.69** | ~130 |
| 12h | 119.76 | **80.50** | 123.94 | ~180 |
| 24h | 188.24 | **180.66** | 227.33 | ~268 |

\* phase3 LSTM values are from `phase3/model_comparison.json`; no phase3 weights were delivered, so they are cited-not-verified.

The champion **loses to the movement-vector baseline at every horizon**, and **loses to persistence at 6 h** (wins only at 12 h/24 h vs persistence). These numbers are honestly reported in the delivered `FINAL_COMPARISON.json`. The delivered forecasting capability does **not** beat the trivial baselines on track — this is the single most important accuracy finding of the audit.

## 5. Selection hygiene — CLEAN

- `champion_model.json` selection rule: lowest equal-weight mean of validation track errors; tie-break wind MAE then RMSE; **validation only, test explicitly unused**. Confirmed `val_primary_score = mean(val_track_6h, val_track_12h, val_track_24h)` exactly (Δ 1.4e-14).
- Ranking from registry: EXP005 (113.074) < EXP003 (127.153) < EXP006 (127.153) < EXP004 (140.476) < EXP002 (147.229) < EXP001 (166.823) — matches `champion_model.json` ranking.
- Test evaluated **once** (only EXP005 has `test_results.json`). No model selection on the test set. **Good practice — credit.**

## 6. Registry anomaly — EXP006 ≈ EXP003 (flag, reproducible)

`experiment_registry.csv` lists EXP003 (`loss=mse`) and EXP006 (`loss=weighted`) with **different losses yet validation-track metrics identical to ~5 significant figures at every horizon (98.7229… / 111.0280… / 171.7082… vs EXP003 98.7229… / 111.0280… / 171.7082…) and the same best_epoch=31**. They are not byte-identical at float precision, but such agreement across differing loss functions is effectively impossible by chance; the registry row for EXP006 looks copied from EXP003. EXP006’s own checkpoint exists but its `test_results.json` does not (only the champion is tested). Recommendation: do not treat EXP006 as an independent experiment result.

## 7. Feature/normalization hygiene — CLEAN

- Feature contract `(5,16)`: first 7 raw + 9 causal derived; step 0 zero-filled for predecessor-dependent features; verified no NaN/Inf anywhere in `feature_dataset` X/Y.
- `normalization_stats.json`: `computed_from: train`, `n_train_samples: 1212`; zero-std handled via scale-1 policy; directional features documented (wrap limitation disclosed, not hidden).

## p4_metrics.json / P4_EXPERIMENT_RANKING.csv

`p4_metrics.json`: prefixes, recomputed-vs-stored test metrics & Δ, baseline recomputation, registry rows, champion/ranking facts, EXP006 anomaly. `P4_EXPERIMENT_RANKING.csv`: the 6 experiments ranked by val_primary with model/loss/horizon metrics and anomaly flag.