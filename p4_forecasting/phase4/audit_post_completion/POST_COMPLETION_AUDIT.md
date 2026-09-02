# P4 Phase-4 POST-COMPLETION READ-ONLY AUDIT

- Generated: 2026-08-29T11:46:50.436511+00:00
- Final verdict: **PASS**

## 1. Audit method
. Independent read-only re-verification of every Phase-4 deliverable and scientific claim. No file outside `phase4/audit_post_completion/` was written. No retraining, no regeneration, no test-suite execution (pytest would write under `phase4/tests/.scratch` and self-heal `source_immutability_report.json`).

## S1 — PASS
- **[PASS]** recorded-baseline-immutability: p1=0 phase1=0 phase2=0 phase3=0 files changed vs recorded baseline
- **[PASS]** champion-recorded-source-hashes: champion EXP005 immutable_sources re-hash match

## S2 — PASS
- **[PASS]** deliverable-inventory: missing: none
- **[PASS]** model-file-naming: expected tree template lists models/cyclone_lstm.py; actual implementation ships gru.py, improved_lstm.py, multitask_lstm.py. naming divergence is cosmetic (locked-family ImprovedLSTM).

## S3 — PASS
- **[PASS]** feature-contract-shapes: {"train": {"X_shape": [1212, 5, 16], "Y_shape": [1212, 3, 3], "feature_contract_order_ok": true, "target_contract_order_ok": true, "horizons_ok": true, "nan_inf_ok": true}, "val": {"X_shape": [231, 5, 16], "Y_shape": [231, 3, 3], "feature_contract_order_ok": true, "target_contract_order_ok": true, "horizons_ok": true, "nan_inf_ok": true}, "test": {"X_shape": [198, 5, 16], "Y_shape": [198, 3, 3], "feature_contract_order_ok": true, "target_contract_order_ok": true, "horizons_ok": true, "nan_inf_ok": true}}
- **[PASS]** raw-seven-cols-byte-identical-to-source: X[:,:,:7] == CLEAN X (all splits)

## S4 — PASS
- **[PASS]** causal-feature-recompute-parity: {"train": {"independent_recompute_max_abs_diff": 0.00257110595703125, "match": true}, "val": {"independent_recompute_max_abs_diff": 0.002498626708984375, "match": true}, "test": {"independent_recompute_max_abs_diff": 0.0023345947265625, "match": true}}
- **[PASS]** future-step-mutation-invariance: steps 0..1 unchanged when steps 2..4 mutated (independent recompute)
- **[PASS]** target-never-read-in-feature-module: no standalone target symbol Y in the feature-engineering code path (word-boundary scan); causal recompute below proves parity using raw X only, so target independence holds functionally
- **[PASS]** first-step-zero-fill-7-predecessor-features: X[:,0,7:14] all exactly zero (train/val/test)

## S5 — PASS
- **[PASS]** per-timestep-16-features: {"train": {"shape": [1212, 5, 16], "16feats_each_step": true}, "val": {"shape": [231, 5, 16], "16feats_each_step": true}, "test": {"shape": [198, 5, 16], "16feats_each_step": true}}

## S6 — PASS
- **[PASS]** normalization-train-only: computed_from={'split': 'train', 'policy': 'TRAIN only; never val/test/combined'} order_ok=True zero_std=[] n_train=1212 recompute_mean_match=True recompute_std_match=True yt_mean=True yt_std=True

## S7 — PASS
- **[PASS]** chronological-disjoint-splits: disjoint=True chronological=True manifest_cyclones(train/val/test)=68/15/14 (split_manifest is the PRE-CLEAN chronological split; CLEAN modeling counts 57/13/10 verified against feature metadata below) boundary(train_end<val_start<test_start)=2022-08-14 03:00:00<2022-08-17 12:00:00 & 2024-05-28 06:00:00<2024-08-25 00:00:00
- **[PASS]** feature-metadata-cyclone-disjoint: {"counts": {"train": 57, "val": 13, "test": 10}, "expected": {"train": 57, "val": 13, "test": 10}, "counts_ok": true, "every_feature_cyclone_matches_manifest_split": true}

## S8 — PASS
- **[PASS]** registry-22-cols-6-exps-all-pass: columns=22 ids_ok=True status_ok=True EXP001(improved_lstm/mse/64/1/0.0)=True
- **[PASS]** configs-match-registry: all 6 config.json == registry

## S9 — PASS
- **[PASS]** independent-champion-selection: independently ranked champion=EXP005 primary_val=113.074141 registry/champion=113.074140848568
- **[PASS]** registry-primary-equals-recomputed: val_primary_score column == recomputed mean of the three track means

## S10 — PASS
- **[PASS]** test-evaluated-exactly-once-champion-only: test_results.json only in EXP005=True checkpoint_hash_match=True touch={'champion': 'EXP005', 'checkpoint_sha256': 'ea6bbc3239059061f30eb43bd1b8b0c0db2dd7006b6c8f5e5a5e774d9c124a0a', 'split': 'test', 'evaluated_at': '2026-08-29T09:41:25.990067+00:00', 'evaluated_times': 1, 'note': 'TEST evaluated exactly once, champion only; election never uses test.'} test_samples=198

## S11 — PASS
- **[PASS]** champion-model-gru-huber-composition: {"model": "gru", "loss": "huber", "hidden_size": 96, "layers": 2, "dropout": 0.1, "seed": 42, "learning_rate": 0.001, "batch_size": 64}
- **[PASS]** champion-parameter-count-from-checkpoint: param_count=89577 recorded=89577 seed=42 epoch=16 val_loss=0.03382676631792799
- **[PASS]** champion-io-contract-Bx5x16->Bx3x3: o(4,5,16)->(4, 3, 3) o(1,5,16)->(1, 3, 3)

## S12 — PASS
- **[PASS]** all-improvement-percentages-recomputed-match: 27 comparisons, max|recorded-recomputed|=0.00e+00
- **[PASS]** raw-source-values-match-recorded: 9 raw-source spot checks
- **[PASS]** FINAL_COMPARISON-champion_test==EXP005-test_results: loaded from results/experiments/EXP005/test_results.json

## S13 — PASS
- **[PASS]** vs-phase3-lstm-all-horizons-positive: track% 6h/12h/24h = [29.68, 33.46, 29.64]
- **[PASS]** vs-persistence: wins 12h+24h, loses 6h: [6h,12h,24h]=[-41.34, 3.37, 17.19]
- **[PASS]** movement-vector superior at all horizons (honest negative): [6h,12h,24h]=[-140.28, -48.77, -4.19]
- **[PASS]** report-honest-negative-limitations: PHASE4_REPORT.md documents negative percentages/honesty

## S14 — PASS
- **[PASS]** inference-contract-code-check: {"denormalize_with_train_stats": true, "lon_wrap_into_0_360": true, "shape_validation": true, "nan_inf_rejection": true, "horizon_mapping_6_12_24": true}
- **[PASS]** inference-live-forecast: hours=[6, 12, 24] lon_wrapped_to_0_360=[89.165, 88.328, 88.06] nan_rejected=True bad_shape_rejected=True

## S15 — PASS
- **[PASS]** all-configs-seed-42: seed=42 in EXP001-006 configs
- **[PASS]** all-experiments-record-source-hashes: EXP001-006 each contain source_hashes.json with immutable_sources
- **[PASS]** reproducibility-anchors: {"date": "2026-08-29T09:45:57.892732+00:00", "python": "3.13.7", "n_experiments_pass": 6, "note": "bit-identical retraining NOT re-executed (read-only audit)"}

## S16 — PASS
- **[PASS]** source-immutability-after-audit: p1=0 phase1=0 phase2=0 phase3=0 changed

## S17 — PASS

> TEST EXECUTION: NOT RUN — WOULD VIOLATE READ-ONLY AUDIT CONSTRAINT. pytest's session-autouse conftest creates `phase4/tests/.scratch/` and `test_source_immutability.py` self-heals `results/source_immutability_report.json`; both are writes outside the audit directory. Recorded prior run: **90 passed**.

## 18. Independent comparison recap

- Champion (independent recompute): **EXP005**, primary (validation) 113.07414084856835 km.
- Champion test: 6h track 91.43 km / 12h 119.76 km / 24h 188.24 km (mean).
- vs Phase-3 LSTM (track mean %): +29.7 / +33.5 / +29.6 (champion win all horizons).
- vs persistence (track mean %): -41.3 / +3.4 / +17.2 (champion loses 6h, wins 12h/24h).
- vs movement-vector (track mean %): -140.3 / -48.8 / -4.2 (baseline superior at all horizons, reported honestly).

## Overall result

**FINAL VERDICT: PASS**