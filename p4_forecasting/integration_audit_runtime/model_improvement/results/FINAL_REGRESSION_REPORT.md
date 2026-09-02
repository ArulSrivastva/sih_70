# Final Regression/Immutability Report

## Checks

- [PASS] integration_api_config_exists
- [PASS] phase5_predictor_exists
- [PASS] phase6_api_exists
- [PASS] forecasting_adapter_exists
- [PASS] exp005_checkpoint_exists
- [PASS] normalization_stats_exists
- [PASS] exp005_config_exists
- [PASS] ps70_zip_exists
- [PASS] p3_tabular_in_zip
- [PASS] p3_image_in_zip
- [PASS] predictor_uses_gru
- [PASS] predictor_config_check
- [PASS] horizons_6_12_24

## Summary
All checks PASSED.

## Pipeline verification
- Integration API config: P3 uses LightGBM (stored model), P4 uses EXP005 GRU via phase6 adapter
- Dashboard serves P3 classification + P4 track forecast + heuristics (landfall/risk)
- No experimental models (CatBoost displacement) are used in production inference
- CatBoost results are from model_improvement experiments only
- All horizons are +6h, +12h, +24h as required