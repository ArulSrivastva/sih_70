# P3 Classification Audit

**Status: PASS_ON_IMAGE_MODEL (exact) / NOT_RUN_ON_TABULAR (lightgbm absent) / FLAWED_COMPARISON.**

## 1. Image-only model (ResNet18) — independently reproduced

Ran the delivered `image_only_model.pt` (backbone ResNet18, 7-class head + wind-regressor head) on the 21 test images with the shipped preprocessing (PIL resize 256×256, /255, HWC→CHW):

| Metric | Stored (`metrics_comparison.json`) | Independent rerun | Diff |
|---|---|---|---|
| category_accuracy_percent | 38.10 | 38.1 | 0.0 |
| category_macro_f1 | 0.2076 | 0.2076 | 0.0 |
| wind_speed_mae_kmh | 109.81 | 109.81 | 0.0 |
| wind_speed_rmse_kmh | 118.39 | 118.39 | 0.0 |

The metric *values* are exact, but note what they imply: 38.1% accuracy vs 14.3% random on 7 classes, and a wind RMSE of ≈118 km/h on a target with range ≈35–250 km/h — the image regression is, in practice, a weak baseline.

## 2. Multi-source tabular model — NOT independently verifiable in this environment

- `models/classification/tabular_multisource_model.pkl` pickle contains LightGBM estimators; `lightgbm` is **not installed** in the audit environment, so the checkpoint cannot be loaded or re-scored (unpicking fails at the class reference). **Recorded NOT_RUN with a real dependency-reason (never faked PASS).**
- Reported values (stored, not independently verified): acc 47.0%, macro_f1 0.3703, wind_MAE 18.84, wind_RMSE 27.28, pressure_MAE 5.11, pressure_RMSE 8.33 on 651 test rows.

## 3. The headline “multi-source advantage” is an improper comparison

`evaluate.py` folds two **different test populations** into one `performance_delta`:
- Image model → 21 INSAT images (its own test).
- Tabular model → 651 ERA5/IBTrACS records (its own test).

These are not the same storms, the same samples, or even the same input modality. The stored delta block (“+8.9% accuracy lift”, “−90.97 km/h wind MAE”) therefore does **not** establish that multi-source tabular input beats imagery. It is an apples-to-oranges artifact of the evaluation, not evidence.

## 4. Label provenance

Image labels come from the Kaggle INSAT-3D sheet (wind knots → IMD class). The test “Super Cyclonic Storm” has 1 frame and 6 classes have ≤6 frames (see `p1_metrics.json.img_test_rare_classes`); macro-F1 on such a tiny, imbalanced 21-image test is statistically fragile (95% CI on 38% accuracy is roughly ±21%).

## Verdicts

| Item | Verdict |
|---|---|
| Image-model metrics independently reproduce | **YES (exact)** |
| Tabular-model metrics independently reproduce | **NOT_RUN** (no lightgbm) |
| “Multi-source beats image” claim | **NOT SUPPORTED** (different populations); do not present as verified |
| Frontend consumes classification | NO (no `/api/classify` endpoint; dashboard is mock-only) |

## p3_metrics.json

Checkpoint-state-dict check, per-sample predictions, stored-vs-recomputed comparison, and `NOT_RUN` reason for tabular.