"""EXP-P4-13 — Movement-vector residual model.

Instead of predicting displacement from scratch, predict the RESIDUAL:
  residual = actual_displacement - movement_vector_displacement

Then: final = movement_vector + predicted_residual

Protocol:
  * Compute movement-vector displacement targets.
  * Compute actual displacement targets.
  * residual = actual - mv_displacement.
  * Train LightGBM and CatBoost to predict the residual.
  * Evaluate on validation and test.
  * Compare against movement-vector, LightGBM displacement, CatBoost displacement.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=DeprecationWarning)

SCRIPTS = Path(__file__).resolve().parent
RESULTS = SCRIPTS.parent / "results"
sys.path.insert(0, str(SCRIPTS))
import p4_common as C
import p4_experiments as E

SEED = 42
np.random.seed(SEED)


def mv_displacement(X_raw):
    """Compute movement-vector displacement (km) from t0 position."""
    lat0 = X_raw[:, 4, 0]; lon0 = X_raw[:, 4, 1]
    lat_m = X_raw[:, 3, 0]; lon_m = X_raw[:, 3, 1]
    dlat_deg = lat0 - lat_m
    dlon_deg = C.wrap_lon_delta(lon_m, lon0)
    N = X_raw.shape[0]
    dlat_km = np.empty((N, 3))
    dlon_km = np.empty((N, 3))
    for hi, h in enumerate(C.HORIZONS):
        m = C.MULT[h]
        target_lat = lat0 + m * dlat_deg
        target_lon = (lon0 + m * dlon_deg) % 360.0
        dlat_km[:, hi] = C.haversine_km(lat0, lon0, target_lat, lon0) * np.sign(target_lat - lat0)
        dlon_km[:, hi] = C.haversine_km(lat0, lon0, lat0, target_lon) * np.sign(C.wrap_lon_delta(lon0, target_lon))
    return dlat_km, dlon_km


def train_lgb_residual(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va):
    models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, trg_tr, trg_va in [("lat", res_lat_tr[:, hi], res_lat_va[:, hi]),
                                       ("lon", res_lon_tr[:, hi], res_lon_va[:, hi])]:
            m = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=5,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1e-2, random_state=SEED, verbose=-1)
            m.fit(Ftr.values, trg_tr,
                  eval_set=[(Fva.values, trg_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            models[f"{h}h_{coord}"] = m
    return models


def train_catboost_residual(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va):
    models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, trg_tr, trg_va in [("lat", res_lat_tr[:, hi], res_lat_va[:, hi]),
                                       ("lon", res_lon_tr[:, hi], res_lon_va[:, hi])]:
            m = CatBoostRegressor(
                iterations=400, learning_rate=0.05, depth=5,
                l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
                early_stopping_rounds=30)
            m.fit(Ftr.values, trg_tr,
                  eval_set=(Fva.values, trg_va))
            models[f"{h}h_{coord}"] = m
    return models


def residual_predict(models, F, mv_dlat, mv_dlon, Xr):
    """Predict residual, add to MV displacement, convert to position."""
    res_dlat = np.column_stack([models[f"{h}h_lat"].predict(F.values) for h in C.HORIZONS])
    res_dlon = np.column_stack([models[f"{h}h_lon"].predict(F.values) for h in C.HORIZONS])
    final_dlat = mv_dlat + res_dlat
    final_dlon = mv_dlon + res_dlon
    pos = E.displacement_to_position(Xr, final_dlat, final_dlon)
    return pos, res_dlat, res_dlon


def main():
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    Xte, Yte, Xrte = C.load_split_data("test")

    Ftr = E.build_tabular_features(Xtr, Xrtr)
    Fva = E.build_tabular_features(Xva, Xrva)
    Fte = E.build_tabular_features(Xte, Xrte)

    # Actual displacement targets
    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)
    dlat_te, dlon_te = E.displacement_targets(Xrte, Yte)

    # MV displacement
    mv_dlat_tr, mv_dlon_tr = mv_displacement(Xrtr)
    mv_dlat_va, mv_dlon_va = mv_displacement(Xrva)
    mv_dlat_te, mv_dlon_te = mv_displacement(Xrte)

    # Residuals = actual - MV
    res_lat_tr = dlat_tr - mv_dlat_tr
    res_lon_tr = dlon_tr - mv_dlon_tr
    res_lat_va = dlat_va - mv_dlat_va
    res_lon_va = dlon_va - mv_dlon_va

    print(f"Train: {Xtr.shape[0]}, Val: {Xva.shape[0]}, Test: {Xte.shape[0]}")
    print(f"Residual stats (train): lat mean={np.mean(res_lat_tr):.2f} std={np.std(res_lat_tr):.2f}, "
          f"lon mean={np.mean(res_lon_tr):.2f} std={np.std(res_lon_tr):.2f}")

    # --- LightGBM residual ---
    print("\nTraining LightGBM residual model...")
    lgb_res_models = train_lgb_residual(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va)

    lgb_res_val, _, _ = residual_predict(lgb_res_models, Fva, mv_dlat_va, mv_dlon_va, Xrva)
    lgb_res_te, _, _ = residual_predict(lgb_res_models, Fte, mv_dlat_te, mv_dlon_te, Xrte)
    lgb_res_val_m = C.summarize(lgb_res_val, Yva[:, :, :2])
    lgb_res_te_m = C.summarize(lgb_res_te, Yte[:, :, :2])

    # --- CatBoost residual ---
    print("Training CatBoost residual model...")
    cat_res_models = train_catboost_residual(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va)

    cat_res_val, _, _ = residual_predict(cat_res_models, Fva, mv_dlat_va, mv_dlon_va, Xrva)
    cat_res_te, _, _ = residual_predict(cat_res_models, Fte, mv_dlat_te, mv_dlon_te, Xrte)
    cat_res_val_m = C.summarize(cat_res_val, Yva[:, :, :2])
    cat_res_te_m = C.summarize(cat_res_te, Yte[:, :, :2])

    # --- Baselines ---
    mv_val = C.summarize(C.movement_vector_pred(Xrva), Yva[:, :, :2])
    mv_te = C.summarize(C.movement_vector_pred(Xrte), Yte[:, :, :2])

    # LightGBM displacement (non-residual)
    lgb_dlat_tr, lgb_dlon_tr = E.displacement_targets(Xrtr, Ytr)
    lgb_models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, trg_tr, trg_va in [("lat", dlat_tr[:, hi], dlat_va[:, hi]),
                                       ("lon", dlon_tr[:, hi], dlon_va[:, hi])]:
            m = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=5,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1e-2, random_state=SEED, verbose=-1)
            m.fit(Ftr.values, trg_tr,
                  eval_set=[(Fva.values, trg_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            lgb_models[f"{h}h_{coord}"] = m
    lgb_dlat_p = np.column_stack([lgb_models[f"{h}h_lat"].predict(Fva.values) for h in C.HORIZONS])
    lgb_dlon_p = np.column_stack([lgb_models[f"{h}h_lon"].predict(Fva.values) for h in C.HORIZONS])
    lgb_disp_val = E.displacement_to_position(Xrva, lgb_dlat_p, lgb_dlon_p)
    lgb_dlat_tp = np.column_stack([lgb_models[f"{h}h_lat"].predict(Fte.values) for h in C.HORIZONS])
    lgb_dlon_tp = np.column_stack([lgb_models[f"{h}h_lon"].predict(Fte.values) for h in C.HORIZONS])
    lgb_disp_te = E.displacement_to_position(Xrte, lgb_dlat_tp, lgb_dlon_tp)
    lgb_disp_val_m = C.summarize(lgb_disp_val, Yva[:, :, :2])
    lgb_disp_te_m = C.summarize(lgb_disp_te, Yte[:, :, :2])

    # CatBoost displacement
    cat_models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, trg_tr, trg_va in [("lat", dlat_tr[:, hi], dlat_va[:, hi]),
                                       ("lon", dlon_tr[:, hi], dlon_va[:, hi])]:
            m = CatBoostRegressor(
                iterations=400, learning_rate=0.05, depth=5,
                l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
                early_stopping_rounds=30)
            m.fit(Ftr.values, trg_tr,
                  eval_set=(Fva.values, trg_va))
            cat_models[f"{h}h_{coord}"] = m
    cat_dlat_p = np.column_stack([cat_models[f"{h}h_lat"].predict(Fva.values) for h in C.HORIZONS])
    cat_dlon_p = np.column_stack([cat_models[f"{h}h_lon"].predict(Fva.values) for h in C.HORIZONS])
    cat_disp_val = E.displacement_to_position(Xrva, cat_dlat_p, cat_dlon_p)
    cat_dlat_tp = np.column_stack([cat_models[f"{h}h_lat"].predict(Fte.values) for h in C.HORIZONS])
    cat_dlon_tp = np.column_stack([cat_models[f"{h}h_lon"].predict(Fte.values) for h in C.HORIZONS])
    cat_disp_te = E.displacement_to_position(Xrte, cat_dlat_tp, cat_dlon_tp)
    cat_disp_val_m = C.summarize(cat_disp_val, Yva[:, :, :2])
    cat_disp_te_m = C.summarize(cat_disp_te, Yte[:, :, :2])

    # Print comparison
    print("\n" + "=" * 70)
    print("VALIDATION (mean track error km)")
    print("=" * 70)
    print(f"{'Model':<30s} {'6h':>8s} {'12h':>8s} {'24h':>8s}")
    for name, m in [("Movement-Vector", mv_val), ("LGB Displacement", lgb_disp_val_m),
                     ("CatBoost Displacement", cat_disp_val_m),
                     ("LGB Residual", lgb_res_val_m), ("CatBoost Residual", cat_res_val_m)]:
        print(f"{name:<30s} {m['6']['track_error_km_mean']:>8.2f} "
              f"{m['12']['track_error_km_mean']:>8.2f} {m['24']['track_error_km_mean']:>8.2f}")

    print("\n" + "=" * 70)
    print("TEST (mean track error km)")
    print("=" * 70)
    print(f"{'Model':<30s} {'6h':>8s} {'12h':>8s} {'24h':>8s}")
    for name, m in [("Movement-Vector", mv_te), ("LGB Displacement", lgb_disp_te_m),
                     ("CatBoost Displacement", cat_disp_te_m),
                     ("LGB Residual", lgb_res_te_m), ("CatBoost Residual", cat_res_te_m)]:
        print(f"{name:<30s} {m['6']['track_error_km_mean']:>8.2f} "
              f"{m['12']['track_error_km_mean']:>8.2f} {m['24']['track_error_km_mean']:>8.2f}")

    # Determine best on validation
    val_scores = {
        "mv": np.mean([mv_val[str(h)]["track_error_km_mean"] for h in C.HORIZONS]),
        "lgb_disp": np.mean([lgb_disp_val_m[str(h)]["track_error_km_mean"] for h in C.HORIZONS]),
        "cat_disp": np.mean([cat_disp_val_m[str(h)]["track_error_km_mean"] for h in C.HORIZONS]),
        "lgb_res": np.mean([lgb_res_val_m[str(h)]["track_error_km_mean"] for h in C.HORIZONS]),
        "cat_res": np.mean([cat_res_val_m[str(h)]["track_error_km_mean"] for h in C.HORIZONS]),
    }
    best_val = min(val_scores, key=val_scores.get)
    print(f"\nBest on validation (mean across horizons): {best_val} ({val_scores[best_val]:.2f} km)")

    out = RESULTS / "P4_E13_RESIDUAL_MODEL.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "validation": {
                "movement_vector": mv_val,
                "lgb_displacement": lgb_disp_val_m,
                "catboost_displacement": cat_disp_val_m,
                "lgb_residual": lgb_res_val_m,
                "catboost_residual": cat_res_val_m,
            },
            "test": {
                "movement_vector": mv_te,
                "lgb_displacement": lgb_disp_te_m,
                "catboost_displacement": cat_disp_te_m,
                "lgb_residual": lgb_res_te_m,
                "catboost_residual": cat_res_te_m,
            },
            "val_scores": val_scores,
            "best_val_model": best_val,
        }, f, indent=2)
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
