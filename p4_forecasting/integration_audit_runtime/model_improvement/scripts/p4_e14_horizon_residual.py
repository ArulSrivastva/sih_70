"""EXP-P4-14 — Horizon-specific residual models.

Compare:
  A. Single shared residual model (same model predicts all 3 horizons)
  B. Horizon-specific residual models (separate model per horizon)

Select using validation performance only. Evaluate once on test.
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
    lat0 = X_raw[:, 4, 0]; lon0 = X_raw[:, 4, 1]
    lat_m = X_raw[:, 3, 0]; lon_m = X_raw[:, 3, 1]
    dlat_deg = lat0 - lat_m
    dlon_deg = C.wrap_lon_delta(lon_m, lon0)
    N = X_raw.shape[0]
    dlat_km = np.empty((N, 3)); dlon_km = np.empty((N, 3))
    for hi, h in enumerate(C.HORIZONS):
        m = C.MULT[h]
        target_lat = lat0 + m * dlat_deg
        target_lon = (lon0 + m * dlon_deg) % 360.0
        dlat_km[:, hi] = C.haversine_km(lat0, lon0, target_lat, lon0) * np.sign(target_lat - lat0)
        dlon_km[:, hi] = C.haversine_km(lat0, lon0, lat0, target_lon) * np.sign(C.wrap_lon_delta(lon0, target_lon))
    return dlat_km, dlon_km


def train_shared_lgb(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va):
    """Shared model: train on all horizons stacked (features replicated 3x)."""
    models = {}
    for coord, res_tr_all, res_va_all in [("lat", res_lat_tr, res_lat_va),
                                           ("lon", res_lon_tr, res_lon_va)]:
        # Stack features 3x (once per horizon) and stack targets
        Ftr_stack = np.tile(Ftr.values, (3, 1))
        Fva_stack = np.tile(Fva.values, (3, 1))
        y_stack = res_tr_all.ravel()
        y_va_stack = res_va_all.ravel()
        m = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1e-2, random_state=SEED, verbose=-1)
        m.fit(Ftr_stack, y_stack,
              eval_set=[(Fva_stack, y_va_stack)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        models[f"shared_{coord}"] = m
    return models


def train_shared_catboost(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va):
    models = {}
    for coord, res_tr_all, res_va_all in [("lat", res_lat_tr, res_lat_va),
                                           ("lon", res_lon_tr, res_lon_va)]:
        Ftr_stack = np.tile(Ftr.values, (3, 1))
        Fva_stack = np.tile(Fva.values, (3, 1))
        y_stack = res_tr_all.ravel()
        y_va_stack = res_va_all.ravel()
        m = CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=5,
            l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
            early_stopping_rounds=30)
        m.fit(Ftr_stack, y_stack,
              eval_set=(Fva_stack, y_va_stack))
        models[f"shared_{coord}"] = m
    return models


def train_perhorizon_lgb(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va):
    models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, res_tr, res_va in [("lat", res_lat_tr[:, hi], res_lat_va[:, hi]),
                                       ("lon", res_lon_tr[:, hi], res_lon_va[:, hi])]:
            m = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=5,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1e-2, random_state=SEED, verbose=-1)
            m.fit(Ftr.values, res_tr,
                  eval_set=[(Fva.values, res_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            models[f"{h}h_{coord}"] = m
    return models


def train_perhorizon_catboost(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va):
    models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, res_tr, res_va in [("lat", res_lat_tr[:, hi], res_lat_va[:, hi]),
                                       ("lon", res_lon_tr[:, hi], res_lon_va[:, hi])]:
            m = CatBoostRegressor(
                iterations=400, learning_rate=0.05, depth=5,
                l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
                early_stopping_rounds=30)
            m.fit(Ftr.values, res_tr,
                  eval_set=(Fva.values, res_va))
            models[f"{h}h_{coord}"] = m
    return models


def predict_shared(models, F, mv_dlat, mv_dlon, Xr):
    res_lat = np.column_stack([models["shared_lat"].predict(F.values)] * 3)
    res_lon = np.column_stack([models["shared_lon"].predict(F.values)] * 3)
    final_dlat = mv_dlat + res_lat
    final_dlon = mv_dlon + res_lon
    return E.displacement_to_position(Xr, final_dlat, final_dlon)


def predict_perhorizon(models, F, mv_dlat, mv_dlon, Xr):
    res_lat = np.column_stack([models[f"{h}h_lat"].predict(F.values) for h in C.HORIZONS])
    res_lon = np.column_stack([models[f"{h}h_lon"].predict(F.values) for h in C.HORIZONS])
    final_dlat = mv_dlat + res_lat
    final_dlon = mv_dlon + res_lon
    return E.displacement_to_position(Xr, final_dlat, final_dlon)


def main():
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    Xte, Yte, Xrte = C.load_split_data("test")

    Ftr = E.build_tabular_features(Xtr, Xrtr)
    Fva = E.build_tabular_features(Xva, Xrva)
    Fte = E.build_tabular_features(Xte, Xrte)

    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)

    mv_dlat_tr, mv_dlon_tr = mv_displacement(Xrtr)
    mv_dlat_va, mv_dlon_va = mv_displacement(Xrva)
    mv_dlat_te, mv_dlon_te = mv_displacement(Xrte)

    res_lat_tr = dlat_tr - mv_dlat_tr
    res_lon_tr = dlon_tr - mv_dlon_tr
    res_lat_va = dlat_va - mv_dlat_va
    res_lon_va = dlon_va - mv_dlon_va

    results = {}

    # --- LightGBM shared ---
    print("LightGBM shared...")
    lgb_shared = train_shared_lgb(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va)
    lgb_shared_val = predict_shared(lgb_shared, Fva, mv_dlat_va, mv_dlon_va, Xrva)
    lgb_shared_te = predict_shared(lgb_shared, Fte, mv_dlat_te, mv_dlon_te, Xrte)
    lgb_shared_val_m = C.summarize(lgb_shared_val, Yva[:, :, :2])
    lgb_shared_te_m = C.summarize(lgb_shared_te, Yte[:, :, :2])
    results["lgb_shared"] = {"val": lgb_shared_val_m, "test": lgb_shared_te_m}

    # --- LightGBM per-horizon ---
    print("LightGBM per-horizon...")
    lgb_per = train_perhorizon_lgb(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va)
    lgb_per_val = predict_perhorizon(lgb_per, Fva, mv_dlat_va, mv_dlon_va, Xrva)
    lgb_per_te = predict_perhorizon(lgb_per, Fte, mv_dlat_te, mv_dlon_te, Xrte)
    lgb_per_val_m = C.summarize(lgb_per_val, Yva[:, :, :2])
    lgb_per_te_m = C.summarize(lgb_per_te, Yte[:, :, :2])
    results["lgb_perhorizon"] = {"val": lgb_per_val_m, "test": lgb_per_te_m}

    # --- CatBoost shared ---
    print("CatBoost shared...")
    cat_shared = train_shared_catboost(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va)
    cat_shared_val = predict_shared(cat_shared, Fva, mv_dlat_va, mv_dlon_va, Xrva)
    cat_shared_te = predict_shared(cat_shared, Fte, mv_dlat_te, mv_dlon_te, Xrte)
    cat_shared_val_m = C.summarize(cat_shared_val, Yva[:, :, :2])
    cat_shared_te_m = C.summarize(cat_shared_te, Yte[:, :, :2])
    results["cat_shared"] = {"val": cat_shared_val_m, "test": cat_shared_te_m}

    # --- CatBoost per-horizon ---
    print("CatBoost per-horizon...")
    cat_per = train_perhorizon_catboost(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va)
    cat_per_val = predict_perhorizon(cat_per, Fva, mv_dlat_va, mv_dlon_va, Xrva)
    cat_per_te = predict_perhorizon(cat_per, Fte, mv_dlat_te, mv_dlon_te, Xrte)
    cat_per_val_m = C.summarize(cat_per_val, Yva[:, :, :2])
    cat_per_te_m = C.summarize(cat_per_te, Yte[:, :, :2])
    results["cat_perhorizon"] = {"val": cat_per_val_m, "test": cat_per_te_m}

    # --- Baselines ---
    mv_val_m = C.summarize(C.movement_vector_pred(Xrva), Yva[:, :, :2])
    mv_te_m = C.summarize(C.movement_vector_pred(Xrte), Yte[:, :, :2])
    results["movement_vector"] = {"val": mv_val_m, "test": mv_te_m}

    # Print comparison
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print(f"{'Model':<30s} {'6h':>8s} {'12h':>8s} {'24h':>8s} {'mean':>8s}")
    for name in ["movement_vector", "lgb_shared", "lgb_perhorizon", "cat_shared", "cat_perhorizon"]:
        m = results[name]["val"]
        mean_e = np.mean([m[str(h)]["track_error_km_mean"] for h in C.HORIZONS])
        print(f"{name:<30s} {m['6']['track_error_km_mean']:>8.2f} "
              f"{m['12']['track_error_km_mean']:>8.2f} {m['24']['track_error_km_mean']:>8.2f} "
              f"{mean_e:>8.2f}")

    print("\n" + "=" * 70)
    print("TEST")
    print("=" * 70)
    print(f"{'Model':<30s} {'6h':>8s} {'12h':>8s} {'24h':>8s} {'mean':>8s}")
    for name in ["movement_vector", "lgb_shared", "lgb_perhorizon", "cat_shared", "cat_perhorizon"]:
        m = results[name]["test"]
        mean_e = np.mean([m[str(h)]["track_error_km_mean"] for h in C.HORIZONS])
        print(f"{name:<30s} {m['6']['track_error_km_mean']:>8.2f} "
              f"{m['12']['track_error_km_mean']:>8.2f} {m['24']['track_error_km_mean']:>8.2f} "
              f"{mean_e:>8.2f}")

    # Best on validation
    val_scores = {}
    for name in ["lgb_shared", "lgb_perhorizon", "cat_shared", "cat_perhorizon"]:
        val_scores[name] = np.mean([results[name]["val"][str(h)]["track_error_km_mean"]
                                    for h in C.HORIZONS])
    best_val = min(val_scores, key=val_scores.get)
    print(f"\nBest on validation: {best_val} ({val_scores[best_val]:.2f} km)")

    out = RESULTS / "P4_E14_HORIZON_RESIDUAL.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
