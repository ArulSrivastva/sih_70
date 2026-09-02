"""P4-E07 — Boosting algorithm comparison.

Compare XGBoost, CatBoost, RandomForest, and LightGBM displacement regressors
on the same 26-feature tabular representation.  All models use the same
protocol: fit on TRAIN, early-stop on VALIDATION, evaluate ONCE on test.

Protocol:
  * Train separate models per (horizon, coordinate).
  * Identical feature set (build_tabular_features).
  * No test-set tuning; each model evaluated exactly once on test.
"""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

SCRIPTS = Path(__file__).resolve().parent
RESULTS = SCRIPTS.parent / "results"
sys.path.insert(0, str(SCRIPTS))
import p4_common as C
import p4_experiments as E

SEED = 42
np.random.seed(SEED)


def train_lgb(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va):
    import lightgbm as lgb
    models = {}
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
            models[f"{h}h_{coord}"] = m
    return models


def train_xgb(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va):
    from xgboost import XGBRegressor
    models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, trg_tr, trg_va in [("lat", dlat_tr[:, hi], dlat_va[:, hi]),
                                       ("lon", dlon_tr[:, hi], dlon_va[:, hi])]:
            m = XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=0.01,
                random_state=SEED, verbosity=0,
                early_stopping_rounds=30)
            m.fit(Ftr.values, trg_tr,
                  eval_set=[(Fva.values, trg_va)], verbose=False)
            models[f"{h}h_{coord}"] = m
    return models


def train_catboost(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va):
    from catboost import CatBoostRegressor
    models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, trg_tr, trg_va in [("lat", dlat_tr[:, hi], dlat_va[:, hi]),
                                       ("lon", dlon_tr[:, hi], dlon_va[:, hi])]:
            m = CatBoostRegressor(
                iterations=400, learning_rate=0.05, depth=5,
                l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
                early_stopping_rounds=30)
            m.fit(Ftr.values, trg_tr,
                  eval_set=(Fva.values, trg_va))
            models[f"{h}h_{coord}"] = m
    return models


def train_rf(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va):
    from sklearn.ensemble import RandomForestRegressor
    models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, trg_tr in [("lat", dlat_tr[:, hi]), ("lon", dlon_tr[:, hi])]:
            m = RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=5,
                random_state=SEED, n_jobs=-1)
            m.fit(Ftr.values, trg_tr)
            models[f"{h}h_{coord}"] = m
    return models


def evaluate_model(models, F, Xr):
    dlat = np.column_stack([models[f"{h}h_lat"].predict(F.values) for h in C.HORIZONS])
    dlon = np.column_stack([models[f"{h}h_lon"].predict(F.values) for h in C.HORIZONS])
    pos = E.displacement_to_position(Xr, dlat, dlon)
    return pos, dlat, dlon


def main():
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    Xte, Yte, Xrte = C.load_split_data("test")

    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)

    Ftr = E.build_tabular_features(Xtr, Xrtr)
    Fva = E.build_tabular_features(Xva, Xrva)
    Fte = E.build_tabular_features(Xte, Xrte)
    print(f"Features: {Ftr.shape}")

    trainers = {
        "lightgbm": train_lgb,
        "xgboost": train_xgb,
        "catboost": train_catboost,
        "random_forest": train_rf,
    }

    results = {}
    for name, train_fn in trainers.items():
        t0 = time.time()
        models = train_fn(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va)
        train_time = time.time() - t0

        val_pos, _, _ = evaluate_model(models, Fva, Xrva)
        te_pos, _, _ = evaluate_model(models, Fte, Xrte)
        val_m = C.summarize(val_pos, Yva[:, :, :2])
        te_m = C.summarize(te_pos, Yte[:, :, :2])
        results[name] = {"val": val_m, "test": te_m, "train_time_s": train_time}
        print(f"\n{name} (train {train_time:.1f}s):")
        print(f"  VAL  6h={val_m['6']['track_error_km_mean']:.2f}  "
              f"12h={val_m['12']['track_error_km_mean']:.2f}  "
              f"24h={val_m['24']['track_error_km_mean']:.2f}")
        print(f"  TEST 6h={te_m['6']['track_error_km_mean']:.2f}  "
              f"12h={te_m['12']['track_error_km_mean']:.2f}  "
              f"24h={te_m['24']['track_error_km_mean']:.2f}")

    mv_pos = C.movement_vector_pred(Xrte)
    mv_m = C.summarize(mv_pos, Yte[:, :, :2])
    results["movement_vector"] = {"test": mv_m}
    print(f"\nmovement-vector (baseline):")
    print(f"  TEST 6h={mv_m['6']['track_error_km_mean']:.2f}  "
          f"12h={mv_m['12']['track_error_km_mean']:.2f}  "
          f"24h={mv_m['24']['track_error_km_mean']:.2f}")

    out = RESULTS / "P4_E07_BOOSTING_COMPARE.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
