"""EXP-P4-11 — Storm-level cross-validation.

Cyclone-disjoint GroupKFold CV on train+val data.
Never touches the test set for model selection.

Protocol:
  * Combine train+val metadata to get all non-test cyclones.
  * GroupKFold by cyclone_id (5 folds).
  * For each fold: train on 4/5, validate on 1/5 (cyclone-disjoint).
  * Preprocessing (feature engineering) is fit on train fold only.
  * Normalization is fit on train fold only.
  * Report per-fold and aggregate: mean, median, std of track error at 6h/12h/24h.
  * Compare LightGBM, CatBoost, movement-vector.
"""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
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
N_FOLDS = 5


def load_all_non_test():
    """Load train + val as a combined pool for CV."""
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    meta_tr = C.load_meta("train")
    meta_va = C.load_meta("val")
    # Stack
    X_eng = np.concatenate([Xtr, Xva], axis=0)
    Y = np.concatenate([Ytr, Yva], axis=0)
    X_raw = np.concatenate([Xrtr, Xrva], axis=0)
    meta = pd.concat([meta_tr, meta_va], axis=0, ignore_index=True)
    return X_eng, Y, X_raw, meta


def train_lgb_fold(Xtr, Ytr, Xrtr, Xva, Yva, Xrva):
    """Train LightGBM displacement models on train fold, return val predictions."""
    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)
    Ftr = E.build_tabular_features(Xtr, Xrtr)
    Fva = E.build_tabular_features(Xva, Xrva)

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

    dlat_p = np.column_stack([models[f"{h}h_lat"].predict(Fva.values) for h in C.HORIZONS])
    dlon_p = np.column_stack([models[f"{h}h_lon"].predict(Fva.values) for h in C.HORIZONS])
    pos = E.displacement_to_position(Xrva, dlat_p, dlon_p)
    return pos


def train_catboost_fold(Xtr, Ytr, Xrtr, Xva, Yva, Xrva):
    """Train CatBoost displacement models on train fold, return val predictions."""
    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)
    Ftr = E.build_tabular_features(Xtr, Xrtr)
    Fva = E.build_tabular_features(Xva, Xrva)

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

    dlat_p = np.column_stack([models[f"{h}h_lat"].predict(Fva.values) for h in C.HORIZONS])
    dlon_p = np.column_stack([models[f"{h}h_lon"].predict(Fva.values) for h in C.HORIZONS])
    pos = E.displacement_to_position(Xrva, dlat_p, dlon_p)
    return pos


def compute_mv_predictions(Xrva, Yva):
    """Compute movement-vector predictions and track error."""
    mv_pos = C.movement_vector_pred(Xrva)
    return mv_pos


def main():
    X_eng, Y, X_raw, meta = load_all_non_test()
    print(f"Combined pool: {X_eng.shape[0]} samples, {meta['cyclone_id'].nunique()} cyclones")

    cyclone_ids = meta["cyclone_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    results = {"lgb": [], "catboost": [], "mv": []}
    fold_details = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_eng, Y, cyclone_ids)):
        t0 = time.time()
        Xtr_f, Y_f, Xrtr_f = X_eng[tr_idx], Y[tr_idx], X_raw[tr_idx]
        Xva_f, Yva_f, Xrva_f = X_eng[va_idx], Y[va_idx], X_raw[va_idx]
        va_cyclones = set(cyclone_ids[va_idx])
        print(f"\nFold {fold_idx+1}/{N_FOLDS}: train={len(tr_idx)}, val={len(va_idx)}, "
              f"val_cyclones={len(va_cyclones)}")

        # LightGBM
        lgb_pos = train_lgb_fold(Xtr_f, Y_f, Xrtr_f, Xva_f, Yva_f, Xrva_f)
        lgb_errs = C.track_error_per_sample(lgb_pos, Yva_f[:, :, :2])
        lgb_m = C.summarize(lgb_pos, Yva_f[:, :, :2])

        # CatBoost
        cat_pos = train_catboost_fold(Xtr_f, Y_f, Xrtr_f, Xva_f, Yva_f, Xrva_f)
        cat_errs = C.track_error_per_sample(cat_pos, Yva_f[:, :, :2])
        cat_m = C.summarize(cat_pos, Yva_f[:, :, :2])

        # Movement-vector
        mv_pos = compute_mv_predictions(Xrva_f, Yva_f)
        mv_errs = C.track_error_per_sample(mv_pos, Yva_f[:, :, :2])
        mv_m = C.summarize(mv_pos, Yva_f[:, :, :2])

        elapsed = time.time() - t0
        print(f"  LGB:  6h={lgb_m['6']['track_error_km_mean']:.2f}  "
              f"12h={lgb_m['12']['track_error_km_mean']:.2f}  "
              f"24h={lgb_m['24']['track_error_km_mean']:.2f}  ({elapsed:.1f}s)")
        print(f"  CAT:  6h={cat_m['6']['track_error_km_mean']:.2f}  "
              f"12h={cat_m['12']['track_error_km_mean']:.2f}  "
              f"24h={cat_m['24']['track_error_km_mean']:.2f}")
        print(f"  MV:   6h={mv_m['6']['track_error_km_mean']:.2f}  "
              f"12h={mv_m['12']['track_error_km_mean']:.2f}  "
              f"24h={mv_m['24']['track_error_km_mean']:.2f}")

        results["lgb"].append(lgb_m)
        results["catboost"].append(cat_m)
        results["mv"].append(mv_m)

        fold_details.append({
            "fold": fold_idx + 1,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "val_cyclones": sorted(va_cyclones),
            "lgb": lgb_m,
            "catboost": cat_m,
            "mv": mv_m,
            "time_s": elapsed,
        })

    # Aggregate across folds (mean of per-fold means)
    agg = {}
    for model_name in ["lgb", "catboost", "mv"]:
        agg[model_name] = {}
        for h in C.HORIZONS:
            means = [r[str(h)]["track_error_km_mean"] for r in results[model_name]]
            medians = [r[str(h)]["track_error_km_median"] for r in results[model_name]]
            agg[model_name][str(h)] = {
                "mean_of_means": float(np.mean(means)),
                "std_of_means": float(np.std(means)),
                "mean_of_medians": float(np.mean(medians)),
                "fold_values": [float(m) for m in means],
            }

    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATE STORM-CV RESULTS (mean of per-fold means, km)")
    print("=" * 70)
    for model_name, label in [("lgb", "LightGBM"), ("catboost", "CatBoost"), ("mv", "Movement-Vector")]:
        print(f"\n{label}:")
        for h in C.HORIZONS:
            a = agg[model_name][str(h)]
            print(f"  {h:2d}h: {a['mean_of_means']:.2f} +/- {a['std_of_means']:.2f}  "
                  f"folds: {[f'{v:.1f}' for v in a['fold_values']]}")

    # Winner per horizon
    print("\n" + "=" * 70)
    print("WINNER PER HORIZON (lowest mean-of-means)")
    print("=" * 70)
    for h in C.HORIZONS:
        scores = {name: agg[name][str(h)]["mean_of_means"] for name in ["lgb", "catboost", "mv"]}
        winner = min(scores, key=scores.get)
        print(f"  {h:2d}h: {winner} ({scores[winner]:.2f} km)")

    out = RESULTS / "P4_E11_STORM_CV.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "n_folds": N_FOLDS,
            "total_samples": X_eng.shape[0],
            "total_cyclones": int(meta["cyclone_id"].nunique()),
            "fold_details": fold_details,
            "aggregate": agg,
        }, f, indent=2)
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
