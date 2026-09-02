"""EXP-P4-12 — Error stratification analysis.

Diagnostic analysis on the untouched test set (NOT tuning).
Analyzes error patterns by storm characteristics.
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


def train_lgb(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va):
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


def train_catboost(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va):
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


def eval_model(models, F, Xr):
    dlat = np.column_stack([models[f"{h}h_lat"].predict(F.values) for h in C.HORIZONS])
    dlon = np.column_stack([models[f"{h}h_lon"].predict(F.values) for h in C.HORIZONS])
    pos = E.displacement_to_position(Xr, dlat, dlon)
    return pos, dlat, dlon


def per_cyclone_error(errs, cyclone_ids):
    """Compute per-cyclone mean error."""
    df = pd.DataFrame({"cyclone": cyclone_ids, "err_6h": errs[:, 0],
                        "err_12h": errs[:, 1], "err_24h": errs[:, 2]})
    return df.groupby("cyclone").agg(["mean", "count"]).reset_index()


def main():
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    Xte, Yte, Xrte = C.load_split_data("test")
    meta_te = C.load_meta("test")

    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)
    Ftr = E.build_tabular_features(Xtr, Xrtr)
    Fva = E.build_tabular_features(Xva, Xrva)
    Fte = E.build_tabular_features(Xte, Xrte)

    # Train models
    print("Training LightGBM...")
    lgb_models = train_lgb(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va)
    print("Training CatBoost...")
    cat_models = train_catboost(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va)

    # Evaluate on test
    lgb_pos, _, _ = eval_model(lgb_models, Fte, Xrte)
    cat_pos, _, _ = eval_model(cat_models, Fte, Xrte)
    mv_pos = C.movement_vector_pred(Xrte)

    lgb_errs = C.track_error_per_sample(lgb_pos, Yte[:, :, :2])
    cat_errs = C.track_error_per_sample(cat_pos, Yte[:, :, :2])
    mv_errs = C.track_error_per_sample(mv_pos, Yte[:, :, :2])

    # --- 1. Per-cyclone error ---
    cyclone_ids = meta_te["cyclone_id"].values
    per_cyc = {}
    for name, errs in [("lgb", lgb_errs), ("catboost", cat_errs), ("mv", mv_errs)]:
        df = pd.DataFrame({"cyclone": cyclone_ids, "err_6h": errs[:, 0],
                            "err_12h": errs[:, 1], "err_24h": errs[:, 2]})
        grp = df.groupby("cyclone").agg({"err_6h": ["mean", "count"],
                                          "err_12h": "mean", "err_24h": "mean"})
        grp.columns = ["_".join(c) for c in grp.columns]
        per_cyc[name] = grp.to_dict(orient="index")

    # Top error cyclones
    top_cyclones = {}
    for name, errs in [("lgb", lgb_errs), ("catboost", cat_errs), ("mv", mv_errs)]:
        df = pd.DataFrame({"cyclone": cyclone_ids, "err_24h": errs[:, 2]})
        top = df.groupby("cyclone")["err_24h"].mean().sort_values(ascending=False).head(5)
        top_cyclones[name] = {c: float(v) for c, v in top.items()}

    # --- 2. Error by wind intensity ---
    wind_t0 = Xrte[:, 4, 2]
    q25, q50, q75 = np.percentile(wind_t0, [25, 50, 75])
    intensity_bins = {
        f"weak_<={q25:.0f}": wind_t0 <= q25,
        f"moderate_{q25:.0f}-{q75:.0f}": (wind_t0 > q25) & (wind_t0 <= q75),
        f"strong_>{q75:.0f}": wind_t0 > q75,
    }
    intensity_errs = {}
    for bname, mask in intensity_bins.items():
        if mask.sum() == 0:
            continue
        for model_name, errs in [("lgb", lgb_errs), ("catboost", cat_errs), ("mv", mv_errs)]:
            e = errs[mask]
            key = f"{model_name}_{bname}"
            intensity_errs[key] = {
                "n": int(mask.sum()),
                "6h": float(np.mean(e[:, 0])),
                "12h": float(np.mean(e[:, 1])),
                "24h": float(np.mean(e[:, 2])),
            }

    # --- 3. Error by movement speed ---
    speed = C.haversine_km(Xrte[:, 3, 0], Xrte[:, 3, 1], Xrte[:, 4, 0], Xrte[:, 4, 1])
    sq25, sq75 = np.percentile(speed, [25, 75])
    speed_bins = {
        f"slow_<={sq25:.1f}": speed <= sq25,
        f"medium": (speed > sq25) & (speed <= sq75),
        f"fast_>{sq75:.1f}": speed > sq75,
    }
    speed_errs = {}
    for bname, mask in speed_bins.items():
        if mask.sum() == 0:
            continue
        for model_name, errs in [("lgb", lgb_errs), ("catboost", cat_errs), ("mv", mv_errs)]:
            e = errs[mask]
            key = f"{model_name}_{bname}"
            speed_errs[key] = {
                "n": int(mask.sum()),
                "6h": float(np.mean(e[:, 0])),
                "12h": float(np.mean(e[:, 1])),
                "24h": float(np.mean(e[:, 2])),
            }

    # --- 4. Error by turning rate ---
    h1 = C.bearing_degrees(Xrte[:, 3, 0], Xrte[:, 3, 1], Xrte[:, 4, 0], Xrte[:, 4, 1])
    h2 = C.bearing_degrees(Xrte[:, 2, 0], Xrte[:, 2, 1], Xrte[:, 3, 0], Xrte[:, 3, 1])
    turn = np.abs((h1 - h2 + 180) % 360 - 180)
    tq = np.median(turn)
    turn_bins = {
        f"straight_<{tq:.0f}deg": turn < tq,
        f"turning_>={tq:.0f}deg": turn >= tq,
    }
    turn_errs = {}
    for bname, mask in turn_bins.items():
        if mask.sum() == 0:
            continue
        for model_name, errs in [("lgb", lgb_errs), ("catboost", cat_errs), ("mv", mv_errs)]:
            e = errs[mask]
            key = f"{model_name}_{bname}"
            turn_errs[key] = {
                "n": int(mask.sum()),
                "6h": float(np.mean(e[:, 0])),
                "12h": float(np.mean(e[:, 1])),
                "24h": float(np.mean(e[:, 2])),
            }

    # --- 5. Stationary vs moving ---
    stationary = speed < 1.0
    move_errs = {}
    for bname, mask in [("stationary_<1km", stationary), ("moving_>1km", ~stationary)]:
        if mask.sum() == 0:
            continue
        for model_name, errs in [("lgb", lgb_errs), ("catboost", cat_errs), ("mv", mv_errs)]:
            e = errs[mask]
            key = f"{model_name}_{bname}"
            move_errs[key] = {
                "n": int(mask.sum()),
                "6h": float(np.mean(e[:, 0])),
                "12h": float(np.mean(e[:, 1])),
                "24h": float(np.mean(e[:, 2])),
            }

    # --- 6. MV vs ML regime comparison ---
    mv_better_6h = mv_errs[:, 0] < lgb_errs[:, 0]
    lgb_better_6h = lgb_errs[:, 0] < mv_errs[:, 0]
    regime_analysis = {
        "mv_better_6h_count": int(mv_better_6h.sum()),
        "lgb_better_6h_count": int(lgb_better_6h.sum()),
        "mv_better_24h_count": int((mv_errs[:, 2] < lgb_errs[:, 2]).sum()),
        "lgb_better_24h_count": int((lgb_errs[:, 2] < mv_errs[:, 2]).sum()),
    }

    # --- 7. Along-track vs cross-track ---
    # Decompose error into along-track and cross-track components
    def decompose_error(pred_pos, true_pos, Xr):
        """Decompose track error into along-track (speed) and cross-track (direction)."""
        lat0 = Xr[:, 4, 0]; lon0 = Xr[:, 4, 1]
        along = np.empty((pred_pos.shape[0], 3))
        cross = np.empty((pred_pos.shape[0], 3))
        for hi in range(3):
            # predicted and true displacement from t0
            pred_dlat = pred_pos[:, hi, 0] - lat0
            pred_dlon = pred_pos[:, hi, 1] - lon0
            true_dlat = true_pos[:, hi, 0] - lat0
            true_dlon = true_pos[:, hi, 1] - lon0
            # along-track: component of error in direction of true movement
            true_mag = np.sqrt(true_dlat**2 + true_dlon**2) + 1e-10
            unit_lat = true_dlat / true_mag
            unit_lon = true_dlon / true_mag
            err_dlat = pred_dlat - true_dlat
            err_dlon = pred_dlon - true_dlon
            along[:, hi] = np.abs(err_dlat * unit_lat + err_dlon * unit_lon)
            cross[:, hi] = np.abs(err_dlat * (-unit_lon) + err_dlon * unit_lat)
        return along, cross

    lgb_along, lgb_cross = decompose_error(lgb_pos, Yte[:, :, :2], Xrte)
    mv_along, mv_cross = decompose_error(mv_pos, Yte[:, :, :2], Xrte)

    track_decomposition = {}
    for name, along, cross in [("lgb", lgb_along, lgb_cross), ("mv", mv_along, mv_cross)]:
        for hi, h in enumerate(C.HORIZONS):
            track_decomposition[f"{name}_{h}h"] = {
                "along_track_mean": float(np.mean(along[:, hi])),
                "cross_track_mean": float(np.mean(cross[:, hi])),
                "total_mean": float(np.mean(along[:, hi] + cross[:, hi])),
            }

    # --- 8. Predicted displacement magnitude ---
    lgb_dlat = np.column_stack([lgb_models[f"{h}h_lat"].predict(Fte.values) for h in C.HORIZONS])
    lgb_dlon = np.column_stack([lgb_models[f"{h}h_lon"].predict(Fte.values) for h in C.HORIZONS])
    lgb_disp_mag = np.sqrt(lgb_dlat**2 + lgb_dlon**2)
    true_dlat, true_dlon = E.displacement_targets(Xrte, Yte)
    true_disp_mag = np.sqrt(true_dlat**2 + true_dlon**2)

    disp_analysis = {}
    for hi, h in enumerate(C.HORIZONS):
        corr = np.corrcoef(lgb_disp_mag[:, hi], lgb_errs[:, hi])[0, 1]
        disp_analysis[str(h)] = {
            "pred_disp_mean": float(np.mean(lgb_disp_mag[:, hi])),
            "true_disp_mean": float(np.mean(true_disp_mag[:, hi])),
            "disp_error_corr": float(corr),
        }

    results = {
        "n_test": int(Xte.shape[0]),
        "n_cyclones": int(meta_te["cyclone_id"].nunique()),
        "test_cyclones": sorted(cyclone_ids.tolist()),
        "top_error_cyclones": top_cyclones,
        "by_intensity": intensity_errs,
        "by_speed": speed_errs,
        "by_turning": turn_errs,
        "stationary_vs_moving": move_errs,
        "regime_analysis": regime_analysis,
        "track_decomposition": track_decomposition,
        "displacement_analysis": disp_analysis,
        "overall": {
            "lgb": C.summarize(lgb_pos, Yte[:, :, :2]),
            "catboost": C.summarize(cat_pos, Yte[:, :, :2]),
            "mv": C.summarize(mv_pos, Yte[:, :, :2]),
        },
    }

    out = RESULTS / "P4_E12_ERROR_STRATIFICATION.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"WROTE {out}")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print(f"Test: {Xte.shape[0]} samples, {meta_te['cyclone_id'].nunique()} cyclones")
    print(f"\nTop error cyclones (LGB, 24h):")
    for c, v in list(top_cyclones["lgb"].items())[:3]:
        print(f"  {c}: {v:.1f} km")
    print(f"\nRegime: MV better at 6h: {regime_analysis['mv_better_6h_count']}, "
          f"LGB better at 6h: {regime_analysis['lgb_better_6h_count']}")
    print(f"Regime: MV better at 24h: {regime_analysis['mv_better_24h_count']}, "
          f"LGB better at 24h: {regime_analysis['lgb_better_24h_count']}")


if __name__ == "__main__":
    main()
