"""P4-E10 — Error analysis.

Analyze the LightGBM champion model's errors across the test set:
  * Per-horizon error distribution (mean, median, std, percentiles).
  * Error by storm intensity (wind speed quartiles).
  * Error by lead time correlation.
  * Worst-case errors (top-10 cases).
  * Error vs initial speed / heading change.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

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


def main():
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    Xte, Yte, Xrte = C.load_split_data("test")

    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)

    Ftr = E.build_tabular_features(Xtr, Xrtr)
    Fva = E.build_tabular_features(Xva, Xrva)
    Fte = E.build_tabular_features(Xte, Xrte)

    models = train_lgb(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va)

    # Predictions
    dlat_pred = np.column_stack([models[f"{h}h_lat"].predict(Fte.values) for h in C.HORIZONS])
    dlon_pred = np.column_stack([models[f"{h}h_lon"].predict(Fte.values) for h in C.HORIZONS])
    pred_pos = E.displacement_to_position(Xrte, dlat_pred, dlon_pred)

    # Per-sample errors
    errs = C.track_error_per_sample(pred_pos, Yte[:, :, :2])  # (N_test, 3)

    # 1. Per-horizon error distribution
    distributions = {}
    for hi, h in enumerate(C.HORIZONS):
        e = errs[:, hi]
        distributions[str(h)] = {
            "mean": float(np.mean(e)),
            "median": float(np.median(e)),
            "std": float(np.std(e)),
            "p25": float(np.percentile(e, 25)),
            "p75": float(np.percentile(e, 75)),
            "p90": float(np.percentile(e, 90)),
            "p95": float(np.percentile(e, 95)),
            "max": float(np.max(e)),
            "min": float(np.min(e)),
        }

    # 2. Error by storm intensity (wind speed at t0 quartiles)
    wind_t0 = Xrte[:, 4, 2]
    quartiles = np.percentile(wind_t0, [25, 50, 75])
    intensity_groups = {
        "weak": wind_t0 <= quartiles[0],
        "moderate": (wind_t0 > quartiles[0]) & (wind_t0 <= quartiles[2]),
        "strong": wind_t0 > quartiles[2],
    }
    intensity_errors = {}
    for gname, mask in intensity_groups.items():
        if mask.sum() == 0:
            continue
        e = errs[mask]
        intensity_errors[gname] = {
            "n": int(mask.sum()),
            "6h_mean": float(np.mean(e[:, 0])),
            "12h_mean": float(np.mean(e[:, 1])),
            "24h_mean": float(np.mean(e[:, 2])),
        }

    # 3. Error by initial movement speed
    speed_t0 = C.haversine_km(Xrte[:, 3, 0], Xrte[:, 3, 1],
                               Xrte[:, 4, 0], Xrte[:, 4, 1])
    speed_quartiles = np.percentile(speed_t0, [25, 50, 75])
    speed_groups = {
        "slow": speed_t0 <= speed_quartiles[0],
        "medium": (speed_t0 > speed_quartiles[0]) & (speed_t0 <= speed_quartiles[2]),
        "fast": speed_t0 > speed_quartiles[2],
    }
    speed_errors = {}
    for gname, mask in speed_groups.items():
        if mask.sum() == 0:
            continue
        e = errs[mask]
        speed_errors[gname] = {
            "n": int(mask.sum()),
            "6h_mean": float(np.mean(e[:, 0])),
            "12h_mean": float(np.mean(e[:, 1])),
            "24h_mean": float(np.mean(e[:, 2])),
        }

    # 4. Worst-case errors (top-10 per horizon)
    worst_cases = {}
    for hi, h in enumerate(C.HORIZONS):
        idx = np.argsort(errs[:, hi])[-10:][::-1]
        worst_cases[str(h)] = [{
            "sample_idx": int(i),
            "error_km": float(errs[i, hi]),
            "wind_t0": float(wind_t0[i]),
            "lat_t0": float(Xrte[i, 4, 0]),
            "lon_t0": float(Xrte[i, 4, 1]),
        } for i in idx]

    # 5. Feature importance from first model (6h_lat)
    fi = models["6h_lat"].feature_importances_
    fi_dict = dict(zip(Fte.columns, fi.tolist()))
    fi_sorted = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)

    results = {
        "n_test": int(errs.shape[0]),
        "horizon_distributions": distributions,
        "by_storm_intensity": intensity_errors,
        "by_initial_speed": speed_errors,
        "worst_cases": worst_cases,
        "top15_features_6h_lat": fi_sorted[:15],
        "mean_track_error_by_horizon": {
            str(h): float(np.mean(errs[:, hi]))
            for hi, h in enumerate(C.HORIZONS)
        },
    }

    out = RESULTS / "P4_E10_ERROR_ANALYSIS.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"WROTE {out}")

    # Print summary
    print(f"\nTest set: {errs.shape[0]} samples")
    print("\nError distribution (km):")
    for h in C.HORIZONS:
        d = distributions[str(h)]
        print(f"  {h:2d}h: mean={d['mean']:.1f}  median={d['median']:.1f}  "
              f"std={d['std']:.1f}  p90={d['p90']:.1f}  max={d['max']:.1f}")
    print("\nError by storm intensity:")
    for gname, info in intensity_errors.items():
        print(f"  {gname:10s} (n={info['n']:3d}): "
              f"6h={info['6h_mean']:.1f}  12h={info['12h_mean']:.1f}  "
              f"24h={info['24h_mean']:.1f}")
    print("\nError by initial speed:")
    for gname, info in speed_errors.items():
        print(f"  {gname:10s} (n={info['n']:3d}): "
              f"6h={info['6h_mean']:.1f}  12h={info['12h_mean']:.1f}  "
              f"24h={info['24h_mean']:.1f}")


if __name__ == "__main__":
    main()
