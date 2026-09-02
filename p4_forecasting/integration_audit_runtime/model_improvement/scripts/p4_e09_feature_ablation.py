"""P4-E09 — Feature ablation study.

Remove each feature group one at a time from the 26-feature tabular set and
measure the impact on validation and test track error.  This identifies which
feature groups contribute most to forecast accuracy.

Feature groups:
  1. state0_*: current state (lat, lon, wind, pressure, sst, wind_u, wind_v)
  2. velocity_*: step velocity (v_lat_step1, v_lon_step1, v_lat_step2, v_lon_step2)
  3. velocity_old: oldest velocity (v_lat_step4, v_lon_step4)
  4. acceleration: (a_lat, a_lon)
  5. trend: (v24_lat, v24_lon)
  6. turn_rate + heading_step1
  7. rolling mean: (v_lat_mean2, v_lon_mean2)
  8. tendencies: (wind_tend, pres_tend)
  9. environment: (env_speed0, env_u0, env_v0)

Protocol:
  * Full model = baseline LightGBM displacement (26 features).
  * Each ablation removes one group, trains on TRAIN, evaluates on val+test.
  * No test-set tuning; ablations evaluated identically.
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

FEATURE_GROUPS = {
    "state0": ["state0_lat", "state0_lon", "state0_wind_speed", "state0_pressure",
               "state0_sst", "state0_wind_u", "state0_wind_v"],
    "velocity_recent": ["v_lat_step1", "v_lon_step1", "v_lat_step2", "v_lon_step2"],
    "velocity_old": ["v_lat_step4", "v_lon_step4"],
    "acceleration": ["a_lat", "a_lon"],
    "trend_24h": ["v24_lat", "v24_lon"],
    "turning": ["turn_rate", "heading_step1"],
    "rolling_mean": ["v_lat_mean2", "v_lon_mean2"],
    "tendencies": ["wind_tend", "pres_tend"],
    "environment": ["env_speed0", "env_u0", "env_v0"],
}


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


def eval_model(models, F, Xr):
    dlat = np.column_stack([models[f"{h}h_lat"].predict(F.values) for h in C.HORIZONS])
    dlon = np.column_stack([models[f"{h}h_lon"].predict(F.values) for h in C.HORIZONS])
    pos = E.displacement_to_position(Xr, dlat, dlon)
    return pos


def run_ablation(name, Ftr_all, Fva_all, Fte_all, Xrva, Xrte, Yva, Yte,
                 dlat_tr, dlon_tr, dlat_va, dlon_va, drop_cols):
    cols = [c for c in Ftr_all.columns if c not in drop_cols]
    Ftr = Ftr_all[cols]
    Fva = Fva_all[cols]
    Fte = Fte_all[cols]

    models = train_lgb(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va)
    val_pos = eval_model(models, Fva, Xrva)
    te_pos = eval_model(models, Fte, Xrte)
    val_m = C.summarize(val_pos, Yva[:, :, :2])
    te_m = C.summarize(te_pos, Yte[:, :, :2])
    return val_m, te_m, len(cols)


def main():
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    Xte, Yte, Xrte = C.load_split_data("test")

    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)

    Ftr_all = E.build_tabular_features(Xtr, Xrtr)
    Fva_all = E.build_tabular_features(Xva, Xrva)
    Fte_all = E.build_tabular_features(Xte, Xrte)
    full_cols = list(Ftr_all.columns)
    print(f"Full features: {len(full_cols)}")

    # Baseline: all features
    print("Training baseline (all features)...")
    bl_models = train_lgb(Ftr_all, Fva_all, dlat_tr, dlon_tr, dlat_va, dlon_va)
    bl_val_pos = eval_model(bl_models, Fva_all, Xrva)
    bl_te_pos = eval_model(bl_models, Fte_all, Xrte)
    bl_val = C.summarize(bl_val_pos, Yva[:, :, :2])
    bl_te = C.summarize(bl_te_pos, Yte[:, :, :2])
    bl_val_mean = np.mean([bl_val[str(h)]["track_error_km_mean"] for h in C.HORIZONS])
    bl_te_mean = np.mean([bl_te[str(h)]["track_error_km_mean"] for h in C.HORIZONS])
    print(f"  Full: val={bl_val_mean:.2f} test={bl_te_mean:.2f}")

    results = {
        "baseline": {"val": bl_val, "test": bl_te, "n_features": len(full_cols)},
        "ablations": {},
    }

    for gname, drop_cols in FEATURE_GROUPS.items():
        valid_drop = [c for c in drop_cols if c in full_cols]
        if not valid_drop:
            continue
        print(f"  Ablating {gname} ({len(valid_drop)} features)...")
        val_m, te_m, n = run_ablation(
            gname, Ftr_all, Fva_all, Fte_all, Xrva, Xrte, Yva, Yte,
            dlat_tr, dlon_tr, dlat_va, dlon_va, valid_drop)
        val_mean = np.mean([val_m[str(h)]["track_error_km_mean"] for h in C.HORIZONS])
        te_mean = np.mean([te_m[str(h)]["track_error_km_mean"] for h in C.HORIZONS])
        delta_val = val_mean - bl_val_mean
        delta_te = te_mean - bl_te_mean
        results["ablations"][gname] = {
            "dropped": valid_drop,
            "n_features_remaining": n,
            "val": val_m,
            "test": te_m,
            "delta_val_km": delta_val,
            "delta_test_km": delta_te,
        }
        print(f"    val={val_mean:.2f} (delta {delta_val:+.2f}) "
              f"test={te_mean:.2f} (delta {delta_te:+.2f})")

    out = RESULTS / "P4_E09_FEATURE_ABLATION.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWROTE {out}")

    # Print summary
    print("\n=== Feature importance ranking (by test delta) ===")
    ranked = sorted(results["ablations"].items(),
                    key=lambda x: x[1]["delta_test_km"], reverse=True)
    for gname, info in ranked:
        print(f"  {gname:20s}  val_delta={info['delta_val_km']:+.2f}  "
              f"test_delta={info['delta_test_km']:+.2f}")


if __name__ == "__main__":
    main()
