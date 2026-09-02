"""P4-E06 — Movement-vector + LightGBM displacement blend (model_improvement).

Blending is performed in DISPLACEMENT space (dlat_km, dlon_km from the current
t0 position). This is equivalent to the position blend
    blended_pos = w * movement_pos + (1-w) * lightgbm_pos
when both positions live in the same continuous coordinate frame, and it is the
numerically robust form to take (longitude wraparound is handled once, at
reconstruction) and it guarantees the reconstruction uses the exact same
current position the model received.

Protocol:
  * LightGBM displacement regressors (6h/12h/24h x lat/lon) are trained on
    TRAIN only with early stopping on VALIDATION.
  * Blend weight w in {0.00, 0.05, ..., 1.00} is chosen per horizon on
    VALIDATION ONLY (minimizing mean track error).
  * Selected validation weights are frozen, then evaluated ONCE on the
    untouched CLEAN test set.
  * No test-set tuning, no test selection, no cherry-picking.

Authoritative frozen baselines (CLEAN 198-row test):
  persistence    = 64.686 / 123.936 / 227.327 km
  movement-vector= 38.051 /  80.505 / 180.661 km
  EXP005         = 91.43  / 119.76  / 188.24  km
  LightGBM E04   = 35.128 /  73.072 / 159.823 km (test; config identical below)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

SCRIPTS = Path(__file__).resolve().parent
RESULTS = SCRIPTS.parent / "results"
sys.path.insert(0, str(SCRIPTS))
import p4_common as C
import p4_experiments as E   # build_tabular_features, displacement_*, displacement_to_position

SEED = 42
np.random.seed(SEED)

GRID = np.round(np.arange(0.00, 1.0001, 0.05), 2)


def train_displacement_models(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va):
    """Train LightGBM displacement regressors, identical to E04 config."""
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


def displacement_from_latlon(X_raw, latlon):
    """Convert a predicted (N,3,2) absolute latlon back to displacement (N,3) km
    from the same t0 position used by the model. Mirrors E.displacement_targets.
    """
    lat0 = X_raw[:, 4, 0]; lon0 = X_raw[:, 4, 1]
    N = X_raw.shape[0]
    dlat = np.empty((N, 3)); dlon = np.empty((N, 3))
    for hi in range(3):
        dlat[:, hi] = C.haversine_km(lat0, lon0, latlon[:, hi, 0], lon0) * \
                      np.sign(latlon[:, hi, 0] - lat0)
        dlon[:, hi] = C.haversine_km(lat0, lon0, lat0, latlon[:, hi, 1]) * \
                      np.sign(C.wrap_lon_delta(lon0, latlon[:, hi, 1]))
    return dlat, dlon


def mean_track_error_column(pos, true_latlon):
    err = C.track_error_per_sample(pos, true_latlon)   # (N,3)
    return np.mean(err, axis=0)                          # (3,)


def main():
    # ---- Load splits ----
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    Xte, Yte, Xrte = C.load_split_data("test")
    print("splits  train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)

    # ---- Displacement targets ----
    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)
    dlat_te, dlon_te = E.displacement_targets(Xrte, Yte)

    # ---- Tabular features ----
    Ftr = E.build_tabular_features(Xtr, Xrtr)
    Fva = E.build_tabular_features(Xva, Xrva)
    Fte = E.build_tabular_features(Xte, Xrte)
    print("tabular features", Ftr.shape)

    # ---- Train LightGBM displacement models (IDENTICAL E04 config) ----
    models = train_displacement_models(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va)

    # ---- Displacement predictions (val + test) ----
    def lgb_displacement(F):
        dl = np.column_stack([models[f"{h}h_lat"].predict(F.values) for h in C.HORIZONS])
        dlo = np.column_stack([models[f"{h}h_lon"].predict(F.values) for h in C.HORIZONS])
        return dl, dlo
    lgb_dlat_va, lgb_dlon_va = lgb_displacement(Fva)
    lgb_dlat_te, lgb_dlon_te = lgb_displacement(Fte)

    # ---- Movement-vector displacement from CURRENT position ----
    mv_va = C.movement_vector_pred(Xrva)
    mv_te = C.movement_vector_pred(Xrte)
    mv_dlat_va, mv_dlon_va = displacement_from_latlon(Xrva, mv_va[:, :, :2])
    mv_dlat_te, mv_dlon_te = displacement_from_latlon(Xrte, mv_te[:, :, :2])

    # ---- Validation performance of the two components ----
    lgb_pos_va = E.displacement_to_position(Xrva, lgb_dlat_va, lgb_dlon_va)
    mv_pos_va = E.displacement_to_position(Xrva, mv_dlat_va, mv_dlon_va)
    lgb_val = mean_track_error_column(lgb_pos_va, Yva[:, :, :2])
    mv_val = mean_track_error_column(mv_pos_va, Yva[:, :, :2])
    print("VAL  movement-vector:", np.round(mv_val, 3))
    print("VAL  LightGBM       :", np.round(lgb_val, 3))

    # ---- Search w per horizon on VALIDATION ----
    selected_w = {}
    val_best = {}
    for hi, h in enumerate(C.HORIZONS):
        best_w, best_e = None, np.inf
        for w in GRID:
            bdl_h = w * mv_dlat_va[:, hi] + (1 - w) * lgb_dlat_va[:, hi]
            bdo_h = w * mv_dlon_va[:, hi] + (1 - w) * lgb_dlon_va[:, hi]
            bdl_full = np.zeros_like(mv_dlat_va)
            bdo_full = np.zeros_like(mv_dlon_va)
            bdl_full[:, hi] = bdl_h
            bdo_full[:, hi] = bdo_h
            bp = E.displacement_to_position(Xrva, bdl_full, bdo_full)
            errs = C.track_error_per_sample(bp, Yva[:, :, :2])  # (N,3)
            e = np.mean(errs[:, hi])
            if e < best_e:
                best_e, best_w = e, float(w)
        selected_w[str(h)] = best_w
        val_best[str(h)] = float(best_e)
        print(f"VAL {h}h  best w={best_w:.2f}  blend_mte={best_e:.3f}  "
              f"(mv={mv_val[hi]:.3f}, lgb={lgb_val[hi]:.3f})")
    print("selected weights (from validation):", selected_w)

    # ---- Freeze and evaluate ONCE on test ----
    te_out = {}
    for hi, h in enumerate(C.HORIZONS):
        w = selected_w[str(h)]
        bdl_h = w * mv_dlat_te[:, hi] + (1 - w) * lgb_dlat_te[:, hi]
        bdo_h = w * mv_dlon_te[:, hi] + (1 - w) * lgb_dlon_te[:, hi]
        bdl_full = np.zeros_like(mv_dlat_te)
        bdo_full = np.zeros_like(mv_dlon_te)
        bdl_full[:, hi] = bdl_h
        bdo_full[:, hi] = bdo_h
        bp = E.displacement_to_position(Xrte, bdl_full, bdo_full)
        errs = C.track_error_per_sample(bp, Yte[:, :, :2])  # (N,3)
        e_h = errs[:, hi]
        te_out[str(h)] = {
            "w": w,
            "track_error_km_mean": float(np.mean(e_h)),
            "track_error_km_median": float(np.median(e_h)),
            "track_error_km_std": float(np.std(e_h)),
        }
        print(f"TEST {h}h  w={w:.2f}  blend={np.mean(e_h):.3f}")

    mv_te_pos = E.displacement_to_position(Xrte, mv_dlat_te, mv_dlon_te)
    lgb_te_pos = E.displacement_to_position(Xrte, lgb_dlat_te, lgb_dlon_te)
    mv_test = C.summarize(mv_te_pos, Yte[:, :, :2])
    lgb_test = C.summarize(lgb_te_pos, Yte[:, :, :2])

    # ---- Improvement table ----
    def row(metric):  # returns {6,12,24} mean track error
        return {h: metric[str(h)]["track_error_km_mean"] for h in C.HORIZONS}

    mv_m, lgb_m = row(mv_test), row(lgb_test)
    blend_m = {h: te_out[str(h)]["track_error_km_mean"] for h in C.HORIZONS}

    print("\n=== FROZEN TEST (mean track error km) ===")
    print("MODEL           " + "".join(f"{h:>10s}" for h in ["6h", "12h", "24h"]))
    print("movement-vector " + "".join(f"{mv_m[h]:>10.3f}" for h in C.HORIZONS))
    print("LightGBM        " + "".join(f"{lgb_m[h]:>10.3f}" for h in C.HORIZONS))
    print("E06 blend       " + "".join(f"{blend_m[h]:>10.3f}" for h in C.HORIZONS))

    print("\n=== E06 blend vs movement-vector (test) ===")
    for h in C.HORIZONS:
        d = blend_m[h] - mv_m[h]
        print(f"{h}h  delta={d:+.3f} km  pct={100*d/mv_m[h]:+.2f}%")
    print("=== E06 blend vs LightGBM (test) ===")
    for h in C.HORIZONS:
        d = blend_m[h] - lgb_m[h]
        print(f"{h}h  delta={d:+.3f} km  pct={100*d/lgb_m[h]:+.2f}%")

    # ---- Verdict on VALIDATION evidence ----
    improved_over_both = all(val_best[str(h)] < min(mv_val[hi], lgb_val[hi]) - 1e-9
                             for hi, h in enumerate(C.HORIZONS))
    print("\nVALIDATION winner per horizon (better than both components):",
          {h: val_best[str(h)] < min(mv_val[hi], lgb_val[hi]) - 1e-9
           for hi, h in enumerate(C.HORIZONS)})
    verdict = "IMPROVED" if improved_over_both else \
              ("INCONCLUSIVE" if any(val_best[str(h)] < min(mv_val[hi], lgb_val[hi]) - 1e-9
                                     for hi, h in enumerate(C.HORIZONS)) else "NOT_IMPROVED")
    print("E06 VERDICT (validation evidence):", verdict)

    evidence = {
        "experiment": "P4-E06",
        "description": "Movement-vector + LightGBM displacement blend",
        "dataset": "canonical_chronological_clean (train/val/test)",
        "protocol": {
            "fit": "train",
            "selection": "validation (weight w per horizon, grid 0..1 step 0.05)",
            "test": "untouched CLEAN test, evaluated once with frozen weights"
        },
        "split_sizes": {"train": Xtr.shape[0], "val": Xva.shape[0], "test": Xte.shape[0]},
        "leakage": "cyclone-disjoint splits; displacement from same t0 used by model; "
                   "w chosen on validation only",
        "validation_movement_vector_km": {h: float(mv_val[hi]) for hi, h in enumerate(C.HORIZONS)},
        "validation_lightgbm_km": {h: float(lgb_val[hi]) for hi, h in enumerate(C.HORIZONS)},
        "validation_blend_best_km": val_best,
        "selected_weights_w": selected_w,
        "frozen_test_movement_vector_km": mv_m,
        "frozen_test_lightgbm_km": lgb_m,
        "frozen_test_e06_blend_km": blend_m,
        "e06_vs_movement_km": {h: blend_m[h] - mv_m[h] for h in C.HORIZONS},
        "e06_vs_movement_pct": {h: 100*(blend_m[h]-mv_m[h])/mv_m[h] for h in C.HORIZONS},
        "e06_vs_lightgbm_km": {h: blend_m[h] - lgb_m[h] for h in C.HORIZONS},
        "e06_vs_lightgbm_pct": {h: 100*(blend_m[h]-lgb_m[h])/lgb_m[h] for h in C.HORIZONS},
        "verdict": verdict,
        "control_baselines_km": {
            "persistence": {"6": 64.686, "12": 123.936, "24": 227.327},
            "movement_vector": mv_m,
            "exp005": {"6": 91.43, "12": 119.76, "24": 188.24},
        },
    }
    out = RESULTS / "P4_E06_BLEND.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    print("WROTE", out)


if __name__ == "__main__":
    main()
