"""EXP-P4-15 — Conditional hybrid / gated forecast.

Only justified if E13/E14 show that different methods win in different regimes.
Uses validation data to learn a regime-dependent selector.
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


def build_regime_features(X_raw):
    """Build simple regime features for gating."""
    wind_t0 = X_raw[:, 4, 2]
    speed = C.haversine_km(X_raw[:, 3, 0], X_raw[:, 3, 1], X_raw[:, 4, 0], X_raw[:, 4, 1])
    h1 = C.bearing_degrees(X_raw[:, 3, 0], X_raw[:, 3, 1], X_raw[:, 4, 0], X_raw[:, 4, 1])
    h2 = C.bearing_degrees(X_raw[:, 2, 0], X_raw[:, 2, 1], X_raw[:, 3, 0], X_raw[:, 3, 1])
    turn = np.abs((h1 - h2 + 180) % 360 - 180)
    lat_t0 = X_raw[:, 4, 0]
    env_speed = np.hypot(X_raw[:, 4, 5], X_raw[:, 4, 6])
    df = pd.DataFrame({
        "wind_t0": wind_t0,
        "speed": speed,
        "turn_rate": turn,
        "lat_t0": lat_t0,
        "env_speed": env_speed,
    })
    return df.astype(np.float32)


def train_displacement_models(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va):
    """Train LightGBM displacement models."""
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


def train_residual_models(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va):
    """Train LightGBM residual models."""
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


def predict_displacement(models, F, Xr):
    dlat = np.column_stack([models[f"{h}h_lat"].predict(F.values) for h in C.HORIZONS])
    dlon = np.column_stack([models[f"{h}h_lon"].predict(F.values) for h in C.HORIZONS])
    return E.displacement_to_position(Xr, dlat, dlon)


def predict_residual(models, F, mv_dlat, mv_dlon, Xr):
    res_lat = np.column_stack([models[f"{h}h_lat"].predict(F.values) for h in C.HORIZONS])
    res_lon = np.column_stack([models[f"{h}h_lon"].predict(F.values) for h in C.HORIZONS])
    return E.displacement_to_position(Xr, mv_dlat + res_lat, mv_dlon + res_lon)


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

    # Regime features
    Rtr = build_regime_features(Xrtr)
    Rva = build_regime_features(Xrva)
    Rte = build_regime_features(Xrte)

    # --- Baselines ---
    mv_val_m = C.summarize(C.movement_vector_pred(Xrva), Yva[:, :, :2])
    mv_te_m = C.summarize(C.movement_vector_pred(Xrte), Yte[:, :, :2])

    # --- Displacement models ---
    print("Training displacement models...")
    disp_models = train_displacement_models(Ftr, Fva, dlat_tr, dlon_tr, dlat_va, dlon_va)
    disp_val = predict_displacement(disp_models, Fva, Xrva)
    disp_te = predict_displacement(disp_models, Fte, Xrte)
    disp_val_m = C.summarize(disp_val, Yva[:, :, :2])
    disp_te_m = C.summarize(disp_te, Yte[:, :, :2])

    # --- Residual models ---
    print("Training residual models...")
    res_models = train_residual_models(Ftr, Fva, res_lat_tr, res_lon_tr, res_lat_va, res_lon_va)
    res_val = predict_residual(res_models, Fva, mv_dlat_va, mv_dlon_va, Xrva)
    res_te = predict_residual(res_models, Fte, mv_dlat_te, mv_dlon_te, Xrte)
    res_val_m = C.summarize(res_val, Yva[:, :, :2])
    res_te_m = C.summarize(res_te, Yte[:, :, :2])

    # --- Per-sample errors for each model on validation ---
    mv_errs = C.track_error_per_sample(C.movement_vector_pred(Xrva), Yva[:, :, :2])
    disp_errs = C.track_error_per_sample(disp_val, Yva[:, :, :2])
    res_errs = C.track_error_per_sample(res_val, Yva[:, :, :2])

    # --- Per-horizon gating: choose best model per sample ---
    # For each sample and horizon, pick the model with lowest error
    # This is the theoretical upper bound for a perfect gate
    best_per_sample = np.minimum(np.minimum(mv_errs, disp_errs), res_errs)

    # --- Learn a simple gate on validation ---
    # For each horizon, train a classifier: which model wins?
    # Use simple threshold rules based on regime features
    gate_results = {}
    for hi, h in enumerate(C.HORIZONS):
        mv_e = mv_errs[:, hi]
        disp_e = disp_errs[:, hi]
        res_e = res_errs[:, hi]

        # Determine winner per sample
        winners = np.zeros(len(mv_e), dtype=int)  # 0=MV, 1=disp, 2=res
        winners[disp_e < mv_e] = 1
        winners[(res_e < mv_e) & (res_e < disp_e)] = 2

        # Simple rule: if speed < threshold, use MV; else use residual
        speed = Rtr["speed"].values if len(Rtr) == len(mv_e) else Rva["speed"].values
        wind = Rva["wind_t0"].values

        # Try threshold-based rules
        best_rule_acc = 0
        best_rule = None
        for speed_thresh in [2, 5, 10, 15, 20]:
            for wind_thresh in [40, 55, 70, 85]:
                rule_pred = np.where(speed < speed_thresh, 0, 2)  # slow->MV, fast->residual
                acc = np.mean(rule_pred == winners)
                if acc > best_rule_acc:
                    best_rule_acc = acc
                    best_rule = {"speed_thresh": speed_thresh, "wind_thresh": wind_thresh}

        # Compute gated error
        gate_pred = np.where(Rva["speed"].values < best_rule["speed_thresh"], 0, 2)
        gated_errs = np.where(gate_pred == 0, mv_e, res_e)
        gate_results[str(h)] = {
            "rule": best_rule,
            "accuracy": float(best_rule_acc),
            "gated_mte": float(np.mean(gated_errs)),
            "mv_mte": float(np.mean(mv_e)),
            "disp_mte": float(np.mean(disp_e)),
            "res_mte": float(np.mean(res_e)),
            "oracle_mte": float(np.mean(best_per_sample[:, hi])),
            "winner_counts": {
                "mv": int((winners == 0).sum()),
                "disp": int((winners == 1).sum()),
                "res": int((winners == 2).sum()),
            },
        }

    # Apply gate to test
    test_gate_results = {}
    for hi, h in enumerate(C.HORIZONS):
        speed_thresh = gate_results[str(h)]["rule"]["speed_thresh"]
        mv_te_errs = C.track_error_per_sample(C.movement_vector_pred(Xrte), Yte[:, :, :2])
        res_te_errs = C.track_error_per_sample(res_te, Yte[:, :, :2])
        gate_pred = np.where(Rte["speed"].values < speed_thresh, 0, 2)
        gated_te_errs = np.where(gate_pred == 0, mv_te_errs[:, hi], res_te_errs[:, hi])
        test_gate_results[str(h)] = {
            "gated_mte": float(np.mean(gated_te_errs)),
            "mv_mte": float(np.mean(mv_te_errs[:, hi])),
            "res_mte": float(np.mean(res_te_errs[:, hi])),
        }

    # Print comparison
    print("\n" + "=" * 70)
    print("GATE ANALYSIS (validation)")
    print("=" * 70)
    for h in C.HORIZONS:
        g = gate_results[str(h)]
        print(f"  {h}h: rule=speed<{g['rule']['speed_thresh']}  "
              f"gate={g['gated_mte']:.2f}  mv={g['mv_mte']:.2f}  "
              f"res={g['res_mte']:.2f}  oracle={g['oracle_mte']:.2f}")

    print("\n" + "=" * 70)
    print("GATE ON TEST")
    print("=" * 70)
    for h in C.HORIZONS:
        g = test_gate_results[str(h)]
        print(f"  {h}h: gate={g['gated_mte']:.2f}  mv={g['mv_mte']:.2f}  res={g['res_mte']:.2f}")

    # Decide if gating helps
    gate_helps = all(test_gate_results[str(h)]["gated_mte"] <
                     min(test_gate_results[str(h)]["mv_mte"],
                         test_gate_results[str(h)]["res_mte"])
                     for h in C.HORIZONS)

    if gate_helps:
        verdict = "IMPROVED — gated forecast is better than individual models"
    else:
        verdict = "NO IMPROVEMENT — retain existing champion"

    print(f"\nVerdict: {verdict}")

    out = RESULTS / "P4_E15_HYBRID_GATE.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "validation": {
                "movement_vector": mv_val_m,
                "displacement": disp_val_m,
                "residual": res_val_m,
                "gate_analysis": gate_results,
            },
            "test": {
                "movement_vector": mv_te_m,
                "displacement": disp_te_m,
                "residual": res_te_m,
                "gated": test_gate_results,
            },
            "verdict": verdict,
            "gate_helps": gate_helps,
        }, f, indent=2)
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
