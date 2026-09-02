"""P4 model-improvement experiment runner (model_improvement area).

Displacement-based forecasting protocol:
  * Each sample predicts displacement (dlat_km, dlon_km) at +6/+12/+24h relative
    to t0 position. Movement-vector & persistence baselines are identical
    3-horizon displacement extrapolations.
  * Train/val/test are the authoritative CLEAN splits (cyclone-disjoint,
    chronological). Normalization computed on TRAIN only.
  * Champion selection on VALIDATION only; each chosen config is evaluated on
    the untouched TEST split exactly once.
  * No test-set tuning, no horizon-specific test selection.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_common as C

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)


def build_tabular_features(X_eng, X_raw):
    """Flatten engineered (N,5,F) into a tabular feature matrix.

    X_eng includes raw 7 + extra derived (14 total per step). We extract,
    causally: current state (t0), last-step velocity, prior-step velocity,
    acceleration, 24h trend, turn rate, and rolling stats.
    """
    N = X_eng.shape[0]
    col = {}

    def agg(name, vals, idx):
        col[f"{name}_{idx}"] = vals[:, idx]

    # t0 raw state (cols 0..6)
    for i, name in enumerate(C.RAW_NAMES):
        col[f"state0_{name}"] = X_raw[:, 4, i]
    # last-step velocity (t-6 -> t) : rows 3->4
    vlat = X_raw[:,4,0] - X_raw[:,3,0]
    vlon = C.wrap_lon_delta(X_raw[:,3,1], X_raw[:,4,1])
    col["v_lat_step1"] = vlat
    col["v_lon_step1"] = vlon
    # velocity previous step (t-12 -> t-6)
    col["v_lat_step2"] = X_raw[:,3,0] - X_raw[:,2,0]
    col["v_lon_step2"] = C.wrap_lon_delta(X_raw[:,2,1], X_raw[:,3,1])
    # velocity oldest (t-24 -> t-18)
    col["v_lat_step4"] = X_raw[:,1,0] - X_raw[:,0,0]
    col["v_lon_step4"] = C.wrap_lon_delta(X_raw[:,0,1], X_raw[:,1,1])
    # acceleration (step2 -> step1)
    col["a_lat"] = vlat - col["v_lat_step2"]
    col["a_lon"] = vlon - col["v_lon_step2"]
    # 24h trend velocity (t-24 -> t)
    col["v24_lat"] = (X_raw[:,4,0] - X_raw[:,0,0]) / 4.0
    col["v24_lon"] = C.wrap_lon_delta(X_raw[:,0,1], X_raw[:,4,1]) / 4.0
    # turn rate (heading change step1 - step2)
    h1 = C.bearing_degrees(X_raw[:,3,0], X_raw[:,3,1], X_raw[:,4,0], X_raw[:,4,1])
    h2 = C.bearing_degrees(X_raw[:,2,0], X_raw[:,2,1], X_raw[:,3,0], X_raw[:,3,1])
    col["turn_rate"] = (h1 - h2 + 180.0) % 360.0 - 180.0
    col["heading_step1"] = h1
    # rolling: mean lat/lon velocity over last 2 steps
    col["v_lat_mean2"] = 0.5*(vlat + col["v_lat_step2"])
    col["v_lon_mean2"] = 0.5*(vlon + col["v_lon_step2"])
    # wind/pressure tendency (t-24 -> t)
    col["wind_tend"] = X_raw[:,4,2] - X_raw[:,0,2]
    col["pres_tend"] = X_raw[:,4,3] - X_raw[:,0,3]
    # environmental wind at t0
    col["env_speed0"] = np.hypot(X_raw[:,4,5], X_raw[:,4,6])
    col["env_u0"] = X_raw[:,4,5]
    col["env_v0"] = X_raw[:,4,6]

    df = pd.DataFrame(col)
    return df.astype(np.float32)


def displacement_targets(X_raw, Y):
    """Displacement (km) from t0 lat/lon to each target."""
    # t0 position
    lat0 = X_raw[:,4,0]; lon0 = X_raw[:,4,1]
    N = X_raw.shape[0]
    # convert to local km using haversine x (lon) / y (lat) approx at t0
    dlat_km = np.empty((N,3))  # per horizon
    dlon_km = np.empty((N,3))
    for hi in range(3):
        dlat_km[:,hi] = C.haversine_km(
            lat0, lon0, Y[:,hi,0], lon0) * np.sign(Y[:,hi,0]-lat0)
        dlon_km[:,hi] = C.haversine_km(
            lat0, lon0, lat0, Y[:,hi,1]) * np.sign(C.wrap_lon_delta(lon0, Y[:,hi,1]))
    return dlat_km, dlon_km


def displacement_to_position(X_raw, dlat_km, dlon_km):
    """Inverse: t0 + displacement km -> absolute lat/lon."""
    lat0 = X_raw[:,4,0]; lon0 = X_raw[:,4,1]
    N = X_raw.shape[0]
    latlon = np.empty((N,3,2))
    lat_pm = 111.0  # km per deg lat (approx, fine for NIO)
    for hi in range(3):
        latlon[:,hi,0] = lat0 + dlat_km[:,hi]/lat_pm
        # lon deg per km depends on cos(lat); use average lat
        coslat = np.cos(np.radians((lat0 + latlon[:,hi,0])/2))
        lon_perm = 111.0*coslat
        lon_perm[lon_perm < 1e-6] = 1e-6
        lonlon = lon0 + dlon_km[:,hi]/lon_perm
        lonlon = lonlon % 360.0
        latlon[:,hi,1] = lonlon
    return latlon


def main():
    # Load engineered data (14 features/step) + raw for all splits
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    Xte, Yte, Xrte = C.load_split_data("test")
    print("train", Xtr.shape, Xrtr.shape, "val", Xva.shape, "test", Xte.shape)

    # displacement targets
    dlat_tr, dlon_tr = displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = displacement_targets(Xrva, Yva)
    dlat_te, dlon_te = displacement_targets(Xrte, Yte)

    # Tabular features
    Ftr = build_tabular_features(Xtr, Xrtr)
    Fva = build_tabular_features(Xva, Xrva)
    Fte = build_tabular_features(Xte, Xrte)
    print("feature cols:", Ftr.shape)

    # ---- Baselines on test (for reference) ----
    pp = C.persistence_pred(Xrte); pm = C.movement_vector_pred(Xrte)
    print("baseline persistence:", C.summarize(pp, Yte[:,:,:2]))
    print("baseline movement   :", C.summarize(pm, Yte[:,:,:2]))

    # ---- CHAMPION GRU (from feature_dataset, using existing 16-feature NPZ) ----
    # We evaluate the pre-trained EXP005 if available via its saved predictions is
    # NOT reproducible here without weights; instead we re-train comparable models.
    # Use LightGBM displacement regressors (E04) as primary fast experiments.

    try:
        import lightgbm as lgb
        have_lgb = True
    except Exception:
        have_lgb = False
    print("lightgbm available:", have_lgb)

    # Feature set variants
    variants = {
        "base16": Ftr,             # all engineered tabular features
    }

    # ---- LightGBM per-horizon displacement regression ----
    # Target stacking: we train ONE model per (horizon, coordinate) for clarity,
    # tuned ONLY on validation.
    best = {}
    test_metrics = {}
    for hname, Ftr_df in variants.items():
        per_h = {}
        for hi, h in enumerate(C.HORIZONS):
            for coord, target in [("lat", dlat_tr[:,hi]), ("lon", dlon_tr[:,hi])]:
                key = f"{h}h_{coord}"
                model = lgb.LGBMRegressor(
                    n_estimators=400, learning_rate=0.05, max_depth=5,
                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1e-2, random_state=SEED, verbose=-1)
                # train/val
                model.fit(Ftr_df.values, target,
                          eval_set=[(Fva.values, dlat_va[:,hi] if coord=="lat" else dlon_va[:,hi])],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
                per_h[key] = model
        # assemble val predictions
        dlat_va_p = np.column_stack([per_h[f"{h}h_lat"].predict(Fva.values) for h in C.HORIZONS])
        dlon_va_p = np.column_stack([per_h[f"{h}h_lon"].predict(Fva.values) for h in C.HORIZONS])
        val_pos = displacement_to_position(Xrva, dlat_va_p, dlon_va_p)
        vm = C.summarize(val_pos, Yva[:,:,:2])
        print(f"[{hname}] VALIDATION: 6h={vm['6']['track_error_km_mean']:.2f} "
              f"12h={vm['12']['track_error_km_mean']:.2f} 24h={vm['24']['track_error_km_mean']:.2f}")

        # test (once)
        dlat_te_p = np.column_stack([per_h[f"{h}h_lat"].predict(Fte.values) for h in C.HORIZONS])
        dlon_te_p = np.column_stack([per_h[f"{h}h_lon"].predict(Fte.values) for h in C.HORIZONS])
        te_pos = displacement_to_position(Xrte, dlat_te_p, dlon_te_p)
        tm = C.summarize(te_pos, Yte[:,:,:2])
        test_metrics[hname] = tm
        print(f"[{hname}] TEST: 6h={tm['6']['track_error_km_mean']:.2f} "
              f"12h={tm['12']['track_error_km_mean']:.2f} 24h={tm['24']['track_error_km_mean']:.2f}")

    out = RESULTS / "P4_LGBM_QUICK.json"
    with open(out,"w",encoding="utf-8") as f:
        json.dump({"test_metrics": test_metrics}, f, indent=2, default=str)
    print("WROTE", out)


if __name__ == "__main__":
    main()
