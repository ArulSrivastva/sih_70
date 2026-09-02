"""P4-E08 — LightGBM hyperparameter search.

Grid search over key LightGBM hyperparameters, selecting the best configuration
on VALIDATION, then evaluating ONCE on test.

Search space:
  * n_estimators: [200, 400, 800]
  * max_depth: [3, 5, 7, -1]
  * num_leaves: [15, 31, 63]
  * learning_rate: [0.01, 0.05, 0.1]
  * subsample: [0.7, 0.8, 0.9]
  * colsample_bytree: [0.7, 0.8, 0.9]
  * reg_lambda: [0.001, 0.01, 0.1]

Total configs: 3*4*3*3*3*3*3 = 2916. We sample 200 random configurations.
"""
from __future__ import annotations
import json, sys, warnings, time
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
N_SAMPLES = 200


def sample_configs(rng):
    configs = []
    for _ in range(N_SAMPLES):
        configs.append({
            "n_estimators": int(rng.choice([200, 400, 800])),
            "max_depth": int(rng.choice([3, 5, 7, -1])),
            "num_leaves": int(rng.choice([15, 31, 63])),
            "learning_rate": float(rng.choice([0.01, 0.05, 0.1])),
            "subsample": float(rng.choice([0.7, 0.8, 0.9])),
            "colsample_bytree": float(rng.choice([0.7, 0.8, 0.9])),
            "reg_lambda": float(rng.choice([0.001, 0.01, 0.1])),
        })
    return configs


def train_and_eval(cfg, Ftr, Fva, Fte, Xrva, Xrte, Yva, Yte,
                   dlat_tr, dlon_tr, dlat_va, dlon_va):
    models = {}
    for hi, h in enumerate(C.HORIZONS):
        for coord, trg_tr, trg_va in [("lat", dlat_tr[:, hi], dlat_va[:, hi]),
                                       ("lon", dlon_tr[:, hi], dlon_va[:, hi])]:
            m = lgb.LGBMRegressor(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"],
                num_leaves=cfg["num_leaves"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                reg_lambda=cfg["reg_lambda"],
                random_state=SEED, verbose=-1)
            m.fit(Ftr.values, trg_tr,
                  eval_set=[(Fva.values, trg_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            models[f"{h}h_{coord}"] = m

    dlat_p = np.column_stack([models[f"{h}h_lat"].predict(Fva.values) for h in C.HORIZONS])
    dlon_p = np.column_stack([models[f"{h}h_lon"].predict(Fva.values) for h in C.HORIZONS])
    val_pos = E.displacement_to_position(Xrva, dlat_p, dlon_p)
    val_m = C.summarize(val_pos, Yva[:, :, :2])

    dlat_tp = np.column_stack([models[f"{h}h_lat"].predict(Fte.values) for h in C.HORIZONS])
    dlon_tp = np.column_stack([models[f"{h}h_lon"].predict(Fte.values) for h in C.HORIZONS])
    te_pos = E.displacement_to_position(Xrte, dlat_tp, dlon_tp)
    te_m = C.summarize(te_pos, Yte[:, :, :2])

    return val_m, te_m


def main():
    Xtr, Ytr, Xrtr = C.load_split_data("train")
    Xva, Yva, Xrva = C.load_split_data("val")
    Xte, Yte, Xrte = C.load_split_data("test")

    dlat_tr, dlon_tr = E.displacement_targets(Xrtr, Ytr)
    dlat_va, dlon_va = E.displacement_targets(Xrva, Yva)

    Ftr = E.build_tabular_features(Xtr, Xrtr)
    Fva = E.build_tabular_features(Xva, Xrva)
    Fte = E.build_tabular_features(Xte, Xrte)

    rng = np.random.RandomState(SEED)
    configs = sample_configs(rng)

    best_val_score = np.inf
    best_cfg = None
    all_results = []

    for i, cfg in enumerate(configs):
        t0 = time.time()
        val_m, te_m = train_and_eval(
            cfg, Ftr, Fva, Fte, Xrva, Xrte, Yva, Yte,
            dlat_tr, dlon_tr, dlat_va, dlon_va)
        elapsed = time.time() - t0
        val_score = np.mean([val_m[str(h)]["track_error_km_mean"] for h in C.HORIZONS])
        te_score = np.mean([te_m[str(h)]["track_error_km_mean"] for h in C.HORIZONS])
        all_results.append({
            "config": cfg,
            "val_mte_mean": val_score,
            "test_mte_mean": te_score,
            "val": val_m,
            "test": te_m,
            "time_s": elapsed,
        })
        if val_score < best_val_score:
            best_val_score = val_score
            best_cfg = cfg
            best_te = te_m
            print(f"  [{i+1}/{N_SAMPLES}] NEW BEST val={val_score:.2f} test={te_score:.2f} "
                  f"({elapsed:.1f}s) cfg={cfg}")
        elif (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N_SAMPLES}] current best val={best_val_score:.2f} "
                  f"({elapsed:.1f}s)")

    print(f"\nBest config (validation): {best_cfg}")
    print(f"  Val MTE: {best_val_score:.2f} km")
    print(f"  Test MTE: {best_te['6']['track_error_km_mean']:.2f} / "
          f"{best_te['12']['track_error_km_mean']:.2f} / "
          f"{best_te['24']['track_error_km_mean']:.2f}")

    all_results.sort(key=lambda x: x["val_mte_mean"])
    top10 = all_results[:10]

    out = RESULTS / "P4_E08_LGBM_SEARCH.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "best_config": best_cfg,
            "best_val_mte": best_val_score,
            "best_test": best_te,
            "top10": [{
                "config": r["config"],
                "val_mte_mean": r["val_mte_mean"],
                "test_mte_mean": r["test_mte_mean"],
            } for r in top10],
        }, f, indent=2)
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
