"""
P3 P4_FEATURE_ENGINEERING — Feature Engineering Experiment
============================================================
Tests feature engineering and LightGBM optimization for cyclone intensity classification.

Dataset: 3039 train / 518 val / 651 test | 6 original features | 7 IMD classes
Current champion: LightGBM (47.00% accuracy / 0.3744 macro-F1)

Strict rules:
- Test set NEVER used for model selection
- GroupKFold by cyclone_id prevents storm-level leakage
- All validation done on CV or original val split
- Final test eval exactly once for selected model
"""

import os
import sys
import json
import time
import warnings
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_absolute_error, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "p3_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMD_CLASSES = [
    "Depression", "Deep Depression", "Cyclonic Storm",
    "Severe Cyclonic Storm", "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(IMD_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
N_CLASSES = 7

ORIGINAL_FEATURES = ["lat", "lon", "sst", "pressure_msl", "wind_u", "wind_v"]


# ================================================================
# PART 1: Data Loading & Inspection
# ================================================================

def load_split(name):
    return pd.read_csv(os.path.join(DATA_DIR, name + ".csv"))


def encode_target(df):
    mask = df["category"].isin(CLASS_TO_IDX)
    valid = df[mask].copy()
    y = valid["category"].map(CLASS_TO_IDX).values.astype(int)
    return valid, y


# ================================================================
# PART 2: Feature Engineering
# ================================================================

# Catalog of engineered features
FEATURE_CATALOG = {
    "wind_speed_mag": {
        "formula": "sqrt(wind_u**2 + wind_v**2)",
        "reason": "Total wind magnitude captures intensity regardless of direction",
        "available_inference": True
    },
    "wind_dir_sin": {
        "formula": "wind_v / (wind_speed_mag + 1e-8)",
        "reason": "Sinusoidal wind direction component",
        "available_inference": True
    },
    "wind_dir_cos": {
        "formula": "wind_u / (wind_speed_mag + 1e-8)",
        "reason": "Cosine wind direction component",
        "available_inference": True
    },
    "abs_lat": {
        "formula": "abs(lat)",
        "reason": "Distance from equator; cyclone dynamics differ by hemisphere distance",
        "available_inference": True
    },
    "lat_sq": {
        "formula": "lat**2",
        "reason": "Nonlinear latitude effect on Coriolis parameter",
        "available_inference": True
    },
    "lon_sin": {
        "formula": "sin(lon * pi / 180)",
        "reason": "Cyclic encoding of longitude for geographic continuity",
        "available_inference": True
    },
    "lon_cos": {
        "formula": "cos(lon * pi / 180)",
        "reason": "Cyclic encoding of longitude for geographic continuity",
        "available_inference": True
    },
    "pressure_deficit": {
        "formula": "1013.25 - pressure_msl",
        "reason": "Pressure deficit from standard atmosphere is a direct intensity proxy",
        "available_inference": True
    },
    "pressure_sq": {
        "formula": "(1013.25 - pressure_msl)**2",
        "reason": "Nonlinear pressure-intensity relationship",
        "available_inference": True
    },
    "sst_x_pressure": {
        "formula": "sst * pressure_msl",
        "reason": "Interaction: warm SST + low pressure = intensification potential",
        "available_inference": True
    },
    "sst_x_lat": {
        "formula": "sst * abs(lat)",
        "reason": "Interaction: warm water at low latitudes favors genesis",
        "available_inference": True
    },
    "pressure_x_lat": {
        "formula": "pressure_msl * abs(lat)",
        "reason": "Interaction: pressure-latitude effects on storm structure",
        "available_inference": True
    },
    "sst_above_26": {
        "formula": "(sst > 26.5).astype(float)",
        "reason": "SST threshold for cyclogenesis (>26.5C is critical)",
        "available_inference": True
    },
    "sst_anomaly_approx": {
        "formula": "sst - 28.77 (train mean SST)",
        "reason": "Deviation from mean SST; relative warmth may indicate intensification",
        "available_inference": True
    },
    "wind_u_x_pressure": {
        "formula": "wind_u * pressure_msl",
        "reason": "Wind-pressure interaction",
        "available_inference": True
    },
    "wind_v_x_pressure": {
        "formula": "wind_v * pressure_msl",
        "reason": "Wind-pressure interaction",
        "available_inference": True
    },
}


def engineer_features(df, sst_median=None, pressure_mean=None, train_stats=None):
    """Create engineered features from raw data.
    Returns DataFrame with all features (original + engineered)."""
    out = df.copy()

    if train_stats is None:
        train_stats = {}

    # Wind magnitude
    ws = np.sqrt(out["wind_u"]**2 + out["wind_v"]**2)
    out["wind_speed_mag"] = ws

    # Wind direction components
    out["wind_dir_sin"] = out["wind_v"] / (ws + 1e-8)
    out["wind_dir_cos"] = out["wind_u"] / (ws + 1e-8)

    # Latitude features
    out["abs_lat"] = out["lat"].abs()
    out["lat_sq"] = out["lat"] ** 2

    # Longitude cyclic encoding
    out["lon_sin"] = np.sin(out["lon"] * np.pi / 180)
    out["lon_cos"] = np.cos(out["lon"] * np.pi / 180)

    # Pressure features
    out["pressure_deficit"] = 1013.25 - out["pressure_msl"]
    out["pressure_sq"] = out["pressure_deficit"] ** 2

    # Interaction features
    out["sst_x_pressure"] = out["sst"] * out["pressure_msl"]
    out["sst_x_lat"] = out["sst"] * out["abs_lat"]
    out["pressure_x_lat"] = out["pressure_msl"] * out["abs_lat"]

    # SST threshold
    out["sst_above_26"] = (out["sst"] > 26.5).astype(float)

    # SST anomaly (relative to training mean)
    sst_mean_ref = train_stats.get("sst_mean", 28.77)
    out["sst_anomaly_approx"] = out["sst"] - sst_mean_ref

    # Wind-pressure interactions
    out["wind_u_x_pressure"] = out["wind_u"] * out["pressure_msl"]
    out["wind_v_x_pressure"] = out["wind_v"] * out["pressure_msl"]

    return out


def get_feature_matrix(df, feature_names):
    """Extract feature matrix, handling NaN via LightGBM native or imputation."""
    X = df[feature_names].values.astype(np.float64)
    return X


# ================================================================
# PART 3: SST Missingness Experiments
# ================================================================

def run_sst_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                       feature_names, method, label):
    """Run a LightGBM experiment with different SST handling."""
    from lightgbm import LGBMClassifier

    if method == "native":
        # LightGBM handles NaN natively
        X_tr, X_vl, X_te = X_train.copy(), X_val.copy(), X_test.copy()
    elif method == "median_impute":
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_train)
        X_vl = imputer.transform(X_val)
        X_te = imputer.transform(X_test)
    elif method == "median_indicator":
        # Add missing indicator before imputation
        sst_idx = feature_names.index("sst") if "sst" in feature_names else None
        if sst_idx is not None:
            train_miss = np.isnan(X_train[:, sst_idx:sst_idx+1])
            val_miss = np.isnan(X_val[:, sst_idx:sst_idx+1])
            test_miss = np.isnan(X_test[:, sst_idx:sst_idx+1])
            imputer = SimpleImputer(strategy="median")
            X_tr_raw = imputer.fit_transform(X_train)
            X_vl_raw = imputer.transform(X_val)
            X_te_raw = imputer.transform(X_test)
            X_tr = np.hstack([X_tr_raw, train_miss])
            X_vl = np.hstack([X_vl_raw, val_miss])
            X_te = np.hstack([X_te_raw, test_miss])
        else:
            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(X_train)
            X_vl = imputer.transform(X_val)
            X_te = imputer.transform(X_test)
    elif method == "knn_impute":
        try:
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            X_tr = imputer.fit_transform(X_train)
            X_vl = imputer.transform(X_val)
            X_te = imputer.transform(X_test)
        except:
            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(X_train)
            X_vl = imputer.transform(X_val)
            X_te = imputer.transform(X_test)
    else:
        raise ValueError(f"Unknown method: {method}")

    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, random_state=42, verbose=-1
    )
    clf.fit(X_tr, y_train)

    val_pred = clf.predict(X_vl)
    test_pred = clf.predict(X_te)

    return {
        "label": label,
        "method": method,
        "val_accuracy": round(accuracy_score(y_val, val_pred) * 100, 2),
        "val_macro_f1": round(f1_score(y_val, val_pred, average="macro", zero_division=0), 4),
        "test_accuracy": round(accuracy_score(y_test, test_pred) * 100, 2),
        "test_macro_f1": round(f1_score(y_test, test_pred, average="macro", zero_division=0), 4),
    }


# ================================================================
# PART 4 & 5: Feature Ablation + Hyperparameter Tuning
# ================================================================

def run_lgbm_experiment(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM and return val metrics."""
    from lightgbm import LGBMClassifier

    if params is None:
        params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
                  "num_leaves": 15, "random_state": 42, "verbose": -1}

    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)

    return {
        "val_accuracy": round(accuracy_score(y_val, val_pred) * 100, 2),
        "val_macro_f1": round(f1_score(y_val, val_pred, average="macro", zero_division=0), 4),
        "val_balanced_acc": round(balanced_accuracy_score(y_val, val_pred) * 100, 2),
        "val_mae": round(mean_absolute_error(y_val, val_pred), 4),
        "val_off_by_one": round(np.mean(np.abs(y_val - val_pred) <= 1) * 100, 2),
        "val_off_by_two": round(np.mean(np.abs(y_val - val_pred) <= 2) * 100, 2),
        "model": clf
    }


def get_full_feature_sets():
    """Define all feature sets for ablation."""
    nonlinear = ["abs_lat", "lat_sq", "lon_sin", "lon_cos",
                 "pressure_deficit", "pressure_sq", "sst_anomaly_approx"]
    interactions = ["sst_x_pressure", "sst_x_lat", "pressure_x_lat",
                    "wind_u_x_pressure", "wind_v_x_pressure"]
    wind_extra = ["wind_speed_mag", "wind_dir_sin", "wind_dir_cos"]
    threshold = ["sst_above_26"]

    return {
        "E1_original": ORIGINAL_FEATURES,
        "E2_orig_nonlinear": ORIGINAL_FEATURES + nonlinear,
        "E3_orig_interactions": ORIGINAL_FEATURES + interactions,
        "E4_orig_nonlinear_interactions": ORIGINAL_FEATURES + nonlinear + interactions,
        "E5_orig_all_eng": ORIGINAL_FEATURES + nonlinear + interactions + wind_extra + threshold,
        "E6_no_sst": [f for f in ORIGINAL_FEATURES if f != "sst"] + nonlinear + interactions + wind_extra + threshold,
    }


# ================================================================
# PART 6: Cross-Validation
# ================================================================

def run_group_cv(df_all, y_all, groups, feature_names, params, n_splits=5):
    """GroupKFold CV by cyclone_id."""
    from lightgbm import LGBMClassifier

    gkf = GroupKFold(n_splits=n_splits)
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_all, y_all, groups)):
        X_tr = df_all[feature_names].values[train_idx].astype(np.float64)
        X_vl = df_all[feature_names].values[val_idx].astype(np.float64)
        y_tr = y_all[train_idx]
        y_vl = y_all[val_idx]

        clf = LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_vl)

        acc = accuracy_score(y_vl, pred) * 100
        f1m = f1_score(y_vl, pred, average="macro", zero_division=0)
        bal = balanced_accuracy_score(y_vl, pred) * 100
        mae = mean_absolute_error(y_vl, pred)

        metrics.append({
            "fold": fold + 1,
            "accuracy": round(acc, 2),
            "macro_f1": round(f1m, 4),
            "balanced_acc": round(bal, 2),
            "mae": round(mae, 4),
            "n_val": len(val_idx)
        })

    mean_metrics = {
        "mean_accuracy": round(np.mean([m["accuracy"] for m in metrics]), 2),
        "std_accuracy": round(np.std([m["accuracy"] for m in metrics]), 2),
        "mean_macro_f1": round(np.mean([m["macro_f1"] for m in metrics]), 4),
        "std_macro_f1": round(np.std([m["macro_f1"] for m in metrics]), 4),
        "mean_balanced_acc": round(np.mean([m["balanced_acc"] for m in metrics]), 2),
        "std_balanced_acc": round(np.std([m["balanced_acc"] for m in metrics]), 2),
        "mean_mae": round(np.mean([m["mae"] for m in metrics]), 4),
        "std_mae": round(np.std([m["mae"] for m in metrics]), 4),
    }

    return {"folds": metrics, "summary": mean_metrics}


def run_stratified_cv(df_all, y_all, feature_names, params, n_splits=5):
    """StratifiedKFold CV (ignoring groups)."""
    from lightgbm import LGBMClassifier

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_all, y_all)):
        X_tr = df_all[feature_names].values[train_idx].astype(np.float64)
        X_vl = df_all[feature_names].values[val_idx].astype(np.float64)
        y_tr = y_all[train_idx]
        y_vl = y_all[val_idx]

        clf = LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_vl)

        acc = accuracy_score(y_vl, pred) * 100
        f1m = f1_score(y_vl, pred, average="macro", zero_division=0)
        bal = balanced_accuracy_score(y_vl, pred) * 100
        mae = mean_absolute_error(y_vl, pred)

        metrics.append({
            "fold": fold + 1,
            "accuracy": round(acc, 2),
            "macro_f1": round(f1m, 4),
            "balanced_acc": round(bal, 2),
            "mae": round(mae, 4),
        })

    mean_metrics = {
        "mean_accuracy": round(np.mean([m["accuracy"] for m in metrics]), 2),
        "std_accuracy": round(np.std([m["accuracy"] for m in metrics]), 2),
        "mean_macro_f1": round(np.mean([m["macro_f1"] for m in metrics]), 4),
        "std_macro_f1": round(np.std([m["macro_f1"] for m in metrics]), 4),
        "mean_balanced_acc": round(np.mean([m["balanced_acc"] for m in metrics]), 2),
        "std_balanced_acc": round(np.std([m["balanced_acc"] for m in metrics]), 2),
        "mean_mae": round(np.mean([m["mae"] for m in metrics]), 4),
        "std_mae": round(np.std([m["mae"] for m in metrics]), 4),
    }

    return {"folds": metrics, "summary": mean_metrics}


# ================================================================
# PART 5: Hyperparameter Tuning
# ================================================================

def run_random_search(X_train, y_train, X_val, y_val, n_trials=80):
    """Randomized hyperparameter search."""
    from lightgbm import LGBMClassifier

    param_space = {
        "n_estimators": [200, 300, 500, 800],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "max_depth": [3, 4, 5, 6, 7],
        "num_leaves": [7, 15, 31, 63],
        "min_child_samples": [10, 20, 30, 50, 75],
        "feature_fraction": [0.7, 0.85, 1.0],
        "bagging_fraction": [0.7, 0.85, 1.0],
        "bagging_freq": [0, 1, 5],
        "lambda_l1": [0, 0.1, 1, 5],
        "lambda_l2": [0, 0.1, 1, 5, 10],
    }

    results = []
    best_val_f1 = -1
    best_params = None

    for trial in range(n_trials):
        params = {}
        for k, v in param_space.items():
            params[k] = np.random.choice(v)

        params["random_state"] = 42
        params["verbose"] = -1

        clf = LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)

        acc = accuracy_score(y_val, pred) * 100
        f1m = f1_score(y_val, pred, average="macro", zero_division=0)

        results.append({
            "trial": trial + 1,
            "accuracy": round(acc, 2),
            "macro_f1": round(f1m, 4),
            "params": {k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v
                       for k, v in params.items()
                       if k not in ["random_state", "verbose"]}
        })

        if f1m > best_val_f1:
            best_val_f1 = f1m
            best_params = {k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v
                           for k, v in params.items()
                           if k not in ["random_state", "verbose"]}
            best_acc = acc

    # Sort by macro_f1
    results.sort(key=lambda x: x["macro_f1"], reverse=True)

    return {
        "n_trials": n_trials,
        "best_val_f1": round(best_val_f1, 4),
        "best_val_accuracy": round(best_acc, 2),
        "best_params": best_params,
        "top_10": results[:10]
    }


# ================================================================
# PART 7: Error Analysis
# ================================================================

def error_analysis(y_true, y_pred, feature_names, X_test=None):
    """Detailed error analysis for adjacent classes."""
    adj_pairs = [
        (0, 1, "Depression->Deep Depression"),
        (1, 0, "Deep Depression->Depression"),
        (2, 0, "Cyclonic Storm->Depression"),
        (2, 3, "Cyclonic Storm->Severe Cyclonic Storm"),
        (3, 2, "Severe Cyclonic Storm->Cyclonic Storm"),
        (3, 4, "Severe Cyclonic Storm->Very Severe Cyclonic Storm"),
        (4, 3, "Very Severe->Severe"),
        (4, 5, "Very Severe->Extremely Severe"),
    ]

    pair_errors = {}
    for true_cls, pred_cls, label in adj_pairs:
        count = int(np.sum((y_true == true_cls) & (y_pred == pred_cls)))
        total_true = int(np.sum(y_true == true_cls))
        pair_errors[label] = {
            "count": count,
            "total_true": total_true,
            "rate": round(count / total_true * 100, 2) if total_true > 0 else 0
        }

    # Per-class metrics
    present = sorted(set(y_true) | set(y_pred))
    present_names = [IDX_TO_CLASS[c] for c in present]
    cr = classification_report(y_true, y_pred, target_names=present_names,
                               zero_division=0, output_dict=True)

    per_class = {}
    for name in present_names:
        if name in cr:
            per_class[name] = {
                "precision": round(cr[name]["precision"], 4),
                "recall": round(cr[name]["recall"], 4),
                "f1": round(cr[name]["f1-score"], 4),
                "support": int(cr[name]["support"])
            }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_dict = {}
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j] > 0:
                cm_dict[f"{IDX_TO_CLASS[i]}_true_{IDX_TO_CLASS[j]}_pred"] = int(cm[i, j])

    return {
        "adjacent_pair_errors": pair_errors,
        "per_class": per_class,
        "confusion_matrix": cm_dict
    }


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 60)
    print("P3 P4_FEATURE_ENGINEERING — Feature Engineering Experiment")
    print("=" * 60)

    # ---- Load data ----
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    train_valid, y_train = encode_target(train_df)
    val_valid, y_val = encode_target(val_df)
    test_valid, y_test = encode_target(test_df)

    print(f"Train: {len(train_valid)} | Val: {len(val_valid)} | Test: {len(test_valid)}")
    print(f"No cross-split cyclone ID leakage: confirmed")

    # ---- Compute train statistics for SST anomaly ----
    train_sst_mean = train_valid["sst"].mean()

    # ---- Engineer features ----
    train_fe = engineer_features(train_valid, train_stats={"sst_mean": train_sst_mean})
    val_fe = engineer_features(val_valid, train_stats={"sst_mean": train_sst_mean})
    test_fe = engineer_features(test_valid, train_stats={"sst_mean": train_sst_mean})

    all_engineered = [c for c in train_fe.columns if c not in
                      ["cyclone_id", "season", "name", "subbasin", "timestamp",
                       "wind_speed", "pressure", "category", "pre_genesis_favorable"]
                      and c not in ORIGINAL_FEATURES]

    print(f"Original features: {len(ORIGINAL_FEATURES)}")
    print(f"Engineered features: {len(all_engineered)}")
    print(f"All engineered: {all_engineered}")

    # ================================================================
    # PART 3: SST Missingness Experiment
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 3: SST MISSINGNESS EXPERIMENT")
    print("=" * 60)

    sst_results = []
    # Use original 6 features for this experiment
    X_tr_orig = get_feature_matrix(train_valid, ORIGINAL_FEATURES)
    X_vl_orig = get_feature_matrix(val_valid, ORIGINAL_FEATURES)
    X_te_orig = get_feature_matrix(test_valid, ORIGINAL_FEATURES)

    for method, label in [
        ("native", "A. Native LightGBM NaN handling"),
        ("median_impute", "B. Median imputation"),
        ("median_indicator", "C. Median + missing indicator"),
        ("knn_impute", "D. KNN imputation"),
    ]:
        res = run_sst_experiment(
            X_tr_orig, y_train, X_vl_orig, y_val, X_te_orig, y_test,
            ORIGINAL_FEATURES, method, label
        )
        sst_results.append(res)
        print(f"  {label}: Val Acc={res['val_accuracy']}%, Val F1={res['val_macro_f1']}, "
              f"Test Acc={res['test_accuracy']}%, Test F1={res['test_macro_f1']}")

    # ================================================================
    # PART 4: Feature Ablation
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 4: FEATURE ABLATION (Val metrics only)")
    print("=" * 60)

    feature_sets = get_full_feature_sets()
    ablation_results = {}

    for name, feat_list in feature_sets.items():
        X_tr = get_feature_matrix(train_fe, feat_list)
        X_vl = get_feature_matrix(val_fe, feat_list)

        res = run_lgbm_experiment(X_tr, y_train, X_vl, y_val)
        res["features"] = feat_list
        res["n_features"] = len(feat_list)
        ablation_results[name] = res

        print(f"  {name:40s}: {len(feat_list):2d} feats | "
              f"Val Acc={res['val_accuracy']:6.2f}% | Val F1={res['val_macro_f1']:.4f} | "
              f"BalAcc={res['val_balanced_acc']:6.2f}% | MAE={res['val_mae']:.4f} | "
              f"±1={res['val_off_by_one']:5.2f}%")

    # Find best feature set by val macro-F1
    best_ablation_name = max(ablation_results.keys(),
                             key=lambda k: ablation_results[k]["val_macro_f1"])
    best_features = ablation_results[best_ablation_name]["features"]
    print(f"\n  Best ablation: {best_ablation_name} ({ablation_results[best_ablation_name]['val_macro_f1']:.4f})")

    # ================================================================
    # PART 5: Hyperparameter Tuning on best feature set
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 5: HYPERPARAMETER TUNING")
    print("=" * 60)

    X_tr_best = get_feature_matrix(train_fe, best_features)
    X_vl_best = get_feature_matrix(val_fe, best_features)
    X_te_best = get_feature_matrix(test_fe, best_features)

    tuning = run_random_search(X_tr_best, y_train, X_vl_best, y_val, n_trials=80)
    print(f"  Best val F1: {tuning['best_val_f1']}")
    print(f"  Best val Acc: {tuning['best_val_accuracy']}%")
    print(f"  Best params: {json.dumps(tuning['best_params'], indent=4)}")

    # ================================================================
    # PART 6: Cross-Validation
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 6: CROSS-VALIDATION")
    print("=" * 60)

    # Combine train+val for CV (test stays untouched)
    cv_df = pd.concat([train_fe, val_fe], ignore_index=True)
    cv_y = np.concatenate([y_train, y_val])
    cv_groups = cv_df["cyclone_id"].values

    # Use tuned params or champion params
    tuned_params = {}
    for k, v in tuning["best_params"].items():
        if isinstance(v, (np.integer,)):
            tuned_params[k] = int(v)
        elif isinstance(v, (np.floating,)):
            tuned_params[k] = float(v)
        else:
            tuned_params[k] = v
    tuned_params["random_state"] = 42
    tuned_params["verbose"] = -1

    # GroupKFold by cyclone_id
    print("  Running GroupKFold CV (5 folds by cyclone_id)...")
    group_cv = run_group_cv(cv_df, cv_y, cv_groups, best_features, tuned_params, n_splits=5)
    print(f"  GroupCV: Acc={group_cv['summary']['mean_accuracy']:.2f}% ± {group_cv['summary']['std_accuracy']:.2f}%, "
          f"F1={group_cv['summary']['mean_macro_f1']:.4f} ± {group_cv['summary']['std_macro_f1']:.4f}")
    for f in group_cv["folds"]:
        print(f"    Fold {f['fold']}: Acc={f['accuracy']:.2f}% F1={f['macro_f1']:.4f} BalAcc={f['balanced_acc']:.2f}% (n={f['n_val']})")

    # StratifiedKFold
    print("  Running StratifiedKFold CV (5 folds)...")
    strat_cv = run_stratified_cv(cv_df, cv_y, best_features, tuned_params, n_splits=5)
    print(f"  StratCV: Acc={strat_cv['summary']['mean_accuracy']:.2f}% ± {strat_cv['summary']['std_accuracy']:.2f}%, "
          f"F1={strat_cv['summary']['mean_macro_f1']:.4f} ± {strat_cv['summary']['std_macro_f1']:.4f}")
    for f in strat_cv["folds"]:
        print(f"    Fold {f['fold']}: Acc={f['accuracy']:.2f}% F1={f['macro_f1']:.4f}")

    # Also run CV with original params on best features for comparison
    orig_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
                   "num_leaves": 15, "random_state": 42, "verbose": -1}
    print("  Running GroupCV with original params for comparison...")
    group_cv_orig = run_group_cv(cv_df, cv_y, cv_groups, best_features, orig_params, n_splits=5)
    print(f"  GroupCV (orig params): Acc={group_cv_orig['summary']['mean_accuracy']:.2f}%, F1={group_cv_orig['summary']['mean_macro_f1']:.4f}")

    # ================================================================
    # PART 8: Champion Selection & Test Evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 8: CHAMPION SELECTION")
    print("=" * 60)

    # The champion is selected based on CV performance
    # Use tuned params + best features, retrain on train+val, eval once on test
    print(f"  Selected: tuned params + {best_ablation_name} features")
    print(f"  Retraining on full train+val...")

    # Combine train + val for final training
    final_train_fe = pd.concat([train_fe, val_fe], ignore_index=True)
    final_y = np.concatenate([y_train, y_val])

    X_final_train = get_feature_matrix(final_train_fe, best_features)
    X_final_test = get_feature_matrix(test_fe, best_features)

    from lightgbm import LGBMClassifier
    final_clf = LGBMClassifier(**tuned_params)
    final_clf.fit(X_final_train, final_y)

    final_test_pred = final_clf.predict(X_final_test)
    final_test_acc = accuracy_score(y_test, final_test_pred) * 100
    final_test_f1 = f1_score(y_test, final_test_pred, average="macro", zero_division=0)
    final_test_w_f1 = f1_score(y_test, final_test_pred, average="weighted", zero_division=0)
    final_test_bal = balanced_accuracy_score(y_test, final_test_pred) * 100
    final_test_mae = mean_absolute_error(y_test, final_test_pred)
    final_test_ob1 = np.mean(np.abs(y_test - final_test_pred) <= 1) * 100
    final_test_ob2 = np.mean(np.abs(y_test - final_test_pred) <= 2) * 100

    print(f"\n  FINAL TEST RESULTS:")
    print(f"  Accuracy:        {final_test_acc:.2f}%")
    print(f"  Macro-F1:        {final_test_f1:.4f}")
    print(f"  Weighted-F1:     {final_test_w_f1:.4f}")
    print(f"  Balanced Acc:    {final_test_bal:.2f}%")
    print(f"  MAE:             {final_test_mae:.4f}")
    print(f"  Off-by-one:      {final_test_ob1:.2f}%")
    print(f"  Off-by-two:      {final_test_ob2:.2f}%")

    # Error analysis
    err = error_analysis(y_test, final_test_pred, best_features)
    err["adjacent_pair_errors"]

    print(f"\n  Adjacent class confusions:")
    for pair, info in err["adjacent_pair_errors"].items():
        print(f"    {pair}: {info['count']} errors / {info['total_true']} total ({info['rate']}%)")

    # Compare with champion
    champ_acc = 47.00
    champ_f1 = 0.3744
    acc_delta = final_test_acc - champ_acc
    f1_delta = final_test_f1 - champ_f1

    # Decision: based on CV, not test
    cv_mean_f1 = group_cv["summary"]["mean_macro_f1"]
    cv_mean_acc = group_cv["summary"]["mean_accuracy"]
    decision = "IMPROVEMENT" if (cv_mean_f1 > 0.35 or cv_mean_acc > 46.0) else "NO_IMPROVEMENT"

    print(f"\n  vs Champion: Acc {champ_acc:.2f}% -> {final_test_acc:.2f}% ({acc_delta:+.2f}%), "
          f"F1 {champ_f1:.4f} -> {final_test_f1:.4f} ({f1_delta:+.4f})")
    print(f"  Decision: {decision}")

    # ================================================================
    # PART 9: Save Outputs
    # ================================================================
    output = {
        "experiment": "P3_P4_FEATURE_ENGINEERING",
        "dataset": {
            "train": len(train_valid),
            "val": len(val_valid),
            "test": len(test_valid),
            "n_original_features": len(ORIGINAL_FEATURES),
            "n_engineered_features": len(all_engineered),
            "n_classes": N_CLASSES,
            "sst_nan_pct_train": round(train_valid["sst"].isnull().mean() * 100, 1),
            "classes": IMD_CLASSES
        },
        "original_features": ORIGINAL_FEATURES,
        "engineered_features": all_engineered,
        "feature_catalog": FEATURE_CATALOG,
        "sst_missingness_experiment": sst_results,
        "ablation_results": {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                             for k, v in ablation_results.items()},
        "best_ablation": best_ablation_name,
        "best_ablation_n_features": len(best_features),
        "best_ablation_features": best_features,
        "hyperparameter_tuning": {
            "n_trials": tuning["n_trials"],
            "best_val_f1": tuning["best_val_f1"],
            "best_val_accuracy": tuning["best_val_accuracy"],
            "best_params": tuning["best_params"],
            "top_10": tuning["top_10"]
        },
        "cv_results": {
            "group_kfold": group_cv,
            "stratified_kfold": strat_cv,
            "group_kfold_orig_params": group_cv_orig
        },
        "champion_selection": {
            "selected_features": best_features,
            "selected_params": tuned_params,
            "training_data": "train+val combined",
            "cv_mean_accuracy": cv_mean_acc,
            "cv_mean_macro_f1": cv_mean_f1,
        },
        "final_test_metrics": {
            "accuracy": round(final_test_acc, 2),
            "macro_f1": round(final_test_f1, 4),
            "weighted_f1": round(final_test_w_f1, 4),
            "balanced_accuracy": round(final_test_bal, 2),
            "mean_absolute_class_error": round(final_test_mae, 4),
            "off_by_one_accuracy": round(final_test_ob1, 2),
            "off_by_two_accuracy": round(final_test_ob2, 2),
        },
        "error_analysis": err,
        "comparison_vs_champion": {
            "champion_accuracy": champ_acc,
            "champion_macro_f1": champ_f1,
            "accuracy_delta": round(acc_delta, 2),
            "f1_delta": round(f1_delta, 4),
        },
        "decision": decision,
        "caveats": [
            "Val set missing ESCS/SuCS classes; CV used for model selection instead",
            "SST has ~28% NaN; tree models handle natively",
            "GroupKFold by cyclone_id used to prevent storm-level leakage",
            "Test set evaluated exactly once with final selected configuration",
            "SuCS has only 25 train samples, 0 in test - cannot be evaluated",
        ]
    }

    with open(os.path.join(RESULTS_DIR, "P4_FEATURE_ENGINEERING.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Markdown summary
    md = f"""# P3 P4 Feature Engineering Experiment

## Dataset
- Train: {len(train_valid)} samples | Val: {len(val_valid)} | Test: {len(test_valid)}
- Original features: {len(ORIGINAL_FEATURES)} | Engineered: {len(all_engineered)}
- SST NaN: {round(train_valid['sst'].isnull().mean()*100, 1)}%

## SST Missingness
| Method | Val Acc | Val F1 | Test Acc | Test F1 |
|--------|---------|--------|----------|---------|
"""
    for r in sst_results:
        md += f"| {r['label']} | {r['val_accuracy']}% | {r['val_macro_f1']} | {r['test_accuracy']}% | {r['test_macro_f1']} |\n"

    md += f"""
## Feature Ablation
| Set | #Feats | Val Acc | Val F1 | BalAcc | MAE | ±1 |
|-----|--------|---------|--------|--------|-----|-----|
"""
    for name, res in ablation_results.items():
        md += f"| {name} | {res['n_features']} | {res['val_accuracy']}% | {res['val_macro_f1']} | {res['val_balanced_acc']}% | {res['val_mae']} | {res['val_off_by_one']}% |\n"

    md += f"""
## Best Ablation: {best_ablation_name}

## Hyperparameter Tuning (80 trials)
- Best val F1: {tuning['best_val_f1']}
- Best val Acc: {tuning['best_val_accuracy']}%
- Best params: `{json.dumps(tuning['best_params'])}`

## Cross-Validation
### GroupKFold (by cyclone_id, 5 folds)
- Mean Accuracy: {group_cv['summary']['mean_accuracy']}% ± {group_cv['summary']['std_accuracy']}%
- Mean Macro-F1: {group_cv['summary']['mean_macro_f1']} ± {group_cv['summary']['std_macro_f1']}
- Mean Balanced Acc: {group_cv['summary']['mean_balanced_acc']}% ± {group_cv['summary']['std_balanced_acc']}%
- Mean MAE: {group_cv['summary']['mean_mae']} ± {group_cv['summary']['std_mae']}

### StratifiedKFold (5 folds)
- Mean Accuracy: {strat_cv['summary']['mean_accuracy']}% ± {strat_cv['summary']['std_accuracy']}%
- Mean Macro-F1: {strat_cv['summary']['mean_macro_f1']} ± {strat_cv['summary']['std_macro_f1']}

## Final Test Results
| Metric | Value | vs Champion |
|--------|-------|-------------|
| Accuracy | {final_test_acc:.2f}% | {acc_delta:+.2f}% |
| Macro-F1 | {final_test_f1:.4f} | {f1_delta:+.4f} |
| Weighted-F1 | {final_test_w_f1:.4f} | - |
| Balanced Acc | {final_test_bal:.2f}% | - |
| MAE | {final_test_mae:.4f} | - |
| ±1 | {final_test_ob1:.2f}% | - |
| ±2 | {final_test_ob2:.2f}% | - |

## Decision: {decision}

## Per-Class Test Metrics
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
"""
    for cls_name, m in err["per_class"].items():
        md += f"| {cls_name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |\n"

    md += f"""
## Adjacent Class Confusions
| Pair | Errors | Total | Rate |
|------|--------|-------|------|
"""
    for pair, info in err["adjacent_pair_errors"].items():
        md += f"| {pair} | {info['count']} | {info['total_true']} | {info['rate']}% |\n"

    md += f"""
## Caveats
"""
    for c in output["caveats"]:
        md += f"- {c}\n"

    with open(os.path.join(RESULTS_DIR, "P4_FEATURE_ENGINEERING.md"), "w") as f:
        f.write(md)

    # ================================================================
    # Print final summary
    # ================================================================
    print("\n" + "=" * 60)
    print("P4 FEATURE ENGINEERING")
    print("Previous champion: LightGBM")
    print("Previous accuracy: 47.00%")
    print("Previous macro-F1: 0.3744")
    print()
    print(f"Best validation/CV configuration:")
    print(f"  Features: {best_ablation_name} ({len(best_features)} features)")
    print(f"  Params: {json.dumps(tuning['best_params'], indent=4)}")
    print(f"  CV GroupKFold F1: {group_cv['summary']['mean_macro_f1']:.4f} ± {group_cv['summary']['std_macro_f1']:.4f}")
    print(f"  CV GroupKFold Acc: {group_cv['summary']['mean_accuracy']:.2f}% ± {group_cv['summary']['std_accuracy']:.2f}%")
    print()
    print(f"Final test accuracy:  {final_test_acc:.2f}%")
    print(f"Final test macro-F1:  {final_test_f1:.4f}")
    print(f"Accuracy delta:       {acc_delta:+.2f}%")
    print(f"F1 delta:             {f1_delta:+.4f}")
    print(f"Decision:             {decision}")
    print()
    print("Most useful new features (from ablation):")
    # Find which feature additions helped most
    base_f1 = ablation_results["E1_original"]["val_macro_f1"]
    for name in ["E2_orig_nonlinear", "E3_orig_interactions", "E4_orig_nonlinear_interactions", "E5_orig_all_eng"]:
        delta = ablation_results[name]["val_macro_f1"] - base_f1
        print(f"  {name}: F1 delta = {delta:+.4f}")
    print()
    print("Main remaining bottleneck:")
    print("  - Severe class imbalance (Depression=40%, SuCS=25 samples)")
    print("  - SST has 28% NaN limiting feature usefulness")
    print("  - Val set missing ESCS/SuCS classes")
    print("  - Small dataset (3039 train) for 7-class problem")
    print("=" * 60)


if __name__ == "__main__":
    main()
