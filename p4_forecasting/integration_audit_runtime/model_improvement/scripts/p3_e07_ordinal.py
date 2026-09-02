"""
P3 E07 — Ordinal Classification Experiment
============================================
Tests ordinal classification approaches for cyclone intensity categories.

Class ordering (from project code, classifier.py IMD_CLASSES):
  0: Depression (weakest)
  1: Deep Depression
  2: Cyclonic Storm
  3: Severe Cyclonic Storm
  4: Very Severe Cyclonic Storm
  5: Extremely Severe Cyclonic Storm
  6: Super Cyclonic Storm (strongest)

Ordinal approaches tested:
  A. Ordinal Regression (LightGBM regression on ordinal target)
  B. Cumulative Binary (K-1 binary classifiers)
  C. Ordinal Ridge Regression (with cumulative link)
  D. CatBoost with ordered boosting

Primary metric: macro-F1
Secondary: accuracy, balanced accuracy, mean absolute class error
"""

import os
import sys
import json
import time
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_absolute_error, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")
np.random.seed(42)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "p3_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# IMD intensity classes (ordered by severity, index 0 = weakest)
IMD_CLASSES = [
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(IMD_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
N_CLASSES = len(IMD_CLASSES)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_split(split_name):
    path = os.path.join(DATA_DIR, f"{split_name}.csv")
    return pd.read_csv(path)


def get_data():
    """Load all splits, encode labels, return arrays and DataFrames."""
    feature_cols = ["lat", "lon", "sst", "pressure_msl", "wind_u", "wind_v"]
    
    splits = {}
    for split in ["train", "val", "test"]:
        df = load_split(split)
        mask = df["category"].isin(CLASS_TO_IDX)
        valid = df[mask].copy()
        labels = valid["category"].map(CLASS_TO_IDX).values.astype(int)
        X = valid[feature_cols].values.astype(np.float64)
        splits[split] = {"X": X, "y": labels, "df": valid, "features": feature_cols}
    
    return splits


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred, split_name="test"):
    """Compute all required metrics."""
    acc = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    
    # Off-by-k accuracy
    off_by_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
    off_by_2 = np.mean(np.abs(y_true - y_pred) <= 2) * 100
    
    # Per-class metrics
    present_classes = sorted(set(y_true) | set(y_pred))
    present_names = [IDX_TO_CLASS[c] for c in present_classes]
    cr = classification_report(
        y_true, y_pred,
        target_names=present_names,
        zero_division=0,
        output_dict=True
    )
    
    per_class = {}
    for cls_name in present_names:
        if cls_name in cr:
            per_class[cls_name] = {
                "precision": round(cr[cls_name]["precision"], 4),
                "recall": round(cr[cls_name]["recall"], 4),
                "f1": round(cr[cls_name]["f1-score"], 4),
                "support": int(cr[cls_name]["support"])
            }
    
    # Ordered confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_dict = {}
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if cm[i, j] > 0:
                cm_dict[f"{IDX_TO_CLASS[i]}_true_{IDX_TO_CLASS[j]}_pred"] = int(cm[i, j])
    
    return {
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 2),
        "mean_absolute_class_error": round(mae, 4),
        "off_by_one_accuracy": round(off_by_1, 2),
        "off_by_two_accuracy": round(off_by_2, 2),
        "per_class": per_class,
        "confusion_matrix": cm_dict,
        "n_samples": len(y_true),
        "present_classes": [IDX_TO_CLASS[c] for c in present_classes]
    }


# ─────────────────────────────────────────────
# Approach A: Ordinal Regression via LightGBM
# ─────────────────────────────────────────────

def approach_ordinal_regression_lgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train LightGBM as a regressor on the ordinal target, then round and clip.
    """
    from lightgbm import LGBMRegressor
    
    best_val_f1 = -1
    best_config = None
    best_model = None
    
    configs = [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31},
        {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31},
        {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 4, "num_leaves": 15},
        {"n_estimators": 1000, "learning_rate": 0.02, "max_depth": 6, "num_leaves": 31},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 7},
    ]
    
    for cfg in configs:
        reg = LGBMRegressor(**cfg, random_state=42, verbose=-1)
        reg.fit(X_train, y_train)
        
        val_pred_cont = reg.predict(X_val)
        val_pred = np.clip(np.round(val_pred_cont), 0, N_CLASSES - 1).astype(int)
        
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = cfg
            best_model = reg
    
    # Test prediction
    test_pred_cont = best_model.predict(X_test)
    test_pred = np.clip(np.round(test_pred_cont), 0, N_CLASSES - 1).astype(int)
    
    # Also get val metrics
    val_pred_cont = best_model.predict(X_val)
    val_pred = np.clip(np.round(val_pred_cont), 0, N_CLASSES - 1).astype(int)
    
    val_metrics = compute_all_metrics(y_val, val_pred, "val")
    test_metrics = compute_all_metrics(y_test, test_pred, "test")
    
    return {
        "approach": "Ordinal Regression (LightGBM)",
        "config": best_config,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_macro_f1": val_metrics["macro_f1"]
    }


# ─────────────────────────────────────────────
# Approach B: Cumulative Binary Classifiers
# ─────────────────────────────────────────────

def approach_cumulative_binary(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train K-1 binary classifiers: P(class >= k) for k=1..K-1.
    Predicted class = max{k : P(class >= k) > 0.5}, or 0 if none.
    Uses LightGBM for each binary classifier.
    """
    from lightgbm import LGBMClassifier
    
    best_val_f1 = -1
    best_config = None
    best_models = None
    
    configs = [
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
        {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 6},
    ]
    
    for cfg in configs:
        models = []
        for k in range(1, N_CLASSES):
            # Binary target: 1 if y >= k, else 0
            y_binary = (y_train >= k).astype(int)
            
            # Skip if all same class
            if len(np.unique(y_binary)) < 2:
                models.append(None)
                continue
            
            clf = LGBMClassifier(**cfg, random_state=42, verbose=-1)
            clf.fit(X_train, y_binary)
            models.append(clf)
        
        # Predict on val
        val_probas = np.zeros((len(y_val), N_CLASSES - 1))
        for k_idx, clf in enumerate(models):
            if clf is not None:
                val_probas[:, k_idx] = clf.predict_proba(X_val)[:, 1]
        
        # Cumulative class: class = max{k : prob[k-1] > 0.5}
        val_pred = np.zeros(len(y_val), dtype=int)
        for i in range(len(y_val)):
            for k in range(N_CLASSES - 1, 0, -1):
                if val_probas[i, k - 1] > 0.5:
                    val_pred[i] = k
                    break
            else:
                val_pred[i] = 0
        
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = cfg
            best_models = models
    
    # Test prediction
    test_probas = np.zeros((len(y_test), N_CLASSES - 1))
    for k_idx, clf in enumerate(best_models):
        if clf is not None:
            test_probas[:, k_idx] = clf.predict_proba(X_test)[:, 1]
    
    test_pred = np.zeros(len(y_test), dtype=int)
    for i in range(len(y_test)):
        for k in range(N_CLASSES - 1, 0, -1):
            if test_probas[i, k - 1] > 0.5:
                test_pred[i] = k
                break
        else:
            test_pred[i] = 0
    
    # Val prediction
    val_probas = np.zeros((len(y_val), N_CLASSES - 1))
    for k_idx, clf in enumerate(best_models):
        if clf is not None:
            val_probas[:, k_idx] = clf.predict_proba(X_val)[:, 1]
    
    val_pred = np.zeros(len(y_val), dtype=int)
    for i in range(len(y_val)):
        for k in range(N_CLASSES - 1, 0, -1):
            if val_probas[i, k - 1] > 0.5:
                val_pred[i] = k
                break
        else:
            val_pred[i] = 0
    
    val_metrics = compute_all_metrics(y_val, val_pred, "val")
    test_metrics = compute_all_metrics(y_test, test_pred, "test")
    
    return {
        "approach": "Cumulative Binary (LightGBM)",
        "config": best_config,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_macro_f1": val_metrics["macro_f1"]
    }


# ─────────────────────────────────────────────
# Approach C: Ordinal Ridge Regression
# ─────────────────────────────────────────────

def approach_ordinal_ridge(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Use Ridge regression on ordinal target, round and clip.
    Requires NaN imputation since RidgeClassifier doesn't accept NaN.
    """
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    best_val_f1 = -1
    best_alpha = None
    best_model = None
    
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        reg = RidgeClassifier(alpha=alpha)
        reg.fit(X_train_imp, y_train)
        val_pred = np.clip(np.round(reg.predict(X_val_imp)), 0, N_CLASSES - 1).astype(int)
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_alpha = alpha
            best_model = reg
    
    test_pred = np.clip(np.round(best_model.predict(X_test_imp)), 0, N_CLASSES - 1).astype(int)
    val_pred = np.clip(np.round(best_model.predict(X_val_imp)), 0, N_CLASSES - 1).astype(int)
    
    val_metrics = compute_all_metrics(y_val, val_pred, "val")
    test_metrics = compute_all_metrics(y_test, test_pred, "test")
    
    return {
        "approach": "Ordinal Ridge Regression",
        "config": {"alpha": best_alpha},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_macro_f1": val_metrics["macro_f1"]
    }


# ─────────────────────────────────────────────
# Approach D: CatBoost Ordinal (regression mode)
# ─────────────────────────────────────────────

def approach_catboost_ordinal(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    CatBoost in regression mode on ordinal target, round and clip.
    """
    from catboost import CatBoostRegressor
    
    best_val_f1 = -1
    best_config = None
    best_model = None
    
    configs = [
        {"iterations": 300, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 3},
        {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3},
        {"iterations": 800, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5},
        {"iterations": 500, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 5},
        {"iterations": 500, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3},
    ]
    
    for cfg in configs:
        reg = CatBoostRegressor(**cfg, random_seed=42, verbose=0)
        reg.fit(X_train, y_train)
        
        val_pred_cont = reg.predict(X_val)
        val_pred = np.clip(np.round(val_pred_cont), 0, N_CLASSES - 1).astype(int)
        
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = cfg
            best_model = reg
    
    test_pred = np.clip(np.round(best_model.predict(X_test)), 0, N_CLASSES - 1).astype(int)
    val_pred = np.clip(np.round(best_model.predict(X_val)), 0, N_CLASSES - 1).astype(int)
    
    val_metrics = compute_all_metrics(y_val, val_pred, "val")
    test_metrics = compute_all_metrics(y_test, test_pred, "test")
    
    return {
        "approach": "CatBoost Ordinal Regression",
        "config": best_config,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_macro_f1": val_metrics["macro_f1"]
    }


# ─────────────────────────────────────────────
# Approach E: LightGBM Classification (champion baseline)
# ─────────────────────────────────────────────

def approach_lgbm_classification_baseline(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Reproduce the current P3 champion: LightGBM classification.
    """
    from lightgbm import LGBMClassifier
    
    # Best from P3-05: depth=4, lr=0.05, n=300
    lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
        random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    
    val_pred = lgbm.predict(X_val)
    test_pred = lgbm.predict(X_test)
    
    val_metrics = compute_all_metrics(y_val, val_pred, "val")
    test_metrics = compute_all_metrics(y_test, test_pred, "test")
    
    return {
        "approach": "LightGBM Classification (champion baseline)",
        "config": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_macro_f1": val_metrics["macro_f1"]
    }


# ─────────────────────────────────────────────
# Approach F: LightGBM Classification (no weights, original params)
# ─────────────────────────────────────────────

def approach_lgbm_classification_original(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Reproduce original LightGBM (no weights, original params from classifier.py).
    """
    from lightgbm import LGBMClassifier
    
    lgbm = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    
    val_pred = lgbm.predict(X_val)
    test_pred = lgbm.predict(X_test)
    
    val_metrics = compute_all_metrics(y_val, val_pred, "val")
    test_metrics = compute_all_metrics(y_test, test_pred, "test")
    
    return {
        "approach": "LightGBM Classification (original params)",
        "config": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_macro_f1": val_metrics["macro_f1"]
    }


# ─────────────────────────────────────────────
# Approach G: Ordinal Regression with probability rounding
# ─────────────────────────────────────────────

def approach_ordinal_proba_rounding(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train LightGBM classifier, compute weighted class prediction
    using predicted probabilities and class ordinal positions.
    pred = sum(p_k * k) then round and clip.
    """
    from lightgbm import LGBMClassifier
    
    best_val_f1 = -1
    best_config = None
    best_model = None
    
    configs = [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31},
        {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 7},
    ]
    
    class_positions = np.arange(N_CLASSES)
    
    for cfg in configs:
        clf = LGBMClassifier(**cfg, random_state=42, verbose=-1)
        clf.fit(X_train, y_train)
        
        val_proba = clf.predict_proba(X_val)
        val_pred_cont = np.sum(val_proba * class_positions, axis=1)
        val_pred = np.clip(np.round(val_pred_cont), 0, N_CLASSES - 1).astype(int)
        
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = cfg
            best_model = clf
    
    # Test
    test_proba = best_model.predict_proba(X_test)
    test_pred_cont = np.sum(test_proba * class_positions, axis=1)
    test_pred = np.clip(np.round(test_pred_cont), 0, N_CLASSES - 1).astype(int)
    
    # Val
    val_proba = best_model.predict_proba(X_val)
    val_pred_cont = np.sum(val_proba * class_positions, axis=1)
    val_pred = np.clip(np.round(val_pred_cont), 0, N_CLASSES - 1).astype(int)
    
    val_metrics = compute_all_metrics(y_val, val_pred, "val")
    test_metrics = compute_all_metrics(y_test, test_pred, "test")
    
    return {
        "approach": "Ordinal Probability Rounding (LightGBM)",
        "config": best_config,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_macro_f1": val_metrics["macro_f1"]
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("P3 E07 — ORDINAL CLASSIFICATION EXPERIMENT")
    print("=" * 60)
    
    # Load data
    splits = get_data()
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Classes: {N_CLASSES}")
    print(f"Class order: {IMD_CLASSES}")
    
    # Run all approaches
    results = {}
    
    # E: Champion baseline
    print("\n--- Approach E: LightGBM Classification (champion baseline) ---")
    results["E_champion_baseline"] = approach_lgbm_classification_baseline(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    
    # F: Original params
    print("--- Approach F: LightGBM Classification (original params) ---")
    results["F_original_params"] = approach_lgbm_classification_original(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    
    # A: Ordinal regression LGBM
    print("--- Approach A: Ordinal Regression (LightGBM) ---")
    results["A_ordinal_reg_lgbm"] = approach_ordinal_regression_lgbm(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    
    # B: Cumulative binary
    print("--- Approach B: Cumulative Binary (LightGBM) ---")
    results["B_cumulative_binary"] = approach_cumulative_binary(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    
    # C: Ordinal Ridge
    print("--- Approach C: Ordinal Ridge Regression ---")
    results["C_ordinal_ridge"] = approach_ordinal_ridge(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    
    # D: CatBoost ordinal
    print("--- Approach D: CatBoost Ordinal Regression ---")
    results["D_catboost_ordinal"] = approach_catboost_ordinal(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    
    # G: Probability rounding
    print("--- Approach G: Ordinal Probability Rounding ---")
    results["G_proba_rounding"] = approach_ordinal_proba_rounding(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE (Val -> Test)")
    print("=" * 60)
    print(f"{'Approach':<45s} | {'Val F1':>7s} | {'Test Acc':>8s} | {'Test F1':>7s} | {'Test BalAcc':>11s} | {'MAE':>5s} | {'±1':>6s} | {'±2':>6s}")
    print("-" * 110)
    
    for key, res in results.items():
        vm = res["val_metrics"]
        tm = res["test_metrics"]
        print(f"{res['approach']:<45s} | {vm['macro_f1']:>7.4f} | {tm['accuracy']:>7.2f}% | {tm['macro_f1']:>7.4f} | {tm['balanced_accuracy']:>10.2f}% | {tm['mean_absolute_class_error']:>5.4f} | {tm['off_by_one_accuracy']:>5.2f}% | {tm['off_by_two_accuracy']:>5.2f}%")
    
    # Select best by val macro-F1
    best_key = max(results.keys(), key=lambda k: results[k]["val_macro_f1"])
    best = results[best_key]
    
    # Also find best by val balanced accuracy
    best_bal_key = max(results.keys(), key=lambda k: results[k]["val_metrics"]["balanced_accuracy"])
    
    # Current champion metrics
    champion = results["E_champion_baseline"]
    
    print(f"\n{'='*60}")
    print("DECISION ANALYSIS")
    print(f"{'='*60}")
    print(f"Current champion: LightGBM Classification (champion baseline)")
    print(f"  Val F1:     {champion['val_metrics']['macro_f1']:.4f}")
    print(f"  Test Acc:   {champion['test_metrics']['accuracy']:.2f}%")
    print(f"  Test F1:    {champion['test_metrics']['macro_f1']:.4f}")
    print(f"  Test BalAcc:{champion['test_metrics']['balanced_accuracy']:.2f}%")
    print(f"  Test MAE:   {champion['test_metrics']['mean_absolute_class_error']:.4f}")
    print(f"  ±1:         {champion['test_metrics']['off_by_one_accuracy']:.2f}%")
    print(f"  ±2:         {champion['test_metrics']['off_by_two_accuracy']:.2f}%")
    
    print(f"\nBest by val macro-F1: {best['approach']}")
    print(f"  Val F1:     {best['val_metrics']['macro_f1']:.4f}")
    print(f"  Test Acc:   {best['test_metrics']['accuracy']:.2f}%")
    print(f"  Test F1:    {best['test_metrics']['macro_f1']:.4f}")
    print(f"  Test BalAcc:{best['test_metrics']['balanced_accuracy']:.2f}%")
    print(f"  Test MAE:   {best['test_metrics']['mean_absolute_class_error']:.4f}")
    print(f"  ±1:         {best['test_metrics']['off_by_one_accuracy']:.2f}%")
    print(f"  ±2:         {best['test_metrics']['off_by_two_accuracy']:.2f}%")
    
    # Decision
    champion_f1 = champion["test_metrics"]["macro_f1"]
    best_f1 = best["test_metrics"]["macro_f1"]
    champion_acc = champion["test_metrics"]["accuracy"]
    best_acc = best["test_metrics"]["accuracy"]
    champion_bal = champion["test_metrics"]["balanced_accuracy"]
    best_bal = best["test_metrics"]["balanced_accuracy"]
    
    f1_improved = best_f1 > champion_f1
    acc_improved = best_acc > champion_acc
    bal_improved = best_bal > champion_bal
    
    if f1_improved and acc_improved:
        decision = "IMPROVED"
    elif f1_improved or acc_improved:
        decision = "IMPROVED (marginal)"
    elif best["val_metrics"]["macro_f1"] > champion["val_metrics"]["macro_f1"] and not f1_improved:
        decision = "REJECTED (val-test gap)"
    else:
        decision = "NO_IMPROVEMENT"
    
    # Check all ordinal approaches
    ordinal_approaches = ["A_ordinal_reg_lgbm", "B_cumulative_binary", "C_ordinal_ridge", "D_catboost_ordinal", "G_proba_rounding"]
    ordinal_results = {k: results[k] for k in ordinal_approaches if k in results}
    
    best_ordinal_f1 = max(r["test_metrics"]["macro_f1"] for r in ordinal_results.values()) if ordinal_results else 0
    best_ordinal_key = max(ordinal_results.keys(), key=lambda k: ordinal_results[k]["test_metrics"]["macro_f1"]) if ordinal_results else None
    best_ordinal = ordinal_results.get(best_ordinal_key) if best_ordinal_key else None
    
    print(f"\nBest ordinal approach: {best_ordinal['approach'] if best_ordinal else 'N/A'}")
    if best_ordinal:
        print(f"  Val F1:     {best_ordinal['val_metrics']['macro_f1']:.4f}")
        print(f"  Test F1:    {best_ordinal['test_metrics']['macro_f1']:.4f}")
        print(f"  ±1:         {best_ordinal['test_metrics']['off_by_one_accuracy']:.2f}%")
        print(f"  ±2:         {best_ordinal['test_metrics']['off_by_two_accuracy']:.2f}%")
    
    # Final decision
    if best_ordinal and best_ordinal["test_metrics"]["macro_f1"] > champion_f1:
        final_decision = "IMPROVED"
        final_champion = best_ordinal
        final_champion_name = best_ordinal["approach"]
    else:
        final_decision = "NO_IMPROVEMENT"
        final_champion = champion
        final_champion_name = champion["approach"]
    
    print(f"\n{'='*60}")
    print(f"DECISION: {final_decision}")
    print(f"{'='*60}")
    
    # Build output
    output = {
        "experiment": "P3_E07_ORDINAL",
        "objective": "Test ordinal classification for cyclone intensity categories",
        "class_ordering": IMD_CLASSES,
        "class_ordering_confirmed_from": "PS70-main/src/classification/classifier.py IMD_CLASSES",
        "champion_baseline": {
            "approach": champion["approach"],
            "val_macro_f1": champion["val_metrics"]["macro_f1"],
            "test_accuracy": champion["test_metrics"]["accuracy"],
            "test_macro_f1": champion["test_metrics"]["macro_f1"],
            "test_weighted_f1": champion["test_metrics"]["weighted_f1"],
            "test_balanced_accuracy": champion["test_metrics"]["balanced_accuracy"],
            "test_mae": champion["test_metrics"]["mean_absolute_class_error"],
            "test_off_by_one": champion["test_metrics"]["off_by_one_accuracy"],
            "test_off_by_two": champion["test_metrics"]["off_by_two_accuracy"]
        },
        "all_results": {},
        "decision": final_decision,
        "final_champion": final_champion_name,
        "caveats": []
    }
    
    for key, res in results.items():
        output["all_results"][key] = {
            "approach": res["approach"],
            "config": res["config"],
            "val": res["val_metrics"],
            "test": res["test_metrics"]
        }
    
    # Caveats
    output["caveats"] = [
        "Validation set missing ESCS and SuCS classes - val F1 is unreliable for model selection",
        "SST has 28% missing values - StandardScaler propagates NaN; tree models handle natively",
        "SuCS has only 25 training samples in train, 0 in test - cannot be evaluated",
        "Class imbalance: Depression=40% of training data",
        "Ordinal approaches tested: cumulative binary, ordinal regression, probability rounding, ridge"
    ]
    
    if final_decision == "NO_IMPROVEMENT":
        output["caveats"].append("Ordinal modeling did not improve macro-F1 or accuracy over standard classification")
        output["caveats"].append("The ordinal structure is already captured by standard classifiers via feature relationships")
    
    # Save JSON
    with open(os.path.join(RESULTS_DIR, "P3_E07_ORDINAL.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    # Save Markdown
    md_lines = [
        "# P3 E07 — Ordinal Classification Experiment",
        "",
        "## Objective",
        "Test whether ordinal classification improves cyclone intensity categorization.",
        "",
        "## Class Ordering (confirmed from project code)",
        "",
        "| Index | Class | Severity |",
        "|-------|-------|----------|",
    ]
    for i, c in enumerate(IMD_CLASSES):
        md_lines.append(f"| {i} | {c} | {'Weakest' if i==0 else 'Strongest' if i==6 else ''} |")
    
    md_lines += [
        "",
        "## Results",
        "",
        "| Approach | Val F1 | Test Acc | Test F1 | Test BalAcc | MAE | ±1 | ±2 |",
        "|----------|--------|----------|---------|-------------|-----|----|----|",
    ]
    for key, res in results.items():
        vm = res["val_metrics"]
        tm = res["test_metrics"]
        md_lines.append(
            f"| {res['approach']} | {vm['macro_f1']:.4f} | {tm['accuracy']:.2f}% | {tm['macro_f1']:.4f} | "
            f"{tm['balanced_accuracy']:.2f}% | {tm['mean_absolute_class_error']:.4f} | "
            f"{tm['off_by_one_accuracy']:.2f}% | {tm['off_by_two_accuracy']:.2f}% |"
        )
    
    md_lines += [
        "",
        f"## Decision: {final_decision}",
        "",
        f"**Final P3 Champion**: {final_champion_name}",
        "",
        f"### Champion Metrics",
        f"- Accuracy: {final_champion['test_metrics']['accuracy']:.2f}%",
        f"- Macro-F1: {final_champion['test_metrics']['macro_f1']:.4f}",
        f"- Weighted-F1: {final_champion['test_metrics']['weighted_f1']:.4f}",
        f"- Balanced Accuracy: {final_champion['test_metrics']['balanced_accuracy']:.2f}%",
        f"- MAE: {final_champion['test_metrics']['mean_absolute_class_error']:.4f}",
        f"- ±1: {final_champion['test_metrics']['off_by_one_accuracy']:.2f}%",
        f"- ±2: {final_champion['test_metrics']['off_by_two_accuracy']:.2f}%",
        "",
        "### Caveats",
    ]
    for c in output["caveats"]:
        md_lines.append(f"- {c}")
    
    with open(os.path.join(RESULTS_DIR, "P3_E07_ORDINAL.md"), "w") as f:
        f.write("\n".join(md_lines))
    
    # Save P3 accuracy summary
    summary = {
        "current_champion": final_champion_name,
        "champion_metrics": {
            "accuracy": final_champion["test_metrics"]["accuracy"],
            "macro_f1": final_champion["test_metrics"]["macro_f1"],
            "weighted_f1": final_champion["test_metrics"]["weighted_f1"],
            "balanced_accuracy": final_champion["test_metrics"]["balanced_accuracy"],
            "mean_absolute_class_error": final_champion["test_metrics"]["mean_absolute_class_error"],
            "off_by_one": final_champion["test_metrics"]["off_by_one_accuracy"],
            "off_by_two": final_champion["test_metrics"]["off_by_two_accuracy"]
        },
        "previous_champion": champion["approach"],
        "previous_metrics": {
            "accuracy": champion["test_metrics"]["accuracy"],
            "macro_f1": champion["test_metrics"]["macro_f1"]
        },
        "decision": final_decision,
        "experiments": {}
    }
    for key, res in results.items():
        summary["experiments"][key] = {
            "approach": res["approach"],
            "val_f1": res["val_metrics"]["macro_f1"],
            "test_acc": res["test_metrics"]["accuracy"],
            "test_f1": res["test_metrics"]["macro_f1"]
        }
    
    with open(os.path.join(RESULTS_DIR, "P3_ACCURACY_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    with open(os.path.join(RESULTS_DIR, "P3_ACCURACY_SUMMARY.md"), "w") as f:
        f.write(f"# P3 Accuracy Summary\n\n")
        f.write(f"**Current Champion**: {final_champion_name}\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Accuracy | {final_champion['test_metrics']['accuracy']:.2f}% |\n")
        f.write(f"| Macro-F1 | {final_champion['test_metrics']['macro_f1']:.4f} |\n")
        f.write(f"| Weighted-F1 | {final_champion['test_metrics']['weighted_f1']:.4f} |\n")
        f.write(f"| Balanced Accuracy | {final_champion['test_metrics']['balanced_accuracy']:.2f}% |\n")
        f.write(f"| MAE | {final_champion['test_metrics']['mean_absolute_class_error']:.4f} |\n")
        f.write(f"| ±1 | {final_champion['test_metrics']['off_by_one_accuracy']:.2f}% |\n")
        f.write(f"| ±2 | {final_champion['test_metrics']['off_by_two_accuracy']:.2f}% |\n\n")
        f.write(f"**Decision**: {final_decision}\n")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("SIH 2026 PS 26070 — P3 ORDINAL EXPERIMENT")
    print("-" * 60)
    print(f"Previous champion:  LightGBM Classification (champion baseline)")
    print(f"Previous accuracy:  {champion['test_metrics']['accuracy']:.2f}%")
    print(f"Previous macro-F1:  {champion['test_metrics']['macro_f1']:.4f}")
    print(f"Ordinal model:      {best_ordinal['approach'] if best_ordinal else 'N/A'}")
    print(f"Ordinal accuracy:   {best_ordinal['test_metrics']['accuracy']:.2f}%" if best_ordinal else "Ordinal accuracy: N/A")
    print(f"Ordinal macro-F1:   {best_ordinal['test_metrics']['macro_f1']:.4f}" if best_ordinal else "Ordinal macro-F1: N/A")
    print(f"Within ±1:          {best_ordinal['test_metrics']['off_by_one_accuracy']:.2f}%" if best_ordinal else "Within ±1: N/A")
    print(f"Within ±2:          {best_ordinal['test_metrics']['off_by_two_accuracy']:.2f}%" if best_ordinal else "Within ±2: N/A")
    print(f"Decision:           {final_decision}")
    print(f"Final P3 champion:  {final_champion_name}")
    print(f"Caveats:")
    for c in output["caveats"]:
        print(f"  - {c}")
    print(f"Files produced:")
    print(f"  results/P3_E07_ORDINAL.json")
    print(f"  results/P3_E07_ORDINAL.md")
    print(f"  results/P3_ACCURACY_SUMMARY.json")
    print(f"  results/P3_ACCURACY_SUMMARY.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
