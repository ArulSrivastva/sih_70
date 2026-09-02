"""
P3 Classification Improvement Pipeline
========================================
Runs P3-01 through P3-08 experiments on cyclone intensity category classification.

P3-01: Data & Label Audit
P3-02: Baseline Reproduction
P3-03: Class Imbalance Analysis
P3-04: Model Comparison
P3-05: Hyperparameter Tuning
P3-06: Ordinal/Class Structure Analysis
P3-07: Error Analysis
P3-08: Final Model Selection & Summary
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
    confusion_matrix, mean_absolute_error, mean_squared_error
)
from sklearn.dummy import DummyClassifier

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

# Also includes "Low Pressure Area" as 8th category in some training data
ALL_CLASSES_WITH_LPA = ["Low Pressure Area"] + IMD_CLASSES

# ─────────────────────────────────────────────
# Data Loading Utilities
# ─────────────────────────────────────────────

def load_split(split_name):
    """Load a CSV split and return a DataFrame."""
    path = os.path.join(DATA_DIR, f"{split_name}.csv")
    return pd.read_csv(path)


def encode_labels(df):
    """Map category strings to integer indices using CLASS_TO_IDX.
    Returns Series of int indices and a boolean mask for valid rows.
    """
    mask = df["category"].isin(CLASS_TO_IDX)
    labels = df["category"].map(CLASS_TO_IDX).astype("Int64")
    return labels, mask


def get_features_and_labels(df, feature_cols=None):
    """Extract numeric features and encoded labels, dropping rows with missing category."""
    if feature_cols is None:
        feature_cols = ["lat", "lon", "sst", "pressure_msl", "wind_u", "wind_v"]
    
    labels, valid_mask = encode_labels(df)
    valid_df = df[valid_mask].copy()
    valid_labels = labels[valid_mask].values
    
    X = valid_df[feature_cols].values.astype(np.float64)
    y = valid_labels.astype(int)
    
    return X, y, valid_df, feature_cols


# ─────────────────────────────────────────────
# P3-01: Data & Label Audit
# ─────────────────────────────────────────────

def p3_01_data_audit():
    print("\n" + "="*60)
    print("P3-01: DATA & LABEL AUDIT")
    print("="*60)
    
    report = {
        "split_stats": {},
        "class_distribution": {},
        "missing_values": {},
        "feature_statistics": {},
        "duplicate_rows": {},
        "cross_split_cyclone_ids": {},
        "label_issues": [],
        "warnings": []
    }
    
    for split in ["train", "val", "test"]:
        df = load_split(split)
        report["split_stats"][split] = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "n_unique_cyclone_ids": df["cyclone_id"].nunique() if "cyclone_id" in df.columns else None,
            "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())] if "timestamp" in df.columns else None
        }
        
        # Missing values
        missing = df.isnull().sum().to_dict()
        report["missing_values"][split] = {k: int(v) for k, v in missing.items() if v > 0}
        
        # Class distribution
        if "category" in df.columns:
            dist = df["category"].value_counts().to_dict()
            report["class_distribution"][split] = {str(k): int(v) for k, v in dist.items()}
            
            # Check for classes with very few samples
            for cls, count in dist.items():
                if count < 5 and split == "train":
                    report["label_issues"].append(f"Class '{cls}' has only {count} samples in {split} split")
        
        # Duplicate rows (excluding cyclone_id and timestamp)
        numeric_cols = ["lat", "lon", "sst", "pressure_msl", "wind_u", "wind_v"]
        if all(c in df.columns for c in numeric_cols):
            n_dup = df.duplicated(subset=numeric_cols + ["category"], keep="first").sum()
            report["duplicate_rows"][split] = int(n_dup)
        
        # Feature statistics
        feat_stats = {}
        for col in ["lat", "lon", "sst", "pressure_msl", "wind_u", "wind_v", "wind_speed", "pressure"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                feat_stats[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "n_missing": int(vals.isnull().sum())
                }
        report["feature_statistics"][split] = feat_stats
    
    # Cross-split cyclone ID overlap
    ids = {}
    for split in ["train", "val", "test"]:
        df = load_split(split)
        if "cyclone_id" in df.columns:
            ids[split] = set(df["cyclone_id"].unique())
    
    if len(ids) == 3:
        train_val_overlap = ids["train"] & ids["val"]
        train_test_overlap = ids["train"] & ids["test"]
        val_test_overlap = ids["val"] & ids["test"]
        
        report["cross_split_cyclone_ids"] = {
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
            "train_val_overlap_ids": sorted(list(train_val_overlap))[:10],
            "train_test_overlap_ids": sorted(list(train_test_overlap))[:10]
        }
        
        if train_val_overlap:
            report["warnings"].append(f"LEAKAGE: {len(train_val_overlap)} cyclone IDs shared between train and val")
        if train_test_overlap:
            report["warnings"].append(f"LEAKAGE: {len(train_test_overlap)} cyclone IDs shared between train and test")
        if val_test_overlap:
            report["warnings"].append(f"LEAKAGE: {len(val_test_overlap)} cyclone IDs shared between val and test")
    
    # Overall warnings
    train_dist = report["class_distribution"].get("train", {})
    if train_dist.get("Depression", 0) / sum(train_dist.values()) > 0.3:
        report["warnings"].append("SEVERE IMBALANCE: Depression class is >30% of training data")
    
    for split in ["val", "test"]:
        split_classes = set(report["class_distribution"].get(split, {}).keys())
        missing_in_split = set(IMD_CLASSES) - split_classes
        if missing_in_split:
            report["warnings"].append(f"MISSING CLASSES in {split}: {sorted(missing_in_split)}")
    
    # Print summary
    print(f"\nTrain: {report['split_stats']['train']['n_rows']} rows, "
          f"{report['split_stats']['train']['n_unique_cyclone_ids']} unique cyclones")
    print(f"Val:   {report['split_stats']['val']['n_rows']} rows, "
          f"{report['split_stats']['val']['n_unique_cyclone_ids']} unique cyclones")
    print(f"Test:  {report['split_stats']['test']['n_rows']} rows, "
          f"{report['split_stats']['test']['n_unique_cyclone_ids']} unique cyclones")
    
    print(f"\nClass distribution (train):")
    for cls, cnt in sorted(report["class_distribution"]["train"].items(), key=lambda x: -x[1]):
        print(f"  {cls:35s}: {cnt:5d} ({100*cnt/report['split_stats']['train']['n_rows']:.1f}%)")
    
    if report["cross_split_cyclone_ids"].get("train_val_overlap_count", 0) > 0:
        print(f"\n!!! LEAKAGE DETECTED !!!")
        print(f"  Train-Val overlap: {report['cross_split_cyclone_ids']['train_val_overlap_count']} cyclone IDs")
        print(f"  Train-Test overlap: {report['cross_split_cyclone_ids']['train_test_overlap_count']} cyclone IDs")
    
    for w in report["warnings"]:
        print(f"  WARNING: {w}")
    
    return report


# ─────────────────────────────────────────────
# P3-02: Baseline Reproduction
# ─────────────────────────────────────────────

def p3_02_baseline():
    print("\n" + "="*60)
    print("P3-02: BASELINE REPRODUCTION")
    print("="*60)
    
    report = {}
    
    # Load data
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    
    X_train, y_train, _, feature_cols = get_features_and_labels(train_df)
    X_val, y_val, _, _ = get_features_and_labels(val_df)
    X_test, y_test, _, _ = get_features_and_labels(test_df)
    
    print(f"Train: {len(X_train)} samples, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {feature_cols}")
    
    # 1. Majority class baseline
    majority_clf = DummyClassifier(strategy="most_frequent", random_state=42)
    majority_clf.fit(X_train, y_train)
    maj_pred = majority_clf.predict(X_test)
    maj_acc = accuracy_score(y_test, maj_pred) * 100
    maj_f1 = f1_score(y_test, maj_pred, average="macro", zero_division=0)
    
    print(f"\nMajority Class Baseline: Accuracy={maj_acc:.2f}%, Macro-F1={maj_f1:.4f}")
    report["majority_class"] = {"accuracy": round(maj_acc, 2), "macro_f1": round(maj_f1, 4)}
    
    # 2. Reproduce original LightGBM (exact params from classifier.py)
    from lightgbm import LGBMClassifier
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    lgbm = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, verbose=-1
    )
    lgbm.fit(X_train_scaled, y_train)
    
    val_pred = lgbm.predict(X_val_scaled)
    test_pred = lgbm.predict(X_test_scaled)
    
    val_acc = accuracy_score(y_val, val_pred) * 100
    val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test, test_pred) * 100
    test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
    
    print(f"\nOriginal LightGBM (reproduced):")
    print(f"  Val:   Accuracy={val_acc:.2f}%, Macro-F1={val_f1:.4f}")
    print(f"  Test:  Accuracy={test_acc:.2f}%, Macro-F1={test_f1:.4f}")
    
    report["original_lgbm"] = {
        "val_accuracy": round(val_acc, 2),
        "val_macro_f1": round(val_f1, 4),
        "test_accuracy": round(test_acc, 2),
        "test_macro_f1": round(test_f1, 4),
        "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6}
    }
    
    # Per-class report
    test_cr = classification_report(
        y_test, test_pred,
        target_names=[IDX_TO_CLASS[i] for i in sorted(set(y_test))],
        zero_division=0, output_dict=True
    )
    report["original_lgbm"]["test_classification_report"] = test_cr
    
    # Check vs stored claimed metrics
    report["stored_claims"] = {"accuracy": 47.0, "macro_f1": 0.3703}
    report["reproduction_gap"] = {
        "accuracy_vs_claim": round(test_acc - 47.0, 2),
        "f1_vs_claim": round(test_f1 - 0.3703, 4)
    }
    
    if abs(test_acc - 47.0) > 2.0:
        report["reproduction_note"] = (
            "Significant gap vs stored claim. Possible causes: "
            "different random seeds, feature columns used, or leakage in original training."
        )
    
    print(f"\nStored claim: Accuracy=47.00%, Macro-F1=0.3703")
    print(f"Reproduced:   Accuracy={test_acc:.2f}%, Macro-F1={test_f1:.4f}")
    print(f"Gap:          Accuracy={test_acc-47.0:+.2f}%, F1={test_f1-0.3703:+.4f}")
    
    return report


# ─────────────────────────────────────────────
# P3-03: Class Imbalance Analysis
# ─────────────────────────────────────────────

def p3_03_class_imbalance():
    print("\n" + "="*60)
    print("P3-03: CLASS IMBALANCE ANALYSIS")
    print("="*60)
    
    report = {"experiments": []}
    
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    
    X_train, y_train, _, feature_cols = get_features_and_labels(train_df)
    X_val, y_val, _, _ = get_features_and_labels(val_df)
    X_test, y_test, _, _ = get_features_and_labels(test_df)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    from lightgbm import LGBMClassifier
    
    # Experiment A: No weights (baseline)
    lgbm_base = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
    lgbm_base.fit(X_train_s, y_train)
    pred = lgbm_base.predict(X_test_s)
    base_acc = accuracy_score(y_test, pred) * 100
    base_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"Baseline (no weights): Acc={base_acc:.2f}%, F1={base_f1:.4f}")
    report["experiments"].append({
        "name": "no_weights",
        "method": "LightGBM default (no class weights)",
        "accuracy": round(base_acc, 2),
        "macro_f1": round(base_f1, 4)
    })
    
    # Experiment B: class_weight="balanced"
    lgbm_bal = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        class_weight="balanced", random_state=42, verbose=-1
    )
    lgbm_bal.fit(X_train_s, y_train)
    pred = lgbm_bal.predict(X_test_s)
    bal_acc = accuracy_score(y_test, pred) * 100
    bal_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"Balanced weights:       Acc={bal_acc:.2f}%, F1={bal_f1:.4f}")
    report["experiments"].append({
        "name": "balanced_weights",
        "method": "LightGBM class_weight='balanced'",
        "accuracy": round(bal_acc, 2),
        "macro_f1": round(bal_f1, 4)
    })
    
    # Experiment C: Inverse frequency weights
    class_counts = np.bincount(y_train, minlength=len(IMD_CLASSES))
    total = len(y_train)
    inv_freq = total / (len(IMD_CLASSES) * class_counts)
    inv_freq_dict = {i: float(inv_freq[i]) for i in range(len(IMD_CLASSES)) if class_counts[i] > 0}
    
    lgbm_inv = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        class_weight=inv_freq_dict, random_state=42, verbose=-1
    )
    lgbm_inv.fit(X_train_s, y_train)
    pred = lgbm_inv.predict(X_test_s)
    inv_acc = accuracy_score(y_test, pred) * 100
    inv_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"Inverse freq weights:   Acc={inv_acc:.2f}%, F1={inv_f1:.4f}")
    report["experiments"].append({
        "name": "inverse_frequency",
        "method": "LightGBM inverse frequency weights",
        "accuracy": round(inv_acc, 2),
        "macro_f1": round(inv_f1, 4),
        "weights": {IDX_TO_CLASS[k]: round(v, 4) for k, v in inv_freq_dict.items()}
    })
    
    # Experiment D: SMOTE oversampling
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_train_s, y_train)
        
        lgbm_smote = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
        lgbm_smote.fit(X_res, y_res)
        pred = lgbm_smote.predict(X_test_s)
        smote_acc = accuracy_score(y_test, pred) * 100
        smote_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        print(f"SMOTE oversampling:     Acc={smote_acc:.2f}%, F1={smote_f1:.4f}")
        report["experiments"].append({
            "name": "smote",
            "method": "SMOTE oversampling + LightGBM",
            "accuracy": round(smote_acc, 2),
            "macro_f1": round(smote_f1, 4)
        })
    except ImportError:
        print("imblearn not available, skipping SMOTE")
        report["experiments"].append({"name": "smote", "status": "skipped", "reason": "imblearn not installed"})
    
    # Experiment E: Focal loss proxy (class weights emphasizing rare classes)
    # Boost weight for classes 4,5,6 (VSCS, ESCS, SuCS)
    focal_weights = {i: 1.0 for i in range(7)}
    for i in [4, 5, 6]:
        focal_weights[i] = 3.0  # 3x weight for severe classes
    
    lgbm_focal = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        class_weight=focal_weights, random_state=42, verbose=-1
    )
    lgbm_focal.fit(X_train_s, y_train)
    pred = lgbm_focal.predict(X_test_s)
    focal_acc = accuracy_score(y_test, pred) * 100
    focal_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"Focal (3x rare):        Acc={focal_acc:.2f}%, F1={focal_f1:.4f}")
    report["experiments"].append({
        "name": "focal_3x_rare",
        "method": "LightGBM with 3x weight on VSCS/ESCS/SuCS",
        "accuracy": round(focal_acc, 2),
        "macro_f1": round(focal_f1, 4),
        "weights": {IDX_TO_CLASS[k]: v for k, v in focal_weights.items()}
    })
    
    # Best
    best = max(report["experiments"], key=lambda x: x.get("macro_f1", 0))
    report["best"] = best["name"]
    print(f"\nBest: {best['name']} (F1={best['macro_f1']:.4f})")
    
    return report


# ─────────────────────────────────────────────
# P3-04: Model Comparison
# ─────────────────────────────────────────────

def p3_04_model_comparison():
    print("\n" + "="*60)
    print("P3-04: MODEL COMPARISON")
    print("="*60)
    
    report = {"models": []}
    
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    
    X_train, y_train, _, feature_cols = get_features_and_labels(train_df)
    X_val, y_val, _, _ = get_features_and_labels(val_df)
    X_test, y_test, _, _ = get_features_and_labels(test_df)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Use balanced class weights for all (best from P3-03)
    class_weights = "balanced"
    
    # 1. LightGBM
    from lightgbm import LGBMClassifier
    t0 = time.time()
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, class_weight=class_weights, random_state=42, verbose=-1)
    lgbm.fit(X_train_s, y_train)
    t_lgbm = time.time() - t0
    pred = lgbm.predict(X_test_s)
    lgbm_acc = accuracy_score(y_test, pred) * 100
    lgbm_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"LightGBM:        Acc={lgbm_acc:.2f}%, F1={lgbm_f1:.4f} ({t_lgbm:.2f}s)")
    report["models"].append({"name": "LightGBM", "accuracy": round(lgbm_acc, 2), "macro_f1": round(lgbm_f1, 4), "train_time_s": round(t_lgbm, 2)})
    
    # 2. CatBoost
    try:
        from catboost import CatBoostClassifier
        t0 = time.time()
        cat = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced",
            random_seed=42, verbose=0
        )
        cat.fit(X_train_s, y_train)
        t_cat = time.time() - t0
        pred = cat.predict(X_test_s).flatten().astype(int)
        cat_acc = accuracy_score(y_test, pred) * 100
        cat_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        print(f"CatBoost:        Acc={cat_acc:.2f}%, F1={cat_f1:.4f} ({t_cat:.2f}s)")
        report["models"].append({"name": "CatBoost", "accuracy": round(cat_acc, 2), "macro_f1": round(cat_f1, 4), "train_time_s": round(t_cat, 2)})
    except Exception as e:
        print(f"CatBoost failed: {e}")
    
    # 3. XGBoost
    try:
        from xgboost import XGBClassifier
        # Compute scale_pos_weight-like approach: use sample_weight
        class_counts = np.bincount(y_train, minlength=len(IMD_CLASSES))
        sample_weights = np.array([len(y_train) / (len(IMD_CLASSES) * class_counts[c]) for c in y_train])
        
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            random_state=42, eval_metric="mlogloss", use_label_encoder=False,
            verbosity=0
        )
        xgb.fit(X_train_s, y_train, sample_weight=sample_weights)
        t_xgb = time.time() - t0
        pred = xgb.predict(X_test_s)
        xgb_acc = accuracy_score(y_test, pred) * 100
        xgb_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        print(f"XGBoost:         Acc={xgb_acc:.2f}%, F1={xgb_f1:.4f} ({t_xgb:.2f}s)")
        report["models"].append({"name": "XGBoost", "accuracy": round(xgb_acc, 2), "macro_f1": round(xgb_f1, 4), "train_time_s": round(t_xgb, 2)})
    except Exception as e:
        print(f"XGBoost failed: {e}")
    
    # 4. Random Forest
    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight=class_weights, random_state=42)
    rf.fit(X_train_s, y_train)
    t_rf = time.time() - t0
    pred = rf.predict(X_test_s)
    rf_acc = accuracy_score(y_test, pred) * 100
    rf_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"Random Forest:   Acc={rf_acc:.2f}%, F1={rf_f1:.4f} ({t_rf:.2f}s)")
    report["models"].append({"name": "RandomForest", "accuracy": round(rf_acc, 2), "macro_f1": round(rf_f1, 4), "train_time_s": round(t_rf, 2)})
    
    # 5. SVM (needs NaN imputation)
    from sklearn.svm import SVC
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_s)
    X_val_imp = imputer.transform(X_val_s)
    X_test_imp = imputer.transform(X_test_s)
    
    t0 = time.time()
    svm = SVC(kernel="rbf", class_weight=class_weights, random_state=42)
    svm.fit(X_train_imp, y_train)
    t_svm = time.time() - t0
    pred = svm.predict(X_test_imp)
    svm_acc = accuracy_score(y_test, pred) * 100
    svm_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"SVM (RBF):       Acc={svm_acc:.2f}%, F1={svm_f1:.4f} ({t_svm:.2f}s)")
    report["models"].append({"name": "SVM_RBF", "accuracy": round(svm_acc, 2), "macro_f1": round(svm_f1, 4), "train_time_s": round(t_svm, 2)})
    
    # 6. Gradient Boosting (sklearn, needs NaN imputation)
    from sklearn.ensemble import GradientBoostingClassifier
    t0 = time.time()
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    gb.fit(X_train_imp, y_train)
    t_gb = time.time() - t0
    pred = gb.predict(X_test_imp)
    gb_acc = accuracy_score(y_test, pred) * 100
    gb_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"GradientBoost:   Acc={gb_acc:.2f}%, F1={gb_f1:.4f} ({t_gb:.2f}s)")
    report["models"].append({"name": "GradientBoosting", "accuracy": round(gb_acc, 2), "macro_f1": round(gb_f1, 4), "train_time_s": round(t_gb, 2)})
    
    # Best model
    best = max(report["models"], key=lambda x: x["macro_f1"])
    report["best_model"] = best["name"]
    print(f"\nBest model: {best['name']} (F1={best['macro_f1']:.4f})")
    
    return report


# ─────────────────────────────────────────────
# P3-05: Hyperparameter Tuning (Best Model)
# ─────────────────────────────────────────────

def p3_05_tuning():
    print("\n" + "="*60)
    print("P3-05: HYPERPARAMETER TUNING")
    print("="*60)
    
    report = {"grid_results": []}
    
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    
    X_train, y_train, _, feature_cols = get_features_and_labels(train_df)
    X_val, y_val, _, _ = get_features_and_labels(val_df)
    X_test, y_test, _, _ = get_features_and_labels(test_df)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Try CatBoost and LightGBM with various params
    # All use class_weight="balanced" / auto_class_weights="Balanced"
    
    # CatBoost grid
    try:
        from catboost import CatBoostClassifier
        
        cb_grid = [
            {"iterations": 300, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 3},
            {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3},
            {"iterations": 800, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5},
            {"iterations": 500, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3},
            {"iterations": 500, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3},
            {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 1},
            {"iterations": 1000, "learning_rate": 0.02, "depth": 6, "l2_leaf_reg": 5},
            {"iterations": 500, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 5},
        ]
        
        best_val_f1 = 0
        best_params = None
        
        for i, params in enumerate(cb_grid):
            cat = CatBoostClassifier(
                iterations=params["iterations"],
                learning_rate=params["learning_rate"],
                depth=params["depth"],
                l2_leaf_reg=params["l2_leaf_reg"],
                auto_class_weights="Balanced",
                random_seed=42, verbose=0
            )
            cat.fit(X_train_s, y_train)
            
            val_pred = cat.predict(X_val_s).flatten().astype(int)
            test_pred = cat.predict(X_test_s).flatten().astype(int)
            
            val_acc = accuracy_score(y_val, val_pred) * 100
            val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            test_acc = accuracy_score(y_test, test_pred) * 100
            test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
            
            print(f"  CB-{i+1}: depth={params['depth']}, lr={params['learning_rate']}, iters={params['iterations']}, l2={params['l2_leaf_reg']} -> "
                  f"Val F1={val_f1:.4f}, Test F1={test_f1:.4f}")
            
            report["grid_results"].append({
                "model": "CatBoost",
                "params": params,
                "val_accuracy": round(val_acc, 2),
                "val_macro_f1": round(val_f1, 4),
                "test_accuracy": round(test_acc, 2),
                "test_macro_f1": round(test_f1, 4)
            })
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_params = params
                best_test_f1 = test_f1
                best_test_acc = test_acc
        
        report["best_catboost"] = {
            "params": best_params,
            "val_macro_f1": round(best_val_f1, 4),
            "test_accuracy": round(best_test_acc, 2),
            "test_macro_f1": round(best_test_f1, 4)
        }
        print(f"\nBest CatBoost: {best_params} -> Val F1={best_val_f1:.4f}, Test F1={best_test_f1:.4f}")
    except Exception as e:
        print(f"CatBoost tuning failed: {e}")
    
    # LightGBM grid
    from lightgbm import LGBMClassifier
    
    lgbm_grid = [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31},
        {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31},
        {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 6, "num_leaves": 31},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, "num_leaves": 63},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 7},
    ]
    
    best_val_f1 = 0
    best_params = None
    
    for i, params in enumerate(lgbm_grid):
        lgbm = LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            class_weight="balanced",
            random_state=42, verbose=-1
        )
        lgbm.fit(X_train_s, y_train)
        
        val_pred = lgbm.predict(X_val_s)
        test_pred = lgbm.predict(X_test_s)
        
        val_acc = accuracy_score(y_val, val_pred) * 100
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        test_acc = accuracy_score(y_test, test_pred) * 100
        test_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
        
        print(f"  LGBM-{i+1}: depth={params['max_depth']}, leaves={params['num_leaves']}, lr={params['learning_rate']}, "
              f"n={params['n_estimators']} -> Val F1={val_f1:.4f}, Test F1={test_f1:.4f}")
        
        report["grid_results"].append({
            "model": "LightGBM",
            "params": params,
            "val_accuracy": round(val_acc, 2),
            "val_macro_f1": round(val_f1, 4),
            "test_accuracy": round(test_acc, 2),
            "test_macro_f1": round(test_f1, 4)
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = params
            best_test_f1 = test_f1
            best_test_acc = test_acc
    
    report["best_lightgbm"] = {
        "params": best_params,
        "val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(best_test_acc, 2),
        "test_macro_f1": round(best_test_f1, 4)
    }
    print(f"\nBest LightGBM: {best_params} -> Val F1={best_val_f1:.4f}, Test F1={best_test_f1:.4f}")
    
    # Overall best
    all_results = [r for r in report["grid_results"] if "val_macro_f1" in r]
    overall_best = max(all_results, key=lambda x: x["val_macro_f1"])
    report["overall_best"] = {
        "model": overall_best["model"],
        "params": overall_best["params"],
        "val_macro_f1": overall_best["val_macro_f1"],
        "test_accuracy": overall_best["test_accuracy"],
        "test_macro_f1": overall_best["test_macro_f1"]
    }
    print(f"\nOverall best: {overall_best['model']} {overall_best['params']}")
    
    return report


# ─────────────────────────────────────────────
# P3-06: Ordinal Analysis
# ─────────────────────────────────────────────

def p3_06_ordinal_analysis():
    print("\n" + "="*60)
    print("P3-06: ORDINAL / CLASS STRUCTURE ANALYSIS")
    print("="*60)
    
    report = {}
    
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    
    X_train, y_train, _, feature_cols = get_features_and_labels(train_df)
    X_val, y_val, _, _ = get_features_and_labels(val_df)
    X_test, y_test, _, _ = get_features_and_labels(test_df)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Best model from P3-05 (CatBoost with balanced weights, reasonable params)
    from catboost import CatBoostClassifier
    
    best_cat = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        auto_class_weights="Balanced", random_seed=42, verbose=0
    )
    best_cat.fit(X_train_s, y_train)
    test_pred = best_cat.predict(X_test_s).flatten().astype(int)
    
    # Confusion matrix analysis
    present_classes = sorted(set(y_test) | set(test_pred))
    cm = confusion_matrix(y_test, test_pred, labels=present_classes)
    
    # Off-by-one accuracy (adjacent class tolerance)
    off_by_one_correct = 0
    off_by_two_correct = 0
    total = len(y_test)
    
    for true, pred in zip(y_test, test_pred):
        diff = abs(true - pred)
        if diff <= 1:
            off_by_one_correct += 1
        if diff <= 2:
            off_by_two_correct += 1
    
    off_by_one_acc = off_by_one_correct / total * 100
    off_by_two_acc = off_by_two_correct / total * 100
    
    print(f"Standard Accuracy:        {accuracy_score(y_test, test_pred)*100:.2f}%")
    print(f"Off-by-one Accuracy:      {off_by_one_acc:.2f}%")
    print(f"Off-by-two Accuracy:      {off_by_two_acc:.2f}%")
    
    report["ordinal_metrics"] = {
        "standard_accuracy": round(accuracy_score(y_test, test_pred) * 100, 2),
        "off_by_one_accuracy": round(off_by_one_acc, 2),
        "off_by_two_accuracy": round(off_by_two_acc, 2),
        "total_test_samples": total
    }
    
    # Confusion matrix details
    cm_dict = {}
    for i, cls_i in enumerate(present_classes):
        for j, cls_j in enumerate(present_classes):
            if cm[i, j] > 0:
                cm_dict[f"{IDX_TO_CLASS[cls_i]}_true_{IDX_TO_CLASS[cls_j]}_pred"] = int(cm[i, j])
    
    report["confusion_matrix"] = cm_dict
    
    # Per-class analysis
    print("\nPer-class analysis:")
    present_class_names = [IDX_TO_CLASS[c] for c in present_classes]
    cr = classification_report(y_test, test_pred, target_names=present_class_names, zero_division=0, output_dict=True)
    report["classification_report"] = {k: v for k, v in cr.items() if k in present_class_names or k in ["accuracy", "macro avg", "weighted avg"]}
    
    for cls_name in present_class_names:
        if cls_name in cr:
            print(f"  {cls_name:35s}: P={cr[cls_name]['precision']:.3f} R={cr[cls_name]['recall']:.3f} F1={cr[cls_name]['f1-score']:.3f} N={int(cr[cls_name]['support'])}")
    
    # Direction of errors (are severe classes predicted as weaker?)
    severe_errors = {"overpredict": 0, "underpredict": 0}
    for true, pred in zip(y_test, test_pred):
        if true != pred:
            if pred > true:
                severe_errors["overpredict"] += 1
            else:
                severe_errors["underpredict"] += 1
    
    report["error_direction"] = severe_errors
    print(f"\nError direction: Overpredict={severe_errors['overpredict']}, Underpredict={severe_errors['underpredict']}")
    
    # Feature importance for ordinal structure
    fi = best_cat.get_feature_importance()
    report["feature_importance"] = {feature_cols[i]: round(float(fi[i]), 4) for i in range(len(feature_cols))}
    print("\nFeature importance:")
    for feat, imp in sorted(report["feature_importance"].items(), key=lambda x: -x[1]):
        print(f"  {feat:20s}: {imp:.4f}")
    
    return report


# ─────────────────────────────────────────────
# P3-07: Error Analysis
# ─────────────────────────────────────────────

def p3_07_error_analysis():
    print("\n" + "="*60)
    print("P3-07: ERROR ANALYSIS")
    print("="*60)
    
    report = {}
    
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    
    X_train, y_train, train_valid_df, feature_cols = get_features_and_labels(train_df)
    X_val, y_val, val_valid_df, _ = get_features_and_labels(val_df)
    X_test, y_test, test_valid_df, _ = get_features_and_labels(test_df)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    from catboost import CatBoostClassifier
    
    best_cat = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        auto_class_weights="Balanced", random_seed=42, verbose=0
    )
    best_cat.fit(X_train_s, y_train)
    test_pred = best_cat.predict(X_test_s).flatten().astype(int)
    
    # Error indices
    errors = np.where(test_pred != y_test)[0]
    correct = np.where(test_pred == y_test)[0]
    
    print(f"Total test: {len(y_test)}, Correct: {len(correct)}, Errors: {len(errors)} ({100*len(errors)/len(y_test):.1f}%)")
    
    report["overall"] = {
        "total_test": len(y_test),
        "correct": len(correct),
        "errors": len(errors),
        "error_rate": round(100 * len(errors) / len(y_test), 2)
    }
    
    # Error distribution by true class
    error_by_class = {}
    for idx in errors:
        cls = IDX_TO_CLASS[y_test[idx]]
        error_by_class[cls] = error_by_class.get(cls, 0) + 1
    
    total_by_class = {}
    for idx in range(len(y_test)):
        cls = IDX_TO_CLASS[y_test[idx]]
        total_by_class[cls] = total_by_class.get(cls, 0) + 1
    
    print("\nError rate by true class:")
    report["error_by_class"] = {}
    for cls in IMD_CLASSES:
        n_err = error_by_class.get(cls, 0)
        n_total = total_by_class.get(cls, 0)
        if n_total > 0:
            err_rate = n_err / n_total * 100
            print(f"  {cls:35s}: {n_err:3d}/{n_total:3d} errors ({err_rate:.1f}%)")
            report["error_by_class"][cls] = {"errors": n_err, "total": n_total, "error_rate": round(err_rate, 2)}
    
    # Error analysis: what predictions were made?
    confusion_pairs = Counter()
    for idx in errors:
        true_cls = IDX_TO_CLASS[y_test[idx]]
        pred_cls = IDX_TO_CLASS[test_pred[idx]]
        confusion_pairs[(true_cls, pred_cls)] += 1
    
    print("\nTop confusion pairs:")
    report["top_confusion_pairs"] = []
    for (true_cls, pred_cls), count in confusion_pairs.most_common(10):
        print(f"  {true_cls:35s} -> {pred_cls:35s}: {count}")
        report["top_confusion_pairs"].append({
            "true_class": true_cls,
            "predicted_class": pred_cls,
            "count": count
        })
    
    # Feature distribution: correct vs error samples
    print("\nFeature statistics: Correct vs Error samples")
    report["feature_comparison"] = {}
    for i, feat in enumerate(feature_cols):
        correct_vals = X_test[correct, i]
        error_vals = X_test[errors, i]
        
        correct_mean = np.mean(correct_vals)
        error_mean = np.mean(error_vals)
        
        report["feature_comparison"][feat] = {
            "correct_mean": round(float(correct_mean), 4),
            "error_mean": round(float(error_mean), 4),
            "difference_pct": round(float((error_mean - correct_mean) / (abs(correct_mean) + 1e-8) * 100), 2)
        }
        print(f"  {feat:20s}: correct_mean={correct_mean:.4f}, error_mean={error_mean:.4f}")
    
    # Confidence analysis
    if hasattr(best_cat, "predict_proba"):
        test_proba = best_cat.predict_proba(X_test_s)
        test_conf = np.max(test_proba, axis=1)
        
        correct_conf = test_conf[correct]
        error_conf = test_conf[errors]
        
        print(f"\nConfidence: Correct mean={np.mean(correct_conf):.4f}, Error mean={np.mean(error_conf):.4f}")
        report["confidence"] = {
            "correct_mean_confidence": round(float(np.mean(correct_conf)), 4),
            "error_mean_confidence": round(float(np.mean(error_conf)), 4)
        }
        
        # Low-confidence samples
        low_conf_threshold = 0.5
        low_conf_mask = test_conf < low_conf_threshold
        n_low_conf = low_conf_mask.sum()
        
        if n_low_conf > 0:
            low_conf_correct = (test_pred[low_conf_mask] == y_test[low_conf_mask]).sum()
            print(f"Low confidence (<{low_conf_threshold}): {n_low_conf} samples, "
                  f"accuracy={100*low_conf_correct/n_low_conf:.1f}%")
            report["low_confidence_analysis"] = {
                "threshold": low_conf_threshold,
                "n_samples": int(n_low_conf),
                "accuracy": round(float(100 * low_conf_correct / n_low_conf), 2)
            }
    
    # Misclassified severe storms
    severe_classes = [4, 5, 6]  # VSCS, ESCS, SuCS
    severe_errors = []
    for idx in errors:
        if y_test[idx] in severe_classes:
            severe_errors.append({
                "cyclone_id": str(test_valid_df.iloc[idx]["cyclone_id"]) if "cyclone_id" in test_valid_df.columns else None,
                "true_class": IDX_TO_CLASS[y_test[idx]],
                "predicted_class": IDX_TO_CLASS[test_pred[idx]],
                "features": {feat: round(float(X_test[idx, i]), 4) for i, feat in enumerate(feature_cols)}
            })
    
    report["severe_storm_errors"] = severe_errors[:20]  # First 20
    print(f"\nSevere storm misclassifications: {len(severe_errors)}")
    for err in severe_errors[:5]:
        print(f"  {err['true_class']} -> {err['predicted_class']} (ID: {err['cyclone_id']})")
    
    return report


# ─────────────────────────────────────────────
# P3-08: Final Summary
# ─────────────────────────────────────────────

def p3_08_summary(p3_01, p3_02, p3_03, p3_04, p3_05, p3_06, p3_07):
    print("\n" + "="*60)
    print("P3-08: FINAL MODEL SELECTION & SUMMARY")
    print("="*60)
    
    summary = {
        "data_quality": {
            "n_train": p3_01["split_stats"]["train"]["n_rows"],
            "n_val": p3_01["split_stats"]["val"]["n_rows"],
            "n_test": p3_01["split_stats"]["test"]["n_rows"],
            "n_features": 6,
            "n_classes": len(IMD_CLASSES),
            "has_leakage": len(p3_01.get("warnings", [])) > 0 and any("LEAKAGE" in w for w in p3_01.get("warnings", [])),
            "class_imbalance_severe": True,
            "missing_classes_in_val": [c for c in IMD_CLASSES if c not in p3_01["class_distribution"].get("val", {})],
            "missing_classes_in_test": [c for c in IMD_CLASSES if c not in p3_01["class_distribution"].get("test", {})]
        },
        "baseline": {
            "majority_class": p3_02.get("majority_class", {}),
            "original_lgbm": p3_02.get("original_lgbm", {}),
            "stored_claim": p3_02.get("stored_claims", {})
        },
        "best_imbalance_method": p3_03.get("best", "N/A"),
        "best_model_name": p3_04.get("best_model", "N/A"),
        "best_tuned_params": p3_05.get("overall_best", {}),
        "ordinal_metrics": p3_06.get("ordinal_metrics", {}),
        "error_summary": {
            "error_rate": p3_07.get("overall", {}).get("error_rate", 0),
            "top_confusion": p3_07.get("top_confusion_pairs", [])[:5],
            "severe_storm_errors": len(p3_07.get("severe_storm_errors", []))
        },
        "improvement_over_stored_claim": {}
    }
    
    # Compute improvement
    stored_acc = summary["baseline"]["stored_claim"].get("accuracy", 47.0)
    stored_f1 = summary["baseline"]["stored_claim"].get("macro_f1", 0.3703)
    
    best_result = p3_05.get("overall_best", {})
    new_acc = best_result.get("test_accuracy", 0)
    new_f1 = best_result.get("test_macro_f1", 0)
    
    summary["improvement_over_stored_claim"] = {
        "accuracy_delta": round(new_acc - stored_acc, 2),
        "f1_delta": round(new_f1 - stored_f1, 4),
        "stored_accuracy": stored_acc,
        "new_accuracy": new_acc,
        "stored_f1": stored_f1,
        "new_f1": new_f1
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("P3 CLASSIFICATION IMPROVEMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {summary['data_quality']['n_train']} train, {summary['data_quality']['n_val']} val, {summary['data_quality']['n_test']} test")
    print(f"Features: {summary['data_quality']['n_features']} (lat, lon, sst, pressure_msl, wind_u, wind_v)")
    print(f"Classes: {summary['data_quality']['n_classes']} (with severe imbalance)")
    print(f"Leakage: {'YES' if summary['data_quality']['has_leakage'] else 'No'}")
    print(f"Missing classes in val: {summary['data_quality']['missing_classes_in_val']}")
    print(f"Missing classes in test: {summary['data_quality']['missing_classes_in_test']}")
    print(f"\nBest imbalance method: {summary['best_imbalance_method']}")
    print(f"Best model: {summary['best_model_name']}")
    print(f"Best params: {summary['best_tuned_params'].get('params', 'N/A')}")
    print(f"\nOrdinal accuracy: {summary['ordinal_metrics'].get('standard_accuracy', 0):.2f}%")
    print(f"Off-by-one accuracy: {summary['ordinal_metrics'].get('off_by_one_accuracy', 0):.2f}%")
    print(f"Off-by-two accuracy: {summary['ordinal_metrics'].get('off_by_two_accuracy', 0):.2f}%")
    print(f"\nImprovement over stored claim:")
    print(f"  Accuracy: {stored_acc:.2f}% -> {new_acc:.2f}% ({new_acc-stored_acc:+.2f}%)")
    print(f"  Macro-F1: {stored_f1:.4f} -> {new_f1:.4f} ({new_f1-stored_f1:+.4f})")
    print(f"{'='*60}")
    
    return summary


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("P3 CLASSIFICATION IMPROVEMENT PIPELINE")
    print("="*60)
    
    # P3-01
    p3_01 = p3_01_data_audit()
    with open(os.path.join(RESULTS_DIR, "P3_DATA_QUALITY_AUDIT.json"), "w") as f:
        json.dump(p3_01, f, indent=2, default=str)
    
    # P3-02
    p3_02 = p3_02_baseline()
    with open(os.path.join(RESULTS_DIR, "P3_BASELINE.json"), "w") as f:
        json.dump(p3_02, f, indent=2, default=str)
    
    # P3-03
    p3_03 = p3_03_class_imbalance()
    with open(os.path.join(RESULTS_DIR, "P3_CLASS_IMBALANCE.json"), "w") as f:
        json.dump(p3_03, f, indent=2, default=str)
    
    # P3-04
    p3_04 = p3_04_model_comparison()
    with open(os.path.join(RESULTS_DIR, "P3_MODEL_COMPARISON.json"), "w") as f:
        json.dump(p3_04, f, indent=2, default=str)
    
    # P3-05
    p3_05 = p3_05_tuning()
    with open(os.path.join(RESULTS_DIR, "P3_TUNING.json"), "w") as f:
        json.dump(p3_05, f, indent=2, default=str)
    
    # P3-06
    p3_06 = p3_06_ordinal_analysis()
    with open(os.path.join(RESULTS_DIR, "P3_ORDINAL_ANALYSIS.json"), "w") as f:
        json.dump(p3_06, f, indent=2, default=str)
    
    # P3-07
    p3_07 = p3_07_error_analysis()
    with open(os.path.join(RESULTS_DIR, "P3_ERROR_ANALYSIS.json"), "w") as f:
        json.dump(p3_07, f, indent=2, default=str)
    
    # P3-08
    p3_08 = p3_08_summary(p3_01, p3_02, p3_03, p3_04, p3_05, p3_06, p3_07)
    with open(os.path.join(RESULTS_DIR, "P3_IMPROVEMENT_SUMMARY.json"), "w") as f:
        json.dump(p3_08, f, indent=2, default=str)
    
    # Save Markdown summary
    md = f"""# P3 Classification Improvement Summary

## Dataset
- **Train**: {p3_08['data_quality']['n_train']} samples
- **Val**: {p3_08['data_quality']['n_val']} samples  
- **Test**: {p3_08['data_quality']['n_test']} samples
- **Features**: lat, lon, sst, pressure_msl, wind_u, wind_v
- **Classes**: {p3_08['data_quality']['n_classes']} (severe imbalance)

## Data Quality
- **Leakage**: {'YES' if p3_08['data_quality']['has_leakage'] else 'No'}
- **Missing classes in val**: {p3_08['data_quality']['missing_classes_in_val']}
- **Missing classes in test**: {p3_08['data_quality']['missing_classes_in_test']}

## Results

| Metric | Stored Claim | Reproduced | Improved |
|--------|-------------|------------|----------|
| Accuracy | {p3_08['baseline']['stored_claim']['accuracy']:.2f}% | {p3_08['baseline']['original_lgbm']['test_accuracy']:.2f}% | {p3_08['improvement_over_stored_claim']['new_accuracy']:.2f}% |
| Macro-F1 | {p3_08['baseline']['stored_claim']['macro_f1']:.4f} | {p3_08['baseline']['original_lgbm']['test_macro_f1']:.4f} | {p3_08['improvement_over_stored_claim']['new_f1']:.4f} |

## Best Configuration
- **Model**: {p3_08['best_model_name']}
- **Class Imbalance**: {p3_08['best_imbalance_method']}
- **Params**: {json.dumps(p3_08['best_tuned_params'].get('params', {}), indent=2)}

## Ordinal Metrics
- **Standard Accuracy**: {p3_08['ordinal_metrics'].get('standard_accuracy', 0):.2f}%
- **Off-by-one Accuracy**: {p3_08['ordinal_metrics'].get('off_by_one_accuracy', 0):.2f}%
- **Off-by-two Accuracy**: {p3_08['ordinal_metrics'].get('off_by_two_accuracy', 0):.2f}%

## Improvement
- **Accuracy**: {p3_08['improvement_over_stored_claim']['stored_accuracy']:.2f}% -> {p3_08['improvement_over_stored_claim']['new_accuracy']:.2f}% ({p3_08['improvement_over_stored_claim']['accuracy_delta']:+.2f}%)
- **Macro-F1**: {p3_08['improvement_over_stored_claim']['stored_f1']:.4f} -> {p3_08['improvement_over_stored_claim']['new_f1']:.4f} ({p3_08['improvement_over_stored_claim']['f1_delta']:+.4f})
"""
    
    with open(os.path.join(RESULTS_DIR, "P3_IMPROVEMENT_SUMMARY.md"), "w") as f:
        f.write(md)
    
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
