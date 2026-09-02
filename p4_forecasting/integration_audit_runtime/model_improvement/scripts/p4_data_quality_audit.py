"""P4 data-quality and leakage audit (model_improvement).

Runs on the authoritative P4 feature dataset + raw canonical data that are on
disk (NOT the zip). Establishes the frozen baseline and detects leakage /
duplicates / NaN / outliers / normalization provenance.

Outputs: results/P4_DATA_QUALITY_AUDIT.json (evidence) — no source modified.
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # .../integration_audit_runtime
P4 = ROOT / ".." / "phase4"                          # p4_forecasting/phase4
PROJECT = P4.parent.parent

FEAT = P4 / "results" / "feature_dataset"
RAW = P4.parent / "canonical_chrono"
NORM = P4 / "results" / "normalization_stats.json"

HORIZONS = [6, 12, 24]
FEAT_NAMES = ["lat","lon","wind_speed","pressure","sst","wind_u","wind_v",
              "delta_lat","delta_lon","movement_speed","movement_direction",
              "wind_change","pressure_change","sst_change",
              "environmental_wind_speed","environmental_wind_direction"]
TARGET_NAMES = ["lat","lon","wind_speed"]

def load_split(split):
    d = np.load(FEAT / f"{split}.npz", allow_pickle=True)
    X = d["X"]; Y = d["Y"]
    meta = pd.read_csv(FEAT / f"{split}_metadata.csv")
    return X, Y, meta

def load_raw(split):
    d = np.load(RAW / f"{split}.npz", allow_pickle=True)
    return d["X"], d["Y"]

report = {"feature_dataset": FEAT.as_posix(), "raw_canonical": RAW.as_posix(),
          "generated": "P4 data-quality & leakage audit (model_improvement)"}

splits = {}
all_ids = {}
for split in ["train", "val", "test"]:
    X, Y, meta = load_split(split)
    ids = set(meta["cyclone_id"].astype(str))
    all_ids.setdefault(split, ids)
    tz = pd.to_datetime(meta["t_zero"], utc=True) if "t_zero" in meta else None
    splits[split] = {
        "rows": int(X.shape[0]),
        "unique_cyclones": int(meta["cyclone_id"].nunique()),
        "time_span": {"min": str(tz.min()) if tz is not None else None,
                      "max": str(tz.max()) if tz is not None else None},
        "feature_shape": list(X.shape),
        "target_shape": list(Y.shape),
        "nan_features": int(np.isnan(X).sum()),
        "inf_features": int(np.isinf(X).sum()),
        "nan_targets": int(np.isnan(Y).sum()),
        "dup_feature_rows": int((np.array([tuple(np.round(r,4)) for r in X.reshape(X.shape[0],-1)]) if X.shape[0] else np.array([])).shape[0] -
                                len(set(map(tuple, np.round(X.reshape(X.shape[0],-1),4))))),
        "cyclone_id_examples": sorted(ids)[:5],
    }

# Cyclone identity overlap across splits
ov = {}
pairs = [("train","val"),("train","test"),("val","test")]
for a,b in pairs:
    inter = all_ids[a] & all_ids[b]
    ov[f"{a}_vs_{b}"] = {"overlap_count": len(inter),
                         "overlap_ids": sorted(inter)[:20]}
report["cyclone_overlap"] = ov

# Chronological ordering: min start of each split
starts = {s: splits[s]["time_span"]["min"] for s in splits}
report["chronological_split_starts"] = starts
report["chronological_ok"] = bool(
    splits["train"]["time_span"]["max"] and
    starts["test"] and starts["val"] and
    str(starts["train"]).lower().strip() and
    str(starts["val"]) <= str(starts["test"])) if starts["val"] and starts["test"] else None

# To be robust, do explicit storm-start ordering from metadata
storm_starts = {}
for split in ["train","val","test"]:
    _, _, meta = load_split(split)
    g = meta.groupby("cyclone_id")["t_zero"].min()
    storm_starts[split] = {str(k): str(v) for k, v in g.items()}
report["storm_count_and_min_start"] = {s: sorted(storm_starts[s].items(), key=lambda kv: kv[1])[:3] for s in storm_starts}

def min_start(split):
    vals = [pd.to_datetime(v, utc=True) for v in storm_starts[split].values()]
    return min(vals) if vals else None
ms = {s: min_start(s) for s in ["train","val","test"]}
report["chronological_storm_ordering"] = {
    "val_latest_start_ge_train_latest": (ms["val"] is None or ms["val"] > ms["train"]) if ms["train"] else None,
    "test_latest_start_ge_val_latest": (ms["test"] is None or ms["test"] > ms["val"]) if ms["val"] else None,
}

# Target leakage: ensure no input feature column equals any target at t=0 (raw)
# Check that targets are strictly future: compare t_zero step raw lat/lon vs target1 lat
for split in ["train","val","test"]:
    X, Y, meta = load_split(split)
    # step index 4 is t=0 (latest history)
    # raw lat/lon are columns 0,1
    t0_lat = X[:, 4, 0]; t0_lon = X[:, 4, 1]
    tgt1_lat = Y[:, 0, 0]; tgt1_lon = Y[:, 0, 1]
    same_pairs = int(((np.abs(t0_lat - tgt1_lat) < 1e-6) & (np.abs(t0_lon - tgt1_lon) < 1e-6)).sum())
    splits[split]["future_target_check_t0_eq_target1_count"] = same_pairs
    # any input column identical to any target column across all samples
    Xf = X.reshape(X.shape[0], -1)
    ident = []
    for c in range(Xf.shape[1]):
        for t in range(3):
            if np.allclose(Xf[:, c], Y[:, 0, t], atol=1e-6):
                ident.append((c, t))
    splits[split]["input_col_identical_to_target"] = ident

report["splits"] = splits

# Feature-level stats (outliers) trained on all splits for audit visibility
feat_stats = {}
for fi, name in enumerate(FEAT_NAMES):
    col = np.concatenate([load_split(s)[0][:, :, fi] for s in splits])
    feat_stats[name] = {"min": float(np.nanmin(col)), "max": float(np.nanmax(col)),
                        "mean": float(np.nanmean(col)), "std": float(np.nanstd(col)),
                        "p1": float(np.nanpercentile(col,1)), "p99": float(np.nanpercentile(col,99))}
report["feature_stats"] = feat_stats

# Train-only normalization provenance
try:
    norm = json.load(open(NORM, encoding="utf-8"))
    report["normalization"] = {"computed_from": (norm.get("computed_from") or {}),
                                "n_features": len(norm.get("features", {}))}
except Exception as e:
    report["normalization"] = {"error": str(e)}

# Per-cyclone sample counts (largest storms)
_, _, meta = load_split("test")
report["test_top_cyclones"] = meta.groupby("cyclone_id").size().sort_values(ascending=False).head(5).to_dict()

out = ROOT / "model_improvement" / "results" / "P4_DATA_QUALITY_AUDIT.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, indent=2, default=str)[:4000])
print("\nWROTE", out)
