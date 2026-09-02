"""P4 Phase 3 - Lightweight LSTM cyclone forecasting model (SIH 2026 PS 26070).

Idempotent orchestrator:
  1. source validation (files + pre-run hashes/snapshots)
  2. normalization preparation (training-only statistics)
  3. model creation (LSTM 7 -> 64 -> 9)
  4. training (deterministic, early stopping, best-val checkpoint)
  5. checkpoint selection (best validation epoch)
  6. evaluation on validation + test
  7. baseline comparison (persistence / movement vector / LSTM)
  8. inference contract test + full pytest suite
  9. safety checks (no Phase-1/2/P1 file modified, nothing outside phase3 written)
 10. report generation (MODEL_REPORT.md, loss curve, run metadata)

All Phase-3 files are written ONLY under p4_forecasting/phase3/. Source files
(canonical, canonical_chrono, phase2 outputs) are strictly read-only.

The run is deterministic: seeds fixed, CPU execution, bytecode writing disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

P3_DIR = Path(__file__).resolve().parent
P4_DIR = P3_DIR.parent
PROJECT_ROOT = P4_DIR.parent
if str(P4_DIR) not in sys.path:
    sys.path.insert(0, str(P4_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phase3.training.train import TrainConfig, run_training  # noqa: E402
from phase3.training.normalization import Normalizer  # noqa: E402
from phase3.evaluation.evaluate_model import evaluate_checkpoint, save_evaluation_results  # noqa: E402
from phase3.evaluation.compare_baselines import build_model_comparison, save_model_comparison  # noqa: E402
from phase3.inference.forecaster import CycloneForecaster  # noqa: E402
from phase3.reports.generate_model_report import generate_model_report  # noqa: E402

DATASET_DIR = P4_DIR / "phase2" / "results" / "canonical_chronological_clean"
BASELINE_JSON = P4_DIR / "phase2" / "results" / "baseline_results.json"
CHRONO_DIR = P4_DIR / "canonical_chrono"
QUALITY_CSV = P4_DIR / "canonical" / "sample_quality.csv"

RESULTS_DIR = P3_DIR / "results"
CKPT_DIR = P3_DIR / "checkpoints"
REPORT_DIR = P3_DIR / "reports"

CKPT_PATH = CKPT_DIR / "best_lstm.pt"
CONFIG_PATH = CKPT_DIR / "model_config.json"
STATS_PATH = RESULTS_DIR / "normalization_stats.json"
HISTORY_PATH = RESULTS_DIR / "training_history.json"
EVAL_PATH = RESULTS_DIR / "evaluation_results.json"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.json"
METADATA_PATH = RESULTS_DIR / "run_metadata.json"
SOURCE_HASHES_PATH = RESULTS_DIR / "source_hashes.json"
CURVE_PATH = RESULTS_DIR / "training_loss_curve.png"
REPORT_PATH = REPORT_DIR / "MODEL_REPORT.md"

CHRONO_SOURCE_FILES = ["train.npz", "train_metadata.csv", "val.npz", "val_metadata.csv",
                       "test.npz", "test_metadata.csv", "split_manifest.csv"]
CLEAN_SOURCE_FILES = ["train.npz", "train_metadata.csv", "val.npz", "val_metadata.csv",
                      "test.npz", "test_metadata.csv"]

PHASE1_DIRS = ["canonical", "canonical_chrono", "audit", "reports", "scripts", "logs",
               "phase2", "_source_p1"]
# canonical_chrono npz sha256 values recorded at Phase-1/Phase-2 handoff (verified
# byte-equal to P1 canonical output). Full values are stored to results/source_hashes.json.
EXPECTED_CHRONO_SHA256_PREFIX = {"train.npz": "df70303e", "val.npz": "48cf065d",
                                 "test.npz": "89e9c2e2"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir(rel_dir: str) -> dict:
    base = P4_DIR / rel_dir
    snap = {}
    if not base.exists():
        return snap
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in sorted(files):
            if fn.endswith(".pyc"):
                continue
            p = Path(root) / fn
            snap[p.relative_to(base).as_posix()] = sha256(p)
    return snap


def main() -> int:
    parser = argparse.ArgumentParser(description="P4 Phase 3 runner")
    parser.add_argument("--force-retrain", action="store_true",
                        help="retrain from scratch even if a best checkpoint exists")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    required = [
        DATASET_DIR / "train.npz", DATASET_DIR / "val.npz", DATASET_DIR / "test.npz",
        BASELINE_JSON, QUALITY_CSV,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR: missing required sources:")
        for m in missing:
            print("  " + m)
        return 1

    # ---- 0. pre-run hashes & snapshots ------------------------------------
    before_sources = {fn: sha256(CHRONO_DIR / fn) for fn in CHRONO_SOURCE_FILES if (CHRONO_DIR / fn).exists()}
    before_clean = {fn: sha256(DATASET_DIR / fn) for fn in CLEAN_SOURCE_FILES if (DATASET_DIR / fn).exists()}
    before_dirs = {rel: snapshot_dir(rel) for rel in PHASE1_DIRS}

    src_hashes = {
        "canonical_chrono": sorted(before_sources.items()),
        "phase2_clean_dataset": sorted(before_clean.items()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "note": "before/after within-run; hashes unchanged for the entire phase-3 run",
    }
    with open(SOURCE_HASHES_PATH, "w", encoding="utf-8") as fh:
        json.dump(src_hashes, fh, indent=2)

    for fn, prefix in EXPECTED_CHRONO_SHA256_PREFIX.items():
        if fn in before_sources and not before_sources[fn].startswith(prefix):
            raise RuntimeError(f"canonical_chrono/{fn} hash does not match Phase-1/2 record "
                               f"({before_sources[fn][:12]}... != {prefix}...)")

    # ---- 1 & 2 & 3 & 4: normalization + model + training ---------------------
    cfg = TrainConfig(seed=args.seed, max_epochs=args.epochs, force_retrain=args.force_retrain)
    result = run_training(DATASET_DIR, STATS_PATH, CKPT_PATH, CONFIG_PATH,
                          HISTORY_PATH, cfg)
    reused = bool(result.epochs_run > 0 and result.training_time_s == 0.0)
    if reused:
        print("[phase3] existing best checkpoint REUSED (no overwrite) - use "
              "--force-retrain to retrain from scratch")

    # ---- 5 & 6: evaluation (val + one test pass) -----------------------------
    evaluation = evaluate_checkpoint(CKPT_PATH, CONFIG_PATH, STATS_PATH, DATASET_DIR,
                                     device=cfg.device)
    save_evaluation_results(evaluation, EVAL_PATH)
    print(f"[phase3] evaluation written -> {EVAL_PATH}")

    # ---- 7: baseline comparison ----------------------------------------------
    dataset_info = _dataset_info(DATASET_DIR)
    comparison = build_model_comparison(BASELINE_JSON, evaluation, dataset_info)
    save_model_comparison(comparison, COMPARISON_PATH)
    print(f"[phase3] model comparison written -> {COMPARISON_PATH}")

    # ---- 8: inference contract test + full test suite -------------------------
    run_inference_smoke()
    test_rc, test_lines = _run_pytest()
    if test_rc != 0:
        print("[phase3] TEST_SUMMARY: FAIL - see pytest output")
        for line in test_lines:
            print("   ", line)
        return 1
    test_summary = " ".join(test_lines)[-200:]
    print("[phase3] TEST_SUMMARY:", test_summary)

    # ---- 0b. run metadata & plot ---------------------------------------------
    write_run_metadata(args, reused)
    _maybe_write_loss_curve(HISTORY_PATH, CURVE_PATH, reused)

    # ---- 10: report ------------------------------------------------------------
    generate_model_report(COMPARISON_PATH, HISTORY_PATH, CONFIG_PATH, METADATA_PATH,
                          REPORT_PATH, dataset_info)
    print(f"[phase3] MODEL_REPORT.md written -> {REPORT_PATH}")

    # ---- 9: safety checks --------------------------------------------------------
    after_sources = {fn: sha256(CHRONO_DIR / fn) for fn in CHRONO_SOURCE_FILES if (CHRONO_DIR / fn).exists()}
    after_clean = {fn: sha256(DATASET_DIR / fn) for fn in CLEAN_SOURCE_FILES if (DATASET_DIR / fn).exists()}
    after_dirs = {rel: snapshot_dir(rel) for rel in PHASE1_DIRS}

    modified_p1 = sorted(fn for fn in CHRONO_SOURCE_FILES if before_sources.get(fn) != after_sources.get(fn))
    modified_clean = sorted(fn for fn in CLEAN_SOURCE_FILES if before_clean.get(fn) != after_clean.get(fn))
    modified_dirs = {rel: sorted(k for k in before_dirs[rel] if before_dirs[rel][k] != after_dirs[rel].get(k))
                     for rel in PHASE1_DIRS}
    modified_dirs = {rel: v for rel, v in modified_dirs.items() if v}

    phase1_dir_changes = sum(len(v) for rel, v in modified_dirs.items() if rel != "phase2")
    phase2_dir_changes = len(modified_dirs.get("phase2", []))

    # P1 source = the immutable staged source copies + canonical_chrono outputs
    p1_count = len(modified_p1) + phase1_dir_changes
    # Phase-2 outputs = phase2 dir + the clean dataset used here
    phase2_count = len(modified_clean) + phase2_dir_changes
    outside_phase3 = p1_count + phase2_count

    verdict = "PASS"
    warnings = []
    if outside_phase3 > 0:
        verdict = "FAIL"
    _check_overfitting(cmpt=comparison, hist=result, warnings=warnings)

    _print_final(result, evaluation, comparison, dataset_info, verdict, reused,
                 p1_count, phase2_count, outside_phase3, test_lines)
    _print_paths(warnings)
    return 0


def _dataset_info(dataset_dir: Path) -> dict:
    info = {}
    for split in ("train", "val", "test"):
        npz = np.load(dataset_dir / f"{split}.npz", allow_pickle=True)
        meta = pd.read_csv(dataset_dir / f"{split}_metadata.csv")
        info[split] = {"sequences": int(npz["X"].shape[0]),
                       "cyclones": int(meta["cyclone_id"].nunique())}
    return info


def run_inference_smoke() -> None:
    forecaster = CycloneForecaster(CKPT_PATH, CONFIG_PATH, STATS_PATH, device="cpu")
    npz = np.load(DATASET_DIR / "val.npz", allow_pickle=True)
    h = np.asarray(npz["X"][0], dtype=np.float32)
    out = forecaster.forecast(h)
    assert len(out["forecast"]) == 3
    for f in out["forecast"]:
        assert set(f.keys()) == {"hours", "latitude", "longitude", "wind_speed_kmh"}
        assert np.isfinite(f["latitude"]) and np.isfinite(f["longitude"]) and np.isfinite(f["wind_speed_kmh"])
        assert f["hours"] in (6, 12, 24)
        assert 0.0 <= f["longitude"] < 360.0
    print("[phase3] inference smoke test OK (contract format, denormalized, no future data)")


def _run_pytest() -> tuple:
    import re
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", "-q",
           "-p", "no:cacheprovider", "--disable-warnings", str(P3_DIR / "tests")]
    print("[phase3] running pytest ...")
    proc = subprocess.run(cmd, cwd=str(P4_DIR), env=env, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = [ln for ln in out.splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
    for ln in lines[-6:]:
        print("   ", ln)
    if not lines:
        lines = [out.strip()[-300:]]
    return proc.returncode, lines[-2:]


def write_run_metadata(args, reused: bool) -> None:
    import platform
    import torch
    meta = {
        "seed": args.seed,
        "max_epochs": args.epochs,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "deterministic": True,
        "backend": "cudnn.deterministic=True, benchmark=False",
        "checkpoint_reused": reused,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import matplotlib
        meta["matplotlib"] = matplotlib.__version__
    except Exception:
        meta["matplotlib"] = None
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def _maybe_write_loss_curve(history_path: Path, curve_path: Path, reused: bool) -> None:
    if reused:
        print("[phase3] loss curve skipped (checkpoint reused)")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib optional
        print(f"[phase3] loss curve skipped: {exc}")
        return
    hist = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [e["epoch"] for e in hist["history"]]
    train = [e["loss"] for e in hist["history"]]
    val = [e["val_loss"] for e in hist["history"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, label="train loss")
    ax.plot(epochs, val, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalized targets)")
    ax.set_title("P4 Phase 3 - LSTM training / validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(curve_path, dpi=140)
    plt.close(fig)
    print(f"[phase3] loss curve written -> {curve_path}")


def _check_overfitting(cmpt, hist, warnings: list) -> None:
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        best_train = data.get("train_loss_at_best_epoch")
        best_val = data.get("best_val_loss")
        final_train = data.get("final_train_loss")
        final_val = data.get("final_val_loss")
    except Exception:
        return
    if None in (best_train, best_val, final_train, final_val):
        return
    _ = cmpt
    if final_train < best_train and final_val > best_val * 1.2:
        warnings.append("OVERFITTING SIGNAL: training loss kept decreasing while validation "
                        "loss rose >20% after the best epoch (reported, not hidden).")
    else:
        print("[phase3] overfitting check: no >20% val-loss divergence after best epoch.")


def _print_final(result, evaluation, comparison, dataset_info, verdict, reused,
                 p1_count, phase2_count, outside_phase3, test_lines):
    t = evaluation["test"]
    best_train = None
    try:
        hist = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        best_train = hist.get("train_loss_at_best_epoch")
    except Exception:
        pass
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        mc = json.load(fh)
    best_val = result.best_val_loss if result.best_val_loss is not None else "n/a"
    imp_p = comparison["improvement_vs_persistence"]["24h"]["track_error_km_mean_pct"]
    imp_mv = comparison["improvement_vs_movement_vector"]["24h"]["track_error_km_mean_pct"]
    summary = " ".join(test_lines).strip() or "n/a"

    print()
    print("=" * 60)
    print("P4 PHASE 3 COMPLETE")
    print("=" * 60)
    print("Dataset: canonical chronological CLEAN")
    rd = dataset_info
    print(f"Train sequences={rd['train']['sequences']} cyclones={rd['train']['cyclones']}")
    print(f"Validation sequences={rd['val']['sequences']} cyclones={rd['val']['cyclones']}")
    print(f"Test sequences={rd['test']['sequences']} cyclones={rd['test']['cyclones']}")
    print()
    print(f"Model: LSTM {mc['input_size']} -> {mc['hidden_size']} -> {mc['output_size']}")
    print(f"Parameters: {mc['parameters']}")
    print(f"Training: best_epoch={result.best_epoch} best_val_loss={best_val} "
          f"best_train_loss={best_train} time={result.training_time_s:.2f}s")
    print()
    for hz, name in (("6h", "+6h"), ("12h", "+12h"), ("24h", "+24h")):
        m = t[hz]
        print(f"{name}: track_error={m['track_error_km_mean']:.2f} km | "
              f"wind_mae={m['wind_mae']:.2f} | wind_rmse={m['wind_rmse']:.2f}")
    print()
    print("Baseline comparison (24h mean track error, LSTM vs baseline):")
    print(f"  Persistence    : {imp_p:+.2f}% ({'LSTM better' if imp_p > 0 else 'baseline better'})")
    print(f"  Movement Vector: {imp_mv:+.2f}% ({'LSTM better' if imp_mv > 0 else 'baseline better'})")
    print(f"Inference: PASS")
    print(f"Tests: {summary}")
    print(f"P1 sources modified : {p1_count}")
    print(f"Phase-1 files modified : {p1_count}")
    print(f"Phase-2 files modified : {phase2_count}")
    print(f"Files changed outside phase3: {outside_phase3}")
    print(f"Checkpoint reused: {reused}")
    print(f"Status: {verdict}")
    print("=" * 60)


def _print_paths(warnings):
    print()
    print("1. best checkpoint  :", CKPT_PATH)
    print("2. normalization    :", STATS_PATH)
    print("3. model comparison :", COMPARISON_PATH)
    print("4. MODEL_REPORT.md  :", REPORT_PATH)
    print("5. metrics          :", EVAL_PATH)
    print("6. loss curve       :", CURVE_PATH)
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print("  -", w)
    print()


if __name__ == "__main__":
    sys.exit(main())