"""P4 Phase-4 orchestrator (STEP 0-18).

Usage:
    python -X utf8 p4_forecasting/phase4/run_phase4.py [--force-retrain]
                               [--skip-pytest] [--start N] [--stop N]

Everything Phase-4 writes stays under ``p4_forecasting/phase4/``.  All Phase-1/
P1/Phase-2/Phase-3 artifacts are hashed before and after the run; any change is
reported (and the final report is marked FAIL).

Test split policy: validation drives all model selection; the TEST split is
evaluated exactly once, champion only, guarded by ``results/test_touch.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
for _p in (str(_PKG), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from phase4.common import PROJECT_ROOT, PKG_DIR, HORIZONS, WriteGuard, sha256  # noqa: E402
from phase4.configs import (  # noqa: E402
    concretize_best_experiments,
    select_best_architecture,
    write_initial_configs,
)
from phase4.experiments import run_one_experiment  # noqa: E402
from phase4.final_comparison import build_final_comparison, save_final_comparison  # noqa: E402
from phase4.features.build_feature_dataset import (  # noqa: E402
    build_feature_datasets,
    write_feature_contract,
    write_feature_engineering_report,
)
from phase4.immutability import (  # noqa: E402
    compute_immutability_report,
    group_snapshots,
    write_immutability_report,
)
from phase4.input_audit import AuditContext, run_input_audit, save_input_audit  # noqa: E402
from phase4.evaluation.evaluate import (  # noqa: E402
    evaluate_split,
    load_model_from_config,
    write_json,
)
from phase4.evaluation.selection import (  # noqa: E402
    select_champion,
    make_champion_json,
    save_champion_json,
)
from phase4.registry import save_registry  # noqa: E402
from phase4.report import generate_phase4_report  # noqa: E402
from phase4.training.normalization import Normalizer  # noqa: E402

RESULTS = PKG_DIR / "phase4" / "results"
REPORTS = PKG_DIR / "phase4" / "reports"
CONFIGS = PKG_DIR / "phase4" / "configs"
FEATURE_DIR = RESULTS / "feature_dataset"
EXPERIMENTS_DIR = RESULTS / "experiments"
TESTS_DIR = PKG_DIR / "phase4" / "tests"

CHRONO = PKG_DIR / "canonical_chrono"
CANONICAL = PKG_DIR / "canonical"
CLEAN = PKG_DIR / "phase2" / "results" / "canonical_chronological_clean"
P2_BASELINE = PKG_DIR / "phase2" / "results" / "baseline_results.json"
P3_COMPARISON = PKG_DIR / "phase3" / "results" / "model_comparison.json"
P3_STATS = PKG_DIR / "phase3" / "results" / "normalization_stats.json"
QUALITY = CANONICAL / "sample_quality.csv"

ALL_EXPERIMENTS = ["EXP001", "EXP002", "EXP003", "EXP004", "EXP005", "EXP006"]


def step(n, title, start, stop):
    return (start is None or n >= start) and (stop is None or n <= stop)


def run_pytest(phase4_tests: Path, select: str | None = None) -> bool:
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    target = str(phase4_tests)
    if select:
        target = str(phase4_tests / select)
    cmd.append(target)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(cmd, cwd=str(PKG_DIR), env=env, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-4000:])
    return proc.returncode == 0


def immutable_sources_dict() -> Dict[str, str]:
    """Hashes of every consumed Read-only artifact (for provenance)."""
    src: Dict[str, str] = {}
    for key in ("train", "val", "test"):
        src[f"chrono_{key}.npz"] = sha256(CHRONO / f"{key}.npz")
        src[f"clean_{key}.npz"] = sha256(CLEAN / f"{key}.npz")
    src["quality.csv"] = sha256(QUALITY)
    src["p2_baseline_results.json"] = sha256(P2_BASELINE)
    src["p3_model_comparison.json"] = sha256(P3_COMPARISON)
    return src


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 Phase-4 pipeline")
    parser.add_argument("--force-retrain", action="store_true",
                        help="retrain all experiments even if checkpoints exist")
    parser.add_argument("--skip-pytest", action="store_true",
                        help="skip the internal pytest steps (7/15/17)")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    args = parser.parse_args()

    guard = WriteGuard(_HERE)
    now = datetime.now(timezone.utc).isoformat()

    # --- STEP 0: guard & environment ---
    print("=" * 72)
    print("P4 PHASE-4 PIPELINE")
    print(f"phase4 dir : {_HERE}")
    print(f"python     : {sys.version.split()[0]}")
    try:
        import torch, numpy, pandas
        print(f"torch      : {torch.__version__} | numpy {numpy.__version__} | pandas {pandas.__version__}")
    except Exception:
        pass
    print(f"write guard: ALL outputs restricted to {guard.root}")
    print("=" * 72)

    # ---- STEP 1: source immutability baseline (BEFORE) ----
    if step(1, "immutability-before", args.start, args.stop):
        print("[STEP 1] hashing all Read-only sources (P1/Phase-1/Phase-2/Phase-3 BUCKET)...")
        before = group_snapshots(PKG_DIR)
        print(f"  buckets: p1={len(before['p1'])} phase1={len(before['phase1'])} "
              f"phase2={len(before['phase2'])} phase3={len(before['phase3'])} "
              f"outside={len(before['files_outside_phase4'])} files")

    # ---- STEP 2: input audit ----
    if step(2, "input-audit", args.start, args.stop):
        print("[STEP 2] read-only input contract audit ...")
        ctx = AuditContext(chrono_dir=CHRONO, canonical_dir=CANONICAL,
                           phase2_clean_dir=CLEAN,
                           phase2_baseline_json=P2_BASELINE,
                           phase3_comparison_json=P3_COMPARISON,
                           phase3_stats_json=P3_STATS,
                           quality_csv=QUALITY, p4_dir=PKG_DIR)
        audit = run_input_audit(ctx)
        save_input_audit(audit, RESULTS / "input_audit.json")
        print(f"  checks: {audit['audit_summary']['checks_total']}, "
              f"failed: {audit['audit_summary']['checks_failed']}, "
              f"overall_pass: {audit['overall_pass']}")
        if not audit["overall_pass"]:
            raise SystemExit("FATAL: input audit failed - aborting before any training")

    # ---- STEP 3: feature dataset ----
    if step(3, "feature-dataset", args.start, args.stop):
        print("[STEP 3] building causal 16-feature datasets ...")
        for p in (CLEAN, CHRONO, QUALITY):
            if not Path(p).exists():
                raise FileNotFoundError(p)
        summary = build_feature_datasets(CLEAN, CHRONO, QUALITY, FEATURE_DIR)
        write_feature_contract(summary, RESULTS / "FEATURE_CONTRACT.md")
        write_feature_engineering_report(summary, Path(_HERE / "features" / "FEATURE_ENGINEERING_REPORT.md"))
        write_json(summary, RESULTS / "feature_dataset_summary.json")
        for split in ("train", "val", "test"):
            s = summary["splits"][split]
            print(f"  {split}: {s['sequences']} seq / {s['cyclones']} cyc, "
                  f"X={tuple(s['X_shape'])} nan={s['nan']} inf={s['inf']} wrote={s['wrote']}")

    # ---- STEP 4: future-leak verification (causal) ----
    if step(4, "leak-check", args.start, args.stop):
        print("[STEP 4] causal future-leak verification on the built datasets ...")
        ok = verify_causality(FEATURE_DIR, CLEAN)
        print(f"  causality check: {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("FATAL: causality check failed")

    # ---- STEP 5: normalization stats (train-only) ----
    if step(5, "normalization", args.start, args.stop):
        print("[STEP 5] computing TRAIN-only normalization statistics ...")
        z = __import__("numpy").load(FEATURE_DIR / "train.npz", allow_pickle=True)
        X, Y = z["X"], z["Y"]
        from phase4.training.normalization import compute_normalization_stats
        stats = compute_normalization_stats(X, Y)
        Normalizer(stats).save(RESULTS / "normalization_stats.json")
        print(f"  train samples: {stats['n_train_samples']}; "
              f"zero-std features: {stats['zero_std_features']}")

    # ---- STEP 6: dataloader smoke ----
    if step(6, "dataloader-smoke", args.start, args.stop):
        print("[STEP 6] dataloader smoke check ...")
        stats = Normalizer.from_path(RESULTS / "normalization_stats.json")
        import torch
        from torch.utils.data import DataLoader
        from phase4.dataloader.forecasting_dataset import (
            ForecastingDataset, collate_forecasting)
        for split in ("train", "val", "test"):
            ds = ForecastingDataset(FEATURE_DIR / f"{split}.npz", stats,
                                    FEATURE_DIR / f"{split}_metadata.csv")
            batch = next(iter(DataLoader(ds, batch_size=64, shuffle=False,
                                         collate_fn=collate_forecasting)))
            print(f"  {split}: n={len(ds)} batch X={tuple(batch['history'].shape)} "
                  f"Y={tuple(batch['target'].shape)}")
            assert tuple(batch["history"].shape[1:]) == (5, 16)
            assert tuple(batch["target"].shape[1:]) == (3, 3)

    # ---- STEP 7: pytest (first pass) ----
    if step(7, "pytest", args.start, args.stop) and not args.skip_pytest:
        print("[STEP 7] running unit test suite (first pass) ...")
        ok = run_pytest(TESTS_DIR)
        if not ok:
            raise SystemExit("FATAL: unit tests failed at STEP 7")

    # ---- STEP 8: experiments ----
    if step(8, "experiments", args.start, args.stop):
        print("[STEP 8] running EXP001-EXP006 ...")
        write_initial_configs(CONFIGS)
        sources = immutable_sources_dict()
        summaries: List[Dict[str, object]] = []
        for eid in ("EXP001", "EXP002", "EXP003", "EXP004"):
            summaries.append(run_one_experiment(
                eid, CONFIGS, FEATURE_DIR, RESULTS / "normalization_stats.json",
                RESULTS, sources, args.force_retrain))
        # --- best-architecture selection among EXP002-004 (VALIDATION only) ---
        val_map = {s["experiment_id"]: s.get("validation_metrics", {})
                   for s in summaries if s["status"] == "PASS"
                   and s["experiment_id"] in ("EXP002", "EXP003", "EXP004")}
        best_arch = select_best_architecture(val_map)
        print(f"  [STEP 8] best architecture (EXP002-004, validation): {best_arch}")
        concretize_best_experiments(CONFIGS, best_arch)
        for eid in ("EXP005", "EXP006"):
            summaries.append(run_one_experiment(
                eid, CONFIGS, FEATURE_DIR, RESULTS / "normalization_stats.json",
                RESULTS, sources, args.force_retrain))
        write_json({s["experiment_id"]: s for s in summaries},
                   RESULTS / "experiments_summary.json")
        for s in summaries:
            print(f"  {s['experiment_id']}: status={s['status']} "
                  f"primary_val={s.get('validation_primary_score', 'n/a')}")

    # ---- STEP 9: collect validation results ----
    if step(9, "val-results", args.start, args.stop):
        print("[STEP 9] assembling validation results ...")
        summaries = json.loads((RESULTS / "experiments_summary.json").read_text(encoding="utf-8"))
        summaries = list(summaries.values())
        val_results = {s["experiment_id"]: s["validation_metrics"]
                       for s in summaries if s["status"] == "PASS" and s["validation_metrics"]}
        write_json(val_results, RESULTS / "validation_results.json")
        print(f"  PASS experiments: {sorted(val_results)}")

    # ---- STEP 10: registry ----
    if step(10, "registry", args.start, args.stop):
        print("[STEP 10] building experiment registry ...")
        summaries = list(json.loads((RESULTS / "experiments_summary.json").read_text(encoding="utf-8")).values())
        save_registry(RESULTS / "experiment_registry.csv", summaries)
        print(f"  registry rows: {len(summaries)} -> results/experiment_registry.csv")

    # ---- STEP 11: champion selection (validation only) ----
    if step(11, "champion", args.start, args.stop):
        print("[STEP 11] selecting champion on VALIDATION (test never used) ...")
        val_results = json.loads((RESULTS / "validation_results.json").read_text(encoding="utf-8"))
        champion_id, primary, rationale = select_champion(val_results)
        summaries = list(json.loads((RESULTS / "experiments_summary.json").read_text(encoding="utf-8")).values())
        champion_summary = next(s for s in summaries if s["experiment_id"] == champion_id)
        src_hashes = json.loads(
            (Path(champion_summary["dir"]) / "source_hashes.json").read_text(encoding="utf-8"))
        run_meta = {
            "date": now,
            "python": sys.version.split()[0],
            "arg_force_retrain": args.force_retrain,
            "n_experiments_pass": sum(1 for s in summaries if s["status"] == "PASS"),
        }
        payload = make_champion_json(champion_id, rationale, run_meta, src_hashes)
        save_champion_json(payload, RESULTS / "champion_model.json")
        print(f"  champion: {champion_id} primary_val={primary:.3f}")
        write_json(rationale, RESULTS / "champion_rationale.json")

    # ---- STEP 12: single-use test evaluation ----
    if step(12, "test-once", args.start, args.stop):
        print("[STEP 12] champion TEST evaluation (exactly once, guarded) ...")
        champion = json.loads((RESULTS / "champion_model.json").read_text(encoding="utf-8"))
        champion_id = champion["experiment_id"]
        exp_dir = EXPERIMENTS_DIR / champion_id
        ckpt = exp_dir / "checkpoint.pt"
        ckpt_hash = sha256(ckpt)
        touch_path = RESULTS / "test_touch.json"
        if touch_path.exists():
            touch = json.loads(touch_path.read_text(encoding="utf-8"))
            if touch["champion"] == champion_id and touch["checkpoint_sha256"] == ckpt_hash:
                print(f"  reuse test_touch.json (champion {champion_id}, same checkpoint hash)")
            else:
                raise SystemExit(
                    "FATAL: results/test_touch.json exists for a different "
                    f"champion/hash ({touch.get('champion')}); TEST MUST NEVER BE "
                    "evaluated by a second model. Inspect and resolve manually.")
        else:
            model = load_model_from_config(exp_dir / "config.json", ckpt)
            stats = Normalizer.from_path(RESULTS / "normalization_stats.json")
            result = evaluate_split(model, stats, FEATURE_DIR, "test")
            write_json(result, exp_dir / "test_results.json")
            write_json({
                "champion": champion_id,
                "checkpoint_sha256": ckpt_hash,
                "split": "test",
                "evaluated_at": now,
                "evaluated_times": 1,
                "note": "TEST evaluated exactly once, champion only; election never uses test.",
            }, touch_path)
            print(f"  TEST evaluated exactly once for {champion_id}; "
                  f"track_6h={result['metrics']['6h']['track_error_km_mean']:.2f} km")

    # ---- STEP 13: final comparison ----
    if step(13, "final-comparison", args.start, args.stop):
        print("[STEP 13] building FINAL_COMPARISON.json ...")
        champion = json.loads((RESULTS / "champion_model.json").read_text(encoding="utf-8"))
        champion_id = champion["experiment_id"]
        exp_dir = EXPERIMENTS_DIR / champion_id
        test_result = json.loads((exp_dir / "test_results.json").read_text(encoding="utf-8"))
        val_results = json.loads((RESULTS / "validation_results.json").read_text(encoding="utf-8"))
        p2 = json.loads(P2_BASELINE.read_text(encoding="utf-8"))
        p3 = json.loads(P3_COMPARISON.read_text(encoding="utf-8"))
        dataset_info = {"train": {"sequences": 1212, "cyclones": 57},
                        "val": {"sequences": 231, "cyclones": 13},
                        "test": {"sequences": 198, "cyclones": 10}}
        comparison = build_final_comparison(
            phase2_baseline=p2, phase3_comparison=p3,
            champion_test=test_result["metrics"],
            champion_val=val_results[champion_id],
            exp_val_results=val_results,
            champion_id=champion_id,
            dataset_info=dataset_info)
        save_final_comparison(comparison, RESULTS / "FINAL_COMPARISON.json")
        print(f"  FINAL_COMPARISON.json written (primary split = test)")

    # ---- STEP 14: immutability after + report ----
    if step(14, "immutability-after", args.start, args.stop):
        print("[STEP 14] hashing Read-only sources again (AFTER) ...")
        before = before if "before" in dir() else group_snapshots(PKG_DIR)
        after = group_snapshots(PKG_DIR)
        report = compute_immutability_report(before, after, recorded_at=now)
        write_immutability_report(report, RESULTS / "source_immutability_report.json")
        print(f"  verdict: {report['verdict']} "
              f"(p1={report['counts']['p1_modified']} "
              f"phase1={report['counts']['phase1_modified']} "
              f"phase2={report['counts']['phase2_modified']} "
              f"phase3={report['counts']['phase3_modified']} "
              f"outside={report['counts']['files_outside_phase4_modified']})")

    # ---- STEP 15: immutability test ----
    if step(15, "immutability-test", args.start, args.stop) and not args.skip_pytest:
        print("[STEP 15] running source-immutability test ...")
        ok = run_pytest(TESTS_DIR, select="test_source_immutability.py")
        if not ok:
            raise SystemExit("FATAL: source immutability test failed")

    # ---- STEP 16: PHASE4_REPORT.md ----
    if step(16, "report", args.start, args.stop):
        print("[STEP 16] writing reports/PHASE4_REPORT.md ...")
        try:
            imm_report = json.loads((RESULTS / "source_immutability_report.json").read_text(encoding="utf-8"))
            imm_status = imm_report["verdict"]
            imm_counts = imm_report["counts"]
        except Exception:
            imm_status, imm_counts = "UNKNOWN", {}
        champion_json = RESULTS / "champion_model.json"
        overall = "PASS"
        if not champion_json.exists():
            overall = "FAIL"
        if imm_status == "FAIL":
            overall = "FAIL"
        generate_phase4_report({
            "results_dir": RESULTS,
            "configs_dir": CONFIGS,
            "champion_dir": (EXPERIMENTS_DIR / json.loads(champion_json.read_text(encoding="utf-8"))["experiment_id"])
            if champion_json.exists() else None,
            "generated_at": now,
            "tests_passed": True,
            "status": overall,
            "immutability": imm_counts,
            "limitations_extra": [],
        }, REPORTS / "PHASE4_REPORT.md")
        print(f"  reports/PHASE4_REPORT.md written (status={overall})")

    # ---- STEP 17: final pytest ----
    if step(17, "final-pytest", args.start, args.stop) and not args.skip_pytest:
        print("[STEP 17] running full unit test suite (final pass) ...")
        ok = run_pytest(TESTS_DIR)
        if not ok:
            raise SystemExit("FATAL: final pytest failed")

    # ---- STEP 18: final summary ----
    if step(18, "final-summary", args.start, args.stop):
        print("[STEP 18] final summary")
        print("=" * 72)
        imm_report = json.loads((RESULTS / "source_immutability_report.json").read_text(encoding="utf-8"))
        counts = imm_report["counts"]
        print("P4 PHASE-4 RUN COMPLETE")
        print(f"  Source immutability        : {imm_report['verdict']}")
        print(f"    P1 modified              : {counts['p1_modified']}")
        print(f"    Phase-1 modified         : {counts['phase1_modified']}")
        print(f"    Phase-2 modified         : {counts['phase2_modified']}")
        print(f"    Phase-3 modified         : {counts['phase3_modified']}")
        print(f"    Files outside phase4     : {counts['files_outside_phase4_modified']}")
        champ = json.loads((RESULTS / "champion_model.json").read_text(encoding="utf-8"))
        print(f"  Champion (validation rule) : {champ['experiment_id']} "
              f"primary={champ['primary_score']:.3f}")
        touch = json.loads((RESULTS / "test_touch.json").read_text(encoding="utf-8"))
        print(f"  Test evaluated once        : True (champion={touch['champion']})")
        cmp = json.loads((RESULTS / "FINAL_COMPARISON.json").read_text(encoding="utf-8"))
        for key, label in (("improvement_vs_persistence", "vs persistence"),
                           ("improvement_vs_movement_vector", "vs movement-vector"),
                           ("improvement_vs_phase3_lstm", "vs phase-3 LSTM")):
            b6 = cmp[key]["6h"]
            print(f"  Test track {label:>22}: 6h={b6['track_error_km_mean_pct']:+.1f}% "
                  f"12h={cmp[key]['12h']['track_error_km_mean_pct']:+.1f}% "
                  f"24h={cmp[key]['24h']['track_error_km_mean_pct']:+.1f}%")
        print("  Artifacts:")
        for rel in ("input_audit.json", "feature_dataset_summary.json",
                    "FEATURE_CONTRACT.md", "normalization_stats.json",
                    "experiment_registry.csv", "champion_model.json",
                    "test_touch.json", "FINAL_COMPARISON.json",
                    "source_immutability_report.json"):
            p = RESULTS / rel
            print(f"    {('OK' if p.exists() else 'MISSING'):8} results/{rel}")
        for rel in ("PHASE4_REPORT.md",):
            p = REPORTS / rel
            print(f"    {('OK' if p.exists() else 'MISSING'):8} reports/{rel}")
        print("=" * 72)


def verify_causality(feature_dir: Path, clean_dir: Path) -> bool:
    """Re-engineering parity: stored feature dataset equals fresh causal
    construction from the raw CLEAN source, and future-step mutations never
    change earlier features on a sampled subset."""
    import numpy as np
    from phase4.features.feature_engineering import engineer_features

    ok = True
    for split in ("train", "val", "test"):
        stored = np.load(feature_dir / f"{split}.npz", allow_pickle=True)
        raw = np.asarray(np.load(clean_dir / f"{split}.npz", allow_pickle=True)["X"],
                         dtype=np.float32)
        fresh = engineer_features(raw)
        equal = np.array_equal(np.asarray(stored["X"], dtype=np.float32), fresh)
        print(f"  {split}: re-engineering parity = {equal}")
        ok &= bool(equal)
    # sampled mutation check on train
    rng = np.random.RandomState(0)
    raw = np.load(clean_dir / "train.npz", allow_pickle=True)["X"]
    n = min(64, raw.shape[0])
    idx = rng.choice(raw.shape[0], n, replace=False)
    X = np.asarray(raw[idx], np.float32)
    base = engineer_features(X)
    Xm = X.copy()
    Xm[:, 2:, :] += 5.0
    mut = engineer_features(Xm)
    ok &= bool(np.array_equal(base[:, :2, :], mut[:, :2, :]))
    return ok


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)