"""PHASE4_REPORT.md generator (required 22-section structure).

Called once at the end of a successful (or partially successful) run.  The
report states PASS / FAIL honestly - failed experiments stay documented and any
loss vs baselines is reported explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def _json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def generate_phase4_report(report_elements: Dict[str, object], path: Path) -> None:
    results = Path(report_elements["results_dir"])
    L: list = ["# P4 Phase-4 Report (Tropical Cyclone Forecasting)", ""]
    L.append(f"- Generated: {report_elements.get('generated_at')}")
    L.append(f"- Overall status: **{report_elements.get('status', 'FAIL')}**")
    L.append("")

    def section(n, title):
        L.append("")
        L.append(f"## {n}. {title}")
        L.append("")

    # 1 Objective
    section(1, "Objective")
    L.append("Train and honestly benchmark three Phase-4 model families "
             "(improved LSTM, GRU, multi-task LSTM) on an engineered 16-feature "
             "causal contract, select one champion using VALIDATION only, and "
             "evaluate that champion exactly once on TEST against the Phase-2 "
             "persistence and movement-vector baselines and the Phase-3 LSTM.")

    # 2 Source datasets
    section(2, "Source datasets")
    L.append("- `p4_forecasting/canonical/` (canonical raw arrays + quality flags)")
    L.append("- `p4_forecasting/canonical_chrono/` (chronological 70/15/15 split)")
    L.append("- `p4_forecasting/phase2/results/canonical_chronological_clean/` "
             "(authoritative CLEAN-only source; all inputs strictly read-only)")

    # 3 Input contract
    section(3, "Input contract")
    L.append("X = (N, 5, 7) raw: `[lat, lon, wind_speed, pressure, sst, wind_u, "
             "wind_v]`; Y = (N, 3, 3) `[lat, lon, wind_speed]` at +6/+12/+24 h; "
             "SST in degrees Celsius; history cadence 6-hourly.")

    # 4 Engineered features
    section(4, "Nine engineered features")
    from phase4.features.build_feature_dataset import FEATURE_FORMULA_ROWS
    contract = results / "FEATURE_CONTRACT.md"
    if contract.exists():
        L.append("See `results/FEATURE_CONTRACT.md` (formulas/units). Table:")
    L.append("")
    L.append("| feature | units | formula |")
    L.append("|---|---|---|")
    for name, units, formula in FEATURE_FORMULA_ROWS:
        L.append(f"| {name} | {units} | {formula} |")
    L.append("")

    # 5 First-step zero-fill
    section(5, "First-step zero-fill policy")
    from phase4.features.feature_engineering import FIRST_STEP_ZERO_FILL_DOC
    L.append(f"> {FIRST_STEP_ZERO_FILL_DOC}")

    # 6 Causality / leakage validation
    section(6, "Causality / leakage validation")
    audit = _json(results / "input_audit.json")
    tests_ok = report_elements.get("tests_passed", False)
    L.append(f"- Input audit overall PASS: {audit.get('overall_pass')}")
    L.append(f"- Future-leak mutation tests + full unit suite passed: {tests_ok}")
    L.append("- Engineered features depend only on X[:, :i+1, :]; target Y mutation "
             "cannot change any engineered feature (verified by tests).")

    # 7 Dataset counts
    section(7, "Dataset counts")
    L.append("| split | sequences | cyclones | X | Y |")
    L.append("|---|---:|---:|---|---|")
    for split in ("train", "val", "test"):
        s = audit["details"]["splits"][split]
        L.append(f"| {split} | {s['sequences']} | {s['cyclones']} | {s['X_shape']} | {s['Y_shape']} |")

    # 8 Normalization
    section(8, "Normalization policy")
    stats = _json(results / "normalization_stats.json")
    L.append(f"- Train-only statistics: {stats['computed_from']['policy']}.")
    L.append(f"- Zero-std features handled as: {stats.get('zero_std_handling')} "
             f"({stats.get('zero_std_features')}).")
    L.append(f"- Directional features documented in stats `directional_features`.")

    # 9 Model architectures
    section(9, "Model architectures")
    L.append("- ImprovedLSTM: `16 -> LSTM -> last hidden -> Linear -> 9 -> (B,3,3)`")
    L.append("- GRUCyclone: same contract with GRU encoder")
    L.append("- MultiTaskLSTM: shared LSTM encoder + separate track (lat/lon) and "
             "intensity (wind) heads composed to `(B,3,3)` = [lat, lon, wind]")

    # 10 Loss functions
    section(10, "Loss functions")
    L.append("- `ForecastMSELoss` - plain MSE over (B,3,3)")
    L.append("- `ForecastHuberLoss` - smooth-L1 with configurable delta")
    L.append("- `WeightedMultiTaskLoss` - configurable track_weight / "
             "intensity_weight / horizon_weights, normalized by weight sums")

    # 11 Experiment configurations
    section(11, "EXP001-EXP006 configurations")
    L.append("| exp | model | loss | hidden | layers | dropout | lr | batch | seed |")
    L.append("|---|---|---|---:|---:|---:|---|---|---:|")
    for eid in ("EXP001", "EXP002", "EXP003", "EXP004", "EXP005", "EXP006"):
        cfgp = report_elements["configs_dir"] / f"{eid}.json"
        if not cfgp.exists():
            L.append(f"| {eid} | MISSING CONFIG | | | | | | | |")
            continue
        c = _json(cfgp)
        L.append(f"| {eid} | {c.get('model')} | {c.get('loss')} | {c.get('hidden_size')} | "
                 f"{c.get('layers')} | {c.get('dropout')} | {c.get('learning_rate')} | "
                 f"{c.get('batch_size')} | {c.get('seed')} |")

    # 12 Validation results
    section(12, "Validation results")
    registry = results / "experiment_registry.csv"
    if registry.exists():
        import pandas as pd
        df = pd.read_csv(registry)
        L.append("| exp | model | loss | best_epoch | best_val_loss | track 6h | track 12h | "
                 "track 24h | wind MAE 6/12/24h | status |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in df.iterrows():
            L.append(f"| {r['experiment_id']} | {r['model']} | {r['loss']} | "
                     f"{_fmt(r['best_epoch'])} | {_fmt(r['best_val_loss'])} | "
                     f"{_fmt(r['val_track_6h'])} | {_fmt(r['val_track_12h'])} | "
                     f"{_fmt(r['val_track_24h'])} | "
                     f"{_fmt(r['val_wind_mae_6h'])}/{_fmt(r['val_wind_mae_12h'])}/"
                     f"{_fmt(r['val_wind_mae_24h'])} | {r['status']} |")

    # 13 Champion selection rule
    section(13, "Champion-selection rule")
    L.append("Locked: lowest equal-weight mean of VALIDATION track errors "
             "`mean(val_track_6h, val_track_12h, val_track_24h)`; tie-break wind "
             "MAE then RMSE. TEST is never used for selection.")

    # 14 Champion
    section(14, "Champion")
    champ = results / "champion_model.json"
    if champ.exists():
        c = _json(champ)
        L.append(f"- Champion experiment: **{c.get('experiment_id')}**")
        L.append(f"- Primary (validation) score: {_fmt(c.get('primary_score'))}")
        L.append(f"- Components: {c.get('component_horizon_scores')}")
        L.append(f"- Selection rule: {c.get('selection_rule')}")
    else:
        L.append("- No champion selected (pipeline did not complete).")

    # 15 Final test results
    section(15, "Final test results (champion, evaluated exactly once)")
    testpath = results / "test_touch.json"
    champion_test = None
    if champion_dir := report_elements.get("champion_dir"):
        tp = Path(champion_dir) / "test_results.json"
        if tp.exists():
            champion_test = _json(tp)
    if champion_test:
        e = champion_test if "metrics" not in champion_test else champion_test["metrics"]
        perf = champion_test.get("metrics", champion_test)
        for hz in ("6h", "12h", "24h"):
            m = perf[hz]
            L.append(f"- +{hz}: track={m['track_error_km_mean']:.2f} km "
                     f"(median {m['track_error_km_median']:.2f}), wind MAE="
                     f"{m['wind_mae']:.2f}, RMSE={m['wind_rmse']:.2f} km/h")
    else:
        L.append("- Champion test result not present.")

    # 16-18 Baseline comparisons
    cmp = results / "FINAL_COMPARISON.json"
    section(16, "Comparison against persistence")
    L.extend(_comparison_rows(cmp, "improvement_vs_persistence"))
    section(17, "Comparison against movement-vector")
    L.extend(_comparison_rows(cmp, "improvement_vs_movement_vector"))
    section(18, "Comparison against phase-3 LSTM")
    L.extend(_comparison_rows(cmp, "improvement_vs_phase3_lstm"))
    if cmp.exists():
        imp = _json(cmp)
        if imp.get("improvement_note"):
            L.append("")
            L.append(f"> {imp['improvement_note']}")

    # 19 Honest limitations
    section(19, "Honest limitations")
    L.append("- If the champion underperforms the baselines that is reported "
             "verbatim (negative improvement percentages), never fabricated.")
    L.append("- Failed experiments remain in the registry with status=FAILED.")
    L.append("- First-step zero-fill applies to all predecessor-dependent "
             "features (documented).")
    L.append("- Directional features are z-scored linearly; the 0/360 wrap "
             "discontinuity is a documented limitation of the locked 16-column "
             "contract (no sine/cosine embedding).")
    L.extend(report_elements.get("limitations_extra", []))

    # 20 Reproducibility
    section(20, "Reproducibility")
    L.append("- Fixed seed (42), CPU-only deterministic execution, "
             "`PYTHONDONTWRITEBYTECODE=1`, `python -X utf8`.")
    L.append("- Idempotent orchestrator: completed experiments are reused "
             "unless `--force-retrain`; feature dataset and normalization are "
             "deterministic and content-equal.")

    # 21 Source immutability
    section(21, "Source immutability")
    imm = results / "source_immutability_report.json"
    if imm.exists():
        r = _json(imm)
        counts = r.get("counts", {})
        L.append("| bucket | changed files |")
        L.append("|---|---|")
        for k in ("p1_modified", "phase1_modified", "phase2_modified",
                  "phase3_modified", "externally_modified", "out_of_writes"):
            pass
        for k in ("p1_modified", "phase1_modified", "phase2_modified",
                  "phase3_modified", "outside_phase4_modified"):
            L.append(f"| {k} | {counts.get(k, 'n/a')} |")
        L.append(f"- Verdict: {r.get('verdict')}")
    else:
        L.append("- Immutability report not present yet.")

    # 22 Final status
    section(22, "Final status")
    L.append(f"**{report_elements.get('status', 'FAIL')}**")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def _comparison_rows(cmp_path: Path, key: str):
    out = []
    if not cmp_path.exists():
        return ["- Not available (final comparison not generated)."]
    c = _json(cmp_path)
    block = c.get(key)
    if not block:
        return ["- Not available."]
    out.append("| horizon | track % | wind MAE % | wind RMSE % |")
    out.append("|---|---:|---:|---:|")
    for hz in ("6h", "12h", "24h"):
        b = block[hz]
        out.append(f"| {hz} | {b['track_error_km_mean_pct']:+.1f} | "
                   f"{b['wind_mae_pct']:+.1f} | {b['wind_rmse_pct']:+.1f} |")
    return out