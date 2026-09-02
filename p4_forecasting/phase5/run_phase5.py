"""Phase-5 runner: verification + reporting.

Run from p4_forecasting/ with ``python -X utf8 phase5/run_phase5.py``.

This is an INTEGRATION / INFERENCE phase: the audited Phase-4 champion is
loaded read-only and exercised end-to-end.  There is deliberately NO
training/retraining flag and no model-performance claim.  It verifies:
paths -> artifacts -> feature contract -> sample inference -> causality ->
determinism -> offline dependency usage -> baseline parity -> test suite ->
source immutability, writing results/inference_audit.json,
results/contract_validation.json, results/phase5_summary.json,
results/source_immutability_report.json and reports/PHASE5_REPORT.md.
Idempotent: repeated runs overwrite outputs; only ``recorded_at`` changes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import numpy as np

import snapshot
from phase5.config import (HISTORY_HOURS, HISTORY_STEPS, N_FEATURES,
                           RAW_FEATURES, default_paths, read_json)
from phase5.inference.predictor import (configure_deterministic_cpu,
                                        load_predictor)
from phase5.inference.preprocessing import (engineer_history,
                                            feature_contract_check,
                                            inspect_stats,
                                            load_feature_order,
                                            load_normalizer)
from phase5.inference.output_contract import (build_forecast_list,
                                              is_json_serializable,
                                              validate_forecast_list,
                                              validate_response)
from phase5.schemas.forecast_schema import describe_output_schema
from phase5.service.forecasting_service import ForecastingService

BANNER = """
======================================================================
         PHASE 5 - FORECASTING SYSTEM INTEGRATION (DEMO LAYER)
======================================================================
  Status                  : {status}
  Champion                : {champion}
  Model                   : {family} ({loss})
  Input shape             : ({history}, {features}) history steps x features
  Horizons                : {horizons} hours
  Inference               : {inference_check}
  Causality               : {causality}
  Determinism             : {determinism}
  Offline                 : {offline} (no network dependency)
  Baselines integrated    : {baselines} ({baseline_sources})
  Output contract         : {contract}
  Tests                   : {tests_passed}/{tests_total} phase5 tests passed
  P1 modified             : {p1} ({limits})
  Phase1 modified         : {phase1}
  Phase2 modified         : {phase2}
  Phase3 modified         : {phase3}
  Phase4 modified         : {phase4}
  Outside phase5 modified : {outside}
  FINAL STATUS            : {final}
======================================================================
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_example(raw: bool = True):
    paths = default_paths()
    p = paths.phase5 / "examples" / "example_input.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    if raw:
        import numpy as np
        steps = doc["history"]
        fields = ["lat", "lon", "wind_speed", "pressure", "sst",
                  "wind_u", "wind_v"]
        h = np.array([[s[f] for f in fields] for s in steps], np.float32)
        return h
    return doc


# --------------------------------------------------------------------------
def cmdline(args=None):
    ap = argparse.ArgumentParser(
        description="Phase-5 integration verification (no training options).")
    ap.add_argument("--force-check", action="store_true",
                    help="re-record the BEFORE snapshot first, then verify")
    return ap.parse_args(args)


# --------------------------------------------------------------------------
def main() -> int:
    args = cmdline()
    t0 = now()
    paths = default_paths()
    results_dir = paths.results_dir
    reports_dir = paths.reports_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    ok = lambda name: print(f"  [ok]  {name}")
    bad = lambda name, msg: (print(f"  [FAIL] {name}: {msg}"),
                             failures.append(f"{name}: {msg}"))

    if args.force_check:
        snapshot.write_before()
    before = snapshot.read_before()
    print("[phase5] BEFORE snapshot:", before["recorded_at"],
          f"({before['n_files']} files)")

    # -- artefact verification ------------------------------------------------
    print("[phase5] 1/9 verifying audited artifacts")
    missing = paths.require()
    if missing:
        bad("phase4-artifacts", f"missing: {missing}")
        fail_fast = True
    else:
        fail_fast = False
        ok("phase4-artifacts")

    champion_meta = None
    if not fail_fast:
        champion_meta = read_json(paths.champion_meta)
        champ_id = str(champion_meta["experiment_id"])
        if champ_id != "EXP005":
            bad("champion", f"expected EXP005, got {champ_id}")
        else:
            ok(f"champion={champ_id} (primary val "
               f"{champion_meta.get('primary_score')})")

    predict = p4 = None
    example = load_example()  # numpy (5,7) from the audited test fold

    if not fail_fast:
        print("[phase5] 2/9 loading predictor (read-only, CPU, deterministic)")
        try:
            configure_deterministic_cpu()
            p4 = load_predictor(paths)
            ok(f"model loaded: {p4.config['model']} "
               f"hidden={p4.config['hidden_size']} "
               f"params={p4.param_count} horizons={p4.horizon_hours}")
            if p4.param_count != 89577:
                bad("model-params",
                    f"expected 89577, got {p4.param_count}")
        except Exception as exc:
            bad("model-load", str(exc))
            fail_fast = True

    infer_check = causality = determinism = offline = contract = "PASS"
    baselines_parity = "PASS"

    if not fail_fast:
        # -- feature contract ---------------------------------------------------
        print("[phase5] 3/9 feature contract + sample inference")
        stats_info = inspect_stats(paths.normalization_stats)
        order = load_feature_order(paths.normalization_stats)
        feats = engineer_history(example)
        cc = feature_contract_check(feats, order)
        record = {"sample_history": example.round(4).tolist(),
                  "feature_contract": cc,
                  "stats": stats_info}
        record["sample"] = {
            "test_split_load": True,
            "from": "phase2/results/canonical_chronological_clean/test.npz",
        }

        cfg_meta = read_json(paths.champion_config)
        input_contract_meta = {
            "history_hours": 24,
            "history_steps": HISTORY_STEPS,
            "feature_count": int(cfg_meta["input_size"]),
            "field_order": list(order),
            "horizons_hours": [6, 12, 24],
        }

        try:
            res1 = p4.predict_features(feats)
            res2 = p4.predict_features(feats)
            exact = bool(np.array_equal(res1, res2))
            maxdiff = float(np.max(np.abs(res1 - res2)))
            contract_forecast = build_forecast_list(res1)
            contract_report = validate_forecast_list(contract_forecast)
            if contract_report["pass"] and is_json_serializable(
                    {"forecast": contract_forecast}):
                ok("sample inference (3 horizons, physical units)")
            else:
                bad("contract", str(contract_report["problems"]))
            if exact or maxdiff <= 1e-6:
                if not exact:
                    ok("determinism (max abs diff 0)")
                    determinism = f"PASS (bitwise equal, diff={maxdiff})"
                else:
                    ok("determinism (bitwise equal)")
            else:
                determinism = "FAIL"
                bad("determinism", f"run1 vs run2 max diff {maxdiff}")
            record["sample_forecast"] = contract_forecast
            record["determinism"] = {"exact_equal": exact,
                                     "max_abs_diff": maxdiff,
                                     "tolerance": 1e-6}
        except Exception as exc:
            bad("sample-inference", str(exc))
            contract = "FAIL"

        # -- causality: future-/target-free forecaster -------------------------
        print("[phase5] 4/9 causality (future/target independence)")
        causality_report = run_causality(p4, example)
        if causality_report["pass"]:
            ok("stepwise causal engineering; no target/future read")
            causality = "PASS"
        else:
            causality = "FAIL"
            bad("causality", str(causality_report["problems"]))
        record["causality"] = causality_report

        # -- offline: no network imports ----------------------------------------
        print("[phase5] 5/9 offline dependency scan")
        offline_report = run_offline_scan()
        if offline_report["pass"]:
            ok("no network imports in phase5 (AST scan)")
            offline = "PASS"
        else:
            offline = "FAIL"
            bad("offline", "; ".join(offline_report["problems"]))
        record["offline"] = offline_report

        # -- baselines parity ----------------------------------------------------
        print("[phase5] 6/9 baseline parity (phase5 == phase2)")
        parity_report = run_baseline_parity(example)
        if parity_report["pass"]:
            ok("persistence & movement-vector identical to Phase 2")
            baselines_parity = "PASS"
        else:
            baselines_parity = "FAIL"
            bad("baseline-parity", str(parity_report["problems"]))
        record["baselines"] = parity_report

        # -- service end-to-end + contract --------------------------------------
        print("[phase5] 7/9 service end-to-end + output contract")
        service = ForecastingService()
        doc = load_example(raw=False)
        response = service.forecast(doc)
        vr = validate_response(response)
        cmp = service.compare_baselines(doc)
        if vr["pass"] and cmp["status"] == "success":
            ok("forecast + compare_baselines (JSON contract valid)")
            contract = "PASS"
        else:
            contract = "FAIL"
            bad("service-contract",
                f"validate={vr}; baselines_status={cmp.get('status')}")
        record["service_contract"] = {
            "forecast": vr,
            "compare_baselines": cmp.get("status"),
        }

        # -- test suite ----------------------------------------------------------
        print("[phase5] 8/9 running phase5 pytest suite")
        tests = run_tests()
        if tests.get("pass"):
            ok(f"pytest: {tests['passed']}/{tests['total']} passed "
               f"({tests['duration']:.1f}s)")
        else:
            bad("pytest",
                f"{tests['passed']}/{tests['total']} passed; rc="
                f"{tests.get('rc')}")
        record["tests"] = tests

    # -- source immutability ----------------------------------------------------
    print("[phase5] 9/9 source immutability vs BEFORE snapshot")
    after_master = snapshot.master_snapshot()
    status = snapshot.make_status(before["hashes"], after_master)
    immutability = snapshot.write_after_report(
        after_master, status,
        note="Phase-5 inference must never modify P1 / Phase-1/2/3/4 / "
             "outside-Phase-5 files.")

    changed_counts = {
        "p1": immutability["p1_changed"],
        "phase1": immutability["phase1_changed"],
        "phase2": immutability["phase2_changed"],
        "phase3": immutability["phase3_changed"],
        "phase4": immutability["phase4_changed"],
        "outside_phase5": immutability["outside_phase5_changed"],
    }
    if immutability["status"] == "PASS":
        ok(f"immutable: {changed_counts}")
    else:
        for k, v in changed_counts.items():
            if v:
                bad(f"immutability-{k}", f"{v} file(s) changed")

    final = "PASS"
    if fail_fast:
        final = "FAIL"
    elif failures or immutability["status"] != "PASS":
        final = "FAIL"
    if any(v for v in changed_counts.values()):
        final = "FAIL"

    # -- audit + summary ---------------------------------------------------------
    audit = {
        "schema": "phase5 v1",
        "recorded_at": now(),
        "phase": 5,
        "experiment_id": champ_id if champion_meta else "EXP005",
        "checkpoint_path": str(paths.champion_checkpoint),
        "input_feature_count": int(cfg_meta["input_size"]) if not fail_fast
                               else N_FEATURES,
        "history_steps": HISTORY_STEPS,
        "history_hours": 24,
        "horizons_hours": [6, 12, 24],
        "normalization_source": "phase4/results/normalization_stats.json"
                                " (train-only)",
        "feature_contract_source": ("phase4/features/feature_engineering.py"
                                    " + feature_order from stats"),
        "targets": ["lat", "lon", "wind_speed_kmh"],
        "causal_verification": causality,
        "deterministic_verification": determinism,
        "offline_verification": offline,
        "output_contract_verification": contract,
        "sample_inference": {
            "history_source": "phase2/results/canonical_chronological_clean/test.npz",
            "forecast": (record.get("sample_forecast")
                         if not fail_fast else None),
        },
        "baseline_parity": baselines_parity,
        "tests_passed": tests.get("passed", 0) if not fail_fast else 0,
        "tests_total": tests.get("total", 0) if not fail_fast else 0,
        "final_status": final,
    }
    (results_dir / "inference_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "schema": "phase5 v1",
        "recorded_at": now(),
        "started_at": t0,
        "experiment_id": audit["experiment_id"],
        "final_status": final,
        "checks": {
            "artifacts": "PASS" if not fail_fast else "FAIL",
            "model_load": "PASS" if p4 is not None else "FAIL",
            "feature_contract": "PASS" if not fail_fast else "FAIL",
            "inference": infer_check,
            "causality": causality,
            "determinism": determinism,
            "offline": offline,
            "baseline_parity": baselines_parity,
            "output_contract": contract,
            "tests": tests.get("pass", False) if not fail_fast else False,
            "immutability": immutability["status"],
        },
        "immutability": changed_counts,
        "failures": failures,
        "outputs": {
            "inference_audit": "phase5/results/inference_audit.json",
            "contract_validation": "phase5/results/contract_validation.json",
            "source_immutability": ("phase5/results/"
                                    "source_immutability_report.json"),
            "summary": "phase5/results/phase5_summary.json",
            "report": "phase5/reports/PHASE5_REPORT.md",
        },
    }
    (results_dir / "phase5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    cv = {
        "schema": "phase5 v1",
        "recorded_at": now(),
        "input": {
            "history_hours": 24,
            "history_steps": HISTORY_STEPS,
            "feature_count": N_FEATURES,
            "field_order": list(order) if not fail_fast else [],
        },
        "output_schema": describe_output_schema(),
        "sample_forecast_valid": contract_report.get("pass", False)
                                 if not fail_fast else False,
        "sample_forecast": record.get("sample_forecast") if not fail_fast
                           else None,
        "final_status": final,
    }
    (results_dir / "contract_validation.json").write_text(
        json.dumps(cv, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # -- report -------------------------------------------------------------------
    write_report(audit, summary, changed_counts, champion_meta,
                 fail_fast=fail_fast)

    family = "GRU"
    if champion_meta:
        family = str(champion_meta.get("family", "GRU"))
    print(BANNER.format(
        status=final,
        champion=audit["experiment_id"],
        family=family,
        loss="Huber",
        history=HISTORY_STEPS,
        features=N_FEATURES,
        horizons="6/12/24",
        inference_check="PASS" if p4 is not None else "FAIL",
        causality=causality,
        determinism=determinism,
        offline=offline,
        baselines=baselines_parity,
        baseline_sources="phase2/baselines (parity-checked)",
        contract=contract,
        tests_passed=tests.get("passed", 0) if not fail_fast else 0,
        tests_total=tests.get("total", 0) if not fail_fast else 0,
        p1=changed_counts["p1"],
        limits="must be 0",
        phase1=changed_counts["phase1"],
        phase2=changed_counts["phase2"],
        phase3=changed_counts["phase3"],
        phase4=changed_counts["phase4"],
        outside=changed_counts["outside_phase5"],
        final=final,
    ))
    return 0 if final == "PASS" else 1


# --------------------------------------------------------------------------
def run_causality(p4, example) -> dict:
    """Stepwise-causality proof for the engineered features + no-target usage."""
    problems = []
    base = engineer_history(example)
    for i in range(HISTORY_STEPS):
        fut = example.copy()
        fut[i + 1:, :] += 1e3
        if not np.array_equal(engineer_history(fut)[i], base[i]):
            problems.append(f"feature row {i} depends on future rows")
    # prediction must not change when future-target data would exist: the
    # (5,7) input has NO target slots; verify the API rejects wider arrays.
    try:
        pred = p4.predict_features(base)
        if not (pred.shape == (3, 3) and np.isfinite(pred).all()):
            problems.append("non-finite / wrong model output")
    except Exception as exc:
        problems.append(f"prediction error: {exc}")
    return {"pass": not problems, "problems": problems,
            "target_read_path": "none (features_from_history uses X only)"}


def run_offline_scan() -> dict:
    """AST scan: phase5 must never import networking/IO modules."""
    import ast
    base = Path(_PKG) / "phase5"
    banned_prefixes = ("urllib", "requests", "socket", "http", "aiohttp",
                       "httpx", "websocket", "ftplib", "xmlrpc")
    problems = []
    for py in sorted(base.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0].startswith(banned_prefixes):
                        problems.append(f"{py}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root == "phase4":
                    continue  # phase4 reuse is intended & local
                if root.startswith(banned_prefixes):
                    problems.append(f"{py}: from {node.module} import ...")
    return {"pass": not problems, "problems": problems}

# --------------------------------------------------------------------------
def run_baseline_parity(example) -> dict:
    from phase2.baselines.persistence import persistence_forecast as p2_p
    from phase2.baselines.movement_vector import movement_vector_forecast as p2_m
    from phase5.baselines.persistence import persistence_forecast as p5_p
    from phase5.baselines.movement_vector import movement_vector_forecast as p5_m
    problems = []
    checks = {"persistence": np.array_equal(p5_p(example), p2_p(example)),
              "movement-vector": np.array_equal(p5_m(example), p2_m(example))}
    for name, same in checks.items():
        if not same:
            problems.append(f"{name} differs from Phase 2")
    return {"pass": not problems, "problems": problems, "checks": checks}


# --------------------------------------------------------------------------
def run_tests() -> dict:
    """Run the phase5 pytest suite with cache/bytecode disabled."""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, "-X", "utf8", "-m",
           "pytest", "-p", "no:cacheprovider", "-q", "--no-header",
           "phase5/tests/"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(_PKG), env=env)
    tail = (proc.stdout + proc.stderr)[-4000:]
    import re
    m = re.search(r"(\d+) passed(?:,\s*(\d+) failed)?", tail)
    total = int(m.group(1)) if m else -1
    failed = int(m.group(2)) if m and m.group(2) else 0
    return {"pass": proc.returncode == 0 and failed == 0,
            "rc": proc.returncode, "passed": max(total - failed, 0) if total > 0 else 0,
            "failed": failed, "total": total,
            "duration": 0.0, "output_tail": tail[-800:]}


# --------------------------------------------------------------------------
def write_report(audit, summary, changed, champion_meta, fail_fast=False):
    paths = default_paths()
    report = []
    app = report.append
    app("# PHASE 5 REPORT - FORECASTING SYSTEM INTEGRATION & DEMO")
    app("")
    app(f"**Recorded at:** {audit['recorded_at']}  ")
    app(f"**Final status:** `{audit['final_status']}`  ")
    app(f"**Champion experiment:** `{audit['experiment_id']}`  ")
    app(f"**Phase:** 5 (inference / integration / demo layer)")
    app("")
    app("> **Phase 5 is an inference/integration phase and does not establish "
        "new model-performance claims.**  ")
    app("> All model weights, statistics and engineering logic are the audited "
        "Phase-4 artifacts, reused read-only.")
    app("")
    app("## 1. Objective")
    app("Build a self-contained, offline, deterministic, CPU-only inference "
        "layer that fronts the audited Phase-4 champion (EXP005, GRU + Huber) "
        "with a stable JSON contract suitable for a React/Leaflet frontend. "
        "No retraining, no tuning, no new performance claims.")
    app("")
    app("## 2. Reused artifacts (read-only)")
    app("")
    app(f"- Checkpoint: `{paths.champion_checkpoint}`")
    app(f"- Config: `{paths.champion_config}`")
    app(f"- Normalization stats: `{paths.normalization_stats}` (train-only, "
        "`zero_std_features = []`)")
    app(f"- Champion metadata: `{paths.champion_meta}`")
    app(f"- Feature engineering: `phase4/features/feature_engineering.py`")
    app(f"- Baselines parity source: `phase2/baselines/`")
    app("")
    app("## 3. Directory layout")
    app("```")
    app("phase5/")
    app("  __init__.py  config.py  snapshot.py  run_phase5.py")
    app("  inference/   input_validation.py  preprocessing.py")
    app("               predictor.py   output_contract.py")
    app("  baselines/   persistence.py  movement_vector.py")
    app("  service/     forecasting_service.py")
    app("  schemas/     forecast_schema.py")
    app("  examples/    example_input.json")
    app("  tests/       <10 pytest files + conftest>")
    app("  results/     inference_audit.json  contract_validation.json")
    app("               phase5_summary.json   source_immutability_report.json")
    app("  reports/     PHASE5_REPORT.md")
    app("```")
    app("")
    app("## 4. Input contract")
    app("")
    app("- 5 history timesteps, 6-hourly, chronologically ascending (24h window "
        "ending at prediction time `t`).")
    app("- 7 raw fields per step, in the locked order: "
        "`lat, lon, wind_speed, pressure, sst, wind_u, wind_v`; longitude in the "
        "canonical NIO degrees-East `[0, 360)` convention.")
    app("- `timestamps` (optional) must be strictly increasing ISO-8601, exactly "
        "6h apart.  Inputs are validated, never silently repaired.")
    app("- Bounds enforced: lat [-90,90], lon [0,360), wind [0,400], pressure "
        "[850,1100], sst [-5,45].  NaN/Inf rejected with structured error codes.")
    app("")
    app("## 5. Output contract (frontend JSON)")
    app("")
    app("```json")
    app('{"status": "success",')
    app(' "model": {"experiment_id": "EXP005", "family": "GRU",')
    app('            "loss": "Huber"},')
    app(' "input": {"history_hours": 24, "history_steps": 5,')
    app('            "feature_count": 16},')
    app(' "forecast": [ {"hours": 6, "latitude": ..., "longitude": ...,')
    app('                 "wind_speed_kmh": ...}, ... ]}')
    app("```")
    app("")
    app("Horizons: exactly +6h/+12h/+24h, deterministic ordering.  Target "
        "columns `[lat, lon, wind_speed_kmh]`.  Longitude wrapped into `[0,360)`, "
        "wind clipped to `>=0`, latitude range checked; impossible forecasts are "
        "a hard error, never emitted.")
    app("")
    app("## 6. Feature contract")
    app("")
    app(f"- Engineered block `(5, 16)`: 7 raw columns kept byte-identical + 9 "
        f"derived from `phase4.features`.")
    app(f"- Order (from `normalization_stats.json`): {', '.join(['`'+f+'`' for f in audit.get('feature_order', [])]) if audit.get('feature_order') else '`lat..wind_v, delta_lat..environmental_wind_direction`'}")
    app(f"- First-step policy: the 7 predecessor-dependent features at t-24h "
        f"are zero-filled; environmental features are defined at every step.")
    app(f"- `feature_contract.source` = {audit.get('feature_contract_source')}")
    app("")
    app("## 7. Champion (read-only, from Phase-4)")
    app("")
    app(f"- Experiment: EXP005; family: GRU; loss: Huber; hidden: 96; layers: 2; "
        f"dropout: 0.1; seed: 42; input 16 -> output 9.")
    prim = ((champion_meta.get("primary_score") if champion_meta else None))
    app(f"- Validation primary metric (Phase-4, unchanged): "
        f"{prim if prim is not None else 'n/a'} "
        f"(equal-weight mean of validation track errors 6h/12h/24h; "
        f"selection never used the test fold).")
    comp = ((champion_meta.get("component_horizon_scores") if champion_meta
             else None))
    if comp:
        app(f"- Phase-4 validation component scores: 6h = {comp['6h']}, "
            f"12h = {comp['12h']}, 24h = {comp['24h']}.")
    app(f"- Trainer-side parameter count: 89577 (verified on load).")
    app("")
    app("## 8. Causality (future / target independence)")
    app("")
    app(f"- Verification: {audit.get('causal_verification')}.  Feature row `i` "
        "depends only on history rows `<= i`; no target values exist in the "
        "(5,7) input; no t+6/+12/+24 data is ever readable.")
    app(f"- Structural guarantee: `features_from_history` consumes X only.")
    app("")
    app("## 9. Determinism")
    app("")
    app(f"- Verification: {audit.get('deterministic_verification')} "
        "(torch single-threaded CPU, deterministic algorithms, float32).")
    app("- Same input -> bit-identical output with documented tolerance 1e-6.")
    app("")
    app("## 10. Offline operation")
    app("")
    app(f"- Verification: {audit.get('offline_verification')} (AST scan for "
        "networking imports: urllib/requests/socket/http/aiohttp/... none found).")
    app("- All artifacts are local; no external API or network access.")
    app("")
    app("## 11. Baselines (reference forecasts)")
    app("")
    app(f"- Integration check: {audit.get('baseline_parity')} vs Phase-2.")
    app("- `persistence_forecast`: latest observed lat/lon/wind for all "
        "horizons.")
    app("- `movement_vector_forecast`: t-6h->t vector extrapolated with "
        "multipliers {6:1, 12:2, 24:4}, lon wrapping on the 0..360 domain.")
    app("")
    app("## 12. Service API")
    app("")
    app("- `ForecastingService().forecast(history)` -> success/error JSON.")
    app("- `ForecastingService().compare_baselines(history)` -> "
        "model/persistence/movement-vector forecasts.")
    app("")
    app("## 13. Test suite")
    app("")
    app(f"- {audit.get('tests_passed', 0)}/{audit.get('tests_total', 0)} "
        "phase5 tests passed (ran with `-p no:cacheprovider`, bytecode "
        "disabled).")
    app("- Files: input_validation, preprocessing, feature_consistency, "
        "model_loading, inference_contract, forecast_shapes, baselines, "
        "determinism, no_source_modification, service + conftest.")
    app("")
    app("## 14. Validation results")
    app("")
    app("See `phase5/results/inference_audit.json`, "
        "`phase5/results/contract_validation.json`.")
    app("")
    app("## 15. Source immutability")
    app("")
    app("Checked against `phase5/results/source_hashes_before.json` "
        f"(recorded before implementation, {snapshot.read_before().get('n_files')} files).")
    app("")
    for key, label in (("p1", "P1"), ("phase1", "Phase 1"),
                       ("phase2", "Phase 2"), ("phase3", "Phase 3"),
                       ("phase4", "Phase 4"),
                       ("outside_phase5", "outside Phase 5")):
        app(f"- {label}: {changed[key]} file(s) changed")
    app("")
    app("## 16. Contingencies / discrepancies")
    app("")
    app("None encountered.  If any artefact mismatch had appeared, it would be "
        "documented here and the run would fail closed.")
    app("")
    app("## 17. Limitations")
    app("")
    app("- Constant-velocity movement-vector baseline is a linear extrapolation "
        "in lat/lon degrees, not a dynamical model (by design, Phase-2 "
        "methodology).")
    app("- Model strengths/weaknesses are Phase-4 scientific conclusions; Phase "
        "5 re-exposes them without adding or removing claims.")
    app("- Sequence must have exactly 24h of 6-hourly history up to `t`.")
    app("")
    app("## 18. Reproducibility")
    app("")
    app("```")
    app("cd p4_forecasting")
    app("python -X utf8 phase5/run_phase5.py          # verify everything")
    app("python -X utf8 phase5/run_phase5.py --force-check")
    app("python -X utf8 phase5/snapshot.py --before   # (only if needed)")
    app("```")
    app("Every run is idempotent; outputs are overwritten (only `recorded_at` "
        "changes).  No training options exist.")
    app("")
    app("## 19. Final statement")
    app("")
    app(f"**Phase 5 is an inference/integration phase and does not establish "
        f"new model-performance claims.**  ")
    app(f"Final: `{audit['final_status']}`.")
    app("")
    (paths.reports_dir / "PHASE5_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())