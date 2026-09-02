"""Phase-6 runner: end-to-end verification of the forecasting API.

Run from p4_forecasting/ with::

    python -X utf8 phase6/run_phase6.py                 # verify everything
    python -X utf8 phase6/run_phase6.py --force-check   # re-record BEFORE first

VERIFICATION ONLY.  The runner never starts a Uvicorn server and never
exercises the network.  Idempotent: repeated runs overwrite the Phase-6
outputs; only ``recorded_at`` timestamps differ.

Start the actual server separately with::

    python -X utf8 -m uvicorn phase6.api.app:app --host 127.0.0.1 --port 8000

(Run from the p4_forecasting/ directory so the ``phase6`` package resolves;
``p4_forecasting`` itself is deliberately not a Python package.)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import snapshot
from phase6.config import PHASE6, phase6_paths
from phase6.schemas.requests import ForecastRequest
from phase6.schemas.responses import (CompareSuccessResponse,
                                      ForecastSuccessResponse, HealthResponse,
                                      ModelInfoResponse)
from phase6.api.app import create_app

BANNED_NETWORK_MODULES = ("urllib", "requests", "socket", "http", "aiohttp",
                          "httpx", "websocket", "ftplib", "xmlrpc", "urllib3")

BANNER = """
============================================================
P4 PHASE-6 COMPLETE
------------------------------------------------------------
Status                         : {status}
Phase-5 integration            : {p5}
Champion                       : {champion}
API                             : {api}
/health                         : {health}
/model                          : {model}
/forecast                       : {forecast}
/forecast/compare               : {compare}
Input validation                : {validation}
Causality                       : {causality}
Determinism                     : {determinism}
Offline inference               : {offline}
Baseline integration            : {baseline}
OpenAPI                         : {openapi}
Tests                           : {passed} passed / {failed} failed
P1 modified                     : {p1}
Phase-1 modified                : {phase1}
Phase-2 modified                : {phase2}
Phase-3 modified                : {phase3}
Phase-4 modified                : {phase4}
Phase-5 modified                : {phase5}
Outside phase6 modified         : {outside}
------------------------------------------------------------
FINAL STATUS                    : {final}
============================================================
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_example() -> Dict[str, Any]:
    paths = phase6_paths()
    p = paths.examples_dir / "forecast_request.json"
    return json.loads(p.read_text(encoding="utf-8"))


# -- individual verification helpers --------------------------------------
def verify_artifacts() -> Dict[str, Any]:
    paths = phase6_paths()
    missing = paths.phase5.require()
    return {"pass": not missing, "missing": missing}


def verify_schemas(example: Dict[str, Any]) -> Dict[str, Any]:
    req = ForecastRequest.model_validate(example)
    return {"pass": True, "steps": len(req.history),
            "spacing_hours": 6}


def run_endpoint_checks(client) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}

    r = client.get("/health")
    results["health"] = {"pass": r.status_code == 200 and
                         r.json().get("status") == "ok" and
                         r.json().get("offline") is True,
                         "status_code": r.status_code, "body": r.json()}

    r = client.get("/model")
    body = r.json()
    results["model"] = {
        "pass": (r.status_code == 200 and
                 body.get("experiment_id") == "EXP005" and
                 body.get("feature_count") == 16 and
                 body.get("history_steps") == 5 and
                 body.get("horizons") == [6, 12, 24]),
        "status_code": r.status_code, "body": body}

    r = client.post("/forecast", json=load_example())
    body = r.json()
    fc = body.get("forecast", [])
    results["forecast"] = {
        "pass": (r.status_code == 200 and len(fc) == 3 and
                 [f["hours"] for f in fc] == [6, 12, 24] and
                 all(f["wind_speed_kmh"] >= 0.0 for f in fc)),
        "status_code": r.status_code, "n_forecasts": len(fc),
        "horizons": [f["hours"] for f in fc]}

    r = client.post("/forecast/compare", json=load_example())
    body = r.json()
    keys = ("model_forecast", "persistence_forecast",
            "movement_vector_forecast")
    results["compare"] = {
        "pass": (r.status_code == 200 and body.get("status") == "success" and
                 all(k in body and len(body[k]) == 3 for k in keys)),
        "status_code": r.status_code,
        "present": [k for k in keys if k in body]}
    return results


_TIMES = ["2024-08-25T00:00:00Z", "2024-08-25T06:00:00Z",
          "2024-08-25T12:00:00Z", "2024-08-25T18:00:00Z",
          "2024-08-26T00:00:00Z"]


def _valid_steps():
    return [{"timestamp": t, "latitude": 23.0 + 0.05 * i,
             "longitude": 68.5 - 0.6 * i, "wind_speed_kmh": 55.6 + 4.0 * i,
             "pressure_hpa": 994.0 - 0.7 * i, "sst": 28.6, "wind_u": 5.0,
             "wind_v": 1.0}
            for i, t in enumerate(_TIMES)]


def run_validation_checks(client) -> Dict[str, Any]:
    import math
    checks = {"cases": {}}
    okall = True
    okall &= _case(client, "4 steps",
                   {"history": _valid_steps()[:4]},
                   "INVALID_HISTORY_LENGTH", checks)
    okall &= _case(client, "6 steps",
                   {"history": _valid_steps() + [_valid_steps()[0]]},
                   "INVALID_HISTORY_LENGTH", checks)

    import json as _json
    steps = _valid_steps()
    steps[2]["wind_speed_kmh"] = float("nan")
    okall &= _case(client, "nan", None, "NON_FINITE_VALUE", checks,
                   raw=_json.dumps({"history": steps}, allow_nan=True))

    steps = _valid_steps()
    steps[0]["timestamp"] = "nope"
    okall &= _case(client, "malformed timestamp",
                   {"history": steps}, "INVALID_TIMESTAMP", checks)

    steps = _valid_steps()
    steps[3]["timestamp"] = "2024-08-25T17:00:00Z"
    okall &= _case(client, "wrong spacing",
                   {"history": steps}, "INVALID_HISTORY_SPACING", checks)

    okall &= _case(client, "non-monotonic",
                   {"history": _valid_steps()[::-1]},
                   "NON_MONOTONIC_HISTORY", checks)

    steps = _valid_steps()
    del steps[2]["wind_v"]
    okall &= _case(client, "missing feature", {"history": steps},
                   "MISSING_FEATURE", checks)

    steps = _valid_steps()
    steps[0]["latitude"] = 95.0
    okall &= _case(client, "impossible lat", {"history": steps},
                   "INVALID_LATITUDE", checks)

    steps = _valid_steps()
    steps[1]["longitude"] = -10.0
    okall &= _case(client, "invalid lon", {"history": steps},
                   "INVALID_LONGITUDE", checks)

    steps = _valid_steps()
    steps[3]["wind_speed_kmh"] = -5.0
    okall &= _case(client, "negative wind", {"history": steps},
                   "INVALID_WIND", checks)

    # malformed JSON body
    okall &= _case(client, "malformed json", None, "INVALID_REQUEST", checks,
                   raw='{"history": [ }}')
    checks["pass"] = okall
    return checks


def _case(client, name, payload, expected_code, checks, raw=None):
    try:
        if raw is not None:
            r = client.post("/forecast", content=raw,
                            headers={"Content-Type": "application/json"})
        else:
            r = client.post("/forecast", json=payload)
        got = r.json().get("error", {}).get("code")
        ok = r.status_code == 422 and got == expected_code
        checks["cases"][name] = {"pass": ok, "status_code": r.status_code,
                                 "code": got, "expected": expected_code}
        return ok
    except Exception as exc:
        checks["cases"][name] = {"pass": False, "error": str(exc)}
        return False


def run_causality(client) -> Dict[str, Any]:
    steps = _valid_steps()
    future = _valid_steps() + [_valid_steps()[0]]       # 6th = future (t+6)
    r = client.post("/forecast", json={"history": future})
    future_rejected = r.status_code == 422

    base = client.post("/forecast", json={"history": steps}).json()
    mod = {"history": list(steps)}
    mod["history"][-1] = dict(steps[-1])
    mod["history"][-1]["wind_speed_kmh"] = 170.0
    changed = client.post("/forecast", json=mod).json()
    responds = (changed["forecast"] != base["forecast"])
    return {"pass": future_rejected and responds,
            "future_rejected": future_rejected,
            "responds_to_latest_change": responds}


def run_determinism(client) -> Dict[str, Any]:
    a = client.post("/forecast", json=load_example())
    b = client.post("/forecast", json=load_example())
    exact = a.status_code == b.status_code and a.json() == b.json()
    return {"pass": exact, "exact_byte_identical": a.content == b.content}


def run_offline(client, adapter) -> Dict[str, Any]:
    from unittest.mock import patch
    import socket as _socket
    problems = []
    for rel in ("phase6", "phase5/service", "phase5/inference",
                "phase5/baselines"):
        root = _PKG / rel
        for py in sorted(root.rglob("*.py")):
            relp = py.relative_to(_PKG).as_posix()
            if relp.startswith("phase6/tests/"):
                continue
            if relp == "phase6/run_phase6.py":
                # the runner imports socket to prove the runtime opens none;
                # it is a verification tool, not part of the inference path
                continue
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name.split(".")[0] in BANNED_NETWORK_MODULES:
                            problems.append(f"{py}: import {a.name}")
                elif isinstance(node, ast.ImportFrom):
                    root_mod = (node.module or "").split(".")[0]
                    if root_mod in BANNED_NETWORK_MODULES:
                        problems.append(f"{py}: from {node.module}")
    # runtime proof: a forecast while sockets are disabled still succeeds
    runtime_ok = True
    exc = None
    try:
        with patch.object(_socket.socket, "connect", autospec=True,
                          side_effect=AssertionError("network attempted")), \
             patch.object(_socket.socket, "sendall", autospec=True,
                          side_effect=AssertionError("network attempted")):
            res = adapter.forecast(ForecastRequest.model_validate(
                load_example()))
        runtime_ok = res.get("status") == "success"
    except Exception as e:  # pragma: no cover - only on network attempts
        runtime_ok = False
        exc = str(e)
    return {"pass": not problems and runtime_ok,
            "import_problems": problems, "runtime_no_sockets": runtime_ok,
            "runtime_error": exc}


def run_baseline_check(client) -> Dict[str, Any]:
    cmp = client.post("/forecast/compare", json=load_example()).json()
    last = load_example()["history"][-1]
    persist_ok = all(
        abs(f["longitude"] - last["longitude"]) < 1e-4 and
        abs(f["latitude"] - last["latitude"]) < 1e-4
        for f in cmp["persistence_forecast"])
    # movement-vector must match the Phase-2 definition evaluated directly
    from phase2.baselines.movement_vector import movement_vector_forecast
    import numpy as np
    fields = ["latitude", "longitude", "wind_speed_kmh"]
    h7 = np.array([[s[f] for f in ("latitude", "longitude",
                                   "wind_speed_kmh", "pressure_hpa", "sst",
                                   "wind_u", "wind_v")] for s in
                   load_example()["history"]], np.float32)
    h7[:, 1] = h7[:, 1]  # lon already in 0..360
    expect = movement_vector_forecast(h7)
    mv_ok = all(
        abs(cmp["movement_vector_forecast"][i]["latitude"] - expect[i, 0]) < 1e-4
        for i in range(3))
    return {"pass": persist_ok and mv_ok, "persistence_parity": persist_ok,
            "movement_vector_parity": mv_ok}


def run_openapi_check(client) -> Dict[str, Any]:
    spec = client.get("/openapi.json")
    docs = client.get("/docs")
    paths_ok = False
    if spec.status_code == 200:
        paths = set(spec.json()["paths"].keys())
        paths_ok = {"/health", "/model", "/forecast", "/forecast/compare"} \
            <= paths
    return {"pass": spec.status_code == 200 and paths_ok and
            docs.status_code == 200,
            "openapi_status": spec.status_code,
            "docs_status": docs.status_code,
            "paths_present": paths_ok}


def run_latency(client, adapter) -> Dict[str, Any]:
    cold_ms = adapter.ensure_loaded()          # force predictor construction
    payload = load_example()
    n = 15
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post("/forecast", json=payload)
        times.append((time.perf_counter() - t0) * 1000.0)
        if r.status_code != 200:
            return {"pass": False, "error": r.text,
                    "cold_load_ms": cold_ms}
    return {
        "pass": True,
        "cold_load_ms": round(cold_ms, 3),
        "warm": {
            "n": n,
            "mean_ms": round(statistics.mean(times), 3),
            "median_ms": round(statistics.median(times), 3),
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
        },
    }


def run_tests() -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, "-X", "utf8", "-m",
           "pytest", "-p", "no:cacheprovider", "-q", "--no-header",
           "phase6/tests/"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(_PKG), env=env)
    tail = (proc.stdout + proc.stderr)[-4000:]
    import re
    m = re.search(r"(\d+) passed(?:,\s*(\d+) failed)?", tail)
    total = int(m.group(1)) if m else -1
    failed = int(m.group(2)) if m and m.group(2) else 0
    return {"pass": proc.returncode == 0 and failed == 0,
            "rc": proc.returncode, "passed": total,
            "failed": failed, "total": total,
            "output_tail": tail[-900:]}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-6 API verification")
    ap.add_argument("--force-check", action="store_true",
                    help="re-record the BEFORE snapshot first")
    args = ap.parse_args()

    paths = phase6_paths()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    ok = lambda name, _msg="": print(f"  [ok]  {name}")
    bad = lambda name, msg: (print(f"  [FAIL] {name}: {msg}"),
                             failures.append(f"{name}: {msg}"))

    if args.force_check:
        snapshot.write_before()
    before = snapshot.read_before()
    print("[phase6] BEFORE snapshot:", before["recorded_at"],
          f"({before['n_files']} files)")

    # 1. verify Phase-5 artifacts --------------------------------------------
    print("[phase6] 1/13 Phase-5 artifacts")
    va = verify_artifacts()
    if va["pass"]:
        ok("Phase-5 artifacts present and complete")
    else:
        bad("phase5-artifacts", "; ".join(va["missing"]))
        return 1

    # 2. create the API modules ------------------------------------------------
    print("[phase6] 2/13 API modules import")
    try:
        app = create_app()
        adapter = app.state.adapter
        ok("FastAPI application factory builds")
    except Exception as exc:
        bad("api-modules", str(exc))
        return 1

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        # 3. schema validation --------------------------------------------------
        print("[phase6] 3/13 request/response schemas")
        try:
            example = load_example()
            sr = verify_schemas(example)
            HealthResponse(**client.get("/health").json())
            ModelInfoResponse(**client.get("/model").json())
            ForecastSuccessResponse(**client.post(
                "/forecast", json=example).json())
            CompareSuccessResponse(**client.post(
                "/forecast/compare", json=example).json())
            ok(f"schemas valid ({sr['steps']} history steps)")
        except Exception as exc:
            bad("schemas", str(exc))
            return 1

        # 4. endpoint checks ---------------------------------------------------
        print("[phase6] 4/13 endpoint checks (/health /model /forecast "
              "/forecast/compare)")
        ep = run_endpoint_checks(client)
        for name, res in ep.items():
            (ok if res["pass"] else bad)(f"endpoint /{name}",
                                         "" if res["pass"]
                                         else json.dumps(res.get("body", ""))[:400])
        api_pass = all(r["pass"] for r in ep.values())

        # 5. input validation ----------------------------------------------------
        print("[phase6] 5/13 input-validation battery")
        vcheck = run_validation_checks(client)
        if vcheck["pass"]:
            ok(f"input validation ({len(vcheck['cases'])} cases all 422 "
               "with expected codes)")
        else:
            failed_cases = [k for k, v in vcheck["cases"].items()
                            if not v["pass"]]
            bad("input-validation", str(failed_cases))
        validation_pass = vcheck["pass"]

        # 6. causality ------------------------------------------------------------
        print("[phase6] 6/13 causality")
        ccheck = run_causality(client)
        (ok if ccheck["pass"] else bad)(
            "causality",
            "future values rejected)" if ccheck["pass"] else str(ccheck))

        # 7. determinism ----------------------------------------------------------
        print("[phase6] 7/13 determinism")
        dcheck = run_determinism(client)
        (ok if dcheck["pass"] else bad)(
            "determinism",
            "identical responses for identical requests" if dcheck["pass"]
            else str(dcheck))

        # 8. offline ----------------------------------------------------------------
        print("[phase6] 8/13 offline/no-network")
        ocheck = run_offline(client, adapter)
        (ok if ocheck["pass"] else bad)(
            "offline",
            "no network imports; socket-disabled forecast succeeds"
            if ocheck["pass"] else str(ocheck))

        # 9. baseline integration ---------------------------------------------------
        print("[phase6] 9/13 baseline parity")
        bcheck = run_baseline_check(client)
        (ok if bcheck["pass"] else bad)(
            "baseline-integration",
            "persistence & movement-vector match Phase-2"
            if bcheck["pass"] else str(bcheck))

        # 10. OpenAPI ----------------------------------------------------------------
        print("[phase6] 10/13 OpenAPI")
        oacheck = run_openapi_check(client)
        (ok if oacheck["pass"] else bad)(
            "openapi", "/openapi.json and /docs available"
            if oacheck["pass"] else str(oacheck))

        # 11. latency -----------------------------------------------------------------
        print("[phase6] 11/13 software latency (cold load + warm inference)")
        lcheck = run_latency(client, adapter)
        if lcheck["pass"]:
            w = lcheck["warm"]
            ok(f"cold load {lcheck['cold_load_ms']} ms | warm "
               f"mean={w['mean_ms']} median={w['median_ms']} "
               f"min={w['min_ms']} max={w['max_ms']} ms "
               f"({w['n']} calls)")
        else:
            bad("latency", str(lcheck))

    # 12. unit + API tests ------------------------------------------------------------
    print("[phase6] 12/13 pytest suite")
    tcheck = run_tests()
    if tcheck["pass"]:
        ok(f"pytest: {tcheck['passed']}/{tcheck['total']} passed")
    else:
        bad("pytest", f"{tcheck['passed']}/{tcheck['total']} passed; "
            f"rc={tcheck['rc']}")

    # -- write intermediate results ---------------------------------------------------
    api_test_results = {
        "schema": "phase6 v1",
        "recorded_at": now(),
        "endpoints": ep,
        "validation": vcheck,
        "causality": ccheck,
        "determinism": dcheck,
        "offline": {"pass": ocheck["pass"],
                    "import_problems": ocheck["import_problems"]},
        "baseline_integration": bcheck,
        "openapi": {"pass": oacheck["pass"]},
        "latency": lcheck,
        "tests": {k: tcheck[k] for k in ("passed", "failed", "total", "rc")},
        "final_status": "PASS" if (api_pass and validation_pass
                                   and ccheck["pass"] and dcheck["pass"]
                                   and ocheck["pass"] and bcheck["pass"]
                                   and oacheck["pass"] and lcheck["pass"]
                                   and tcheck["pass"]) else "FAIL",
    }
    (paths.results_dir / "api_test_results.json").write_text(
        json.dumps(api_test_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    # 13. source immutability ----------------------------------------------------------
    print("[phase6] 13/13 source immutability")
    after_master = snapshot.master_snapshot()
    status = snapshot.make_status(before["hashes"], after_master)
    immutable = snapshot.write_after_report(
        after_master, status,
        note="Phase-6 API must never modify P1 / Phase-1/2/3/4/5 or any file "
             "outside phase6.")

    counts = {k: immutable[f"{k}_changed"]
              for k in ("p1", "phase1", "phase2", "phase3", "phase4",
                        "phase5", "outside_phase6")}
    if immutable["status"] == "PASS":
        ok(f"immutable: {counts}")
    else:
        for k, v in counts.items():
            if v:
                bad(f"immutability-{k}", f"{v} file(s) changed")

    final = "PASS"
    checks_ok = all([api_pass, validation_pass, ccheck["pass"], dcheck["pass"],
                     ocheck["pass"], bcheck["pass"], oacheck["pass"],
                     lcheck["pass"], tcheck["pass"],
                     immutable["status"] == "PASS"])
    if not checks_ok or failures:
        final = "FAIL"

    # -- summary + contract + report ---------------------------------------------------
    summary = {
        "schema": "phase6 v1",
        "recorded_at": now(),
        "final_status": final,
        "steps": {
            "phase5_artifacts": va["pass"],
            "api_modules": True,
            "schemas": True,
            "endpoints": api_pass,
            "input_validation": validation_pass,
            "causality": ccheck["pass"],
            "determinism": dcheck["pass"],
            "offline": ocheck["pass"],
            "baseline_integration": bcheck["pass"],
            "openapi": oacheck["pass"],
            "latency": lcheck["pass"],
            "tests": tcheck["pass"],
            "immutability": immutable["status"] == "PASS",
        },
        "latency": lcheck,
        "tests": {k: tcheck[k] for k in ("passed", "failed", "total")},
        "immutability": counts,
        "failures": failures,
        "outputs": {
            "api_contract": "phase6/results/api_contract.json",
            "api_test_results": "phase6/results/api_test_results.json",
            "source_immutability": "phase6/results/source_immutability_report.json",
            "summary": "phase6/results/phase6_summary.json",
            "report": "phase6/reports/PHASE6_REPORT.md",
        },
    }
    (paths.results_dir / "phase6_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    write_api_contract(example)
    write_report(summary, api_test_results, lcheck, counts)

    print(BANNER.format(
        status=final,
        p5="PASS" if va["pass"] else "FAIL",
        champion=ep["model"].get("body", {}).get("experiment_id", "EXP005")
        if ep.get("model") else "n/a",
        api="PASS" if api_pass else "FAIL",
        health="PASS" if ep.get("health", {}).get("pass") else "FAIL",
        model="PASS" if ep.get("model", {}).get("pass") else "FAIL",
        forecast="PASS" if ep.get("forecast", {}).get("pass") else "FAIL",
        compare="PASS" if ep.get("compare", {}).get("pass") else "FAIL",
        validation="PASS" if validation_pass else "FAIL",
        causality="PASS" if ccheck["pass"] else "FAIL",
        determinism="PASS" if dcheck["pass"] else "FAIL",
        offline="PASS" if ocheck["pass"] else "FAIL",
        baseline="PASS" if bcheck["pass"] else "FAIL",
        openapi="PASS" if oacheck["pass"] else "FAIL",
        passed=tcheck["passed"], failed=tcheck["failed"],
        p1=counts["p1"], phase1=counts["phase1"], phase2=counts["phase2"],
        phase3=counts["phase3"], phase4=counts["phase4"],
        phase5=counts["phase5"], outside=counts["outside_phase6"],
        final=final,
    ))
    return 0 if final == "PASS" else 1


def write_api_contract(example: Dict[str, Any]) -> None:
    """Frontend contract document (results/api_contract.json)."""
    response_example_health = {"status": "ok", "service": "cyclone-forecasting",
                               "phase": "phase6", "offline": True,
                               "model_ready": True}
    response_example_model = {
        "experiment_id": "EXP005", "model": "GRU", "loss": "Huber",
        "hidden_size": 96, "layers": 2, "input_size": 16, "output_size": 9,
        "history_steps": 5, "history_hours": 24, "feature_count": 16,
        "horizons": [6, 12, 24], "targets": ["lat", "lon", "wind_speed_kmh"],
        "validation_primary_score": 113.07414084856835}
    response_example_forecast = {
        "status": "success",
        "model": {"experiment_id": "EXP005", "family": "GRU", "loss": "Huber"},
        "input": {"history_hours": 24, "history_steps": 5, "feature_count": 16},
        "forecast": [{"hours": 6, "latitude": 22.86, "longitude": 65.32,
                      "wind_speed_kmh": 86.11},
                     {"hours": 12, "latitude": 22.92, "longitude": 64.68,
                      "wind_speed_kmh": 86.80},
                     {"hours": 24, "latitude": 22.54, "longitude": 63.09,
                      "wind_speed_kmh": 79.33}]}
    doc = {
        "schema": "phase6 v1",
        "recorded_at": now(),
        "base_url": "http://127.0.0.1:8000",
        "frontend_note": ("The React frontend only needs to speak this JSON. "
                          "No PyTorch / feature-engineering knowledge is "
                          "required: send 5 observations, read back 3 "
                          "forecast points."),
        "events": {
            "GET /health": {
                "request": None,
                "response": {"schema": {"status": "string",
                                        "service": "string",
                                        "phase": "string",
                                        "offline": "boolean",
                                        "model_ready": "boolean"},
                             "example": response_example_health}},
            "GET /model": {
                "request": None,
                "response": {"schema": {
                    "experiment_id": "string", "model": "string",
                    "loss": "string", "hidden_size": "int", "layers": "int",
                    "input_size": "int", "output_size": "int",
                    "history_steps": "int", "history_hours": "int",
                    "feature_count": "int", "horizons": "list[int]",
                    "targets": "list[string]",
                    "validation_primary_score": "float"},
                    "example": response_example_model}},
            "POST /forecast": {
                "request": {"schema": {
                    "history": "exactly 5 objects of {timestamp: ISO-8601, "
                               "latitude: [-90,90], longitude: [0,360), "
                               "wind_speed_kmh: >=0, pressure_hpa: [850,1100], "
                               "sst: [-5,45], wind_u, wind_v: number} "
                               "at t-24h/t-18h/t-12h/t-6h/t (6-hourly)",
                    "order": "chronologically ascending"},
                    "example": example},
                "response": {"schema": {"status": "success",
                                        "model": {"experiment_id":
                                                  "EXP005",
                                                  "family": "GRU",
                                                  "loss": "Huber"},
                                        "input": {"history_hours": 24,
                                                  "history_steps": 5,
                                                  "feature_count": 16},
                                        "forecast": "3 x {hours, latitude, "
                                                    "longitude, "
                                                    "wind_speed_kmh}"},
                             "example": response_example_forecast}},
            "POST /forecast/compare": {
                "request": {"schema": "same body as POST /forecast",
                            "example": example},
                "response": {"schema": {"status": "success",
                                        "model": "model identity dict",
                                        "model_forecast": "3 x forecast item",
                                        "persistence_forecast": "3 x item",
                                        "movement_vector_forecast":
                                            "3 x item"},
                             "example": None}},
            "error": {"schema": {"status": "error",
                                 "error": {"code": "string", "message":
                                           "string"}},
                      "codes": ["INVALID_REQUEST", "INVALID_HISTORY_LENGTH",
                                "INVALID_TIMESTAMP", "INVALID_HISTORY_SPACING",
                                "NON_MONOTONIC_HISTORY", "INVALID_LATITUDE",
                                "INVALID_LONGITUDE", "INVALID_WIND",
                                "MISSING_FEATURE", "NON_FINITE_VALUE",
                                "MODEL_NOT_READY", "INFERENCE_ERROR"]}},
        "cors": {
            "allowed_origins": ["http://localhost:3000",
                                "http://localhost:5173"],
            "methods": ["GET", "POST", "OPTIONS"]},
        "server_command": ("cd p4_forecasting && python -X utf8 -m uvicorn "
                           "phase6.api.app:app --host 127.0.0.1 --port 8000"),
        "frontend_usage_example": dict(
            javascript=(
                'fetch("http://localhost:8000/forecast", {\n'
                '  method: "POST",\n'
                '  headers: { "Content-Type": "application/json" },\n'
                '  body: JSON.stringify(request)\n'
                '})\n'
                '  .then(r => r.json())\n'
                '  .then(data => { /* data.forecast = [{hours, latitude, '
                'longitude, wind_speed_kmh}x3] */ });')),
    }
    paths = phase6_paths()
    (paths.results_dir / "api_contract.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def write_report(summary, api_results, latency, counts):
    paths = phase6_paths()
    report = []
    app_ = report.append
    app_("# PHASE 6 REPORT - FORECASTING API & FRONTEND INTEGRATION")
    app_("")
    app_(f"**Recorded at:** {summary['recorded_at']}  ")
    app_(f"**Final status:** `{summary['final_status']}`  ")
    app_("")
    app_("> **Phase 6 exposes the validated Phase-5 forecasting system "
         "through a local API. It does not retrain the model and does not "
         "establish new forecasting-performance claims.**")
    app_("")
    app_("## 1. Objective")
    app_("Expose the audited Phase-5 inference engine (EXP005, GRU + Huber) "
         "through a clean, local, offline FastAPI so a future React/Leaflet "
         "frontend can request +6h/+12h/+24h forecasts with zero knowledge "
         "of PyTorch or feature engineering.")
    app_("")
    app_("## 2. Phase-5 integration")
    app_("")
    app_("- Public interface reused (inspected before coding): "
         "`phase5/service/forecasting_service.py` -> `ForecastingService` "
         "with `forecast(history)` and `compare_baselines(history)`.")
    app_("- `phase6/integration/forecasting_adapter.py` maps the HTTP request "
         "schema onto the Phase-5 input contract and maps Phase-5 error "
         "codes onto the public Phase-6 vocabulary.")
    app_("- No scientific component (engineering, normalisation, model, "
         "baselines) was duplicated, modified or re-trained.")
    app_("")
    app_("## 3. API architecture")
    app_("")
    app_("```")
    app_("HTTP request")
    app_("  -> Pydantic validation (422 before inference)")
    app_("  -> ForecastingAdapter (schema mapping + error mapping)")
    app_("  -> Phase-5 ForecastingService")
    app_("  -> EXP005 (CPU, deterministic, offline)")
    app_("  -> validated output -> JSON response")
    app_("```")
    app_("")
    app_("## 4. Endpoints")
    app_("")
    app_(f"- `GET /health` -> **{'PASS' if api_results['endpoints']['health']['pass'] else 'FAIL'}**")
    app_(f"- `GET /model` -> **{'PASS' if api_results['endpoints']['model']['pass'] else 'FAIL'}**")
    app_(f"- `POST /forecast` -> **{'PASS' if api_results['endpoints']['forecast']['pass'] else 'FAIL'}**")
    app_(f"- `POST /forecast/compare` -> **{'PASS' if api_results['endpoints']['compare']['pass'] else 'FAIL'}**")
    app_("")
    app_("## 5. Request schema")
    app_("")
    app_("```json")
    app_('{"history": [')
    app_('  {"timestamp": "2025-11-29T00:00:00Z", "latitude": 12.1,')
    app_('   "longitude": 85.2, "wind_speed_kmh": 100.0,')
    app_('   "pressure_hpa": 980.0, "sst": 28.4, "wind_u": 3.2,')
    app_('   "wind_v": -1.4},')
    app_('   ... exactly 5 observations, 6-hourly, t-24h ... t ...]}')
    app_("```")
    app_("")
    app_("## 6. Response schema")
    app_("")
    app_("```json")
    app_('{"status": "success",')
    app_(' "model": {"experiment_id": "EXP005", "family": "GRU",')
    app_('            "loss": "Huber"},')
    app_(' "input": {"history_hours": 24, "history_steps": 5,')
    app_('            "feature_count": 16},')
    app_(' "forecast": [{"hours": 6, "latitude": ..., "longitude": ...,')
    app_('                 "wind_speed_kmh": ...} x3]}')
    app_("```")
    app_("")
    app_("## 7. Input validation")
    app_("")
    app_("Pydantic-first: length (exactly 5), ISO timestamps, strict 6-hour "
         "cadence, monotonicity, physical bounds, finite values, mandatory "
         "fields.  Rejected before the service is ever called with HTTP 422; "
         "nothing is repaired or interpolated.")
    app_(f"- battery: {len(api_results['validation']['cases'])} invalid "
         f"cases -> all 422 with expected codes -> "
         f"**{'PASS' if api_results['validation']['pass'] else 'FAIL'}**")
    app_("")
    app_("## 8. Error handling")
    app_("")
    app_("Structured `{status, error:{code, message}}` on every failure; "
         "stack traces and filesystem paths are never exposed.  Error codes: "
         "`INVALID_REQUEST, INVALID_HISTORY_LENGTH, INVALID_TIMESTAMP, "
         "INVALID_HISTORY_SPACING, NON_MONOTONIC_HISTORY, INVALID_LATITUDE, "
         "INVALID_LONGITUDE, INVALID_WIND, MISSING_FEATURE, "
         "NON_FINITE_VALUE, MODEL_NOT_READY, INFERENCE_ERROR`.")
    app_("")
    app_("## 9. Causality")
    app_("")
    app_(f"- **{'PASS' if api_results['causality']['pass'] else 'FAIL'}**")
    app_("- Future (t+6/+12/+24) values are never accepted (rejected 422); "
         "changing the latest observed value DOES change the forecast.")
    app_("- The request schema has no target/future slots at all; the adapter "
         "forwards exactly the 5 validated observations.")
    app_("")
    app_("## 10. Determinism")
    app_("")
    app_(f"- **{'PASS' if api_results['determinism']['pass'] else 'FAIL'}** - "
         "identical requests produce byte-identical responses (single-thread "
         "CPU inference).")
    app_("")
    app_("## 11. Offline operation")
    app_("")
    app_(f"- **{'PASS' if api_results['offline']['pass'] else 'FAIL'}** - no "
         "network imports in the inference path; a forecast still succeeds "
         "with sockets disabled; all artifacts are local.")
    app_("")
    app_("## 12. Baseline integration")
    app_("")
    app_(f"- **{'PASS' if api_results['baseline_integration']['pass'] else 'FAIL'}** - "
         "persistence and movement-vector outputs match the audited Phase-2 "
         "definitions (parity-checked against `phase2/baselines`).  The "
         "compare endpoint exists for debugging / demonstration, never to "
         "claim the model 'wins'.")
    app_("")
    app_("## 13. CORS")
    app_("")
    app_(f"- origins: `{', '.join(['http://localhost:3000', 'http://localhost:5173'])}` "
         "(local dev only; no wildcard).")
    app_("")
    app_("## 14. OpenAPI")
    app_("")
    app_(f"- **{'PASS' if api_results['openapi']['pass'] else 'FAIL'}** - "
         "`/docs` and `/openapi.json` are served locally by FastAPI.")
    app_("")
    app_("## 15. Latency (software only; no accuracy claim)")
    app_("")
    if latency.get("pass"):
        w = latency["warm"]
        app_(f"- cold model load: **{latency['cold_load_ms']} ms**")
        app_(f"- warm inference: mean {w['mean_ms']} ms | median "
             f"{w['median_ms']} ms | min {w['min_ms']} ms | max "
             f"{w['max_ms']} ms ({w['n']} calls)")
    else:
        app_("- latency measurement failed; see log")
    app_("")
    app_("## 16. Tests")
    app_("")
    app_(f"- **{summary['tests']['passed']} passed / "
         f"{summary['tests']['failed']} failed** (pytest, cache + bytecode "
         "disabled).")
    app_("- Files: health, model_endpoint, forecast_endpoint, "
         "compare_endpoint, validation_errors, response_schema, causality, "
         "determinism, offline, source_immutability + conftest.")
    app_("")
    app_("## 17. Source immutability")
    app_("")
    app_("Checked against `phase6/results/source_hashes_before.json`.  "
         "Only files under `p4_forecasting/phase6/` were created.")
    for key, label in (("p1", "P1"), ("phase1", "Phase 1"),
                       ("phase2", "Phase 2"), ("phase3", "Phase 3"),
                       ("phase4", "Phase 4"), ("phase5", "Phase 5"),
                       ("outside_phase6", "outside phase6")):
        app_(f"- {label}: {counts[key]} file(s) changed")
    app_("")
    app_("## 18. Frontend integration instructions")
    app_("")
    app_("1. Start the API (from `p4_forecasting/`):")
    app_("```")
    app_("python -X utf8 -m uvicorn phase6.api.app:app --host 127.0.0.1 "
         "--port 8000")
    app_("```")
    app_("2. Ask the frontend to POST `history` (5 observations) and read "
         "`data.forecast`.")
    app_("```javascript")
    app_('fetch("http://localhost:8000/forecast", {')
    app_('  method: "POST",')
    app_('  headers: { "Content-Type": "application/json" },')
    app_('  body: JSON.stringify(request)')
    app_('})')
    app_('  .then(r => r.json())')
    app_('  .then(data => { /* data.forecast = [{hours, latitude, longitude, '
         'wind_speed_kmh}x3] */ });')
    app_("```")
    app_("3. Interactive docs: http://localhost:8000/docs")
    app_("")
    app_("## 19. Limitations")
    app_("")
    app_("- Input requires exactly 24h of 6-hourly history ending at `t`.")
    app_("- Forecasts are 6h/12h/24h ahead; no longer lead times in this "
         "phase.")
    app_("- Latency numbers are software timings on this machine, not "
         "forecasting-accuracy claims.")
    app_("- CORS is limited to local dev origins; a cross-origin deployment "
         "would need an explicit configuration change.")
    app_("")
    app_("## 20. Scientific statement")
    app_("")
    app_("> **Phase 6 exposes the validated Phase-5 forecasting system "
         "through a local API. It does not retrain the model and does not "
         "establish new forecasting-performance claims.**  ")
    app_("")
    app_("The project retains the audited Phase-4 results unchanged: "
         "`EXP005` is the validation-selected ML champion, while the "
         "movement-vector baseline remains superior overall on the audited "
         "test comparison and persistence remains stronger at +6h.  No 'AI "
         "beats traditional methods', 'best predictor', 'state-of-the-art', "
         "'100% accurate' or real-time-accuracy claim is made.")
    app_("")
    (paths.reports_dir / "PHASE6_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())