"""Offline operation: the inference path must make no network calls and
import no networking modules."""

import ast
import pathlib
import socket
from unittest.mock import patch

import pytest

BASE = pathlib.Path(__file__).resolve().parent.parent.parent  # p4_forecasting

BANNED = ("urllib", "requests", "socket", "http", "aiohttp", "httpx",
          "websocket", "ftplib", "xmlrpc", "urllib3")


def _modules_to_scan():
    """The actual inference path: phase6 runtime code + the Phase-5 service,
    inference and baseline packages.  Test files are excluded (tests are
    allowed to import httpx for the FastAPI TestClient)."""
    roots = ["phase6", "phase5/service", "phase5/inference",
             "phase5/baselines"]
    out = []
    for rel in roots:
        root = BASE / rel
        for py in sorted(root.rglob("*.py")):
            if py.name.startswith("__"):
                continue
            relp = py.relative_to(BASE).as_posix()
            if relp.startswith("phase6/tests/"):
                continue
            if relp == "phase6/run_phase6.py":
                # the runner itself imports socket to prove that the runtime
                # opens none; it is a verification tool, not inference code
                continue
            out.append(py)
    return out


def test_no_network_imports_in_inference_path():
    problems = []
    for py in _modules_to_scan():
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in BANNED:
                        problems.append(f"{py}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in BANNED:
                    problems.append(f"{py}: from {node.module} import ...")
    assert not problems, problems


def test_runtime_inference_uses_no_sockets(adapter, valid_request_model):
    """If any socket were opened during a forecast, this would raise."""
    with patch.object(socket.socket, "connect",
                      autospec=True, side_effect=AssertionError(
                          "socket.connect called during inference")):
        with patch.object(socket.socket, "sendall",
                          autospec=True, side_effect=AssertionError(
                              "socket.sendall called during inference")):
            res = adapter.forecast(valid_request_model)
    assert res["status"] == "success"


def test_health_reports_offline(client):
    assert client.get("/health").json()["offline"] is True


def test_all_artifacts_local():
    from phase5.config import default_paths
    paths = default_paths()
    for rel in ("champion_checkpoint", "champion_config",
                "normalization_stats", "champion_meta"):
        assert getattr(paths, rel).exists(), rel
    # all artifact values must be local absolute paths (no URLs)
    assert str(paths.champion_checkpoint).startswith(tuple(
        ["C:", "C:\\", "/"]))


def test_no_database_server_dependency(client, valid_request):
    # a successful forecast requires no DB host config anywhere in config.py
    import phase6.config as cfg
    source = pathlib.Path(cfg.__file__).read_text(encoding="utf-8")
    assert "postgres" not in source.lower()
    assert "mongodb" not in source.lower()
    r = client.post("/forecast", json=valid_request)
    assert r.status_code == 200