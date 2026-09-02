"""Offline proof: /api/analyze completes with sockets disabled."""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_analyze_is_offline(client: TestClient, analyze_request, adapter):
    """No network access is needed anywhere in the analyze path."""
    import socket as _socket
    with patch.object(_socket.socket, "connect", autospec=True,
                      side_effect=AssertionError("network attempted")), \
         patch.object(_socket.socket, "sendall", autospec=True,
                      side_effect=AssertionError("network attempted")):
        r = client.post("/api/analyze", json=analyze_request)
    assert r.status_code == 200
    assert r.json()["status"] == "success"