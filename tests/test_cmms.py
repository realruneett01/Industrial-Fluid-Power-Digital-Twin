import json
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from hydro.data.schema import PHYSICAL_CHANNELS
from hydro.online.cmms import build_work_order, dispatch_cmms_work_order, ensure_erpnext_schema


class _Recorder(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Recorder.received.append((self.path, dict(self.headers), json.loads(body)))
        self.send_response(201)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def mock_cmms():
    _Recorder.received = []
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_payload_schema_and_action_mapping():
    ts = datetime(2026, 10, 19, tzinfo=UTC)
    wo = build_work_order("HYD-01", "accumulator", 3, 0.42, ts)
    assert "Recharge nitrogen bladder" in wo["description"]
    assert wo["priority"] == "High"
    assert wo["hydro_timestamp"] == ts.isoformat()
    assert {"hydro_asset_tag", "hydro_fault_type", "hydro_severity", "hydro_metric"} <= wo.keys()


def test_unknown_fault_type_raises():
    with pytest.raises(ValueError):
        build_work_order("HYD-01", "gearbox", 1, 0.1)


def test_dispatch_to_mock_returns_201_with_token_header(mock_cmms, monkeypatch):
    monkeypatch.setenv("ERPNEXT_API_KEY", "k")
    monkeypatch.setenv("ERPNEXT_API_SECRET", "s")
    status = dispatch_cmms_work_order(build_work_order("HYD-01", "pump", 2, 1.5), mock_cmms)
    assert status == 201
    path, headers, body = _Recorder.received[0]
    assert path == "/api/resource/Issue"
    assert headers["Authorization"] == "token k:s"
    assert body["hydro_fault_type"] == "pump"


def test_unreachable_cmms_returns_zero_quickly():
    t0 = time.perf_counter()
    assert dispatch_cmms_work_order({"test": True}, "http://127.0.0.1:9") == 0
    assert time.perf_counter() - t0 < 5


def test_redirect_is_not_followed(monkeypatch):
    """A redirect must not forward the Authorization header to another host."""

    class _Redirect(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/steal")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Redirect)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ERPNEXT_API_KEY", "k")
    monkeypatch.setenv("ERPNEXT_API_SECRET", "s")
    try:
        assert (
            dispatch_cmms_work_order({"test": True}, f"http://127.0.0.1:{server.server_port}")
            == 302
        )
    finally:
        server.shutdown()


def test_malformed_url_returns_zero():
    assert dispatch_cmms_work_order({"test": True}, "invalid_url") == 0


def test_build_work_order_coerces_numpy_types_for_json():
    severity = np.int64(2)
    metric = np.float32(0.85)
    payload = build_work_order("HYD-01", "valve", severity, metric)
    dumped = json.dumps(payload)
    assert isinstance(dumped, str)
    assert type(payload["hydro_severity"]) is int
    assert type(payload["hydro_metric"]) is float
    assert payload["hydro_severity"] == 2
    assert payload["hydro_metric"] == pytest.approx(0.85)


def test_encode_block_frame_reversed_key_order_round_trip():
    from hydro.online.replay_server import (
        decode_block_frame,
        encode_block_frame,
        frame_layout,
    )

    values = {}
    for name, width in reversed(frame_layout(PHYSICAL_CHANNELS)):
        values[name] = np.arange(width, dtype=float) + 1.0
    frame = encode_block_frame(7, values)
    decoded_idx, decoded_values = decode_block_frame(frame, PHYSICAL_CHANNELS)
    assert decoded_idx == 7
    assert set(decoded_values.keys()) == set(PHYSICAL_CHANNELS)
    for name in PHYSICAL_CHANNELS:
        np.testing.assert_array_equal(decoded_values[name], values[name])

    missing = {k: v for k, v in values.items() if k != "PS1"}
    with pytest.raises(KeyError) as exc_info:
        encode_block_frame(7, missing)
    assert "PS1" in str(exc_info.value)


def test_credentials_from_env_missing_raises_system_exit(monkeypatch):
    from hydro.online.replay_server import _credentials_from_env

    monkeypatch.delenv("HYDRO_OPCUA_USER", raising=False)
    monkeypatch.setenv("HYDRO_OPCUA_PASSWORD", "pass")
    with pytest.raises(SystemExit) as exc_info:
        _credentials_from_env()
    assert "HYDRO_OPCUA_USER" in str(exc_info.value)

    monkeypatch.setenv("HYDRO_OPCUA_USER", "user")
    monkeypatch.delenv("HYDRO_OPCUA_PASSWORD", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        _credentials_from_env()
    assert "HYDRO_OPCUA_PASSWORD" in str(exc_info.value)

    monkeypatch.setenv("HYDRO_OPCUA_USER", "")
    monkeypatch.setenv("HYDRO_OPCUA_PASSWORD", "pass")
    with pytest.raises(SystemExit) as exc_info:
        _credentials_from_env()
    assert "HYDRO_OPCUA_USER" in str(exc_info.value)


class _SchemaServer(BaseHTTPRequestHandler):
    priorities: set[str] = set()
    custom_fields: set[str] = set()
    requests: list = []
    forbidden: bool = False

    def do_GET(self):  # noqa: N802 - http.server API
        _SchemaServer.requests.append((self.path, dict(self.headers)))
        if _SchemaServer.forbidden:
            self.send_response(403)
            self.end_headers()
            return
        if self.path.startswith("/api/resource/Issue%20Priority/"):
            name = self.path.split("/api/resource/Issue%20Priority/")[1]
            status = 200 if name in _SchemaServer.priorities else 404
        elif self.path.startswith("/api/resource/Custom%20Field/"):
            field_id = self.path.split("/api/resource/Custom%20Field/")[1]
            status = 200 if field_id in _SchemaServer.custom_fields else 404
        else:
            status = 404
        self.send_response(status)
        self.end_headers()

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        _SchemaServer.requests.append((self.path, dict(self.headers), body))
        if _SchemaServer.forbidden:
            self.send_response(403)
            self.end_headers()
            return
        if self.path == "/api/resource/Issue%20Priority":
            _SchemaServer.priorities.add(body["name"])
            status = 200
        elif self.path == "/api/resource/Custom%20Field":
            _SchemaServer.custom_fields.add(f"Issue-{body['fieldname']}")
            status = 200
        else:
            status = 404
        self.send_response(status)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def mock_schema_server():
    _SchemaServer.priorities = set()
    _SchemaServer.custom_fields = set()
    _SchemaServer.requests = []
    _SchemaServer.forbidden = False
    server = HTTPServer(("127.0.0.1", 0), _SchemaServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_ensure_erpnext_schema_first_and_second_run(mock_schema_server, monkeypatch):
    monkeypatch.setenv("ERPNEXT_API_KEY", "test_key")
    monkeypatch.setenv("ERPNEXT_API_SECRET", "test_secret")
    counts1 = ensure_erpnext_schema(base_url=mock_schema_server)
    assert counts1 == {"created": 8, "existing": 0}
    counts2 = ensure_erpnext_schema(base_url=mock_schema_server)
    assert counts2 == {"created": 0, "existing": 8}


def test_ensure_erpnext_schema_url_encoding(mock_schema_server):
    ensure_erpnext_schema(base_url=mock_schema_server)
    paths = [req[0] for req in _SchemaServer.requests]
    assert any(p.startswith("/api/resource/Issue%20Priority/") for p in paths)
    assert any(p.startswith("/api/resource/Custom%20Field/") for p in paths)
    assert all(" " not in p for p in paths)


def test_ensure_erpnext_schema_forbidden_raises_cleanly(mock_schema_server, monkeypatch):
    _SchemaServer.forbidden = True
    secret_key = "super_secret_key_123"
    secret_token = "super_secret_token_456"
    monkeypatch.setenv("ERPNEXT_API_KEY", secret_key)
    monkeypatch.setenv("ERPNEXT_API_SECRET", secret_token)
    with pytest.raises(RuntimeError) as exc_info:
        ensure_erpnext_schema(base_url=mock_schema_server)
    err_text = str(exc_info.value)
    assert "403" in err_text
    assert secret_key not in err_text
    assert secret_token not in err_text


def test_cli_cmms_setup_subcommand(mock_schema_server):
    from hydro.cli import main

    code = main(["cmms-setup", "--url", mock_schema_server])
    assert code == 0


def test_cli_cmms_setup_failure_returns_nonzero(mock_schema_server):
    from hydro.cli import main

    _SchemaServer.forbidden = True
    code = main(["cmms-setup", "--url", mock_schema_server])
    assert code != 0
