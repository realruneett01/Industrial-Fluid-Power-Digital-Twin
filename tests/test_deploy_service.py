from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time

import pytest

from hydro.deploy import selfcheck as sc
from hydro.deploy import service, site
from hydro.deploy.gate import GateOptions, run_gate
from tests.test_deploy_gate import _write_dataset


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    root = tmp_path_factory.mktemp("svc")
    _write_dataset(root / "data" / "raw")
    sc.toolchain_bundle(root / "models" / "online")
    code, _ = run_gate(GateOptions(root=root, mode="replay", replay_speed=60.0))
    assert code == 0
    return root


def test_service_processes_cycles_and_writes_heartbeat_and_alerts(deployed, monkeypatch):
    monkeypatch.delenv("ERPNEXT_URL", raising=False)
    site_dir = deployed / "site"
    rc = asyncio.run(service.serve(site_dir, duration_s=9))
    assert rc == 0
    beat = json.loads((site_dir / site.HEARTBEAT_FILE).read_text())
    assert beat["cycles"] >= 1 and 0.0 <= beat["last_hi"] <= 1.0
    assert beat["cycle_errors"] == 0 and beat["bad_frames"] == 0
    ok, msg = service.health(site_dir, 120)
    assert ok, msg
    alerts = (site_dir / site.ALERTS_FILE).read_text().splitlines()
    assert alerts and "time" in json.loads(alerts[0])


def test_service_refuses_when_gate_failed(tmp_path):
    site.write_json_atomic(tmp_path / site.SITE_FILE, {"version": 1, "gate_passed": False})
    with pytest.raises(ValueError, match="did not pass"):
        asyncio.run(service.serve(tmp_path))


def test_service_refuses_without_site(tmp_path):
    with pytest.raises(ValueError, match="hydro deploy"):
        asyncio.run(service.serve(tmp_path))


def test_bounded_run_gives_up_after_one_failed_attempt(deployed, tmp_path):
    bad = tmp_path / "site"
    bad.mkdir()
    cfg = json.loads((deployed / "site" / site.SITE_FILE).read_text())
    cfg.update(source="live", endpoint="opc.tcp://127.0.0.1:1/x", insecure=True)
    site.write_json_atomic(bad / site.SITE_FILE, cfg)
    t0 = time.time()
    assert asyncio.run(service.serve(bad, duration_s=3)) == 1
    assert time.time() - t0 < 60


def test_health_without_heartbeat_is_unhealthy(tmp_path):
    ok, msg = service.health(tmp_path, 120)
    assert not ok and "no heartbeat" in msg


def test_health_stale_heartbeat_is_unhealthy(tmp_path):
    site.write_json_atomic(tmp_path / site.HEARTBEAT_FILE, {"time": time.time() - 500})
    assert not service.health(tmp_path, 120)[0]
    assert service.health(tmp_path, 1000)[0]


@pytest.mark.parametrize("content", ["", "{", "[]", '{"time": "x"}', "{}"])
def test_corrupt_heartbeat_counts_as_none(tmp_path, content):
    (tmp_path / site.HEARTBEAT_FILE).write_text(content)
    assert site.heartbeat_age(tmp_path) is None


def test_dispatch_goes_to_cmms_when_configured(tmp_path, monkeypatch):
    from hydro.online.cmms import dispatch_cmms_work_order

    monkeypatch.setenv("ERPNEXT_URL", "http://127.0.0.1:9")
    assert service.make_dispatch(tmp_path) is dispatch_cmms_work_order


def test_dispatch_falls_back_to_jsonl(tmp_path, monkeypatch):
    monkeypatch.delenv("ERPNEXT_URL", raising=False)
    fn = service.make_dispatch(tmp_path)
    assert fn({"a": 1}) == 0 and fn({"a": 2}) == 0
    lines = (tmp_path / site.ALERTS_FILE).read_text().splitlines()
    assert [json.loads(x)["a"] for x in lines] == [1, 2]


def test_loopback_credentials_are_random_stable_and_private(tmp_path):
    a = site.loopback_credentials(tmp_path)
    b = site.loopback_credentials(tmp_path)
    assert a == b and len(a["password"]) >= 24
    other = site.loopback_credentials(tmp_path / "other")
    assert other["password"] != a["password"]
    if sys.platform != "win32":
        mode = stat.S_IMODE(os.stat(tmp_path / site.CREDENTIALS_FILE).st_mode)
        assert mode == 0o600


def test_corrupt_credentials_file_is_regenerated(tmp_path):
    (tmp_path / site.CREDENTIALS_FILE).write_text("garbage")
    assert site.loopback_credentials(tmp_path)["password"]


@pytest.mark.parametrize("version", [None, 0, 2, "1"])
def test_load_site_rejects_other_versions(tmp_path, version):
    site.write_json_atomic(tmp_path / site.SITE_FILE, {"version": version, "gate_passed": True})
    with pytest.raises(ValueError):
        site.load_site(tmp_path)


def test_write_json_atomic_leaves_no_temp(tmp_path):
    site.write_json_atomic(tmp_path / "x" / "a.json", {"k": 1})
    assert [p.name for p in (tmp_path / "x").iterdir()] == ["a.json"]


def test_default_site_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRO_SITE_DIR", str(tmp_path / "custom"))
    assert site.default_site_dir(tmp_path) == tmp_path / "custom"
    monkeypatch.delenv("HYDRO_SITE_DIR")
    assert site.default_site_dir(tmp_path) == tmp_path / "site"
