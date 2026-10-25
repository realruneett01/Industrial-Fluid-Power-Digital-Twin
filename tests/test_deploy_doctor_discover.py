from __future__ import annotations

import asyncio
import shutil
import socket
import sys
from datetime import UTC, datetime

import numpy as np
import pytest

from hydro.data.schema import CHANNEL_REGISTRY
from hydro.deploy import discover, doctor
from hydro.deploy.machine import Machine
from hydro.deploy.results import Status, worst


def _m(**kw) -> Machine:
    base = dict(os_name="Linux", arch="x86_64", python="3.12.1", cpu_count=4, ram_gb=8.0,
                in_container=False, has_systemd=False, hostname="h")  # fmt: skip
    return Machine(**{**base, **kw})


def test_doctor_passes_on_this_machine(tmp_path):
    results = doctor.run_doctor(tmp_path / "site")
    failed = [r for r in results if r.status == Status.FAIL]
    assert not failed, failed
    assert (tmp_path / "site").is_dir()


def test_check_python_too_old(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 0, "final", 0))
    assert doctor.check_python(_m(python="3.11.0")).status == Status.FAIL


@pytest.mark.parametrize("year", [1970, 1999, 2000, 2023])
def test_clock_before_2024_fails(year):
    assert doctor.check_clock(datetime(year, 1, 1, tzinfo=UTC)).status == Status.FAIL


@pytest.mark.parametrize("year", [2024, 2026, 2040])
def test_clock_plausible_passes(year):
    assert doctor.check_clock(datetime(year, 6, 1, tzinfo=UTC)).status == Status.PASS


@pytest.mark.parametrize(
    ("ram", "status"),
    [(0.0, Status.WARN), (0.5, Status.WARN), (1.0, Status.PASS), (16.0, Status.PASS)],
)
def test_ram_check(ram, status):
    assert doctor.check_ram(_m(ram_gb=ram)).status == status


def test_readonly_site_dir_fails(tmp_path, monkeypatch):
    target = tmp_path / "ro"

    def boom(*a, **k):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(type(target), "mkdir", boom)
    res = doctor.check_writable(target)
    assert res.status == Status.FAIL and "read-only" in res.detail


def test_site_dir_that_is_a_file_fails(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    assert doctor.check_writable(f / "site").status == Status.FAIL


def test_low_disk_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        doctor.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(1, 1, 10 * 2**20)
    )
    res = doctor.check_disk(tmp_path)
    assert res.status == Status.FAIL and "10 MB" in res.detail


def test_missing_module_is_reported(monkeypatch):
    real = doctor.importlib.import_module

    def fake(name):
        if name == "asyncua":
            raise ImportError("gone")
        return real(name)

    monkeypatch.setattr(doctor.importlib, "import_module", fake)
    by = {r.name: r for r in doctor.check_modules()}
    assert by["module:asyncua"].status == Status.FAIL and by["module:numpy"].status == Status.PASS


def test_broken_wheel_oserror_is_a_failure_not_a_crash(monkeypatch):
    def fake(name):
        raise OSError("DLL load failed")

    monkeypatch.setattr(doctor.importlib, "import_module", fake)
    assert worst(doctor.check_modules()) == Status.FAIL


def test_port_in_use_warns():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert doctor.check_port(port).status == Status.WARN
    assert doctor.check_port(port).status == Status.PASS


# ---------------- discovery ----------------


def test_candidate_endpoints_order_and_dedup(monkeypatch):
    monkeypatch.setenv(discover.ENDPOINTS_ENV, "opc.tcp://a:1, opc.tcp://b:2 ,,opc.tcp://a:1")
    got = discover.candidate_endpoints(["opc.tcp://x:9", "opc.tcp://b:2"])
    assert got[:3] == ["opc.tcp://x:9", "opc.tcp://b:2", "opc.tcp://a:1"]
    assert len(got) == len(set(got)) and discover.DEFAULT_ENDPOINT in got


def test_candidate_endpoints_without_env(monkeypatch):
    monkeypatch.delenv(discover.ENDPOINTS_ENV, raising=False)
    assert discover.candidate_endpoints()[0] == discover.DEFAULT_ENDPOINT


def test_unreachable_endpoint_is_reported_not_raised():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    info = asyncio.run(discover.probe_endpoint(f"opc.tcp://127.0.0.1:{port}/x", timeout=1))
    assert info["reachable"] is False and info["error"]


def test_garbage_url_is_reported_not_raised():
    info = asyncio.run(discover.probe_endpoint("not a url", timeout=1))
    assert info["reachable"] is False


def test_probe_finds_live_server_and_its_policies():
    from hydro.deploy.selfcheck import _free_port
    from hydro.online.replay_server import HydroReplayServer

    async def go():
        ep = f"opc.tcp://127.0.0.1:{_free_port()}/hydro/"
        server = HydroReplayServer({"PS1": np.zeros((1, 6000))}, ep, {"u": "p"}, cert_dir=None)
        await server.start()
        try:
            return await discover.probe_endpoint(ep, timeout=3)
        finally:
            await server.stop()

    info = asyncio.run(go())
    assert info["reachable"] and info["policies"]


def _write_dataset(d, skip=None):
    d.mkdir(parents=True, exist_ok=True)
    names = ["profile"] + [n for n, c in CHANNEL_REGISTRY.items() if not c.is_virtual]
    for n in names:
        if n != skip:
            (d / f"{n}.txt").write_text("1\n")


def test_find_datasets_complete_and_nested(tmp_path):
    _write_dataset(tmp_path / "data" / "raw")
    assert discover.find_datasets(tmp_path) == [str(tmp_path.resolve() / "data" / "raw")]


@pytest.mark.parametrize("skip", ["profile", "PS1", "TS4", "VS1", "EPS1"])
def test_find_datasets_ignores_incomplete(tmp_path, skip):
    _write_dataset(tmp_path / "d", skip=skip)
    assert discover.find_datasets(tmp_path) == []


def test_find_datasets_depth_limited_and_skips_site(tmp_path):
    _write_dataset(tmp_path / "a" / "b" / "c" / "d")
    _write_dataset(tmp_path / "site" / "x")
    assert discover.find_datasets(tmp_path) == []


def test_serial_listing_never_raises():
    assert isinstance(discover.list_serial_ports(), list)


def test_discovery_result_helpers():
    d = discover.Discovery(
        endpoints=[{"url": "a", "reachable": False}, {"url": "b", "reachable": True}]
    )
    assert [e["url"] for e in d.reachable_endpoints()] == ["b"]
    assert set(d.to_dict()) == {"endpoints", "datasets", "serial_ports"}
