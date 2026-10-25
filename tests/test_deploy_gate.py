from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from hydro.data.schema import CHANNEL_REGISTRY
from hydro.deploy import manifest
from hydro.deploy import selfcheck as sc
from hydro.deploy.gate import GateOptions, run_gate
from hydro.deploy.site import CALIBRATION_FILE, REPORT_FILE, SITE_FILE, load_site
from hydro.deploy.synthetic import synthetic_cycle


def _write_dataset(d: Path, n: int = 6, mutate=None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    cycles = [synthetic_cycle(i) for i in range(n)]
    for name, ch in CHANNEL_REGISTRY.items():
        arr = (
            np.concatenate([c[name] for c in cycles])
            if name in cycles[0]
            else np.ones((n, ch.n_samples))
        )
        if mutate:
            arr = mutate(name, arr)
        np.savetxt(d / f"{name}.txt", arr, delimiter="\t", fmt="%.5f")
    np.savetxt(
        d / "profile.txt", np.tile([[100, 100, 0, 130, 0]], (n, 1)), fmt="%d", delimiter="\t"
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("drive")
    _write_dataset(root / "data" / "raw")
    sc.toolchain_bundle(root / "models" / "online")
    return root


@pytest.fixture
def drive(prepared, tmp_path):
    dst = tmp_path / "drive"
    shutil.copytree(prepared, dst)
    return dst


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "HYDRO_OPCUA_USER",
        "HYDRO_OPCUA_PASSWORD",
        "HYDRO_OPCUA_ENDPOINTS",
        manifest.KEY_ENV,
    ):
        monkeypatch.delenv(var, raising=False)


def _by_name(report):
    return {s["name"]: s for s in report["steps"]}


def test_replay_gate_passes_with_warnings_and_writes_site_files(drive):
    code, report = run_gate(GateOptions(root=drive, mode="replay"))
    assert code == 0 and report["verdict"] == "WARN" and report["gate_passed"]
    site_dir = drive / "site"
    for f in (SITE_FILE, REPORT_FILE, CALIBRATION_FILE):
        assert (site_dir / f).is_file()
    site = load_site(site_dir)
    assert site["source"] == "replay" and site["calibration"]
    assert _by_name(report)["source"]["status"] == "WARN"
    assert "REPLAY, not live" in _by_name(report)["source"]["detail"]
    assert all(_by_name(report)[f"sensor:{c}"]["status"] == "PASS" for c in synthetic_cycle(0))


def test_no_source_fails_closed(prepared, tmp_path):
    root = tmp_path / "empty"
    shutil.copytree(prepared / "models", root / "models")
    code, report = run_gate(GateOptions(root=root))
    assert code == 2 and not report["gate_passed"]
    with pytest.raises(ValueError, match="did not pass"):
        load_site(root / "site")


def test_live_mode_without_endpoint_fails(drive):
    code, report = run_gate(GateOptions(root=drive, mode="live"))
    assert code == 2 and _by_name(report)["source"]["status"] == "FAIL"


def test_missing_bundle_fails(drive):
    shutil.rmtree(drive / "models")
    code, report = run_gate(GateOptions(root=drive, mode="replay"))
    assert code == 2 and _by_name(report)["bundle-files"]["status"] == "FAIL"


def test_tampered_drive_fails_manifest(drive):
    manifest.write_manifest(drive)
    (drive / "models" / "online" / "bundle.json").write_text("{}")
    code, report = run_gate(GateOptions(root=drive, mode="replay"))
    assert code == 2 and _by_name(report)["manifest"]["status"] == "FAIL"


def test_valid_manifest_passes(drive):
    manifest.write_manifest(drive)
    code, report = run_gate(GateOptions(root=drive, mode="replay"))
    assert code == 0 and _by_name(report)["manifest"]["status"] == "PASS"


def test_signed_manifest_with_wrong_key_fails(drive, monkeypatch):
    manifest.write_manifest(drive, key="right")
    monkeypatch.setenv(manifest.KEY_ENV, "wrong")
    code, report = run_gate(GateOptions(root=drive, mode="replay"))
    assert code == 2 and _by_name(report)["manifest"]["status"] == "FAIL"


def test_dead_sensor_in_data_fails_calibration(drive):
    def kill_ps2(name, arr):
        return np.full_like(arr, np.nan) if name == "PS2" else arr

    shutil.rmtree(drive / "data")
    _write_dataset(drive / "data" / "raw", mutate=kill_ps2)
    code, report = run_gate(GateOptions(root=drive, mode="replay"))
    steps = _by_name(report)
    assert (
        code == 2
        and steps["sensor:PS2"]["status"] == "FAIL"
        and steps["sensor:PS1"]["status"] == "PASS"
    )
    assert (
        not (drive / "site" / CALIBRATION_FILE).exists()
        or "PS2" not in json.loads((drive / "site" / CALIBRATION_FILE).read_text())["channels"]
    )


def test_flat_sensor_warns_but_strict_fails(drive):
    def flat(name, arr):
        return np.full_like(arr, 3.0) if name == "FS2" else arr

    shutil.rmtree(drive / "data")
    _write_dataset(drive / "data" / "raw", mutate=flat)
    code, report = run_gate(GateOptions(root=drive, mode="replay"))
    assert code == 0 and _by_name(report)["sensor:FS2"]["status"] == "WARN"
    code, _ = run_gate(GateOptions(root=drive, mode="replay", strict=True))
    assert code == 2


def test_dataset_with_fewer_cycles_than_requested_fails(drive):
    code, report = run_gate(GateOptions(root=drive, mode="replay", calibration_cycles=50))
    assert code == 2 and _by_name(report)["capture"]["status"] == "FAIL"


def test_invalid_replay_speed_is_rejected(drive):
    with pytest.raises(ValueError):
        run_gate(GateOptions(root=drive, replay_speed=0))


def test_gate_is_repeatable_and_calibration_stable(drive):
    run_gate(GateOptions(root=drive, mode="replay"))
    first = json.loads((drive / "site" / CALIBRATION_FILE).read_text())["digest"]
    run_gate(GateOptions(root=drive, mode="replay"))
    assert json.loads((drive / "site" / CALIBRATION_FILE).read_text())["digest"] == first


def test_report_is_valid_json_with_machine_and_profile(drive):
    run_gate(GateOptions(root=drive, mode="replay"))
    rep = json.loads((drive / "site" / REPORT_FILE).read_text())
    assert rep["machine"]["cpu_count"] >= 1 and rep["profile"]["tier"]
    assert not list((drive / "site").glob("*.tmp"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def live_server(drive):
    port = _free_port()
    env = {**os.environ, "HYDRO_OPCUA_USER": "u", "HYDRO_OPCUA_PASSWORD": "p"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "hydro.online.replay_server", "--insecure", "--data-dir",
         str(drive / "data" / "raw"), "--cycles", "6", "--speed", "60",
         "--endpoint", f"opc.tcp://127.0.0.1:{port}/hydro/"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    deadline = time.time() + 30
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.2)
    yield f"opc.tcp://127.0.0.1:{port}/hydro/"
    proc.terminate()
    proc.wait(timeout=10)


def test_live_gate_calibrates_from_real_opcua_stream(drive, live_server, monkeypatch):
    monkeypatch.setenv("HYDRO_OPCUA_USER", "u")
    monkeypatch.setenv("HYDRO_OPCUA_PASSWORD", "p")
    code, report = run_gate(
        GateOptions(
            root=drive, mode="live", endpoint=live_server, insecure=True, calibration_cycles=1
        )
    )
    steps = _by_name(report)
    assert code == 0, report["text"]
    assert steps["source"]["status"] == "PASS" and steps["capture"]["status"] == "PASS"
    assert load_site(drive / "site")["source"] == "live"


def test_live_gate_without_credentials_fails(drive, live_server):
    code, report = run_gate(
        GateOptions(
            root=drive, mode="live", endpoint=live_server, insecure=True, calibration_cycles=1
        )
    )
    assert code == 2 and _by_name(report)["credentials"]["status"] == "FAIL"


def test_live_gate_with_wrong_password_fails(drive, live_server, monkeypatch):
    monkeypatch.setenv("HYDRO_OPCUA_USER", "u")
    monkeypatch.setenv("HYDRO_OPCUA_PASSWORD", "WRONG")
    code, report = run_gate(
        GateOptions(root=drive, mode="live", endpoint=live_server, insecure=True,
                    calibration_cycles=1, capture_timeout_s=10)
    )  # fmt: skip
    assert code == 2 and _by_name(report)["capture"]["status"] == "FAIL"
