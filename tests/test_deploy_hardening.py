"""Regression tests for problems found while reviewing the deployment layer."""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hydro.deploy import manifest
from hydro.deploy import selfcheck as sc
from hydro.deploy.gate import GateOptions, run_gate
from tests.test_deploy_gate import _by_name, _write_dataset

ROOT = Path(__file__).resolve().parents[1]

FAKE_PYTHON = """#!/bin/sh
case "$1" in
  -c) exit 0 ;;
  -m)
    if [ "$2" = venv ]; then mkdir -p "$3/bin"; cp "$0" "$3/bin/python"; exit 0; fi
    printf '%s\\n' "$@" > "$FAKE_OUT"; exit 0 ;;
esac
exit 0
"""


@pytest.fixture
def launcher(tmp_path):
    drive = tmp_path / "drive dir"  # a space in the path on purpose
    drive.mkdir()
    shutil.copy(ROOT / "start-hydro.sh", drive / "start-hydro.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python3.13", "python3.12", "python3", "python"):
        fake = bin_dir / name
        fake.write_text(FAKE_PYTHON)
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "argv.txt"

    def run(*args):
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "FAKE_OUT": str(out)}
        res = subprocess.run(
            ["/bin/sh", str(drive / "start-hydro.sh"), *args],
            env=env,
            capture_output=True,
            text=True,
        )
        argv = out.read_text().splitlines() if out.exists() else []
        return res, argv

    return run


def test_launcher_starts_service_by_default(launcher):
    res, argv = launcher()
    assert res.returncode == 0, res.stderr
    assert argv == ["-m", "hydro", "deploy", "--start"]


def test_launcher_check_only_does_not_start(launcher):
    res, argv = launcher("--check-only")
    assert res.returncode == 0 and argv == ["-m", "hydro", "deploy"]


def test_launcher_preserves_arguments_with_spaces(launcher):
    _, argv = launcher("--check-only", "--data-dir", "/media/my usb/raw data", "--strict")
    assert argv == ["-m", "hydro", "deploy", "--data-dir", "/media/my usb/raw data", "--strict"]


def test_launcher_repair_recreates_environment(launcher, tmp_path):
    launcher("--check-only")
    venvs = list((tmp_path / ".cache" / "hydro").iterdir())
    assert len(venvs) == 1
    marker = venvs[0] / "marker"
    marker.write_text("x")
    launcher("--check-only")
    assert marker.exists()
    launcher("--check-only", "--repair")
    assert not marker.exists()


def test_wheelhouse_is_covered_by_the_manifest(tmp_path):
    (tmp_path / "wheelhouse").mkdir()
    (tmp_path / "wheelhouse" / "numpy.whl").write_bytes(b"wheel")
    manifest.write_manifest(tmp_path)
    (tmp_path / "wheelhouse" / "numpy.whl").write_bytes(b"evil")
    assert manifest.verify_manifest(tmp_path)["changed"] == ["wheelhouse/numpy.whl"]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("hard")
    _write_dataset(root / "data" / "raw")
    sc.toolchain_bundle(root / "models" / "online")
    return root


@pytest.fixture
def drive(prepared, tmp_path):
    dst = tmp_path / "drive"
    shutil.copytree(prepared, dst)
    return dst


def test_secure_live_gate_explains_missing_server_certificate(drive, monkeypatch):
    monkeypatch.setenv("HYDRO_OPCUA_USER", "u")
    monkeypatch.setenv("HYDRO_OPCUA_PASSWORD", "p")
    code, report = run_gate(
        GateOptions(root=drive, mode="live", endpoint="opc.tcp://127.0.0.1:1/x")
    )
    step = _by_name(report)["server-certificate"]
    assert code == 2 and step["status"] == "FAIL" and "server_cert.der" in step["detail"]


def test_recalibration_flags_drift_against_previous_baseline(drive):
    assert run_gate(GateOptions(root=drive, mode="replay"))[0] == 0
    raw = drive / "data" / "raw"
    ps3 = np.loadtxt(raw / "PS3.txt", delimiter="\t") + 40.0  # sensor offset appears
    np.savetxt(raw / "PS3.txt", ps3, delimiter="\t", fmt="%.5f")
    code, report = run_gate(GateOptions(root=drive, mode="replay"))
    steps = _by_name(report)
    assert code == 0  # drift is a warning, not a stop
    assert steps["drift:PS3"]["status"] == "WARN" and steps["drift:PS1"]["status"] == "PASS"


def test_first_calibration_has_no_drift_steps(drive):
    _, report = run_gate(GateOptions(root=drive, mode="replay"))
    assert not [n for n in _by_name(report) if n.startswith("drift:")]


def test_corrupted_previous_calibration_is_ignored_not_fatal(drive):
    run_gate(GateOptions(root=drive, mode="replay"))
    (drive / "site" / "calibration.json").write_text("garbage")
    code, report = run_gate(GateOptions(root=drive, mode="replay"))
    assert code == 0 and not [n for n in _by_name(report) if n.startswith("drift:")]
