from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hydro.deploy import manifest
from hydro.deploy import selfcheck as sc
from hydro.deploy.gate import GateOptions, run_gate

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_portable", ROOT / "scripts" / "build_portable.py"
)
bp = importlib.util.module_from_spec(_spec)
sys.modules["build_portable"] = bp
_spec.loader.exec_module(bp)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return sc.toolchain_bundle(tmp_path_factory.mktemp("m") / "online")


def test_image_without_wheels_has_everything_and_verifies(bundle, tmp_path):
    out = tmp_path / "usb"
    assert bp.main(["--out", str(out), "--models", str(bundle), "--no-wheels"]) == 0
    for name in (
        "start-hydro.sh",
        "Start-Hydro.bat",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
    ):
        assert (out / name).is_file(), name
    assert (out / "src" / "hydro" / "deploy" / "gate.py").is_file()
    assert (out / "models" / "online" / "bundle.json").is_file()
    assert (out / "start-hydro.sh").stat().st_mode & 0o111
    assert manifest.verify_manifest(out)["ok"]
    assert not list(out.rglob("__pycache__")) and not list(out.rglob("*.pyc"))
    assert not (out / "tests").exists() and not (out / "reports").exists()


def test_image_does_not_leak_private_or_generated_files(bundle, tmp_path):
    out = tmp_path / "usb"
    bp.main(["--out", str(out), "--models", str(bundle), "--no-wheels"])
    names = {p.name for p in out.rglob("*")}
    for forbidden in (
        ".git",
        "certs",
        "site",
        "credentials.json",
        ".env",
        "PROGRESS.md",
        "CLAUDE.md",
    ):
        assert forbidden not in names


def test_tampering_with_the_image_is_caught_by_the_gate(bundle, tmp_path):
    out = tmp_path / "usb"
    bp.main(["--out", str(out), "--models", str(bundle), "--no-wheels"])
    (out / "src" / "hydro" / "deploy" / "gate.py").write_text("# tampered\n")
    res = manifest.verify_manifest(out)
    assert not res["ok"] and res["changed"] == ["src/hydro/deploy/gate.py"]
    code, report = run_gate(GateOptions(root=out, mode="replay"))
    assert code == 2 and any(
        s["name"] == "manifest" and s["status"] == "FAIL" for s in report["steps"]
    )


def test_incomplete_bundle_is_refused(bundle, tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "bundle.json").write_text("{}")
    with pytest.raises(SystemExit, match="incomplete"):
        bp.main(["--out", str(tmp_path / "usb"), "--models", str(broken), "--no-wheels"])


def test_non_dataset_folder_is_refused(bundle, tmp_path):
    with pytest.raises(SystemExit, match="ZeMA"):
        bp.main(
            [
                "--out",
                str(tmp_path / "u"),
                "--models",
                str(bundle),
                "--dataset",
                str(tmp_path),
                "--no-wheels",
            ]
        )


def test_unknown_platform_rejected_by_argparse(bundle, tmp_path):
    with pytest.raises(SystemExit):
        bp.main(["--out", str(tmp_path / "u"), "--models", str(bundle), "--platform", "amiga"])


@pytest.mark.parametrize("group", sorted(bp.PLATFORM_GROUPS))
def test_every_linux_group_lists_the_modern_manylinux_tags(group):
    tags = bp.PLATFORM_GROUPS[group]
    if group.startswith("linux"):
        assert any("manylinux_2_28" in t for t in tags) and any("manylinux2014" in t for t in tags)


def test_wheelhouse_requirements_file_lists_runtime_only(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bp.subprocess, "run", lambda cmd, check: calls.append(cmd))
    bp.build_wheelhouse(tmp_path, ["linux-x86_64", "windows-x64"], ["3.12", "3.13"])
    reqs = (tmp_path / "wheelhouse" / "requirements.txt").read_text().split()
    assert "torch" not in " ".join(reqs) and "asyncua>=1.1" in reqs
    assert len(calls) == 4  # 2 platforms x 2 Python versions
    assert all("--only-binary=:all:" in c for c in calls)
