"""CLI wiring plus static checks of launchers, packaging, service files and documentation."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from hydro.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]


def _run(*args, env=None, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "hydro", *args], capture_output=True, text=True, env=env, cwd=cwd
    )


# ---------------- CLI ----------------


@pytest.mark.parametrize(
    "argv",
    [
        ["doctor"],
        ["discover"],
        ["deploy"],
        ["deploy", "--mode", "replay", "--strict", "--start", "--replay-speed", "5"],
        ["run", "--duration-sec", "3"],
        ["health", "--max-age", "5"],
        ["manifest", "build"],
        ["manifest", "verify"],
    ],
)
def test_new_commands_parse(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


@pytest.mark.parametrize("argv", [["deploy", "--mode", "bogus"], ["manifest", "explode"], ["nope"]])
def test_bad_arguments_exit_2(argv):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(argv)
    assert e.value.code == 2


def test_doctor_command_exit_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HYDRO_SITE_DIR", str(tmp_path))
    assert main(["doctor"]) == 0
    assert "[PASS] python" in capsys.readouterr().out


def test_deploy_fails_closed_in_empty_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hydro.cli.REPO_ROOT", tmp_path)
    assert main(["deploy"]) == 2
    out = capsys.readouterr().out
    assert "VERDICT: FAIL" in out and "gate FAILED" in out


def test_deploy_rejects_bad_replay_speed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hydro.cli.REPO_ROOT", tmp_path)
    assert main(["deploy", "--replay-speed", "0"]) == 1
    assert "replay_speed" in capsys.readouterr().out


def test_run_without_site_is_a_clean_error(tmp_path, capsys):
    assert main(["run", "--site-dir", str(tmp_path)]) == 1
    assert "hydro deploy" in capsys.readouterr().out


def test_health_exit_codes(tmp_path):
    assert main(["health", "--site-dir", str(tmp_path)]) == 1
    from hydro.deploy.site import write_heartbeat

    write_heartbeat(tmp_path)
    assert main(["health", "--site-dir", str(tmp_path)]) == 0


def test_manifest_command_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("hydro.cli.REPO_ROOT", tmp_path)
    (tmp_path / "a.txt").write_text("a")
    assert main(["manifest", "verify"]) == 2
    assert main(["manifest", "build"]) == 0
    assert main(["manifest", "verify"]) == 0
    (tmp_path / "a.txt").write_text("b")
    assert main(["manifest", "verify"]) == 2


def test_hydro_root_env_moves_the_root(tmp_path):
    import os

    env = {**os.environ, "HYDRO_ROOT": str(tmp_path), "HYDRO_SITE_DIR": str(tmp_path / "s")}
    res = _run("deploy", env=env)
    assert res.returncode == 2 and (tmp_path / "s" / "deploy_report.json").is_file()


def test_runtime_import_does_not_pull_training_stack():
    code = (
        "import sys, hydro.cli, hydro.deploy.gate, hydro.deploy.service, hydro.online.predictor;"
        "bad=[m for m in ('torch','catboost','sklearn','onnx') if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---------------- packaging ----------------

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_runtime_dependencies_exclude_training_libs():
    runtime = " ".join(PYPROJECT["project"]["dependencies"]).lower()
    for lib in ("torch", "catboost", "scikit-learn", "onnx>=", "pyarrow"):
        assert lib not in runtime
    train = " ".join(PYPROJECT["project"]["optional-dependencies"]["train"]).lower()
    for lib in ("torch", "catboost", "scikit-learn", "pyarrow"):
        assert lib in train


def test_pyproject_is_not_open_source_licensed():
    proj = PYPROJECT["project"]
    assert "AGPL" not in json.dumps(proj) and "Proprietary" in proj["license"]["text"]
    assert "Private :: Do Not Upload" in proj["classifiers"]


def test_console_script_exists():
    assert PYPROJECT["project"]["scripts"]["hydro"] == "hydro.cli:main"


def test_runtime_requirements_match_pyproject():
    spec = (ROOT / "scripts" / "build_portable.py").read_text()
    for dep in PYPROJECT["project"]["dependencies"]:
        assert f'"{dep}"' in spec, dep


def test_gitignore_keeps_generated_state_out():
    ignored = (ROOT / ".gitignore").read_text()
    for entry in ("/site/", "/wheelhouse/", "/certs/", "/models/", "*.onnx"):
        assert entry in ignored


# ---------------- launchers, service files ----------------


def test_posix_launcher_syntax_and_contract():
    sh = ROOT / "start-hydro.sh"
    text = sh.read_text()
    assert subprocess.run(["sh", "-n", str(sh)]).returncode == 0
    assert text.startswith("#!/usr/bin/env sh") and "set -eu" in text
    assert sh.stat().st_mode & 0o111
    for needle in (
        "--check-only",
        "--repair",
        "wheelhouse",
        "deploy --start",
        "PYTHONDONTWRITEBYTECODE",
    ):
        assert needle in text
    assert "3, 12" in text


def test_windows_launcher_contract():
    bat = (ROOT / "Start-Hydro.bat").read_text()
    for needle in (
        "--check-only",
        "--repair",
        "wheelhouse",
        "deploy --start",
        "LOCALAPPDATA",
        "3, 12",
    ):
        assert needle in bat
    assert bat.isascii()


def test_posix_launcher_fails_cleanly_without_python(tmp_path):
    shutil.copy(ROOT / "start-hydro.sh", tmp_path / "start-hydro.sh")
    res = subprocess.run(
        ["/bin/sh", str(tmp_path / "start-hydro.sh")],
        env={"PATH": "/nonexistent", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert res.returncode == 3 and "Python 3.12" in res.stderr


def test_systemd_unit_is_wellformed():
    text = (ROOT / "deploy" / "hydro.service").read_text()
    sections = re.findall(r"^\[(\w+)\]$", text, re.M)
    assert sections == ["Unit", "Service", "Install"]
    assert "StartLimitBurst" in text.split("[Service]")[0]  # belongs in [Unit]
    required = (
        "NoNewPrivileges=true",
        "ProtectSystem=strict",
        "RestartPreventExitStatus=2 3",
        "ExecStartPre=",
        "ExecStart=/opt/hydro/venv/bin/hydro run",
        "User=hydro",
    )
    for needle in required:
        assert needle in text


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed")
def test_systemd_analyze_accepts_unit():
    res = subprocess.run(
        ["systemd-analyze", "verify", str(ROOT / "deploy" / "hydro.service")],
        capture_output=True,
        text=True,
    )
    bad = [ln for ln in res.stderr.splitlines() if ln.strip() and "is not executable" not in ln]
    assert not bad, bad


def test_env_example_has_no_real_secrets():
    text = (ROOT / "deploy" / "hydro.env.example").read_text()
    assert re.search(r"^HYDRO_OPCUA_PASSWORD=$", text, re.M)


def test_dockerfile_contract():
    text = (ROOT / "Dockerfile").read_text()
    assert "USER hydro" in text and "HEALTHCHECK" in text and "HYDRO_ROOT" in text
    assert "torch" not in text and "catboost" not in text
    assert 'ENTRYPOINT ["hydro"]' in text


def test_ci_workflow_covers_windows_arm_docker_systemd():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert set(wf["jobs"]) == {"test", "runtime-only", "docker", "systemd-unit"}
    for needle in ("windows-latest", "ubuntu-24.04-arm", "docker build", "systemd-analyze"):
        assert needle in text


# ---------------- licensing and documentation honesty ----------------

README = (ROOT / "README.md").read_text()
LICENSE = (ROOT / "LICENSE").read_text()


def test_license_is_proprietary_not_open_source():
    assert "ALL RIGHTS RESERVED" in LICENSE and "NOT OPEN-SOURCE" in LICENSE
    assert "GNU AFFERO" not in LICENSE.upper() and "GENERAL PUBLIC LICENSE" not in LICENSE.upper()
    for clause in ("PROHIBITED ACTS", "CONFIDENTIALITY", "NO WARRANTY", "LIMITATION OF LIABILITY",
                   "TERMINATION", "AUDIT", "reverse engineer", "machine-learning"):  # fmt: skip
        assert clause.lower() in LICENSE.lower()


def test_license_is_consistent_everywhere():
    assert "AGPL" not in (ROOT / "pyproject.toml").read_text()
    assert "Proprietary" in README and "AGPL-3.0-or-later). See" not in README
    assert "must make its source code available" not in README
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()


def test_third_party_notice_discloses_lgpl_and_dataset_license():
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert "LGPL" in text and "CC BY 4.0" in text and "asyncua" in text


def test_readme_has_no_unverifiable_claims():
    for banned in (
        "repowise",
        "186+",
        "deployed on an edge box",
        "Verified 2026-10-20",
        "production-ready",
    ):
        assert banned not in README, banned


def test_readme_states_limits_of_calibration_and_replay():
    required = (
        "acceptance and baselining, not model re-tuning",
        "REPLAY, not live",
        "OPC-UA only",
        "research prototype",
        "accumulator",
    )
    for needle in required:
        assert needle in README


def test_readme_local_links_resolve():
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", README):
        if "://" in target:
            continue
        assert (ROOT / target).exists(), target


def test_readme_anchor_for_deployment_section_exists():
    assert "(#plug-and-play-deployment)" in README and "## Plug-and-play deployment" in README


def test_readme_reported_numbers_match_reports():
    """The headline LDA numbers in the README must equal reports/baseline_results.md."""
    base = (ROOT / "reports" / "baseline_results.md").read_text()
    for value in ("0.9823", "0.9977", "0.9915", "0.8019"):
        assert value in base and value in README


def test_readme_documented_commands_are_real_subcommands():
    parser = build_parser()
    for cmd in re.findall(r"python -m hydro (\w[\w-]*)", README):
        if cmd in {"online"}:
            continue
        assert cmd in parser._subparsers._group_actions[0].choices, cmd  # noqa: SLF001


def test_deployment_doc_commands_reference_existing_files():
    doc = (ROOT / "docs" / "deploy" / "DEPLOYMENT.md").read_text()
    for path in ("deploy/hydro.service", "scripts/build_portable.py", ".github/workflows/ci.yml"):
        assert path in doc and (ROOT / path).exists()


def test_every_documented_unverified_target_is_flagged():
    doc = (ROOT / "docs" / "deploy" / "DEPLOYMENT.md").read_text()
    assert "unverified until CI runs" in doc and "not run" in doc
