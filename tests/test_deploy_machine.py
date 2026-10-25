from __future__ import annotations

import itertools

import pytest

from hydro.deploy.machine import Machine, auto_profile, detect_machine
from hydro.deploy.results import CheckResult, Status, render, worst


def _machine(cpus: int = 4, ram: float = 8.0) -> Machine:
    return Machine("Linux", "x86_64", "3.12.0", cpus, ram, False, False, "h")


def test_detect_machine_is_sane():
    m = detect_machine()
    assert m.cpu_count >= 1 and m.ram_gb >= 0 and m.os_name and m.arch and m.python
    assert set(m.to_dict()) >= {"os_name", "arch", "cpu_count", "ram_gb", "in_container"}


@pytest.mark.parametrize(
    ("cpus", "ram"),
    list(
        itertools.product(
            [1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128], [0.0, 0.5, 1, 1.9, 2, 4, 8, 64]
        )
    ),
)
def test_auto_profile_grid(cpus, ram):
    p = auto_profile(_machine(cpus, ram))
    assert p["tier"] in {"edge-lite", "edge", "workstation"}
    assert 1 <= p["onnx_intra_op_threads"] <= max(1, cpus)
    assert p["opcua_publish_ms"] in {50, 100}
    assert p["latency_sla_ms"] == 150.0
    assert p["calibration_cycles"] >= 1
    if cpus <= 2:
        assert p["tier"] == "edge-lite"
    if 0 < ram < 2:
        assert p["tier"] == "edge-lite"


@pytest.mark.parametrize("cpus", range(-3, 1))
def test_auto_profile_degenerate_cpu_counts_do_not_crash(cpus):
    assert auto_profile(_machine(cpus, 4.0))["onnx_intra_op_threads"] == 1


def test_auto_profile_never_changes_feature_window():
    assert "history_len" not in auto_profile(_machine())


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], Status.PASS),
        ([Status.PASS], Status.PASS),
        ([Status.PASS, Status.WARN], Status.WARN),
        ([Status.WARN, Status.FAIL, Status.PASS], Status.FAIL),
        ([Status.FAIL], Status.FAIL),
    ],
)
def test_worst(statuses, expected):
    assert worst([CheckResult("x", s, "d") for s in statuses]) == expected


def test_render_shows_hint_only_for_non_pass():
    text = render(
        [
            CheckResult("a", Status.PASS, "fine", "ignored"),
            CheckResult("b", Status.FAIL, "broken", "do this"),
        ]
    )
    assert "[PASS] a: fine" in text and "[FAIL] b: broken" in text
    assert "do this" in text and "ignored" not in text
    assert text.isascii()


def test_check_result_serialises_status_name():
    assert CheckResult("n", Status.WARN, "d").to_dict()["status"] == "WARN"
