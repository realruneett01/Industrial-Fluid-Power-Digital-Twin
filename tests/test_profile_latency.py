"""Tests for scripts/profile_latency.py: inference latency and CMMS dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from hydro.data.schema import CHANNEL_REGISTRY
from scripts.profile_latency import (
    _run_cmms_heavy_check,
    gate,
    infer_cycle,
    main,
    profile,
    train_models,
)

if TYPE_CHECKING:
    pass


@pytest.fixture
def synthetic_dataset() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Generate minimal deterministic dataset for fast unit testing."""
    rng = np.random.default_rng(42)
    n = 8
    data = {
        name: rng.normal(50.0, 2.0, (n, ch.n_samples)).astype(np.float32)
        for name, ch in CHANNEL_REGISTRY.items()
        if not ch.is_virtual
    }
    data["PS1"] = np.linspace(100.0, 200.0, 6000, dtype=np.float32)[None, :].repeat(n, axis=0)
    targets = pd.DataFrame(
        {
            "cooler": [3, 20, 100, 3, 20, 100, 3, 20],
            "valve": [100, 73, 80, 100, 73, 80, 100, 73],
            "pump": [0, 1, 2, 0, 1, 2, 0, 1],
            "accumulator": [90, 100, 115, 130, 90, 100, 115, 130],
            "stable_flag": [0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    return data, targets


@pytest.fixture
def trained_artifacts(
    synthetic_dataset: tuple[dict[str, np.ndarray], pd.DataFrame],
) -> tuple[dict[str, np.ndarray], float, dict[str, ort.InferenceSession]]:
    """Train fast baseline models and return input data, baseline, and ONNX sessions."""
    data, targets = synthetic_dataset
    train_idx = np.arange(0, 6)
    baseline, onnx_paths = train_models(data, targets, train_idx, iterations=5)
    sessions = {
        tgt: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for tgt, path in onnx_paths.items()
    }
    return data, baseline, sessions


@pytest.mark.parametrize(
    ("metrics", "expected_code"),
    [
        ({"p95_ms": 149.9, "p99_ms": 300.0}, 0),
        ({"p95_ms": 150.0, "p99_ms": 300.0}, 1),
        ({"p95_ms": 160.0, "p99_ms": 300.0}, 1),
    ],
)
def test_gate_standard_thresholds(metrics: dict[str, float], expected_code: int) -> None:
    """Verify standard gate logic triggers exit 1 only on p95 >= 150 ms."""
    assert gate(metrics, heavy=False) == expected_code


@pytest.mark.parametrize(
    ("metrics", "expected_code"),
    [
        ({"p95_ms": 140.0, "p99_ms": 499.0}, 0),
        ({"p95_ms": 140.0, "p99_ms": 500.0}, 1),
        ({"p95_ms": 155.0, "p99_ms": 400.0}, 1),
    ],
)
def test_gate_heavy_thresholds(metrics: dict[str, float], expected_code: int) -> None:
    """Verify heavy-check gate triggers exit 1 on p95 >= 150 ms or p99 >= 500 ms."""
    assert gate(metrics, heavy=True) == expected_code


def test_infer_cycle_probabilities(
    trained_artifacts: tuple[dict[str, np.ndarray], float, dict[str, ort.InferenceSession]],
) -> None:
    """Test infer_cycle returns 4 targets with valid probabilities summing to 1."""
    data, baseline, sessions = trained_artifacts
    cycle = {ch: data[ch][6:7] for ch in data}
    probs = infer_cycle(cycle, baseline, sessions)

    expected_targets = {"cooler", "valve", "pump", "accumulator"}
    assert set(probs.keys()) == expected_targets

    for tgt, prob_arr in probs.items():
        assert prob_arr.ndim == 2
        assert prob_arr.shape[0] == 1
        assert np.isclose(float(np.sum(prob_arr)), 1.0, atol=1e-5), f"{tgt} probs do not sum to 1"
        assert np.all(prob_arr >= 0.0) and np.all(prob_arr <= 1.0)


def test_profile_positive_timings(
    trained_artifacts: tuple[dict[str, np.ndarray], float, dict[str, ort.InferenceSession]],
) -> None:
    """Test profile returns N positive timings for evaluated cycles."""
    data, baseline, sessions = trained_artifacts
    eval_idx = np.array([6, 7])
    timings = profile(data, eval_idx, baseline, sessions, warmup=1)

    assert len(timings) == len(eval_idx)
    assert np.all(timings > 0)


def test_heavy_cmms_mock_dispatch() -> None:
    """Test CMMS mock dispatch succeeds with HTTP 201."""
    assert _run_cmms_heavy_check() is None


@pytest.mark.heavy
def test_real_data_profile_latency_heavy(raw_dir: Path, tmp_path: Path) -> None:
    """Heavy integration test against real dataset verifying CLI flow passes SLA."""
    report_file = tmp_path / "latency.md"
    exit_code = main(
        [
            "--data-dir",
            str(raw_dir),
            "--cycles",
            "2",
            "--heavy-check",
            "--report",
            str(report_file),
        ]
    )
    assert exit_code == 0
    assert report_file.exists()


def test_main_report_lands_in_tmp_path(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    """Assert main writes the latency report to the path specified in tmp_path."""
    report_path = tmp_path / "custom_latency.md"
    exit_code = main(
        [
            "--data-dir",
            str(synthetic_raw_dir),
            "--cycles",
            "2",
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0
    assert report_path.is_file()
    assert "Inference Latency Profile" in report_path.read_text(encoding="utf-8")
