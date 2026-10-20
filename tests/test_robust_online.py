"""Parametrized robustness tests for online streaming and advisor components.

Verifies edge cases for frame decoding, cycle assembly, feature computation under degraded
signals, and health index evaluation with invalid probability distributions.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.online.advisor_client import (
    CycleAssembler,
    SupervisoryAdvisorClient,
    health_index,
)
from hydro.online.predictor import OnnxCyclePredictor
from hydro.online.replay_server import block_values, decode_block_frame

FRAME_WIDTH: int = 78  # 1 block index + 77 physical sensor channel samples
EXPECTED_FEATURES: int = 161


def _make_frame(index: float, fill: float = 0.0) -> list[float]:
    """Create a 78-element frame with specified block index and fill value."""
    return [float(index)] + [float(fill)] * (FRAME_WIDTH - 1)


def _make_cycle(fill: float = 1.0) -> dict[str, np.ndarray]:
    """Create a 1-cycle sensor dictionary with given fill value."""
    return {
        ch: np.full((1, CHANNEL_REGISTRY[ch].n_samples), fill, dtype=np.float64)
        for ch in PHYSICAL_CHANNELS
    }


def _assert_cycle_shapes(cycle: dict[str, np.ndarray]) -> None:
    """Assert cycle contains expected 2D shape for every physical channel."""
    for ch in PHYSICAL_CHANNELS:
        assert cycle[ch].shape == (1, CHANNEL_REGISTRY[ch].n_samples)


@pytest.mark.parametrize(
    "corrupt_frame",
    [
        [],
        [0.0] * 10,
        [0.0] * 50,
        [0.0] * 100,
    ],
)
def test_decode_block_frame_invalid_length(corrupt_frame: list[float]) -> None:
    """Verify decode_block_frame raises ValueError on empty or mismatched length frames."""
    with pytest.raises(ValueError):
        decode_block_frame(corrupt_frame, PHYSICAL_CHANNELS)


@pytest.mark.parametrize(
    "corrupt_frame",
    [
        [],
        [0.0] * 50,
        [0.0] * 100,
        _make_frame(float("nan")),
        _make_frame(float("inf")),
        _make_frame(-1.0),
        _make_frame(1.5),
        _make_frame(1e18),
        [0.0] + [float("nan")] * (FRAME_WIDTH - 1),
        [0.0] + [float("inf")] * (FRAME_WIDTH - 1),
    ],
)
def test_on_frame_robustness_no_escape_and_counts_bad(corrupt_frame: list[float]) -> None:
    """Verify SupervisoryAdvisorClient._on_frame catches corrupt frames without escaping."""
    advisor = SupervisoryAdvisorClient("opc.tcp://127.0.0.1:4840")
    asyncio.run(advisor._on_frame(corrupt_frame))
    assert advisor.bad_frames == 1
    assert advisor.assembler._blocks_in_cycle == 0
    assert len(advisor.hi_history) == 0


def test_cycle_assembler_shuffled_never_raises_or_corrupts() -> None:
    """Verify CycleAssembler fed shuffled block indices never raises and maintains shapes."""
    rng = np.random.default_rng(42)
    data = {ch: rng.normal(size=(2, CHANNEL_REGISTRY[ch].n_samples)) for ch in PHYSICAL_CHANNELS}
    assembler = CycleAssembler()
    indices = list(range(600))
    rng.shuffle(indices)

    for idx in indices:
        assert assembler.add_block(idx, block_values(data, 0, idx % 600)) is None

    completed: list[dict[str, np.ndarray]] = []
    for idx in range(600):
        if (c := assembler.add_block(idx, block_values(data, 0, idx))) is not None:
            completed.append(c)

    next_indices = list(range(600, 1200)) * 2
    rng.shuffle(next_indices)
    for idx in next_indices:
        assert assembler.add_block(idx, block_values(data, 1, idx % 600)) is None

    assert len(completed) == 1
    _assert_cycle_shapes(completed[0])


def test_cycle_assembler_duplicated_never_raises_or_corrupts() -> None:
    """Verify CycleAssembler fed duplicate block indices ignores duplicates cleanly."""
    rng = np.random.default_rng(43)
    data = {ch: rng.normal(size=(2, CHANNEL_REGISTRY[ch].n_samples)) for ch in PHYSICAL_CHANNELS}
    assembler = CycleAssembler()
    completed = []

    for idx in range(600):
        for _ in range(2):
            res = assembler.add_block(idx, block_values(data, 0, idx))
            if res is not None:
                completed.append(res)

    assert len(completed) == 1
    _assert_cycle_shapes(completed[0])


def test_cycle_assembler_reversed_never_raises() -> None:
    """Verify CycleAssembler fed reversed block indices handles resets gracefully."""
    rng = np.random.default_rng(44)
    data = {ch: rng.normal(size=(2, CHANNEL_REGISTRY[ch].n_samples)) for ch in PHYSICAL_CHANNELS}
    assembler = CycleAssembler()

    for idx in reversed(range(600)):
        cycle = assembler.add_block(idx, block_values(data, 0, idx))
        assert cycle is None


def test_cycle_assembler_wrap_around_completes_valid_cycle() -> None:
    """Verify CycleAssembler correctly handles stream wrap-around and completes cycle."""
    rng = np.random.default_rng(45)
    data = {ch: rng.normal(size=(2, CHANNEL_REGISTRY[ch].n_samples)) for ch in PHYSICAL_CHANNELS}
    assembler = CycleAssembler()
    completed = []

    for idx in list(range(500, 600)) + list(range(600)):
        res = assembler.add_block(idx, block_values(data, 0, idx % 600))
        if res is not None:
            completed.append(res)

    assert len(completed) == 1
    _assert_cycle_shapes(completed[0])


@pytest.mark.parametrize(
    "signal_val",
    [
        0.0,
        42.0,
        float("nan"),
        float("inf"),
    ],
)
def test_predictor_compute_features_degenerate_signals(signal_val: float) -> None:
    """Verify OnnxCyclePredictor.compute_features produces finite output on degraded signals."""
    pred = OnnxCyclePredictor(valve_baseline_s=10.0, sessions={}, history_len=5)
    cycle = _make_cycle(signal_val)
    features = pred.compute_features(cycle)

    assert features.shape == (1, EXPECTED_FEATURES)
    assert np.all(np.isfinite(features))
    assert pred.last_features is not None
    assert np.all(np.isfinite(pred.last_features))


def test_predictor_history_never_exceeds_history_len() -> None:
    """Verify predictor history deque length is strictly bounded by history_len."""
    history_limit = 3
    pred = OnnxCyclePredictor(valve_baseline_s=10.0, sessions={}, history_len=history_limit)
    cycle = _make_cycle(1.0)

    for _ in range(10):
        pred.compute_features(cycle)
        assert len(pred._history) <= history_limit

    assert len(pred._history) == history_limit


@pytest.mark.parametrize(
    ("classes", "probs"),
    [
        (np.array([3, 20, 100]), np.array([0.0, 0.0, 0.0])),
        (np.array([3, 20, 100]), np.array([np.nan, 0.5, 0.5])),
        (np.array([3, 20, 100]), np.array([np.inf, 0.0, 0.0])),
        (np.array([3, 20, 100]), np.array([0.1, 0.2, 0.2])),
        (np.array([3, 20, 100]), np.array([0.5, 0.5, 0.5])),
        (np.array([3, 20, 100]), np.array([-0.2, 0.2, 1.0])),
    ],
)
def test_health_index_invalid_probabilities_raise(classes: np.ndarray, probs: np.ndarray) -> None:
    """Verify health_index raises ValueError on degenerate or unnormalized probabilities."""
    with pytest.raises(ValueError):
        health_index({"cooler": (classes, probs)})


@pytest.mark.parametrize(
    "bad_probs",
    [
        np.array([0.0, 0.0, 0.0]),
        np.array([np.nan, 0.5, 0.5]),
        np.array([0.2, 0.2, 0.2]),
    ],
)
def test_process_cycle_counts_cycle_errors_on_invalid_probs(bad_probs: np.ndarray) -> None:
    """Verify SupervisoryAdvisorClient._process_cycle increments cycle_errors on bad probs."""
    advisor = SupervisoryAdvisorClient("opc.tcp://127.0.0.1:4840")
    dummy_classes = np.array([3, 20, 100])
    advisor.predictor = lambda _c: {"cooler": (dummy_classes, bad_probs)}
    cycle = _make_cycle(1.0)

    asyncio.run(advisor._process_cycle(cycle))
    assert advisor.cycle_errors == 1
    assert len(advisor.hi_history) == 0
