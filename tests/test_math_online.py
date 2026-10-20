from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.online.advisor_client import HEALTHY_CLASSES, CycleAssembler, health_index
from hydro.online.replay_server import block_values, decode_block_frame, encode_block_frame

# Closed-form hand-derived frame length oracle:
# Frame contains: 1 block index float + sum of channel sample widths.
# 7 channels at 100 Hz (PS1-PS6, EPS1): 7 * 10 = 70 samples
# 2 channels at 10 Hz (FS1, FS2): 2 * 1 = 2 samples
# 5 channels at 1 Hz (TS1-TS4, VS1): 5 * 1 = 5 samples
# Total frame length = 1 + 70 + 2 + 5 = 78 floats.
EXPECTED_FRAME_LENGTH: int = 78


@dataclass(frozen=True)
class HealthCase:
    name: str
    probs: tuple[float, float, float, float]
    expected_hi: float


@dataclass(frozen=True)
class InvalidProbCase:
    name: str
    classes: np.ndarray
    probs: np.ndarray


def _synthetic_data(n_cycles: int = 1, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        ch: rng.normal(size=(n_cycles, CHANNEL_REGISTRY[ch].n_samples)) for ch in PHYSICAL_CHANNELS
    }


def _random_block_values(rng: np.random.Generator) -> dict[str, np.ndarray]:
    vals: dict[str, np.ndarray] = {}
    for ch in PHYSICAL_CHANNELS:
        width = max(1, CHANNEL_REGISTRY[ch].rate_hz // 10)
        vals[ch] = rng.uniform(-100.0, 100.0, size=width)
    return vals


@pytest.mark.parametrize("block_index", [0, 1, 500, 100_000, 1_000_000])
def test_frame_length_matches_rate_derived_oracle(block_index: int) -> None:
    # Oracle: 1 index + 7*10 (100Hz) + 2*1 (10Hz) + 5*1 (1Hz) = 78
    rng = np.random.default_rng(block_index)
    vals = _random_block_values(rng)
    frame = encode_block_frame(block_index, vals)
    expected_samples = sum(max(1, CHANNEL_REGISTRY[ch].rate_hz // 10) for ch in PHYSICAL_CHANNELS)
    assert len(frame) == 1 + expected_samples
    assert len(frame) == EXPECTED_FRAME_LENGTH


def test_encode_decode_round_trip_200_seeded_blocks() -> None:
    # Verifies exact round-trip for 200 seeded random blocks across 0..10^6
    rng = np.random.default_rng(20261020)
    for _ in range(200):
        block_idx = int(rng.integers(0, 1_000_001))
        vals = _random_block_values(rng)
        frame = encode_block_frame(block_idx, vals)
        assert len(frame) == EXPECTED_FRAME_LENGTH
        dec_idx, dec_vals = decode_block_frame(frame, PHYSICAL_CHANNELS)
        assert dec_idx == block_idx
        for ch in PHYSICAL_CHANNELS:
            np.testing.assert_array_equal(dec_vals[ch], vals[ch])


@pytest.mark.parametrize(
    ("desc", "fill_val"),
    [
        ("zeros", 0.0),
        ("negative_constant", -42.5),
        ("large_constant", 1e6),
        ("high_pressure_bar", 350.0),
        ("nan_values", np.nan),
        ("inf_values", np.inf),
    ],
)
def test_encode_decode_edge_cases(desc: str, fill_val: float) -> None:
    vals = {
        ch: np.full(max(1, CHANNEL_REGISTRY[ch].rate_hz // 10), fill_val)
        for ch in PHYSICAL_CHANNELS
    }
    frame = encode_block_frame(42, vals)
    dec_idx, dec_vals = decode_block_frame(frame, PHYSICAL_CHANNELS)
    assert dec_idx == 42
    for ch in PHYSICAL_CHANNELS:
        np.testing.assert_array_equal(dec_vals[ch], vals[ch])


@pytest.mark.parametrize("missing_ch", ["PS1", "EPS1", "FS1", "TS1", "VS1"])
def test_encode_missing_channel_raises_key_error(missing_ch: str) -> None:
    rng = np.random.default_rng(7)
    vals = _random_block_values(rng)
    del vals[missing_ch]
    with pytest.raises(KeyError, match=missing_ch):
        encode_block_frame(0, vals)


@pytest.mark.parametrize("length", [0, 1, 77, 79, 100])
def test_decode_invalid_frame_length_raises(length: int) -> None:
    bad_frame = [0.0] * length
    with pytest.raises(ValueError, match="layout expects"):
        decode_block_frame(bad_frame, PHYSICAL_CHANNELS)


@pytest.mark.parametrize("channel", ["PS1", "PS2", "EPS1", "FS1", "FS2"])
def test_block_values_partitions_cycle_high_rate(channel: str) -> None:
    # 600 blocks * per_block samples reconstructs full 60s cycle array exactly
    data = _synthetic_data(n_cycles=1, seed=101)
    blocks = [block_values(data, cycle=0, block=b) for b in range(600)]
    concatenated = np.concatenate([blocks[b][channel] for b in range(600)])
    np.testing.assert_array_equal(concatenated, data[channel][0])


@pytest.mark.parametrize("channel", ["TS1", "TS2", "TS3", "TS4", "VS1"])
def test_block_values_partitions_cycle_1hz(channel: str) -> None:
    # 1 Hz channels sampled every 10 blocks (blocks 0, 10, ... 590) reproduce 60-sample array
    data = _synthetic_data(n_cycles=1, seed=202)
    blocks = [block_values(data, cycle=0, block=b) for b in range(0, 600, 10)]
    concatenated = np.concatenate([blocks[i][channel] for i in range(60)])
    np.testing.assert_array_equal(concatenated, data[channel][0])


@pytest.mark.parametrize(
    ("signal_type", "generator"),
    [
        ("zeros", lambda s: np.zeros((1, s))),
        ("constants", lambda s: np.full((1, s), 42.0)),
        ("ramp", lambda s: np.linspace(0.0, 100.0, s).reshape(1, -1)),
        ("physical_bounds", lambda s: np.linspace(-50.0, 250.0, s).reshape(1, -1)),
    ],
)
def test_block_values_signal_edge_cases(
    signal_type: str, generator: Callable[[int], np.ndarray]
) -> None:
    data = {ch: generator(CHANNEL_REGISTRY[ch].n_samples) for ch in PHYSICAL_CHANNELS}
    for ch in ("PS1", "FS1"):
        blocks = [block_values(data, cycle=0, block=b) for b in range(600)]
        concat = np.concatenate([blocks[b][ch] for b in range(600)])
        np.testing.assert_array_equal(concat, data[ch][0])


@pytest.mark.parametrize("seed", [10, 20, 30, 40, 50])
def test_health_index_bounds_in_zero_to_one(seed: int) -> None:
    rng = np.random.default_rng(seed)
    probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in ("cooler", "valve", "pump", "accumulator"):
        k = 4 if t in ("valve", "accumulator") else 3
        raw = rng.uniform(0.1, 1.0, size=k)
        classes = np.arange(k)
        classes[0] = HEALTHY_CLASSES[t]
        probs[t] = (classes, raw / np.sum(raw))
    hi = health_index(probs)
    assert 0.0 <= hi <= 1.0


@pytest.mark.parametrize(
    "targets",
    [
        ("cooler", "valve", "pump", "accumulator"),
        ("cooler", "valve"),
        ("pump", "accumulator"),
        ("cooler",),
    ],
)
def test_health_index_equals_one_for_all_healthy(targets: tuple[str, ...]) -> None:
    # All healthy one-hot probs have p(healthy) = 1.0; closed-form mean is exactly 1.0.
    probs = {t: (np.array([HEALTHY_CLASSES[t], 999]), np.array([1.0, 0.0])) for t in targets}
    assert health_index(probs) == 1.0


@pytest.mark.parametrize(
    "targets",
    [
        ("cooler", "valve", "pump", "accumulator"),
        ("cooler", "valve"),
        ("pump", "accumulator"),
        ("valve",),
    ],
)
def test_health_index_equals_zero_for_all_faulty(targets: tuple[str, ...]) -> None:
    # All faulty one-hot probs have p(healthy) = 0.0; closed-form mean is exactly 0.0.
    probs = {t: (np.array([HEALTHY_CLASSES[t], 999]), np.array([0.0, 1.0])) for t in targets}
    assert health_index(probs) == 0.0


@pytest.mark.parametrize("target", ["cooler", "valve", "pump", "accumulator"])
def test_health_index_monotonically_decreases(target: str) -> None:
    # Hand derivation: HI = mean(p_h). For target t, d(HI)/d(p_h) = 1/4 > 0.
    # Decreasing p_h strictly decreases HI by delta / 4.
    p_healthy_steps = [1.0, 0.75, 0.5, 0.25, 0.0]
    hi_values: list[float] = []
    for p_h in p_healthy_steps:
        probs = {
            t: (
                np.array([HEALTHY_CLASSES[t], 999]),
                np.array([p_h if t == target else 1.0, 1.0 - p_h if t == target else 0.0]),
            )
            for t in ("cooler", "valve", "pump", "accumulator")
        }
        hi_values.append(health_index(probs))
    for i in range(len(hi_values) - 1):
        assert hi_values[i] > hi_values[i + 1]


@pytest.mark.parametrize(
    "case",
    [
        # Case 1: Cooler=0.70, Valve=0.50, Pump=0.85, Acc=0.40
        # Hand derivation: HI = (0.70 + 0.50 + 0.85 + 0.40) / 4 = 2.45 / 4 = 0.6125
        HealthCase("asymmetric_mix", (0.70, 0.50, 0.85, 0.40), 0.6125),
        # Case 2: Cooler=0.10, Valve=0.20, Pump=0.30, Acc=0.40
        # Hand derivation: HI = (0.10 + 0.20 + 0.30 + 0.40) / 4 = 1.00 / 4 = 0.2500
        HealthCase("uniform_ramp", (0.10, 0.20, 0.30, 0.40), 0.25),
        # Case 3: Cooler=1.0, Valve=1.0, Pump=0.0, Acc=0.0
        # Hand derivation: HI = (1.0 + 1.0 + 0.0 + 0.0) / 4 = 2.00 / 4 = 0.5000
        HealthCase("half_healthy", (1.0, 1.0, 0.0, 0.0), 0.50),
    ],
)
def test_health_index_matches_hand_computed_oracle(case: HealthCase) -> None:
    targets = ("cooler", "valve", "pump", "accumulator")
    probs = {
        t: (np.array([HEALTHY_CLASSES[t], 999]), np.array([p, 1.0 - p]))
        for t, p in zip(targets, case.probs, strict=True)
    }
    np.testing.assert_allclose(health_index(probs), case.expected_hi, atol=1e-12)


@pytest.mark.parametrize(
    "case",
    [
        InvalidProbCase("empty", np.array([100]), np.array([])),
        InvalidProbCase("nan", np.array([100]), np.array([np.nan])),
        InvalidProbCase("inf", np.array([100]), np.array([np.inf])),
        InvalidProbCase("negative", np.array([100, 999]), np.array([-0.2, 1.2])),
        InvalidProbCase("sum_not_one", np.array([100, 999]), np.array([0.3, 0.3])),
    ],
)
def test_health_index_invalid_target_probs_raise(case: InvalidProbCase) -> None:
    probs = {"cooler": (case.classes, case.probs)}
    with pytest.raises(ValueError):
        health_index(probs)


@pytest.mark.parametrize("payload", [{}, {"unknown_target": (np.array([1]), np.array([1.0]))}])
def test_health_index_empty_or_unknown_dict_raises(payload: dict) -> None:
    with pytest.raises(ValueError):
        health_index(payload)


@pytest.mark.parametrize(
    ("n_blocks", "expected_cycles"),
    [
        (0, 0),
        (1, 0),
        (599, 0),
        (600, 1),
        (601, 1),
        (1199, 1),
        (1200, 2),
        (1799, 2),
        (1800, 3),
        (2450, 4),
    ],
)
def test_cycle_assembler_emits_floor_n_over_600(n_blocks: int, expected_cycles: int) -> None:
    # Closed-form math: clean stream starting at 0 emits floor(n / 600) cycles.
    data = _synthetic_data(n_cycles=5, seed=303)
    assembler = CycleAssembler()
    emitted = 0
    for b in range(n_blocks):
        cycle = assembler.add_block(b, block_values(data, cycle=b // 600, block=b % 600))
        if cycle is not None:
            emitted += 1
    assert emitted == expected_cycles
    assert emitted == n_blocks // 600


@pytest.mark.parametrize("start_idx", [0, 600, 1200, 3000])
def test_cycle_assembler_stream_starting_at_boundary_offset(start_idx: int) -> None:
    # Clean stream starting at any boundary (start_idx % 600 == 0)
    # emits exactly 1 cycle per 600 blocks.
    data = _synthetic_data(n_cycles=10, seed=404)
    assembler = CycleAssembler()
    emitted = 0
    for b in range(start_idx, start_idx + 600):
        cycle = assembler.add_block(b, block_values(data, cycle=0, block=b % 600))
        if cycle is not None:
            emitted += 1
    assert emitted == 1


def test_cycle_assembler_reconstructed_cycle_matches_input() -> None:
    # Hand-derived property: concatenated blocks reproduce complete cycle shapes and values
    data = _synthetic_data(n_cycles=1, seed=505)
    assembler = CycleAssembler()
    cycle = None
    for b in range(600):
        cycle = assembler.add_block(b, block_values(data, cycle=0, block=b))
    assert cycle is not None
    for ch in PHYSICAL_CHANNELS:
        assert ch in cycle
        assert cycle[ch].shape == (1, CHANNEL_REGISTRY[ch].n_samples)
        np.testing.assert_array_equal(cycle[ch], data[ch])
