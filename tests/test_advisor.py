from __future__ import annotations

import asyncio
import socket
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.online.advisor_client import (
    AdvisorConfig,
    CycleAssembler,
    SupervisoryAdvisorClient,
    constant_healthy_predictor,
    health_index,
    main,
)
from hydro.online.replay_server import (
    HydroReplayServer,
    block_values,
    decode_block_frame,
    encode_block_frame,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _synthetic_cycle(n_cycles: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        ch: rng.normal(size=(n_cycles, CHANNEL_REGISTRY[ch].n_samples)) for ch in PHYSICAL_CHANNELS
    }


def test_cycle_assembler_completes_cycle():
    data = _synthetic_cycle()
    assembler = CycleAssembler()
    result = None
    for block in range(600):
        blk = block_values(data, cycle=0, block=block)
        res = assembler.add_block(block, blk)
        if block < 599:
            assert res is None
        else:
            result = res

    assert result is not None
    assert assembler.dropped_blocks == 0
    assert assembler.discarded_cycles == 0
    for ch in PHYSICAL_CHANNELS:
        assert ch in result
        assert result[ch].shape == (1, CHANNEL_REGISTRY[ch].n_samples)
        np.testing.assert_allclose(result[ch], data[ch])


def test_cycle_assembler_skipped_block_discards():
    data = _synthetic_cycle()
    assembler = CycleAssembler()
    for block in range(10):
        blk = block_values(data, cycle=0, block=block)
        assert assembler.add_block(block, blk) is None

    blk_11 = block_values(data, cycle=0, block=11)
    assert assembler.add_block(11, blk_11) is None
    assert assembler.dropped_blocks == 1
    assert assembler.discarded_cycles == 1


def test_health_index_certainty():
    probs_healthy: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "cooler": (np.array([3, 20, 100]), np.array([0.0, 0.0, 1.0])),
        "valve": (np.array([73, 80, 90, 100]), np.array([0.0, 0.0, 0.0, 1.0])),
        "pump": (np.array([0, 1, 2]), np.array([1.0, 0.0, 0.0])),
        "accumulator": (np.array([90, 100, 115, 130]), np.array([0.0, 0.0, 0.0, 1.0])),
    }
    assert health_index(probs_healthy) == 1.0

    probs_faulty: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "cooler": (np.array([3, 20, 100]), np.array([1.0, 0.0, 0.0])),
        "valve": (np.array([73, 80, 90, 100]), np.array([1.0, 0.0, 0.0, 0.0])),
        "pump": (np.array([0, 1, 2]), np.array([0.0, 1.0, 0.0])),
        "accumulator": (np.array([90, 100, 115, 130]), np.array([1.0, 0.0, 0.0, 0.0])),
    }
    assert health_index(probs_faulty) == 0.0

    with pytest.raises(ValueError):
        health_index({})


def test_advisor_e2e_speed5():
    port = _free_port()
    endpoint = f"opc.tcp://127.0.0.1:{port}/hydro/"
    data = _synthetic_cycle(2)
    users = {"operator": "s3cret-test-only"}
    server = HydroReplayServer(data, endpoint, users=users)

    dispatched: list[dict[str, Any]] = []

    def stub_dispatch(payload: dict[str, Any]) -> int:
        dispatched.append(payload)
        return 201

    def stub_faulty_predictor(
        _cycle: dict[str, np.ndarray],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            "cooler": (np.array([3, 20, 100]), np.array([0.9, 0.1, 0.0])),
            "valve": (np.array([73, 80, 90, 100]), np.array([0.8, 0.2, 0.0, 0.0])),
            "pump": (np.array([0, 1, 2]), np.array([0.0, 0.5, 0.5])),
            "accumulator": (np.array([90, 100, 115, 130]), np.array([0.7, 0.3, 0.0, 0.0])),
        }

    client = SupervisoryAdvisorClient(
        endpoint=endpoint,
        username="operator",
        password="s3cret-test-only",
        predictor=stub_faulty_predictor,
        config=AdvisorConfig(
            hi_threshold=0.5,
            publish_ms=10,
            dispatch=stub_dispatch,
        ),
    )

    async def _run_server() -> None:
        await asyncio.sleep(0.5)
        await server.run(range(2), speed=5.0)

    async def _run_test() -> list[float]:
        await server.start()
        try:
            _, hi_results = await asyncio.gather(_run_server(), client.run(duration_s=25.5))
            return hi_results
        finally:
            await server.stop()

    hi_values = asyncio.run(_run_test())
    assert len(hi_values) >= 1
    assert hi_values[0] == 0.0
    assert len(dispatched) >= 1
    assert dispatched[0]["hydro_fault_type"] in {"cooler", "valve", "pump", "accumulator"}
    assert dispatched[0]["priority"] in {"Medium", "High"}


def test_main_cli_smoke(monkeypatch):
    monkeypatch.setenv("HYDRO_OPCUA_USER", "operator")
    monkeypatch.setenv("HYDRO_OPCUA_PASSWORD", "s3cret")
    import hydro.online.predictor as predictor_mod

    monkeypatch.setattr(
        predictor_mod.OnnxCyclePredictor, "from_dir", classmethod(lambda cls, _p: object())
    )
    with pytest.raises(OSError):
        main(
            [
                "--endpoint",
                "opc.tcp://127.0.0.1:9/hydro/",
                "--duration-sec",
                "0.01",
                "--models-dir",
                "unused",
            ]
        )


def test_assembler_waits_for_cycle_boundary():
    asm = CycleAssembler()
    data = {n: np.zeros((1, CHANNEL_REGISTRY[n].n_samples)) for n in PHYSICAL_CHANNELS}
    outs = [asm.add_block(i, block_values(data, 0, i % 600)) for i in range(250, 1200)]
    done = [o for o in outs if o is not None]
    assert len(done) == 1  # blocks 250..599 skipped; 600..1199 form exactly one cycle


def test_cycle_assembler_sample_count_mismatch_resets_and_counts():
    data = _synthetic_cycle()
    assembler = CycleAssembler()
    for block in range(600):
        blk = block_values(data, cycle=0, block=block)
        if block == 50:
            blk = dict(blk)
            blk["PS1"] = np.zeros(5)
        res = assembler.add_block(block, blk)
        assert res is None
    assert assembler.discarded_cycles == 1

    cycle_1_result = None
    for block in range(600, 1200):
        blk = block_values(data, cycle=0, block=block % 600)
        res = assembler.add_block(block, blk)
        if block == 1199:
            cycle_1_result = res
    assert cycle_1_result is not None
    assert assembler.discarded_cycles == 1


def test_cycle_assembler_restart_resets_and_discards():
    data = _synthetic_cycle()
    assembler = CycleAssembler()
    for block in range(50):
        blk = block_values(data, cycle=0, block=block)
        assert assembler.add_block(block, blk) is None
    assert assembler.discarded_cycles == 0

    blk_0 = block_values(data, cycle=0, block=0)
    assert assembler.add_block(0, blk_0) is None
    assert assembler.discarded_cycles == 1
    assert assembler._blocks_in_cycle == 1
    assert len(assembler._buffers["PS1"]) == 1


def test_block_frame_round_trip_and_bad_frame():
    data = _synthetic_cycle(1)
    values = block_values(data, 0, 37)
    idx, decoded = decode_block_frame(encode_block_frame(37, values), PHYSICAL_CHANNELS)
    assert idx == 37
    for name in PHYSICAL_CHANNELS:
        np.testing.assert_array_equal(decoded[name], values[name])

    client = SupervisoryAdvisorClient(endpoint="opc.tcp://127.0.0.1:4840/hydro/")
    asyncio.run(client._on_frame([5.0, 1.0, 2.0]))  # truncated frame
    assert client.bad_frames == 1
    assert client.assembler._last_block_index is None


def test_predictor_runs_in_worker_thread():
    import threading

    loop_thread = threading.current_thread()
    predictor_threads: list[threading.Thread] = []

    def tracking_predictor(
        _cycle: dict[str, np.ndarray],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        predictor_threads.append(threading.current_thread())
        return constant_healthy_predictor(_cycle)

    async def _test() -> None:
        client = SupervisoryAdvisorClient(
            endpoint="opc.tcp://127.0.0.1:4840/hydro/",
            predictor=tracking_predictor,
        )
        cycle = _synthetic_cycle()
        await client._process_cycle(cycle)

    asyncio.run(_test())
    assert len(predictor_threads) == 1
    assert predictor_threads[0] != loop_thread


def test_advisor_config():
    cfg = AdvisorConfig()
    assert cfg.hi_threshold == 0.5
    assert cfg.publish_ms == 50
    assert cfg.asset_tag == "HYD-01"
    with pytest.raises(FrozenInstanceError):
        cfg.hi_threshold = 0.9  # type: ignore[misc]


def test_cycle_assembler_duplicate_handling():
    data = _synthetic_cycle()
    assembler = CycleAssembler()
    blk = block_values(data, cycle=0, block=0)
    assert assembler.add_block(0, blk) is None
    assert assembler._is_duplicate(0) is True
    assert assembler.add_block(0, blk) is None


def test_cycle_assembler_gap_at_boundary_recovers_next_cycle():
    data = _synthetic_cycle(2)
    assembler = CycleAssembler()
    cycles = []
    for block in range(599):
        blk = block_values(data, cycle=0, block=block)
        res = assembler.add_block(block, blk)
        if res is not None:
            cycles.append(res)

    for block in range(600, 1200):
        blk = block_values(data, cycle=1, block=block % 600)
        res = assembler.add_block(block, blk)
        if res is not None:
            cycles.append(res)

    assert len(cycles) == 1
    for ch in PHYSICAL_CHANNELS:
        np.testing.assert_allclose(cycles[0][ch], data[ch][1:2])
    assert assembler.dropped_blocks == 1
    assert assembler.discarded_cycles == 1


def test_cycle_assembler_midstream_join_gap_before_boundary():
    data = _synthetic_cycle(1)
    assembler = CycleAssembler()
    blk_250 = block_values(data, cycle=0, block=250)
    assert assembler.add_block(250, blk_250) is None

    blk_255 = block_values(data, cycle=0, block=255)
    assert assembler.add_block(255, blk_255) is None
    assert assembler.dropped_blocks == 4
    assert assembler.discarded_cycles == 0


def test_advisor_cycle_error_resilience():
    data = _synthetic_cycle(2)
    cycle_1 = {ch: data[ch][0:1] for ch in PHYSICAL_CHANNELS}
    cycle_2 = {ch: data[ch][1:2] for ch in PHYSICAL_CHANNELS}
    call_count = 0

    def flaky_predictor(
        _cycle: dict[str, np.ndarray],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Predictor failure")
        return constant_healthy_predictor(_cycle)

    client = SupervisoryAdvisorClient(
        endpoint="opc.tcp://127.0.0.1:4840/hydro/",
        predictor=flaky_predictor,
    )
    assert client.cycle_errors == 0

    async def _run() -> None:
        await client._process_cycle(cycle_1)
        await client._process_cycle(cycle_2)

    asyncio.run(_run())
    assert client.cycle_errors == 1
    assert len(client.hi_history) == 1
    assert client.hi_history[0] == 1.0


class _FakePredictorWithReset:
    """Fake predictor tracking call and reset invocations."""

    def __init__(self) -> None:
        self.reset_count: int = 0
        self.call_count: int = 0

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, cycle: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self.call_count += 1
        return constant_healthy_predictor(cycle)


def test_advisor_predictor_reset_on_gap() -> None:
    """Verify clean stream causes no reset and a mid-stream gap triggers one reset."""
    data = _synthetic_cycle()
    fake = _FakePredictorWithReset()
    client = SupervisoryAdvisorClient(
        endpoint="opc.tcp://127.0.0.1:4840/hydro/",
        predictor=fake,
    )
    for b in range(10):
        blk = block_values(data, cycle=0, block=b)
        asyncio.run(client._on_frame(encode_block_frame(b, blk)))
    assert fake.reset_count == 0

    blk_11 = block_values(data, cycle=0, block=11)
    asyncio.run(client._on_frame(encode_block_frame(11, blk_11)))
    assert fake.reset_count == 1

    for b in range(12, 20):
        blk = block_values(data, cycle=0, block=b)
        asyncio.run(client._on_frame(encode_block_frame(b, blk)))
    assert fake.reset_count == 1


def test_advisor_predictor_reset_on_restart() -> None:
    """Verify restart resets the predictor even without gap."""
    data = _synthetic_cycle()
    fake = _FakePredictorWithReset()
    client = SupervisoryAdvisorClient(
        endpoint="opc.tcp://127.0.0.1:4840/hydro/",
        predictor=fake,
    )
    for b in range(50):
        blk = block_values(data, cycle=0, block=b)
        asyncio.run(client._on_frame(encode_block_frame(b, blk)))
    assert fake.reset_count == 0

    blk_0 = block_values(data, cycle=0, block=0)
    asyncio.run(client._on_frame(encode_block_frame(0, blk_0)))
    assert fake.reset_count == 1


def test_onnx_predictor_reset_clears_history() -> None:
    """Verify OnnxCyclePredictor.reset() empties rolling history and clears cached features."""
    from hydro.online.predictor import OnnxCyclePredictor

    predictor = OnnxCyclePredictor(valve_baseline_s=10.0)
    cycle = {ch: np.zeros((1, CHANNEL_REGISTRY[ch].n_samples)) for ch in PHYSICAL_CHANNELS}
    predictor.compute_features(cycle)
    assert len(predictor._history) == 1
    assert predictor.last_features is not None

    predictor.reset()
    assert len(predictor._history) == 0
    assert predictor.last_features is None


def test_extract_cycle_features_filters_physical_channels() -> None:
    """Verify extract_cycle_features ignores extra keys and preserves channel order."""
    from hydro.online.predictor import extract_cycle_features

    data = {ch: np.zeros((1, CHANNEL_REGISTRY[ch].n_samples)) for ch in PHYSICAL_CHANNELS}
    data_with_extra = dict(data)
    data_with_extra["EXTRA_UNKNOWN"] = np.ones((1, 6000))
    feat_base = extract_cycle_features(data, baseline=10.0)
    feat_extra = extract_cycle_features(data_with_extra, baseline=10.0)
    assert feat_base.shape == feat_extra.shape
    np.testing.assert_allclose(feat_base, feat_extra)
