"""Supervisory advisor client for hydraulic condition monitoring over OPC-UA (Level 2).

Subscribes to BlockIndex, reads multi-rate sensor channels in batches, assembles 60 s cycles,
evaluates health index using plugged-in predictors, and dispatches CMMS advisory work orders.
Read-only client: never writes to Level 1 sensor nodes.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from asyncua import Client, Node, ua

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.online.cmms import build_work_order, dispatch_cmms_work_order
from hydro.online.replay_server import (
    BLOCKS_PER_CYCLE,
    DEFAULT_ENDPOINT,
    NAMESPACE_URI,
    decode_block_frame,
    sensor_node_id,
)
from hydro.online.security import DEFAULT_CERT_DIR, secure_client

log = logging.getLogger(__name__)

HEALTHY_CLASSES: dict[str, int] = {
    "cooler": 100,
    "valve": 100,
    "pump": 0,
    "accumulator": 130,
}


def _validate_target_probs(target: str, probs_arr: np.ndarray) -> None:
    if probs_arr.size == 0 or not np.isfinite(probs_arr).all():
        raise ValueError(f"Probabilities for {target} must be non-empty and finite.")
    if np.any(probs_arr < 0.0):
        raise ValueError(f"Probabilities for {target} must be non-negative.")
    if not np.isclose(float(np.sum(probs_arr)), 1.0, atol=1e-3):
        raise ValueError(f"Probabilities for {target} must sum to 1.0.")


def _target_healthy_prob(target: str, classes: np.ndarray, probs: np.ndarray) -> float:
    healthy_val = HEALTHY_CLASSES[target]
    classes_arr = np.asarray(classes)
    probs_arr = np.asarray(probs, dtype=np.float64).ravel()
    _validate_target_probs(target, probs_arr)
    matches = np.where(classes_arr == healthy_val)[0]
    if len(matches) > 0:
        return float(probs_arr[matches[0]])
    return 0.0


def health_index(probabilities: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """Compute overall hydraulic Health Index HI in [0, 1] as the mean healthy class probability.

    Parameters
    ----------
    probabilities : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping of target name to (class_values, probs) arrays.
        Healthy class reference values:
        - cooler: 100 (full cooler efficiency)
        - valve: 100 (optimal switching behavior)
        - pump: 0 (no internal leakage)
        - accumulator: 130 (optimal pre-charge pressure in bar)

    Returns
    -------
    float
        Health Index in the range [0.0, 1.0], where 1.0 indicates certainty of all
        components being healthy, and 0.0 indicates certainty of faults.
    """
    if not probabilities:
        raise ValueError("Probabilities dictionary cannot be empty.")
    h_probs = [
        _target_healthy_prob(t, *probabilities[t]) for t in HEALTHY_CLASSES if t in probabilities
    ]
    if not h_probs:
        raise ValueError("No recognized diagnostic targets found in probabilities.")
    return float(np.clip(np.mean(h_probs), 0.0, 1.0))


def _find_worst_target(
    probabilities: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[str, float]:
    target_probs = [
        (t, _target_healthy_prob(t, *probabilities[t]))
        for t in HEALTHY_CLASSES
        if t in probabilities
    ]
    if not target_probs:
        return "valve", 1.0
    worst_target, min_p = min(target_probs, key=lambda x: x[1])
    degradation_metric = float(1.0 - min_p)
    return worst_target, degradation_metric


def constant_healthy_predictor(
    _cycle: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Trivial constant predictor yielding 100% confidence in the healthy class for all targets."""
    return {
        "cooler": (np.array([3, 20, 100]), np.array([0.0, 0.0, 1.0])),
        "valve": (np.array([73, 80, 90, 100]), np.array([0.0, 0.0, 0.0, 1.0])),
        "pump": (np.array([0, 1, 2]), np.array([1.0, 0.0, 0.0])),
        "accumulator": (np.array([90, 100, 115, 130]), np.array([0.0, 0.0, 0.0, 1.0])),
    }


class CycleAssembler:
    """Accumulates 100 ms sensor blocks into complete 60 s cycle frames."""

    def __init__(self) -> None:
        self.dropped_blocks: int = 0
        self.discarded_cycles: int = 0
        self._last_block_index: int | None = None
        self._blocks_in_cycle: int = 0
        self._buffers: dict[str, list[np.ndarray]] = {}

    def _reset_cycle(self) -> None:
        self._blocks_in_cycle = 0
        self._buffers = {}

    def _is_duplicate(self, block_index: int) -> bool:
        return block_index == self._last_block_index

    def _handle_restart(self, block_index: int) -> None:
        if self._last_block_index is not None and block_index < self._last_block_index:
            if self._blocks_in_cycle > 0:
                self.discarded_cycles += 1
            self._reset_cycle()

    def _handle_gap(self, block_index: int) -> None:
        if self._last_block_index is not None and block_index > self._last_block_index + 1:
            gap = block_index - self._last_block_index - 1
            self.dropped_blocks += gap
            if self._blocks_in_cycle > 0:
                self.discarded_cycles += 1
            self._reset_cycle()

    def _complete_cycle(self) -> dict[str, np.ndarray] | None:
        try:
            return self._build_cycle()
        except (ValueError, TypeError, KeyError):
            self.discarded_cycles += 1
            return None
        finally:
            self._reset_cycle()

    def add_block(
        self, block_index: int, values: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray] | None:
        """Add one block of sensor values and return completed cycle if ready."""
        if self._is_duplicate(block_index):
            return None
        self._handle_restart(block_index)
        self._handle_gap(block_index)

        self._last_block_index = block_index
        if self._blocks_in_cycle == 0 and block_index % BLOCKS_PER_CYCLE != 0:
            return None
        self._append_values(block_index, values)
        self._blocks_in_cycle += 1

        if self._blocks_in_cycle == BLOCKS_PER_CYCLE:
            return self._complete_cycle()
        return None

    def _append_values(self, block_index: int, values: dict[str, np.ndarray]) -> None:
        for name, arr in values.items():
            schema = CHANNEL_REGISTRY.get(name)
            if schema is not None and schema.rate_hz == 1 and (block_index % 10 != 0):
                continue
            if name not in self._buffers:
                self._buffers[name] = []
            self._buffers[name].append(np.asarray(arr, dtype=np.float64).ravel())

    def _build_cycle(self) -> dict[str, np.ndarray]:
        cycle: dict[str, np.ndarray] = {}
        for name, chunks in self._buffers.items():
            concatenated = np.concatenate(chunks)
            expected = (
                CHANNEL_REGISTRY[name].n_samples if name in CHANNEL_REGISTRY else len(concatenated)
            )
            if len(concatenated) != expected:
                self._reset_cycle()
                raise ValueError(
                    f"Channel {name} sample count mismatch: {len(concatenated)} != {expected}"
                )
            cycle[name] = concatenated.reshape(1, -1)
        return cycle


class _AdvisorSubscriptionHandler:
    """Asyncua subscription notification handler for BlockIndex changes."""

    def __init__(self, advisor: SupervisoryAdvisorClient) -> None:
        self._advisor = advisor

    async def datachange_notification(self, _node: Node, val, _data: ua.DataValue) -> None:
        if val is None or len(val) < 2 or val[0] < 0:
            return
        await self._advisor._on_frame(val)


MAX_BLOCK_INDEX: int = 2**31 - 1


def _is_valid_block_index(raw_idx: Any) -> bool:
    try:
        val = float(raw_idx)
    except (ValueError, TypeError):
        return False
    if not np.isfinite(val) or not val.is_integer():
        return False
    return 0 <= val <= MAX_BLOCK_INDEX


def _is_valid_values(values: dict[str, np.ndarray]) -> bool:
    return all(np.isfinite(arr).all() for arr in values.values())


@dataclass(frozen=True)
class AdvisorConfig:
    """Runtime configuration for SupervisoryAdvisorClient."""

    hi_threshold: float = 0.5
    publish_ms: int = 50
    dispatch: Callable[[dict[str, Any]], Any] = dispatch_cmms_work_order
    asset_tag: str = "HYD-01"
    cert_dir: Path | None = None  # None = unencrypted session


class SupervisoryAdvisorClient:
    """Supervisory condition monitoring client subscribing to ZeMA sensor stream."""

    def __init__(
        self,
        endpoint: str,
        username: str = "",
        password: str = "",
        predictor: Callable[
            [dict[str, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]
        ] = constant_healthy_predictor,
        config: AdvisorConfig | None = None,
    ) -> None:
        cfg = config if config is not None else AdvisorConfig()
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.predictor = predictor
        self.config = cfg
        self.hi_threshold = cfg.hi_threshold
        self.publish_ms = cfg.publish_ms
        self.dispatch = cfg.dispatch
        self.asset_tag = cfg.asset_tag
        self.cert_dir = cfg.cert_dir
        self.assembler = CycleAssembler()
        self.hi_history: list[float] = []
        self.bad_frames: int = 0
        self.cycle_errors: int = 0
        self._client: Client | None = None
        self._subscription: Any = None
        self._channel_names: tuple[str, ...] = PHYSICAL_CHANNELS
        self._last_dropped_blocks: int = 0
        self._last_discarded_cycles: int = 0

    async def _setup_subscription(self) -> None:
        if self._client is None:
            return
        idx = await self._client.get_namespace_index(NAMESPACE_URI)
        frame_node = self._client.get_node(ua.NodeId(sensor_node_id("BlockFrame"), idx))
        handler = _AdvisorSubscriptionHandler(self)
        self._subscription = await self._client.create_subscription(self.publish_ms, handler)
        await self._subscription.subscribe_data_change(frame_node, queuesize=BLOCKS_PER_CYCLE)

    async def _on_frame(self, frame: Any) -> None:
        """Feed one atomic BlockFrame (index + all samples) into the assembler.

        Notifications are queued server-side (queuesize = one cycle), so no reads are
        needed and a frame can never mix samples of two blocks.
        """
        try:
            arr = np.asarray(frame, dtype=float)
            if arr.size == 0 or not _is_valid_block_index(arr[0]):
                self.bad_frames += 1
                return
            block_index, values = decode_block_frame(arr, self._channel_names)
            if not _is_valid_values(values):
                self.bad_frames += 1
                return
        except (ValueError, TypeError, IndexError, KeyError):
            self.bad_frames += 1
            return
        last_idx = self.assembler._last_block_index
        cycle = self.assembler.add_block(block_index, values)
        self._check_predictor_reset(last_idx, block_index)
        if cycle is not None:
            await self._process_cycle(cycle)

    def _check_predictor_reset(self, last_idx: int | None, block_index: int) -> None:
        dropped = self.assembler.dropped_blocks > self._last_dropped_blocks
        discarded = self.assembler.discarded_cycles > self._last_discarded_cycles
        restarted = last_idx is not None and block_index < last_idx
        if any((dropped, discarded, restarted)):
            reset_fn = getattr(self.predictor, "reset", None)
            if callable(reset_fn):
                reset_fn()
        self._last_dropped_blocks = self.assembler.dropped_blocks
        self._last_discarded_cycles = self.assembler.discarded_cycles

    async def _process_cycle(self, cycle: dict[str, np.ndarray]) -> None:
        try:
            probs = await asyncio.to_thread(self.predictor, cycle)
            hi = health_index(probs)
            self.hi_history.append(hi)
            if hi < self.hi_threshold:
                await self._dispatch_alert(probs, hi)
        except Exception:
            # Broad catch protects the OPC-UA subscription loop against predictor or alert failures.
            self.cycle_errors += 1
            logging.getLogger(__name__).exception("Error processing cycle")

    async def _dispatch_alert(
        self, probs: dict[str, tuple[np.ndarray, np.ndarray]], hi: float
    ) -> None:
        worst_target, metric = _find_worst_target(probs)
        severity = 3 if hi < 0.3 else 2
        payload = build_work_order(self.asset_tag, worst_target, severity, metric)
        if inspect.iscoroutinefunction(self.dispatch):
            await self.dispatch(payload)
        else:
            await asyncio.to_thread(self.dispatch, payload)

    async def run(self, duration_s: float) -> list[float]:
        """Connect to OPC-UA server, monitor cycles, and return list of Health Index values."""
        self._client = Client(self.endpoint)
        if self.username:
            self._client.set_user(self.username)
        if self.password:
            self._client.set_password(self.password)
        if self.cert_dir is not None:
            await secure_client(self._client, self.cert_dir)
        async with self._client:
            await self._setup_subscription()
            try:
                await asyncio.sleep(duration_s)
            finally:
                if self._subscription is not None:
                    await self._subscription.delete()
        return list(self.hi_history)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for running SupervisoryAdvisorClient."""
    parser = argparse.ArgumentParser(description="ZeMA Supervisory Advisor Client (Read-Only)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--duration-sec", type=float, default=65.0)
    parser.add_argument("--hi-threshold", type=float, default=0.5)
    parser.add_argument("--cert-dir", default=str(DEFAULT_CERT_DIR))
    parser.add_argument(
        "--insecure", action="store_true", help="unencrypted session, localhost testing only"
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        help="Directory containing bundle.json and ONNX models",
    )
    args = parser.parse_args(argv)

    from hydro.online.predictor import OnnxCyclePredictor

    predictor = OnnxCyclePredictor.from_dir(args.models_dir)
    username = os.environ.get("HYDRO_OPCUA_USER", "")
    password = os.environ.get("HYDRO_OPCUA_PASSWORD", "")
    client = SupervisoryAdvisorClient(
        endpoint=args.endpoint,
        username=username,
        password=password,
        predictor=predictor,
        config=AdvisorConfig(
            hi_threshold=args.hi_threshold,
            cert_dir=None if args.insecure else Path(args.cert_dir),
        ),
    )
    results = asyncio.run(client.run(args.duration_sec))
    print(f"Evaluated {len(results)} cycles, Health Index values: {results}")


if __name__ == "__main__":
    main()
