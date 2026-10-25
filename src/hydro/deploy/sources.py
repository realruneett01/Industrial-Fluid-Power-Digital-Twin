"""Cycle sources: where calibration and self-checks get real sensor cycles from."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from hydro.data.schema import CYCLE_SECONDS, PHYSICAL_CHANNELS


def grab_from_dataset(data_dir: Path | str, n_cycles: int) -> list[dict[str, np.ndarray]]:
    """First ``n_cycles`` cycles of an on-disk ZeMA-format dataset, each as (1, n) arrays."""
    from hydro.data.loader import load_cycle_dataset

    data, _ = load_cycle_dataset(Path(data_dir))
    total = len(next(iter(data.values())))
    count = min(n_cycles, total)
    return [{k: v[i : i + 1] for k, v in data.items()} for i in range(count)]


async def grab_from_opcua(
    endpoint: str,
    n_cycles: int,
    timeout_s: float,
    username: str = "",
    password: str = "",
    cert_dir: Path | None = None,
) -> list[dict[str, np.ndarray]]:
    """Capture complete cycles from a live OPC-UA BlockFrame stream (read-only subscription)."""
    from asyncua import Client, ua

    from hydro.online.advisor_client import CycleAssembler, _is_valid_block_index
    from hydro.online.replay_server import (
        BLOCKS_PER_CYCLE,
        NAMESPACE_URI,
        decode_block_frame,
        sensor_node_id,
    )
    from hydro.online.security import secure_client

    assembler, cycles = CycleAssembler(), []
    done = asyncio.Event()

    class _Handler:
        def datachange_notification(self, _node, val, _data) -> None:
            try:
                arr = np.asarray(val, dtype=float)
                if arr.size < 2 or not _is_valid_block_index(arr[0]):
                    return
                idx, values = decode_block_frame(arr, PHYSICAL_CHANNELS)
            except (ValueError, TypeError, IndexError):
                return
            cycle = assembler.add_block(idx, values)
            if cycle is not None:
                cycles.append(cycle)
                if len(cycles) >= n_cycles:
                    done.set()

    client = Client(endpoint, timeout=10)
    if username:
        client.set_user(username)
    if password:
        client.set_password(password)
    if cert_dir is not None:
        await secure_client(client, cert_dir)
    async with client:
        idx = await client.get_namespace_index(NAMESPACE_URI)
        node = client.get_node(ua.NodeId(sensor_node_id("BlockFrame"), idx))
        sub = await client.create_subscription(50, _Handler())
        await sub.subscribe_data_change(node, queuesize=BLOCKS_PER_CYCLE)
        try:
            await asyncio.wait_for(done.wait(), timeout_s)
        except TimeoutError:
            pass  # return whatever complete cycles arrived; the caller judges the count
        finally:
            await sub.delete()
    return cycles


def expected_capture_seconds(n_cycles: int, speed: float = 1.0) -> float:
    """Wall-clock time to stream n complete cycles at the given replay speed (+ one partial)."""
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    return (n_cycles + 1) * CYCLE_SECONDS / speed
