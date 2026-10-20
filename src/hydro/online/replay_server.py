"""Soft real-time asyncua OPC-UA replay server with a server-side read-only boundary.

Each 100 ms block publishes array-valued nodes: 10 samples per 100 Hz channel, 1 per 10 Hz
channel, and the current 1 Hz value (refreshed every 10 blocks). ``BlockIndex`` increases by
one per block so clients can detect dropped blocks (plan G16). ``BlockFrame`` carries the
index and every sample of the block in one atomic array for subscribers.

Security: asyncua lets ``UserRole.Admin`` bypass AccessLevel checks, so this server installs
``ReadOnlyUserManager``, which never returns Admin. Every node is CurrentRead only.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import os
from pathlib import Path

import numpy as np
from asyncua import Node, Server, ua
from asyncua.crypto.permission_rules import User, UserRole

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.online.security import DEFAULT_CERT_DIR, insecure_server, secure_server

NAMESPACE_URI = "urn:hydro:zema:replay"
DEFAULT_ENDPOINT = "opc.tcp://127.0.0.1:4840/hydro/"
BLOCKS_PER_CYCLE = 600
BLOCK_SECONDS = 0.1
READ_ONLY = ua.AccessLevel.CurrentRead.mask


def sensor_node_id(channel: str) -> str:
    """Siemens-style string NodeId for a channel, e.g. "DB101"."Sensors"."PS1"."""
    return f'"DB101"."Sensors"."{channel}"'


class ReadOnlyUserManager:
    """Accepts only known username/password pairs as plain users; never Admin, never anonymous."""

    def __init__(self, users: dict[str, str] | None = None) -> None:
        self._users = users or {}
        if any(not name or not pw for name, pw in self._users.items()):
            raise ValueError("OPC-UA users need a non-empty username and password")

    def get_user(self, iserver, username=None, password=None, certificate=None) -> User | None:
        expected = self._users.get(username or "")
        if expected is None or not hmac.compare_digest(expected, password or ""):
            return None
        return User(role=UserRole.User, name=username)


def frame_layout(channels) -> list[tuple[str, int]]:
    """(channel, samples per block) in frame order; 1 Hz channels carry their current value."""
    return [(n, max(1, CHANNEL_REGISTRY[n].rate_hz // 10)) for n in channels]


def encode_block_frame(block_index: int, values: dict[str, np.ndarray]) -> list[float]:
    """One atomic array per block: [block_index, ch1 samples..., ch2 samples..., ...]."""
    for n in PHYSICAL_CHANNELS:
        if n not in values:
            raise KeyError(n)
    parts = [np.asarray(values[n], dtype=float).ravel() for n in PHYSICAL_CHANNELS]
    return [float(block_index), *np.concatenate(parts).tolist()]


def decode_block_frame(frame, channels) -> tuple[int, dict[str, np.ndarray]]:
    """Inverse of encode_block_frame for the given channel order."""
    arr, out, pos = np.asarray(frame, dtype=float), {}, 1
    for name, width in frame_layout(channels):
        out[name], pos = arr[pos : pos + width], pos + width
    if pos != arr.size:
        raise ValueError(f"frame has {arr.size} values, layout expects {pos}")
    return int(arr[0]), out


def block_values(data: dict[str, np.ndarray], cycle: int, block: int) -> dict[str, np.ndarray]:
    """Samples of every channel that belong to one 100 ms block of one cycle."""
    out = {}
    for name, arr in data.items():
        per_block = CHANNEL_REGISTRY[name].rate_hz * BLOCK_SECONDS
        if per_block >= 1:
            start = int(block * per_block)
            out[name] = arr[cycle, start : start + int(per_block)]
        else:
            out[name] = arr[cycle, block // 10 : block // 10 + 1]
    return out


class HydroReplayServer:
    """Replays cycles of the ZeMA dataset over OPC-UA with read-only sensor nodes."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        endpoint: str = DEFAULT_ENDPOINT,
        users: dict[str, str] | None = None,
        cert_dir: Path | None = None,
    ) -> None:
        self.data = data
        self.endpoint = endpoint
        self.cert_dir = cert_dir  # None = unencrypted endpoint (tests / explicit --insecure)
        self.server = Server(user_manager=ReadOnlyUserManager(users))
        self.nodes: dict[str, Node] = {}
        self.block_node: Node | None = None
        self.frame_node: Node | None = None

    async def start(self) -> None:
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        if self.cert_dir is None:
            insecure_server(self.server)
        else:
            await secure_server(self.server, self.cert_dir)
        self.server.set_identity_tokens([ua.UserNameIdentityToken])
        idx = await self.server.register_namespace(NAMESPACE_URI)
        folder = await self.server.nodes.objects.add_object(ua.NodeId('"DB101"', idx), "DB101")
        for name in self.data:
            self.nodes[name] = await self._add_read_only(folder, idx, name, [0.0])
        self.block_node = await self._add_read_only(folder, idx, "BlockIndex", -1)
        self.frame_node = await self._add_read_only(folder, idx, "BlockFrame", [0.0])
        await self.server.start()

    async def _add_read_only(self, folder, idx: int, name: str, initial):
        node_id = ua.NodeId(sensor_node_id(name), idx)
        variant_type = ua.VariantType.Int64 if isinstance(initial, int) else ua.VariantType.Double
        node = await folder.add_variable(node_id, name, ua.Variant(initial, variant_type))
        for attr in (ua.AttributeIds.AccessLevel, ua.AttributeIds.UserAccessLevel):
            await node.write_attribute(
                attr, ua.DataValue(ua.Variant(READ_ONLY, ua.VariantType.Byte))
            )
        return node

    async def publish_block(self, cycle: int, block: int, block_index: int) -> None:
        """Write one block server-side (internal writes are not subject to client access).

        ``BlockFrame`` is written in a single call, so subscribers get index and samples
        atomically; the per-channel nodes are kept for browsing clients.
        """
        values = block_values(self.data, cycle, block)
        for name, vals in values.items():
            variant = ua.Variant(vals.astype(float).tolist(), ua.VariantType.Double)
            await self.nodes[name].write_value(variant)
        await self.block_node.write_value(ua.Variant(block_index, ua.VariantType.Int64))
        frame = encode_block_frame(block_index, values)
        await self.frame_node.write_value(ua.Variant(frame, ua.VariantType.Double))

    async def run(self, cycles: range, speed: float = 1.0) -> int:
        """Replay cycles on an absolute schedule so jitter does not accumulate."""
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        loop = asyncio.get_running_loop()
        t0, block_index = loop.time(), 0
        for cycle in cycles:
            for block in range(BLOCKS_PER_CYCLE):
                await self.publish_block(cycle, block, block_index)
                block_index += 1
                await asyncio.sleep(
                    max(0.0, t0 + block_index * BLOCK_SECONDS / speed - loop.time())
                )
        return block_index

    async def stop(self) -> None:
        await self.server.stop()


def _credentials_from_env() -> dict[str, str]:
    """Read OPC-UA credentials from environment variables."""
    for var in ("HYDRO_OPCUA_USER", "HYDRO_OPCUA_PASSWORD"):
        if not os.environ.get(var):
            raise SystemExit(f"Missing required environment variable: {var}")
    return {os.environ["HYDRO_OPCUA_USER"]: os.environ["HYDRO_OPCUA_PASSWORD"]}


async def _serve(args: argparse.Namespace) -> None:
    from hydro.data.loader import load_cycle_dataset

    data, _ = load_cycle_dataset(Path(args.data_dir))
    users = _credentials_from_env()
    cert_dir = None if args.insecure else Path(args.cert_dir)
    server = HydroReplayServer(data, args.endpoint, users, cert_dir)
    await server.start()
    try:
        if args.test_startup:
            print(f"replay server up at {args.endpoint} with {len(server.nodes)} sensor nodes")
            return
        await server.run(range(args.start, args.start + args.cycles), args.speed)
    finally:
        await server.stop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ZeMA OPC-UA replay server (read-only)")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--test-startup", action="store_true")
    parser.add_argument("--cert-dir", default=str(DEFAULT_CERT_DIR))
    parser.add_argument(
        "--insecure", action="store_true", help="unencrypted endpoint, localhost testing only"
    )
    asyncio.run(_serve(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
