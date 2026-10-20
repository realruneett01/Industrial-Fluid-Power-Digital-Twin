"""Server-side read-only audit (plan 6.5): every client write to a sensor node is rejected."""

import asyncio
import socket
from pathlib import Path

import numpy as np
import pytest
from asyncua import Client, ua

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.online.replay_server import HydroReplayServer, block_values, sensor_node_id

USERS = {"operator": "s3cret-test-only"}
REJECTED = {ua.StatusCodes.BadNotWritable, ua.StatusCodes.BadUserAccessDenied}
ADVISOR = Path(__file__).resolve().parents[1] / "src" / "hydro" / "online" / "advisor_client.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tiny_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {n: rng.normal(size=(2, CHANNEL_REGISTRY[n].n_samples)) for n in PHYSICAL_CHANNELS}


async def _with_server(check, username=None, password=None):
    endpoint = f"opc.tcp://127.0.0.1:{_free_port()}/hydro/"
    server = HydroReplayServer(_tiny_data(), endpoint, users=USERS)
    await server.start()
    try:
        client = Client(endpoint)
        if username is not None:
            client.set_user(username)
            client.set_password(password)
        async with client:
            return await check(client)
    finally:
        await server.stop()


async def _attempt_writes(client: Client) -> list[int]:
    idx = await client.get_namespace_index("urn:hydro:zema:replay")
    codes = []
    for name in (*PHYSICAL_CHANNELS, "BlockIndex"):
        node = client.get_node(ua.NodeId(sensor_node_id(name), idx))
        with pytest.raises(ua.UaStatusCodeError) as exc:
            await node.write_value(ua.Variant([1.0], ua.VariantType.Double))
        codes.append(exc.value.code)
    return codes


def test_server_rejects_writes():
    codes = asyncio.run(_with_server(_attempt_writes, "operator", USERS["operator"]))
    assert len(codes) == len(PHYSICAL_CHANNELS) + 1
    assert set(codes) <= REJECTED


@pytest.mark.parametrize("creds", [(None, None), ("admin", "admin"), ("operator", "wrong")])
def test_unknown_or_anonymous_login_refused(creds):
    """asyncua lets Admin bypass AccessLevel, so no login may ever map to Admin."""
    with pytest.raises(Exception):  # noqa: B017 - asyncua raises several types on auth failure
        asyncio.run(_with_server(_attempt_writes, *creds))


def test_access_level_attributes_are_read_only():
    async def read_levels(client: Client):
        idx = await client.get_namespace_index("urn:hydro:zema:replay")
        node = client.get_node(ua.NodeId(sensor_node_id("PS1"), idx))
        al = await node.read_attribute(ua.AttributeIds.AccessLevel)
        ual = await node.read_attribute(ua.AttributeIds.UserAccessLevel)
        return al.Value.Value, ual.Value.Value

    assert asyncio.run(_with_server(read_levels, "operator", USERS["operator"])) == (1, 1)


def test_block_values_rates():
    blk = block_values(_tiny_data(), cycle=0, block=25)
    assert blk["PS1"].shape == (10,)
    assert blk["FS1"].shape == (1,)
    assert blk["TS1"].shape == (1,)


def test_client_has_no_write_credentials():
    if not ADVISOR.exists():
        pytest.skip("advisor_client.py not written yet")
    source = ADVISOR.read_text(encoding="utf-8")
    for forbidden in ("write_value", "set_value", "write_attribute", "call_method", "set_writable"):
        assert forbidden not in source


@pytest.mark.parametrize("users", [{"": "x"}, {"operator": ""}])
def test_empty_credentials_rejected_at_startup(users):
    with pytest.raises(ValueError):
        HydroReplayServer(_tiny_data(), users=users)


def test_zero_speed_rejected():
    server = HydroReplayServer(_tiny_data(), users=USERS)
    with pytest.raises(ValueError):
        asyncio.run(server.run(range(1), speed=0.0))
