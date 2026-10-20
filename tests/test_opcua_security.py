"""Transport security tests for OPC-UA replay server and advisor client (Basic256Sha256)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from asyncua import Client, ua
from asyncua.client.ua_session import SessionState
from cryptography import x509
from cryptography.x509.oid import ExtensionOID

from hydro.online.replay_server import NAMESPACE_URI, HydroReplayServer, sensor_node_id
from hydro.online.security import SERVER_URI, ensure_cert, secure_client
from tests.test_security import REJECTED, USERS, _free_port, _tiny_data


def _make_client(endpoint: str, user: str = "operator") -> Client:
    """Create a client configured with credentials for the given user."""
    client = Client(endpoint)
    client.set_user(user)
    client.set_password(USERS[user])
    return client


@asynccontextmanager
async def _running_server(
    cert_dir: Path | None = None,
) -> AsyncIterator[tuple[HydroReplayServer, str]]:
    """Yield a started HydroReplayServer on a free localhost port and stop it on exit."""
    endpoint = f"opc.tcp://127.0.0.1:{_free_port()}/hydro/"
    server = HydroReplayServer(_tiny_data(), endpoint, users=USERS, cert_dir=cert_dir)
    await server.start()
    try:
        yield server, endpoint
    finally:
        await server.stop()


async def _read_block_index(endpoint: str, cert_dir: Path | None = None) -> Any:
    """Connect to endpoint, optionally securing session, and read the BlockIndex value."""
    client = _make_client(endpoint)
    if cert_dir is not None:
        await secure_client(client, cert_dir)
    async with client:
        idx = await client.get_namespace_index(NAMESPACE_URI)
        node = client.get_node(ua.NodeId(sensor_node_id("BlockIndex"), idx))
        return await node.read_value()


async def _assert_plain_client_fails(endpoint: str) -> None:
    """Verify that a plain client connection is rejected and no session is activated."""
    client = _make_client(endpoint)
    with pytest.raises(ua.UaError, match="No matching endpoints") as exc_info:
        async with client:
            pass
    assert "SecurityPolicy#None" in str(exc_info.value)

    session = getattr(getattr(client, "uaclient", None), "session", None)
    assert getattr(session, "state", None) != SessionState.ACTIVATED


async def _attempt_sensor_write(client: Client, channel: str = "PS1") -> int:
    """Attempt writing to a read-only sensor node and return the status code."""
    idx = await client.get_namespace_index(NAMESPACE_URI)
    node = client.get_node(ua.NodeId(sensor_node_id(channel), idx))
    with pytest.raises(ua.UaStatusCodeError) as exc_info:
        await node.write_value(ua.Variant([1.0], ua.VariantType.Double))
    return exc_info.value.code


def _extract_san_uris(cert_der: bytes) -> list[str]:
    """Extract URI entries from the Subject Alternative Name extension."""
    cert = x509.load_der_x509_certificate(cert_der)
    ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
    return ext.value.get_values_for_type(x509.UniformResourceIdentifier)


async def _check_secure_connect_and_read(cert_dir: Path) -> None:
    """Verify secure client reads BlockIndex from secure server."""
    async with _running_server(cert_dir) as (_, endpoint):
        val = await _read_block_index(endpoint, cert_dir)
        assert val == -1


async def _check_secure_server_rejects_plain(cert_dir: Path) -> None:
    """Verify secure server rejects unencrypted connection attempt."""
    async with _running_server(cert_dir) as (_, endpoint):
        val = await _read_block_index(endpoint, cert_dir)
        assert val == -1
        await _assert_plain_client_fails(endpoint)


async def _check_secure_session_rejects_writes(cert_dir: Path) -> None:
    """Verify writes are denied even over an authenticated secure session."""
    async with _running_server(cert_dir) as (_, endpoint):
        client = _make_client(endpoint)
        await secure_client(client, cert_dir)
        async with client:
            code = await _attempt_sensor_write(client, "PS1")
            assert code in REJECTED


async def _check_cert_reuse_and_san(cert_dir: Path) -> None:
    """Verify certificate idempotency and presence of SERVER_URI in SAN."""
    pair1 = await ensure_cert(cert_dir, "server")
    c1 = pair1.cert.read_bytes()
    k1 = pair1.key.read_bytes()

    pair2 = await ensure_cert(cert_dir, "server")
    c2 = pair2.cert.read_bytes()
    k2 = pair2.key.read_bytes()

    assert c1 == c2
    assert k1 == k2
    assert SERVER_URI in _extract_san_uris(c1)


async def _check_insecure_server_accepts_plain() -> None:
    """Verify insecure server allows plain client connections."""
    async with _running_server(None) as (_, endpoint):
        val = await _read_block_index(endpoint, None)
        assert val == -1


def test_secure_server_and_client_connects_and_reads(tmp_path: Path) -> None:
    """Secure server with secured client reads BlockIndex successfully."""
    asyncio.run(_check_secure_connect_and_read(tmp_path))


def test_secure_server_rejects_plain_client(tmp_path: Path) -> None:
    """Secure server refuses plain unencrypted client connections."""
    asyncio.run(_check_secure_server_rejects_plain(tmp_path))


def test_secure_session_rejects_writes(tmp_path: Path) -> None:
    """Secure session still rejects writes to read-only sensor nodes."""
    asyncio.run(_check_secure_session_rejects_writes(tmp_path))


def test_ensure_cert_reuses_valid_pair_and_san(tmp_path: Path) -> None:
    """ensure_cert reuses existing files without changes and sets SERVER_URI SAN."""
    asyncio.run(_check_cert_reuse_and_san(tmp_path))


def test_insecure_server_accepts_plain_client() -> None:
    """Insecure server accepts plain client without transport security."""
    asyncio.run(_check_insecure_server_accepts_plain())
