"""OPC-UA transport security: self-signed application certificates and policy setup.

The replay server offers only Basic256Sha256 SignAndEncrypt unless explicitly started
insecure; the advisor client encrypts with its own certificate and pins the server's.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path

from asyncua import Client, Server, ua
from asyncua.crypto import cert_gen
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from cryptography.x509.oid import ExtendedKeyUsageOID

SERVER_URI = "urn:hydro:replay-server"
CLIENT_URI = "urn:hydro:advisor-client"
DEFAULT_CERT_DIR = Path(__file__).resolve().parents[3] / "certs"
SUBJECT = {"organizationName": "hydro condition twin"}


@dataclass(frozen=True)
class CertPair:
    """Paths of one application instance certificate (DER) and its private key (PEM)."""

    cert: Path
    key: Path


def cert_pair(cert_dir: Path, role: str) -> CertPair:
    """Standard file names for the server or client pair inside cert_dir."""
    return CertPair(cert_dir / f"{role}_cert.der", cert_dir / f"{role}_key.pem")


async def ensure_cert(cert_dir: Path, role: str) -> CertPair:
    """Create (or reuse while valid) a self-signed certificate for role server/client."""
    uri, usage = {
        "server": (SERVER_URI, ExtendedKeyUsageOID.SERVER_AUTH),
        "client": (CLIENT_URI, ExtendedKeyUsageOID.CLIENT_AUTH),
    }[role]
    pair = cert_pair(cert_dir, role)
    cert_dir.mkdir(parents=True, exist_ok=True)
    await cert_gen.setup_self_signed_certificate(
        pair.key, pair.cert, uri, socket.gethostname(), [usage], SUBJECT
    )
    return pair


async def secure_server(server: Server, cert_dir: Path) -> None:
    """Restrict the server to Basic256Sha256 SignAndEncrypt. Call after init(), before start()."""
    pair = await ensure_cert(cert_dir, "server")
    await server.set_application_uri(SERVER_URI)
    await server.load_certificate(pair.cert)
    await server.load_private_key(pair.key)
    server.set_security_policy([ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt])


def insecure_server(server: Server) -> None:
    """Explicit opt-in to an unencrypted endpoint (localhost testing only)."""
    server.set_security_policy([ua.SecurityPolicyType.NoSecurity])


async def secure_client(client: Client, cert_dir: Path) -> None:
    """Encrypt the session with the client pair and pin the server certificate on disk."""
    pair = await ensure_cert(cert_dir, "client")
    client.application_uri = CLIENT_URI
    await client.set_security(
        SecurityPolicyBasic256Sha256,
        str(pair.cert),
        str(pair.key),
        server_certificate=str(cert_pair(cert_dir, "server").cert),
        mode=ua.MessageSecurityMode.SignAndEncrypt,
    )
