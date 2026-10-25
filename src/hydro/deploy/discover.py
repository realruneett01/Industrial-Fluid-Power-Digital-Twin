"""Find what sensor sources are reachable from this machine.

Discovery is read-only and local-first: configured OPC-UA endpoints, datasets on the drive,
and serial ports (listed for information; hydro itself reads sensors through OPC-UA).
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydro.data.schema import CHANNEL_REGISTRY
from hydro.online.replay_server import DEFAULT_ENDPOINT

ENDPOINTS_ENV = "HYDRO_OPCUA_ENDPOINTS"  # comma separated
EXTRA_ENDPOINTS = ("opc.tcp://localhost:4840", "opc.tcp://127.0.0.1:4840")
PROBE_TIMEOUT_S = 3.0


@dataclass
class Discovery:
    """Everything found; ``best`` ranks live OPC-UA above a dataset replay."""

    endpoints: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    serial_ports: list[str] = field(default_factory=list)

    def reachable_endpoints(self) -> list[dict[str, Any]]:
        return [e for e in self.endpoints if e["reachable"]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoints": self.endpoints,
            "datasets": self.datasets,
            "serial_ports": self.serial_ports,
        }


def candidate_endpoints(extra: list[str] | None = None) -> list[str]:
    """Ordered, de-duplicated endpoint list: explicit, environment, defaults."""
    env = [e.strip() for e in os.environ.get(ENDPOINTS_ENV, "").split(",") if e.strip()]
    ordered = [*(extra or []), *env, DEFAULT_ENDPOINT, *EXTRA_ENDPOINTS]
    return list(dict.fromkeys(ordered))


async def probe_endpoint(url: str, timeout: float = PROBE_TIMEOUT_S) -> dict[str, Any]:
    """Ask an OPC-UA server which security policies it offers (no session, no credentials)."""
    from asyncua import Client

    info: dict[str, Any] = {"url": url, "reachable": False, "policies": [], "error": ""}
    client = Client(url, timeout=timeout)
    try:
        eps = await asyncio.wait_for(client.connect_and_get_server_endpoints(), timeout + 1)
    except Exception as err:  # noqa: BLE001  # any failure means "not reachable"
        info["error"] = type(err).__name__
        return info
    finally:
        with contextlib.suppress(Exception):  # nothing to disconnect when connect failed
            await client.disconnect()
    info["reachable"] = True
    info["policies"] = sorted({ep.SecurityPolicyUri.rsplit("#", 1)[-1] for ep in eps})
    return info


def find_datasets(root: Path) -> list[str]:
    """Directories below root (max depth 3) that hold a complete ZeMA-format dataset."""
    needed = {"profile.txt", *(f"{n}.txt" for n, c in CHANNEL_REGISTRY.items() if not c.is_virtual)}
    found = []
    base = root.resolve()
    for dirpath, dirnames, filenames in os.walk(base):
        depth = len(Path(dirpath).relative_to(base).parts)
        dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "__pycache__", "site"}]
        if depth >= 3:
            dirnames[:] = []
        if needed.issubset(filenames):
            found.append(dirpath)
    return sorted(found)


def list_serial_ports() -> list[str]:
    """Serial devices the OS exposes; informational only."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM")
        except (ImportError, OSError):
            return []
        ports = []
        with key:
            i = 0
            while True:
                try:
                    ports.append(str(winreg.EnumValue(key, i)[1]))
                except OSError:
                    break
                i += 1
        return sorted(ports)
    patterns = ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/cu.usbserial*", "/dev/cu.usbmodem*")
    return sorted(p for pat in patterns for p in glob.glob(pat))


async def discover(
    root: Path, extra_endpoints: list[str] | None = None, timeout: float = PROBE_TIMEOUT_S
) -> Discovery:
    """Probe all candidate endpoints concurrently and scan the drive for datasets."""
    urls = candidate_endpoints(extra_endpoints)
    probes = await asyncio.gather(*(probe_endpoint(u, timeout) for u in urls))
    return Discovery(
        endpoints=list(probes),
        datasets=find_datasets(root),
        serial_ports=list_serial_ports(),
    )
