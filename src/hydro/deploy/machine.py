"""Detect the host machine and derive an automatic runtime profile from it."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MIN_PYTHON = (3, 12)


@dataclass(frozen=True)
class Machine:
    """Facts about the host that influence configuration."""

    os_name: str
    arch: str
    python: str
    cpu_count: int
    ram_gb: float
    in_container: bool
    has_systemd: bool
    hostname: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ram_bytes() -> int | None:
    """Total physical memory in bytes, or None when the platform will not say."""
    try:
        if sys.platform == "win32":

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys)
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(size)
    except (AttributeError, OSError, ValueError):
        return None


def _in_container() -> bool:
    if Path("/.dockerenv").exists() or os.environ.get("container"):  # noqa: SIM112
        return True
    try:
        text = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(tag in text for tag in ("docker", "kubepods", "containerd", "lxc"))


def detect_machine() -> Machine:
    """Inspect the current host."""
    ram = _ram_bytes()
    return Machine(
        os_name=platform.system() or "unknown",
        arch=platform.machine().lower() or "unknown",
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=round(ram / 2**30, 2) if ram else 0.0,
        in_container=_in_container(),
        has_systemd=Path("/run/systemd/system").is_dir() and shutil.which("systemctl") is not None,
        hostname=platform.node() or "unknown",
    )


def auto_profile(machine: Machine) -> dict[str, Any]:
    """Pick runtime settings from the hardware.

    Only knobs that do not change model inputs are tuned: ONNX threads, OPC-UA publish
    interval and the number of calibration cycles. The feature window length is fixed by
    the trained models and is never adapted.
    """
    cpus = max(1, machine.cpu_count)
    if cpus <= 2 or 0 < machine.ram_gb < 2:
        tier, threads, publish_ms = "edge-lite", 1, 100
    elif cpus <= 4:
        tier, threads, publish_ms = "edge", 2, 50
    else:
        tier, threads, publish_ms = "workstation", min(4, cpus // 2), 50
    return {
        "tier": tier,
        "onnx_intra_op_threads": threads,
        "opcua_publish_ms": publish_ms,
        "calibration_cycles": 3 if tier == "edge-lite" else 5,
        "latency_sla_ms": 150.0,
    }
