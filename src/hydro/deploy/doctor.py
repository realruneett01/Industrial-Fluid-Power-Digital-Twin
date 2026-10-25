"""Environment preflight: can this machine run hydro at all?"""

from __future__ import annotations

import importlib
import shutil
import socket
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from hydro.deploy.machine import MIN_PYTHON, Machine, detect_machine
from hydro.deploy.results import CheckResult, Status

RUNTIME_MODULES = (
    "numpy",
    "scipy",
    "pandas",
    "yaml",
    "onnxruntime",
    "asyncua",
    "cryptography",
)
MIN_FREE_MB = 200
MIN_RAM_GB = 1.0
MIN_PLAUSIBLE_YEAR = 2024


def check_python(machine: Machine) -> CheckResult:
    ok = sys.version_info[:2] >= MIN_PYTHON
    need = ".".join(map(str, MIN_PYTHON))
    return CheckResult(
        "python",
        Status.PASS if ok else Status.FAIL,
        f"Python {machine.python} on {machine.os_name}/{machine.arch}",
        "" if ok else f"install Python {need} or newer",
    )


def check_modules() -> list[CheckResult]:
    results = []
    for mod in RUNTIME_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as err:  # broken wheels raise more than ImportError
            results.append(
                CheckResult(
                    f"module:{mod}",
                    Status.FAIL,
                    f"cannot import ({type(err).__name__}: {err})",
                    "reinstall from the wheelhouse: launch script with --repair",
                )
            )
        else:
            results.append(CheckResult(f"module:{mod}", Status.PASS, "importable"))
    return results


def check_writable(site_dir: Path) -> CheckResult:
    try:
        site_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=site_dir, prefix=".probe", delete=True) as f:
            f.write(b"x")
            f.flush()
    except OSError as err:
        return CheckResult(
            "site-dir-writable",
            Status.FAIL,
            f"{site_dir}: {err}",
            "the drive is read-only or full; unlock it or pass --site-dir on a writable path",
        )
    return CheckResult("site-dir-writable", Status.PASS, str(site_dir))


def check_disk(site_dir: Path) -> CheckResult:
    probe = site_dir if site_dir.exists() else site_dir.parent
    try:
        free_mb = shutil.disk_usage(probe).free // 2**20
    except OSError as err:
        return CheckResult("disk-space", Status.WARN, f"cannot determine free space: {err}")
    if free_mb >= MIN_FREE_MB:
        return CheckResult("disk-space", Status.PASS, f"{free_mb} MB free")
    return CheckResult(
        "disk-space", Status.FAIL, f"{free_mb} MB free", f"free at least {MIN_FREE_MB} MB"
    )


def check_ram(machine: Machine) -> CheckResult:
    if machine.ram_gb == 0:
        return CheckResult("memory", Status.WARN, "total RAM unknown")
    ok = machine.ram_gb >= MIN_RAM_GB
    return CheckResult(
        "memory",
        Status.PASS if ok else Status.WARN,
        f"{machine.ram_gb} GB",
        "" if ok else "under 1 GB: expect slow inference",
    )


def check_clock(now: datetime | None = None) -> CheckResult:
    """A dead RTC battery resets industrial PCs to 1970/2000 and breaks certificate validity."""
    current = now or datetime.now(UTC)
    ok = current.year >= MIN_PLAUSIBLE_YEAR
    return CheckResult(
        "system-clock",
        Status.PASS if ok else Status.FAIL,
        current.isoformat(timespec="seconds"),
        "" if ok else "set the system date and time; OPC-UA certificates depend on it",
    )


def check_port(port: int, host: str = "127.0.0.1") -> CheckResult:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        busy = s.connect_ex((host, port)) == 0
    return CheckResult(
        f"port:{port}",
        Status.WARN if busy else Status.PASS,
        "in use (another OPC-UA server may already run)" if busy else "free",
        "stop the other service or pass --endpoint with a different port" if busy else "",
    )


def run_doctor(site_dir: Path, machine: Machine | None = None) -> list[CheckResult]:
    """All environment checks, in dependency order."""
    m = machine or detect_machine()
    return [
        check_python(m),
        *check_modules(),
        check_writable(site_dir),
        check_disk(site_dir),
        check_ram(m),
        check_clock(),
    ]
