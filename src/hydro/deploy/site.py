"""Site folder: the generated, machine-specific state next to the portable program."""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

SITE_VERSION = 1
SITE_FILE = "site.json"
CALIBRATION_FILE = "calibration.json"
REPORT_FILE = "deploy_report.json"
CREDENTIALS_FILE = "credentials.json"
HEARTBEAT_FILE = "heartbeat.json"
ALERTS_FILE = "alerts.jsonl"
HEARTBEAT_MAX_AGE_S = 120.0


def write_json_atomic(path: Path, payload: dict[str, Any], private: bool = False) -> Path:
    """Write JSON via a temp file and rename, so an unplugged drive never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if private:
        tmp.chmod(0o600)  # no effect on FAT/exFAT or Windows ACLs; documented limitation
    tmp.replace(path)
    return path


def load_site(site_dir: Path) -> dict[str, Any]:
    """Read site.json written by ``hydro deploy``; ValueError when absent or incompatible."""
    path = site_dir / SITE_FILE
    try:
        site = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"no usable {path} - run `hydro deploy` first ({err})") from err
    if not isinstance(site, dict) or site.get("version") != SITE_VERSION:
        raise ValueError(f"{path} has an unsupported version")
    if not site.get("gate_passed"):
        raise ValueError(f"{path}: the deployment gate did not pass; refusing to run")
    return site


def loopback_credentials(site_dir: Path) -> dict[str, str]:
    """Random credentials for the self-hosted loopback replay; created once, reused after."""
    path = site_dir / CREDENTIALS_FILE
    if path.is_file():
        try:
            creds = json.loads(path.read_text(encoding="utf-8"))
            if creds.get("user") and creds.get("password"):
                return {"user": str(creds["user"]), "password": str(creds["password"])}
        except (OSError, ValueError):
            pass
    creds = {"user": "hydro", "password": secrets.token_urlsafe(24)}
    write_json_atomic(path, creds, private=True)
    return creds


def write_heartbeat(site_dir: Path, **fields: Any) -> None:
    write_json_atomic(site_dir / HEARTBEAT_FILE, {"time": time.time(), **fields})


def heartbeat_age(site_dir: Path) -> float | None:
    """Seconds since the last heartbeat, or None when there is none."""
    try:
        beat = json.loads((site_dir / HEARTBEAT_FILE).read_text(encoding="utf-8"))
        return max(0.0, time.time() - float(beat["time"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def default_site_dir(root: Path) -> Path:
    return Path(os.environ.get("HYDRO_SITE_DIR", root / "site"))
