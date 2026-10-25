"""SHA-256 manifest of a deployment folder (pendrive): tamper and corruption evidence.

An optional HMAC-SHA256 signature (key from ``HYDRO_MANIFEST_KEY``) binds the manifest to a
secret, so a file edit plus a regenerated manifest is detected by anyone holding the key.
This is integrity checking, not copy protection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_NAME = "MANIFEST.sha256.json"
KEY_ENV = "HYDRO_MANIFEST_KEY"
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "site",
        "certs",
        ".pytest_cache",
        ".ruff_cache",
    }
)
EXCLUDED_SUFFIXES = (".pyc", ".log", ".tmp")
CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if name == MANIFEST_NAME or name.endswith(EXCLUDED_SUFFIXES) or path.is_symlink():
                continue
            yield path


def _signature(files: dict[str, dict[str, Any]], key: str) -> str:
    body = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(key.encode(), body, hashlib.sha256).hexdigest()


def build_manifest(root: Path | str, key: str | None = None) -> dict[str, Any]:
    """Hash every deployable file below root (posix-style relative paths)."""
    base = Path(root).resolve()
    files = {
        p.relative_to(base).as_posix(): {"sha256": _sha256(p), "size": p.stat().st_size}
        for p in _iter_files(base)
    }
    manifest: dict[str, Any] = {
        "version": 1,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    if key:
        manifest["signature"] = _signature(files, key)
    return manifest


def write_manifest(root: Path | str, key: str | None = None) -> Path:
    """Write MANIFEST.sha256.json into root and return its path."""
    base = Path(root)
    path = base / MANIFEST_NAME
    path.write_text(json.dumps(build_manifest(base, key), indent=2) + "\n", encoding="utf-8")
    return path


def verify_manifest(root: Path | str, key: str | None = None) -> dict[str, Any]:
    """Compare root against its manifest.

    Returns ``{"ok", "missing", "changed", "extra", "signature"}`` where signature is
    "valid", "invalid", "unsigned" or "unchecked" (manifest signed but no key given).
    """
    base = Path(root).resolve()
    path = base / MANIFEST_NAME
    result: dict[str, Any] = {
        "ok": False,
        "missing": [],
        "changed": [],
        "extra": [],
        "signature": "unchecked",
    }
    if not path.is_file():
        result["error"] = f"{MANIFEST_NAME} not found"
        return result
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = manifest["files"]
        if not isinstance(expected, dict):
            raise TypeError("files must be an object")
    except (OSError, ValueError, KeyError, TypeError) as err:
        result["error"] = f"unreadable manifest: {err}"
        return result
    for rel, meta in expected.items():
        target = (base / rel).resolve()
        if base not in target.parents or not target.is_file():
            result["missing"].append(rel)
        elif not isinstance(meta, dict) or _sha256(target) != meta.get("sha256"):
            result["changed"].append(rel)
    result["extra"] = sorted(
        p.relative_to(base).as_posix()
        for p in _iter_files(base)
        if p.relative_to(base).as_posix() not in expected
    )
    sig = manifest.get("signature")
    if sig is None:
        result["signature"] = "unsigned" if not key else "invalid"
    elif key:
        ok = hmac.compare_digest(str(sig), _signature(expected, key))
        result["signature"] = "valid" if ok else "invalid"
    result["ok"] = not (result["missing"] or result["changed"]) and result["signature"] != "invalid"
    return result
