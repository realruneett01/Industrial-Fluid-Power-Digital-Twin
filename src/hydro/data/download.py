"""Dataset download, verification, and extraction for ZeMA hydraulic system data."""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.request
import zipfile
from pathlib import Path

import yaml

from hydro.data.schema import CHANNEL_REGISTRY
from hydro.paths import project_root

CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming buffer
CONFIG_PATH = project_root() / "configs" / "data.yaml"


def _load_defaults() -> tuple[str, str]:
    """Load default dataset URL and SHA-256 digest from configs/data.yaml."""
    if not CONFIG_PATH.is_file():
        return "", ""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return "", ""
    dataset_cfg = cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        return "", ""
    url = str(dataset_cfg.get("url") or "")
    sha = str(dataset_cfg.get("sha256") or "")
    return url, sha


def _dump_digest_fallback(cfg_path: Path, digest: str) -> None:
    """Write digest via yaml.safe_dump when no sha256 line exists."""
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    dataset_cfg = cfg.setdefault("dataset", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
        cfg["dataset"] = dataset_cfg
    dataset_cfg["sha256"] = digest
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _write_digest(cfg_path: Path, digest: str) -> None:
    """Write computed SHA-256 digest to dataset.sha256 in config file."""
    if not cfg_path.is_file():
        return
    content = cfg_path.read_text(encoding="utf-8")
    pattern = r"(^dataset:[^\r\n]*\r?\n(?:[ \t]+[^\r\n]*\r?\n|[ \t]*\r?\n)*?[ \t]+sha256:)[^\r\n]*"
    updated, count = re.subn(pattern, rf"\g<1> {digest}", content, count=1, flags=re.MULTILINE)
    if count > 0:
        cfg_path.write_text(updated, encoding="utf-8")
        return
    _dump_digest_fallback(cfg_path, digest)


def _is_complete(dest_dir: Path) -> bool:
    """Return True if profile.txt and all 17 channel files exist in dest_dir."""
    if not (dest_dir / "profile.txt").is_file():
        return False
    return all((dest_dir / f"{ch}.txt").is_file() for ch in CHANNEL_REGISTRY)


def _download(url: str, zip_path: Path) -> str:
    """Download url to zip_path in 1 MiB chunks, returning computed hex digest."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(zip_path, "wb") as out_file:
        while chunk := response.read(CHUNK_SIZE):
            hasher.update(chunk)
            out_file.write(chunk)
    return hasher.hexdigest()


def _verify(zip_path: Path, computed: str, expected: str | None) -> None:
    """Verify computed SHA-256 against expected digest, deleting zip on mismatch."""
    if not expected:
        logging.warning("No expected SHA-256 digest provided (trust on first use): %s", computed)
        return
    if computed.lower() != expected.lower():
        if zip_path.exists():
            zip_path.unlink()
        raise ValueError(
            f"SHA-256 digest mismatch for {zip_path.name}: expected {expected}, got {computed}"
        )


def _extract(zip_path: Path, dest_dir: Path) -> None:
    """Extract 17 channel txt files and profile.txt into dest_dir, guarding against zip-slip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    allowed = {f"{ch}.txt" for ch in CHANNEL_REGISTRY} | {"profile.txt"}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if any(sep in name for sep in ("/", "\\", "..")):
                continue
            if name in allowed:
                (dest_dir / name).write_bytes(zf.read(name))


def _resolve_url_and_sha(url: str | None, sha256: str | None) -> tuple[str, str]:
    """Resolve effective dataset URL and expected SHA-256 digest."""
    if url is None:
        default_url, default_sha = _load_defaults()
        eff_sha = default_sha if sha256 is None else str(sha256)
        return default_url, eff_sha
    eff_sha = "" if sha256 is None else str(sha256)
    return str(url), eff_sha


def _download_and_verify(url: str, zip_path: Path, expected_sha: str) -> str:
    """Download archive from url and verify digest, returning computed hash."""
    computed = _download(url, zip_path)
    _verify(zip_path, computed, expected_sha)
    return computed


def _handle_tofu(
    url_param: str | None,
    sha_param: str | None,
    effective_sha: str,
    computed: str,
) -> None:
    """Write computed digest if default url is used and config sha is missing (TOFU)."""
    if all((url_param is None, sha_param is None, not effective_sha)):
        _write_digest(CONFIG_PATH, computed)


def fetch_zema_dataset(
    dest_dir: Path | str,
    force: bool = False,
    url: str | None = None,
    sha256: str | None = None,
) -> Path:
    """Download, verify, and extract the ZeMA hydraulic dataset to dest_dir."""
    dest = Path(dest_dir)
    if not force and _is_complete(dest):
        return dest

    effective_url, effective_sha = _resolve_url_and_sha(url, sha256)
    if not effective_url:
        raise ValueError("Dataset URL is not specified or configured.")

    zip_path = dest.parent / "zema.zip"
    computed = _download_and_verify(effective_url, zip_path, effective_sha)
    _handle_tofu(url, sha256, effective_sha, computed)
    _extract(zip_path, dest)
    return dest
