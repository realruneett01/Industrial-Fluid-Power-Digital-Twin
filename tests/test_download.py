"""Tests for ZeMA dataset downloading, verification, and extraction."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
import yaml

from hydro.data.download import fetch_zema_dataset
from hydro.data.schema import CHANNEL_REGISTRY

EXPECTED_FILES = [f"{ch}.txt" for ch in CHANNEL_REGISTRY] + ["profile.txt"]


def _create_sample_zip(
    target_path: Path,
    extra_members: list[tuple[str, bytes]] | None = None,
) -> str:
    """Create a synthetic dataset zip file and return its SHA-256 hex digest."""
    with zipfile.ZipFile(target_path, "w") as zf:
        for name in EXPECTED_FILES:
            zf.writestr(name, f"header\ncontent of {name}\n".encode())
        if extra_members:
            for extra_name, content in extra_members:
                zf.writestr(extra_name, content)
    return hashlib.sha256(target_path.read_bytes()).hexdigest()


def test_fetch_zema_dataset_correct_digest(tmp_path: Path) -> None:
    """Correct SHA-256 digest extracts all 18 dataset files into dest_dir."""
    source_zip = tmp_path / "source.zip"
    digest = _create_sample_zip(source_zip)
    dest_dir = tmp_path / "extracted"

    res = fetch_zema_dataset(dest_dir, url=source_zip.as_uri(), sha256=digest)

    assert res == dest_dir
    assert (dest_dir / "profile.txt").is_file()
    assert all((dest_dir / f"{ch}.txt").is_file() for ch in CHANNEL_REGISTRY)
    assert len(list(dest_dir.iterdir())) == 18


def test_fetch_zema_dataset_wrong_digest(tmp_path: Path) -> None:
    """Mismatched SHA-256 digest raises ValueError and removes downloaded zip."""
    source_zip = tmp_path / "source.zip"
    actual_digest = _create_sample_zip(source_zip)
    dest_dir = tmp_path / "extracted"
    zip_path = dest_dir.parent / "zema.zip"
    bad_digest = "0" * 64

    with pytest.raises(ValueError, match="SHA-256") as exc_info:
        fetch_zema_dataset(dest_dir, url=source_zip.as_uri(), sha256=bad_digest)

    error_msg = str(exc_info.value)
    assert bad_digest in error_msg
    assert actual_digest in error_msg
    assert not zip_path.exists()


def test_fetch_zema_dataset_complete_cache(tmp_path: Path) -> None:
    """Complete dest_dir returns immediately without network access."""
    dest_dir = tmp_path / "cached"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_FILES:
        (dest_dir / name).write_text("cached")

    res = fetch_zema_dataset(
        dest_dir,
        force=False,
        url="invalid://not-a-valid-url",
        sha256="bad_digest",
    )
    assert res == dest_dir


def test_fetch_zema_dataset_zip_slip_ignored(tmp_path: Path) -> None:
    """Zip-slip archive members containing slash, backslash or .. are skipped."""
    source_zip = tmp_path / "source.zip"
    malicious = [
        ("../slip.txt", b"slip content"),
        ("subdir/inner.txt", b"nested content"),
        ("bad\\back.txt", b"backslash content"),
    ]
    digest = _create_sample_zip(source_zip, extra_members=malicious)
    dest_dir = tmp_path / "extracted"

    fetch_zema_dataset(dest_dir, url=source_zip.as_uri(), sha256=digest)

    assert not (tmp_path / "slip.txt").exists()
    assert not (dest_dir / "slip.txt").exists()
    assert not (dest_dir / "subdir").exists()
    extracted_names = {p.name for p in dest_dir.iterdir()}
    assert extracted_names == set(EXPECTED_FILES)


def test_fetch_zema_dataset_trust_on_first_use(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty sha256 string logs a warning with computed digest and extracts."""
    source_zip = tmp_path / "source.zip"
    computed = _create_sample_zip(source_zip)
    dest_dir = tmp_path / "extracted"

    with caplog.at_level("WARNING"):
        res = fetch_zema_dataset(dest_dir, url=source_zip.as_uri(), sha256="")

    assert res == dest_dir
    assert computed in caplog.text
    assert (dest_dir / "profile.txt").is_file()


def test_fetch_zema_dataset_null_sha256_preserves_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Null sha256 in config does not delete the downloaded zip file."""
    source_zip = tmp_path / "source.zip"
    _create_sample_zip(source_zip)
    dest_dir = tmp_path / "extracted"
    zip_path = dest_dir.parent / "zema.zip"

    cfg_path = tmp_path / "data.yaml"
    cfg_data = {"dataset": {"url": source_zip.as_uri(), "sha256": None}}
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    monkeypatch.setattr("hydro.data.download.CONFIG_PATH", cfg_path)

    res = fetch_zema_dataset(dest_dir)

    assert res == dest_dir
    assert zip_path.is_file()
    assert (dest_dir / "profile.txt").is_file()


def test_fetch_zema_dataset_trust_on_first_use_persists_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TOFU writes digest to config; subsequent run with corrupt zip raises ValueError."""
    source_zip = tmp_path / "source.zip"
    digest = _create_sample_zip(source_zip)
    dest_dir = tmp_path / "extracted"

    cfg_path = tmp_path / "data.yaml"
    cfg_data = {"dataset": {"url": source_zip.as_uri(), "sha256": None}, "extra_key": 42}
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    monkeypatch.setattr("hydro.data.download.CONFIG_PATH", cfg_path)

    fetch_zema_dataset(dest_dir)
    updated_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert updated_cfg["dataset"]["sha256"] == digest
    assert updated_cfg["extra_key"] == 42

    source_zip.write_bytes(b"corrupted zip content")
    with pytest.raises(ValueError, match="SHA-256 digest mismatch"):
        fetch_zema_dataset(dest_dir, force=True)


def test_fetch_zema_dataset_custom_url_no_sha_ignores_config_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom url with sha256=None keeps zip and leaves config digest untouched."""
    source_zip = tmp_path / "source.zip"
    _create_sample_zip(source_zip)
    dest_dir = tmp_path / "extracted"
    zip_path = dest_dir.parent / "zema.zip"

    cfg_path = tmp_path / "data.yaml"
    original_cfg = (
        "dataset:\n"
        "  url: https://example.com/other.zip\n"
        "  sha256: deadbeef00000000000000000000000000000000000000000000000000000000\n"
    )
    cfg_path.write_text(original_cfg, encoding="utf-8")
    monkeypatch.setattr("hydro.data.download.CONFIG_PATH", cfg_path)

    res = fetch_zema_dataset(dest_dir, url=source_zip.as_uri())

    assert res == dest_dir
    assert zip_path.is_file()
    assert (dest_dir / "profile.txt").is_file()
    assert cfg_path.read_text(encoding="utf-8") == original_cfg


def test_fetch_zema_dataset_tofu_preserves_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TOFU writes digest via regex substitution, preserving yaml comments."""
    source_zip = tmp_path / "source.zip"
    digest = _create_sample_zip(source_zip)
    dest_dir = tmp_path / "extracted"

    cfg_path = tmp_path / "data.yaml"
    cfg_text = (
        f"# Top\ndataset:\n  # URL\n  url: {source_zip.as_uri()}\n"
        "  # TOFU\n  sha256: null\n  n_cycles: 2205\n# Trailing\n"
    )
    cfg_path.write_text(cfg_text, encoding="utf-8")
    monkeypatch.setattr("hydro.data.download.CONFIG_PATH", cfg_path)

    fetch_zema_dataset(dest_dir)

    updated_text = cfg_path.read_text(encoding="utf-8")
    for comment in ("# Top", "# URL", "# TOFU", "# Trailing"):
        assert comment in updated_text
    updated_cfg = yaml.safe_load(updated_text)
    assert updated_cfg["dataset"]["sha256"] == digest


@pytest.mark.heavy
def test_fetch_zema_dataset_real_data(raw_dir: Path) -> None:
    """Verify complete check against real dataset directory when present."""
    res = fetch_zema_dataset(raw_dir)
    assert res == raw_dir
