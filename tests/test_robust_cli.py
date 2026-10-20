"""Robustness tests for CLI and dataset loader against malformed inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hydro.cli import main
from hydro.data.loader import load_cycle_dataset
from hydro.data.schema import CHANNEL_REGISTRY


def _make_fake_raw(dest: Path, n_cycles: int = 4) -> None:
    """Create a minimal synthetic raw dataset in dest directory."""
    rng = np.random.default_rng(42)
    for name, ch in CHANNEL_REGISTRY.items():
        if not ch.is_virtual:
            data = rng.normal(50.0, 1.0, (n_cycles, ch.n_samples)).astype(np.float32)
            np.savetxt(dest / f"{name}.txt", data, delimiter="\t", fmt="%.2f")
    profile = np.tile([[100, 100, 0, 130, 0], [3, 73, 2, 90, 1]], (n_cycles // 2, 1))
    np.savetxt(dest / "profile.txt", profile, delimiter="\t", fmt="%d")


def test_cli_unknown_subcommand() -> None:
    """Unknown subcommands must exit with a non-zero exit code."""
    with pytest.raises(SystemExit) as exc_info:
        main(["unknown_subcommand"])
    assert exc_info.value.code != 0


def test_cli_missing_required_args() -> None:
    """Missing required arguments must exit with a non-zero exit code."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_cli_missing_data_dir(tmp_path: Path) -> None:
    """Missing data directory must raise FileNotFoundError naming the directory or file."""
    missing_dir = tmp_path / "nonexistent_dir"
    with pytest.raises(FileNotFoundError) as exc_info:
        main(["bundle", "--data-dir", str(missing_dir)])
    msg = str(exc_info.value)
    assert any((missing_dir.name in msg, "PS1.txt" in msg, "profile.txt" in msg))


def test_cli_empty_data_dir(tmp_path: Path) -> None:
    """Empty data directory must raise FileNotFoundError naming the missing file."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError) as exc_info:
        main(["bundle", "--data-dir", str(empty_dir)])
    msg = str(exc_info.value)
    assert any(("PS1.txt" in msg, "profile.txt" in msg, empty_dir.name in msg))


@pytest.mark.parametrize("channel", ["PS1", "FS1"])
def test_cli_truncated_channel(tmp_path: Path, channel: str) -> None:
    """Truncated channel file must raise ValueError naming the offending channel file."""
    data_dir = tmp_path / f"trunc_{channel}"
    data_dir.mkdir()
    _make_fake_raw(data_dir, n_cycles=4)
    ch = CHANNEL_REGISTRY[channel]
    bad = np.zeros((2, ch.n_samples), dtype=np.float32)
    np.savetxt(data_dir / f"{channel}.txt", bad, delimiter="\t", fmt="%.2f")

    with pytest.raises(ValueError) as exc_info:
        main(["bundle", "--data-dir", str(data_dir)])
    msg = str(exc_info.value)
    assert any((f"{channel}.txt" in msg, channel in msg))


@pytest.mark.parametrize("channel", ["PS1", "FS1"])
def test_cli_wrong_column_count(tmp_path: Path, channel: str) -> None:
    """Channel with wrong column count must raise ValueError naming the offending file."""
    data_dir = tmp_path / f"bad_cols_{channel}"
    data_dir.mkdir()
    _make_fake_raw(data_dir, n_cycles=4)
    bad = np.zeros((4, 10), dtype=np.float32)
    np.savetxt(data_dir / f"{channel}.txt", bad, delimiter="\t", fmt="%.2f")

    with pytest.raises(ValueError) as exc_info:
        main(["bundle", "--data-dir", str(data_dir)])
    msg = str(exc_info.value)
    assert any((f"{channel}.txt" in msg, channel in msg))


@pytest.mark.parametrize("channel", ["PS1", "FS1"])
def test_cli_non_numeric_text(tmp_path: Path, channel: str) -> None:
    """Channel with non-numeric text must raise ValueError naming the offending file."""
    data_dir = tmp_path / f"text_{channel}"
    data_dir.mkdir()
    _make_fake_raw(data_dir, n_cycles=4)
    (data_dir / f"{channel}.txt").write_text("invalid_float\t1.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        main(["bundle", "--data-dir", str(data_dir)])
    msg = str(exc_info.value)
    assert any((f"{channel}.txt" in msg, channel in msg))


@pytest.mark.parametrize("col_count", [4, 6])
def test_cli_profile_wrong_columns(tmp_path: Path, col_count: int) -> None:
    """profile.txt with wrong column count must raise ValueError naming profile.txt."""
    data_dir = tmp_path / f"profile_cols_{col_count}"
    data_dir.mkdir()
    _make_fake_raw(data_dir, n_cycles=4)
    bad_profile = np.zeros((4, col_count), dtype=int)
    np.savetxt(data_dir / "profile.txt", bad_profile, delimiter="\t", fmt="%d")

    with pytest.raises(ValueError) as exc_info:
        main(["bundle", "--data-dir", str(data_dir)])
    assert "profile.txt" in str(exc_info.value)


def test_loader_validates_shapes_and_rows(tmp_path: Path) -> None:
    """load_cycle_dataset must validate per-channel shapes and row counts against profile.txt."""
    _make_fake_raw(tmp_path, n_cycles=4)
    data, targets = load_cycle_dataset(tmp_path)
    assert data["PS1"].shape == (4, CHANNEL_REGISTRY["PS1"].n_samples)
    assert len(targets) == 4

    ch = CHANNEL_REGISTRY["PS1"]
    extra_rows = np.zeros((6, ch.n_samples), dtype=np.float32)
    np.savetxt(tmp_path / "PS1.txt", extra_rows, delimiter="\t", fmt="%.2f")
    with pytest.raises(ValueError) as exc_info:
        load_cycle_dataset(tmp_path)
    assert any(("PS1.txt" in str(exc_info.value), "PS1" in str(exc_info.value)))


def test_loader_profile_non_numeric(tmp_path: Path) -> None:
    """load_cycle_dataset must raise ValueError naming profile.txt when non-numeric."""
    _make_fake_raw(tmp_path, n_cycles=4)
    (tmp_path / "profile.txt").write_text("bad\t100\t0\t130\t0\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_cycle_dataset(tmp_path)
    assert "profile.txt" in str(exc_info.value)
