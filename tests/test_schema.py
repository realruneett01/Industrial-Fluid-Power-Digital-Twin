"""Tests for multi-rate channel schema, cycle dataset loader, and interim Parquet storage."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydro.data.loader import load_cycle_dataset, save_interim
from hydro.data.schema import (
    CHANNEL_REGISTRY,
    PHYSICAL_CHANNELS,
    TARGET_COLUMNS,
    VIRTUAL_CHANNELS,
)


def test_channel_registry_composition() -> None:
    """Verify registry size and physical vs virtual channel partition."""
    assert len(CHANNEL_REGISTRY) == 17
    assert len(PHYSICAL_CHANNELS) == 14
    assert len(VIRTUAL_CHANNELS) == 3
    assert set(PHYSICAL_CHANNELS).isdisjoint(set(VIRTUAL_CHANNELS))


def test_synthetic_load_shapes_and_dtype(synthetic_raw_dir: Path) -> None:
    """Verify loaded arrays match channel sample rates and use float32."""
    data, targets = load_cycle_dataset(synthetic_raw_dir)
    assert data["PS1"].shape == (8, 6000)
    assert data["FS1"].shape == (8, 600)
    assert data["TS1"].shape == (8, 60)
    for arr in data.values():
        assert arr.dtype == np.float32
    assert targets.shape == (8, 5)
    assert list(targets.columns) == list(TARGET_COLUMNS)
    assert targets.dtypes.apply(lambda dt: dt == np.int64).all()


def test_synthetic_load_virtual_channel_filter(synthetic_raw_dir: Path) -> None:
    """Verify virtual channels are excluded by default and included on request."""
    data_default, _ = load_cycle_dataset(synthetic_raw_dir)
    for ch in VIRTUAL_CHANNELS:
        assert ch not in data_default
    assert len(data_default) == 14

    data_all, _ = load_cycle_dataset(synthetic_raw_dir, exclude_virtual=False)
    for ch in VIRTUAL_CHANNELS:
        assert ch in data_all
    assert len(data_all) == 17


def test_synthetic_load_mismatched_columns_raises(
    synthetic_raw_dir: Path,
    tmp_path: Path,
) -> None:
    """Verify corrupted column count raises ValueError naming the channel."""
    corrupted_dir = tmp_path / "corrupted_raw"
    shutil.copytree(synthetic_raw_dir, corrupted_dir)
    bad_data = np.random.default_rng(0).normal(50, 5, (8, 5999))
    np.savetxt(corrupted_dir / "PS1.txt", bad_data, delimiter="\t")

    with pytest.raises(ValueError, match="PS1"):
        load_cycle_dataset(corrupted_dir)


def test_save_interim_roundtrip(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    """Verify save_interim writes 4 Parquet files readable with expected row counts."""
    data, targets = load_cycle_dataset(synthetic_raw_dir)
    out_dir = tmp_path / "interim"
    paths = save_interim(data, targets, out_dir)

    assert len(paths) == 4
    expected_names = {
        "rate_100hz.parquet",
        "rate_10hz.parquet",
        "rate_1hz.parquet",
        "targets.parquet",
    }
    assert {p.name for p in paths} == expected_names

    for path in paths:
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 8

    df_targets = pd.read_parquet(out_dir / "targets.parquet")
    assert list(df_targets.columns) == list(TARGET_COLUMNS)


@pytest.mark.heavy
def test_real_dataset_shapes_and_sanitary(raw_dir: Path) -> None:
    """Validate full 2205-cycle real dataset dimensions, targets, and lack of NaN/Inf."""
    data, targets = load_cycle_dataset(raw_dir, exclude_virtual=True)

    assert data["PS1"].shape == (2205, 6000)
    assert data["FS1"].shape == (2205, 600)
    assert data["TS1"].shape == (2205, 60)
    assert targets.shape == (2205, 5)
    assert list(targets.columns) == list(TARGET_COLUMNS)

    for ch in ("CE", "CP", "SE"):
        assert ch not in data

    for name, arr in data.items():
        assert not np.isnan(arr).any(), f"NaN found in {name}"
        assert not np.isinf(arr).any(), f"Inf found in {name}"
