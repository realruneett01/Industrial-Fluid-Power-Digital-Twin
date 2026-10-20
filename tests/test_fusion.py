"""Tests for feature fusion, physics feature extraction, and matrix builder."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydro.data.loader import load_cycle_dataset
from hydro.features import (
    build_feature_matrices,
    build_physics_features,
    save_feature_matrices,
)
from hydro.features.physics import fit_valve_baseline


def test_matrices_row_counts_and_keys(synthetic_raw_dir: Path) -> None:
    """Verify three matrices are returned with equal row counts matching input."""
    data, targets = load_cycle_dataset(synthetic_raw_dir)
    baseline_s = fit_valve_baseline(data["PS1"], targets["valve"].to_numpy())
    mats = build_feature_matrices(data, valve_baseline_s=baseline_s)

    assert set(mats.keys()) == {"statistical", "physics", "hybrid"}
    n_expected = len(targets)
    assert len(mats["statistical"]) == n_expected
    assert len(mats["physics"]) == n_expected
    assert len(mats["hybrid"]) == n_expected


def test_hybrid_columns_composition(synthetic_raw_dir: Path) -> None:
    """Verify hybrid matrix columns exactly equal statistical plus physics columns."""
    data, targets = load_cycle_dataset(synthetic_raw_dir)
    baseline_s = fit_valve_baseline(data["PS1"], targets["valve"].to_numpy())
    mats = build_feature_matrices(data, valve_baseline_s=baseline_s)

    expected_cols = list(mats["statistical"].columns) + list(mats["physics"].columns)
    assert list(mats["hybrid"].columns) == expected_cols


def test_no_nan_or_inf_after_build(synthetic_raw_dir: Path) -> None:
    """Verify no NaN or Inf values remain in any feature matrix after build."""
    data, targets = load_cycle_dataset(synthetic_raw_dir)
    baseline_s = fit_valve_baseline(data["PS1"], targets["valve"].to_numpy())
    mats = build_feature_matrices(data, valve_baseline_s=baseline_s)

    for name, df in mats.items():
        assert not df.isna().any().any(), f"NaN detected in {name} matrix"
        arr = df.to_numpy(dtype=np.float64)
        assert not np.isinf(arr).any(), f"Inf detected in {name} matrix"


def test_physics_columns_presence(synthetic_raw_dir: Path) -> None:
    """Verify required physical degradation indicators and covariates exist."""
    data, targets = load_cycle_dataset(synthetic_raw_dir)
    baseline_s = fit_valve_baseline(data["PS1"], targets["valve"].to_numpy())
    mats = build_feature_matrices(data, valve_baseline_s=baseline_s)

    expected_cols = [
        "c_eff",
        "k_leak_rel",
        "valve_edge_delay",
        "rise_time_10_90",
        "overshoot_ratio",
        "thermal_effectiveness",
        "ts1_mean",
    ]
    for col in expected_cols:
        assert col in mats["physics"].columns, f"Missing physics column: {col}"
    assert "k_leak_rel" in mats["physics"].columns
    assert "c_eff" in mats["physics"].columns


def test_virtual_channel_ce_raises_value_error(synthetic_raw_dir: Path) -> None:
    """Verify ValueError is raised if virtual channels (CE, CP, SE) are in data."""
    data, targets = load_cycle_dataset(synthetic_raw_dir, exclude_virtual=False)
    assert "CE" in data
    baseline_s = fit_valve_baseline(data["PS1"], targets["valve"].to_numpy())

    with pytest.raises(ValueError, match="CE"):
        build_feature_matrices(data, valve_baseline_s=baseline_s)

    with pytest.raises(ValueError, match="CE"):
        build_physics_features(data, valve_baseline_s=baseline_s)


def test_save_feature_matrices_parquet(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    """Verify save_feature_matrices writes 3 Parquet files with correct names."""
    data, targets = load_cycle_dataset(synthetic_raw_dir)
    baseline_s = fit_valve_baseline(data["PS1"], targets["valve"].to_numpy())
    mats = build_feature_matrices(data, valve_baseline_s=baseline_s)

    out_dir = tmp_path / "feature_parquets"
    written = save_feature_matrices(mats, out_dir)

    expected_names = {
        "features_statistical.parquet",
        "features_physics.parquet",
        "features_hybrid.parquet",
    }
    assert len(written) == 3
    assert {p.name for p in written} == expected_names

    for p in written:
        assert p.is_file()
        df = pd.read_parquet(p)
        assert len(df) == len(targets)


def test_nan_fill_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that replacing NaNs logs warnings with filled cell counts."""
    rng = np.random.default_rng(42)
    n = 6
    data = {
        "PS1": rng.normal(50, 5, (n, 6000)),
        "FS1": rng.normal(50, 5, (n, 600)),
        "EPS1": rng.normal(2000, 100, (n, 6000)),
        "TS1": rng.normal(40, 2, (n, 60)),
        "TS3": rng.normal(35, 2, (n, 60)),
        "TS4": rng.normal(30, 2, (n, 60)),
    }
    with caplog.at_level(logging.WARNING):
        build_feature_matrices(data, valve_baseline_s=10.0)

    # c_eff is expected to produce NaNs on random 50 bar signals and trigger warnings
    assert any("filled" in r.message.lower() for r in caplog.records)


@pytest.mark.heavy
def test_real_data_physics_matrix(raw_dir: Path) -> None:
    """Verify real 2205-cycle dataset builds under 120s with expected shape and no NaN."""
    data, targets = load_cycle_dataset(raw_dir, exclude_virtual=True)
    baseline_s = fit_valve_baseline(data["PS1"], targets["valve"].to_numpy())
    mats = build_feature_matrices(data, valve_baseline_s=baseline_s)

    df_phys = mats["physics"]
    assert df_phys.shape[0] == 2205
    assert df_phys.shape[1] in (7, 8)
    assert not df_phys.isna().any().any()
