"""Tests for statistical feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydro.features.stats import extract_statistical_features


def test_random_rows_shape() -> None:
    rng = np.random.default_rng(42)
    data = {"PS1": rng.standard_normal((10, 6000))}
    df = extract_statistical_features(data)
    assert df.shape == (10, 11)
    assert list(df.index) == list(range(10))


def test_linear_row_slope() -> None:
    n_samples = 100
    x = np.arange(n_samples, dtype=np.float64)
    row = 2.0 * x + 5.0
    data = {"PS1": row[np.newaxis, :]}
    df = extract_statistical_features(data)
    assert np.isclose(df.loc[0, "PS1_slope"], 2.0)


def test_constant_row_values() -> None:
    data = {"PS1": np.full((3, 200), 7.5)}
    df = extract_statistical_features(data)
    assert (df["PS1_std"] == 0.0).all()
    assert (df["PS1_var"] == 0.0).all()
    assert (df["PS1_slope"] == 0.0).all()
    assert (df["PS1_skew"] == 0.0).all()
    assert (df["PS1_kurtosis"] == 0.0).all()
    assert not df.isna().any().any()


def test_two_channels_columns() -> None:
    rng = np.random.default_rng(0)
    data = {
        "PS1": rng.standard_normal((5, 100)),
        "FS1": rng.standard_normal((5, 50)),
    }
    df = extract_statistical_features(data)
    expected_metrics = [
        "mean",
        "std",
        "var",
        "skew",
        "kurtosis",
        "slope",
        "min",
        "max",
        "median",
        "p05",
        "p95",
    ]
    expected_columns = [f"PS1_{m}" for m in expected_metrics] + [
        f"FS1_{m}" for m in expected_metrics
    ]
    assert list(df.columns) == expected_columns
    assert df.shape == (5, 22)


def test_mismatched_cycles_raises_value_error() -> None:
    data = {
        "PS1": np.zeros((10, 50)),
        "FS1": np.zeros((8, 50)),
    }
    with pytest.raises(ValueError, match="cycles"):
        extract_statistical_features(data)


def test_1d_input_raises_value_error() -> None:
    data = {"PS1": np.zeros(50)}
    with pytest.raises(ValueError, match="2-D"):
        extract_statistical_features(data)


def test_3d_input_raises_value_error() -> None:
    data = {"PS1": np.zeros((5, 10, 2))}
    with pytest.raises(ValueError, match="2-D"):
        extract_statistical_features(data)


def test_empty_dict_returns_empty_dataframe() -> None:
    df = extract_statistical_features({})
    assert isinstance(df, pd.DataFrame)
    assert df.empty


@pytest.mark.heavy
def test_heavy_raw_data(raw_dir: Path) -> None:
    ps1_path = raw_dir / "PS1.txt"
    arr = np.loadtxt(ps1_path, delimiter="\t")
    df = extract_statistical_features({"PS1": arr})
    assert df.shape == (arr.shape[0], 11)
