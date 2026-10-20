"""Comprehensive numerical and mathematical tests for statistical features.

Verifies statistical metrics against independent mathematical oracles:
- NumPy / SciPy reference implementations
- Closed-form derivations derived by hand
- Mathematical invariances (shift, scale, symmetry, constant)
- Multi-rate signals across 100 Hz (6000), 10 Hz (600), and 1 Hz (60)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import stats

from hydro.features.stats import extract_statistical_features

if TYPE_CHECKING:
    import pandas as pd


CHANNEL_CASES = ((6000, "PS1"), (600, "FS1"), (60, "TS1"))


def _make_seeded_signal(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate reproducible 4-cycle multi-distribution test signal array."""
    rng = np.random.default_rng(seed)
    c0 = rng.standard_normal(n_samples)
    c1 = rng.uniform(-3.0, 3.0, n_samples)
    mask = rng.random(n_samples) > 0.5
    c2 = np.where(
        mask,
        rng.normal(2.0, 0.5, n_samples),
        rng.normal(-1.0, 1.2, n_samples),
    )
    c3 = rng.exponential(scale=2.0, size=n_samples) - 2.0
    return np.vstack([c0, c1, c2, c3])


def _extract_seeded(n_samples: int, channel: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract statistical features from seeded test signal array."""
    arr = _make_seeded_signal(n_samples)
    return arr, extract_statistical_features({channel: arr})


def _assert_metrics_allclose(
    df: pd.DataFrame,
    channel: str,
    expected_metrics: dict[str, float | np.ndarray],
    rtol: float = 1e-10,
) -> None:
    """Assert multiple extracted statistical features match expected values."""
    for metric, exp_val in expected_metrics.items():
        arr_val = df[f"{channel}_{metric}"].to_numpy().squeeze()
        atol = 1e-10 if np.all(np.asarray(exp_val) == 0.0) else 0.0
        np.testing.assert_allclose(arr_val, exp_val, rtol=rtol, atol=atol)


def _evaluate_transform_invariance(
    transform_fn: Callable[[np.ndarray], np.ndarray],
    expected_metrics_fn: Callable[[pd.DataFrame], dict[str, float | np.ndarray]],
) -> None:
    """Verify statistical feature behavior under signal transformation across seeds."""
    for seed in (42, 101):
        arr = _make_seeded_signal(600, seed=seed)
        df_orig = extract_statistical_features({"FS1": arr})
        df_trans = extract_statistical_features({"FS1": transform_fn(arr)})
        _assert_metrics_allclose(
            df_trans,
            "FS1",
            expected_metrics_fn(df_orig),
            rtol=1e-10,
        )


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_mean_against_numpy_oracle(n_samples: int, channel: str) -> None:
    # Derivation: Sample mean is (1/N) * sum_{i=0}^{N-1} x_i
    arr, df = _extract_seeded(n_samples, channel)
    expected = np.mean(arr, axis=1)
    np.testing.assert_allclose(df[f"{channel}_mean"].to_numpy(), expected, rtol=1e-12)


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_std_and_var_ddof0_oracle(n_samples: int, channel: str) -> None:
    # Derivation: ddof=0 population variance is (1/N) * sum((x_i - mu)^2), std = sqrt(var)
    arr, df = _extract_seeded(n_samples, channel)
    exp_std = np.std(arr, axis=1, ddof=0)
    exp_var = np.var(arr, axis=1, ddof=0)
    _assert_metrics_allclose(df, channel, {"std": exp_std, "var": exp_var}, rtol=1e-12)
    np.testing.assert_allclose(df[f"{channel}_std"] ** 2, df[f"{channel}_var"], rtol=1e-12)


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_min_max_extremes_oracle(n_samples: int, channel: str) -> None:
    # Derivation: Order statistics min = x_{(0)}, max = x_{(N-1)} with min <= max
    arr, df = _extract_seeded(n_samples, channel)
    _assert_metrics_allclose(
        df,
        channel,
        {"min": np.min(arr, axis=1), "max": np.max(arr, axis=1)},
        rtol=1e-12,
    )
    assert (df[f"{channel}_min"] <= df[f"{channel}_max"]).all()


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_median_and_percentiles_oracle(n_samples: int, channel: str) -> None:
    # Derivation: Quantiles satisfy monotonicity: min <= p05 <= median (p50) <= p95 <= max
    arr, df = _extract_seeded(n_samples, channel)
    _assert_metrics_allclose(
        df,
        channel,
        {
            "median": np.median(arr, axis=1),
            "p05": np.percentile(arr, 5, axis=1),
            "p95": np.percentile(arr, 95, axis=1),
        },
        rtol=1e-12,
    )
    assert (
        (df[f"{channel}_p05"] <= df[f"{channel}_median"])
        & (df[f"{channel}_median"] <= df[f"{channel}_p95"])
    ).all()


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_skewness_against_scipy_fisher_pearson(n_samples: int, channel: str) -> None:
    # Derivation: Fisher-Pearson skewness is m_3 / (m_2^(3/2)) with biased sample moments
    arr, df = _extract_seeded(n_samples, channel)
    expected = stats.skew(arr, axis=1, bias=True)
    np.testing.assert_allclose(df[f"{channel}_skew"].to_numpy(), expected, rtol=1e-10)


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_kurtosis_against_scipy_fisher_excess(n_samples: int, channel: str) -> None:
    # Derivation: Fisher excess kurtosis is (m_4 / m_2^2) - 3 (normal distribution = 0)
    arr, df = _extract_seeded(n_samples, channel)
    expected = stats.kurtosis(arr, axis=1, fisher=True, bias=True)
    np.testing.assert_allclose(df[f"{channel}_kurtosis"].to_numpy(), expected, rtol=1e-10)


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_slope_against_numpy_polyfit(n_samples: int, channel: str) -> None:
    # Derivation: OLS slope beta_1 = sum((i - i_bar)*y_i) / sum((i - i_bar)^2)
    arr, df = _extract_seeded(n_samples, channel)
    x = np.arange(n_samples, dtype=np.float64)
    expected = np.array([np.polyfit(x, row, deg=1)[0] for row in arr])
    np.testing.assert_allclose(df[f"{channel}_slope"].to_numpy(), expected, rtol=1e-10)


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_rms_mathematical_identity_oracle(n_samples: int, channel: str) -> None:
    # Derivation: RMS^2 = (1/N) * sum(x_i^2) = Var(X) + Mean(X)^2
    arr, df = _extract_seeded(n_samples, channel)
    expected_rms = np.sqrt(np.mean(arr**2, axis=1))
    feature_rms = np.sqrt(df[f"{channel}_mean"] ** 2 + df[f"{channel}_var"])
    np.testing.assert_allclose(feature_rms.to_numpy(), expected_rms, rtol=1e-12)


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_peak_to_peak_range_oracle(n_samples: int, channel: str) -> None:
    # Derivation: Peak-to-peak amplitude is max(x) - min(x) == np.ptp(x)
    arr, df = _extract_seeded(n_samples, channel)
    feature_ptp = df[f"{channel}_max"] - df[f"{channel}_min"]
    expected_ptp = np.ptp(arr, axis=1)
    np.testing.assert_allclose(feature_ptp.to_numpy(), expected_ptp, rtol=1e-12)


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_parseval_spectral_energy_oracle(n_samples: int, channel: str) -> None:
    # Derivation: Parseval identity: sum(x_n^2)/N = (1/N^2) * sum(|X_k|^2)
    # Time power mean^2 + var must equal frequency energy of one-sided rfft
    arr, df = _extract_seeded(n_samples, channel)
    time_power = (df[f"{channel}_mean"] ** 2 + df[f"{channel}_var"]).to_numpy()
    fft_vals = np.fft.rfft(arr, axis=1)
    weights = np.full(fft_vals.shape[1], 2.0)
    weights[0] = 1.0
    if n_samples % 2 == 0:
        weights[-1] = 1.0
    freq_power = np.sum(np.abs(fft_vals) ** 2 * weights, axis=1) / (n_samples**2)
    np.testing.assert_allclose(time_power, freq_power, rtol=1e-10)


@pytest.mark.parametrize("shift", [10.0, -250.0, 10000.0])
def test_shift_invariance(shift: float) -> None:
    # Derivation: Var(X + c) = Var(X), Std(X + c) = Std(X), Slope(X + c) = Slope(X)
    _evaluate_transform_invariance(
        lambda arr: arr + shift,
        lambda orig: {
            "std": orig["FS1_std"].to_numpy(),
            "var": orig["FS1_var"].to_numpy(),
            "slope": orig["FS1_slope"].to_numpy(),
            "skew": orig["FS1_skew"].to_numpy(),
            "kurtosis": orig["FS1_kurtosis"].to_numpy(),
            "mean": (orig["FS1_mean"] + shift).to_numpy(),
        },
    )


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0, -1.0, -2.5, -0.1])
def test_scale_behavior(scale: float) -> None:
    # Derivation: Std(aX) = |a|*Std(X), Var(aX) = a^2*Var(X), Slope(aX) = a*Slope(X)
    # Skew(aX) = sgn(a)*Skew(X), Kurt(aX) = Kurt(X)
    sign = 1.0 if scale > 0 else -1.0
    _evaluate_transform_invariance(
        lambda arr: scale * arr,
        lambda orig: {
            "std": (abs(scale) * orig["FS1_std"]).to_numpy(),
            "var": ((scale**2) * orig["FS1_var"]).to_numpy(),
            "slope": (scale * orig["FS1_slope"]).to_numpy(),
            "skew": (sign * orig["FS1_skew"]).to_numpy(),
            "kurtosis": orig["FS1_kurtosis"].to_numpy(),
        },
    )


@pytest.mark.parametrize("const_val", [0.0, 7.5, -42.0, 1_000_000.0])
def test_constant_signal_invariance_and_no_nan(const_val: float) -> None:
    # Derivation: Constant signal has identically zero variance, skewness, and slope
    arr = np.full((3, 60), const_val, dtype=np.float64)
    df = extract_statistical_features({"TS1": arr})
    _assert_metrics_allclose(
        df,
        "TS1",
        {
            "std": 0.0,
            "var": 0.0,
            "slope": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "mean": const_val,
            "min": const_val,
            "max": const_val,
        },
        rtol=1e-12,
    )
    assert not df.isna().any().any()


@pytest.mark.parametrize(("n_samples", "channel"), CHANNEL_CASES)
def test_time_reversal_slope_antisymmetry(n_samples: int, channel: str) -> None:
    # Derivation: Time reversal t -> (N - 1 - t) negates OLS slope while leaving moments unchanged
    arr, df = _extract_seeded(n_samples, channel)
    arr_rev = np.ascontiguousarray(arr[:, ::-1])
    df_rev = extract_statistical_features({channel: arr_rev})
    _assert_metrics_allclose(
        df_rev,
        channel,
        {
            "slope": -df[f"{channel}_slope"].to_numpy(),
            "mean": df[f"{channel}_mean"].to_numpy(),
            "std": df[f"{channel}_std"].to_numpy(),
        },
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    ("slope", "offset", "n_samples"),
    [(0.5, 10.0, 60), (-2.0, 100.0, 600), (0.01, 5.0, 6000)],
)
def test_closed_form_linear_ramp(slope: float, offset: float, n_samples: int) -> None:
    # Derivation: x_n = m*n + c -> mean = c + m*(N-1)/2, slope = m, var = m^2*(N^2 - 1)/12
    n = np.arange(n_samples, dtype=np.float64)
    ramp = (offset + slope * n)[np.newaxis, :]
    df = extract_statistical_features({"PS1": ramp})
    exp_mean = offset + slope * (n_samples - 1.0) / 2.0
    exp_var = (slope**2) * (n_samples**2 - 1.0) / 12.0
    _assert_metrics_allclose(
        df,
        "PS1",
        {
            "slope": slope,
            "mean": exp_mean,
            "var": exp_var,
            "std": np.sqrt(exp_var),
            "skew": 0.0,
        },
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    ("amplitude", "n_samples"),
    [(5.0, 60), (12.0, 600), (50.0, 6000)],
)
def test_closed_form_bimodal_square_wave(amplitude: float, n_samples: int) -> None:
    # Derivation: Alternating +A, -A for even N has mean=0, var=A^2, std=A,
    # skew=0, and Fisher excess kurtosis E[X^4]/Var(X)^2 - 3 = A^4/A^4 - 3 = -2.0
    pattern = np.tile([amplitude, -amplitude], n_samples // 2)[np.newaxis, :]
    df = extract_statistical_features({"FS1": pattern})
    _assert_metrics_allclose(
        df,
        "FS1",
        {
            "mean": 0.0,
            "std": amplitude,
            "var": amplitude**2,
            "skew": 0.0,
            "kurtosis": -2.0,
            "min": -amplitude,
            "max": amplitude,
        },
        rtol=1e-10,
    )


@pytest.mark.parametrize("n_samples", [1, 2])
def test_minimal_lengths(n_samples: int) -> None:
    # Derivation: N=1 is constant -> std=0, slope=0. N=2 [y0, y1] -> slope=y1-y0, mean=(y0+y1)/2
    data_map = {1: np.array([[42.0]]), 2: np.array([[10.0, 20.0]])}
    arr = data_map[n_samples]
    df = extract_statistical_features({"CH": arr})
    if n_samples == 1:
        assert df["CH_slope"].iloc[0] == 0.0
        assert df["CH_std"].iloc[0] == 0.0
    else:
        _assert_metrics_allclose(
            df,
            "CH",
            {"slope": 10.0, "mean": 15.0, "std": 5.0},
            rtol=1e-12,
        )


@pytest.mark.parametrize(
    ("channel", "low", "high", "n_samples"),
    [
        ("PS1", 0.0, 250.0, 6000),
        ("TS1", 10.0, 60.0, 60),
        ("FS1", 0.0, 12.0, 600),
        ("EPS1", 0.0, 3000.0, 6000),
    ],
)
def test_extreme_physical_sensor_ranges(
    channel: str,
    low: float,
    high: float,
    n_samples: int,
) -> None:
    # Derivation: Signals within physical operating bounds
    # must produce statistics bounded by [low, high]
    rng = np.random.default_rng(123)
    arr = rng.uniform(low, high, (3, n_samples))
    df = extract_statistical_features({channel: arr})
    series_bounded = (
        (df[f"{channel}_min"] >= low)
        & (df[f"{channel}_max"] <= high)
        & (df[f"{channel}_mean"] >= low)
        & (df[f"{channel}_mean"] <= high)
    )
    assert series_bounded.all() and not df.isna().any().any()


@pytest.mark.parametrize(
    ("case", "data"),
    [
        ("empty", {}),
        ("1d", {"P": np.zeros(60)}),
        ("3d", {"P": np.zeros((2, 60, 1))}),
        ("mismatch", {"P": np.zeros((2, 60)), "F": np.zeros((3, 60))}),
    ],
)
def test_empty_and_invalid_shapes_raise_expected(case: str, data: dict[str, np.ndarray]) -> None:
    # Derivation: Invalid input geometries must be rejected or produce empty DataFrame
    if case == "empty":
        df = extract_statistical_features(data)
        assert df.empty
    else:
        with pytest.raises(ValueError):
            extract_statistical_features(data)


@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_inputs_preserve_structure(val: float) -> None:
    # Derivation: Corrupted IEEE 754 floats produce output columns matching standard 11 metrics
    arr = np.array([[1.0, 2.0, val, 4.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = extract_statistical_features({"CH": arr})
    assert "CH_mean" in df.columns
    assert df.shape == (1, 11)
