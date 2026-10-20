"""Statistical feature extraction for hydraulic sensor cycle data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def _compute_slope(arr: np.ndarray, is_const: np.ndarray) -> np.ndarray:
    """Compute OLS slope against sample index for each row in unit/sample."""
    n_samples = arr.shape[1]
    if n_samples <= 1:
        return np.zeros(arr.shape[0], dtype=np.float64)
    x = np.arange(n_samples, dtype=np.float64)
    x_c = x - (n_samples - 1.0) / 2.0
    denom = float(np.dot(x_c, x_c))
    slope: np.ndarray = (arr @ x_c) / denom
    slope[is_const] = 0.0
    return slope


def _compute_channel_stats(arr: np.ndarray, channel: str) -> dict[str, np.ndarray]:
    """Compute 11 statistical metrics for a single channel array."""
    is_const = arr.min(axis=1) == arr.max(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sk = np.nan_to_num(stats.skew(arr, axis=1), nan=0.0)
        kt = np.nan_to_num(stats.kurtosis(arr, axis=1), nan=0.0)
    sk[is_const], kt[is_const] = 0.0, 0.0
    sd = np.where(is_const, 0.0, np.std(arr, axis=1))
    vr = np.where(is_const, 0.0, np.var(arr, axis=1))
    return {
        f"{channel}_mean": np.mean(arr, axis=1),
        f"{channel}_std": sd,
        f"{channel}_var": vr,
        f"{channel}_skew": sk,
        f"{channel}_kurtosis": kt,
        f"{channel}_slope": _compute_slope(arr, is_const),
        f"{channel}_min": np.min(arr, axis=1),
        f"{channel}_max": np.max(arr, axis=1),
        f"{channel}_median": np.median(arr, axis=1),
        f"{channel}_p05": np.percentile(arr, 5, axis=1),
        f"{channel}_p95": np.percentile(arr, 95, axis=1),
    }


def _validate_channel_data(data: dict[str, np.ndarray]) -> int:
    """Validate 2-D shape and consistent cycle count across channels."""
    n_cycles: int | None = None
    for ch, arr in data.items():
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            dims = getattr(arr, "ndim", type(arr).__name__)
            raise ValueError(f"Channel '{ch}' array must be 2-D, got {dims}")
        if n_cycles is None:
            n_cycles = arr.shape[0]
        elif arr.shape[0] != n_cycles:
            raise ValueError(f"Channel '{ch}' has {arr.shape[0]} cycles, expected {n_cycles}")
    return 0 if n_cycles is None else n_cycles


def extract_statistical_features(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """Extract 11 statistical features per channel across all cycles.

    Features per row: mean, std, var, skew, kurtosis, slope, min, max,
    median, p05, p95. Unfitted: computed cycle-by-cycle without label leakage.
    """
    if not data:
        return pd.DataFrame()
    n_cycles = _validate_channel_data(data)
    cols: dict[str, np.ndarray] = {}
    for ch, arr in data.items():
        cols.update(_compute_channel_stats(arr, ch))
    return pd.DataFrame(cols, index=pd.RangeIndex(n_cycles))
