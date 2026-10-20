"""Cycle splitting strategies and fold coverage validation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def blocked_time_ordered_kfold(
    n_cycles: int = 2205,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate time-ordered contiguous K-fold train/test splits.

    Args:
        n_cycles: Total number of cycles (default 2205).
        n_splits: Number of contiguous blocks (folds, default 5).

    Returns:
        List of (train_idx, test_idx) integer index arrays.

    Raises:
        ValueError: If n_splits < 2 or n_splits > n_cycles.
    """
    if n_splits < 2 or n_splits > n_cycles:
        raise ValueError(f"n_splits must be between 2 and n_cycles ({n_cycles}), got {n_splits}")
    indices = np.arange(n_cycles)
    test_blocks = np.array_split(indices, n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_idx in test_blocks:
        train_idx = np.setdiff1d(indices, test_idx)
        folds.append((train_idx, test_idx))
    return folds


def _audit_column_coverage(
    series: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    col_name: str,
) -> bool:
    """Audit single target column coverage across all test folds."""
    all_classes = set(series.unique())
    is_covered = True
    for fold_idx, (_, test_idx) in enumerate(folds):
        test_classes = set(series.iloc[test_idx].unique())
        missing = all_classes - test_classes
        if missing:
            is_covered = False
            warnings.warn(
                f"Target '{col_name}', fold {fold_idx}: missing classes {sorted(missing)}",
                category=UserWarning,
                stacklevel=3,
            )
    return is_covered


def audit_class_coverage(
    targets: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, bool]:
    """Audit class coverage across test folds for each target column except stable_flag.

    Args:
        targets: Target DataFrame containing condition columns.
        folds: Sequence of (train_idx, test_idx) tuples.

    Returns:
        Dict mapping column name to boolean indicating full coverage.
    """
    coverage: dict[str, bool] = {}
    for col in targets.columns:
        if str(col).lower() == "stable_flag":
            continue
        coverage[str(col)] = _audit_column_coverage(targets[col], folds, str(col))
    return coverage


def temperature_shift_split(
    ts1: np.ndarray,
    threshold_c: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split cycle indices by mean temperature for TS1 sensor (degC).

    Cycles with mean TS1 < threshold go to train, mean TS1 >= threshold go to test.

    Args:
        ts1: TS1 sensor readings of shape (n_cycles, 60) in degC.
        threshold_c: Temperature threshold in degC. If None, uses median cycle mean.

    Returns:
        Tuple of (train_idx, test_idx, threshold_used).

    Raises:
        ValueError: If ts1 is not 2-D or either train or test split is empty.
    """
    if ts1.ndim != 2:
        raise ValueError(f"ts1 must be a 2-D array, got ndim={ts1.ndim}")
    cycle_means = np.mean(ts1, axis=1)
    threshold = float(np.median(cycle_means)) if threshold_c is None else float(threshold_c)
    train_idx = np.where(cycle_means < threshold)[0]
    test_idx = np.where(cycle_means >= threshold)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Split resulted in an empty set: train={len(train_idx)}, "
            f"test={len(test_idx)} (threshold={threshold:.2f} degC)"
        )
    return train_idx, test_idx, threshold
