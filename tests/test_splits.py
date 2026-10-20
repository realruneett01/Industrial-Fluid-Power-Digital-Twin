"""Tests for cycle split strategies and coverage audit."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from hydro.eval.splits import (
    audit_class_coverage,
    blocked_time_ordered_kfold,
    temperature_shift_split,
)


def test_blocked_kfold_zero_intersection_and_order() -> None:
    """Zero train/test intersection for all folds and contiguous ordered coverage."""
    n_cycles = 100
    n_splits = 5
    folds = blocked_time_ordered_kfold(n_cycles=n_cycles, n_splits=n_splits)

    assert len(folds) == n_splits
    all_test = []
    for train_idx, test_idx in folds:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert len(train_idx) + len(test_idx) == n_cycles
        all_test.append(test_idx)

    concatenated = np.concatenate(all_test)
    np.testing.assert_array_equal(concatenated, np.arange(n_cycles))


def test_blocked_kfold_default_2205_cycles() -> None:
    """Default 2205 cycles produces 5 test blocks of 441 cycles."""
    folds = blocked_time_ordered_kfold()
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert len(test_idx) == 441
        assert len(train_idx) == 1764


@pytest.mark.parametrize(
    ("n_cycles", "n_splits"),
    [
        (2205, 1),
        (2205, 0),
        (2205, -1),
        (5, 6),
    ],
)
def test_blocked_kfold_invalid_splits_raise(n_cycles: int, n_splits: int) -> None:
    """Invalid n_splits raises ValueError."""
    with pytest.raises(ValueError, match="n_splits"):
        blocked_time_ordered_kfold(n_cycles=n_cycles, n_splits=n_splits)


def test_audit_class_coverage_missing_class_warns() -> None:
    """A class present only in the first block triggers UserWarning and returns False."""
    n_cycles = 100
    folds = blocked_time_ordered_kfold(n_cycles=n_cycles, n_splits=5)

    # Class 99 appears only in cycle 0 (first block)
    cooler = np.ones(n_cycles, dtype=int)
    cooler[0] = 99
    targets = pd.DataFrame({"cooler": cooler, "stable_flag": np.zeros(n_cycles, dtype=int)})

    with pytest.warns(UserWarning) as record:
        coverage = audit_class_coverage(targets, folds)

    assert coverage["cooler"] is False
    assert "stable_flag" not in coverage

    # Verify warning mentions column, fold, and missing class
    messages = [str(w.message) for w in record]
    assert any("cooler" in msg and "fold" in msg and "99" in msg for msg in messages)


def test_audit_class_coverage_full_coverage() -> None:
    """Full class coverage returns True with no UserWarning."""
    n_cycles = 100
    folds = blocked_time_ordered_kfold(n_cycles=n_cycles, n_splits=5)

    # Each block contains both classes 0 and 1
    labels = np.tile([0, 1], n_cycles // 2)
    targets = pd.DataFrame({"pump": labels, "stable_flag": labels})

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        coverage = audit_class_coverage(targets, folds)

    assert coverage == {"pump": True}


def test_temp_shift_split_disjoint_and_covering() -> None:
    """Temp split partitions indices cleanly and covers all cycles."""
    rng = np.random.default_rng(42)
    ts1 = rng.normal(loc=40.0, scale=5.0, size=(100, 60))

    train_idx, test_idx, threshold = temperature_shift_split(ts1)

    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(train_idx) + len(test_idx) == 100
    np.testing.assert_array_equal(
        np.sort(np.concatenate([train_idx, test_idx])),
        np.arange(100),
    )
    cycle_means = np.mean(ts1, axis=1)
    assert np.all(cycle_means[train_idx] < threshold)
    assert np.all(cycle_means[test_idx] >= threshold)


def test_temp_shift_threshold_used() -> None:
    """Threshold equals median when None and equals given value otherwise."""
    ts1 = np.arange(100, dtype=float).reshape(10, 10)
    cycle_means = np.mean(ts1, axis=1)

    _, _, threshold_auto = temperature_shift_split(ts1, threshold_c=None)
    assert threshold_auto == float(np.median(cycle_means))

    custom_thresh = 45.0
    _, _, threshold_custom = temperature_shift_split(ts1, threshold_c=custom_thresh)
    assert threshold_custom == custom_thresh


def test_temp_shift_invalid_input_raises() -> None:
    """Empty side or non 2-D ts1 raises ValueError."""
    ts1 = np.ones((10, 60))

    # All means are 1.0; threshold 0.0 leaves train empty
    with pytest.raises(ValueError, match="empty"):
        temperature_shift_split(ts1, threshold_c=0.0)

    # All means are 1.0; threshold 10.0 leaves test empty
    with pytest.raises(ValueError, match="empty"):
        temperature_shift_split(ts1, threshold_c=10.0)

    # 1-D array
    with pytest.raises(ValueError, match="2-D"):
        temperature_shift_split(np.ones(60))

    # 3-D array
    with pytest.raises(ValueError, match="2-D"):
        temperature_shift_split(np.ones((10, 60, 1)))
