"""Mathematical evaluation tests comparing metrics against sklearn and validating splits."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from hydro.eval.metrics import (
    CLASS_LABELS,
    evaluate_classification_targets,
    latency_percentiles,
)
from hydro.eval.splits import (
    blocked_time_ordered_kfold,
    temperature_shift_split,
)
from tests.test_metrics import (
    assert_latency_percentiles,
    assert_target_classification_metrics,
)


@dataclass(frozen=True)
class ConfusionMetrics:
    """Classification metrics derived directly from confusion matrix entries."""

    macro_f1: float
    macro_f1_all: float
    balanced_accuracy: float


def _cm_f1(cm: np.ndarray) -> np.ndarray:
    """Compute per-class F1 score from confusion matrix cm."""
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0).astype(float) - tp
    fn = np.sum(cm, axis=1).astype(float) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def compute_confusion_oracles(
    yt: np.ndarray,
    yp: np.ndarray,
    all_class_labels: list[int],
) -> ConfusionMetrics:
    """Compute independent oracle metrics using confusion matrix closed-form math.

    Derivation: TP_i = C[i, i], FP_i = sum_k(C[k, i]) - TP_i, FN_i = sum_j(C[i, j]) - TP_i.
    F1_i = 2*TP_i / (2*TP_i + FP_i + FN_i); Recall_i = TP_i / Support_i.
    Macro-F1 is mean(F1) over present labels; Macro-F1-all over all_class_labels;
    Balanced accuracy is mean(Recall) over classes present in ground truth.
    """
    present = sorted(set(yt).union(set(yp)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        cm_p = confusion_matrix(yt, yp, labels=present)
        cm_a = confusion_matrix(yt, yp, labels=all_class_labels)
    f1_p = _cm_f1(cm_p)
    f1_a = _cm_f1(cm_a)
    supp = np.sum(cm_p, axis=1).astype(float)
    tp_p = np.diag(cm_p).astype(float)
    rec = np.divide(tp_p, supp, out=np.zeros_like(tp_p), where=supp > 0)
    bal = float(np.mean(rec[supp > 0])) if np.any(supp > 0) else 0.0
    return ConfusionMetrics(
        macro_f1=float(np.mean(f1_p)),
        macro_f1_all=float(np.mean(f1_a)),
        balanced_accuracy=bal,
    )


def assert_temperature_split_invariants(
    split_result: tuple[np.ndarray, np.ndarray, float],
    expected_threshold: float,
    cycle_values: np.ndarray,
) -> None:
    """Assert temperature shift split partitions cycles cleanly around threshold."""
    train_idx, test_idx, threshold = split_result
    assert threshold == pytest.approx(expected_threshold)
    assert np.all(cycle_values[train_idx] < threshold)
    assert np.all(cycle_values[test_idx] >= threshold)
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(train_idx) + len(test_idx) == len(cycle_values)


@pytest.mark.parametrize(
    ("target", "seed"),
    [
        ("cooler", 42),
        ("cooler", 101),
        ("valve", 42),
        ("valve", 2024),
        ("pump", 777),
        ("pump", 888),
        ("accumulator", 42),
        ("accumulator", 999),
    ],
)
def test_metrics_vs_sklearn_and_confusion_oracle(target: str, seed: int) -> None:
    """Random seeded labels match sklearn reference and confusion matrix math."""
    rng = np.random.default_rng(seed)
    labels = CLASS_LABELS[target]
    yt = rng.choice(labels, size=60)
    yp = rng.choice(labels, size=60)
    res = evaluate_classification_targets(pd.DataFrame({target: yt}), pd.DataFrame({target: yp}))
    row = res.loc[target]
    cm_oracle = compute_confusion_oracles(yt, yp, labels)

    present = sorted(set(yt).union(set(yp)))
    sk_f1 = float(f1_score(yt, yp, labels=present, average="macro", zero_division=0))
    sk_f1_all = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    sk_bal = float(balanced_accuracy_score(yt, yp))

    actual = (row["macro_f1"], row["macro_f1_all_labels"], row["balanced_accuracy"])
    assert actual == pytest.approx((sk_f1, sk_f1_all, sk_bal))
    assert actual == pytest.approx(
        (cm_oracle.macro_f1, cm_oracle.macro_f1_all, cm_oracle.balanced_accuracy)
    )


@dataclass(frozen=True)
class DegenerateCase:
    """Container for degenerate classification test scenario."""

    target: str
    y_true: list[int]
    y_pred: list[int]
    expected_macro_f1: float
    expected_macro_f1_all: float
    expected_bal_acc: float


DEGENERATE_CASES: list[tuple[str, DegenerateCase]] = [
    (
        "single_class_perfect",
        DegenerateCase(
            target="cooler",
            y_true=[3, 3, 3, 3],
            y_pred=[3, 3, 3, 3],
            # Closed-form: 1 present class TP=4 -> F1=1.0. All 3 labels: (1+0+0)/3 = 1/3.
            expected_macro_f1=1.0,
            expected_macro_f1_all=1.0 / 3.0,
            expected_bal_acc=1.0,
        ),
    ),
    (
        "single_class_all_wrong",
        DegenerateCase(
            target="cooler",
            y_true=[3, 3, 3, 3],
            y_pred=[20, 20, 20, 20],
            # Closed-form: TP=0 for all classes -> F1=0, Recall=0.
            expected_macro_f1=0.0,
            expected_macro_f1_all=0.0,
            expected_bal_acc=0.0,
        ),
    ),
    (
        "multi_class_perfect",
        DegenerateCase(
            target="pump",
            y_true=[0, 1, 2, 0, 1, 2],
            y_pred=[0, 1, 2, 0, 1, 2],
            # Closed-form: all classes TP=2 -> F1=1.0, Recall=1.0.
            expected_macro_f1=1.0,
            expected_macro_f1_all=1.0,
            expected_bal_acc=1.0,
        ),
    ),
    (
        "multi_class_all_wrong_cyclic",
        DegenerateCase(
            target="pump",
            y_true=[0, 1, 2],
            y_pred=[1, 2, 0],
            # Closed-form: zero matching predictions -> F1=0.0, Recall=0.0.
            expected_macro_f1=0.0,
            expected_macro_f1_all=0.0,
            expected_bal_acc=0.0,
        ),
    ),
    (
        "missing_classes_in_fold",
        DegenerateCase(
            target="valve",
            y_true=[73, 80],
            y_pred=[73, 80],
            # Closed-form: present [73, 80] F1=1.0. All 4 labels has two 0s -> 2/4 = 0.5.
            expected_macro_f1=1.0,
            expected_macro_f1_all=0.5,
            expected_bal_acc=1.0,
        ),
    ),
    (
        "extreme_imbalance",
        DegenerateCase(
            target="accumulator",
            y_true=[90] * 99 + [100] * 1,
            y_pred=[90] * 100,
            # Closed-form: F1_90 = 198/199, F1_100 = 0. Present = 99/199. All = 99/398.
            expected_macro_f1=99.0 / 199.0,
            expected_macro_f1_all=99.0 / 398.0,
            expected_bal_acc=0.5,
        ),
    ),
    (
        "single_sample",
        DegenerateCase(
            target="cooler",
            y_true=[100],
            y_pred=[100],
            # Closed-form: single sample TP=1 -> F1=1.0. All labels: 1/3.
            expected_macro_f1=1.0,
            expected_macro_f1_all=1.0 / 3.0,
            expected_bal_acc=1.0,
        ),
    ),
]


@pytest.mark.parametrize(("case_name", "case"), DEGENERATE_CASES)
def test_classification_degenerate_cases(case_name: str, case: DegenerateCase) -> None:
    """Degenerate cases match closed-form math and confusion matrix oracle."""
    yt = np.array(case.y_true)
    yp = np.array(case.y_pred)
    df_t = pd.DataFrame({case.target: yt})
    df_p = pd.DataFrame({case.target: yp})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        res = evaluate_classification_targets(df_t, df_p)
    row = res.loc[case.target]
    assert_target_classification_metrics(
        row,
        (
            case.expected_macro_f1,
            case.expected_macro_f1_all,
            case.expected_bal_acc,
            len(case.y_true),
        ),
    )


CLASSIFICATION_INVALID_CASES = [
    (pd.DataFrame({"cooler": [3, 20]}), pd.DataFrame({"cooler": [3]}), "Length mismatch"),
    (pd.DataFrame({"cooler": []}), pd.DataFrame({"cooler": []}), "must not be empty"),
    (pd.DataFrame({"cooler": [3, 20]}), pd.DataFrame({"valve": [73, 80]}), "missing from y_pred"),
]


@pytest.mark.parametrize(
    ("df_t", "df_p", "error_match"),
    CLASSIFICATION_INVALID_CASES,
    ids=["length_mismatch", "empty_dataframe", "missing_target_in_pred"],
)
def test_classification_invalid_inputs_raise(
    df_t: pd.DataFrame,
    df_p: pd.DataFrame,
    error_match: str,
) -> None:
    """Invalid classification input shapes and missing targets raise ValueError."""
    with pytest.raises(ValueError, match=error_match):
        evaluate_classification_targets(df_t, df_p)


def test_classification_unknown_target_fallback() -> None:
    """Target absent from CLASS_LABELS falls back to present labels."""
    df_t = pd.DataFrame({"custom_target": [1, 2, 2]})
    df_p = pd.DataFrame({"custom_target": [1, 2, 1]})
    res = evaluate_classification_targets(df_t, df_p)
    assert res.loc["custom_target", "macro_f1"] == pytest.approx(
        res.loc["custom_target", "macro_f1_all_labels"]
    )


SPLIT_INVARIANT_CASES = [
    (2, 2),
    (5, 5),
    (7, 3),
    (10, 4),
    (14, 7),
    (23, 6),
    (50, 7),
    (100, 5),
    (2205, 5),
    (2205, 7),
]


@pytest.mark.parametrize(("n_cycles", "n_splits"), SPLIT_INVARIANT_CASES)
def test_blocked_kfold_split_invariants(n_cycles: int, n_splits: int) -> None:
    """Validate disjoint, ordered, contiguous, and complement invariants."""
    folds = blocked_time_ordered_kfold(n_cycles=n_cycles, n_splits=n_splits)
    assert len(folds) == n_splits

    all_test = [test_idx for _, test_idx in folds]
    test_sizes = [len(t) for t in all_test]
    assert max(test_sizes) - min(test_sizes) <= 1

    for train_idx, test_idx in folds:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        assert np.array_equal(train_idx, np.setdiff1d(np.arange(n_cycles), test_idx))
        assert len(train_idx) + len(test_idx) == n_cycles

    concatenated = np.concatenate(all_test)
    np.testing.assert_array_equal(concatenated, np.arange(n_cycles))

    for test_idx in all_test:
        if len(test_idx) > 1:
            assert np.all(np.diff(test_idx) == 1)


@pytest.mark.parametrize(
    ("n_cycles", "n_splits"),
    [
        (10, 1),
        (10, 0),
        (10, -3),
        (4, 5),
    ],
)
def test_blocked_kfold_invalid_splits_raise(n_cycles: int, n_splits: int) -> None:
    """Splits with k < 2 or k > n raise ValueError."""
    with pytest.raises(ValueError, match="n_splits must be between 2 and n_cycles"):
        blocked_time_ordered_kfold(n_cycles=n_cycles, n_splits=n_splits)


@pytest.mark.parametrize("seed", [42, 101, 777, 2026])
def test_temperature_shift_split_median_threshold(seed: int) -> None:
    """When threshold_c is None, threshold is median of per-cycle mean TS1."""
    rng = np.random.default_rng(seed)
    cycle_base = np.linspace(35.0, 55.0, 30)
    noise = rng.normal(0.0, 0.2, size=(30, 60))
    ts1 = cycle_base[:, None] + noise

    cycle_means = np.mean(ts1, axis=1)
    expected_threshold = float(np.median(cycle_means))

    split_result = temperature_shift_split(ts1)
    assert_temperature_split_invariants(split_result, expected_threshold, cycle_means)


@pytest.mark.parametrize("explicit_threshold", [38.0, 45.0, 50.0])
def test_temperature_shift_split_explicit_threshold(explicit_threshold: float) -> None:
    """Explicit threshold is strictly honoured for partitioning cycles."""
    cycle_temps = np.linspace(30.0, 60.0, 31)
    ts1 = np.repeat(cycle_temps[:, None], 60, axis=1)

    split_result = temperature_shift_split(ts1, threshold_c=explicit_threshold)
    assert_temperature_split_invariants(split_result, explicit_threshold, cycle_temps)


@pytest.mark.parametrize("bad_shape", [np.ones(60), np.ones((5, 60, 1))])
def test_temperature_shift_split_non_2d_raises(bad_shape: np.ndarray) -> None:
    """Input TS1 array that is not 2-D raises ValueError."""
    with pytest.raises(ValueError, match="ts1 must be a 2-D array"):
        temperature_shift_split(bad_shape)


def test_temperature_shift_split_constant_all_equal_raises() -> None:
    """Constant temperature cycles producing empty train split raise ValueError."""
    ts1 = np.full((10, 60), 45.0)
    with pytest.raises(ValueError, match="Split resulted in an empty set"):
        temperature_shift_split(ts1)


@pytest.mark.parametrize("out_of_bounds_thresh", [10.0, 90.0])
def test_temperature_shift_split_empty_split_explicit_raises(out_of_bounds_thresh: float) -> None:
    """Explicit threshold outside data range causing empty split raises ValueError."""
    ts1 = np.repeat(np.linspace(30.0, 50.0, 10)[:, None], 60, axis=1)
    with pytest.raises(ValueError, match="Split resulted in an empty set"):
        temperature_shift_split(ts1, threshold_c=out_of_bounds_thresh)


def test_temperature_shift_split_minimal_two_cycles() -> None:
    """Minimal 2-cycle array splits into 1 train cycle and 1 test cycle."""
    ts1 = np.array([[35.0] * 60, [55.0] * 60])
    train_idx, test_idx, threshold = temperature_shift_split(ts1)
    assert threshold == pytest.approx(45.0)
    assert list(train_idx) == [0]
    assert list(test_idx) == [1]


def test_latency_percentiles_units_and_percentiles() -> None:
    """Latency conversion from ns to ms and percentile math match numpy oracle."""
    samples_ns = np.array([1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000], dtype=np.int64)
    # samples in ms: [1.0, 2.0, 3.0, 4.0, 5.0]
    # Closed-form: p50 = 3.0 ms, p95 = 4.8 ms, p99 = 4.96 ms.
    res = latency_percentiles(samples_ns)
    assert_latency_percentiles(res, (3.0, 4.8, 4.96))


@pytest.mark.parametrize("seed", [42, 101, 2026])
def test_latency_percentiles_seeded_random(seed: int) -> None:
    """Random latency measurements match numpy percentile calculation in ms."""
    rng = np.random.default_rng(seed)
    samples_ns = rng.uniform(1e6, 1e8, size=500)
    samples_ms = samples_ns / 1e6

    res = latency_percentiles(samples_ns)
    expected = (
        float(np.percentile(samples_ms, 50)),
        float(np.percentile(samples_ms, 95)),
        float(np.percentile(samples_ms, 99)),
    )
    assert_latency_percentiles(res, expected)


LATENCY_UNIFORM_CASES = [
    (np.array([42_000_000.0]), 42.0),
    (np.full(50, 15_000_000.0), 15.0),
    (np.zeros(10), 0.0),
]


@pytest.mark.parametrize(
    ("samples_ns", "expected_ms"),
    LATENCY_UNIFORM_CASES,
    ids=["single_sample", "constant_values", "zeros"],
)
def test_latency_percentiles_uniform_inputs(
    samples_ns: np.ndarray,
    expected_ms: float,
) -> None:
    """Single, constant, and zero latency inputs yield that constant across percentiles."""
    res = latency_percentiles(samples_ns)
    assert_latency_percentiles(res, (expected_ms, expected_ms, expected_ms))


@pytest.mark.parametrize(
    ("invalid_input", "error_match"),
    [
        (np.array([-1.0, 10.0]), "must be non-negative"),
        (np.array([]), "must not be empty"),
        (np.ones((2, 2)), "must be 1D"),
    ],
)
def test_latency_percentiles_invalid_inputs_raise(
    invalid_input: np.ndarray,
    error_match: str,
) -> None:
    """Negative values, empty arrays, or non-1D inputs raise ValueError."""
    with pytest.raises(ValueError, match=error_match):
        latency_percentiles(invalid_input)
