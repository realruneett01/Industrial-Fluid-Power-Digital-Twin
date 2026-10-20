"""Evaluation metrics for hydraulic system fault classification and inference latency."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

CLASS_LABELS: dict[str, list[int]] = {
    "cooler": [3, 20, 100],
    "valve": [73, 80, 90, 100],
    "pump": [0, 1, 2],
    "accumulator": [90, 100, 115, 130],
}


def _evaluate_single_target(
    yt: pd.Series | np.ndarray,
    yp: pd.Series | np.ndarray,
    target: str,
) -> dict[str, float | int]:
    """Compute classification metrics for a single target."""
    present_labels = sorted(set(yt).union(set(yp)))
    all_labels = CLASS_LABELS.get(target, present_labels)
    macro_f1 = float(f1_score(yt, yp, labels=present_labels, average="macro", zero_division=0))
    macro_f1_all = float(f1_score(yt, yp, labels=all_labels, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(yt, yp))
    return {
        "macro_f1": macro_f1,
        "macro_f1_all_labels": macro_f1_all,
        "balanced_accuracy": bal_acc,
        "n_test": len(yt),
    }


def _resolve_targets(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> list[str]:
    """Resolve and validate target columns present in y_true and y_pred."""
    targets = [col for col in y_true.columns if col in CLASS_LABELS]
    if not targets:
        targets = list(y_true.columns)
    for tgt in targets:
        if tgt not in y_pred.columns:
            raise ValueError(f"Target '{tgt}' found in y_true but missing from y_pred.")
    return targets


def evaluate_classification_targets(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate multi-target classification metrics across target components.

    Parameters
    ----------
    y_true : pd.DataFrame
        Ground truth labels per target column.
    y_pred : pd.DataFrame
        Predicted labels per target column.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by target with columns:
        - macro_f1: Macro-F1 over classes present in y_true and y_pred.
        - macro_f1_all_labels: Macro-F1 over all CLASS_LABELS of that target.
        - balanced_accuracy: Balanced accuracy score.
        - n_test: Number of test samples evaluated.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("Input DataFrames must not be empty.")

    targets = _resolve_targets(y_true, y_pred)
    rows = {tgt: _evaluate_single_target(y_true[tgt], y_pred[tgt], tgt) for tgt in targets}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "target"
    return df


def latency_percentiles(samples_ns: np.ndarray) -> dict[str, float]:
    """Compute latency percentiles in milliseconds (ms) from nanosecond (ns) samples.

    Parameters
    ----------
    samples_ns : np.ndarray
        1D array of latency measurements in nanoseconds (ns).

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'p50_ms', 'p95_ms', and 'p99_ms' in milliseconds (ms).
    """
    arr = np.asarray(samples_ns, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Latency sample array must not be empty.")
    if arr.ndim != 1:
        raise ValueError(f"Latency samples must be 1D, got shape {arr.shape}.")
    if (arr < 0).any():
        raise ValueError("Latency measurements must be non-negative.")

    samples_ms = arr / 1e6
    return {
        "p50_ms": float(np.percentile(samples_ms, 50)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
    }
