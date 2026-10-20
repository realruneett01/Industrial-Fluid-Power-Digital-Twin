"""Tests for evaluation metrics and report generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from hydro.eval.metrics import (
    CLASS_LABELS,
    evaluate_classification_targets,
    latency_percentiles,
)
from hydro.eval.report import generate_markdown_table, write_report

if TYPE_CHECKING:
    from collections.abc import Callable


def assert_latency_percentiles(
    metrics: dict[str, float],
    expected: tuple[float, float, float],
) -> None:
    """Assert latency percentiles match expected p50, p95, and p99 values in ms."""
    assert (metrics["p50_ms"], metrics["p95_ms"], metrics["p99_ms"]) == pytest.approx(expected)


def assert_target_classification_metrics(
    row: pd.Series,
    expected: tuple[float, float, float, int],
) -> None:
    """Assert target row matches expected (macro_f1, macro_f1_all, balanced_acc, n_test)."""
    assert (
        row["macro_f1"],
        row["macro_f1_all_labels"],
        row["balanced_accuracy"],
        int(row["n_test"]),
    ) == (
        pytest.approx(expected[0]),
        pytest.approx(expected[1]),
        pytest.approx(expected[2]),
        expected[3],
    )


def test_class_labels_constant() -> None:
    """CLASS_LABELS defines valid condition states for the 4 physical targets."""
    assert CLASS_LABELS["cooler"] == [3, 20, 100]
    assert CLASS_LABELS["valve"] == [73, 80, 90, 100]
    assert CLASS_LABELS["pump"] == [0, 1, 2]
    assert CLASS_LABELS["accumulator"] == [90, 100, 115, 130]


def test_perfect_predictions_macro_f1() -> None:
    """Perfect predictions must yield macro_f1 = 1.0 across all targets."""
    # 12 samples covers full cycles of 3 classes and 4 classes
    y_true = pd.DataFrame(
        {
            "cooler": [3, 20, 100] * 4,
            "valve": [73, 80, 90, 100] * 3,
            "pump": [0, 1, 2] * 4,
            "accumulator": [90, 100, 115, 130] * 3,
        }
    )
    res = evaluate_classification_targets(y_true, y_true.copy())

    assert set(res.index) == set(CLASS_LABELS.keys())
    for target in CLASS_LABELS:
        assert_target_classification_metrics(res.loc[target], (1.0, 1.0, 1.0, len(y_true)))


def test_hand_computed_confusion_case() -> None:
    """Hand-computed confusion case matches macro_f1 and balanced_accuracy."""
    # Cooler classes: [3, 20, 100]
    # Sample 1: true=3, pred=3   (TP_3)
    # Sample 2: true=3, pred=20  (FN_3, FP_20)
    # Sample 3: true=20, pred=20 (TP_20)
    # Sample 4: true=100, pred=100 (TP_100)
    # F1_3 = 2/3, F1_20 = 2/3, F1_100 = 1.0 -> Macro-F1 = (7/3) / 3 = 7/9
    # Recall_3 = 0.5, Recall_20 = 1.0, Recall_100 = 1.0 -> Balanced Acc = 2.5 / 3 = 5/6
    y_true = pd.DataFrame({"cooler": [3, 3, 20, 100]})
    y_pred = pd.DataFrame({"cooler": [3, 20, 20, 100]})

    res = evaluate_classification_targets(y_true, y_pred)
    assert_target_classification_metrics(
        res.loc["cooler"],
        (7.0 / 9.0, 7.0 / 9.0, 5.0 / 6.0, 4),
    )


def test_missing_class_macro_f1_difference() -> None:
    """Missing class in fold causes macro_f1 to differ from macro_f1_all_labels."""
    # Cooler classes: [3, 20, 100]; class 100 absent. Present F1=1.0; all-labels F1=2/3.
    df_fold = pd.DataFrame({"cooler": [3, 3, 20, 20]})
    res = evaluate_classification_targets(df_fold, df_fold)

    f1_present = res.loc["cooler", "macro_f1"]
    f1_all = res.loc["cooler", "macro_f1_all_labels"]
    assert (f1_present, f1_all) == (1.0, pytest.approx(2.0 / 3.0))
    assert f1_present != f1_all


def test_latency_percentiles_known_array() -> None:
    """Latency percentiles match expected p50, p95, p99 on known array."""
    # 0 to 100 ms in 1 ms steps (0 to 100_000_000 ns)
    samples_ns = np.linspace(0, 100_000_000, 101, dtype=np.int64)

    metrics = latency_percentiles(samples_ns)
    assert_latency_percentiles(metrics, (50.0, 95.0, 99.0))


def test_markdown_table_structure() -> None:
    """Markdown table has header, separator, and one row per index entry."""
    df = pd.DataFrame(
        {
            "macro_f1": [1.0, 0.85],
            "balanced_accuracy": [0.95, 0.80],
            "n_test": [100, 100],
        },
        index=pd.Index(["cooler", "valve"], name="target"),
    )

    table = generate_markdown_table(df)
    lines = [line.strip() for line in table.strip().split("\n") if line.strip()]

    # 1 header + 1 separator + 2 data rows = 4 rows
    assert len(lines) == 4
    header_cols = [c.strip() for c in lines[0].split("|")[1:-1]]
    assert header_cols == ["target", "macro_f1", "balanced_accuracy", "n_test"]
    assert "---" in lines[1]
    assert "cooler" in lines[2]
    assert "1.0000" in lines[2]
    assert "valve" in lines[3]
    assert "0.8500" in lines[3]


def test_write_report_headers(tmp_path: Path) -> None:
    """write_report output contains title and every section header."""
    report_file = tmp_path / "reports" / "summary.md"
    sections = {
        "Executive Summary": "System operating within normal limits.",
        "Target Performance": "Detailed table content.",
        "Latency Verification": "p95 latency is 45 ms.",
    }

    ret = write_report(report_file, "Twin Diagnostics Report", sections)

    assert ret == report_file
    assert report_file.is_file()
    content = report_file.read_text(encoding="utf-8")
    assert "# Twin Diagnostics Report" in content
    for heading in sections:
        assert f"## {heading}" in content
    assert "System operating within normal limits." in content


VALIDATION_CASES = [
    (
        lambda: evaluate_classification_targets(
            pd.DataFrame({"cooler": [3, 20]}), pd.DataFrame({"cooler": [3]})
        ),
        "Length mismatch",
    ),
    (
        lambda: evaluate_classification_targets(pd.DataFrame(), pd.DataFrame()),
        "empty",
    ),
    (lambda: latency_percentiles(np.array([])), "empty"),
    (lambda: latency_percentiles(np.array([-1.0, 10.0])), "non-negative"),
]


@pytest.mark.parametrize(
    ("action", "match_msg"),
    VALIDATION_CASES,
    ids=["length_mismatch", "empty_dataframes", "empty_latencies", "negative_latencies"],
)
def test_validation_errors(action: Callable[[], object], match_msg: str) -> None:
    """Validation errors raised on invalid shapes and inputs."""
    with pytest.raises(ValueError, match=match_msg):
        action()
