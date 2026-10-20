"""Benchmark evaluation suite, baseline models, and temperature-shift experiment."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.eval.metrics import CLASS_LABELS
from hydro.eval.report import generate_markdown_table, write_report
from hydro.eval.splits import (
    audit_class_coverage,
    blocked_time_ordered_kfold,
    temperature_shift_split,
)
from hydro.features import build_physics_features, extract_statistical_features
from hydro.features.physics import fit_valve_baseline
from hydro.models.classical import make_classifier
from hydro.models.cnn1d import predict_cnn, train_cnn_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

TARGETS: tuple[str, ...] = ("cooler", "valve", "pump", "accumulator")
PIPELINES: tuple[str, ...] = ("statistical", "physics", "hybrid", "cnn_1d")
BASELINE_PIPELINES: tuple[str, ...] = ("lda", "rf")


@dataclass(frozen=True)
class RunConfig:
    """Benchmark execution configuration parameters.

    Attributes
    ----------
    cnn_epochs : int
        Number of epochs to train the 1D-CNN multi-head model (default 15).
    catboost_iterations : int | None
        Iteration count override for CatBoost classifiers (default None).
    """

    cnn_epochs: int = 15
    catboost_iterations: int | None = None


def _resolve_config(config: RunConfig | int | None) -> RunConfig:
    """Normalize input into a RunConfig instance."""
    if isinstance(config, int):
        return RunConfig(cnn_epochs=config)
    if config is None:
        return RunConfig()
    return config


@dataclass
class _BenchmarkAccumulator:
    """Out-of-fold predictions and fit times accumulated across benchmark folds."""

    oof: dict[str, dict[str, np.ndarray]]
    fit_times: dict[str, dict[str, float]]
    stat_df: pd.DataFrame
    config: RunConfig


def _build_classifier(kind: str, seed: int = 42, iterations: int | None = None) -> Pipeline:
    """Create classification pipeline and apply optional catboost iteration overrides."""
    clf = make_classifier(kind, seed=seed)
    if kind == "catboost" and iterations is not None:
        clf.named_steps["classifier"].set_params(iterations=iterations)
    return clf


def _fit_predict_classical(
    clf: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Fit a classical model pipeline and return test predictions and fit duration (s)."""
    t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    fit_sec = time.perf_counter() - t0
    preds = np.asarray(clf.predict(x_test)).ravel()
    return preds, fit_sec


def _clean_matrix_train_only(df: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """Clean feature matrix using medians computed on training cycles only."""
    cleaned = df.replace([np.inf, -np.inf], np.nan).copy()
    train_slice = cleaned.iloc[train_idx]
    medians = train_slice.median().fillna(0.0)
    return cleaned.fillna(medians)


def fold_feature_sets(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    train_idx: np.ndarray,
    stat_features: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Extract statistical, physics, and hybrid feature matrices for a fold.

    The valve baseline and NaN fill medians are fitted strictly on the training fold (G10).
    """
    if "valve" not in targets.columns:
        raise ValueError("Missing 'valve' target column required for baseline.")
    v_col = targets["valve"]
    v_train = np.asarray(v_col.iloc[train_idx] if hasattr(v_col, "iloc") else v_col[train_idx])
    p_train = data["PS1"][train_idx]
    baseline_s = fit_valve_baseline(p_train, v_train)

    phys_raw = build_physics_features(data, baseline_s)
    phys_df = _clean_matrix_train_only(phys_raw, train_idx)
    stat_raw = extract_statistical_features(data) if stat_features is None else stat_features
    stat_df = _clean_matrix_train_only(stat_raw, train_idx)

    hybrid_df = pd.concat([stat_df, phys_df], axis=1)
    return {
        "statistical": stat_df,
        "physics": phys_df,
        "hybrid": hybrid_df,
    }


def _evaluate_oof_metrics(
    targets: pd.DataFrame,
    oof_preds: dict[str, np.ndarray],
    fit_times: dict[str, float],
    pipeline: str,
) -> list[dict[str, Any]]:
    """Evaluate out-of-fold classification metrics and fit latency per target."""
    rows: list[dict[str, Any]] = []
    for tgt in TARGETS:
        yt = targets[tgt].to_numpy()
        yp = oof_preds[tgt]
        labels = sorted(set(yt).union(set(yp)))
        all_labels = CLASS_LABELS.get(tgt, labels)
        macro_f1 = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
        macro_f1_all = float(f1_score(yt, yp, labels=all_labels, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(yt, yp))
        rows.append(
            {
                "pipeline": pipeline,
                "target": tgt,
                "macro_f1": macro_f1,
                "macro_f1_all_labels": macro_f1_all,
                "balanced_accuracy": bal_acc,
                "fit_seconds": fit_times[tgt],
            }
        )
    return rows


def _init_benchmark_accumulator(
    n_cycles: int,
    config: RunConfig,
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
) -> _BenchmarkAccumulator:
    """Initialize state container for out-of-fold benchmark predictions."""
    stat_df = extract_statistical_features(data)
    oof: dict[str, dict[str, np.ndarray]] = {}
    fit_times: dict[str, dict[str, float]] = {}
    for pipe in PIPELINES:
        oof[pipe] = {tgt: np.empty(n_cycles, dtype=targets[tgt].dtype) for tgt in TARGETS}
        fit_times[pipe] = {tgt: 0.0 for tgt in TARGETS}
    return _BenchmarkAccumulator(oof, fit_times, stat_df, config)


def _fit_predict_catboost_targets(
    feat_dict: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    fold_pair: tuple[np.ndarray, np.ndarray],
    acc: _BenchmarkAccumulator,
) -> None:
    """Train CatBoost models across classical feature sets on a single fold."""
    train_idx, test_idx = fold_pair
    for pipe in ("statistical", "physics", "hybrid"):
        x_tr = feat_dict[pipe].iloc[train_idx]
        x_te = feat_dict[pipe].iloc[test_idx]
        for tgt in TARGETS:
            clf = _build_classifier("catboost", iterations=acc.config.catboost_iterations)
            preds, fit_sec = _fit_predict_classical(clf, x_tr, targets[tgt].iloc[train_idx], x_te)
            acc.oof[pipe][tgt][test_idx] = preds
            acc.fit_times[pipe][tgt] += fit_sec


def _fit_predict_cnn_targets(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    fold_pair: tuple[np.ndarray, np.ndarray],
    acc: _BenchmarkAccumulator,
) -> None:
    """Train MultiRateCNN on training fold and store out-of-fold predictions."""
    train_idx, test_idx = fold_pair
    t0 = time.perf_counter()
    model, meta = train_cnn_model(data, targets, train_idx, epochs=acc.config.cnn_epochs)
    fit_sec = time.perf_counter() - t0
    preds_df = predict_cnn(model, data, test_idx, meta)
    per_target_fit_sec = fit_sec / len(TARGETS)
    for tgt in TARGETS:
        acc.oof["cnn_1d"][tgt][test_idx] = preds_df[tgt].to_numpy()
        acc.fit_times["cnn_1d"][tgt] += per_target_fit_sec


def run_four_way_benchmark(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: RunConfig | int | None = None,
) -> pd.DataFrame:
    """Benchmark statistical, physics, hybrid, and 1D-CNN pipelines across folds."""
    cfg = _resolve_config(config)
    acc = _init_benchmark_accumulator(len(targets), cfg, data, targets)
    for fold_pair in folds:
        train_idx, _ = fold_pair
        feats = fold_feature_sets(data, targets, train_idx, stat_features=acc.stat_df)
        _fit_predict_catboost_targets(feats, targets, fold_pair, acc)
        _fit_predict_cnn_targets(data, targets, fold_pair, acc)

    rows: list[dict[str, Any]] = []
    for pipe in PIPELINES:
        pipe_rows = _evaluate_oof_metrics(targets, acc.oof[pipe], acc.fit_times[pipe], pipe)
        rows.extend(pipe_rows)
    return pd.DataFrame(rows)


def _evaluate_classical_pipeline(
    cleaned_folds: Sequence[tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    targets: pd.DataFrame,
    kind: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Cross-validate a classical model across folds and return predictions and timing."""
    n_cycles = len(targets)
    oof = {tgt: np.empty(n_cycles, dtype=targets[tgt].dtype) for tgt in TARGETS}
    fit_times = {tgt: 0.0 for tgt in TARGETS}

    for train_idx, test_idx, feat_df in cleaned_folds:
        x_tr, x_te = feat_df.iloc[train_idx], feat_df.iloc[test_idx]
        for tgt in TARGETS:
            y_tr = targets[tgt].iloc[train_idx]
            clf = _build_classifier(kind)
            preds, fit_sec = _fit_predict_classical(clf, x_tr, y_tr, x_te)
            oof[tgt][test_idx] = preds
            fit_times[tgt] += fit_sec

    return oof, fit_times


def run_baseline(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Evaluate LDA and Random Forest baselines on statistical features."""
    stat_raw = extract_statistical_features(data)
    cleaned_folds = [
        (train_idx, test_idx, _clean_matrix_train_only(stat_raw, train_idx))
        for train_idx, test_idx in folds
    ]
    rows: list[dict[str, Any]] = []
    for pipe in BASELINE_PIPELINES:
        oof, fit_times = _evaluate_classical_pipeline(cleaned_folds, targets, pipe)
        pipe_rows = _evaluate_oof_metrics(targets, oof, fit_times, pipe)
        rows.extend(pipe_rows)
    return pd.DataFrame(rows)


def _evaluate_temp_shift_models(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    split_indices: tuple[np.ndarray, np.ndarray],
    config: RunConfig,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit all models on cool cycles and predict warm target cycles."""
    train_idx, test_idx = split_indices
    feats = fold_feature_sets(data, targets, train_idx)
    preds: dict[str, dict[str, np.ndarray]] = {pipe: {} for pipe in PIPELINES}

    for pipe in ("statistical", "physics", "hybrid"):
        x_tr, x_te = feats[pipe].iloc[train_idx], feats[pipe].iloc[test_idx]
        for tgt in TARGETS:
            clf = _build_classifier("catboost", iterations=config.catboost_iterations)
            p, _ = _fit_predict_classical(clf, x_tr, targets[tgt].iloc[train_idx], x_te)
            preds[pipe][tgt] = p

    model, meta = train_cnn_model(data, targets, train_idx, epochs=config.cnn_epochs)
    cnn_df = predict_cnn(model, data, test_idx, meta)
    for tgt in TARGETS:
        preds["cnn_1d"][tgt] = cnn_df[tgt].to_numpy()

    return preds


def _assemble_shift_rows(
    cool_df: pd.DataFrame,
    shift_preds: dict[str, dict[str, np.ndarray]],
    test_targets: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Assemble comparative evaluation rows for temperature shift experiment."""
    rows: list[dict[str, Any]] = []
    for pipe in PIPELINES:
        for tgt in TARGETS:
            f1_in_val = cool_df[(cool_df["pipeline"] == pipe) & (cool_df["target"] == tgt)][
                "macro_f1"
            ].iloc[0]
            f1_in = float(f1_in_val)
            yt = test_targets[tgt].to_numpy()
            yp = shift_preds[pipe][tgt]
            labels = sorted(set(yt).union(set(yp)))
            f1_shift = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
            f1_shift_all = float(
                f1_score(yt, yp, labels=CLASS_LABELS[tgt], average="macro", zero_division=0)
            )
            rows.append(
                {
                    "pipeline": pipe,
                    "target": tgt,
                    "f1_in": f1_in,
                    "f1_shift": f1_shift,
                    "f1_shift_all_labels": f1_shift_all,
                    "delta_f1": f1_shift - f1_in,
                }
            )
    return rows


def run_temperature_shift_experiment(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    threshold_c: float | None = None,
    config: RunConfig | int | None = None,
) -> tuple[pd.DataFrame, float]:
    """Evaluate performance degradation under temperature distribution shift."""
    cfg = _resolve_config(config)
    train_idx, test_idx, threshold = temperature_shift_split(data["TS1"], threshold_c)
    cool_data = {k: v[train_idx] for k, v in data.items()}
    cool_targets = targets.iloc[train_idx].reset_index(drop=True)
    cool_folds = blocked_time_ordered_kfold(n_cycles=len(train_idx), n_splits=5)
    cool_df = run_four_way_benchmark(cool_data, cool_targets, cool_folds, config=cfg)

    shift_preds = _evaluate_temp_shift_models(data, targets, (train_idx, test_idx), cfg)
    rows = _assemble_shift_rows(cool_df, shift_preds, targets.iloc[test_idx])
    return pd.DataFrame(rows), threshold


def write_benchmark_report(
    tables: dict[str, pd.DataFrame],
    path: Path | str,
    notes: dict[str, str],
) -> Path:
    """Generate markdown tables and write benchmark report to disk."""
    sections: dict[str, str] = {}
    for title, df in tables.items():
        sections[title] = generate_markdown_table(df)

    if notes:
        note_lines = [f"- **{k}**: {v}" for k, v in notes.items()]
        sections["Experimental Notes and Audit"] = "\n".join(note_lines)

    return write_report(path, "Benchmark Evaluation Report", sections)


def _make_synthetic_channel(name: str, n: int, targets: pd.DataFrame) -> np.ndarray:
    """Generate synthetic channel cycles with class-correlated signals."""
    ch = CHANNEL_REGISTRY[name]
    n_s = ch.n_samples
    rng = np.random.default_rng(42)
    if name == "PS1":
        ramp = np.linspace(120.0, 220.0, n_s, dtype=np.float32)
        base = np.tile(ramp, (n, 1))
        bias = (targets["valve"].to_numpy()[:, None] - 85.0) * 0.1
        arr = (base + bias + rng.normal(0, 0.5, (n, n_s))).astype(np.float32)
        arr[:, 0] = np.arange(n, dtype=np.float32)
        return arr
    if name == "TS1":
        means = np.linspace(35.0, 55.0, n, dtype=np.float32)[:, None]
        return np.tile(means, (1, n_s)) + rng.normal(0, 0.05, (n, n_s)).astype(np.float32)
    if name == "TS3":
        bias = (targets["cooler"].to_numpy()[:, None] - 40.0) * 0.05
        return np.ones((n, n_s), dtype=np.float32) * 40.0 + bias
    if name == "TS4":
        return np.ones((n, n_s), dtype=np.float32) * 30.0
    if name == "FS1":
        return np.ones((n, n_s), dtype=np.float32) * 40.0 + rng.normal(0, 0.1, (n, n_s)).astype(
            np.float32
        )
    if name == "EPS1":
        return np.ones((n, n_s), dtype=np.float32) * 2500.0 + rng.normal(0, 5.0, (n, n_s)).astype(
            np.float32
        )
    return rng.normal(50.0, 2.0, (n, n_s)).astype(np.float32)


def _generate_synthetic_data() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Generate 60-cycle synthetic multi-rate data and balanced targets."""
    n = 60
    targets = pd.DataFrame(
        {
            "cooler": np.tile([3, 20, 100], n // 3),
            "valve": np.tile([73, 80, 90, 100], n // 4),
            "pump": np.tile([0, 1, 2], n // 3),
            "accumulator": np.tile([90, 100, 115, 130], n // 4),
            "stable_flag": np.tile([0, 1], n // 2),
        }
    )
    data: dict[str, np.ndarray] = {}
    for name in PHYSICAL_CHANNELS:
        data[name] = _make_synthetic_channel(name, n, targets)
    return data, targets


def _run_cli_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark run and persist report per CLI arguments."""
    if args.quick_test:
        data, targets = _generate_synthetic_data()
        config = RunConfig(cnn_epochs=1, catboost_iterations=20)
    else:
        from hydro.data.loader import load_cycle_dataset

        data, targets = load_cycle_dataset(args.raw_dir)
        config = RunConfig(cnn_epochs=15, catboost_iterations=None)

    folds = blocked_time_ordered_kfold(n_cycles=len(targets), n_splits=5)
    audit = audit_class_coverage(targets, folds)
    df_four = run_four_way_benchmark(data, targets, folds, config=config)
    df_base = run_baseline(data, targets, folds)
    tables = {"Four-Way Benchmark": df_four, "Baseline Models": df_base}
    notes = {"class_coverage_audit": str(audit)}

    if args.run_temp_shift:
        df_shift, thresh = run_temperature_shift_experiment(data, targets, config=config)
        tables["Temperature Shift Experiment"] = df_shift
        notes["temperature_threshold_c"] = f"{thresh:.2f}"

    out_path = write_benchmark_report(tables, args.output, notes)
    logger.info("Benchmark report written to %s", out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the hydraulic condition benchmark."""
    parser = argparse.ArgumentParser(description="Hydraulic condition monitoring benchmark suite.")
    parser.add_argument(
        "--quick-test", action="store_true", help="Run fast 60-cycle synthetic benchmark"
    )
    parser.add_argument(
        "--run-temp-shift", action="store_true", help="Run temperature shift experiment"
    )
    parser.add_argument(
        "--output", type=str, default="reports/benchmark_results.md", help="Output report path"
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory"
    )
    args = parser.parse_args(argv)
    return _run_cli_benchmark(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
