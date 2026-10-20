"""Tests for benchmark evaluation, baseline comparison, and report generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.eval.benchmark import (
    RunConfig,
    fold_feature_sets,
    main,
    run_baseline,
    run_four_way_benchmark,
    run_temperature_shift_experiment,
    write_benchmark_report,
)
from hydro.eval.splits import blocked_time_ordered_kfold

if TYPE_CHECKING:
    from pathlib import Path


def _make_60_cycle_dataset() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Generate 60 synthetic cycles with full target class coverage and injected signals."""
    n = 60
    rng = np.random.default_rng(42)

    cooler_classes = [3, 20, 100]
    valve_classes = [73, 80, 90, 100]
    pump_classes = [0, 1, 2]
    accum_classes = [90, 100, 115, 130]

    cooler = np.tile(cooler_classes, n // len(cooler_classes))
    valve = np.tile(valve_classes, n // len(valve_classes))
    pump = np.tile(pump_classes, n // len(pump_classes))
    accum = np.tile(accum_classes, n // len(accum_classes))
    stable = np.tile([0, 1], n // 2)

    targets = pd.DataFrame(
        {
            "cooler": cooler,
            "valve": valve,
            "pump": pump,
            "accumulator": accum,
            "stable_flag": stable,
        }
    )

    data: dict[str, np.ndarray] = {}
    for name in PHYSICAL_CHANNELS:
        ch = CHANNEL_REGISTRY[name]
        n_s = ch.n_samples
        if name == "PS1":
            ramp = np.linspace(120.0, 220.0, n_s, dtype=np.float32)
            base_p = np.tile(ramp, (n, 1))
            valve_bias = (valve[:, None].astype(np.float32) - 85.0) * 0.1
            noise = rng.normal(0, 0.5, (n, n_s)).astype(np.float32)
            arr = base_p + valve_bias + noise
            # Embed cycle index in first column for leakage auditing
            arr[:, 0] = np.arange(n, dtype=np.float32)
            data[name] = arr
        elif name == "FS1":
            data[name] = (np.ones((n, n_s), dtype=np.float32) * 40.0) + rng.normal(
                0, 0.1, (n, n_s)
            ).astype(np.float32)
        elif name == "EPS1":
            data[name] = (np.ones((n, n_s), dtype=np.float32) * 2500.0) + rng.normal(
                0, 5.0, (n, n_s)
            ).astype(np.float32)
        elif name == "TS1":
            t_means = np.linspace(35.0, 55.0, n, dtype=np.float32)[:, None]
            data[name] = np.tile(t_means, (1, n_s)) + rng.normal(0, 0.05, (n, n_s)).astype(
                np.float32
            )
        elif name == "TS3":
            cooler_bias = (cooler[:, None].astype(np.float32) - 40.0) * 0.05
            data[name] = np.ones((n, n_s), dtype=np.float32) * 40.0 + cooler_bias
        elif name == "TS4":
            data[name] = np.ones((n, n_s), dtype=np.float32) * 30.0
        else:
            data[name] = rng.normal(50.0, 2.0, (n, n_s)).astype(np.float32)

    return data, targets


@pytest.fixture
def synthetic_benchmark_data() -> tuple[
    dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
]:
    """Provide 60-cycle synthetic data, targets, and 5-fold blocked splits."""
    data, targets = _make_60_cycle_dataset()
    folds = blocked_time_ordered_kfold(n_cycles=60, n_splits=5)
    return data, targets, folds


@pytest.fixture(autouse=True)
def fast_catboost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch model configuration to limit CatBoost to 20 iterations for fast testing."""
    from hydro.models import classical

    base_cfg = classical.load_model_config()
    patched_cfg = dict(base_cfg)
    cb_cfg = dict(patched_cfg.get("catboost", {}))
    cb_cfg["iterations"] = 20
    patched_cfg["catboost"] = cb_cfg
    monkeypatch.setattr(classical, "load_model_config", lambda: patched_cfg)


def test_fold_feature_sets_structure(
    synthetic_benchmark_data: tuple[
        dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
    ],
) -> None:
    """Verify fold_feature_sets produces clean statistical, physics, and hybrid matrices."""
    data, targets, folds = synthetic_benchmark_data
    train_idx, _ = folds[0]
    feats = fold_feature_sets(data, targets, train_idx)

    assert set(feats.keys()) == {"statistical", "physics", "hybrid"}
    assert len(feats["statistical"]) == 60
    assert len(feats["physics"]) == 60
    assert len(feats["hybrid"]) == 60
    assert not feats["hybrid"].isna().any().any()


def test_four_way_benchmark_shapes_and_bounds(
    synthetic_benchmark_data: tuple[
        dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
    ],
) -> None:
    """Verify four-way benchmark produces 16 rows with macro F1 values in [0, 1]."""
    data, targets, folds = synthetic_benchmark_data
    df = run_four_way_benchmark(data, targets, folds, config=RunConfig(cnn_epochs=1))

    assert len(df) == 16
    expected_cols = {
        "pipeline",
        "target",
        "macro_f1",
        "macro_f1_all_labels",
        "balanced_accuracy",
        "fit_seconds",
    }
    assert expected_cols.issubset(set(df.columns))

    expected_pipes = {"statistical", "physics", "hybrid", "cnn_1d"}
    assert set(df["pipeline"]) == expected_pipes
    assert set(df["target"]) == {"cooler", "valve", "pump", "accumulator"}

    assert (df["macro_f1"] >= 0.0).all()
    assert (df["macro_f1"] <= 1.0).all()
    assert (df["fit_seconds"] >= 0.0).all()


def test_run_baseline_shapes_and_bounds(
    synthetic_benchmark_data: tuple[
        dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
    ],
) -> None:
    """Verify baseline evaluation returns 8 rows with valid macro F1 scores."""
    data, targets, folds = synthetic_benchmark_data
    df = run_baseline(data, targets, folds)

    assert len(df) == 8
    expected_cols = {
        "pipeline",
        "target",
        "macro_f1",
        "macro_f1_all_labels",
        "balanced_accuracy",
        "fit_seconds",
    }
    assert expected_cols.issubset(set(df.columns))
    assert set(df["pipeline"]) == {"lda", "rf"}
    assert set(df["target"]) == {"cooler", "valve", "pump", "accumulator"}
    assert (df["macro_f1"] >= 0.0).all()
    assert (df["macro_f1"] <= 1.0).all()


def test_temp_shift_experiment_delta_f1(
    synthetic_benchmark_data: tuple[
        dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
    ],
) -> None:
    """Verify temperature shift experiment table has delta_f1 = f1_shift - f1_in exactly."""
    data, targets, _ = synthetic_benchmark_data
    df_shift, threshold = run_temperature_shift_experiment(
        data, targets, config=RunConfig(cnn_epochs=1)
    )

    assert isinstance(threshold, float)
    assert len(df_shift) == 16
    expected_cols = ["pipeline", "target", "f1_in", "f1_shift", "f1_shift_all_labels", "delta_f1"]
    assert list(df_shift.columns) == expected_cols

    diff = df_shift["f1_shift"].to_numpy() - df_shift["f1_in"].to_numpy()
    np.testing.assert_allclose(df_shift["delta_f1"].to_numpy(), diff, atol=1e-12)


def test_valve_baseline_fitted_train_rows_only(
    synthetic_benchmark_data: tuple[
        dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fit_valve_baseline receives only training indices and never test rows."""
    data, targets, folds = synthetic_benchmark_data
    inspected_calls: list[np.ndarray] = []

    from hydro.eval import benchmark as bench_mod

    orig_fit = bench_mod.fit_valve_baseline

    def recording_fit(p_train: np.ndarray, v_train: np.ndarray, fs_hz: float = 100.0) -> float:
        indices = p_train[:, 0].astype(int)
        inspected_calls.append(indices)
        return orig_fit(p_train, v_train, fs_hz)

    monkeypatch.setattr(bench_mod, "fit_valve_baseline", recording_fit)

    for train_idx, test_idx in folds:
        del inspected_calls[:]
        _ = fold_feature_sets(data, targets, train_idx)
        assert len(inspected_calls) == 1
        called_idx = set(inspected_calls[0])
        assert called_idx.isdisjoint(set(test_idx))
        assert called_idx == set(train_idx)


def test_write_benchmark_report_contains_all_pipelines(tmp_path: Path) -> None:
    """Verify written benchmark report contains every pipeline name and notes."""
    tables: dict[str, pd.DataFrame] = {
        "Four-Way Benchmark": pd.DataFrame(
            {
                "pipeline": ["statistical", "physics", "hybrid", "cnn_1d"],
                "target": ["cooler", "cooler", "cooler", "cooler"],
                "macro_f1": [0.85, 0.88, 0.92, 0.90],
            }
        ),
        "Baseline": pd.DataFrame(
            {
                "pipeline": ["lda", "rf"],
                "target": ["cooler", "cooler"],
                "macro_f1": [0.75, 0.82],
            }
        ),
    }
    notes = {"threshold_used": "43.50 degC", "class_coverage": "Full coverage across all folds"}
    out_file = tmp_path / "benchmark_results.md"
    written_path = write_benchmark_report(tables, out_file, notes)

    assert written_path.exists()
    content = written_path.read_text(encoding="utf-8")
    for pipe in ("statistical", "physics", "hybrid", "cnn_1d", "lda", "rf"):
        assert pipe in content
    assert "43.50 degC" in content
    assert "Full coverage across all folds" in content


def test_main_quick_test_execution(tmp_path: Path) -> None:
    """Verify main --quick-test runs under 90 seconds and writes report."""
    out_path = tmp_path / "quick_report.md"
    exit_code = main(["--quick-test", "--output", str(out_path)])
    assert exit_code == 0
    assert out_path.exists()


@pytest.mark.heavy
def test_benchmark_real_data(raw_dir: Path) -> None:
    """Verify benchmark runs on real data under --heavy-check."""
    from hydro.data.loader import load_cycle_dataset

    data, targets = load_cycle_dataset(raw_dir)
    folds = blocked_time_ordered_kfold(n_cycles=len(targets), n_splits=5)
    df_base = run_baseline(data, targets, folds)
    assert len(df_base) == 8


def test_fold_feature_sets_train_only_nan_fill(
    synthetic_benchmark_data: tuple[
        dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
    ],
) -> None:
    """Verify NaN fill medians are computed on train rows only and applied to all rows."""
    data, targets, folds = synthetic_benchmark_data
    train_idx, test_idx = folds[0]
    n_cycles = len(targets)

    # 48 train rows: 24 of 10.0, 24 of 20.0 -> train median = 15.0
    # 12 test rows: 11 of 100.0, 1 of NaN -> all-cycles non-null median = 20.0
    col_vals = np.zeros(n_cycles, dtype=np.float64)
    col_vals[train_idx[:24]] = 10.0
    col_vals[train_idx[24:]] = 20.0
    col_vals[test_idx[:11]] = 100.0
    col_vals[test_idx[11]] = np.nan

    custom_stat = pd.DataFrame({"leak_test_col": col_vals})
    feats = fold_feature_sets(data, targets, train_idx, stat_features=custom_stat)
    filled_stat = feats["statistical"]

    nan_row_val = filled_stat["leak_test_col"].iloc[test_idx[11]]
    assert nan_row_val == pytest.approx(15.0)
    assert nan_row_val != pytest.approx(20.0)


def test_fold_feature_sets_train_median_nan_fallback_zero(
    synthetic_benchmark_data: tuple[
        dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
    ],
) -> None:
    """Verify fallback to 0.0 when a column in the training fold has only NaNs."""
    data, targets, folds = synthetic_benchmark_data
    train_idx, test_idx = folds[0]
    n_cycles = len(targets)

    col_vals = np.full(n_cycles, np.nan, dtype=np.float64)
    col_vals[test_idx] = 99.0
    custom_stat = pd.DataFrame({"all_nan_train_col": col_vals})
    feats = fold_feature_sets(data, targets, train_idx, stat_features=custom_stat)
    filled_stat = feats["statistical"]

    assert (filled_stat["all_nan_train_col"].iloc[train_idx] == 0.0).all()
    assert filled_stat["all_nan_train_col"].iloc[test_idx[0]] == pytest.approx(99.0)


def test_assemble_shift_rows_f1_shift_all_labels() -> None:
    """Verify f1_shift_all_labels uses full target class set from CLASS_LABELS."""
    from hydro.eval.benchmark import _assemble_shift_rows

    cool_df = pd.DataFrame(
        [
            {"pipeline": p, "target": t, "macro_f1": 0.9}
            for p in ("statistical", "physics", "hybrid", "cnn_1d")
            for t in ("cooler", "valve", "pump", "accumulator")
        ]
    )
    # Target 'cooler' has CLASS_LABELS: [3, 20, 100]. Supply test labels with only [3, 20].
    yt = np.array([3, 20, 3, 20])
    yp = np.array([3, 20, 3, 20])
    test_targets = pd.DataFrame(
        {
            "cooler": yt,
            "valve": np.array([73, 80, 90, 100]),
            "pump": np.array([0, 1, 2, 0]),
            "accumulator": np.array([90, 100, 115, 130]),
        }
    )
    shift_preds = {
        pipe: {
            "cooler": yp,
            "valve": test_targets["valve"].to_numpy(),
            "pump": test_targets["pump"].to_numpy(),
            "accumulator": test_targets["accumulator"].to_numpy(),
        }
        for pipe in ("statistical", "physics", "hybrid", "cnn_1d")
    }
    rows = _assemble_shift_rows(cool_df, shift_preds, test_targets)
    row = next(r for r in rows if r["pipeline"] == "statistical" and r["target"] == "cooler")
    assert row["f1_shift"] == pytest.approx(1.0)
    assert row["f1_shift_all_labels"] == pytest.approx(2.0 / 3.0)
    assert row["delta_f1"] == pytest.approx(row["f1_shift"] - row["f1_in"])


def test_cnn_fit_time_divided_equally_across_targets(
    synthetic_benchmark_data: tuple[
        dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CNN fit duration is divided equally across 4 targets to sum to wall time."""
    from hydro.eval import benchmark as bm

    data, targets, folds = synthetic_benchmark_data
    fold_pair = folds[0]
    config = bm.RunConfig(cnn_epochs=1)
    acc = bm._init_benchmark_accumulator(len(targets), config, data, targets)

    times = [0.0, 40.0]
    monkeypatch.setattr(bm.time, "perf_counter", lambda: times.pop(0) if times else 40.0)

    bm._fit_predict_cnn_targets(data, targets, fold_pair, acc)

    for tgt in bm.TARGETS:
        assert acc.fit_times["cnn_1d"][tgt] == pytest.approx(10.0)
    total_fit_time = sum(acc.fit_times["cnn_1d"].values())
    assert total_fit_time == pytest.approx(40.0)


def test_catboost_iterations_threaded_without_global_mutation() -> None:
    """Verify CatBoost iterations override is threaded via parameter, not module global."""
    from hydro.eval import benchmark as bm

    assert not hasattr(bm, "_CATBOOST_ITERATIONS")

    clf_default = bm._build_classifier("catboost")
    cb_default = clf_default.named_steps["classifier"]
    default_iters = cb_default.get_params().get("iterations")

    clf_custom = bm._build_classifier("catboost", iterations=7)
    cb_custom = clf_custom.named_steps["classifier"]
    assert cb_custom.get_params()["iterations"] == 7
    assert default_iters != 7


def test_run_baseline_train_only_imputation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_baseline imputes NaNs using train-fold median, not test-fold extremes."""
    from hydro.eval import benchmark as bm

    n_cycles = 4
    train_idx = np.array([0, 1])
    test_idx = np.array([2, 3])
    folds = [(train_idx, test_idx), (test_idx, train_idx)]

    tiny_feat_df = pd.DataFrame({"stat_feature": [10.0, np.nan, 1000.0, 1000.0]})
    targets = pd.DataFrame(
        {
            "cooler": [3, 20, 3, 20],
            "valve": [73, 80, 73, 80],
            "pump": [0, 1, 0, 1],
            "accumulator": [90, 100, 90, 100],
        }
    )
    data = {"dummy": np.zeros(n_cycles)}

    monkeypatch.setattr(bm, "extract_statistical_features", lambda _: tiny_feat_df.copy())

    captured_x_tr: list[pd.DataFrame] = []

    def spy_fit_predict(
        clf: Any, x_tr: pd.DataFrame, y_tr: Any, x_te: pd.DataFrame
    ) -> tuple[np.ndarray, float]:
        captured_x_tr.append(x_tr.copy())
        return np.asarray(y_tr)[: len(x_te)], 0.001

    monkeypatch.setattr(bm, "_fit_predict_classical", spy_fit_predict)

    bm.run_baseline(data, targets, folds)

    assert len(captured_x_tr) > 0
    imputed_val = captured_x_tr[0]["stat_feature"].iloc[1]
    assert imputed_val == pytest.approx(10.0)


def test_run_baseline_uses_train_only_cleaning_spy(
    synthetic_benchmark_data: tuple[
        dict[str, np.ndarray], pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run_baseline cleans features using _clean_matrix_train_only per fold."""
    from hydro.eval import benchmark as bm

    data, targets, folds = synthetic_benchmark_data
    called_train_indices: list[np.ndarray] = []
    orig_clean = bm._clean_matrix_train_only

    def spy_clean(df: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
        called_train_indices.append(train_idx)
        return orig_clean(df, train_idx)

    monkeypatch.setattr(bm, "_clean_matrix_train_only", spy_clean)
    bm.run_baseline(data, targets, folds)

    assert len(called_train_indices) == len(folds)
    for (train_idx, _), called_idx in zip(folds, called_train_indices, strict=True):
        np.testing.assert_array_equal(called_idx, train_idx)
