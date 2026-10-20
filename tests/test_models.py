"""Tests for classical fault classification models and cross-validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import catboost.utils
import numpy as np
import pandas as pd
import pytest
import torch
from catboost import CatBoostClassifier, CatBoostError
from sklearn.pipeline import Pipeline

import hydro.models.classical
from hydro.data.loader import load_cycle_dataset
from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.eval.metrics import CLASS_LABELS
from hydro.eval.splits import blocked_time_ordered_kfold
from hydro.features import build_feature_matrices
from hydro.features.physics import fit_valve_baseline
from hydro.models.classical import (
    _evaluate_target_folds,
    cross_validate_targets,
    load_model_config,
    make_classifier,
    train_evaluate_baseline,
)
from hydro.models.cnn1d import MultiRateCNN, get_device, train_cnn_model


def _make_synthetic_targets(x: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Create 4 separable target columns from feature inputs."""
    n = len(x)
    s0 = x[:, 0] + rng.normal(0, 0.1, size=n)
    cooler = np.where(s0 < -1.0, 3, np.where(s0 < 1.0, 20, 100))

    s1 = x[:, 1] + rng.normal(0, 0.1, size=n)
    valve = np.where(s1 < -1.5, 73, np.where(s1 < 0.0, 80, np.where(s1 < 1.5, 90, 100)))

    s2 = x[:, 2] + rng.normal(0, 0.1, size=n)
    pump = np.where(s2 < -1.0, 0, np.where(s2 < 1.0, 1, 2))

    s3 = x[:, 3] + rng.normal(0, 0.1, size=n)
    accum = np.where(s3 < -1.5, 90, np.where(s3 < 0.0, 100, np.where(s3 < 1.5, 115, 130)))

    return pd.DataFrame({"cooler": cooler, "valve": valve, "pump": pump, "accumulator": accum})


def _assert_metrics_frame(df: pd.DataFrame) -> None:
    """Assert metrics DataFrame matches expected target index and valid F1 range."""
    assert set(df.index) == set(CLASS_LABELS.keys())
    assert (df["macro_f1"] >= 0.6).all()
    assert (df["macro_f1"] <= 1.0).all()


def _assert_cv_outputs(
    metrics_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    n_folds: int,
) -> None:
    """Assert common cross-validation metric and fold output properties."""
    _assert_metrics_frame(metrics_df)
    assert len(fold_df) == n_folds * 4


def _patch_fast_rf_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch model configuration with lightweight Random Forest hyperparameters."""
    cfg = dict(load_model_config())
    cfg["random_forest"] = {"n_estimators": 50, "n_jobs": 1}
    monkeypatch.setattr("hydro.models.classical.load_model_config", lambda: cfg)


def _make_tiny_cnn_data(n_cycles: int = 8) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Generate minimal multi-rate channel dictionary and targets for CNN smoke tests."""
    rng = np.random.default_rng(0)
    data = {
        name: rng.normal(50.0, 5.0, (n_cycles, CHANNEL_REGISTRY[name].n_samples)).astype(np.float32)
        for name in PHYSICAL_CHANNELS
    }
    targets = pd.DataFrame(
        {
            "cooler": np.array([3, 20, 100, 3] * (n_cycles // 4 + 1))[:n_cycles],
            "valve": np.array([73, 80, 90, 100] * (n_cycles // 4 + 1))[:n_cycles],
            "pump": np.array([0, 1, 2, 0] * (n_cycles // 4 + 1))[:n_cycles],
            "accumulator": np.array([90, 100, 115, 130] * (n_cycles // 4 + 1))[:n_cycles],
        }
    )
    return data, targets


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """Generate 200 synthetic cycles with 6 features and 4 separable targets."""
    n = 200
    rng = np.random.default_rng(42)
    x = rng.uniform(-3.0, 3.0, size=(n, 4))
    noise = rng.standard_normal((n, 2))
    feats = np.hstack([x, noise])
    features = pd.DataFrame(feats, columns=[f"feat_{i}" for i in range(6)])
    targets = _make_synthetic_targets(x, rng)
    folds = blocked_time_ordered_kfold(n_cycles=n, n_splits=5)
    return features, targets, folds


def test_load_model_config() -> None:
    """Verify loading model hyperparameters from yaml configuration."""
    config = load_model_config()
    assert isinstance(config, dict)
    assert "seed" in config
    assert "lda" in config
    assert "random_forest" in config
    assert "catboost" in config


@pytest.mark.parametrize(
    ("kind", "expected_steps"),
    [
        ("lda", ("scaler", "classifier")),
        ("rf", ("classifier",)),
        ("catboost", ("classifier",)),
    ],
)
def test_make_classifier_types(kind: str, expected_steps: tuple[str, ...]) -> None:
    """Verify classifier pipeline creation and component structure."""
    pipe = make_classifier(kind)
    assert isinstance(pipe, Pipeline)
    for step in expected_steps:
        assert step in pipe.named_steps


def test_make_classifier_unknown_raises() -> None:
    """Unknown classifier kind must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown"):
        make_classifier("unsupported_kind")


def test_cross_validate_targets_lda(
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """LDA pipeline achieves >= 0.6 macro-F1 across all 4 targets."""
    features, targets, folds = synthetic_data
    metrics_df, fold_df = cross_validate_targets(features, targets, folds, kind="lda")

    _assert_cv_outputs(metrics_df, fold_df, len(folds))
    assert (metrics_df["n_test"] == len(features)).all()
    assert set(fold_df.columns) == {"fold", "target", "macro_f1"}


def test_cross_validate_targets_rf(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Random Forest pipeline achieves >= 0.6 macro-F1 across all 4 targets."""
    features, targets, folds = synthetic_data
    _patch_fast_rf_config(monkeypatch)

    metrics_df, fold_df = cross_validate_targets(features, targets, folds, kind="rf")
    _assert_cv_outputs(metrics_df, fold_df, len(folds))


def test_oof_predictions_coverage(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Out-of-fold predictions cover every sample index exactly once."""
    features, targets, folds = synthetic_data
    n = len(features)
    captured: list[pd.DataFrame] = []
    orig_eval = hydro.models.classical.evaluate_classification_targets

    def spy_eval(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        captured.append(y_pred)
        return orig_eval(y_true, y_pred)

    monkeypatch.setattr("hydro.models.classical.evaluate_classification_targets", spy_eval)
    cross_validate_targets(features, targets, folds, kind="lda")
    assert len(captured) == 1
    oof_df = captured[0]
    assert len(oof_df) == n
    assert set(oof_df.index) == set(range(n))
    for col in CLASS_LABELS:
        assert set(oof_df[col]).issubset(set(CLASS_LABELS[col]))

    oof_tgt, _ = _evaluate_target_folds(features, targets["cooler"], "cooler", folds, kind="lda")
    assert len(oof_tgt) == n
    assert set(oof_tgt).issubset(set(CLASS_LABELS["cooler"]))


def test_catboost_iterations_override(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """CatBoost runs fast under iterations override and achieves >= 0.6 macro-F1."""
    features, targets, folds = synthetic_data
    custom_cfg: dict[str, Any] = {
        "seed": 42,
        "catboost": {
            "iterations": 15,
            "depth": 4,
            "learning_rate": 0.1,
            "verbose": 0,
        },
    }
    monkeypatch.setattr("hydro.models.classical.load_model_config", lambda: custom_cfg)
    metrics_df, fold_df = cross_validate_targets(features, targets, folds, kind="catboost")
    _assert_cv_outputs(metrics_df, fold_df, len(folds))


def test_train_evaluate_baseline(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """train_evaluate_baseline returns dict of metrics DataFrames keyed by kind."""
    features, targets, folds = synthetic_data
    _patch_fast_rf_config(monkeypatch)

    results = train_evaluate_baseline(features, targets, folds, kinds=("lda", "rf"))

    assert set(results.keys()) == {"lda", "rf"}
    for kind in ("lda", "rf"):
        _assert_metrics_frame(results[kind])


def test_input_validation_errors(
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Input length mismatch or missing column raises ValueError."""
    features, targets, folds = synthetic_data
    with pytest.raises(ValueError, match="Length mismatch"):
        cross_validate_targets(features.iloc[:10], targets, folds, kind="lda")

    with pytest.raises(ValueError, match="missing from targets"):
        cross_validate_targets(features, targets, folds, kind="lda", target_cols=("nonexistent",))


def test_device_helper_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device helper returns CPU when CUDA unavailable and CUDA when available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device() == torch.device("cpu")
    assert str(get_device()) == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == torch.device("cuda")
    assert str(get_device()) == "cuda"

    assert get_device("cpu") == torch.device("cpu")
    assert get_device("cuda") == torch.device("cuda")


def test_cnn_trains_on_selected_device_and_returns_cpu() -> None:
    """Verify CNN trains on specified device and is moved back to CPU in eval mode."""
    data, targets = _make_tiny_cnn_data(n_cycles=8)
    train_idx = np.arange(8)
    model, meta = train_cnn_model(
        data,
        targets,
        train_idx,
        epochs=1,
        batch_size=4,
        device="cpu",
    )
    assert not model.training
    for p in model.parameters():
        assert p.device == torch.device("cpu")
    assert "loss_history" in meta
    assert len(meta["loss_history"]) == 1


def test_catboost_cpu_when_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CatBoost configures CPU task_type when no GPU is detected."""
    monkeypatch.setattr("catboost.utils.get_gpu_device_count", lambda: 0)
    pipe = make_classifier("catboost", seed=42)
    cb = pipe.named_steps["classifier"]
    assert cb.get_params().get("task_type") == "CPU"


def test_catboost_params_identical_cpu_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify hyperparameters are identical between CPU and GPU configs except task_type/devices."""
    monkeypatch.setattr("catboost.utils.get_gpu_device_count", lambda: 0)
    cb_cpu = make_classifier("catboost", seed=42).named_steps["classifier"]
    cpu_params = {k: v for k, v in cb_cpu.get_params().items() if k not in ("task_type", "devices")}

    monkeypatch.setattr("catboost.utils.get_gpu_device_count", lambda: 1)
    cb_gpu = make_classifier("catboost", seed=42).named_steps["classifier"]
    gpu_params = {k: v for k, v in cb_gpu.get_params().items() if k not in ("task_type", "devices")}

    assert cpu_params == gpu_params
    assert cb_cpu.get_params().get("border_count") == 254
    assert cb_gpu.get_params().get("border_count") == 254
    assert cb_cpu.get_params().get("bootstrap_type") is not None
    assert cb_cpu.get_params().get("bootstrap_type") == cb_gpu.get_params().get("bootstrap_type")


def _verify_gpu_fallback_with_exception(
    exc: Exception,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Verify CatBoost GPU path falls back to CPU when fit raises the given exception."""
    features, targets, folds = synthetic_data
    monkeypatch.setattr("catboost.utils.get_gpu_device_count", lambda: 1)
    pipe = make_classifier("catboost", seed=42)
    cb = pipe.named_steps["classifier"]
    assert str(cb.get_params().get("task_type", "")).upper() == "GPU"

    orig_fit = CatBoostClassifier.fit

    def failing_gpu_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        if str(self.get_params().get("task_type", "")).upper() == "GPU":
            raise exc
        return orig_fit(self, *args, **kwargs)

    monkeypatch.setattr(CatBoostClassifier, "fit", failing_gpu_fit)

    with caplog.at_level(logging.WARNING):
        pipe.fit(features.iloc[folds[0][0]], targets["cooler"].iloc[folds[0][0]])

    assert cb.get_params().get("task_type") == "CPU"
    assert "CatBoost GPU fit failed" in caplog.text
    preds = pipe.predict(features.iloc[folds[0][1]])
    assert len(preds) == len(folds[0][1])


def test_catboost_gpu_fallback_on_catboost_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Verify CatBoost GPU path falls back to CPU when fit raises CatBoostError."""
    _verify_gpu_fallback_with_exception(
        CatBoostError("Simulated GPU failure"),
        monkeypatch,
        caplog,
        synthetic_data,
    )


def test_catboost_gpu_fallback_on_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    synthetic_data: tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Verify CatBoost GPU path falls back to CPU when fit raises RuntimeError."""
    _verify_gpu_fallback_with_exception(
        RuntimeError("CUDA runtime error"),
        monkeypatch,
        caplog,
        synthetic_data,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_smoke_cnn_and_catboost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CNN training on CUDA and CatBoost GPU training on live hardware."""
    training_devices: list[torch.device] = []
    orig_forward = MultiRateCNN.forward

    def spy_forward(self: MultiRateCNN, *args: Any, **kwargs: Any) -> Any:
        training_devices.append(next(self.parameters()).device)
        return orig_forward(self, *args, **kwargs)

    monkeypatch.setattr(MultiRateCNN, "forward", spy_forward)
    data, targets = _make_tiny_cnn_data(n_cycles=8)
    model, _ = train_cnn_model(data, targets, np.arange(8), epochs=1, batch_size=4, device="cuda")
    assert len(training_devices) > 0
    assert all(d.type == "cuda" for d in training_devices)
    assert not model.training
    assert all(p.device == torch.device("cpu") for p in model.parameters())

    if catboost.utils.get_gpu_device_count() > 0:
        pipe = make_classifier("catboost", seed=42)
        pipe.named_steps["classifier"].set_params(iterations=2, verbose=0)
        rng = np.random.default_rng(42)
        x_dummy = pd.DataFrame(rng.normal(size=(20, 4)), columns=[f"f{i}" for i in range(4)])
        y_dummy = pd.Series(rng.integers(0, 2, size=20))
        pipe.fit(x_dummy, y_dummy)
        assert len(pipe.predict(x_dummy[:5])) == 5
        assert pipe.named_steps["classifier"].get_all_params()["task_type"] == "GPU"


@pytest.mark.heavy
def test_real_data_smoke(raw_dir: Path) -> None:
    """Verify classical model training on real dataset achieves above-chance accuracy.

    Assumptions:
    - 300 cycles for training, 100 disjoint cycles for testing.
    - Valve inflection baseline fitted strictly on training cycles.
    - Evaluates pump leakage classification (3 classes, chance ~0.33).
    """
    data, targets = load_cycle_dataset(raw_dir)
    train_data = {k: v[:300] for k, v in data.items()}
    test_data = {k: v[300:400] for k, v in data.items()}

    train_valve = targets["valve"].iloc[:300].to_numpy()
    baseline = fit_valve_baseline(train_data["PS1"], train_valve)

    x_train = build_feature_matrices(train_data, baseline)["statistical"]
    x_test = build_feature_matrices(test_data, baseline)["statistical"]

    y_train = targets["pump"].iloc[:300]
    y_test = targets["pump"].iloc[300:400]

    clf = make_classifier("rf", seed=42)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    accuracy = float(np.mean(preds == y_test))

    assert accuracy >= 0.70, f"Pump RF accuracy {accuracy:.2f} below threshold 0.70"
