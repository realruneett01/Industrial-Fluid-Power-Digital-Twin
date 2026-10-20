"""Tests for ONNX model export and numerical parity validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from catboost import CatBoostClassifier

from hydro.models.cnn1d import TARGET_HEADS, MultiRateCNN
from hydro.models.export import (
    export_catboost_to_onnx,
    export_cnn_to_onnx,
    export_models_to_onnx,
    main,
    validate_parity,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def trained_catboost(
    tmp_path: Path,
) -> tuple[CatBoostClassifier, np.ndarray, int]:
    """Provide a tiny trained CatBoost classifier and test input."""
    rng = np.random.default_rng(42)
    n_features = 12
    x_train = rng.normal(0.0, 1.0, (60, n_features)).astype(np.float32)
    y_train = rng.integers(0, 3, size=60)
    model = CatBoostClassifier(
        iterations=20,
        random_seed=42,
        verbose=0,
        train_dir=str(tmp_path / "cb_train"),
    )
    model.fit(x_train, y_train)
    x_test = rng.normal(0.0, 1.0, (50, n_features)).astype(np.float32)
    return model, x_test, n_features


@pytest.fixture
def eval_cnn_model() -> MultiRateCNN:
    """Provide a deterministically initialized MultiRateCNN in eval mode."""
    torch.manual_seed(42)
    model = MultiRateCNN()
    model.eval()
    return model


def test_catboost_export_parity(
    trained_catboost: tuple[CatBoostClassifier, np.ndarray, int],
    tmp_path: Path,
) -> None:
    """CatBoost export must match predict_proba within rtol 1e-5 on 50 random rows."""
    model, x_test, n_features = trained_catboost
    onnx_file = tmp_path / "catboost_cooler.onnx"
    exported_path = export_catboost_to_onnx(model, n_features, onnx_file)
    assert exported_path.is_file()

    def ref_fn(inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [model.predict_proba(inputs["features"])]

    rel_err = validate_parity(
        exported_path,
        ref_fn,
        {"features": x_test},
        rtol=1e-5,
        atol=1e-6,
    )
    assert rel_err < 1e-5


@pytest.mark.parametrize("batch_size", [1, 3])
def test_cnn_export_parity_dynamic_batch(
    eval_cnn_model: MultiRateCNN,
    batch_size: int,
    tmp_path: Path,
) -> None:
    """CNN export must match PyTorch logits within rtol 1e-4 and atol 1e-5 for variable batches."""
    onnx_file = tmp_path / "cnn.onnx"
    export_cnn_to_onnx(eval_cnn_model, onnx_file)
    assert onnx_file.is_file()

    rng = np.random.default_rng(100 + batch_size)
    inputs = {
        "x100": rng.normal(0.0, 1.0, (batch_size, 7, 6000)).astype(np.float32),
        "x10": rng.normal(0.0, 1.0, (batch_size, 2, 600)).astype(np.float32),
        "x1": rng.normal(0.0, 1.0, (batch_size, 5, 60)).astype(np.float32),
    }

    def ref_fn(inp: dict[str, np.ndarray]) -> list[np.ndarray]:
        with torch.no_grad():
            outs = eval_cnn_model(
                torch.from_numpy(inp["x100"]),
                torch.from_numpy(inp["x10"]),
                torch.from_numpy(inp["x1"]),
            )
        return [outs[head].numpy() for head in TARGET_HEADS]

    rel_err = validate_parity(
        onnx_file,
        ref_fn,
        inputs,
        rtol=1e-4,
        atol=1e-5,
    )
    assert rel_err < 1e-3


def test_export_models_to_onnx_dispatch(
    trained_catboost: tuple[CatBoostClassifier, np.ndarray, int],
    eval_cnn_model: MultiRateCNN,
    tmp_path: Path,
) -> None:
    """export_models_to_onnx must return one .onnx path per key in model_dict."""
    cb_model, _, n_features = trained_catboost
    model_dict: dict[str, object] = {
        "catboost_cooler": (cb_model, n_features),
        "cnn": eval_cnn_model,
    }
    out_dir = tmp_path / "models_out"
    exported_paths = export_models_to_onnx(model_dict, out_dir)

    assert len(exported_paths) == 2
    for path in exported_paths:
        assert path.is_file()
        assert path.suffix == ".onnx"
    names = {p.name for p in exported_paths}
    assert names == {"catboost_cooler.onnx", "cnn.onnx"}


def test_validate_parity_raises_on_mismatch(
    trained_catboost: tuple[CatBoostClassifier, np.ndarray, int],
    tmp_path: Path,
) -> None:
    """validate_parity must raise AssertionError if outputs do not match."""
    model, x_test, n_features = trained_catboost
    onnx_file = tmp_path / "mismatch_test.onnx"
    export_catboost_to_onnx(model, n_features, onnx_file)

    def corrupted_ref(inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        probs = model.predict_proba(inputs["features"])
        return [probs + 1.0]

    with pytest.raises(AssertionError, match="Parity"):
        validate_parity(
            onnx_file,
            corrupted_ref,
            {"features": x_test},
            rtol=1e-5,
            atol=1e-6,
        )


def test_main_validate_returns_zero() -> None:
    """main with --validate must exit or return 0."""
    code = main(["--validate"])
    assert code == 0
