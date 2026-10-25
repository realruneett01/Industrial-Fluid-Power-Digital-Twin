"""ONNX model export and numerical parity validation for hydraulic models."""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from catboost import CatBoostClassifier
from torch import nn

from hydro.models.cnn1d import TARGET_HEADS, MultiRateCNN
from hydro.models.runtime import _convert_ort_output, inference_session

__all__ = ["_convert_ort_output", "inference_session"]


class _CNNExportTupleWrapper(nn.Module):
    """Module wrapper returning an ordered tuple of head logits for ONNX export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x100: torch.Tensor,
        x10: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outs = self.model(x100, x10, x1)
        if isinstance(outs, dict):
            return tuple(outs[head] for head in TARGET_HEADS)
        return tuple(outs)


def _build_cnn_dummy_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dummy multi-rate input tensors on CPU for ONNX tracing."""
    x100 = torch.zeros(1, 7, 6000, dtype=torch.float32, device="cpu")
    x10 = torch.zeros(1, 2, 600, dtype=torch.float32, device="cpu")
    x1 = torch.zeros(1, 5, 60, dtype=torch.float32, device="cpu")
    return x100, x10, x1


def export_catboost_to_onnx(
    model: Any,
    n_features: int,
    out_path: Path | str,
) -> Path:
    """Export trained CatBoost classifier to ONNX format.

    Parameters
    ----------
    model : Any
        Trained CatBoostClassifier instance or pipeline step.
    n_features : int
        Number of input features expected by the model.
    out_path : Path | str
        Destination path for the exported .onnx model file.

    Returns
    -------
    Path
        Resolved destination path of the exported ONNX model.
    """
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    cb_model = getattr(model, "named_steps", {}).get("classifier", model)
    if hasattr(model, "steps") and not hasattr(cb_model, "save_model"):
        cb_model = model.steps[-1][1]

    if not hasattr(cb_model, "save_model"):
        raise ValueError("Model does not provide a save_model method for ONNX export.")

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cb_model.save_model(str(dest), format="onnx")
    return dest


def export_cnn_to_onnx(
    model: nn.Module,
    out_path: Path | str,
) -> Path:
    """Export MultiRateCNN to ONNX with dynamic batching and opset 17.

    Parameters
    ----------
    model : nn.Module
        MultiRateCNN instance to export.
    out_path : Path | str
        Destination path for the exported .onnx model file.

    Returns
    -------
    Path
        Resolved destination path of the exported ONNX model.
    """
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).cpu().eval()
    wrapper = _CNNExportTupleWrapper(export_model)
    dummy_inputs = _build_cnn_dummy_inputs()

    input_names = ["x100", "x10", "x1"]
    output_names = [f"logits_{head}" for head in TARGET_HEADS]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(dest),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
            training=torch.onnx.TrainingMode.EVAL,
        )
    return dest


def _dispatch_single_model(key: str, value: object, out_path: Path) -> Path:
    """Dispatch a single model entry to the appropriate ONNX export function."""
    if key.startswith("catboost"):
        if isinstance(value, (tuple, list)):
            model, n_features = value[0], int(value[1])
        elif hasattr(value, "n_features_in_"):
            model, n_features = value, int(value.n_features_in_)
        else:
            raise ValueError(f"CatBoost entry '{key}' must provide (model, n_features) tuple.")
        return export_catboost_to_onnx(model, n_features, out_path)
    if key.startswith("cnn"):
        return export_cnn_to_onnx(value, out_path)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported model key '{key}'; expected 'catboost_*' or 'cnn'.")


def export_models_to_onnx(
    model_dict: dict[str, object],
    output_dir: Path | str,
) -> list[Path]:
    """Export a dictionary of models to ONNX files in the target directory.

    Parameters
    ----------
    model_dict : dict[str, object]
        Mapping from model identifier to model instance or (model, n_features) tuple.
    output_dir : Path | str
        Target directory to receive .onnx files.

    Returns
    -------
    list[Path]
        List of paths to exported ONNX files in insertion order.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported_paths: list[Path] = []
    for key, value in model_dict.items():
        out_path = out_dir / f"{key}.onnx"
        exported_paths.append(_dispatch_single_model(key, value, out_path))
    return exported_paths


def _compute_output_rel_error(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float,
    atol: float,
) -> float:
    """Verify parity with np.allclose and return maximum relative error."""
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}")
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        max_abs = float(np.max(np.abs(actual - expected)))
        msg = f"Parity failed: max diff {max_abs:.3e} exceeds rtol={rtol}, atol={atol}"
        raise AssertionError(msg)
    diff = np.abs(actual - expected)
    denom = np.abs(expected)
    rel = np.where(denom > 0, diff / denom, 0.0)
    return float(np.max(rel)) if rel.size > 0 else 0.0


def validate_parity(
    onnx_path: Path | str,
    reference_fn: Callable[[dict[str, np.ndarray]], list[np.ndarray] | np.ndarray],
    inputs: dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> float:
    """Validate numerical parity between ONNX Runtime CPU inference and reference callable.

    Parameters
    ----------
    onnx_path : Path | str
        Path to the exported ONNX model.
    reference_fn : Callable[[dict[str, np.ndarray]], list[np.ndarray] | np.ndarray]
        Callable taking inputs dict and returning list of reference output arrays.
    inputs : dict[str, np.ndarray]
        Input tensors passed to ONNX Runtime and the reference function.
    rtol : float, default 1e-5
        Relative tolerance for np.allclose.
    atol : float, default 1e-6
        Absolute tolerance for np.allclose.

    Returns
    -------
    float
        Maximum relative error across all evaluated outputs.
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref_raw = reference_fn(inputs)
    expected_outputs = list(ref_raw) if isinstance(ref_raw, (list, tuple)) else [ref_raw]

    all_names = [o.name for o in session.get_outputs()]
    if len(expected_outputs) == 1 and "probabilities" in all_names:
        raw_outputs = session.run(["probabilities"], inputs)
    else:
        raw_outputs = session.run(None, inputs)

    if len(raw_outputs) != len(expected_outputs):
        msg = f"Output count mismatch: got {len(raw_outputs)}, expected {len(expected_outputs)}"
        raise AssertionError(msg)

    max_rel_error = 0.0
    for act_raw, exp_raw in zip(raw_outputs, expected_outputs, strict=True):
        actual = _convert_ort_output(act_raw)
        expected = np.asarray(exp_raw)
        rel_err = _compute_output_rel_error(actual, expected, rtol, atol)
        max_rel_error = max(max_rel_error, rel_err)

    return max_rel_error


def _validate_cb_parity(tmp_dir: Path) -> float:
    """Train tiny CatBoost on synthetic data, export, and validate probability parity."""
    rng = np.random.default_rng(42)
    x_train = rng.normal(0.0, 1.0, (50, 10)).astype(np.float32)
    y_train = rng.integers(0, 3, size=50)
    model = CatBoostClassifier(
        iterations=20,
        random_seed=42,
        verbose=0,
        train_dir=str(tmp_dir / "cb_train"),
    )
    model.fit(x_train, y_train)

    cb_path = export_catboost_to_onnx(model, 10, tmp_dir / "catboost_cooler.onnx")
    x_val = rng.normal(0.0, 1.0, (50, 10)).astype(np.float32)
    return validate_parity(
        cb_path,
        lambda inp: [model.predict_proba(inp["features"])],
        {"features": x_val},
        rtol=1e-5,
        atol=1e-6,
    )


def _validate_cnn_parity(tmp_dir: Path) -> float:
    """Initialize MultiRateCNN on CPU, export, and validate logit parity."""
    torch.manual_seed(42)
    cnn_model = MultiRateCNN()
    cnn_model.eval()
    cnn_path = export_cnn_to_onnx(cnn_model, tmp_dir / "cnn.onnx")

    rng = np.random.default_rng(42)
    inputs = {
        "x100": rng.normal(0.0, 1.0, (2, 7, 6000)).astype(np.float32),
        "x10": rng.normal(0.0, 1.0, (2, 2, 600)).astype(np.float32),
        "x1": rng.normal(0.0, 1.0, (2, 5, 60)).astype(np.float32),
    }

    def ref_cnn(inp: dict[str, np.ndarray]) -> list[np.ndarray]:
        with torch.no_grad():
            outs = cnn_model(
                torch.from_numpy(inp["x100"]),
                torch.from_numpy(inp["x10"]),
                torch.from_numpy(inp["x1"]),
            )
        return [outs[head].numpy() for head in TARGET_HEADS]

    return validate_parity(
        cnn_path,
        ref_cnn,
        inputs,
        rtol=1e-4,
        atol=1e-5,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for exporting and validating ONNX models.

    Parameters
    ----------
    argv : list[str] | None, default None
        Command-line arguments.

    Returns
    -------
    int
        Exit status code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Export and validate hydraulic twin ONNX models.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Train synthetic models, export, and verify ONNX runtime numerical parity.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.validate:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            cb_err = _validate_cb_parity(tmp_dir)
            cnn_err = _validate_cnn_parity(tmp_dir)
            print(f"CatBoost parity max relative error: {cb_err:.2e}")
            print(f"CNN parity max relative error: {cnn_err:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
