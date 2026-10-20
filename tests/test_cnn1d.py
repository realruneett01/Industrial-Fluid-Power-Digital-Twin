"""Tests for MultiRateCNN 1D convolutional model and training pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import torch

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.models.cnn1d import (
    MultiRateCNN,
    encode_targets,
    fit_channel_stats,
    predict_cnn,
    prepare_inputs,
    train_cnn_model,
)
from hydro.models.export import export_cnn_to_onnx

if TYPE_CHECKING:
    from pathlib import Path


def _generate_synthetic_cycle_data(n_cycles: int = 32, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate synthetic multi-rate channel data dictionary."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for name in PHYSICAL_CHANNELS:
        ch = CHANNEL_REGISTRY[name]
        data[name] = rng.normal(50.0, 5.0, (n_cycles, ch.n_samples)).astype(np.float32)
    return data


def _generate_synthetic_targets(n_cycles: int = 32) -> pd.DataFrame:
    """Generate synthetic targets dataframe with known ZeMA classes."""
    cooler_classes = [3, 20, 100]
    valve_classes = [73, 80, 90, 100]
    pump_classes = [0, 1, 2]
    accum_classes = [90, 100, 115, 130]

    cooler = np.array([cooler_classes[i % len(cooler_classes)] for i in range(n_cycles)])
    valve = np.array([valve_classes[i % len(valve_classes)] for i in range(n_cycles)])
    pump = np.array([pump_classes[i % len(pump_classes)] for i in range(n_cycles)])
    accum = np.array([accum_classes[i % len(accum_classes)] for i in range(n_cycles)])

    return pd.DataFrame(
        {
            "cooler": cooler,
            "valve": valve,
            "pump": pump,
            "accumulator": accum,
        }
    )


def test_param_count_under_250k() -> None:
    """Assert total MultiRateCNN parameter count is strictly under 250,000."""
    model = MultiRateCNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert 0 < total_params < 250_000


def test_forward_output_shapes_batch_4() -> None:
    """Verify forward pass output shapes for batch size 4 match target heads."""
    model = MultiRateCNN()
    model.eval()
    x100 = torch.randn(4, 7, 6000)
    x10 = torch.randn(4, 2, 600)
    x1 = torch.randn(4, 5, 60)

    with torch.no_grad():
        outputs = model(x100, x10, x1)

    expected_heads = {"cooler": 3, "valve": 4, "pump": 3, "accumulator": 4}
    assert set(outputs.keys()) == set(expected_heads.keys())
    for name, num_classes in expected_heads.items():
        assert outputs[name].shape == (4, num_classes)


def test_fit_channel_stats_no_leakage() -> None:
    """Verify mutating non-training rows does not affect fit_channel_stats."""
    data = _generate_synthetic_cycle_data(n_cycles=16, seed=0)
    train_idx = np.array([0, 1, 2, 3, 4, 5])

    stats_before = fit_channel_stats(data, train_idx)

    # Corrupt non-training rows
    for ch in data:
        data[ch][6:] += 9999.0

    stats_after = fit_channel_stats(data, train_idx)

    for ch in data:
        np.testing.assert_allclose(stats_before[ch][0], stats_after[ch][0])
        np.testing.assert_allclose(stats_before[ch][1], stats_after[ch][1])


def test_prepare_inputs_shapes_and_order() -> None:
    """Verify prepare_inputs returns 3 float32 tensors with expected shapes and ordering."""
    data = _generate_synthetic_cycle_data(n_cycles=8, seed=1)
    train_idx = np.arange(8)
    stats = fit_channel_stats(data, train_idx)

    t100, t10, t1 = prepare_inputs(data, train_idx, stats)

    assert t100.shape == (8, 7, 6000)
    assert t10.shape == (8, 2, 600)
    assert t1.shape == (8, 5, 60)
    assert t100.dtype == torch.float32
    assert t10.dtype == torch.float32
    assert t1.dtype == torch.float32


def test_encode_targets() -> None:
    """Verify encode_targets produces integer codes and sorted unique classes."""
    targets = _generate_synthetic_targets(n_cycles=12)
    codes, classes = encode_targets(targets)

    for col in ("cooler", "valve", "pump", "accumulator"):
        assert col in codes
        assert col in classes
        assert np.all(codes[col] >= 0)
        assert np.all(codes[col] < len(classes[col]))
        # Reconstruct and compare
        reconstructed = classes[col][codes[col]]
        np.testing.assert_array_equal(reconstructed, targets[col].to_numpy())


def test_train_loss_decreases() -> None:
    """Verify 3 epochs on 32 synthetic cycles with encoded signal reduces loss."""
    n_cycles = 32
    data = _generate_synthetic_cycle_data(n_cycles=n_cycles, seed=7)
    targets = _generate_synthetic_targets(n_cycles=n_cycles)

    # Encode target signal strongly into channel means so loss drops rapidly
    for i in range(n_cycles):
        data["PS1"][i] += float(targets["cooler"].iloc[i]) * 10.0
        data["FS1"][i] += float(targets["valve"].iloc[i]) * 10.0
        data["TS1"][i] += float(targets["pump"].iloc[i]) * 10.0
        data["VS1"][i] += float(targets["accumulator"].iloc[i]) * 10.0

    train_idx = np.arange(n_cycles)
    model, meta = train_cnn_model(
        data,
        targets,
        train_idx,
        epochs=3,
        batch_size=16,
        lr=5e-3,
        seed=42,
    )

    loss_history = meta.get("loss_history", [])
    assert len(loss_history) == 3
    assert loss_history[-1] < loss_history[0]


def test_predict_cnn_known_classes() -> None:
    """Verify predict_cnn returns 4 target columns containing only known class values."""
    n_cycles = 32
    data = _generate_synthetic_cycle_data(n_cycles=n_cycles, seed=9)
    targets = _generate_synthetic_targets(n_cycles=n_cycles)
    train_idx = np.arange(24)
    eval_idx = np.arange(24, 32)

    model, meta = train_cnn_model(
        data,
        targets,
        train_idx,
        epochs=1,
        batch_size=16,
        seed=42,
    )

    preds = predict_cnn(model, data, eval_idx, meta)

    assert isinstance(preds, pd.DataFrame)
    expected_cols = ["cooler", "valve", "pump", "accumulator"]
    assert list(preds.columns) == expected_cols
    assert len(preds) == len(eval_idx)

    for col in expected_cols:
        known = set(meta["classes"][col])
        pred_vals = set(preds[col].tolist())
        assert pred_vals.issubset(known)


@pytest.mark.heavy
def test_cnn_real_data(raw_dir: Path) -> None:
    """Smoke test MultiRateCNN on real ZeMA raw data subset."""
    from hydro.data.loader import load_cycle_dataset

    cycles, targets = load_cycle_dataset(raw_dir, exclude_virtual=True)
    subset_idx = np.arange(16)
    stats = fit_channel_stats(cycles, subset_idx)
    t100, t10, t1 = prepare_inputs(cycles, subset_idx, stats)

    model = MultiRateCNN()
    model.eval()
    with torch.no_grad():
        out = model(t100, t10, t1)

    assert set(out.keys()) == {"cooler", "valve", "pump", "accumulator"}
    assert out["cooler"].shape == (16, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_predict_and_export_cnn_cuda(tmp_path: Path) -> None:
    """Verify predict_cnn and export_cnn_to_onnx work for CUDA model and keep device."""
    data = _generate_synthetic_cycle_data(n_cycles=8, seed=42)
    targets = _generate_synthetic_targets(n_cycles=8)
    _, classes = encode_targets(targets)
    stats = fit_channel_stats(data, np.arange(8))
    meta = {"stats": stats, "classes": classes}
    model = MultiRateCNN().cuda()
    preds = predict_cnn(model, data, np.arange(4), meta)
    assert len(preds) == 4
    onnx_file = tmp_path / "model_cuda.onnx"
    exported_path = export_cnn_to_onnx(model, onnx_file)
    assert exported_path.is_file()
    assert next(model.parameters()).device.type == "cuda"


def test_train_cnn_model_deterministic_flag() -> None:
    """Verify train_cnn_model sets torch.backends.cudnn.deterministic to True."""
    data = _generate_synthetic_cycle_data(n_cycles=8, seed=42)
    targets = _generate_synthetic_targets(n_cycles=8)
    torch.backends.cudnn.deterministic = False
    train_cnn_model(data, targets, np.arange(8), epochs=1, batch_size=8, seed=42)
    assert torch.backends.cudnn.deterministic is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_seeded_cuda_trainings_identical() -> None:
    """Verify two seeded CUDA trainings of a tiny model yield identical predictions."""
    data = _generate_synthetic_cycle_data(n_cycles=64, seed=123)
    targets = _generate_synthetic_targets(n_cycles=64)
    idx = np.arange(64)
    torch.backends.cudnn.deterministic = False
    m1, meta1 = train_cnn_model(data, targets, idx, epochs=5, batch_size=8, seed=99, device="cuda")
    preds1 = predict_cnn(m1, data, idx, meta1)
    torch.backends.cudnn.deterministic = False
    m2, meta2 = train_cnn_model(data, targets, idx, epochs=5, batch_size=8, seed=99, device="cuda")
    preds2 = predict_cnn(m2, data, idx, meta2)
    pd.testing.assert_frame_equal(preds1, preds2)
