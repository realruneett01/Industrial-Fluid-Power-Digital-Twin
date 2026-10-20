"""Tests for hydro.online.predictor: ONNX bundle export, serving, and feature parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydro.features.physics import fit_valve_baseline
from hydro.online.advisor_client import health_index
from hydro.online.predictor import (
    TARGETS,
    OnnxCyclePredictor,
    extract_cycle_features,
    train_onnx_bundle,
)
from tests.test_profile_latency import synthetic_dataset

__all__ = ["synthetic_dataset"]


def test_train_onnx_bundle_and_predictor_from_dir(
    tmp_path: Path,
    synthetic_dataset: tuple[dict[str, np.ndarray], pd.DataFrame],
) -> None:
    """Verify bundle training exports valid ONNX models and bundle metadata."""
    data, targets = synthetic_dataset
    train_idx = np.arange(0, 6)
    out_dir = train_onnx_bundle(data, targets, train_idx, tmp_path, iterations=5)
    assert out_dir == tmp_path

    bundle_json_path = tmp_path / "bundle.json"
    assert bundle_json_path.exists()
    with bundle_json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    predictor = OnnxCyclePredictor.from_dir(tmp_path)
    cycle = {ch: data[ch][6:7] for ch in data}
    results = predictor(cycle)

    assert set(results.keys()) == set(TARGETS)
    for tgt in TARGETS:
        labels, probs = results[tgt]
        expected_labels = meta["classes"][tgt]
        np.testing.assert_array_equal(labels, expected_labels)
        assert probs.ndim == 1
        assert np.isclose(float(np.sum(probs)), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_online_feature_parity_with_history(
    synthetic_dataset: tuple[dict[str, np.ndarray], pd.DataFrame],
) -> None:
    """Verify online feature rows match offline multi-cycle stacked rows for k >= 4."""
    data, targets = synthetic_dataset
    baseline = fit_valve_baseline(data["PS1"][:6], np.asarray(targets["valve"])[:6])
    stacked_7 = {ch: data[ch][:7] for ch in data}
    offline_features = extract_cycle_features(stacked_7, baseline)

    predictor = OnnxCyclePredictor(valve_baseline_s=baseline)
    for k in range(7):
        cycle = {ch: data[ch][k : k + 1] for ch in data}
        feat_row = predictor.compute_features(cycle)
        if k >= 4:
            np.testing.assert_allclose(feat_row[0], offline_features[k], atol=1e-5)
            assert predictor.last_features is not None
            np.testing.assert_allclose(predictor.last_features[0], offline_features[k], atol=1e-5)


def test_predictor_output_works_with_health_index(
    tmp_path: Path,
    synthetic_dataset: tuple[dict[str, np.ndarray], pd.DataFrame],
) -> None:
    """Verify OnnxCyclePredictor output schema is compatible with health_index."""
    data, targets = synthetic_dataset
    train_idx = np.arange(0, 6)
    train_onnx_bundle(data, targets, train_idx, tmp_path, iterations=5)
    predictor = OnnxCyclePredictor.from_dir(tmp_path)

    cycle = {ch: data[ch][6:7] for ch in data}
    probs = predictor(cycle)
    hi = health_index(probs)
    assert 0.0 <= hi <= 1.0


@pytest.mark.heavy
def test_real_data_predictor_heavy(raw_dir: Path, tmp_path: Path) -> None:
    """Heavy integration test training bundle and serving on real ZeMA data."""
    from hydro.data.loader import load_cycle_dataset

    data, targets = load_cycle_dataset(raw_dir)
    train_idx = np.arange(0, 2000, 10)
    train_onnx_bundle(data, targets, train_idx, tmp_path, iterations=20)
    bundle_classes = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))["classes"]
    predictor = OnnxCyclePredictor.from_dir(tmp_path)

    correct_valve = 0
    for cycle_idx in range(2100, 2106):
        cycle = {ch: data[ch][cycle_idx : cycle_idx + 1] for ch in data}
        probs = predictor(cycle)
        for tgt in TARGETS:
            labels, p = probs[tgt]
            np.testing.assert_array_equal(labels, bundle_classes[tgt])
            assert np.isclose(float(np.sum(p)), 1.0, atol=1e-4)
        hi = health_index(probs)
        assert 0.0 <= hi <= 1.0
        valve_labels, valve_p = probs["valve"]
        if valve_labels[np.argmax(valve_p)] == targets["valve"].iloc[cycle_idx]:
            correct_valve += 1
    assert correct_valve >= 4


def test_extract_cycle_features_missing_channel_raises(
    synthetic_dataset: tuple[dict[str, np.ndarray], pd.DataFrame],
) -> None:
    """Verify missing physical channel raises ValueError naming the missing channel."""
    data, _ = synthetic_dataset
    cycle = {ch: data[ch][0:1] for ch in data}
    cycle_missing = {k: v for k, v in cycle.items() if k != "VS1"}
    with pytest.raises(ValueError, match="VS1"):
        extract_cycle_features(cycle_missing, baseline=10.0)


def test_extract_cycle_features_extra_key_ignored(
    synthetic_dataset: tuple[dict[str, np.ndarray], pd.DataFrame],
) -> None:
    """Verify extra unknown keys are ignored and full cycle feature width is preserved."""
    data, _ = synthetic_dataset
    cycle = {ch: data[ch][0:1] for ch in data}
    cycle_extra = {**cycle, "EXTRA_UNKNOWN": np.zeros((1, 100), dtype=np.float32)}
    features_full = extract_cycle_features(cycle, baseline=10.0)
    features_extra = extract_cycle_features(cycle_extra, baseline=10.0)
    assert features_full.shape == features_extra.shape
    assert features_full.shape[1] == 161
    np.testing.assert_allclose(features_full, features_extra)
