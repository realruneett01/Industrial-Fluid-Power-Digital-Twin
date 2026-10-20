from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydro.cli import build_parser, main
from hydro.data.schema import CHANNEL_REGISTRY


def _synthetic_dataset() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 8
    data = {
        name: rng.normal(50.0, 2.0, (n, ch.n_samples)).astype(np.float32)
        for name, ch in CHANNEL_REGISTRY.items()
        if not ch.is_virtual
    }
    data["PS1"] = np.linspace(100.0, 200.0, 6000, dtype=np.float32)[None, :].repeat(n, axis=0)
    targets = pd.DataFrame(
        {
            "cooler": [3, 20, 100, 3, 20, 100, 3, 20],
            "valve": [100, 73, 80, 100, 73, 80, 100, 73],
            "pump": [0, 1, 2, 0, 1, 2, 0, 1],
            "accumulator": [90, 100, 115, 130, 90, 100, 115, 130],
            "stable_flag": [0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    return data, targets


def test_eval_defaults():
    args = build_parser().parse_args(["eval"])
    assert (args.split, args.mode, args.heavy_check) == ("blocked", "baseline-only", False)


def test_eval_temp_shift_full():
    args = build_parser().parse_args(
        ["eval", "--split", "temp-shift", "--mode", "full-comparison", "--heavy-check"]
    )
    assert (args.split, args.mode, args.heavy_check) == ("temp-shift", "full-comparison", True)


def test_unknown_split_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", "--split", "random"])


def test_command_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_bundle_defaults():
    args = build_parser().parse_args(["bundle"])
    assert (args.data_dir, args.out, args.holdout, args.iterations) == (
        "data/raw",
        "models/online",
        205,
        None,
    )


def test_cmd_bundle_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, targets = _synthetic_dataset()
    monkeypatch.setattr(
        "hydro.data.loader.load_cycle_dataset",
        lambda *args, **kwargs: (data, targets),
    )
    exit_code = main(["bundle", "--out", str(tmp_path), "--holdout", "2", "--iterations", "5"])
    assert exit_code == 0
    assert (tmp_path / "bundle.json").is_file()
    for tgt in ("cooler", "valve", "pump", "accumulator"):
        assert (tmp_path / f"catboost_{tgt}.onnx").is_file()


def test_cmd_bundle_negative_holdout_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_load(*args: object, **kwargs: object) -> None:
        pytest.fail("load_cycle_dataset must not be called when holdout is negative")

    monkeypatch.setattr("hydro.data.loader.load_cycle_dataset", _fail_load)
    with pytest.raises(ValueError, match=r"[Hh]oldout.*negative"):
        main(["bundle", "--holdout", "-1"])


def test_advisor_client_missing_models_dir():
    from hydro.online.advisor_client import main as advisor_main

    with pytest.raises(SystemExit):
        advisor_main([])
