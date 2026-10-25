"""Online cycle predictor with rolling history feature extraction and ONNX inference."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hydro.data.schema import PHYSICAL_CHANNELS
from hydro.features import build_physics_features
from hydro.features.physics import fit_valve_baseline
from hydro.features.stats import extract_statistical_features
from hydro.models.runtime import _convert_ort_output, inference_session

if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnxruntime as ort
    from sklearn.pipeline import Pipeline

TARGETS: tuple[str, ...] = ("cooler", "valve", "pump", "accumulator")


def extract_cycle_features(cycles: dict[str, np.ndarray], baseline: float) -> np.ndarray:
    """Extract concatenated statistical and physics features for cycles as float32.

    Parameters
    ----------
    cycles : dict[str, np.ndarray]
        Sensor cycle arrays. Values are 1D (samples,) or 2D (cycles, samples).
    baseline : float
        Valve inflection baseline time in seconds [s].

    Returns
    -------
    np.ndarray
        Array of shape (n_cycles, n_features) with dtype float32.
    """
    missing = [k for k in PHYSICAL_CHANNELS if k not in cycles]
    if missing:
        raise ValueError(f"Missing physical channel(s): {', '.join(missing)}")
    c2d = {
        k: np.nan_to_num(
            np.atleast_2d(cycles[k].astype(np.float64) if k == "PS1" else cycles[k]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for k in PHYSICAL_CHANNELS
    }
    stat_df = extract_statistical_features(c2d)
    phys_df = build_physics_features(c2d, baseline)
    combined = pd.concat([stat_df, phys_df], axis=1).to_numpy(dtype=np.float32)
    return np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_export_catboost(
    clf: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
) -> list[int]:
    """Fit CatBoost classifier, export to ONNX, and return ordered class labels."""
    from hydro.models.export import export_catboost_to_onnx

    clf.fit(x, y)
    export_catboost_to_onnx(clf, x.shape[1], out_path)
    return [int(c) for c in clf.classes_]


def train_onnx_bundle(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    train_idx: np.ndarray,
    out_dir: Path | str,
    iterations: int | None = None,
) -> Path:
    """Fit valve baseline and CatBoost models on train_idx, exporting bundle.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Sensor channel arrays.
    targets : pd.DataFrame
        Target label DataFrame.
    train_idx : np.ndarray
        Row indices of the training fold.
    out_dir : Path | str
        Target directory for ONNX models and bundle.json.
    iterations : int | None, optional
        Number of gradient boosting iterations.

    Returns
    -------
    Path
        Output directory path containing exported artifacts.
    """
    from hydro.models.classical import make_classifier

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    train_data = {k: v[train_idx] for k, v in data.items()}
    valve_labels = np.asarray(targets["valve"])[train_idx]
    valve_baseline = fit_valve_baseline(train_data["PS1"], valve_labels)
    x_train = extract_cycle_features(train_data, valve_baseline)

    classes_dict: dict[str, list[int]] = {}
    for tgt in TARGETS:
        clf = make_classifier("catboost")
        if iterations is not None:
            clf.set_params(classifier__iterations=iterations)
        out_path = target_dir / f"catboost_{tgt}.onnx"
        y_train = np.asarray(targets[tgt])[train_idx]
        classes_dict[tgt] = _fit_export_catboost(clf, x_train, y_train, out_path)

    bundle_data: dict[str, Any] = {
        "valve_baseline_s": float(valve_baseline),
        "classes": classes_dict,
        "labels": classes_dict,
        **classes_dict,
    }
    with (target_dir / "bundle.json").open("w", encoding="utf-8") as f:
        json.dump(bundle_data, f, indent=2)
    return target_dir


def _run_ort_session(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    """Run single ONNX session and return 1D probability array."""
    in_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    req_outs = ["probabilities"] if "probabilities" in out_names else None
    raw = session.run(req_outs, {in_name: x})
    target_out = raw[0] if req_outs else raw[-1]
    return np.asarray(_convert_ort_output(target_out), dtype=np.float32).ravel()


class CycleHistory:
    """Rolling buffer of cycle arrays for feature extraction."""

    def __init__(self, maxlen: int = 5) -> None:
        """Initialize rolling cycle buffer with maximum capacity."""
        self.maxlen: int = maxlen
        self._buffer: deque[dict[str, np.ndarray]] = deque(maxlen=maxlen)

    def append(self, cycle: dict[str, np.ndarray]) -> None:
        """Append 2D-expanded cycle to rolling history buffer."""
        c2d = {k: np.atleast_2d(v) for k, v in cycle.items()}
        self._buffer.append(c2d)

    def stack(self) -> dict[str, np.ndarray]:
        """Stack buffered cycle arrays along cycle axis."""
        if not self._buffer:
            return {}
        first = self._buffer[0]
        return {ch: np.concatenate([c[ch] for c in self._buffer], axis=0) for ch in first}

    def reset(self) -> None:
        """Clear rolling cycle buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return number of cycles currently in buffer."""
        return len(self._buffer)


class OnnxCyclePredictor:
    """Online streaming cycle predictor using exported ONNX models and history buffer."""

    from_dir: Any = None

    def __init__(
        self,
        valve_baseline_s: float,
        classes: dict[str, Sequence[int] | np.ndarray] | None = None,
        sessions: dict[str, ort.InferenceSession] | None = None,
        history_len: int = 5,
    ) -> None:
        """Initialize predictor with baseline, class labels, and ONNX sessions."""
        self.valve_baseline_s: float = float(valve_baseline_s)
        self.classes: dict[str, np.ndarray] = {t: np.asarray(c) for t, c in (classes or {}).items()}
        self.sessions: dict[str, ort.InferenceSession] = sessions or {}
        self.history_len: int = history_len
        self.history: CycleHistory = CycleHistory(maxlen=history_len)
        self._history: CycleHistory = self.history
        self.last_features: np.ndarray | None = None

    def reset(self) -> None:
        """Clear rolling cycle history."""
        self.history.reset()
        self.last_features = None

    def compute_features(self, cycle: dict[str, np.ndarray]) -> np.ndarray:
        """Append cycle to history deque and extract 1-row feature vector."""
        self.history.append(cycle)
        stacked = self.history.stack()
        features = extract_cycle_features(stacked, self.valve_baseline_s)
        self.last_features = features[-1:, :]
        return self.last_features

    def __call__(self, cycle: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Run feature extraction and inference for a single cycle.

        Parameters
        ----------
        cycle : dict[str, np.ndarray]
            Sensor channels for one 60 s cycle.

        Returns
        -------
        dict[str, tuple[np.ndarray, np.ndarray]]
            Mapping of target name to (class_labels, probabilities) 1D arrays.
        """
        x = self.compute_features(cycle)
        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for tgt, session in self.sessions.items():
            probs = _run_ort_session(session, x)
            classes = self.classes.get(tgt, np.arange(len(probs)))
            results[tgt] = (classes, probs)
        return results


def _from_dir(
    cls: type[OnnxCyclePredictor], path: Path | str, intra_op_threads: int = 0
) -> OnnxCyclePredictor:
    """Instantiate predictor from directory containing bundle.json and ONNX models."""
    bundle_dir = Path(path)
    bundle_file = bundle_dir / "bundle.json"
    if not bundle_file.exists():
        raise FileNotFoundError(f"Bundle metadata not found: {bundle_file}")
    with bundle_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    baseline = float(meta["valve_baseline_s"])
    classes_raw = (
        meta.get("classes") or meta.get("labels") or {t: meta[t] for t in TARGETS if t in meta}
    )
    classes = {t: np.asarray(classes_raw[t]) for t in TARGETS}
    sessions = {
        t: inference_session(bundle_dir / f"catboost_{t}.onnx", intra_op_threads) for t in TARGETS
    }
    return cls(valve_baseline_s=baseline, classes=classes, sessions=sessions)


OnnxCyclePredictor.from_dir = classmethod(_from_dir)
