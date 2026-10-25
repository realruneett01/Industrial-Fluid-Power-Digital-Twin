"""Inference-only ONNX helpers: need numpy and onnxruntime, not torch or catboost."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def inference_session(onnx_path: Path | str, intra_op_threads: int = 0) -> ort.InferenceSession:
    """CPU session with intra-op spinning disabled.

    Spinning worker threads keep cores busy after each run and starve the numpy/scipy
    feature extraction that runs between inferences (measured p95 156 ms -> 19 ms).
    ``intra_op_threads`` of 0 lets onnxruntime choose.
    """
    if intra_op_threads < 0:
        raise ValueError(f"intra_op_threads must be >= 0, got {intra_op_threads}")
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.intra_op_num_threads = intra_op_threads
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def _convert_ort_output(output: Any) -> np.ndarray:
    """Convert raw ONNX Runtime output (including map sequences) to ndarray."""
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
        return np.array(
            [[row[k] for k in sorted(row.keys())] for row in output],
            dtype=np.float32,
        )
    return np.asarray(output)
