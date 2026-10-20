"""Inference latency profiling for the hydraulic condition monitoring digital twin."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402

from hydro.data.loader import load_cycle_dataset  # noqa: E402
from hydro.eval.metrics import latency_percentiles  # noqa: E402
from hydro.eval.report import write_report  # noqa: E402
from hydro.models.export import (  # noqa: E402
    _convert_ort_output,
    inference_session,
)
from hydro.online.cmms import build_work_order, dispatch_cmms_work_order  # noqa: E402
from hydro.online.predictor import (  # noqa: E402
    TARGETS,
    extract_cycle_features,
    train_onnx_bundle,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def train_models(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    train_idx: np.ndarray,
    iterations: int | None = None,
) -> tuple[float, dict[str, Path]]:
    """Fit valve baseline and CatBoost models on train_idx, exporting each to ONNX.

    Units: baseline inflection time in seconds [s]. Returns baseline and ONNX paths.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="hydro_onnx_"))
    train_onnx_bundle(data, targets, train_idx, out_dir, iterations=iterations)
    with (out_dir / "bundle.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    baseline = float(meta["valve_baseline_s"])
    onnx_paths = {tgt: out_dir / f"catboost_{tgt}.onnx" for tgt in TARGETS}
    return baseline, onnx_paths


def infer_cycle(
    cycle: dict[str, np.ndarray],
    baseline: float,
    sessions: dict[str, ort.InferenceSession],
) -> dict[str, np.ndarray]:
    """Extract features for one cycle and return class probabilities per target.

    Input cycle arrays must be shaped (1, n). Returns dict mapping target to probabilities.
    """
    x = extract_cycle_features(cycle, baseline)
    probs: dict[str, np.ndarray] = {}
    for tgt, session in sessions.items():
        in_name = session.get_inputs()[0].name
        out_names = [o.name for o in session.get_outputs()]
        req_outs = ["probabilities"] if "probabilities" in out_names else None
        raw = session.run(req_outs, {in_name: x})
        target_out = raw[0] if req_outs else raw[-1]
        probs[tgt] = np.asarray(_convert_ort_output(target_out), dtype=np.float32)
    return probs


def profile(
    data: dict[str, np.ndarray],
    idx: Sequence[int] | np.ndarray,
    baseline: float,
    sessions: dict[str, ort.InferenceSession],
    warmup: int = 2,
) -> np.ndarray:
    """Measure per-cycle execution nanoseconds for feature extraction plus inference.

    Units: returned array contains execution durations in nanoseconds [ns].
    """
    if len(idx) > 0 and warmup > 0:
        warm_cycle = {k: v[idx[0] : idx[0] + 1] for k, v in data.items()}
        for _ in range(warmup):
            infer_cycle(warm_cycle, baseline, sessions)

    timings: list[int] = []
    for i in idx:
        cycle = {k: v[i : i + 1] for k, v in data.items()}
        t0 = time.perf_counter_ns()
        infer_cycle(cycle, baseline, sessions)
        t1 = time.perf_counter_ns()
        timings.append(t1 - t0)
    return np.asarray(timings, dtype=np.int64)


def gate(percentiles: dict[str, float], heavy: bool = False) -> int:
    """Evaluate latency percentiles against SLA gates.

    Returns 1 if SLA is breached, 0 if within SLA.
    Target p95 < 150 ms; heavy-check gate p99 < 500 ms.
    """
    if percentiles["p95_ms"] >= 150.0:
        return 1
    if heavy and percentiles["p99_ms"] >= 500.0:
        return 1
    return 0


class _MockCMMSHandler(BaseHTTPRequestHandler):
    """Local mock handler responding with HTTP 201 Created."""

    def do_POST(self) -> None:  # noqa: N802
        self.send_response(201)
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


def _run_cmms_heavy_check() -> None:
    """Start local mock HTTP server, assert CMMS dispatch returns 201."""
    server = HTTPServer(("127.0.0.1", 0), _MockCMMSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        payload = build_work_order("HYD-01", "cooler", 2, 0.45)
        status = dispatch_cmms_work_order(payload, base_url=url)
        if status != 201:
            raise AssertionError(f"CMMS heavy check failed with status {status}")
    finally:
        server.shutdown()


def _save_latency_report(
    pcts: dict[str, float],
    n_cycles: int,
    report_path: Path | str,
) -> None:
    """Write markdown summary of latency percentiles using write_report."""
    sections = {
        "Latency Percentiles": (
            f"- p50: {pcts['p50_ms']:.2f} ms\n"
            f"- p95: {pcts['p95_ms']:.2f} ms\n"
            f"- p99: {pcts['p99_ms']:.2f} ms\n"
            f"- Cycles profiled: {n_cycles}"
        ),
        "Methodology": (
            "Measured via time.perf_counter_ns for single-cycle feature extraction "
            "plus 4-target ONNX runtime inference on CPU."
        ),
    }
    write_report(report_path, "Inference Latency Profile", sections)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command line arguments for latency profiling."""
    parser = argparse.ArgumentParser(description="Profile inference latency.")
    parser.add_argument("--cycles", type=int, default=10, help="Number of cycles to profile.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "raw",
        help="Path to raw dataset.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_REPO_ROOT / "reports" / "latency.md",
        help="Path to output markdown report.",
    )
    parser.add_argument(
        "--heavy-check",
        action="store_true",
        default=False,
        help="Run CMMS mock dispatch check.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Profile inference latency on final cycles and evaluate SLA exit code.

    Returns 0 on passing latency SLA, 1 if SLA is breached.
    """
    args = _parse_args(argv)
    data, targets = load_cycle_dataset(args.data_dir)
    n_total = len(data["PS1"])
    if n_total <= args.cycles:
        raise ValueError(f"Total cycles {n_total} must exceed --cycles {args.cycles}")

    train_idx = np.arange(0, n_total - args.cycles)
    test_idx = np.arange(n_total - args.cycles, n_total)
    baseline, onnx_paths = train_models(data, targets, train_idx)
    sessions = {t: inference_session(p) for t, p in onnx_paths.items()}
    timings_ns = profile(data, test_idx, baseline, sessions)
    pcts = latency_percentiles(timings_ns)
    print(f"p50: {pcts['p50_ms']:.2f} ms")
    print(f"p95: {pcts['p95_ms']:.2f} ms")
    print(f"p99: {pcts['p99_ms']:.2f} ms")

    _save_latency_report(pcts, len(test_idx), args.report)
    if args.heavy_check:
        _run_cmms_heavy_check()
    return gate(pcts, args.heavy_check)


if __name__ == "__main__":
    sys.exit(main())
