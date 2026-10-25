"""End-to-end self-checks that must pass before a deployment is allowed to start.

Covers the model bundle (loads, predicts, outputs are valid probabilities), the latency SLA
on *this* machine, and an attestation that the OPC-UA boundary really refuses writes.
"""

from __future__ import annotations

import asyncio
import json
import socket
import tempfile
import time
from pathlib import Path

import numpy as np

from hydro.deploy.results import CheckResult, Status
from hydro.deploy.synthetic import synthetic_cycle
from hydro.online.predictor import TARGETS

BUNDLE_FILES = ("bundle.json", *(f"catboost_{t}.onnx" for t in TARGETS))
PROB_ATOL = 1e-3


def check_bundle_files(models_dir: Path) -> CheckResult:
    missing = [f for f in BUNDLE_FILES if not (models_dir / f).is_file()]
    if missing:
        return CheckResult(
            "bundle-files",
            Status.FAIL,
            f"missing in {models_dir}: {', '.join(missing)}",
            "build the bundle on a training machine: python -m hydro bundle --out models/online",
        )
    try:
        meta = json.loads((models_dir / "bundle.json").read_text(encoding="utf-8"))
        baseline = float(meta["valve_baseline_s"])
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError(f"valve_baseline_s={baseline}")
        for t in TARGETS:
            if not meta["classes"][t]:
                raise ValueError(f"no classes for {t}")
    except (OSError, ValueError, KeyError, TypeError) as err:
        return CheckResult(
            "bundle-files", Status.FAIL, f"bundle.json invalid: {err}", "rebuild the bundle"
        )
    return CheckResult("bundle-files", Status.PASS, f"{len(BUNDLE_FILES)} files, metadata valid")


def check_inference(
    models_dir: Path, runs: int = 20, sla_ms: float = 150.0, threads: int = 0
) -> list[CheckResult]:
    """Load the bundle, predict on synthetic cycles, validate outputs and measure latency."""
    from hydro.models.runtime import inference_session
    from hydro.online.advisor_client import health_index
    from hydro.online.predictor import OnnxCyclePredictor

    try:
        meta = json.loads((models_dir / "bundle.json").read_text(encoding="utf-8"))
        predictor = OnnxCyclePredictor(
            float(meta["valve_baseline_s"]),
            {t: np.asarray(meta["classes"][t]) for t in TARGETS},
            {t: inference_session(models_dir / f"catboost_{t}.onnx", threads) for t in TARGETS},
        )
        out = predictor(synthetic_cycle(0))
    except Exception as err:  # noqa: BLE001  # report any load/predict failure as a gate failure
        return [CheckResult("inference", Status.FAIL, f"{type(err).__name__}: {err}")]
    results = []
    bad = []
    for t in TARGETS:
        classes, probs = out[t]
        if (
            len(classes) != len(probs)
            or not np.isfinite(probs).all()
            or (probs < 0).any()
            or abs(float(probs.sum()) - 1.0) > PROB_ATOL
        ):
            bad.append(t)
    if bad:
        return [CheckResult("inference", Status.FAIL, f"invalid probabilities for {bad}")]
    hi = health_index(out)
    results.append(CheckResult("inference", Status.PASS, f"4 targets valid, HI={hi:.3f}"))
    predictor.reset()
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        predictor(synthetic_cycle(i))
        times.append((time.perf_counter() - t0) * 1000.0)
    p95 = float(np.percentile(times[2:] if runs > 4 else times, 95))
    status = Status.PASS if p95 <= sla_ms else Status.FAIL
    results.append(
        CheckResult(
            "latency-sla",
            status,
            f"p95 {p95:.1f} ms (limit {sla_ms:.0f} ms, {runs} cycles)",
            "" if status == Status.PASS else "machine too slow or overloaded for the 150 ms SLA",
            {"p95_ms": p95},
        )
    )
    return results


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _attest_read_only() -> CheckResult:
    """Start a throwaway loopback server, authenticate, and try to write a sensor node."""
    from asyncua import Client, ua

    from hydro.online.replay_server import NAMESPACE_URI, HydroReplayServer, sensor_node_id

    data = {"PS1": np.zeros((1, 6000))}
    endpoint = f"opc.tcp://127.0.0.1:{_free_port()}/hydro/"
    server = HydroReplayServer(data, endpoint, {"attest": "attest-pw"}, cert_dir=None)
    await server.start()
    try:
        client = Client(endpoint, timeout=5)
        client.set_user("attest")
        client.set_password("attest-pw")
        async with client:
            idx = await client.get_namespace_index(NAMESPACE_URI)
            node = client.get_node(ua.NodeId(sensor_node_id("PS1"), idx))
            try:
                await node.write_value(ua.Variant([1.0], ua.VariantType.Double))
            except (ua.UaStatusCodeError, ua.UaError):
                return CheckResult("opcua-read-only", Status.PASS, "client write was rejected")
            return CheckResult(
                "opcua-read-only",
                Status.FAIL,
                "a client write to a sensor node SUCCEEDED",
                "do not deploy: the read-only boundary is broken",
            )
    finally:
        await server.stop()


def check_read_only_boundary() -> CheckResult:
    try:
        return asyncio.run(_attest_read_only())
    except Exception as err:  # noqa: BLE001
        return CheckResult("opcua-read-only", Status.FAIL, f"attestation failed: {err}")


def toolchain_bundle(out_dir: Path, iterations: int = 5) -> Path:
    """Train a tiny bundle on synthetic data to prove the full stack works (needs `[train]`).

    The result is a smoke-test artifact only; it must never be used for diagnosis.
    """
    import pandas as pd

    from hydro.online.predictor import train_onnx_bundle

    n = 12
    cycles = [synthetic_cycle(i) for i in range(n)]
    data = {k: np.concatenate([c[k] for c in cycles]) for k in cycles[0]}
    cooler, valve = [3, 20, 100] * 4, [73, 80, 90, 100] * 3
    pump, acc = [0, 1, 2] * 4, [90, 100, 115, 130] * 3
    targets = pd.DataFrame(
        {"cooler": cooler, "valve": valve, "pump": pump, "accumulator": acc, "stable_flag": 0}
    )
    return train_onnx_bundle(data, targets, np.arange(n), out_dir, iterations=iterations)


def check_toolchain() -> CheckResult:
    """Train, export and run a synthetic bundle in a temp dir."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = toolchain_bundle(Path(tmp) / "bundle")
            res = check_inference(out, runs=3, sla_ms=1e9)
    except Exception as err:  # noqa: BLE001
        return CheckResult("toolchain", Status.FAIL, f"{type(err).__name__}: {err}")
    ok = all(r.status == Status.PASS for r in res)
    return CheckResult("toolchain", Status.PASS if ok else Status.FAIL, "train/export/infer ok")
