"""Sensor calibration and acceptance: do the attached sensors deliver usable cycles?

Calibration here is *acceptance and baselining*, not an adjustment of the trained models.
Each channel of each captured cycle is checked for sample count, finite values, plausible
physical range, dead (flat) signal and noise level; the surviving statistics are stored as
the site baseline so later drift can be measured against it. Plausibility limits are
deliberately wide sensor-span limits, not tight data-derived bands.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.deploy.results import CheckResult, Status

# (low, high) physical limits per unit. A value outside is a wiring/scaling fault.
LIMITS_BY_UNIT: dict[str, tuple[float, float]] = {
    "bar": (-1.0, 700.0),
    "W": (0.0, 20000.0),
    "l/min": (-5.0, 200.0),
    "degC": (-20.0, 150.0),
    "mm/s": (0.0, 50.0),
}
MAX_BAD_FRACTION = 0.01  # tolerated share of non-finite or out-of-range samples per channel
SCHEMA_VERSION = 1

CycleGrabber = Callable[[int], list[dict[str, np.ndarray]]]


@dataclass(frozen=True)
class ChannelStats:
    mean: float
    std: float
    minimum: float
    maximum: float
    noise: float  # robust sample-to-sample noise estimate

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "noise": self.noise,
        }


def _noise_estimate(x: np.ndarray) -> float:
    """Noise sigma from the median absolute first difference (robust to slow trends)."""
    if x.size < 3:
        return 0.0
    d = np.diff(x)
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2.0))


def check_channel(name: str, samples: np.ndarray) -> tuple[Status, str, ChannelStats | None]:
    """Judge one channel of one cycle. Returns (status, reason, stats or None on FAIL)."""
    schema = CHANNEL_REGISTRY[name]
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size != schema.n_samples:
        return Status.FAIL, f"{x.size} samples, expected {schema.n_samples}", None
    finite = np.isfinite(x)
    bad_frac = 1.0 - float(finite.mean())
    if bad_frac > MAX_BAD_FRACTION:
        return Status.FAIL, f"{bad_frac:.1%} NaN/inf samples", None
    y = x[finite]
    lo, hi = LIMITS_BY_UNIT[schema.unit]
    out_frac = float(np.mean((y < lo) | (y > hi)))
    if out_frac > MAX_BAD_FRACTION:
        return Status.FAIL, f"{out_frac:.1%} samples outside {lo}..{hi} {schema.unit}", None
    stats = ChannelStats(
        float(y.mean()), float(y.std()), float(y.min()), float(y.max()), _noise_estimate(y)
    )
    if np.ptp(y) == 0.0:
        return Status.WARN, "signal is perfectly flat (stuck or disconnected sensor?)", stats
    if bad_frac > 0 or out_frac > 0:
        return Status.WARN, "a few NaN or out-of-range samples", stats
    return Status.PASS, "ok", stats


def _pooled(per_cycle: list[ChannelStats]) -> dict[str, float]:
    means = np.array([s.mean for s in per_cycle])
    return {
        "mean": float(means.mean()),
        "std": float(np.mean([s.std for s in per_cycle])),
        "min": float(min(s.minimum for s in per_cycle)),
        "max": float(max(s.maximum for s in per_cycle)),
        "noise": float(np.mean([s.noise for s in per_cycle])),
        "mean_spread": float(means.std()),
        "cycles": len(per_cycle),
    }


def calibrate_cycles(
    cycles: Sequence[dict[str, np.ndarray]], strict: bool = False
) -> tuple[list[CheckResult], dict[str, Any]]:
    """Check every channel of every cycle and build the site baseline.

    ``strict`` promotes WARN to FAIL. A channel's final status is the worst across cycles.
    """
    results: list[CheckResult] = []
    baseline: dict[str, Any] = {}
    if not cycles:
        return [CheckResult("calibration", Status.FAIL, "no complete cycle captured")], baseline
    for name in PHYSICAL_CHANNELS:
        stats: list[ChannelStats] = []
        status, reasons = Status.PASS, []
        for i, cycle in enumerate(cycles):
            if name not in cycle:
                status, reasons = Status.FAIL, [f"cycle {i}: channel missing"]
                break
            s, why, st = check_channel(name, cycle[name])
            if st is not None:
                stats.append(st)
            if s != Status.PASS:
                reasons.append(f"cycle {i}: {why}")
            status = max(status, s)
        if strict and status == Status.WARN:
            status = Status.FAIL
        unit = CHANNEL_REGISTRY[name].unit
        if stats and status != Status.FAIL:
            baseline[name] = {"unit": unit, **_pooled(stats)}
        detail = f"{len(cycles)} cycle(s)" if status == Status.PASS else "; ".join(reasons[:2])
        hint = "check wiring, scaling and sensor power" if status != Status.PASS else ""
        results.append(CheckResult(f"sensor:{name}", status, detail, hint))
    return results, baseline


def build_calibration_record(
    baseline: dict[str, Any], source: str, results: list[CheckResult]
) -> dict[str, Any]:
    """Self-describing calibration file content with a content digest."""
    body = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "passed": all(r.status != Status.FAIL for r in results),
        "channels": baseline,
    }
    canonical = json.dumps(body["channels"], sort_keys=True, separators=(",", ":"))
    body["digest"] = hashlib.sha256(canonical.encode()).hexdigest()
    return body


def save_calibration(record: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic: a pulled pendrive never leaves half a file
    return path


def load_calibration(path: Path) -> dict[str, Any]:
    """Read and validate a calibration file; raises ValueError when it is unusable."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"cannot read calibration {path}: {err}") from err
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported calibration schema in {path}")
    channels = record.get("channels")
    if not isinstance(channels, dict):
        raise ValueError(f"calibration {path} has no channel table")
    canonical = json.dumps(channels, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != record.get("digest"):
        raise ValueError(f"calibration {path} failed its digest check (edited or corrupted)")
    return record


def drift_report(
    baseline: dict[str, Any], cycle: dict[str, np.ndarray], z_limit: float = 6.0
) -> list[CheckResult]:
    """Compare a fresh cycle to the stored baseline channel by channel.

    A channel drifts when its cycle mean leaves the baseline mean by more than ``z_limit``
    times the larger of the baseline within-cycle std and the between-cycle spread.
    """
    out: list[CheckResult] = []
    for name, ref in baseline.items():
        if name not in cycle:
            out.append(CheckResult(f"drift:{name}", Status.FAIL, "channel missing"))
            continue
        mean = float(np.nanmean(np.asarray(cycle[name], dtype=np.float64)))
        scale = max(ref["std"], ref.get("mean_spread", 0.0), 1e-9)
        z = abs(mean - ref["mean"]) / scale
        status = Status.PASS if z <= z_limit else Status.WARN
        out.append(
            CheckResult(
                f"drift:{name}",
                status,
                f"mean {mean:.4g} vs baseline {ref['mean']:.4g} (z={z:.1f})",
                "recalibrate if the plant state did not change" if status != Status.PASS else "",
            )
        )
    return out
