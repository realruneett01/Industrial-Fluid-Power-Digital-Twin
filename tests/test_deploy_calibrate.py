from __future__ import annotations

import json

import numpy as np
import pytest

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS
from hydro.deploy import calibrate as cal
from hydro.deploy.results import Status
from hydro.deploy.synthetic import synthetic_cycle

CHANNELS = list(PHYSICAL_CHANNELS)


def _good(name: str, seed: int = 0) -> np.ndarray:
    return synthetic_cycle(seed)[name]


@pytest.mark.parametrize("name", CHANNELS)
@pytest.mark.parametrize("seed", range(8))
def test_clean_synthetic_channels_pass(name, seed):
    status, why, stats = cal.check_channel(name, _good(name, seed))
    assert status == Status.PASS, why
    assert stats is not None and stats.std > 0 and stats.minimum <= stats.mean <= stats.maximum


@pytest.mark.parametrize("name", CHANNELS)
@pytest.mark.parametrize("delta", [-1, 1, 7, 0])
def test_wrong_sample_count_fails(name, delta):
    n = CHANNEL_REGISTRY[name].n_samples
    x = np.ones(n + delta if delta else 0)
    status, why, stats = cal.check_channel(name, x)
    assert status == Status.FAIL and stats is None and "expected" in why


@pytest.mark.parametrize("name", CHANNELS)
@pytest.mark.parametrize("frac", [0.02, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_many_non_finite_samples_fail(name, frac, bad):
    x = _good(name).ravel().copy()
    x[: max(1, int(len(x) * frac) + 1)] = bad
    status, _, stats = cal.check_channel(name, x)
    assert status == Status.FAIL and stats is None


@pytest.mark.parametrize("name", CHANNELS)
def test_single_non_finite_sample_tolerated_only_above_100_samples(name):
    """1 NaN is under 1 % of a 100 Hz or 10 Hz cycle (warn) but 1.7 % of a 60-sample channel."""
    x = _good(name).ravel().copy()
    x[0] = np.nan
    status, _, stats = cal.check_channel(name, x)
    if CHANNEL_REGISTRY[name].n_samples >= 100:
        assert status == Status.WARN and stats is not None and np.isfinite(stats.mean)
    else:
        assert status == Status.FAIL and stats is None


@pytest.mark.parametrize("name", CHANNELS)
@pytest.mark.parametrize("side", ["above", "below"])
def test_wiring_or_scaling_fault_out_of_range_fails(name, side):
    lo, hi = cal.LIMITS_BY_UNIT[CHANNEL_REGISTRY[name].unit]
    x = _good(name).ravel().copy()
    x[:] = hi * 2 + 1.0 if side == "above" else lo - 100.0
    status, _, _ = cal.check_channel(name, x)
    assert status == Status.FAIL


@pytest.mark.parametrize("name", CHANNELS)
def test_flat_signal_warns_and_strict_fails(name):
    flat = np.full(CHANNEL_REGISTRY[name].n_samples, 5.0)
    status, why, stats = cal.check_channel(name, flat)
    assert status == Status.WARN and "flat" in why and stats is not None
    cycle = synthetic_cycle(1)
    cycle[name] = flat.reshape(1, -1)
    results, baseline = cal.calibrate_cycles([cycle])
    by_name = {r.name: r for r in results}
    assert by_name[f"sensor:{name}"].status == Status.WARN and name in baseline
    strict, strict_base = cal.calibrate_cycles([cycle], strict=True)
    assert {r.name: r for r in strict}[f"sensor:{name}"].status == Status.FAIL
    assert name not in strict_base


def test_calibrate_cycles_all_good_builds_complete_baseline():
    cycles = [synthetic_cycle(i) for i in range(3)]
    results, baseline = cal.calibrate_cycles(cycles)
    assert len(results) == len(CHANNELS) and all(r.status == Status.PASS for r in results)
    assert set(baseline) == set(CHANNELS)
    for name, ref in baseline.items():
        assert ref["cycles"] == 3 and ref["unit"] == CHANNEL_REGISTRY[name].unit
        assert ref["min"] <= ref["mean"] <= ref["max"] and ref["noise"] >= 0


def test_calibrate_no_cycles_fails():
    results, baseline = cal.calibrate_cycles([])
    assert results[0].status == Status.FAIL and baseline == {}


@pytest.mark.parametrize("name", CHANNELS)
def test_missing_channel_fails_only_that_channel(name):
    cycle = synthetic_cycle(2)
    del cycle[name]
    results, baseline = cal.calibrate_cycles([cycle])
    bad = [r.name for r in results if r.status == Status.FAIL]
    assert bad == [f"sensor:{name}"] and name not in baseline


def test_one_bad_cycle_fails_the_channel():
    good, bad = synthetic_cycle(0), synthetic_cycle(1)
    bad["TS1"] = np.full((1, 60), 900.0)
    results, _ = cal.calibrate_cycles([good, bad])
    assert {r.name: r.status for r in results}["sensor:TS1"] == Status.FAIL


@pytest.mark.parametrize("noise", [0.0, 0.05, 0.5, 5.0])
def test_noise_estimate_tracks_true_sigma(noise):
    rng = np.random.default_rng(0)
    x = 100 + np.linspace(0, 50, 6000) + noise * rng.standard_normal(6000)  # ramp must not matter
    est = cal._noise_estimate(x)
    assert est == pytest.approx(noise, rel=0.1, abs=1e-9) if noise else est < 0.02


@pytest.mark.parametrize("n", [0, 1, 2])
def test_noise_estimate_short_input(n):
    assert cal._noise_estimate(np.arange(n, dtype=float)) == 0.0


def test_calibration_record_roundtrip(tmp_path):
    results, baseline = cal.calibrate_cycles([synthetic_cycle(0)])
    rec = cal.build_calibration_record(baseline, "unit", results)
    assert rec["passed"] is True
    path = cal.save_calibration(rec, tmp_path / "sub" / "calibration.json")
    assert cal.load_calibration(path)["digest"] == rec["digest"]
    assert not list(path.parent.glob("*.tmp"))


def test_calibration_tampering_is_detected(tmp_path):
    results, baseline = cal.calibrate_cycles([synthetic_cycle(0)])
    path = cal.save_calibration(
        cal.build_calibration_record(baseline, "u", results), tmp_path / "c.json"
    )
    data = json.loads(path.read_text())
    data["channels"]["PS1"]["mean"] += 1.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        cal.load_calibration(path)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "nope",
        "[]",
        '{"schema_version": 99}',
        '{"schema_version": 1}',
        '{"schema_version": 1, "channels": 3}',
    ],
)
def test_load_calibration_rejects_garbage(tmp_path, content):
    p = tmp_path / "c.json"
    p.write_text(content)
    with pytest.raises(ValueError):
        cal.load_calibration(p)


def test_load_calibration_missing_file(tmp_path):
    with pytest.raises(ValueError):
        cal.load_calibration(tmp_path / "absent.json")


def test_record_marks_failed_calibration():
    results, baseline = cal.calibrate_cycles([])
    assert cal.build_calibration_record(baseline, "u", results)["passed"] is False


@pytest.mark.parametrize("seed", range(20))
def test_same_plant_state_shows_no_drift(seed):
    _, base = cal.calibrate_cycles([synthetic_cycle(i) for i in range(3)])
    out = cal.drift_report(base, synthetic_cycle(100 + seed))
    assert all(r.status == Status.PASS for r in out), [
        r.detail for r in out if r.status != Status.PASS
    ]


@pytest.mark.parametrize("name", CHANNELS)
@pytest.mark.parametrize("shift", [10.0, -10.0])
def test_sensor_offset_shows_up_as_drift(name, shift):
    _, base = cal.calibrate_cycles([synthetic_cycle(i) for i in range(3)])
    cycle = synthetic_cycle(9)
    cycle[name] = cycle[name] + shift * max(base[name]["std"], 1e-3) * 10
    out = {r.name: r for r in cal.drift_report(base, cycle)}
    assert out[f"drift:{name}"].status == Status.WARN


def test_drift_missing_channel_fails():
    _, base = cal.calibrate_cycles([synthetic_cycle(0)])
    cycle = synthetic_cycle(1)
    del cycle["PS2"]
    assert {r.name: r.status for r in cal.drift_report(base, cycle)}["drift:PS2"] == Status.FAIL
