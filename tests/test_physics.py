"""Tests for grey-box physics parameter extraction."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hydro.eval.report import generate_markdown_table
from hydro.features.physics import (
    _upsample_flow,
    accumulator_gas_compliance,
    compute_lumped_capacitance,
    compute_pressure_derivative,
    compute_thermal_effectiveness,
    compute_valve_edge_dynamics,
    fit_valve_baseline,
)


def test_pressure_derivative_known_ramp() -> None:
    """Pressure derivative for a constant ramp equals nominal Pa/s."""
    fs_hz = 100.0
    # 10 bar/s ramp over 1000 samples (10 s) -> 1.0e6 Pa/s
    t = np.arange(1000) / fs_hz
    p_bar = 50.0 + 10.0 * t
    dp_dt = compute_pressure_derivative(p_bar, fs_hz=fs_hz)

    # Ignore boundaries of Savitzky-Golay filter
    interior = dp_dt[25:-25]
    np.testing.assert_allclose(interior, 1.0e6, rtol=1e-3)


def test_valve_edge_delay_recovery() -> None:
    """Synthetic tanh edge delay is recovered within 1 sample (0.01 s) under noise."""
    fs_hz = 100.0
    t = np.arange(1000) / fs_hz
    t0 = 2.0
    tau = 0.05
    d_samples = 10
    d_seconds = d_samples / fs_hz

    step_base = 100.0 * 0.5 * (1.0 + np.tanh((t - t0) / tau))
    step_delayed = 100.0 * 0.5 * (1.0 + np.tanh((t - (t0 + d_seconds)) / tau))

    rng = np.random.default_rng(42)
    sig_power = float(np.mean(step_base**2))
    noise_power = sig_power / (10 ** (25 / 10))
    noise_std = np.sqrt(noise_power)

    noisy_base = step_base + rng.normal(0, noise_std, step_base.shape)
    noisy_delayed = step_delayed + rng.normal(0, noise_std, step_delayed.shape)

    base_inflect = fit_valve_baseline(noisy_base[None, :], np.array([100]), fs_hz=fs_hz)
    df = compute_valve_edge_dynamics(noisy_delayed[None, :], base_inflect, fs_hz=fs_hz)

    recovered_delay = df["valve_edge_delay"].iloc[0]
    assert abs(recovered_delay - d_seconds) <= (1.0 / fs_hz) + 1e-9


def test_lumped_capacitance_linear_ramp() -> None:
    """Capacitance recovered from a constant ramp equals Q/(dp/dt) within 2%."""
    # Ramp 100 to 200 bar over 60 s (6000 samples)
    p = np.linspace(100.0, 200.0, 6000)[None, :]
    # Flow 40 l/min (600 samples)
    flow = np.ones((1, 600)) * 40.0

    # Theoretical C = (40 * 1e-3 / 60) / (100 * 1e5 / 60) = 4.0e-9 m^3/Pa
    expected_c = (40.0 * 1e-3 / 60.0) / (100.0 * 1e5 / 60.0)
    c_eff = compute_lumped_capacitance(p, flow)

    assert abs(c_eff[0] - expected_c) / expected_c < 0.02


def test_lumped_capacitance_positive_check() -> None:
    """C_eff > 0 on linspace(100, 200, 6000) with flow 40."""
    c_eff = compute_lumped_capacitance(
        np.linspace(100.0, 200.0, 6000)[None, :],
        np.ones((1, 600)) * 40.0,
    )
    assert c_eff[0] > 0


def test_lumped_capacitance_flat_segment_no_warning() -> None:
    """Flat segment with dp/dt == 0 does not trigger RuntimeWarning divide by zero."""
    p = np.concatenate([np.full(3000, 150.0), np.linspace(150.0, 200.0, 3000)])[None, :]
    flow = np.ones((1, 600)) * 40.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c_eff = compute_lumped_capacitance(p, flow)
    assert c_eff[0] > 0


def test_accumulator_gas_compliance() -> None:
    """Accumulator compliance is 0 below p0 and increases with p0 at fixed p."""
    p_pa = 150.0e5
    p0_low = 50.0e5
    p0_high = 80.0e5

    c_zero = accumulator_gas_compliance(30.0e5, p0_low)
    assert c_zero == 0.0

    c_low = accumulator_gas_compliance(p_pa, p0_low)
    c_high = accumulator_gas_compliance(p_pa, p0_high)
    assert c_high > c_low > 0.0


def test_thermal_effectiveness_bounds() -> None:
    """Thermal effectiveness values remain within [0, 1]."""
    rng = np.random.default_rng(123)
    n_cycles = 10
    t_in = rng.uniform(15.0, 55.0, (n_cycles, 60))
    t_out = rng.uniform(15.0, 55.0, (n_cycles, 60))

    eff = compute_thermal_effectiveness(t_in, t_out, window_size=5)
    assert eff.shape == (n_cycles,)
    assert np.all(eff >= 0.0)
    assert np.all(eff <= 1.0)


def test_thermal_effectiveness_causality() -> None:
    """Changing the last 3 cycles leaves earlier outputs identical."""
    rng = np.random.default_rng(42)
    n_cycles = 10
    t_in = rng.uniform(25.0, 50.0, (n_cycles, 60))
    t_out = t_in - rng.uniform(2.0, 8.0, (n_cycles, 60))

    res1 = compute_thermal_effectiveness(t_in, t_out, window_size=5)

    t_in_mod = t_in.copy()
    t_out_mod = t_out.copy()
    t_in_mod[-3:] = 60.0
    t_out_mod[-3:] = 20.0

    res2 = compute_thermal_effectiveness(t_in_mod, t_out_mod, window_size=5)
    np.testing.assert_array_equal(res1[:-3], res2[:-3])


def test_fit_valve_baseline_raises_no_healthy() -> None:
    """fit_valve_baseline raises ValueError when no cycle has valve == 100."""
    rng = np.random.default_rng(0)
    p = rng.standard_normal((5, 6000))
    valve_unhealthy = np.array([90, 80, 73, 80, 90])

    with pytest.raises(ValueError, match="healthy"):
        fit_valve_baseline(p, valve_unhealthy)


def test_valve_edge_dynamics_columns() -> None:
    """compute_valve_edge_dynamics returns DataFrame with expected columns."""
    rng = np.random.default_rng(0)
    p = rng.standard_normal((5, 6000))
    df = compute_valve_edge_dynamics(p, baseline_median_inflect=10.0)

    expected_cols = {"valve_edge_delay", "rise_time_10_90", "overshoot_ratio"}
    assert expected_cols.issubset(df.columns)
    assert len(df) == 5


@pytest.mark.heavy
def test_physics_on_real_data(raw_dir: Path) -> None:
    """Smoke test on real dataset channels if available."""
    ps1 = np.loadtxt(raw_dir / "PS1.txt", delimiter="\t", max_rows=5)
    fs1 = np.loadtxt(raw_dir / "FS1.txt", delimiter="\t", max_rows=5)
    c_eff = compute_lumped_capacitance(ps1, fs1)
    assert c_eff.shape == (5,)
    assert not np.all(np.isnan(c_eff))


def test_report_omits_range_index() -> None:
    """generate_markdown_table omits default RangeIndex but retains named index."""
    df_default = pd.DataFrame({"col_a": [1.0, 2.0], "col_b": [3, 4]})
    table_default = generate_markdown_table(df_default)
    first_line_default = table_default.splitlines()[0]
    assert first_line_default == "| col_a | col_b |"

    df_named = pd.DataFrame(
        {"col_a": [1.0, 2.0]},
        index=pd.Index(["row1", "row2"], name="target"),
    )
    table_named = generate_markdown_table(df_named)
    first_line_named = table_named.splitlines()[0]
    assert first_line_named == "| target | col_a |"


def test_valve_edge_dynamics_rise_time_standby() -> None:
    """Rise time 10-90% is measured from base standby pressure, not zero."""
    fs_hz = 100.0
    t = np.arange(6000) / fs_hz
    p = np.full(6000, 30.0)
    ramp_mask = (t >= 2.0) & (t <= 2.2)
    p[ramp_mask] = 30.0 + (150.0 - 30.0) * (t[ramp_mask] - 2.0) / 0.2
    p[(t > 2.2) & (t < 50.0)] = 150.0

    df = compute_valve_edge_dynamics(p, baseline_median_inflect=0.0, fs_hz=fs_hz)
    rise_time = df["rise_time_10_90"].iloc[0]
    assert abs(rise_time - 0.16) <= 0.03


def test_upsample_flow_linear_ramp_accuracy() -> None:
    """Upsampled flow of linear ramp matches physical time within 1e-9 for t <= 59.9 s."""
    n_source = 600
    n_target = 6000
    t_source = np.arange(n_source) / 10.0
    flow = t_source
    t_target = np.arange(n_target) / 100.0

    up_flow = _upsample_flow(flow, n_target)
    mask = t_target <= 59.9
    np.testing.assert_allclose(up_flow[mask], t_target[mask], atol=1e-9)
