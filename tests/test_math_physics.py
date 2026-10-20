"""Tests for mathematical physics formulas, derivatives, and signal dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from hydro.features.physics import (
    _upsample_flow,
    compute_lumped_capacitance,
    compute_pressure_derivative,
    compute_thermal_effectiveness,
    compute_valve_edge_dynamics,
    fit_valve_baseline,
)


@pytest.mark.parametrize(
    ("slope_bar_s", "flow_l_min"),
    [(1.6, 15.0), (2.0, 30.0), (3.0, 45.0), (5.0, 60.0)],
)
def test_lumped_capacitance_analytic_ramp(slope_bar_s: float, flow_l_min: float) -> None:
    """Capacitance on linear ramp matches analytic C = Q / (dp/dt)."""
    # Oracle: C = (flow_l_min * 1e-3 / 60) / (slope_bar_s * 1e5) [m^3/Pa]
    expected_c = (flow_l_min / 60000.0) / (slope_bar_s * 1.0e5)
    t = np.arange(6000) / 100.0
    p = 100.0 + slope_bar_s * t
    flow = np.full(600, flow_l_min)
    c_eff = compute_lumped_capacitance(p, flow)
    assert abs(c_eff[0] - expected_c) / expected_c < 1.0e-5


@pytest.mark.parametrize(
    ("slope_bar_s", "flow_l_min", "p_start", "expect_nan"),
    [
        (2.0, 0.0, 100.0, False),  # active ramp with zero flow -> C_eff = 0.0
        (0.0, 30.0, 50.0, True),  # flat pressure below threshold -> nan
        (1.0, 30.0, 100.0, True),  # slope 1.0 bar/s < min 1.5 bar/s -> nan
    ],
)
def test_lumped_capacitance_edge_values(
    slope_bar_s: float, flow_l_min: float, p_start: float, expect_nan: bool
) -> None:
    """Verify capacitance on zero flow (0.0) and inactive masks (nan)."""
    t = np.arange(6000) / 100.0
    p = p_start + slope_bar_s * t
    q = np.full(600, flow_l_min)
    c_eff = compute_lumped_capacitance(p, q)
    if expect_nan:
        assert np.isnan(c_eff[0])
    else:
        assert c_eff[0] == 0.0


def test_lumped_capacitance_invalid_inputs() -> None:
    """Mismatched cycle count or short signals raise ValueError."""
    with pytest.raises(ValueError, match="Cycle count mismatch"):
        compute_lumped_capacitance(np.ones((2, 6000)), np.ones((3, 600)))
    with pytest.raises(ValueError, match="at least 21"):
        compute_lumped_capacitance(np.ones(10), np.ones(10))


@pytest.mark.parametrize(
    ("a", "b"),
    [(2.0, 0.0), (1.0, 0.5), (-1.0, 0.3), (0.0, 1.2)],
)
def test_pressure_derivative_quadratic_analytic(a: float, b: float) -> None:
    """dp/dt of p(t)=a*t + b*t^2 matches analytic derivative 1e5*(a + 2*b*t)."""
    fs = 100.0
    t = np.arange(1000) / fs
    p_bar = a * t + b * (t**2)
    # Oracle: d(p_pa)/dt = 1e5 * (a + 2*b*t)
    expected_dp_dt = 1.0e5 * (a + 2.0 * b * t)
    dp_dt = compute_pressure_derivative(p_bar, fs_hz=fs)
    np.testing.assert_allclose(dp_dt[25:-25], expected_dp_dt[25:-25], rtol=1.0e-5)


def test_pressure_derivative_property_loop() -> None:
    """Seeded property test: random (a, b) derivatives match analytic formula."""
    rng = np.random.default_rng(42)
    fs = 100.0
    t = np.arange(1000) / fs
    for a, b in rng.uniform(-5.0, 5.0, size=(10, 2)):
        p_bar = a * t + b * (t**2)
        expected = 1.0e5 * (a + 2.0 * b * t)
        dp_dt = compute_pressure_derivative(p_bar, fs_hz=fs)
        np.testing.assert_allclose(dp_dt[25:-25], expected[25:-25], rtol=1.0e-4)


def test_pressure_derivative_edge_cases() -> None:
    """Invalid fs, short arrays, and constant pressures are handled properly."""
    with pytest.raises(ValueError, match="must be positive"):
        compute_pressure_derivative(np.ones(100), fs_hz=0.0)
    with pytest.raises(ValueError, match="at least 21"):
        compute_pressure_derivative(np.ones(20), fs_hz=100.0)
    const_dp = compute_pressure_derivative(np.full(100, 50.0), fs_hz=100.0)
    np.testing.assert_allclose(const_dp, 0.0, atol=1.0e-5)


@pytest.mark.parametrize("delay_samples", [0, 4, 10, 15, 25])
def test_valve_edge_delay_synthetic_step(delay_samples: int) -> None:
    """Step delay shifted by k samples is recovered within 1/fs resolution."""
    fs = 100.0
    t = np.arange(1000) / fs
    t0, tau = 2.0, 0.05
    delay_s = delay_samples / fs
    p_base = 50.0 + 50.0 * np.tanh((t - t0) / tau)
    p_delayed = 50.0 + 50.0 * np.tanh((t - (t0 + delay_s)) / tau)
    base_inflect = fit_valve_baseline(p_base[None, :], np.array([100]), fs_hz=fs)
    df = compute_valve_edge_dynamics(p_delayed[None, :], base_inflect, fs_hz=fs)
    # Oracle: recovered delay equals injected delay_s within 1 sample resolution
    assert abs(df["valve_edge_delay"].iloc[0] - delay_s) <= (1.0 / fs) + 1.0e-9


@pytest.mark.parametrize("ramp_s", [0.5, 1.0, 2.0, 3.0])
def test_valve_rise_time_linear_ramp(ramp_s: float) -> None:
    """Linear ramp rise time 10-90% matches analytic 0.80 * ramp_duration."""
    fs = 100.0
    t = np.arange(6000) / fs
    p = np.full(6000, 30.0)
    t1, t2 = 5.0, 5.0 + ramp_s
    mask = (t >= t1) & (t <= t2)
    p[mask] = 30.0 + (150.0 - 30.0) * (t[mask] - t1) / ramp_s
    p[t > t2] = 150.0
    df = compute_valve_edge_dynamics(p, 0.0, fs_hz=fs)
    # Oracle: 10% to 90% of linear ramp spans exactly (0.90 - 0.10) * ramp_s
    expected_rise = 0.80 * ramp_s
    assert abs(df["rise_time_10_90"].iloc[0] - expected_rise) <= (1.0 / fs) + 1.0e-6


@pytest.mark.parametrize(
    ("peak_bar", "expected_ov"),
    [(110.0, 0.10), (125.0, 0.25), (150.0, 0.50), (200.0, 1.00)],
)
def test_valve_overshoot_known_peaks(peak_bar: float, expected_ov: float) -> None:
    """Overshoot on 100 bar plateau matches analytic (peak / plateau) - 1.0."""
    p = np.full(1000, 100.0)
    p[500:505] = peak_bar
    df = compute_valve_edge_dynamics(p, 0.0, fs_hz=100.0)
    # Oracle: overshoot = max(p) / percentile(p, 95) - 1.0 = (peak / 100.0) - 1.0
    assert abs(df["overshoot_ratio"].iloc[0] - expected_ov) < 1.0e-6


def test_valve_edge_dynamics_flat_constant() -> None:
    """Constant positive pressure yields zero rise time and zero overshoot."""
    df_const = compute_valve_edge_dynamics(np.full(1000, 50.0), 0.0, fs_hz=100.0)
    assert df_const["rise_time_10_90"].iloc[0] == 0.0
    assert df_const["overshoot_ratio"].iloc[0] == 0.0


@pytest.mark.parametrize(
    ("t_in", "t_out", "expected_eff"),
    [
        (50.0, 35.0, 0.50),
        (40.0, 20.0, 1.00),
        (30.0, 35.0, 0.00),
        (40.0, 10.0, 1.00),
        (30.0, 30.0, 0.00),
    ],
)
def test_thermal_effectiveness_hand_computed(
    t_in: float, t_out: float, expected_eff: float
) -> None:
    """Thermal effectiveness matches hand-calculated (T_in - T_out)/(T_in - T_ref)."""
    # Oracle: clip((T_in - T_out) / max(T_in - 20.0, 1.0), 0.0, 1.0)
    eff = compute_thermal_effectiveness(
        np.array([t_in]), np.array([t_out]), window_size=1, t_ref_c=20.0
    )
    assert abs(eff[0] - expected_eff) < 1.0e-6


def test_thermal_effectiveness_causality() -> None:
    """Future cycle modifications do not affect prior rolling mean values."""
    rng = np.random.default_rng(42)
    t_in = rng.uniform(30.0, 60.0, 10)
    t_out = rng.uniform(20.0, 30.0, 10)
    res_orig = compute_thermal_effectiveness(t_in, t_out, window_size=5)
    t_in_mod = t_in.copy()
    t_out_mod = t_out.copy()
    t_in_mod[-3:] = 75.0
    t_out_mod[-3:] = 15.0
    res_mod = compute_thermal_effectiveness(t_in_mod, t_out_mod, window_size=5)
    # Causal trailing window: cycles 0..6 must remain strictly identical
    np.testing.assert_array_equal(res_orig[:-3], res_mod[:-3])


def test_thermal_effectiveness_edge_cases() -> None:
    """Invalid window_size and mismatched shapes raise ValueError."""
    with pytest.raises(ValueError, match="window_size"):
        compute_thermal_effectiveness(np.array([40.0]), np.array([30.0]), window_size=0)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_thermal_effectiveness(np.ones(5), np.ones(6))


def test_flow_upsampling_linear_ramp() -> None:
    """Linearly upsampled f(t)=t from 10 to 100 Hz matches target grid exactly."""
    t_src = np.arange(600) / 10.0
    t_tgt = np.arange(6000) / 100.0
    # Oracle: Linear interpolation of linear function f(t)=t is identity
    up_flow = _upsample_flow(t_src, 6000)
    np.testing.assert_allclose(up_flow, t_tgt, atol=1.0e-12)


@pytest.mark.parametrize("freq_hz", [0.05, 0.10, 0.20])
def test_flow_upsampling_sine(freq_hz: float) -> None:
    """Upsampled sin matches numpy linear interpolation oracle within bound."""
    t_src = np.arange(600) / 10.0
    t_tgt = np.arange(6000) / 100.0
    omega = 2.0 * np.pi * freq_hz
    flow_sin = np.sin(omega * t_src)
    up_flow = _upsample_flow(flow_sin, 6000)
    # Oracle 1: Independent np.interp on interior grid
    interior = t_tgt <= 59.9
    ref_interp = np.interp(t_tgt[interior], t_src, flow_sin)
    np.testing.assert_allclose(up_flow[interior], ref_interp, atol=1.0e-14)
    # Oracle 2: Error bounded by (h^2 * omega^2) / 8 where h = 0.1 s
    max_err = ((0.1**2) * (omega**2)) / 8.0 + 1.0e-6
    analytic_sin = np.sin(omega * t_tgt[interior])
    assert np.max(np.abs(up_flow[interior] - analytic_sin)) <= max_err


def test_flow_upsampling_edge_cases() -> None:
    """Same size returns identical array; constant input preserves value."""
    flow = np.arange(50, dtype=float)
    assert _upsample_flow(flow, 50) is flow
    const_up = _upsample_flow(np.full(600, 42.0), 6000)
    np.testing.assert_allclose(const_up, 42.0, atol=1.0e-12)
