"""SIMULATION ARTIFACT - not validated against ZeMA data.

Unit tests for hydraulic cylinder extension and cold-start simulation.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from hydro.sim.cold_start_sim import acceleration_limit, simulate_cold_start
from hydro.sim.valve_cylinder import CylinderParams, simulate_extension


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        ("area_m2", 2e-3),
        ("volume_m3", 1e-3),
        ("bulk_modulus_pa", 1.4e9),
        ("supply_pa", 1.6e7),
        ("cd_orifice", 0.62),
        ("valve_area_m2", 2e-5),
        ("load_n", 1e4),
        ("mass_kg", 50.0),
    ],
)
def test_cylinder_params_defaults(attr: str, expected: float) -> None:
    """CylinderParams must have correct defaults."""
    params = CylinderParams()
    assert getattr(params, attr) == pytest.approx(expected)


def test_cylinder_params_immutability() -> None:
    """CylinderParams must be frozen."""
    params = CylinderParams()
    with pytest.raises(FrozenInstanceError):
        params.area_m2 = 3e-3  # type: ignore[misc]


def test_simulate_extension_columns_and_monotonic_pressure() -> None:
    """Output DataFrame has expected columns and pressure rises monotonically before motion."""
    params = CylinderParams()
    df = simulate_extension(params, valve_opening=1.0, t_end_s=1.0, dt_s=1e-3)
    expected_cols = ["t_s", "x_m", "v_m_s", "p_pa"]
    assert list(df.columns) == expected_cols
    assert len(df) == 1001

    motion_mask = df["v_m_s"] > 0.0
    assert motion_mask.any()
    first_motion_idx = int(motion_mask.idxmax())
    assert first_motion_idx > 0

    p_before = df.loc[:first_motion_idx, "p_pa"].to_numpy()
    dp = np.diff(p_before)
    assert np.all(dp > 0.0)


def test_simulate_extension_velocity_smooth_when_force_below_load() -> None:
    """Velocity starts at 0, never negative, and decelerates smoothly when force_net < 0."""
    params = CylinderParams()
    dt_s = 1e-3
    df = simulate_extension(params, valve_opening=0.8, t_end_s=1.5, dt_s=dt_s)
    v = df["v_m_s"].to_numpy()
    p = df["p_pa"].to_numpy()

    assert v[0] == 0.0
    assert np.all(v >= 0.0)

    motion_mask = v > 0.0
    assert motion_mask.any()
    first_motion_idx = int(np.argmax(motion_mask))

    force_net = p[1:] * params.area_m2 - params.load_n
    a = force_net / params.mass_kg
    step_dv = np.abs(v[1:] - v[:-1])

    decel_mask = (np.arange(len(p) - 1) >= first_motion_idx) & (force_net < 0.0)
    assert decel_mask.any()
    max_allowed = np.abs(a[decel_mask]) * dt_s + 1e-6
    assert np.all(step_dv[decel_mask] <= max_allowed)


def test_simulate_extension_valve_opening_stroke_timing() -> None:
    """Larger valve opening must reach a given stroke sooner."""
    params = CylinderParams()
    df_half = simulate_extension(params, valve_opening=0.5, t_end_s=2.0, dt_s=1e-3)
    df_full = simulate_extension(params, valve_opening=1.0, t_end_s=2.0, dt_s=1e-3)

    target_stroke_m = 0.5
    t_half = df_half.loc[df_half["x_m"] >= target_stroke_m, "t_s"].iloc[0]
    t_full = df_full.loc[df_full["x_m"] >= target_stroke_m, "t_s"].iloc[0]
    assert t_full < t_half


def test_simulate_extension_invalid_inputs_raise() -> None:
    """Invalid opening, duration, or timestep must raise ValueError."""
    params = CylinderParams()
    with pytest.raises(ValueError, match="valve_opening"):
        simulate_extension(params, valve_opening=-0.1)
    with pytest.raises(ValueError, match="t_end_s"):
        simulate_extension(params, valve_opening=0.5, t_end_s=0.0)
    with pytest.raises(ValueError, match="dt_s"):
        simulate_extension(params, valve_opening=0.5, dt_s=-1e-3)


def test_acceleration_limit_temperatures() -> None:
    """At 40 C accel equals nominal; at 15 C it is lower and >= 0.1*nominal."""
    nominal = 1.2
    limit_40 = acceleration_limit(40.0, nominal_m_s2=nominal)
    assert limit_40 == pytest.approx(nominal, rel=1e-3)

    limit_15 = acceleration_limit(15.0, nominal_m_s2=nominal)
    assert limit_15 < limit_40
    assert limit_15 >= 0.1 * nominal


def test_acceleration_limit_invalid_nominal_raises() -> None:
    """Non-positive nominal acceleration must raise ValueError."""
    with pytest.raises(ValueError, match="nominal_m_s2"):
        acceleration_limit(40.0, nominal_m_s2=0.0)


def test_simulate_cold_start_frame_and_values() -> None:
    """simulate_cold_start returns DataFrame with valid columns and values."""
    temps = np.array([15.0, 25.0, 40.0, 60.0])
    df = simulate_cold_start(temps)
    expected_cols = ["temp_c", "nu_cst", "accel_limit_m_s2"]
    assert list(df.columns) == expected_cols
    assert len(df) == 4

    assert df.loc[df["temp_c"] == 40.0, "accel_limit_m_s2"].iloc[0] == pytest.approx(1.0)
    accel_15 = df.loc[df["temp_c"] == 15.0, "accel_limit_m_s2"].iloc[0]
    assert 0.1 <= accel_15 < 1.0


def test_simulate_cold_start_non_1d_raises() -> None:
    """Non-1D temperature array must raise ValueError."""
    with pytest.raises(ValueError, match="temps_c must be a 1D array"):
        simulate_cold_start(np.ones((2, 2)))
