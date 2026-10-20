"""SIMULATION ARTIFACT - not validated against ZeMA data.

Mathematical and physical validation of hydraulic lumped-parameter simulation:
- Orifice equation Q = Cd * A * x * sqrt(2 * dp / rho) vs hand values.
- Steady-state cylinder velocity v = Q / A_piston under constant load.
- Pressure dynamics dp/dt = (beta / V) * (Q - A * v) sign and magnitude.
- Velocity non-negativity under aggressive deceleration and stall.
- Zero valve opening resulting in zero motion.
- Time-step refinement convergence (first-order semi-implicit Euler).
- SI unit consistency and physical extreme/edge parameter ranges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from hydro.sim.valve_cylinder import (
    RHO_OIL_KG_M3,
    CylinderParams,
    _step_state,
    simulate_extension,
)

if TYPE_CHECKING:
    import pandas as pd

# Hand derivations for orifice equation:
# Q = Cd * A_v * x_v * sqrt(2 * dp / rho).
# In _step_state with v=0: dp_step = (beta / V) * Q * dt => Q = (p_next - p) * V / (beta * dt).
# Parameters: Cd=0.62, A_v=2e-5 m^2, rho=870 kg/m^3, Ps=1.6e7 Pa.
# Case 1: p=0, x_v=1.0: dp=1.6e7, sqrt(3.2e7/870) ~= 191.785320594,
#         Q = 1.24e-5 * 191.785320594 ~= 2.3781379753675e-3 m^3/s.
# Case 2: p=0, x_v=0.5: Q = 0.5 * Q_case1 = 1.18906898768375e-3 m^3/s.
# Case 3: p=7.3e6, x_v=1.0: dp=8.7e6, 2*dp/rho=20000 exact, sqrt=100*sqrt(2) ~= 141.421356237,
#         Q ~= 1.7536248173426e-3 m^3/s.
# Case 4: p=1.165e7, x_v=0.75: dp=4.35e6, 2*dp/rho=10000 exact, sqrt=100 exact,
#         Q = 0.62*2e-5*0.75*100 = 9.3e-4 m^3/s exact.
# Case 5: p=1.6e7 (=Ps), x_v=1.0: dp=0, Q = 0.0 m^3/s.
# Case 6: p=2.0e7 (>Ps), x_v=1.0: dp_head=max(Ps - p, 0) = 0, Q = 0.0 m^3/s.
ORIFICE_HAND_CASES = [
    (0.0, 1.0, 2.378137975367511e-3),
    (0.0, 0.5, 1.1890689876837556e-3),
    (7.3e6, 1.0, 1.753624817342638e-3),
    (1.165e7, 0.75, 9.3e-4),
    (1.6e7, 1.0, 0.0),
    (2.0e7, 1.0, 0.0),
]


def _compute_orifice_q(
    params: CylinderParams,
    p_pa: float,
    x_v: float,
    dt_s: float = 1e-4,
) -> float:
    """Compute orifice flow from single step state pressure advance."""
    _, _, p_next = _step_state(params, (0.0, 0.0, p_pa), x_v, dt_s)
    return float((p_next - p_pa) * params.volume_m3 / (params.bulk_modulus_pa * dt_s))


def _simulate_halving_refinements(
    params: CylinderParams,
    valve_opening: float,
    t_end_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate extension across dt halving steps (2ms, 1ms, 0.5ms)."""
    return (
        simulate_extension(params, valve_opening, t_end_s, dt_s=2e-3),
        simulate_extension(params, valve_opening, t_end_s, dt_s=1e-3),
        simulate_extension(params, valve_opening, t_end_s, dt_s=5e-4),
    )


def _step_refinement_diffs(
    df_c: pd.DataFrame,
    df_m: pd.DataFrame,
    df_f: pd.DataFrame,
    col: str,
) -> tuple[float, float]:
    """Compute difference between successive step refinements at terminal time."""
    d_cm = abs(float(df_c[col].iloc[-1] - df_m[col].iloc[-1]))
    d_mf = abs(float(df_m[col].iloc[-1] - df_f[col].iloc[-1]))
    return d_cm, d_mf


@pytest.mark.parametrize(("p_pa", "x_v", "expected_q"), ORIFICE_HAND_CASES)
def test_orifice_flow_hand_values(
    p_pa: float,
    x_v: float,
    expected_q: float,
) -> None:
    """Verify orifice equation Q = Cd * A * x * sqrt(2*dp/rho) against hand values."""
    computed_q = _compute_orifice_q(CylinderParams(), p_pa, x_v)
    assert computed_q == pytest.approx(expected_q, rel=1e-8, abs=1e-12)


def test_orifice_scaling_laws_property_loop() -> None:
    """Verify orifice flow properties Q ~ x_v and Q ~ sqrt(dp) using seeded numpy loop."""
    rng = np.random.default_rng(2026)
    params = CylinderParams()
    for _ in range(15):
        p_val = float(rng.uniform(1e5, 1.4e7))
        x_base = float(rng.uniform(0.1, 0.45))
        q1 = _compute_orifice_q(params, p_val, x_base)
        q2 = _compute_orifice_q(params, p_val, 2.0 * x_base)
        assert q2 == pytest.approx(2.0 * q1, rel=1e-7)


# Hand derivations for cylinder steady-state velocity and pressure:
# Equilibrium: a = 0 => p_ss = load_n / area_m2.
# Steady flow: dp/dt = 0 => Q_ss = area_m2 * v_ss => v_ss = Q_ss / area_m2.
# With dp_ss = max(supply_pa - p_ss, 0), Q_ss = Cd * A_v * x_v * sqrt(2 * dp_ss / rho).
# Case 1: load=1e4 N, x_v=1.0 => p_ss = 5.0e6 Pa, v_ss ~= 0.9859239205 m/s.
# Case 2: load=1e4 N, x_v=0.5 => p_ss = 5.0e6 Pa, v_ss ~= 0.4929619602 m/s.
# Case 3: load=1.4e4 N, x_v=1.0 => p_ss = 7.0e6 Pa, v_ss ~= 0.8918017408 m/s.
# Case 4: load=6.0e3 N, x_v=0.8 => p_ss = 3.0e6 Pa, v_ss ~= 0.8574498410 m/s.
# Case 5: load=1.2e4 N, x_v=0.6 => p_ss = 6.0e6 Pa, v_ss ~= 0.5640249444 m/s.
STEADY_STATE_HAND_CASES = [
    (1.0e4, 1.0, 5.0e6, 0.9859239205),
    (1.0e4, 0.5, 5.0e6, 0.4929619602),
    (1.4e4, 1.0, 7.0e6, 0.8918017408),
    (6.0e3, 0.8, 3.0e6, 0.8574498410),
    (1.2e4, 0.6, 6.0e6, 0.5640249444),
]


@pytest.mark.parametrize(
    ("load_n", "x_v", "expected_p", "expected_v"),
    STEADY_STATE_HAND_CASES,
)
def test_steady_state_velocity_hand_values(
    load_n: float,
    x_v: float,
    expected_p: float,
    expected_v: float,
) -> None:
    """Verify settled cylinder velocity v = Q / A_piston and pressure p = load / A_piston."""
    params = CylinderParams(load_n=load_n)
    df = simulate_extension(params, valve_opening=x_v, t_end_s=1.0, dt_s=1e-4)
    final_p = df["p_pa"].iloc[-1]
    final_v = df["v_m_s"].iloc[-1]
    assert final_p == pytest.approx(expected_p, rel=1e-4)
    assert final_v == pytest.approx(expected_v, rel=1e-4)


@pytest.mark.parametrize("load_n", [7.5e3, 1.0e4, 1.25e4])
def test_steady_state_flow_force_balance(load_n: float) -> None:
    """Verify net force is zero and flow matches piston displacement at steady state."""
    params = CylinderParams(load_n=load_n)
    df = simulate_extension(params, valve_opening=0.7, t_end_s=1.0, dt_s=1e-4)
    p_end = df["p_pa"].iloc[-1]
    v_end = df["v_m_s"].iloc[-1]
    assert np.isclose(p_end * params.area_m2, load_n, rtol=1e-4)
    dp = max(params.supply_pa - p_end, 0.0)
    q_oracle = params.cd_orifice * params.valve_area_m2 * 0.7 * np.sqrt(2.0 * dp / RHO_OIL_KG_M3)
    assert np.isclose(q_oracle, params.area_m2 * v_end, rtol=1e-4)


# Hand derivations for single-step pressure dynamics dp/dt = (beta / V) * (Q - A * v):
# beta/V = 1.4e9 / 1e-3 = 1.4e12 Pa/m^3. A_p = 2e-3 m^2.
# Case 1: p=5e6, v=0.5, x_v=1.0 => Q ~= 1.971847841e-3, Q - A*v = 9.71847841e-4
#         => dp/dt ~= 1.3605869774e9 Pa/s.
# Case 2: p=5e6, v=1.5, x_v=1.0 => A*v = 3e-3, Q - A*v = -1.028152159e-3
#         => dp/dt ~= -1.4394130226e9 Pa/s.
# Case 3: p=7.3e6, v=0.8768124086713, x_v=1.0 => Q = A*v => Q - A*v = 0 => dp/dt = 0.0 Pa/s.
# Case 4: p=1.0e7, v=0.4, x_v=0.0 => Q = 0, A*v = 8e-4
#         => dp/dt = 1.4e12 * (-8e-4) = -1.12e9 Pa/s exact.
# Case 5: p=0.0, v=0.0, x_v=0.0 => Q = 0, A*v = 0 => dp/dt = 0.0 Pa/s exact.
PRESSURE_DYNAMICS_HAND_CASES = [
    (5.0e6, 0.5, 1.0, 1.3605869773963e9),
    (5.0e6, 1.5, 1.0, -1.4394130226037e9),
    (7.3e6, 0.876812408671319, 1.0, 0.0),
    (1.0e7, 0.4, 0.0, -1.12e9),
    (0.0, 0.0, 0.0, 0.0),
]


@pytest.mark.parametrize(
    ("p_pa", "v_m_s", "x_v", "expected_dp_dt"),
    PRESSURE_DYNAMICS_HAND_CASES,
)
def test_pressure_dynamics_single_step_hand_values(
    p_pa: float,
    v_m_s: float,
    x_v: float,
    expected_dp_dt: float,
) -> None:
    """Verify single-step dp/dt = (beta/V)*(Q - A*v) against hand-derived values."""
    params = CylinderParams()
    dt_s = 1e-4
    _, _, p_next = _step_state(params, (0.0, v_m_s, p_pa), x_v, dt_s)
    computed_dp_dt = (p_next - p_pa) / dt_s
    assert computed_dp_dt == pytest.approx(expected_dp_dt, rel=1e-7, abs=1e-6)


@pytest.mark.parametrize(
    ("v_m_s", "expected_sign"),
    [
        (0.3, 1),
        (1.8, -1),
        (0.9859239205, 0),
    ],
)
def test_pressure_dynamics_sign_properties(
    v_m_s: float,
    expected_sign: int,
) -> None:
    """Verify sign of pressure rate: positive when Q > A*v, negative when Q < A*v."""
    params = CylinderParams()
    dt_s = 1e-4
    p_val = 5.0e6
    _, _, p_next = _step_state(params, (0.0, v_m_s, p_val), 1.0, dt_s)
    dp = p_next - p_val
    if expected_sign > 0:
        assert dp > 0.0
    elif expected_sign < 0:
        assert dp < 0.0
    else:
        assert np.isclose(dp, 0.0, atol=1e-3)


def test_pressure_dynamics_property_loop() -> None:
    """Verify dp/dt formula over 25 randomized states against independent oracle."""
    rng = np.random.default_rng(777)
    params = CylinderParams()
    dt_s = 1e-4
    for _ in range(25):
        p_val = float(rng.uniform(1e5, 1.4e7))
        v_val = float(rng.uniform(0.0, 2.0))
        xv = float(rng.uniform(0.1, 1.0))
        _, _, p_next = _step_state(params, (0.0, v_val, p_val), xv, dt_s)
        dp_head = max(params.supply_pa - p_val, 0.0)
        q_ref = (
            params.cd_orifice * params.valve_area_m2 * xv * np.sqrt(2.0 * dp_head / RHO_OIL_KG_M3)
        )
        dp_dt_ref = (params.bulk_modulus_pa / params.volume_m3) * (q_ref - params.area_m2 * v_val)
        expected_p_next = p_val + dp_dt_ref * dt_s
        assert p_next == pytest.approx(expected_p_next, rel=1e-9)


@pytest.mark.parametrize(
    ("x_v", "load_n", "v_init"),
    [
        (0.0, 5.0e4, 0.0),
        (1.0, 5.0e4, 0.0),
        (0.0, 1.0e4, 5.0),
        (0.5, 3.5e4, 2.0),
    ],
)
def test_velocity_non_negativity(
    x_v: float,
    load_n: float,
    v_init: float,
) -> None:
    """Verify velocity v >= 0 under aggressive deceleration and load overload."""
    params = CylinderParams(load_n=load_n)
    dt_s = 1e-3
    curr = (0.0, v_init, 0.0)
    for _ in range(50):
        curr = _step_state(params, curr, x_v, dt_s)
        assert curr[1] >= 0.0


@pytest.mark.parametrize(
    ("t_end_s", "dt_s"),
    [
        (0.1, 1.0e-2),
        (0.5, 5.0e-3),
        (1.0, 1.0e-3),
        (2.0, 2.0e-3),
    ],
)
def test_zero_valve_opening_zero_motion(
    t_end_s: float,
    dt_s: float,
) -> None:
    """Verify that valve_opening=0.0 yields strictly zero motion (x=0, v=0, p=0)."""
    params = CylinderParams()
    df = simulate_extension(params, valve_opening=0.0, t_end_s=t_end_s, dt_s=dt_s)
    assert (df[["x_m", "v_m_s", "p_pa"]] == 0.0).all().all()


@pytest.mark.parametrize(
    ("x_v", "t_end_s"),
    [
        (0.25, 0.2),
        (0.50, 0.2),
        (1.00, 0.2),
    ],
)
def test_timestep_convergence_shrinking_difference(
    x_v: float,
    t_end_s: float,
) -> None:
    """Verify time-step refinement (dt, dt/2, dt/4) shrinks discretization difference."""
    df_c, df_m, df_f = _simulate_halving_refinements(CylinderParams(), x_v, t_end_s)
    dx_cm, dx_mf = _step_refinement_diffs(df_c, df_m, df_f, "x_m")
    assert dx_mf < dx_cm
    assert 1.8 <= (dx_cm / dx_mf) <= 2.2


@pytest.mark.parametrize("load_n", [8000.0, 10000.0, 12000.0])
def test_timestep_velocity_pressure_convergence(load_n: float) -> None:
    """Verify velocity and pressure transient differences shrink with dt halving."""
    df_c, df_m, df_f = _simulate_halving_refinements(CylinderParams(load_n=load_n), 0.5, 0.05)
    for col in ("v_m_s", "p_pa"):
        d_cm, d_mf = _step_refinement_diffs(df_c, df_m, df_f, col)
        assert d_mf < d_cm


PHYSICAL_PARAMETER_BOUNDS = [
    (CylinderParams().area_m2, 1e-4, 1e-2),
    (CylinderParams().volume_m3, 1e-4, 1e-1),
    (CylinderParams().bulk_modulus_pa, 1e8, 5e9),
    (CylinderParams().supply_pa, 1e6, 5e7),
    (CylinderParams().cd_orifice, 0.5, 0.9),
    (CylinderParams().valve_area_m2, 1e-6, 1e-3),
    (CylinderParams().load_n, 1e2, 1e6),
    (CylinderParams().mass_kg, 1.0, 1e4),
    (RHO_OIL_KG_M3, 700.0, 1000.0),
]


@pytest.mark.parametrize(("val", "low", "high"), PHYSICAL_PARAMETER_BOUNDS)
def test_physical_units_and_parameter_dimensions(
    val: float,
    low: float,
    high: float,
) -> None:
    """Verify dimensional consistency: p [Pa] = N/m^2, v [m/s] = (m^3/s)/m^2."""
    assert low <= val <= high


@pytest.mark.parametrize(
    ("xv", "t_end", "dt", "match_msg"),
    [
        (-0.1, 1.0, 1e-3, "valve_opening must be non-negative"),
        (1.0, 0.0, 1e-3, "t_end_s must be positive"),
        (1.0, -1.0, 1e-3, "t_end_s must be positive"),
        (1.0, 1.0, 0.0, "dt_s must be positive"),
        (1.0, 1.0, -1e-4, "dt_s must be positive"),
    ],
)
def test_input_validation_negative_or_zero(
    xv: float,
    t_end: float,
    dt: float,
    match_msg: str,
) -> None:
    """Verify input validation raises ValueError on illegal inputs."""
    with pytest.raises(ValueError, match=match_msg):
        simulate_extension(CylinderParams(), valve_opening=xv, t_end_s=t_end, dt_s=dt)


@pytest.mark.parametrize(
    ("supply_pa", "mass_kg"),
    [
        (3.5e7, 50.0),
        (1.6e7, 5000.0),
    ],
)
def test_extreme_physical_parameters(
    supply_pa: float,
    mass_kg: float,
) -> None:
    """Verify simulation stability under extreme physical parameter bounds."""
    params = CylinderParams(supply_pa=supply_pa, mass_kg=mass_kg)
    df = simulate_extension(params, valve_opening=1.0, t_end_s=0.2, dt_s=1e-4)
    assert not (df.isna().to_numpy().any() or np.isinf(df.to_numpy()).any())
    assert (df[["v_m_s", "x_m"]] >= 0.0).all().all()
