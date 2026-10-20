"""Mathematical verification of ASTM D341 Walther viscosity and pump leakage index."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from hydro.features.viscosity import (
    calibrate_walther,
    compute_oil_viscosity,
    normalized_pump_leakage,
    walther_viscosity,
)

# Hand-derived closed-form Walther parameters from two ISO VG 46 calibration points:
# (T1 = 40.0 C, nu1 = 46.0 cSt) -> T_K1 = 313.15 K
# (T2 = 100.0 C, nu2 = 6.8 cSt) -> T_K2 = 373.15 K
# W1 = log10(log10(46.0 + 0.7)) = log10(log10(46.7))
# W2 = log10(log10(6.8 + 0.7))  = log10(log10(7.5))
# B = (W1 - W2) / (log10(373.15) - log10(313.15))
# A = W1 + B * log10(313.15)
_W1_HAND = math.log10(math.log10(46.7))
_W2_HAND = math.log10(math.log10(7.5))
_LOG_TK1_HAND = math.log10(313.15)
_LOG_TK2_HAND = math.log10(373.15)
B_HAND = (_W1_HAND - _W2_HAND) / (_LOG_TK2_HAND - _LOG_TK1_HAND)
A_HAND = _W1_HAND + B_HAND * _LOG_TK1_HAND


def _walther_independent_oracle(temp_c: float, a: float, b: float) -> float:
    """Independent closed-form oracle for ASTM D341 Walther viscosity in cSt."""
    # Derivation: W = A - B * log10(T_K); nu = 10^(10^W) - 0.7
    tk = temp_c + 273.15
    w = a - b * math.log10(tk)
    return 10.0 ** (10.0**w) - 0.7


def _walther_inverse_oracle(nu_cst: float, a: float, b: float) -> float:
    """Independent closed-form inverse oracle for temperature in degC from viscosity in cSt."""
    # Derivation: W = log10(log10(nu + 0.7)) = A - B*log10(T_K)
    # T_K = 10^((A - W) / B); T_C = T_K - 273.15
    w = math.log10(math.log10(nu_cst + 0.7))
    tk = 10.0 ** ((a - w) / b)
    return tk - 273.15


@dataclass(frozen=True)
class LeakageCase:
    """Test parameters and expected analytical leakage coefficient K in m^6/kg."""

    power_w: float
    pressure_bar: float
    flow_lpm: float
    temp_c: float
    eta_motor: float
    expected_k: float


def _make_leakage_inputs(
    case: LeakageCase,
    n_cycles: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic 2D array inputs for normalized_pump_leakage."""
    p_w = np.full((n_cycles, 6000), case.power_w, dtype=np.float64)
    p_bar = np.full((n_cycles, 6000), case.pressure_bar, dtype=np.float64)
    q_lpm = np.full((n_cycles, 600), case.flow_lpm, dtype=np.float64)
    t_c = np.full((n_cycles, 60), case.temp_c, dtype=np.float64)
    return p_w, p_bar, q_lpm, t_c


def test_iso_vg46_anchor_40c() -> None:
    """Verify ISO VG 46 nominal viscosity at 40 C is 46 cSt within ISO 3448 band [41.4, 50.6]."""
    nu_40 = float(compute_oil_viscosity(np.array([40.0]))[0])
    assert 41.4 <= nu_40 <= 50.6
    assert nu_40 == pytest.approx(46.0, rel=1e-6)


def test_iso_vg46_anchor_100c() -> None:
    """Verify ISO VG 46 viscosity at 100 C is approx 6.8 cSt for VI approx 100 mineral oil."""
    nu_100 = float(compute_oil_viscosity(np.array([100.0]))[0])
    assert 6.0 <= nu_100 <= 7.5
    assert nu_100 == pytest.approx(6.8, rel=1e-6)


@pytest.mark.parametrize(
    ("temp_c", "min_nu", "max_nu"),
    [
        (0.0, 400.0, 700.0),
        (20.0, 120.0, 160.0),
        (40.0, 41.4, 50.6),
        (60.0, 18.0, 24.0),
        (80.0, 9.5, 13.0),
        (100.0, 6.0, 7.5),
    ],
)
def test_iso_vg46_standard_temperature_bands(
    temp_c: float,
    min_nu: float,
    max_nu: float,
) -> None:
    """Verify viscosity falls within standard physical bounds for VG 46 mineral oil."""
    nu = float(compute_oil_viscosity(np.array([temp_c]))[0])
    assert min_nu <= nu <= max_nu


def test_calibrate_walther_hand_derived_oracle() -> None:
    """Verify Walther calibration matches hand-derived analytical solution."""
    # Closed-form solution: W1=log10(log10(46.7)), W2=log10(log10(7.5))
    # B = (W1 - W2) / (log10(373.15) - log10(313.15)), A = W1 + B * log10(313.15)
    a, b = calibrate_walther(40.0, 46.0, 100.0, 6.8)
    assert a == pytest.approx(A_HAND, rel=1e-12)
    assert b == pytest.approx(B_HAND, rel=1e-12)


def test_calibrate_walther_order_invariance() -> None:
    """Verify Walther calibration yields identical parameters regardless of point order."""
    a_fwd, b_fwd = calibrate_walther(40.0, 46.0, 100.0, 6.8)
    a_rev, b_rev = calibrate_walther(100.0, 6.8, 40.0, 46.0)
    assert a_fwd == pytest.approx(a_rev, rel=1e-12)
    assert b_fwd == pytest.approx(b_rev, rel=1e-12)


@pytest.mark.parametrize(
    "temp_c",
    [-10.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0],
)
def test_walther_viscosity_vs_hand_oracle(temp_c: float) -> None:
    """Compare Walther viscosity against independent closed-form hand oracle."""
    # Closed-form oracle: nu = 10^(10^(A - B * log10(temp_c + 273.15))) - 0.7
    expected = _walther_independent_oracle(temp_c, A_HAND, B_HAND)
    evaluated = float(walther_viscosity(np.array([temp_c]), A_HAND, B_HAND)[0])
    default_evaluated = float(compute_oil_viscosity(np.array([temp_c]))[0])
    assert evaluated == pytest.approx(expected, rel=1e-12)
    assert default_evaluated == pytest.approx(expected, rel=1e-12)


def test_viscosity_strict_monotonic_seeded_random() -> None:
    """Verify strict monotonic decrease across 100 seeded random temperatures."""
    rng = np.random.default_rng(2026)
    temps = np.sort(rng.uniform(-20.0, 140.0, size=100))
    viscosities = compute_oil_viscosity(temps)
    diffs = np.diff(viscosities)
    assert np.all(diffs < 0.0)


@pytest.mark.parametrize(
    "t_eval",
    [-10.0, 10.0, 40.0, 75.0, 110.0],
)
def test_viscosity_finite_difference_derivative_negative(t_eval: float) -> None:
    """Verify numerical derivative d(nu)/dT is strictly negative at sample temperatures."""
    h = 1e-4
    t_pair = np.array([t_eval - h, t_eval + h])
    nu_pair = compute_oil_viscosity(t_pair)
    d_nu_dt = (nu_pair[1] - nu_pair[0]) / (2.0 * h)
    assert d_nu_dt < 0.0


@pytest.mark.parametrize(
    "temp_c",
    [-5.0, 20.0, 40.0, 70.0, 100.0],
)
def test_walther_inverse_round_trip_from_temperature(temp_c: float) -> None:
    """Verify T -> nu -> T round-trip consistency using closed-form inverse oracle."""
    # Closed-form inverse: T_K = 10^((A - log10(log10(nu + 0.7))) / B), T_C = T_K - 273.15
    nu = float(walther_viscosity(np.array([temp_c]), A_HAND, B_HAND)[0])
    recovered_t = _walther_inverse_oracle(nu, A_HAND, B_HAND)
    assert recovered_t == pytest.approx(temp_c, abs=1e-8)


@pytest.mark.parametrize(
    "target_nu",
    [1.5, 6.8, 20.0, 46.0, 200.0],
)
def test_walther_inverse_round_trip_from_viscosity(target_nu: float) -> None:
    """Verify nu -> T -> nu round-trip consistency using closed-form inverse oracle."""
    t_solved = _walther_inverse_oracle(target_nu, A_HAND, B_HAND)
    evaluated_nu = float(walther_viscosity(np.array([t_solved]), A_HAND, B_HAND)[0])
    assert evaluated_nu == pytest.approx(target_nu, rel=1e-8)


@pytest.mark.parametrize(
    "invalid_temp_c",
    [-273.15, -273.16, -300.0],
)
def test_absolute_temperature_guard_raises(invalid_temp_c: float) -> None:
    """Verify walther_viscosity raises ValueError when T_K <= 0."""
    with pytest.raises(ValueError, match="Absolute temperature must be positive"):
        walther_viscosity(np.array([invalid_temp_c]), A_HAND, B_HAND)


@pytest.mark.parametrize(
    ("t1", "t2", "match_msg"),
    [
        (-273.15, 100.0, "positive"),
        (40.0, -273.15, "positive"),
        (50.0, 50.0, "distinct"),
    ],
)
def test_calibrate_walther_temperature_guards(
    t1: float,
    t2: float,
    match_msg: str,
) -> None:
    """Verify calibrate_walther raises ValueError for non-positive or non-distinct temps."""
    with pytest.raises(ValueError, match=match_msg):
        calibrate_walther(t1, 46.0, t2, 6.8)


def test_kelvin_vs_celsius_distinction() -> None:
    """Verify that passing Kelvin values (e.g. 313.15 K) produces distinct non-40 C values."""
    # 40 C = 313.15 K. If 313.15 is passed as temp_c, it is treated as 313.15 C (586.3 K)
    nu_celsius = float(compute_oil_viscosity(np.array([40.0]))[0])
    nu_kelvin_as_celsius = float(compute_oil_viscosity(np.array([313.15]))[0])
    assert nu_celsius == pytest.approx(46.0, rel=1e-6)
    assert nu_kelvin_as_celsius < 1.0
    assert abs(nu_celsius - nu_kelvin_as_celsius) > 40.0


@pytest.mark.parametrize(
    "shape",
    [(), (1,), (5,), (3, 4), (2, 3, 2), (0,)],
)
def test_walther_viscosity_preserves_shape(shape: tuple[int, ...]) -> None:
    """Verify walther_viscosity and compute_oil_viscosity preserve arbitrary array shapes."""
    arr = np.full(shape, 40.0, dtype=np.float64)
    res_walther = walther_viscosity(arr, A_HAND, B_HAND)
    res_compute = compute_oil_viscosity(arr)
    assert res_walther.shape == shape
    assert res_compute.shape == shape


# Dimensional reasoning for pump leakage index K:
# P_loss = power * eta - pressure_pa * flow_m3s  [W = kg*m^2/s^3]
# nu_m2s = cSt * 1e-6                             [m^2/s]
# pressure_pa = pressure_bar * 1e5                [Pa = kg/(m*s^2)]
# K = (P_loss * nu_m2s) / (pressure_pa)^2
#   = [ (kg*m^2/s^3) * (m^2/s) ] / [ (kg/(m*s^2))^2 ]
#   = [ kg*m^4/s^4 ] / [ kg^2/(m^2*s^4) ] = m^6 / kg
LEAKAGE_CASES = (
    # Standard: P_loss = 10000*0.9 - 5e6*1e-3 = 4000 W; nu = 4.6e-5; p^2 = 25e12
    # K = 4000 * 4.6e-5 / 25e12 = 7.36e-15 m^6/kg
    LeakageCase(10000.0, 50.0, 60.0, 40.0, 0.90, 7.36e-15),
    # Zero loss: power = 5000/0.9 -> P_loss = 0 -> K = 0.0
    LeakageCase(5000.0 / 0.9, 50.0, 60.0, 40.0, 0.90, 0.0),
    # 100 bar: P_loss = 20000*0.9 - 1e7*1e-3 = 8000 W; p^2 = 1e14
    # K = 8000 * 4.6e-5 / 1e14 = 3.68e-15 m^6/kg
    LeakageCase(20000.0, 100.0, 60.0, 40.0, 0.90, 3.68e-15),
    # 100 C: nu = 6.8e-6; P_loss = 4000 W; K = 4000 * 6.8e-6 / 25e12 = 1.088e-15 m^6/kg
    LeakageCase(10000.0, 50.0, 60.0, 100.0, 0.90, 1.088e-15),
)


@pytest.mark.parametrize("case", LEAKAGE_CASES)
def test_pump_leakage_analytic_cases(case: LeakageCase) -> None:
    """Verify normalized pump leakage against exact hand-derived dimensional oracles."""
    p_w, p_bar, q_lpm, t_c = _make_leakage_inputs(case)
    k = normalized_pump_leakage(p_w, p_bar, q_lpm, t_c, eta_motor=case.eta_motor)
    assert k.shape == (2,)
    if case.expected_k == 0.0:
        np.testing.assert_allclose(k, 0.0, atol=1e-22)
    else:
        np.testing.assert_allclose(k, case.expected_k, rtol=1e-6)


@pytest.mark.parametrize(
    ("p_val", "expect_nan"),
    [(0.5, True), (1.0, True), (1.1, False)],
)
def test_pump_leakage_sub_bar_threshold(p_val: float, expect_nan: bool) -> None:
    """Verify samples with pressure <= 1 bar are masked out; all-sub-bar returns NaN."""
    p_w = np.full((1, 6000), 5000.0)
    p_bar = np.full((1, 6000), p_val)
    q_lpm = np.full((1, 600), 40.0)
    t_c = np.full((1, 60), 40.0)
    k = normalized_pump_leakage(p_w, p_bar, q_lpm, t_c)
    assert np.isnan(k[0]) if expect_nan else np.isfinite(k[0])


def test_pump_leakage_empty_cycles() -> None:
    """Verify N=0 cycles returns empty 1D array of shape (0,)."""
    p_w = np.empty((0, 6000))
    p_bar = np.empty((0, 6000))
    q_lpm = np.empty((0, 600))
    t_c = np.empty((0, 60))
    k = normalized_pump_leakage(p_w, p_bar, q_lpm, t_c)
    assert k.shape == (0,)


@pytest.mark.parametrize(
    ("shapes", "match_err"),
    [
        ((6000, (1, 6000), (1, 600), (1, 60)), "2D arrays"),
        (((2, 6000), (1, 6000), (1, 600), (1, 60)), "Cycle count"),
        (((1, 5000), (1, 6000), (1, 600), (1, 60)), "Sample count"),
        (((1, 6000), (1, 6000), (1, 500), (1, 60)), "Sample count"),
    ],
)
def test_pump_leakage_shape_validation(
    shapes: tuple[object, ...],
    match_err: str,
) -> None:
    """Verify input shape validation rejects improper dimensions and sample counts."""
    inputs = [np.ones(s) for s in shapes]
    with pytest.raises(ValueError, match=match_err):
        normalized_pump_leakage(inputs[0], inputs[1], inputs[2], inputs[3])


def test_walther_viscosity_nan_propagation() -> None:
    """Verify NaN in temperature input propagates to NaN in viscosity output."""
    t_arr = np.array([40.0, np.nan, 100.0])
    res = walther_viscosity(t_arr, A_HAND, B_HAND)
    assert np.isfinite(res[0])
    assert np.isnan(res[1])
    assert np.isfinite(res[2])


def test_walther_viscosity_high_temp_asymptote() -> None:
    """Verify that as T -> infinity, Walther viscosity approaches physical asymptote (0.3 cSt)."""
    # As T -> inf, log10(T) -> inf, W = A - B*log10(T) -> -inf
    # 10^W -> 0, 10^0 - 0.7 = 1.0 - 0.7 = 0.3 cSt
    t_extreme = np.array([500.0, 1000.0, 5000.0])
    res = walther_viscosity(t_extreme, A_HAND, B_HAND)
    assert np.all(res > 0.3)
    assert np.all(np.diff(res) < 0.0)
    assert res[-1] == pytest.approx(0.3, abs=0.01)


@pytest.mark.parametrize(
    ("nu1", "nu2"),
    [(0.3, 6.8), (46.0, 0.3), (0.2, 0.2)],
)
def test_calibrate_walther_viscosity_domain_guard(nu1: float, nu2: float) -> None:
    """Verify calibrate_walther raises ValueError when nu + 0.7 <= 1.0 (nu <= 0.3 cSt)."""
    with pytest.raises(ValueError, match="log domain guard"):
        calibrate_walther(40.0, nu1, 100.0, nu2)
