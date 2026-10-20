"""Tests for ASTM D341 Walther viscosity calibration and normalized pump leakage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hydro.features.viscosity import (
    calibrate_walther,
    compute_oil_viscosity,
    normalized_pump_leakage,
    walther_viscosity,
)


def test_calibrate_walther_round_trip() -> None:
    """Walther calibration must round-trip both calibration points within 1e-6 relative."""
    t1_c, nu1_cst = 40.0, 46.0
    t2_c, nu2_cst = 100.0, 6.8
    a, b = calibrate_walther(t1_c, nu1_cst, t2_c, nu2_cst)

    evaluated = walther_viscosity(np.array([t1_c, t2_c]), a, b)
    expected = np.array([nu1_cst, nu2_cst])
    np.testing.assert_allclose(evaluated, expected, rtol=1e-6)


def test_calibrate_walther_guard_raises() -> None:
    """Log domain guard must raise ValueError when nu + 0.7 <= 1.0."""
    with pytest.raises(ValueError, match="log domain"):
        calibrate_walther(40.0, 0.2, 100.0, 6.8)

    with pytest.raises(ValueError, match="log domain"):
        calibrate_walther(40.0, 46.0, 100.0, 0.2)

    with pytest.raises(ValueError, match="log domain"):
        calibrate_walther(40.0, 0.3, 100.0, 6.8)


def test_compute_oil_viscosity_20c() -> None:
    """ISO VG 46 reference viscosity at 20 degC must be approx 135 cSt within 10%."""
    nu_20 = compute_oil_viscosity(np.array([20.0]))[0]
    rel_error = abs(nu_20 - 135.0) / 135.0
    assert rel_error < 0.10


def test_viscosity_temperature_derivative_negative() -> None:
    """Viscosity must strictly decrease with increasing temperature across [10, 80] degC."""
    temps = np.linspace(10.0, 80.0, 200)
    viscosities = compute_oil_viscosity(temps)
    diffs = np.diff(viscosities)
    assert np.all(diffs < 0.0)


def test_normalized_pump_leakage_synthetic_positive_finite() -> None:
    """Leakage coefficient K must be positive and finite on valid synthetic cycles."""
    n = 3
    power_w = np.full((n, 6000), 5000.0)
    pressure_bar = np.full((n, 6000), 50.0)
    flow_lpm = np.full((n, 600), 40.0)
    temp_c = np.full((n, 60), 40.0)

    k = normalized_pump_leakage(power_w, pressure_bar, flow_lpm, temp_c)
    assert k.shape == (n,)
    assert np.all(np.isfinite(k))
    assert np.all(k > 0.0)


def test_normalized_pump_leakage_temperature_scaling() -> None:
    """K ratio at two temperatures with identical kinematics equals viscosity ratio within 1e-6."""
    power_w = np.full((2, 6000), 5000.0)
    pressure_bar = np.full((2, 6000), 50.0)
    flow_lpm = np.full((2, 600), 40.0)
    temp_c = np.array([[30.0] * 60, [60.0] * 60])

    k = normalized_pump_leakage(power_w, pressure_bar, flow_lpm, temp_c)
    nu = compute_oil_viscosity(np.array([30.0, 60.0]))

    k_ratio = k[0] / k[1]
    nu_ratio = nu[0] / nu[1]
    np.testing.assert_allclose(k_ratio, nu_ratio, rtol=1e-6)


def test_normalized_pump_leakage_shape_mismatch_raises() -> None:
    """Incompatible array dimensions or lengths must raise ValueError."""
    p_valid = np.ones((2, 6000))
    q_valid = np.ones((2, 600))
    t_valid = np.ones((2, 60))

    with pytest.raises(ValueError, match="2D arrays"):
        normalized_pump_leakage(np.ones(6000), p_valid, q_valid, t_valid)

    with pytest.raises(ValueError, match="Cycle count mismatch"):
        normalized_pump_leakage(np.ones((3, 6000)), p_valid, q_valid, t_valid)

    with pytest.raises(ValueError, match="Sample count mismatch"):
        normalized_pump_leakage(np.ones((2, 5000)), p_valid, q_valid, t_valid)

    with pytest.raises(ValueError, match="Sample count mismatch"):
        normalized_pump_leakage(p_valid, p_valid, np.ones((2, 500)), t_valid)

    with pytest.raises(ValueError, match="Sample count mismatch"):
        normalized_pump_leakage(p_valid, p_valid, q_valid, np.ones((2, 50)))


@pytest.mark.heavy
def test_viscosity_real_data(raw_dir: Path) -> None:
    """Verify viscosity calculation on real TS1 sensor data against physical bounds.

    Units: temperature in degC, kinematic viscosity in cSt (mm^2/s).
    """
    ts1 = np.loadtxt(raw_dir / "TS1.txt", delimiter="\t", dtype=np.float32)
    viscosity = compute_oil_viscosity(ts1)

    assert np.all(np.isfinite(viscosity)), "Viscosities must be finite"
    assert np.all(viscosity > 0.0), "Viscosities must be strictly positive"

    t_sorted = np.sort(np.unique(ts1))
    v_sorted = compute_oil_viscosity(t_sorted)
    assert np.all(np.diff(v_sorted) < 0.0), "Viscosity must decrease with temperature"

    v_40 = float(compute_oil_viscosity(np.array([40.0]))[0])
    assert 41.4 <= v_40 <= 50.6, f"Viscosity at 40 C ({v_40:.2f} cSt) outside ISO VG 46 band"
