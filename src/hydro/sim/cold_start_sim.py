"""SIMULATION ARTIFACT - not validated against ZeMA data.

Cold-start ram acceleration governor based on ASTM D341 kinematic viscosity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydro.features.viscosity import compute_oil_viscosity


def _get_viscosity_ratio(temp_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute kinematic viscosity and ratio nu(40 C) / nu(T)."""
    nu_40 = float(compute_oil_viscosity(np.array([40.0]))[0])
    nu_t = compute_oil_viscosity(temp_c)
    return nu_t, nu_40 / nu_t


def acceleration_limit(temp_c: float, nominal_m_s2: float = 1.0) -> float:
    """Compute governed ram acceleration limit in m/s^2 for a given oil temperature.

    Scales nominal by nu(40 C) / nu(T), clipped to [0.1 * nominal, 1.0 * nominal].
    """
    if nominal_m_s2 <= 0.0:
        raise ValueError(f"nominal_m_s2 must be positive, got {nominal_m_s2}")
    _, ratio = _get_viscosity_ratio(np.array([float(temp_c)]))
    clipped = np.clip(nominal_m_s2 * ratio[0], 0.1 * nominal_m_s2, 1.0 * nominal_m_s2)
    return float(clipped)


def simulate_cold_start(
    temps_c: np.ndarray,
    nominal_m_s2: float = 1.0,
) -> pd.DataFrame:
    """Simulate cold start ram governor across an array of temperatures in degC.

    Returns DataFrame with columns: temp_c, nu_cst, accel_limit_m_s2.
    """
    if nominal_m_s2 <= 0.0:
        raise ValueError(f"nominal_m_s2 must be positive, got {nominal_m_s2}")
    t_arr = np.asarray(temps_c, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ValueError(f"temps_c must be a 1D array, got ndim={t_arr.ndim}")
    nu_t, ratio = _get_viscosity_ratio(t_arr)
    accel = np.clip(nominal_m_s2 * ratio, 0.1 * nominal_m_s2, 1.0 * nominal_m_s2)
    return pd.DataFrame(
        {
            "temp_c": t_arr,
            "nu_cst": nu_t,
            "accel_limit_m_s2": accel,
        }
    )
