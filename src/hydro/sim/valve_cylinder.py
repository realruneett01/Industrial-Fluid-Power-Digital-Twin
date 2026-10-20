"""SIMULATION ARTIFACT - not validated against ZeMA data.

Lumped-parameter hydraulic valve and cylinder extension simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RHO_OIL_KG_M3: float = 870.0


@dataclass(frozen=True)
class CylinderParams:
    """Hydraulic cylinder and servo-valve lumped parameters (SI units)."""

    area_m2: float = 2e-3
    volume_m3: float = 1e-3
    bulk_modulus_pa: float = 1.4e9
    supply_pa: float = 1.6e7
    cd_orifice: float = 0.62
    valve_area_m2: float = 2e-5
    load_n: float = 1e4
    mass_kg: float = 50.0


def _validate_sim_inputs(valve_opening: float, t_end_s: float, dt_s: float) -> None:
    """Validate numeric ranges for extension simulation inputs."""
    if valve_opening < 0.0:
        raise ValueError(f"valve_opening must be non-negative, got {valve_opening}")
    if t_end_s <= 0.0:
        raise ValueError(f"t_end_s must be positive, got {t_end_s}")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")


def _step_state(
    params: CylinderParams,
    state: tuple[float, float, float],
    valve_opening: float,
    dt_s: float,
) -> tuple[float, float, float]:
    """Advance cylinder position, velocity, and pressure by dt_s via semi-implicit Euler."""
    x, v, p = state
    dp_head = max(params.supply_pa - p, 0.0)
    q = (
        params.cd_orifice
        * params.valve_area_m2
        * valve_opening
        * np.sqrt(2.0 * dp_head / RHO_OIL_KG_M3)
    )
    dp_dt = (params.bulk_modulus_pa / params.volume_m3) * (q - params.area_m2 * v)
    p_next = p + dp_dt * dt_s

    force_net = p_next * params.area_m2 - params.load_n
    a = force_net / params.mass_kg
    v_next = max(0.0, v + a * dt_s)
    x_next = x + v_next * dt_s
    return (x_next, v_next, p_next)


def simulate_extension(
    params: CylinderParams,
    valve_opening: float,
    t_end_s: float = 2.0,
    dt_s: float = 1e-3,
) -> pd.DataFrame:
    """Simulate cylinder extension stroke under orifice flow control.

    Assumes constant density rho=870 kg/m^3 (viscosity enters through rho only).

    Returns DataFrame with columns: t_s, x_m, v_m_s, p_pa.
    """
    _validate_sim_inputs(valve_opening, t_end_s, dt_s)
    n_steps = int(np.round(t_end_s / dt_s))
    t_arr = np.linspace(0.0, t_end_s, n_steps + 1)
    x_arr = np.zeros(n_steps + 1, dtype=np.float64)
    v_arr = np.zeros(n_steps + 1, dtype=np.float64)
    p_arr = np.zeros(n_steps + 1, dtype=np.float64)

    curr = (0.0, 0.0, 0.0)
    for k in range(n_steps):
        curr = _step_state(params, curr, valve_opening, dt_s)
        x_arr[k + 1], v_arr[k + 1], p_arr[k + 1] = curr

    return pd.DataFrame(
        {
            "t_s": t_arr,
            "x_m": x_arr,
            "v_m_s": v_arr,
            "p_pa": p_arr,
        }
    )
