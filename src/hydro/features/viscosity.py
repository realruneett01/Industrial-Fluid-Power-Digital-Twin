"""ASTM D341 Walther viscosity calibration and normalized pump leakage features."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml


def calibrate_walther(
    t1_c: float,
    nu1_cst: float,
    t2_c: float,
    nu2_cst: float,
) -> tuple[float, float]:
    """Calibrate ASTM D341 Walther viscosity: log10(log10(nu + 0.7)) = A - B*log10(T_K).

    Returns (A, B) in float64 from two (T, nu) points in degC and cSt.
    """
    if nu1_cst + 0.7 <= 1.0 or nu2_cst + 0.7 <= 1.0:
        raise ValueError(
            f"Viscosity log domain guard: nu + 0.7 must be > 1.0, got {nu1_cst} and {nu2_cst}"
        )
    tk1 = float(t1_c) + 273.15
    tk2 = float(t2_c) + 273.15
    if tk1 <= 0.0 or tk2 <= 0.0 or tk1 == tk2:
        raise ValueError("Absolute temperatures must be positive (> 0 K) and distinct")

    w1 = np.float64(np.log10(np.log10(nu1_cst + 0.7)))
    w2 = np.float64(np.log10(np.log10(nu2_cst + 0.7)))
    log_tk1 = np.float64(np.log10(tk1))
    log_tk2 = np.float64(np.log10(tk2))

    b = float((w1 - w2) / (log_tk2 - log_tk1))
    a = float(w1 + b * log_tk1)
    return (a, b)


def walther_viscosity(temp_c: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate ASTM D341 Walther kinematic viscosity in cSt (mm^2/s).

    nu = 10^(10^(A - B * log10(T_K))) - 0.7, with T_K = temp_c + 273.15.
    """
    t_arr = np.asarray(temp_c, dtype=np.float64)
    tk = t_arr + 273.15
    if np.any(tk <= 0.0):
        raise ValueError("Absolute temperature must be positive (T_K > 0)")
    w = np.float64(a) - np.float64(b) * np.log10(tk)
    return np.asarray(10.0 ** (10.0**w) - 0.7, dtype=np.float64)


@lru_cache(maxsize=1)
def _load_default_walther_params() -> tuple[float, float]:
    """Load reference oil parameters from configs/oil_vg46.yaml and calibrate Walther."""
    config_path = Path(__file__).resolve().parents[3] / "configs" / "oil_vg46.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return calibrate_walther(
        float(cfg["t1_c"]),
        float(cfg["nu1_cst"]),
        float(cfg["t2_c"]),
        float(cfg["nu2_cst"]),
    )


def compute_oil_viscosity(temp_c: np.ndarray) -> np.ndarray:
    """Compute ISO VG 46 kinematic viscosity in cSt (mm^2/s) using configs/oil_vg46.yaml."""
    a, b = _load_default_walther_params()
    return walther_viscosity(temp_c, a, b)


def _validate_leakage_shapes(
    power_w: np.ndarray,
    pressure_bar: np.ndarray,
    flow_lpm: np.ndarray,
    temp_c: np.ndarray,
) -> int:
    """Validate 2D shapes and dimensions for pump leakage inputs."""
    expected = (
        (power_w, 6000),
        (pressure_bar, 6000),
        (flow_lpm, 600),
        (temp_c, 60),
    )
    if any(arr.ndim != 2 for arr, _ in expected):
        raise ValueError("All inputs to normalized_pump_leakage must be 2D arrays")
    n = power_w.shape[0]
    if any(arr.shape[0] != n for arr, _ in expected):
        raise ValueError(
            f"Cycle count mismatch: power={n}, pressure={pressure_bar.shape[0]}, "
            f"flow={flow_lpm.shape[0]}, temp={temp_c.shape[0]}"
        )
    if any(arr.shape[1] != samples for arr, samples in expected):
        raise ValueError(
            f"Sample count mismatch: expected (*, 6000), (*, 6000), (*, 600), (*, 60), "
            f"got {power_w.shape}, {pressure_bar.shape}, {flow_lpm.shape}, {temp_c.shape}"
        )
    return n


def _compute_leakage_integral(
    p_loss: np.ndarray,
    pressure_pa: np.ndarray,
    mask: np.ndarray,
    nu_m2s: np.ndarray,
) -> np.ndarray:
    """Integrate leakage integrand over masked pressure samples (> 1 bar)."""
    p_pa_safe = np.where(mask, pressure_pa, 1.0)
    integrand = (p_loss * nu_m2s[:, np.newaxis]) / (p_pa_safe**2)
    integrand_masked = np.where(mask, integrand, 0.0)
    counts = np.sum(mask, axis=1)
    return np.divide(
        np.sum(integrand_masked, axis=1),
        counts,
        out=np.full(mask.shape[0], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def normalized_pump_leakage(
    power_w: np.ndarray,
    pressure_bar: np.ndarray,
    flow_lpm: np.ndarray,
    temp_c: np.ndarray,
    eta_motor: float = 0.90,
) -> np.ndarray:
    """Compute normalized pump leakage coefficient K for each cycle.

    P_loss(t) = power*eta - (pressure*1e5)*(flow*1e-3/60) in W.
    nu_m2s = cSt*1e-6 in m^2/s.
    K = mean over samples with pressure > 1 bar of P_loss*nu_m2s/(pressure*1e5)**2.

    Unit: m^6/kg. Lumps motor losses with leakage (relative index only).
    """
    p_w = np.asarray(power_w, dtype=np.float64)
    p_bar = np.asarray(pressure_bar, dtype=np.float64)
    q_lpm = np.asarray(flow_lpm, dtype=np.float64)
    t_c = np.asarray(temp_c, dtype=np.float64)

    n = _validate_leakage_shapes(p_w, p_bar, q_lpm, t_c)
    if n == 0:
        return np.empty((0,), dtype=np.float64)

    nu_m2s = compute_oil_viscosity(np.mean(t_c, axis=1)) * 1e-6
    flow_m3s = np.repeat(q_lpm, 10, axis=1) * (1e-3 / 60.0)
    pressure_pa = p_bar * 1e5
    p_loss = p_w * float(eta_motor) - pressure_pa * flow_m3s

    return _compute_leakage_integral(p_loss, pressure_pa, p_bar > 1.0, nu_m2s)
