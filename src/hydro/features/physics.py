"""Grey-box hydraulic physics parameter observer and feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from hydro.data.schema import CYCLE_SECONDS


def _upsample_flow(flow: np.ndarray, n_target: int) -> np.ndarray:
    """Linearly upsample flow (l/min) along axis=-1 to match target sample count."""
    n_source = flow.shape[-1]
    if n_source == n_target:
        return flow
    rate_source = n_source / CYCLE_SECONDS
    rate_target = n_target / CYCLE_SECONDS
    x_source = np.arange(n_source) / rate_source
    x_target = np.arange(n_target) / rate_target
    interp = interp1d(x_source, flow, axis=-1, kind="linear", fill_value="extrapolate")
    return interp(x_target)


def _extract_inflection_times(pressure: np.ndarray, fs_hz: float) -> np.ndarray:
    """Return inflection time in s as argmax of 2nd derivative of SG-smoothed signal."""
    d2 = savgol_filter(pressure, window_length=21, polyorder=2, deriv=2, axis=-1)
    inflect_idx = np.argmax(d2, axis=-1)
    return inflect_idx / fs_hz


def compute_pressure_derivative(p_bar: np.ndarray, fs_hz: float = 100.0) -> np.ndarray:
    """Compute pressure derivative dp/dt in Pa/s via Savitzky-Golay smoothed gradient."""
    if fs_hz <= 0:
        raise ValueError("Sampling frequency fs_hz must be positive.")
    p = np.asarray(p_bar, dtype=float)
    if p.shape[-1] < 21:
        raise ValueError("Input signal length must be at least 21 for Savitzky-Golay filter.")
    p_pa = p * 1.0e5
    smoothed_pa = savgol_filter(p_pa, window_length=21, polyorder=2, axis=-1)
    return np.gradient(smoothed_pa, 1.0 / fs_hz, axis=-1)


def compute_lumped_capacitance(
    pressure: np.ndarray,
    flow: np.ndarray,
    p_min_bar: float = 130.0,
    dpdt_min_bar_s: float = 1.5,
) -> np.ndarray:
    """Compute lumped capacitance C = Q / (dp/dt) in m^3/Pa over active ramp mask."""
    p = np.atleast_2d(pressure)
    q = np.atleast_2d(flow)
    if p.shape[0] != q.shape[0]:
        raise ValueError(f"Cycle count mismatch: pressure {p.shape[0]} vs flow {q.shape[0]}.")

    q_up = _upsample_flow(q, p.shape[-1])
    q_m3_s = q_up * (1.0e-3 / 60.0)
    dp_dt = compute_pressure_derivative(p, fs_hz=100.0)

    mask = (p > p_min_bar) & (dp_dt >= dpdt_min_bar_s * 1.0e5)
    mask_counts = np.sum(mask, axis=-1)
    c_pointwise = np.divide(q_m3_s, dp_dt, out=np.zeros_like(q_m3_s), where=mask)
    sum_c = np.sum(c_pointwise, axis=-1)

    return np.where(mask_counts > 0, sum_c / np.maximum(mask_counts, 1), np.nan)


def accumulator_gas_compliance(
    p_pa: float | np.ndarray,
    p0_pa: float | np.ndarray,
    v0_m3: float = 0.002,
    n: float = 1.3,
) -> np.ndarray:
    """Compute gas compliance C_gas = dV/dp in m^3/Pa for an accumulator (0 when p <= p0)."""
    if v0_m3 <= 0 or n <= 0:
        raise ValueError("v0_m3 and n must be strictly positive.")
    p = np.asarray(p_pa, dtype=float)
    p0 = np.asarray(p0_pa, dtype=float)

    mask = (p > p0) & (p > 0.0)
    safe_p = np.where(mask, p, 1.0)
    safe_p0 = np.where(mask, p0, 0.0)

    return np.where(
        mask,
        v0_m3 * (safe_p0 ** (1.0 / n)) / (n * (safe_p ** (1.0 + 1.0 / n))),
        0.0,
    )


def fit_valve_baseline(
    pressure_train: np.ndarray,
    valve_train: np.ndarray,
    fs_hz: float = 100.0,
) -> float:
    """Fit median inflection time in s on healthy cycles (valve == 100). Train fold only."""
    p = np.atleast_2d(pressure_train)
    v = np.asarray(valve_train)
    if len(p) != len(v):
        raise ValueError(f"Length mismatch: pressure {len(p)} vs valve labels {len(v)}.")

    healthy_mask = v == 100
    if not np.any(healthy_mask):
        raise ValueError("No healthy valve cycles (valve == 100) found in training data.")

    t_inflect = _extract_inflection_times(p[healthy_mask], fs_hz)
    return float(np.median(t_inflect))


def compute_valve_edge_dynamics(
    pressure: np.ndarray,
    baseline_median_inflect: float,
    fs_hz: float = 100.0,
) -> pd.DataFrame:
    """Compute directional switching dynamics: delay (s), rise time (s), overshoot ratio."""
    p = np.atleast_2d(pressure)
    t_inflect = _extract_inflection_times(p, fs_hz)
    edge_delay = t_inflect - baseline_median_inflect

    base = np.percentile(p, 5, axis=-1, keepdims=True)
    plateau = np.percentile(p, 95, axis=-1, keepdims=True)
    idx_10 = np.argmax(p >= base + 0.10 * (plateau - base), axis=-1)
    idx_90 = np.argmax(p >= base + 0.90 * (plateau - base), axis=-1)
    rise_time = (idx_90 - idx_10) / fs_hz

    max_p = np.max(p, axis=-1, keepdims=True)
    overshoot = np.where(plateau > 0, (max_p / plateau) - 1.0, 0.0).squeeze(-1)

    return pd.DataFrame(
        {
            "valve_edge_delay": edge_delay,
            "rise_time_10_90": rise_time,
            "overshoot_ratio": overshoot,
        }
    )


def compute_thermal_effectiveness(
    temp_in: np.ndarray,
    temp_out: np.ndarray,
    window_size: int = 5,
    t_ref_c: float = 20.0,
) -> np.ndarray:
    """Compute cooler thermal effectiveness index in [0, 1] with trailing rolling mean."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    t_in = np.asarray(temp_in, dtype=float)
    t_out = np.asarray(temp_out, dtype=float)
    if t_in.shape != t_out.shape:
        raise ValueError(f"Shape mismatch: temp_in {t_in.shape} vs temp_out {t_out.shape}.")

    denom = np.maximum(t_in - t_ref_c, 1.0)
    ratio = (t_in - t_out) / denom
    clipped = np.clip(ratio, 0.0, 1.0)

    cycle_eff = np.mean(clipped, axis=-1) if t_in.ndim > 1 else clipped
    rolled = pd.Series(cycle_eff).rolling(window=window_size, min_periods=1).mean()
    return rolled.to_numpy()
