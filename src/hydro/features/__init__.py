"""Feature fusion, grey-box physics feature observer, and matrix construction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hydro.data.schema import VIRTUAL_CHANNELS
from hydro.features.physics import (
    compute_lumped_capacitance,
    compute_thermal_effectiveness,
    compute_valve_edge_dynamics,
)
from hydro.features.stats import extract_statistical_features
from hydro.features.viscosity import normalized_pump_leakage

logger = logging.getLogger(__name__)

PHYSICS_COLUMNS = (
    "c_eff",
    "k_leak_rel",
    "valve_edge_delay",
    "rise_time_10_90",
    "overshoot_ratio",
    "thermal_effectiveness",
    "ts1_mean",
)

REQUIRED_PHYSICS_CHANNELS = ("PS1", "FS1", "EPS1", "TS1", "TS3", "TS4")


def _validate_feature_inputs(data: dict[str, np.ndarray]) -> None:
    """Validate that virtual channels are excluded and required channels exist."""
    forbidden = [ch for ch in VIRTUAL_CHANNELS if ch in data]
    if forbidden:
        msg = f"Virtual channels {forbidden} forbidden in feature extraction"
        raise ValueError(msg)
    missing = [ch for ch in REQUIRED_PHYSICS_CHANNELS if ch not in data]
    if missing:
        msg = f"Missing channels required for physics features: {missing}"
        raise ValueError(msg)


def _clean_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +-inf with NaN, fill NaN with column median, and log filled counts."""
    cleaned = df.replace([np.inf, -np.inf], np.nan).copy()
    for col in cleaned.columns:
        n_nan = int(cleaned[col].isna().sum())
        if n_nan > 0:
            med = float(cleaned[col].median())
            fill_val = 0.0 if np.isnan(med) else med
            cleaned[col] = cleaned[col].fillna(fill_val)
            logging.warning(
                "Filled %d NaN/Inf cells in column '%s' with median %f",
                n_nan,
                col,
                fill_val,
            )
    return cleaned


def build_physics_features(
    data: dict[str, np.ndarray],
    valve_baseline_s: float,
) -> pd.DataFrame:
    """Extract 7 grey-box physical condition indicators across all cycles.

    Features: c_eff [m^3/Pa], k_leak_rel [m^6/kg], valve_edge_delay [s],
    rise_time_10_90 [s], overshoot_ratio [1], thermal_effectiveness [1],
    ts1_mean [degC]. The valve baseline is fitted on the training fold only.
    """
    _validate_feature_inputs(data)
    n_cycles = len(data["PS1"])

    c_eff = compute_lumped_capacitance(data["PS1"], data["FS1"])
    k_leak_rel = normalized_pump_leakage(data["EPS1"], data["PS1"], data["FS1"], data["TS1"])
    valve_df = compute_valve_edge_dynamics(data["PS1"], valve_baseline_s)
    thermal_eff = compute_thermal_effectiveness(data["TS3"], data["TS4"])
    ts1_mean = np.mean(data["TS1"], axis=-1)

    features = {
        "c_eff": c_eff,
        "k_leak_rel": k_leak_rel,
        "valve_edge_delay": valve_df["valve_edge_delay"].to_numpy(),
        "rise_time_10_90": valve_df["rise_time_10_90"].to_numpy(),
        "overshoot_ratio": valve_df["overshoot_ratio"].to_numpy(),
        "thermal_effectiveness": thermal_eff,
        "ts1_mean": ts1_mean,
    }
    return pd.DataFrame(features, index=pd.RangeIndex(n_cycles))


def build_feature_matrices(
    data: dict[str, np.ndarray],
    valve_baseline_s: float,
) -> dict[str, pd.DataFrame]:
    """Build unscaled statistical, physics, and hybrid feature matrices.

    Replaces +-inf with NaN, fills NaN with column median, and logs warning
    for filled cells per column. Virtual channels (CE, CP, SE) raise ValueError.
    """
    _validate_feature_inputs(data)
    df_stat = extract_statistical_features(data)
    df_phys = build_physics_features(data, valve_baseline_s)

    clean_stat = _clean_matrix(df_stat)
    clean_phys = _clean_matrix(df_phys)
    clean_hybrid = pd.concat([clean_stat, clean_phys], axis=1)

    return {
        "statistical": clean_stat,
        "physics": clean_phys,
        "hybrid": clean_hybrid,
    }


def save_feature_matrices(
    mats: dict[str, pd.DataFrame],
    out_dir: Path | str,
) -> list[Path]:
    """Persist feature matrices as Parquet files into target directory.

    Writes features_statistical.parquet, features_physics.parquet, and
    features_hybrid.parquet. Returns list of written file paths.
    """
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    file_map = (
        ("statistical", "features_statistical.parquet"),
        ("physics", "features_physics.parquet"),
        ("hybrid", "features_hybrid.parquet"),
    )
    for key, filename in file_map:
        path = dest / filename
        mats[key].to_parquet(path)
        written.append(path)
    return written


__all__ = [
    "PHYSICS_COLUMNS",
    "build_feature_matrices",
    "build_physics_features",
    "save_feature_matrices",
]
