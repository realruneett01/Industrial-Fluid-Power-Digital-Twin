"""Deterministic, physically plausible synthetic cycles for self-checks (not training data)."""

from __future__ import annotations

import numpy as np

from hydro.data.schema import CHANNEL_REGISTRY, PHYSICAL_CHANNELS

# Nominal mean and noise per channel, chosen inside the plausibility limits of calibrate.py.
_NOMINAL: dict[str, tuple[float, float]] = {
    "PS1": (150.0, 0.5),
    "PS2": (110.0, 0.5),
    "PS3": (100.0, 0.5),
    "PS4": (90.0, 0.5),
    "PS5": (10.0, 0.1),
    "PS6": (9.0, 0.1),
    "EPS1": (2500.0, 20.0),
    "FS1": (6.0, 0.05),
    "FS2": (10.0, 0.05),
    "TS1": (45.0, 0.05),
    "TS2": (50.0, 0.05),
    "TS3": (55.0, 0.05),
    "TS4": (40.0, 0.05),
    "VS1": (0.6, 0.01),
}


def synthetic_cycle(seed: int = 0) -> dict[str, np.ndarray]:
    """One 60 s cycle, shape (1, n_samples) per physical channel, reproducible per seed."""
    rng = np.random.default_rng(seed)
    cycle: dict[str, np.ndarray] = {}
    for name in PHYSICAL_CHANNELS:
        mean, noise = _NOMINAL[name]
        n = CHANNEL_REGISTRY[name].n_samples
        cycle[name] = (mean + noise * rng.standard_normal(n)).reshape(1, n)
    ramp = np.linspace(0.0, 1.0, 6000)
    cycle["PS1"] = (100.0 + 100.0 * ramp + 0.5 * rng.standard_normal(6000)).reshape(1, -1)
    return cycle
