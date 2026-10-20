"""Multi-rate channel registry for the ZeMA hydraulic test stand (UCI 447)."""

from __future__ import annotations

from dataclasses import dataclass

CYCLE_SECONDS = 60
N_CYCLES = 2205
TARGET_COLUMNS = ("cooler", "valve", "pump", "accumulator", "stable_flag")


@dataclass(frozen=True)
class ChannelSchema:
    name: str
    rate_hz: int
    unit: str
    is_virtual: bool = False

    @property
    def n_samples(self) -> int:
        return self.rate_hz * CYCLE_SECONDS


def _channels(names: list[str], rate_hz: int, unit: str, virtual: bool = False):
    return {n: ChannelSchema(n, rate_hz, unit, virtual) for n in names}


CHANNEL_REGISTRY: dict[str, ChannelSchema] = {
    **_channels([f"PS{i}" for i in range(1, 7)], 100, "bar"),
    **_channels(["EPS1"], 100, "W"),
    **_channels(["FS1", "FS2"], 10, "l/min"),
    **_channels([f"TS{i}" for i in range(1, 5)], 1, "degC"),
    **_channels(["VS1"], 1, "mm/s"),
    **_channels(["CE"], 1, "%", virtual=True),
    **_channels(["CP"], 1, "kW", virtual=True),
    **_channels(["SE"], 1, "%", virtual=True),
}

PHYSICAL_CHANNELS = tuple(n for n, c in CHANNEL_REGISTRY.items() if not c.is_virtual)
VIRTUAL_CHANNELS = tuple(n for n, c in CHANNEL_REGISTRY.items() if c.is_virtual)
