"""Dataset loader and interim Parquet storage for ZeMA hydraulic condition monitoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hydro.data.schema import CHANNEL_REGISTRY, TARGET_COLUMNS


def _read_channel_array(file_path: Path, expected_cols: int, channel_name: str) -> np.ndarray:
    """Read a tab-separated float32 channel file and validate its column width."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing channel file '{file_path.name}' at {file_path}")
    try:
        df = pd.read_csv(file_path, sep="\t", header=None, dtype=np.float32, engine="c")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, TypeError, OSError) as exc:
        msg = f"Error reading channel file '{file_path.name}' ({channel_name}): {exc}"
        raise ValueError(msg) from exc
    arr = df.to_numpy()
    if arr.ndim != 2 or arr.shape[1] != expected_cols:
        msg = (
            f"Channel file '{file_path.name}' ({channel_name}) has shape {arr.shape}, "
            f"expected (*, {expected_cols})"
        )
        raise ValueError(msg)
    return arr


def _read_targets(file_path: Path, expected_rows: int | None = None) -> pd.DataFrame:
    """Read profile.txt as int64 targets and validate dimensions and numeric format."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing target file '{file_path.name}' at {file_path}")
    try:
        df = pd.read_csv(file_path, sep="\t", header=None, engine="c")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, TypeError, OSError) as exc:
        msg = f"Error reading target file '{file_path.name}': {exc}"
        raise ValueError(msg) from exc
    expected_cols = len(TARGET_COLUMNS)
    if df.shape[1] != expected_cols:
        msg = f"Target file '{file_path.name}' has {df.shape[1]} columns, expected {expected_cols}"
        raise ValueError(msg)
    try:
        df = df.astype(np.int64)
    except (ValueError, TypeError) as exc:
        msg = f"Target file '{file_path.name}' contains non-numeric values: {exc}"
        raise ValueError(msg) from exc
    if expected_rows is not None and len(df) != expected_rows:
        msg = f"Target file '{file_path.name}' has {len(df)} rows, expected {expected_rows}"
        raise ValueError(msg)
    df.columns = list(TARGET_COLUMNS)
    return df


def load_cycle_dataset(
    data_dir: Path | str,
    exclude_virtual: bool = True,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load multi-rate hydraulic cycle arrays [bar, W, l/min, degC, mm/s] and targets."""
    base_dir = Path(data_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: '{base_dir}'")

    targets = _read_targets(base_dir / "profile.txt")
    expected_rows = len(targets)
    if expected_rows == 0:
        raise ValueError(f"Target file 'profile.txt' in '{base_dir}' is empty")

    data: dict[str, np.ndarray] = {}
    for name, ch in CHANNEL_REGISTRY.items():
        if exclude_virtual and ch.is_virtual:
            continue
        arr = _read_channel_array(base_dir / f"{name}.txt", ch.n_samples, name)
        if arr.shape[0] != expected_rows:
            msg = (
                f"Channel file '{name}.txt' has {arr.shape[0]} rows, "
                f"expected {expected_rows} (from profile.txt)"
            )
            raise ValueError(msg)
        data[name] = arr

    return data, targets


def _write_rate_parquet(
    data: dict[str, np.ndarray],
    rate_hz: int,
    out_file: Path,
) -> None:
    """Write fixed-size list array columns for all channels at a given sampling rate."""
    col_arrays: list[pa.FixedSizeListArray] = []
    col_names: list[str] = []
    for name, ch in CHANNEL_REGISTRY.items():
        if ch.rate_hz != rate_hz or name not in data:
            continue
        arr = np.asarray(data[name], dtype=np.float32)
        arrow_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(arr.ravel(), type=pa.float32()),
            ch.n_samples,
        )
        col_arrays.append(arrow_arr)
        col_names.append(name)
    table = pa.Table.from_arrays(col_arrays, names=col_names)
    pq.write_table(table, out_file)


def save_interim(
    data: dict[str, np.ndarray],
    targets: pd.DataFrame,
    out_dir: Path | str,
) -> list[Path]:
    """Persist multi-rate sensor cycles and targets into interim Parquet files."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for rate in (100, 10, 1):
        rate_path = dest / f"rate_{rate}hz.parquet"
        _write_rate_parquet(data, rate, rate_path)
        written.append(rate_path)

    targets_path = dest / "targets.parquet"
    targets.to_parquet(targets_path, index=False)
    written.append(targets_path)
    return written
