"""Report generation utilities for hydraulic condition monitoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _format_cell(val: object, float_fmt: str) -> str:
    """Format an individual table cell value according to its type."""
    if isinstance(val, (float, np.floating)):
        return f"{val:{float_fmt}}"
    return str(val)


def _format_row(row: tuple[object, ...], include_index: bool, float_fmt: str) -> list[str]:
    """Format row values into string cells, preserving index as string if present."""
    if include_index:
        return [str(row[0])] + [_format_cell(val, float_fmt) for val in row[1:]]
    return [_format_cell(val, float_fmt) for val in row]


def generate_markdown_table(results: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Generate a GitHub pipe table from a DataFrame, omitting default RangeIndex.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame to format as a Markdown table.
    float_fmt : str, default='.4f'
        Format specifier for floating-point values.

    Returns
    -------
    str
        Markdown string containing the GitHub pipe table.
    """
    is_default_range = isinstance(results.index, pd.RangeIndex) and results.index.name is None
    include_index = not is_default_range

    headers = [str(col) for col in results.columns]
    if include_index:
        index_name = str(results.index.name) if results.index.name else "index"
        headers.insert(0, index_name)

    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]
    for row in results.itertuples(index=include_index, name=None):
        lines.append(f"| {' | '.join(_format_row(row, include_index, float_fmt))} |")
    return "\n".join(lines)


def write_report(
    path: Path | str,
    title: str,
    sections: dict[str, str],
) -> Path:
    """Write a markdown report with an H1 title and an H2 per section.

    Parameters
    ----------
    path : Path | str
        Output file path for the markdown report.
    title : str
        Report title rendered as an H1 heading.
    sections : dict[str, str]
        Mapping from section headings to section content.

    Returns
    -------
    Path
        Path object pointing to the written report file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts = [f"# {title}"]
    for heading, content in sections.items():
        parts.append(f"## {heading}\n\n{content.strip()}")

    report_text = "\n\n".join(parts) + "\n"
    out_path.write_text(report_text, encoding="utf-8")
    return out_path
