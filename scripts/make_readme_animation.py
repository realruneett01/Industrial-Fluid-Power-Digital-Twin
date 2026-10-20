"""Generate condition timeline SVG animation for ZeMA hydraulic test stand."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COOLER_COLORS: dict[int, str] = {
    100: "#22c55e",
    20: "#f59e0b",
    3: "#ef4444",
}
VALVE_COLORS: dict[int, str] = {
    100: "#22c55e",
    90: "#f59e0b",
    80: "#f97316",
    73: "#ef4444",
}
PUMP_COLORS: dict[int, str] = {
    0: "#22c55e",
    1: "#f59e0b",
    2: "#ef4444",
}
ACCUMULATOR_COLORS: dict[int, str] = {
    130: "#22c55e",
    115: "#f59e0b",
    100: "#f97316",
    90: "#ef4444",
}
STABLE_COLORS: dict[int, str] = {
    0: "#22c55e",
    1: "#ef4444",
}


@dataclass(frozen=True)
class TimelineLayout:
    """Dimensions and coordinate bounding boxes for condition timeline SVG [pixels]."""

    n_cycles: int
    width: int = 960
    height: int = 420
    plot_x: int = 175
    plot_w: int = 745
    top_y: int = 92
    lane_h: int = 32
    stable_h: int = 14
    lane_gap: int = 10


@dataclass(frozen=True)
class LaneSpec:
    """Specification and color thresholds for a condition monitor heat lane."""

    name: str
    label: str
    unit: str
    colors: dict[int, str]
    height: int


LANES: tuple[LaneSpec, ...] = (
    LaneSpec("cooler", "Cooler", "efficiency %", COOLER_COLORS, 32),
    LaneSpec("valve", "Valve", "switching %", VALVE_COLORS, 32),
    LaneSpec("pump", "Pump Leakage", "state 0..2", PUMP_COLORS, 32),
    LaneSpec("accumulator", "Accumulator", "pressure bar", ACCUMULATOR_COLORS, 32),
    LaneSpec("stable_flag", "Stable Flag", "0=stable 1=transient", STABLE_COLORS, 14),
)


def _resolve_color(val: int, mapping: dict[int, str]) -> str:
    """Return hex color for component state, with fallback to amber or boundary colors."""
    if val in mapping:
        return mapping[val]
    sorted_keys = sorted(mapping.keys())
    if val >= sorted_keys[-1]:
        return "#22c55e"
    if val <= sorted_keys[0]:
        return "#ef4444"
    return "#f59e0b"


def _run_length_encode(values: np.ndarray) -> list[tuple[int, int, int]]:
    """Compute run-length segments (start_idx, length, value) for 1D integer array."""
    if len(values) == 0:
        return []
    diffs = np.diff(values) != 0
    change_indices = np.where(diffs)[0] + 1
    split_points = np.concatenate([[0], change_indices, [len(values)]])
    runs: list[tuple[int, int, int]] = []
    for i in range(len(split_points) - 1):
        s = int(split_points[i])
        e = int(split_points[i + 1])
        runs.append((s, e - s, int(values[s])))
    return runs


def _render_header(layout: TimelineLayout) -> list[str]:
    """Render SVG envelope, accessible metadata, styles, and background panel."""
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {layout.width} {layout.height}" '
        f'width="100%" height="100%">',
        "  <title>2,205 real test-stand cycles - what the twin learns to diagnose</title>",
        "  <desc>Condition timeline showing degradation lanes across test bench cycles.</desc>",
        "  <style>",
        '    text { font-family: system-ui, -apple-system, "Segoe UI", '
        "Helvetica, Arial, sans-serif; }",
        "    @keyframes pulse-glow { 0%, 100% { opacity: 0.25; } 50% { opacity: 0.65; } }",
        "    .cursor-glow { animation: pulse-glow 2s ease-in-out infinite; }",
        "    @media (prefers-reduced-motion: reduce) {",
        "      * { animation: none !important; }",
        "      animate, animateTransform { dur: 0s !important; repeatCount: 0 !important; }",
        "    }",
        "  </style>",
        f'  <rect width="{layout.width}" height="{layout.height}" rx="12" fill="#0b1220" />',
        '  <text x="36" y="44" fill="#e2e8f0" font-size="18" font-weight="600">'
        "2,205 real test-stand cycles - what the twin learns to diagnose</text>",
        '  <text x="36" y="66" fill="#94a3b8" font-size="12">'
        "ZeMA hydraulic test bench · Multi-component physical wear · "
        "Real-time digital twin monitoring</text>",
    ]


def _render_legend(x_start: int, y: int) -> list[str]:
    """Render compact color legend indicators for condition states."""
    items = (
        ("#22c55e", "Healthy"),
        ("#f59e0b", "Warning"),
        ("#ef4444", "Fault"),
        ("#22d3ee", "Twin Scan"),
    )
    lines: list[str] = []
    curr_x = x_start
    for color, text in items:
        lines.append(f'  <circle cx="{curr_x}" cy="{y}" r="4" fill="{color}" />')
        lines.append(
            f'  <text x="{curr_x + 9}" y="{y + 4}" fill="#94a3b8" font-size="11">{text}</text>'
        )
        curr_x += 80
    return lines


def _render_lane_blocks(
    runs: list[tuple[int, int, int]],
    spec: LaneSpec,
    y: int,
    layout: TimelineLayout,
) -> list[str]:
    """Render run-length merged rectangles for a single lane."""
    elements: list[str] = []
    scale = layout.plot_w / max(layout.n_cycles, 1)
    for start, length, val in runs:
        rx = layout.plot_x + start * scale
        rw = max(length * scale, 0.6)
        color = _resolve_color(val, spec.colors)
        elements.append(
            f'  <rect class="condition-block" x="{rx:.2f}" y="{y}" '
            f'width="{rw:.2f}" height="{spec.height}" fill="{color}" />'
        )
    return elements


def _render_single_lane(
    spec: LaneSpec,
    y: int,
    series: np.ndarray,
    layout: TimelineLayout,
) -> list[str]:
    """Render label, background track, and wear blocks for one condition lane."""
    lines: list[str] = [
        f'  <text x="36" y="{y + 19}" fill="#e2e8f0" font-size="13" '
        f'font-weight="600">{spec.label}</text>',
        f'  <text x="36" y="{y + 29}" fill="#64748b" font-size="9">{spec.unit}</text>',
        f'  <rect class="lane-track" x="{layout.plot_x}" y="{y}" '
        f'width="{layout.plot_w}" height="{spec.height}" rx="3" fill="#111a2e" />',
    ]
    runs = _run_length_encode(series)
    lines.extend(_render_lane_blocks(runs, spec, y, layout))
    return lines


def _render_all_lanes(
    profile: np.ndarray,
    layout: TimelineLayout,
) -> tuple[list[str], int]:
    """Render all 5 heat lanes and return SVG elements with plot bottom coordinate [px]."""
    lines: list[str] = []
    curr_y = layout.top_y
    for i, spec in enumerate(LANES):
        col_data = profile[:, i] if i < profile.shape[1] else np.zeros(layout.n_cycles, dtype=int)
        lines.extend(_render_single_lane(spec, curr_y, col_data, layout))
        curr_y += spec.height + layout.lane_gap
    return lines, curr_y - layout.lane_gap


def _render_x_axis(layout: TimelineLayout, bottom_y: int) -> list[str]:
    """Render cycle ticks and vertical grid guidelines across all lanes."""
    lines: list[str] = [
        f'  <line x1="{layout.plot_x}" y1="{bottom_y + 10}" '
        f'x2="{layout.plot_x + layout.plot_w}" y2="{bottom_y + 10}" '
        'stroke="#334155" stroke-width="1" />'
    ]
    step = 250
    tick_values = sorted({*range(0, layout.n_cycles, step), layout.n_cycles})
    scale = layout.plot_w / max(layout.n_cycles, 1)
    for c in tick_values:
        tx = layout.plot_x + c * scale
        lines.append(
            f'  <line x1="{tx:.1f}" y1="{layout.top_y}" x2="{tx:.1f}" y2="{bottom_y}" '
            f'stroke="#1e293b" stroke-width="1" stroke-dasharray="2,3" />'
        )
        lines.append(
            f'  <line x1="{tx:.1f}" y1="{bottom_y + 10}" x2="{tx:.1f}" y2="{bottom_y + 15}" '
            f'stroke="#475569" stroke-width="1" />'
        )
        lbl = f"{c}" if c != layout.n_cycles else f"{c} cyc"
        lines.append(
            f'  <text x="{tx:.1f}" y="{bottom_y + 27}" fill="#94a3b8" '
            f'font-size="10" text-anchor="middle">{lbl}</text>'
        )
    return lines


def _anim_x(attr: str, start: float, end: float, dur: str = "12s") -> str:
    """Generate SMIL x/x1/x2/cx linear motion animation element."""
    return (
        f'    <animate attributeName="{attr}" from="{start}" to="{end}" '
        f'dur="{dur}" repeatCount="indefinite" />'
    )


def _render_readout_bands(layout: TimelineLayout, y_text: int) -> list[str]:
    """Render 4 animated cycle band text readouts synchronized across the 12s scan."""
    lines: list[str] = []
    q = max(layout.n_cycles // 4, 1)
    plot_end = layout.plot_x + layout.plot_w
    bands = (
        (f"Cycles 1 - {q}", 'values="1;1;0;0"', 'keyTimes="0;0.249;0.250;1"'),
        (
            f"Cycles {q + 1} - {2 * q}",
            'values="0;0;1;1;0;0"',
            'keyTimes="0;0.249;0.250;0.499;0.500;1"',
        ),
        (
            f"Cycles {2 * q + 1} - {3 * q}",
            'values="0;0;1;1;0;0"',
            'keyTimes="0;0.499;0.500;0.749;0.750;1"',
        ),
        (
            f"Cycles {3 * q + 1} - {layout.n_cycles}",
            'values="0;0;1;1"',
            'keyTimes="0;0.749;0.750;1"',
        ),
    )
    for text_val, val_attr, key_attr in bands:
        lines.extend(
            [
                f'  <text x="{layout.plot_x}" y="{y_text}" fill="#22d3ee" font-size="11" '
                'font-weight="600" text-anchor="middle">',
                _anim_x("x", layout.plot_x, plot_end),
                f'    <animate attributeName="opacity" {val_attr}\n'
                f'      {key_attr} dur="12s" repeatCount="indefinite" />',
                f"    {text_val}",
                "  </text>",
            ]
        )
    return lines


def _render_cursor_and_readout(layout: TimelineLayout, bottom_y: int) -> list[str]:
    """Render vertical scanning cursor, glow aura, and moving cycle readout box."""
    plot_end = layout.plot_x + layout.plot_w
    box_w = 124
    box_h = 24
    box_y = bottom_y + 40
    lines: list[str] = [
        f'  <line class="cursor-glow" x1="{layout.plot_x}" y1="{layout.top_y - 4}" '
        f'x2="{layout.plot_x}" y2="{bottom_y + 4}" stroke="#22d3ee" stroke-width="6" '
        'opacity="0.3">',
        _anim_x("x1", layout.plot_x, plot_end),
        _anim_x("x2", layout.plot_x, plot_end),
        "  </line>",
        f'  <line class="cursor-line" x1="{layout.plot_x}" y1="{layout.top_y - 4}" '
        f'x2="{layout.plot_x}" y2="{bottom_y + 4}" stroke="#22d3ee" stroke-width="2">',
        _anim_x("x1", layout.plot_x, plot_end),
        _anim_x("x2", layout.plot_x, plot_end),
        "  </line>",
        f'  <circle cx="{layout.plot_x}" cy="{layout.top_y - 4}" r="3.5" fill="#22d3ee">',
        _anim_x("cx", layout.plot_x, plot_end),
        "  </circle>",
        f'  <rect x="{layout.plot_x - box_w // 2}" y="{box_y}" width="{box_w}" height="{box_h}" '
        'rx="4" fill="#111a2e" stroke="#22d3ee" stroke-width="1.2">',
        _anim_x("x", layout.plot_x - box_w // 2, plot_end - box_w // 2),
        "  </rect>",
    ]
    lines.extend(_render_readout_bands(layout, box_y + 16))
    return lines


def _load_profile_array(profile_path: Path) -> np.ndarray:
    """Load profile.txt targets array [cycles, 5] and validate shape."""
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile dataset not found at {profile_path}")
    arr = np.loadtxt(profile_path, delimiter="\t", dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"Invalid profile shape {arr.shape}, expected (*, 5)")
    return arr


def render_condition_timeline(profile_path: Path | str) -> str:
    """Render complete SVG string for hydraulic condition timeline from profile.txt."""
    path = Path(profile_path)
    data = _load_profile_array(path)
    layout = TimelineLayout(n_cycles=len(data))

    parts: list[str] = []
    parts.extend(_render_header(layout))
    parts.extend(_render_legend(620, 40))
    lane_lines, bottom_y = _render_all_lanes(data, layout)
    parts.extend(lane_lines)
    parts.extend(_render_x_axis(layout, bottom_y))
    parts.extend(_render_cursor_and_readout(layout, bottom_y))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _resolve_profile_path(raw_arg: str | None, repo_root: Path) -> Path:
    """Resolve profile path from CLI arg or the repo's data/raw/profile.txt."""
    if raw_arg is not None:
        p = Path(raw_arg)
        return p if p.is_absolute() else (repo_root / p)
    return repo_root / "data" / "raw" / "profile.txt"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint to generate condition timeline SVG."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate hydraulic condition timeline SVG.")
    parser.add_argument("--profile", type=str, default=None, help="Path to profile.txt")
    parser.add_argument("--out", type=str, default=None, help="Path to output SVG")
    args = parser.parse_args(argv)

    profile_path = _resolve_profile_path(args.profile, repo_root)
    out_default = repo_root / "docs" / "assets" / "condition_timeline.svg"
    out_path = Path(args.out) if args.out else out_default
    out_path = out_path if out_path.is_absolute() else (repo_root / out_path)

    svg_content = render_condition_timeline(profile_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg_content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
