"""Tests for condition timeline SVG generator."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from scripts.make_readme_animation import main, render_condition_timeline


def test_render_from_synthetic_profile(tmp_path: Path) -> None:
    """Validate timeline SVG rendered from synthetic profile data [cycles, severity]."""
    profile_file = tmp_path / "synthetic_profile.txt"
    out_svg = tmp_path / "timeline.svg"
    n_rows = 24
    half = n_rows // 2
    healthy_block = np.tile([100, 100, 0, 130, 0], (half, 1))
    fault_block = np.tile([3, 73, 2, 90, 1], (half, 1))
    synth = np.vstack([healthy_block, fault_block])
    np.savetxt(profile_file, synth, fmt="%d", delimiter="\t")

    exit_code = main(["--profile", str(profile_file), "--out", str(out_svg)])
    assert exit_code == 0
    assert out_svg.exists()

    tree = ET.parse(out_svg)
    root = tree.getroot()
    raw_content = out_svg.read_text(encoding="utf-8")

    assert "script" not in [elem.tag.lower() for elem in root.iter()]
    assert "<script" not in raw_content.lower()
    assert "foreignobject" not in raw_content.lower()
    assert 'href="http' not in raw_content.lower()

    lanes = [elem for elem in root.iter() if elem.attrib.get("class") == "lane-track"]
    assert len(lanes) == 5

    data_rects = [elem for elem in root.iter() if elem.attrib.get("class") == "condition-block"]
    assert 0 < len(data_rects) < n_rows

    animates = [elem for elem in root.iter() if "animate" in elem.tag.lower()]
    assert len(animates) > 0


def test_committed_svg_exists_and_size() -> None:
    """Validate committed condition_timeline.svg exists, is valid XML, and fits under 150 KB."""
    svg_path = Path(__file__).resolve().parents[1] / "docs" / "assets" / "condition_timeline.svg"
    assert svg_path.exists(), f"Committed asset missing: {svg_path}"

    size_bytes = svg_path.stat().st_size
    assert size_bytes < 150 * 1024, f"File size {size_bytes} exceeds 150 KB"

    tree = ET.parse(svg_path)
    root = tree.getroot()
    raw = svg_path.read_text(encoding="utf-8")

    assert "<script" not in raw.lower()
    assert "foreignobject" not in raw.lower()
    assert 'href="http' not in raw.lower()

    animates = [elem for elem in root.iter() if "animate" in elem.tag.lower()]
    assert len(animates) > 5


@pytest.mark.heavy
def test_real_dataset_render(raw_dir: Path, tmp_path: Path) -> None:
    """Validate timeline SVG generation on full 2,205 cycle dataset [cycles]."""
    out_svg = tmp_path / "from_raw.svg"
    profile_path = raw_dir / "profile.txt"
    svg_content = render_condition_timeline(profile_path)
    out_svg.write_text(svg_content, encoding="utf-8")

    assert out_svg.exists()
    tree = ET.parse(out_svg)
    assert tree.getroot() is not None
    assert out_svg.stat().st_size < 150 * 1024
