"""Tests for README documentation assets and CLI subcommand references."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from hydro.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_SVG_BYTES = 150 * 1024
MAX_MEDIA_BYTES = 10 * 1024 * 1024
MEDIA_EXTENSIONS = {".gif", ".webp", ".mp4"}


def extract_readme_image_paths(readme_text: str) -> list[str]:
    """Extract local image relative paths from markdown and HTML image tags."""
    md_pattern = r"!\[.*?\]\(([^)]+)\)"
    html_pattern = r'<img\s+[^>]*src=["\']?([^"\'\s>]+)["\']?'
    raw_paths: list[str] = []
    for match in re.findall(md_pattern, readme_text):
        raw_paths.append(match.strip().split()[0].strip("\"'"))
    for match in re.findall(html_pattern, readme_text):
        raw_paths.append(match.strip().split()[0].strip("\"'"))
    return [p for p in raw_paths if not p.startswith(("http://", "https://"))]


def extract_readme_targets(readme_text: str) -> list[str]:
    """Extract local href and src targets from markdown and HTML links and images."""
    patterns = (
        r"!\[.*?\]\(([^)]+)\)",
        r"\[.*?\]\(([^)]+)\)",
        r'<img\s+[^>]*src=["\']?([^"\'\s>]+)["\']?',
        r'<a\s+[^>]*href=["\']?([^"\'\s>]+)["\']?',
    )
    raw_targets: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, readme_text):
            clean = match.strip().split()[0].strip("\"'").split("#")[0]
            if clean and not clean.startswith(("http://", "https://", "mailto:")):
                raw_targets.append(clean)
    return sorted(set(raw_targets))


def extract_hydro_subcommands(readme_text: str) -> set[str]:
    """Extract subcommand names invoked via 'python -m hydro <cmd>'."""
    pattern = r"python\s+-m\s+hydro\s+([a-zA-Z0-9_-]+)"
    return set(re.findall(pattern, readme_text))


def get_cli_subcommands() -> set[str]:
    """Retrieve all subcommand choices defined in hydro CLI parser."""
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices.keys())
    return set()


def validate_svg_content(content: str, filename: str = "asset.svg") -> None:
    """Validate SVG XML syntax, size limit, unsafe tags, and external URLs."""
    raw_bytes = content.encode("utf-8")
    if len(raw_bytes) > MAX_SVG_BYTES:
        raise ValueError(f"{filename} exceeds 150 KB limit ({len(raw_bytes)} bytes)")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as err:
        raise ValueError(f"{filename} failed to parse as XML: {err}") from err

    _check_prohibited_elements(root, filename)
    _check_prohibited_hrefs(root, filename)


def _check_prohibited_elements(root: ET.Element, filename: str) -> None:
    """Ensure SVG contains no script or foreignObject elements."""
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag in {"script", "foreignobject"}:
            raise ValueError(f"{filename} contains prohibited element <{tag}>")


def _check_prohibited_hrefs(root: ET.Element, filename: str) -> None:
    """Ensure SVG attributes contain no external http or https hrefs."""
    for elem in root.iter():
        for key, val in elem.attrib.items():
            attr_name = key.split("}")[-1].lower()
            if attr_name == "href" and val.strip().lower().startswith(("http://", "https://")):
                raise ValueError(f"{filename} contains external href: {val}")


def test_readme_image_paths_exist() -> None:
    """Every local image path referenced in README.md must exist on disk."""
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.is_file(), "README.md does not exist"
    paths = extract_readme_image_paths(readme_path.read_text(encoding="utf-8"))
    assert len(paths) >= 3, f"Expected at least 3 SVG images in README, found {len(paths)}"
    for rel_path in paths:
        target = REPO_ROOT / rel_path
        if target.suffix.lower() in MEDIA_EXTENSIONS and not target.is_file():
            continue
        assert target.is_file(), f"Image referenced in README.md not found: {rel_path}"


def test_readme_href_src_targets_exist() -> None:
    """Every local href and src target referenced in README.md must exist on disk."""
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.is_file(), "README.md does not exist"
    targets = extract_readme_targets(readme_path.read_text(encoding="utf-8"))
    assert len(targets) >= 5, f"Expected at least 5 targets in README, found {len(targets)}"
    for rel_path in targets:
        target = REPO_ROOT / rel_path
        if target.suffix.lower() in MEDIA_EXTENSIONS and not target.is_file():
            continue
        assert target.is_file(), f"Target referenced in README.md not found: {rel_path}"


@pytest.mark.parametrize(
    "media_filename",
    ["explainer.gif", "explainer.webp", "explainer.mp4"],
)
def test_docs_assets_media_sizes(media_filename: str) -> None:
    """Verify explainer media files in docs/assets are under 10 MB limit."""
    media_path = REPO_ROOT / "docs" / "assets" / media_filename
    if not media_path.is_file():
        pytest.skip(f"Media asset not generated yet: {media_filename}")
    size_bytes = media_path.stat().st_size
    assert size_bytes <= MAX_MEDIA_BYTES, (
        f"{media_filename} exceeds 10 MB limit ({size_bytes} > {MAX_MEDIA_BYTES})"
    )


def test_extract_readme_targets_filtering() -> None:
    """extract_readme_targets captures local links/images and ignores external/anchor links."""
    sample = (
        "# Title\n"
        "[Local doc](docs/guide.md) and [External](https://example.com/api).\n"
        "![Local img](docs/img.svg) and [Anchor](#section)\n"
        '<a href="docs/video.mp4"><img src="docs/thumb.gif" alt="demo"></a>\n'
        '<a href="http://external.org">External HTML</a>\n'
    )
    targets = extract_readme_targets(sample)
    assert targets == ["docs/guide.md", "docs/img.svg", "docs/thumb.gif", "docs/video.mp4"]


def test_readme_explainer_section_present() -> None:
    """Verify README includes 37s explainer section, video embed, and regen command."""
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## The problem and how we solved it (37 s)" in readme_text
    assert "docs/assets/explainer.mp4" in readme_text
    assert "docs/assets/explainer.gif" in readme_text
    assert "Click for the full-quality MP4." in readme_text
    assert "python scripts/make_explainer_video.py" in readme_text


def test_docs_assets_svg_specs() -> None:
    """Every SVG file in docs/assets must satisfy XML, size, tag, and href rules."""
    assets_dir = REPO_ROOT / "docs" / "assets"
    if not assets_dir.is_dir():
        return
    svg_files = sorted(assets_dir.glob("*.svg"))
    for svg_path in svg_files:
        content = svg_path.read_text(encoding="utf-8")
        validate_svg_content(content, filename=svg_path.name)


def test_readme_hydro_subcommands() -> None:
    """Every 'python -m hydro <subcommand>' in README must exist in CLI parser."""
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.is_file(), "README.md does not exist"
    subcommands = extract_hydro_subcommands(readme_path.read_text(encoding="utf-8"))
    assert len(subcommands) > 0, "No 'python -m hydro' subcommands found in README"
    valid_subcommands = get_cli_subcommands()
    for cmd in subcommands:
        assert cmd in valid_subcommands, f"Subcommand '{cmd}' not found in hydro CLI parser"


def test_validate_svg_content_valid() -> None:
    """Valid SVG snippet passes validation without error."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="10" cy="10" r="5"/></svg>'
    validate_svg_content(svg, "valid.svg")


def test_validate_svg_prohibited_script() -> None:
    """SVG with script element raises ValueError."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    with pytest.raises(ValueError, match="prohibited element <script>"):
        validate_svg_content(svg, "script.svg")


def test_validate_svg_prohibited_foreign_object() -> None:
    """SVG with foreignObject element raises ValueError."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><foreignObject width="10"/></svg>'
    with pytest.raises(ValueError, match="prohibited element <foreignobject>"):
        validate_svg_content(svg, "fo.svg")


def test_validate_svg_prohibited_href() -> None:
    """SVG with external href raises ValueError."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<a xlink:href="https://example.com"><circle r="1"/></a></svg>'
    )
    with pytest.raises(ValueError, match="external href"):
        validate_svg_content(svg, "link.svg")


def test_validate_svg_oversized() -> None:
    """SVG exceeding 150 KB raises ValueError."""
    large_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg"><!-- ' + ("x" * (150 * 1024)) + " --></svg>"
    )
    with pytest.raises(ValueError, match="exceeds 150 KB"):
        validate_svg_content(large_svg, "oversized.svg")


def extract_readme_layout_paths(readme_text: str) -> list[str]:
    """Extract relative file and directory paths from Repository Layout block."""
    layout_match = re.search(
        r"## Repository Layout\s+```[^\n]*\n(.*?)```",
        readme_text,
        re.DOTALL,
    )
    if not layout_match:
        return []
    paths: list[str] = []
    current_parent = ""
    for raw_line in layout_match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("hydraulic-condition-twin"):
            continue
        code_part = line.split("#")[0].rstrip()
        if code_part.startswith(("├── ", "└── ")):
            name = code_part[4:].strip().rstrip("/")
            current_parent = name
            paths.append(name)
        elif "── " in code_part:
            name = code_part.split("── ")[-1].strip().rstrip("/")
            paths.append(f"{current_parent}/{name}")
    return paths


def parse_latency_p95(text: str) -> float:
    """Extract p95 latency in milliseconds from markdown text."""
    match = re.search(r"p95(?:[^\n|]*\|\s*|:\s*)([0-9.]+)\s*ms", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse p95 latency from text: {text[:100]}...")
    return float(match.group(1))


def test_extract_readme_layout_paths() -> None:
    """extract_readme_layout_paths correctly resolves nested tree entries."""
    sample = (
        "## Repository Layout\n\n```\nroot/\n├── dir_a/\n│   └── file_a.txt\n└── file_b.txt\n```\n"
    )
    paths = extract_readme_layout_paths(sample)
    assert paths == ["dir_a", "dir_a/file_a.txt", "file_b.txt"]


def test_parse_latency_p95() -> None:
    """parse_latency_p95 parses both list and table formats."""
    assert parse_latency_p95("- p95: 15.29 ms") == 15.29
    assert parse_latency_p95("| Latency (CPU p95) | 15.29 ms |") == 15.29


def test_readme_repository_layout_paths_exist() -> None:
    """Every path listed in README Repository Layout block must exist on disk."""
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.is_file(), "README.md does not exist"
    paths = extract_readme_layout_paths(readme_path.read_text(encoding="utf-8"))
    assert len(paths) >= 10, f"Expected at least 10 layout paths, found {len(paths)}"
    for rel_path in paths:
        target = REPO_ROOT / rel_path
        assert target.exists(), f"Path in Repository Layout does not exist: {rel_path}"


def test_readme_latency_p95_matches_report() -> None:
    """Verify p95 latency in README.md matches reports/latency.md exactly."""
    readme_path = REPO_ROOT / "README.md"
    latency_report_path = REPO_ROOT / "reports" / "latency.md"
    assert readme_path.is_file(), "README.md does not exist"
    assert latency_report_path.is_file(), "reports/latency.md does not exist"

    readme_text = readme_path.read_text(encoding="utf-8")
    report_text = latency_report_path.read_text(encoding="utf-8")

    readme_p95 = parse_latency_p95(readme_text)
    report_p95 = parse_latency_p95(report_text)
    assert readme_p95 == pytest.approx(report_p95), (
        f"README p95 ({readme_p95} ms) != reports/latency.md ({report_p95} ms)"
    )
