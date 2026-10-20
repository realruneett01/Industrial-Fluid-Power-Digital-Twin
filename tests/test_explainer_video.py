"""Tests for explainer video generator and filtergraph builder."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.make_explainer_video import (
    DEFAULT_CAPTIONS,
    ExplainerConfig,
    Mp4Options,
    build_explainer_video,
    build_filtergraph,
    build_gif,
    build_mp4_command,
    build_webp,
    detect_h264_encoder,
    escape_drawtext,
    get_scene_paths,
    main,
    wrap_caption,
)

HAS_FFMPEG: bool = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

WRAP_CASES = (
    ("Short under 70 chars.", ["Short under 70 chars."]),
    ("A" * 70, ["A" * 70]),
    (
        "The problem: cooler, valve, pump and accumulator wear is invisible until failure.",
        ["The problem: cooler, valve, pump and", "accumulator wear is invisible until failure."],
    ),
    (
        "Verified: full test suite green, every file health >= 9.6, "
        "ONNX parity checked. Read-only by design.",
        [
            "Verified: full test suite green, every file health",
            ">= 9.6, ONNX parity checked. Read-only by design.",
        ],
    ),
)


def test_drawtext_escaping_colon_and_apostrophe() -> None:
    """Verify colon and apostrophe escaping for ffmpeg drawtext filter."""
    raw = "The problem: cooler, it's wear is invisible until failure."
    escaped = escape_drawtext(raw)
    assert r"\:" in escaped
    assert r"'\''" in escaped
    assert "cooler" in escaped


@pytest.mark.parametrize("n_scenes", [1, 3, 8])
def test_filtergraph_builder_segments(n_scenes: int, tmp_path: Path) -> None:
    """Verify filtergraph contains expected zoompan, xfade, and drawtext segments."""
    paths = [tmp_path / f"scene{i}.jpg" for i in range(1, n_scenes + 1)]
    config = ExplainerConfig(scenes_dir=tmp_path, out_dir=tmp_path)
    fg = build_filtergraph(paths, DEFAULT_CAPTIONS, config)

    assert fg.count("zoompan") == n_scenes
    assert fg.count("xfade") == n_scenes
    assert fg.count("drawtext") == n_scenes + 1


def test_missing_scene_error(tmp_path: Path) -> None:
    """Verify missing scene in sequence raises FileNotFoundError naming missing file."""
    (tmp_path / "scene1.jpg").write_text("data", encoding="utf-8")
    (tmp_path / "scene3.jpg").write_text("data", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="scene2.jpg"):
        get_scene_paths(tmp_path)


def test_empty_scene_dir_error(tmp_path: Path) -> None:
    """Verify empty scene directory raises FileNotFoundError naming scene1.jpg."""
    with pytest.raises(FileNotFoundError, match="scene1.jpg"):
        get_scene_paths(tmp_path)


@pytest.mark.parametrize(("caption", "expected_lines"), WRAP_CASES)
def test_wrap_caption_splits_long_captions(caption: str, expected_lines: list[str]) -> None:
    """Verify wrap_caption splits strings longer than 70 chars at nearest space."""
    assert wrap_caption(caption).split("\n") == expected_lines


def test_wrap_caption_no_spaces_fallback() -> None:
    """Verify wrap_caption returns unmodified string when no spaces exist."""
    no_spaces = "A" * 75
    assert wrap_caption(no_spaces) == no_spaces


def test_default_caption_8_content() -> None:
    """Verify caption 8 contains updated verification text."""
    expected = (
        "Verified: full test suite green, every file health >= 9.6, "
        "ONNX parity checked. Read-only by design."
    )
    assert DEFAULT_CAPTIONS[7] == expected


def test_caption_drawtext_styling(tmp_path: Path) -> None:
    """Verify filtergraph contains font size, box opacity 0.75, padding 14, and 24px margin."""
    paths = [tmp_path / "scene1.jpg"]
    config = ExplainerConfig(scenes_dir=tmp_path, out_dir=tmp_path, height=720)
    fg = build_filtergraph(paths, DEFAULT_CAPTIONS, config)

    assert "fontsize=29" in fg
    assert "boxcolor=black@0.75" in fg
    assert "boxborderw=14" in fg
    assert "text_align=center" in fg
    assert "y=h-th-24" in fg


def test_build_gif_stops_at_first_tier_under_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_gif stops at the first tier meeting max_bytes limit."""
    mp4, gif = tmp_path / "test.mp4", tmp_path / "test.gif"
    mp4.touch()
    gif.touch()

    tier_sizes = [10 * 1024 * 1024, 7 * 1024 * 1024, 5 * 1024 * 1024]
    call_count = 0

    def fake_render(mp4_in: Path, gif_out: Path, tier: object) -> None:
        nonlocal call_count
        call_count += 1
        fake_size = tier_sizes[call_count - 1]
        monkeypatch.setattr(Path, "stat", lambda self: type("Stat", (), {"st_size": fake_size})())

    monkeypatch.setattr("scripts.make_explainer_video.render_gif_tier", fake_render)
    result = build_gif(mp4, gif, max_bytes=8 * 1024 * 1024)

    assert result == gif
    assert call_count == 2


def test_build_gif_raises_when_all_tiers_exceed_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_gif raises RuntimeError when all tiers exceed max_bytes limit."""
    mp4, gif = tmp_path / "test.mp4", tmp_path / "test.gif"
    mp4.touch()
    gif.touch()

    monkeypatch.setattr("scripts.make_explainer_video.render_gif_tier", lambda *args: None)
    monkeypatch.setattr(
        Path,
        "stat",
        lambda self: type("Stat", (), {"st_size": 20 * 1024 * 1024})(),
    )

    with pytest.raises(RuntimeError, match="Could not generate GIF under"):
        build_gif(mp4, gif, max_bytes=8 * 1024 * 1024)


def test_build_webp_stops_at_first_tier_under_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_webp stops at first tier meeting max_bytes limit."""
    mp4, webp = tmp_path / "test.mp4", tmp_path / "test.webp"
    mp4.touch()
    webp.touch()

    tier_sizes = [10 * 1024 * 1024, 6 * 1024 * 1024]
    call_count = 0

    def fake_render(mp4_in: Path, webp_out: Path, tier: object) -> None:
        nonlocal call_count
        call_count += 1
        fake_size = tier_sizes[call_count - 1]
        monkeypatch.setattr(Path, "stat", lambda self: type("Stat", (), {"st_size": fake_size})())

    monkeypatch.setattr("scripts.make_explainer_video.render_webp_tier", fake_render)
    result = build_webp(mp4, webp, max_bytes=8 * 1024 * 1024)

    assert result == webp
    assert call_count == 2


def test_build_webp_raises_when_all_tiers_exceed_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_webp raises RuntimeError when all tiers exceed limit."""
    mp4, webp = tmp_path / "test.mp4", tmp_path / "test.webp"
    mp4.touch()
    webp.touch()

    monkeypatch.setattr("scripts.make_explainer_video.render_webp_tier", lambda *args: None)
    monkeypatch.setattr(
        Path,
        "stat",
        lambda self: type("Stat", (), {"st_size": 15 * 1024 * 1024})(),
    )

    with pytest.raises(RuntimeError, match="Could not generate WebP under"):
        build_webp(mp4, webp, max_bytes=8 * 1024 * 1024)


def test_build_explainer_video_requests_webp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_explainer_video invokes build_webp when skip_webp is False."""
    (tmp_path / "scene1.jpg").write_text("data", encoding="utf-8")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    webp_requested = False

    def fake_build_webp(mp4: Path, out_webp: Path, **kwargs: object) -> Path:
        nonlocal webp_requested
        webp_requested = True
        return out_webp

    monkeypatch.setattr("scripts.make_explainer_video.build_gif", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.make_explainer_video.build_webp", fake_build_webp)

    config = ExplainerConfig(scenes_dir=tmp_path, out_dir=tmp_path, skip_gif=True, skip_webp=False)
    _, _, out_webp = build_explainer_video(config)

    assert webp_requested is True
    assert out_webp == tmp_path / "explainer.webp"


def _generate_test_scenes(target_dir: Path, colors: tuple[str, ...]) -> None:
    """Generate minimal test scenes using ffmpeg color sources."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for idx, col in enumerate(colors, start=1):
        p = target_dir / f"scene{idx}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={col}:s=1280x720:d=0.1",
            "-vframes",
            "1",
            str(p),
        ]
        subprocess.run(cmd, check=True, capture_output=True)


def _probe_duration(media_path: Path) -> float:
    """Query media duration using ffprobe."""
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    probe = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    return float(probe.stdout.strip())


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg and ffprobe required on PATH")
def test_render_tiny_generated_scenes(tmp_path: Path) -> None:
    """Integration test: render 3 tiny generated scenes and verify duration."""
    scenes_dir, out_dir = tmp_path / "scenes", tmp_path / "out"
    _generate_test_scenes(scenes_dir, ("0x224488", "0x336699", "0x112244"))

    code = main(
        [
            "--scenes-dir",
            str(scenes_dir),
            "--out-dir",
            str(out_dir),
            "--seconds-per-scene",
            "1.0",
            "--skip-gif",
        ]
    )
    assert code == 0
    mp4_path = out_dir / "explainer.mp4"
    assert mp4_path.is_file()

    actual_dur = _probe_duration(mp4_path)
    # 1.5s title + 3 * 1.0s scenes - 3 * 0.6s xfades = 2.7s
    expected_dur = 1.5 + (3 * 1.0) - (3 * 0.6)
    rel_error = abs(actual_dur - expected_dur) / expected_dur
    assert rel_error <= 0.15


@pytest.mark.parametrize(
    ("encoder_output", "probe_code", "expected"),
    [
        (" V....D h264_nvenc NVIDIA NVENC H.264 encoder\n", 0, "h264_nvenc"),
        (" V....D libx264 libx264 H.264 encoder\n", 0, "libx264"),
        (" V....D h264_nvenc NVIDIA NVENC H.264 encoder\n", 1, "libx264"),
    ],
)
def test_detect_h264_encoder_scenarios(
    monkeypatch: pytest.MonkeyPatch,
    encoder_output: str,
    probe_code: int,
    expected: str,
) -> None:
    """Verify detect_h264_encoder selects nvenc only when listed and probe succeeds."""
    detect_h264_encoder.cache_clear()

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "-encoders" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=encoder_output, stderr="")
        if "h264_nvenc" in cmd:
            return subprocess.CompletedProcess(cmd, probe_code, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.make_explainer_video.subprocess.run", fake_run)
    try:
        assert detect_h264_encoder() == expected
    finally:
        detect_h264_encoder.cache_clear()


def test_detect_h264_encoder_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify detect_h264_encoder falls back to libx264 on process execution error."""
    detect_h264_encoder.cache_clear()

    def raise_oserror(*args: object, **kwargs: object) -> None:
        raise OSError("ffmpeg missing")

    monkeypatch.setattr("scripts.make_explainer_video.subprocess.run", raise_oserror)
    try:
        assert detect_h264_encoder() == "libx264"
    finally:
        detect_h264_encoder.cache_clear()


def test_build_mp4_command_nvenc_flags(tmp_path: Path) -> None:
    """Verify build_mp4_command configures h264_nvenc flags properly."""
    paths = [tmp_path / "scene1.jpg"]
    out_mp4 = tmp_path / "out.mp4"
    cmd = build_mp4_command(paths, "filtergraph", out_mp4, Mp4Options("h264_nvenc", 28))
    assert "-c:v" in cmd
    idx = cmd.index("-c:v")
    assert cmd[idx + 1] == "h264_nvenc"
    assert cmd[idx + 2 : idx + 8] == ["-preset", "p5", "-rc", "vbr", "-cq", "28"]
    assert "-pix_fmt" in cmd
    assert "-movflags" in cmd


def test_nvenc_reencodes_when_size_exceeds_10mb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify MP4 re-encodes with cq 32 if initial nvenc output exceeds 10 MB."""
    (tmp_path / "scene1.jpg").write_text("data", encoding="utf-8")
    out_mp4 = tmp_path / "explainer.mp4"
    out_mp4.touch()
    executed_cmds: list[list[str]] = []
    fake_sizes = [11 * 1024 * 1024, 7 * 1024 * 1024]
    orig_stat = Path.stat

    def fake_stat(self: Path, *args: object, **kwargs: object) -> object:
        if self == out_mp4:
            idx = min(len(executed_cmds) - 1, len(fake_sizes) - 1)
            return type("Stat", (), {"st_size": fake_sizes[idx], "st_mode": 0o100644})()
        return orig_stat(self, *args, **kwargs)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        executed_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.make_explainer_video.detect_h264_encoder", lambda: "h264_nvenc")
    monkeypatch.setattr("scripts.make_explainer_video.subprocess.run", fake_run)
    monkeypatch.setattr(Path, "stat", fake_stat)
    config = ExplainerConfig(scenes_dir=tmp_path, out_dir=tmp_path, skip_gif=True, skip_webp=True)
    build_explainer_video(config)

    assert len(executed_cmds) == 2
    assert executed_cmds[0][executed_cmds[0].index("-cq") + 1] == "28"
    assert executed_cmds[1][executed_cmds[1].index("-cq") + 1] == "32"
