"""Build explainer video, GIF, and animated WebP from condition monitoring scene images."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_MP4_BYTES: int = 10 * 1024 * 1024

_PROBE_NVENC_CMD: tuple[str, ...] = (
    "ffmpeg",
    "-y",
    "-hide_banner",
    "-f",
    "lavfi",
    "-i",
    "color=s=256x256",
    "-frames:v",
    "1",
    "-c:v",
    "h264_nvenc",
    "-f",
    "null",
    "-",
)

DEFAULT_CAPTIONS: tuple[str, ...] = (
    "Hydraulic systems fail silently. ZeMA test stand, 2,205 real cycles.",
    "The problem: cooler, valve, pump and accumulator wear is invisible until failure.",
    "17 sensor channels at 100 / 10 / 1 Hz - too much raw data to read by eye.",
    "Our fix, part 1: grey-box physics features - viscosity, capacitance, leakage, valve edge.",
    "Part 2: CatBoost models exported to ONNX run on an edge box over read-only, encrypted OPC-UA.",
    (
        "Part 3: a Health Index per cycle - p95 inference 15.3 ms. "
        "Held-out degraded cycles score about 0.26."
    ),
    "Below the threshold, a work order goes to ERPNext automatically.",
    (
        "Verified: full test suite green, every file health >= 9.6, "
        "ONNX parity checked. Read-only by design."
    ),
)


@dataclass(frozen=True)
class GifTier:
    """Configuration settings for GIF rendering tier."""

    width: int
    fps: int
    max_colors: int


@dataclass(frozen=True)
class Mp4Options:
    """Configuration options for MP4 video encoding."""

    encoder: str = "libx264"
    quality: int = 26


@dataclass(frozen=True)
class WebpTier:
    """Configuration settings for animated WebP rendering tier."""

    width: int
    fps: int
    quality: int


GIF_TIERS: tuple[GifTier, ...] = (
    GifTier(560, 8, 128),
    GifTier(560, 8, 96),
    GifTier(560, 8, 64),
    GifTier(480, 8, 96),
    GifTier(480, 6, 64),
    GifTier(400, 6, 64),
    GifTier(320, 6, 64),
)

WEBP_TIERS: tuple[WebpTier, ...] = (
    WebpTier(640, 10, 60),
    WebpTier(560, 8, 60),
    WebpTier(480, 8, 50),
    WebpTier(400, 6, 50),
)


@dataclass(frozen=True)
class ExplainerConfig:
    """Configuration settings for building explainer video."""

    scenes_dir: Path
    out_dir: Path
    seconds_per_scene: float = 5.0
    skip_gif: bool = False
    skip_webp: bool = False
    fps: int = 30
    width: int = 1280
    height: int = 720
    crf: int = 26
    xfade_duration: float = 0.6
    title_duration: float = 1.5
    title_text: str = "Hydraulic Condition Monitoring Digital Twin"


def escape_drawtext(text: str) -> str:
    """Escape colons and apostrophes for ffmpeg drawtext filter."""
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"'\''")
    return escaped


def format_font_path(font_path: Path) -> str:
    """Format font file path for ffmpeg drawtext option."""
    return font_path.as_posix().replace(":", r"\:")


def get_default_font_path() -> Path | None:
    """Return system Segoe UI font path if available, else None."""
    windir = os.environ.get("WINDIR", "C:/Windows")
    candidate = Path(windir) / "Fonts" / "segoeui.ttf"
    if candidate.is_file():
        return candidate
    fallback = Path("C:/Windows/Fonts/segoeui.ttf")
    if fallback.is_file():
        return fallback
    return None


def get_zoompan_expr(idx: int, n_frames: int, fps: int = 30) -> str:
    """Return ffmpeg zoompan filter string with alternating Ken Burns motions."""
    mode = idx % 4
    if mode == 0:
        return (
            f"zoompan=z='min(zoom+0.0015,1.25)':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d={n_frames}:s=1280x720:fps={fps}"
        )
    if mode == 1:
        return (
            f"zoompan=z='1.15':x='(on/{n_frames})*(iw-iw/zoom)':"
            f"y='ih/2-(ih/zoom/2)':d={n_frames}:s=1280x720:fps={fps}"
        )
    if mode == 2:
        return (
            f"zoompan=z='if(lte(on,1),1.20,max(1.0,zoom-0.0015))':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d={n_frames}:s=1280x720:fps={fps}"
        )
    return (
        f"zoompan=z='1.15':x='(1-on/{n_frames})*(iw-iw/zoom)':"
        f"y='ih/2-(ih/zoom/2)':d={n_frames}:s=1280x720:fps={fps}"
    )


def _find_max_scene_idx(scenes_dir: Path) -> int:
    """Find maximum scene index from scene*.jpg files in directory."""
    found = list(scenes_dir.glob("scene*.jpg"))
    if not found:
        raise FileNotFoundError(f"Missing scene file: {scenes_dir / 'scene1.jpg'}")
    indices = [
        int(p.stem.removeprefix("scene")) for p in found if p.stem.removeprefix("scene").isdigit()
    ]
    if not indices:
        raise FileNotFoundError(f"Missing scene file: {scenes_dir / 'scene1.jpg'}")
    return max(indices)


def get_scene_paths(scenes_dir: Path, n_scenes: int | None = None) -> list[Path]:
    """Collect and validate scene image paths in sequential order."""
    if not scenes_dir.exists():
        raise FileNotFoundError(f"Missing scene directory: {scenes_dir}")
    count = n_scenes if n_scenes is not None else _find_max_scene_idx(scenes_dir)
    paths: list[Path] = []
    for i in range(1, count + 1):
        expected_file = scenes_dir / f"scene{i}.jpg"
        if not expected_file.is_file():
            raise FileNotFoundError(f"Missing scene file: {expected_file}")
        paths.append(expected_file)
    return paths


def wrap_caption(text: str, max_chars: int = 70) -> str:
    """Wrap caption longer than max_chars onto two lines at nearest space."""
    if len(text) <= max_chars:
        return text
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return text
    mid = len(text) // 2
    split_idx = min(spaces, key=lambda i: abs(i - mid))
    return f"{text[:split_idx]}\n{text[split_idx + 1 :]}"


def _build_title_segment(config: ExplainerConfig, font_opt: str) -> str:
    """Build initial title card video generator segment."""
    txt = escape_drawtext(config.title_text)
    return (
        f"color=c=0x0a192f:s={config.width}x{config.height}:d={config.title_duration}:r={config.fps},"
        f"drawtext={font_opt}text='{txt}':fontcolor=white:fontsize=36:"
        f"x=(w-text_w)/2:y=(h-text_h)/2[title]"
    )


def _build_scene_segments(
    scene_paths: list[Path],
    captions: tuple[str, ...],
    config: ExplainerConfig,
    font_opt: str,
) -> list[str]:
    """Build zoompan and drawtext caption segments for each scene."""
    n_frames = round(config.seconds_per_scene * config.fps)
    fontsize = round(config.height * 0.04)
    segments: list[str] = []
    for i in range(len(scene_paths)):
        zp = get_zoompan_expr(i, n_frames, config.fps)
        raw_cap = captions[i] if i < len(captions) else f"Scene {i + 1}"
        cap = wrap_caption(raw_cap)
        cap_text = escape_drawtext(cap)
        seg = (
            f"[{i}:v]{zp},"
            f"drawtext={font_opt}text='{cap_text}':fontcolor=white:fontsize={fontsize}:"
            f"box=1:boxcolor=black@0.75:boxborderw=14:text_align=center:"
            f"x=(w-text_w)/2:y=h-th-24[s{i}]"
        )
        segments.append(seg)
    return segments


def _build_xfade_segments(n_scenes: int, config: ExplainerConfig) -> list[str]:
    """Build chained crossfade transitions between title card and scenes."""
    segments: list[str] = []
    prev_out = "title"
    offset = config.title_duration - config.xfade_duration
    dur_accum = config.title_duration
    for i in range(n_scenes):
        out_node = f"x{i}" if i < n_scenes - 1 else "outv"
        seg = (
            f"[{prev_out}][s{i}]xfade=transition=fade:"
            f"duration={config.xfade_duration}:offset={offset:.2f}[{out_node}]"
        )
        segments.append(seg)
        prev_out = out_node
        dur_accum = dur_accum + config.seconds_per_scene - config.xfade_duration
        offset = dur_accum - config.xfade_duration
    return segments


def build_filtergraph(
    scene_paths: list[Path],
    captions: tuple[str, ...],
    config: ExplainerConfig,
    font_path: Path | None = None,
) -> str:
    """Build complete filtergraph for explainer video."""
    font_opt = f"fontfile='{format_font_path(font_path)}':" if font_path else ""
    title_seg = _build_title_segment(config, font_opt)
    scene_segs = _build_scene_segments(scene_paths, captions, config, font_opt)
    xfade_segs = _build_xfade_segments(len(scene_paths), config)
    all_segs = [title_seg, *scene_segs, *xfade_segs]
    return ";".join(all_segs)


@lru_cache(maxsize=1)
def detect_h264_encoder() -> str:
    """Detect if h264_nvenc is available and working, falling back to libx264."""
    try:
        enc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        if not enc or enc.returncode != 0:
            return "libx264"
        if "h264_nvenc" not in (enc.stdout or ""):
            return "libx264"
        probe = subprocess.run(list(_PROBE_NVENC_CMD), capture_output=True, text=True, check=False)
        if probe and probe.returncode == 0:
            return "h264_nvenc"
    except OSError:
        # ffmpeg executable missing or inaccessible
        pass
    return "libx264"


def build_mp4_command(
    scene_paths: list[Path],
    filtergraph: str,
    out_mp4: Path,
    options: Mp4Options | int = 26,
) -> list[str]:
    """Assemble ffmpeg CLI arguments list for MP4 rendering."""
    opts = Mp4Options("libx264", options) if isinstance(options, int) else options
    cmd: list[str] = ["ffmpeg", "-y"]
    for path in scene_paths:
        cmd.extend(["-i", str(path)])
    cmd.extend(["-filter_complex", filtergraph, "-map", "[outv]", "-c:v", opts.encoder])
    if opts.encoder == "h264_nvenc":
        cmd.extend(["-preset", "p5", "-rc", "vbr", "-cq", str(opts.quality)])
    else:
        cmd.extend(["-crf", str(opts.quality)])
    cmd.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)])
    return cmd


def render_gif_tier(
    mp4_path: Path,
    out_gif: Path,
    tier: GifTier,
) -> None:
    """Render GIF using palettegen (stats_mode=diff) and paletteuse (bayer dither)."""
    with tempfile.TemporaryDirectory() as td:
        pal_path = Path(td) / "palette.png"
        filt = f"fps={tier.fps},scale={tier.width}:-1:flags=lanczos"
        cmd_pal = [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            f"{filt},palettegen=stats_mode=diff:max_colors={tier.max_colors}",
            str(pal_path),
        ]
        subprocess.run(cmd_pal, check=True, capture_output=True)

        cmd_gif = [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-i",
            str(pal_path),
            "-filter_complex",
            f"[0:v]{filt}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
            str(out_gif),
        ]
        subprocess.run(cmd_gif, check=True, capture_output=True)


def build_gif(
    mp4_path: Path,
    out_gif: Path,
    max_bytes: int = 9 * 1024 * 1024,
) -> Path:
    """Build GIF meeting target size by looping over tiers, raising on failure."""
    for tier in GIF_TIERS:
        render_gif_tier(mp4_path, out_gif, tier)
        if out_gif.stat().st_size <= max_bytes:
            return out_gif
    raise RuntimeError(
        f"Could not generate GIF under {max_bytes} bytes; smallest tier exceeded limit"
    )


def render_webp_tier(
    mp4_path: Path,
    out_webp: Path,
    tier: WebpTier,
) -> None:
    """Render animated WebP using libwebp encoder with looping enabled."""
    filt = f"fps={tier.fps},scale={tier.width}:-1:flags=lanczos"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-c:v",
        "libwebp",
        "-loop",
        "0",
        "-quality",
        str(tier.quality),
        "-vf",
        filt,
        str(out_webp),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def build_webp(
    mp4_path: Path,
    out_webp: Path,
    max_bytes: int = 8 * 1024 * 1024,
) -> Path:
    """Build animated WebP meeting target size by looping over tiers."""
    for tier in WEBP_TIERS:
        render_webp_tier(mp4_path, out_webp, tier)
        if out_webp.stat().st_size <= max_bytes:
            return out_webp
    raise RuntimeError(
        f"Could not generate WebP under {max_bytes} bytes; smallest tier exceeded limit"
    )


def _render_mp4(
    scene_paths: list[Path],
    filtergraph: str,
    out_mp4: Path,
    config: ExplainerConfig,
) -> None:
    """Render MP4 with h264_nvenc (re-encoding if >10MB) or fallback to libx264."""
    encoder = detect_h264_encoder()
    logger.info("Using H.264 video encoder: %s", encoder)
    if encoder == "h264_nvenc":
        cmd = build_mp4_command(scene_paths, filtergraph, out_mp4, Mp4Options("h264_nvenc", 28))
        subprocess.run(cmd, check=True, capture_output=True)
        if out_mp4.is_file() and out_mp4.stat().st_size > MAX_MP4_BYTES:
            logger.info("NVENC output exceeds limit; re-encoding with cq 32")
            cmd = build_mp4_command(scene_paths, filtergraph, out_mp4, Mp4Options("h264_nvenc", 32))
            subprocess.run(cmd, check=True, capture_output=True)
        return
    cmd = build_mp4_command(scene_paths, filtergraph, out_mp4, Mp4Options("libx264", config.crf))
    subprocess.run(cmd, check=True, capture_output=True)


def build_explainer_video(
    config: ExplainerConfig,
) -> tuple[Path, Path | None, Path | None]:
    """Build MP4 explainer video, optional GIF and WebP from scene images."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    scene_paths = get_scene_paths(config.scenes_dir)
    font_path = get_default_font_path()
    filtergraph = build_filtergraph(scene_paths, DEFAULT_CAPTIONS, config, font_path)
    out_mp4 = config.out_dir / "explainer.mp4"
    _render_mp4(scene_paths, filtergraph, out_mp4, config)

    out_gif: Path | None = None
    if not config.skip_gif:
        out_gif = config.out_dir / "explainer.gif"
        build_gif(out_mp4, out_gif)

    out_webp: Path | None = None
    if not config.skip_webp:
        out_webp = config.out_dir / "explainer.webp"
        build_webp(out_mp4, out_webp)

    return out_mp4, out_gif, out_webp


def _create_arg_parser(default_root: Path) -> argparse.ArgumentParser:
    """Create configured argument parser for explainer CLI."""
    parser = argparse.ArgumentParser(description="Build explainer video from scenes.")
    def_scenes = default_root / "docs" / "assets" / "explainer"
    def_out = default_root / "docs" / "assets"
    parser.add_argument("--scenes-dir", type=Path, default=def_scenes, help="Scene images dir")
    parser.add_argument("--out-dir", type=Path, default=def_out, help="Output directory")
    parser.add_argument(
        "--seconds-per-scene", type=float, default=5.0, help="Scene duration in seconds"
    )
    parser.add_argument("--skip-gif", action="store_true", help="Skip explainer.gif")
    parser.add_argument("--skip-webp", action="store_true", help="Skip explainer.webp")
    return parser


def parse_args(argv: list[str] | None, default_root: Path) -> ExplainerConfig:
    """Parse CLI options into frozen ExplainerConfig."""
    parser = _create_arg_parser(default_root)
    args = parser.parse_args(argv)

    scenes_dir = (
        args.scenes_dir if args.scenes_dir.is_absolute() else (default_root / args.scenes_dir)
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else (default_root / args.out_dir)

    return ExplainerConfig(
        scenes_dir=scenes_dir,
        out_dir=out_dir,
        seconds_per_scene=args.seconds_per_scene,
        skip_gif=args.skip_gif,
        skip_webp=args.skip_webp,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint to generate explainer mp4, gif, and webp."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    config = parse_args(argv, repo_root)
    build_explainer_video(config)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
