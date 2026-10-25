"""Build the plug-and-play drive image: program, models, offline wheels, integrity manifest.

    python scripts/build_portable.py --out /media/usb/hydro --models models/online \
        --platform linux-x86_64 --platform linux-aarch64 --platform windows-x64

Copy ``--out`` to the drive (or point it straight at the mounted drive). On the target machine
run ``start-hydro.sh`` or ``Start-Hydro.bat``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hydro.deploy import manifest  # noqa: E402
from hydro.deploy.selfcheck import BUNDLE_FILES  # noqa: E402

RUNTIME_REQUIREMENTS = [
    "numpy>=2.2",
    "scipy>=1.15",
    "pandas>=2.2",
    "pyyaml>=6",
    "onnxruntime>=1.21",
    "asyncua>=1.1",
]
PROGRAM_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "pyproject.toml",
    "start-hydro.sh",
    "Start-Hydro.bat",
)
PROGRAM_DIRS = ("src", "configs", "docs/deploy")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
# Friendly target name -> pip platform tags. Newer wheels (scipy, onnxruntime) use manylinux_2_28,
# so every Linux group lists the older tags too; pip picks whichever each wheel was built for.
PLATFORM_GROUPS: dict[str, tuple[str, ...]] = {
    "linux-x86_64": (
        "manylinux_2_28_x86_64",
        "manylinux_2_27_x86_64",
        "manylinux_2_17_x86_64",
        "manylinux2014_x86_64",
    ),
    "linux-aarch64": (
        "manylinux_2_28_aarch64",
        "manylinux_2_27_aarch64",
        "manylinux_2_17_aarch64",
        "manylinux2014_aarch64",
    ),
    "windows-x64": ("win_amd64",),
}
DEFAULT_PLATFORMS = tuple(PLATFORM_GROUPS)
DEFAULT_PYTHONS = ("3.12", "3.13")


def copy_program(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in PROGRAM_FILES:
        src = REPO / name
        if src.is_file():
            shutil.copy2(src, out / name)
    for name in PROGRAM_DIRS:
        src = REPO / name
        if src.is_dir():
            shutil.copytree(src, out / name, ignore=IGNORE, dirs_exist_ok=True)
    (out / "start-hydro.sh").chmod(0o755)


def copy_models(models: Path, out: Path) -> None:
    missing = [f for f in BUNDLE_FILES if not (models / f).is_file()]
    if missing:
        raise SystemExit(f"model bundle {models} is incomplete, missing: {', '.join(missing)}")
    shutil.copytree(models, out / "models" / "online", dirs_exist_ok=True)


def copy_dataset(dataset: Path, out: Path) -> None:
    if not (dataset / "profile.txt").is_file():
        raise SystemExit(f"{dataset} does not look like a ZeMA dataset (no profile.txt)")
    shutil.copytree(dataset, out / "data" / "raw", dirs_exist_ok=True)


def build_wheelhouse(out: Path, platforms: list[str], python_versions: list[str]) -> None:
    """Download wheels per (platform, Python version) pair; offline installs need an exact match."""
    wheelhouse = out / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    (wheelhouse / "requirements.txt").write_text("\n".join(RUNTIME_REQUIREMENTS) + "\n")
    for name in platforms:
        tags = PLATFORM_GROUPS.get(name)
        if tags is None:
            raise SystemExit(f"unknown platform {name!r}; choose from {', '.join(PLATFORM_GROUPS)}")
        for version in python_versions:
            cmd = [sys.executable, "-m", "pip", "download", "--only-binary=:all:"]
            for tag in tags:
                cmd += ["--platform", tag]
            cmd += ["--python-version", version, "--dest", str(wheelhouse), *RUNTIME_REQUIREMENTS]
            print("+", " ".join(cmd))
            subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True, help="bundle dir from `hydro bundle`")
    ap.add_argument("--dataset", type=Path, help="optional ZeMA folder for offline replay mode")
    ap.add_argument("--platform", action="append", choices=list(PLATFORM_GROUPS), help="repeatable")
    ap.add_argument(
        "--python-version",
        action="append",
        help="Python version(s) of the target machines, repeatable (default: 3.12 and 3.13)",
    )
    ap.add_argument("--no-wheels", action="store_true", help="skip the offline wheelhouse")
    args = ap.parse_args(argv)

    out = args.out.resolve()
    copy_program(out)
    copy_models(args.models, out)
    if args.dataset:
        copy_dataset(args.dataset, out)
    if not args.no_wheels:
        build_wheelhouse(
            out,
            args.platform or list(DEFAULT_PLATFORMS),
            args.python_version or list(DEFAULT_PYTHONS),
        )
    path = manifest.write_manifest(out)
    check = manifest.verify_manifest(out)
    if not check["ok"]:
        raise SystemExit(f"fresh manifest does not verify: {check}")
    print(f"portable image ready: {out}\nmanifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
