"""Run benchmark evaluation wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hydro.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["eval", "--split", "temp-shift", "--mode", "full-comparison", *sys.argv[1:]]))
