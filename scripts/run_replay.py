"""Run OPC-UA replay server wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hydro.online.replay_server import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
