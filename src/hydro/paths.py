"""Where the project's configs and certificates live, also when installed rather than editable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "HYDRO_ROOT"


def project_root() -> Path:
    """``$HYDRO_ROOT`` when set (portable drive, container, service), else the source checkout."""
    env = os.environ.get(ROOT_ENV)
    return Path(env) if env else Path(__file__).resolve().parents[2]
