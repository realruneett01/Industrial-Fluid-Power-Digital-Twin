from pathlib import Path

import numpy as np
import pytest

from hydro.data.schema import CHANNEL_REGISTRY

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def pytest_addoption(parser):
    parser.addoption(
        "--heavy-check",
        action="store_true",
        default=False,
        help="run heavy tests against the full real dataset in data/raw",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--heavy-check"):
        return
    skip = pytest.mark.skip(reason="needs --heavy-check")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    if not (RAW_DIR / "profile.txt").exists():
        pytest.skip(f"real dataset missing in {RAW_DIR}")
    return RAW_DIR


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    """Tiny dataset in the real on-disk format: 8 cycles, tab-separated, all 17 channels."""
    rng = np.random.default_rng(0)
    n = 8
    for name, ch in CHANNEL_REGISTRY.items():
        np.savetxt(tmp_path / f"{name}.txt", rng.normal(50, 5, (n, ch.n_samples)), delimiter="\t")
    profile = np.tile([[100, 100, 0, 130, 0], [3, 73, 2, 90, 1]], (n // 2, 1))
    np.savetxt(tmp_path / "profile.txt", profile, fmt="%d", delimiter="\t")
    return tmp_path
