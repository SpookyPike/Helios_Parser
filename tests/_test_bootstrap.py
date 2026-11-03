from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# `root` as well as `src`: several tests import `scripts.*` (the validators and
# the payload audit), and without the repo root on the path they die at import
# with ModuleNotFoundError, which reads as a broken test rather than a path miss.
for _path in (SRC, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Most historical backend/UI tests exercise dev-only experimental physics
# modules directly. Production-gate tests override this explicitly.
os.environ.setdefault("HELIOS_ENABLE_EXPERIMENTAL", "1")


def example_data_path(name: str) -> Path:
    """Resolve example HDF5 paths across repo layouts used in different bundles."""

    candidates = (
        ROOT / "examples" / name,
        ROOT / name,
        ROOT / "outputs" / "hdf5" / name,
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
