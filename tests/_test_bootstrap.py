from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
