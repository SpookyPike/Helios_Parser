from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
