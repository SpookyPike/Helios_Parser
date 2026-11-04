"""Kept so ``python scripts/plot_radiation.py`` and the docs keep working.

The implementation moved to ``helios_viewer.radiation_plot`` so it can ship as
the ``helios_plot_radiation`` console script.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helios_viewer.radiation_plot import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())