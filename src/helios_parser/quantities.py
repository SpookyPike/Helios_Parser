"""Validated derived quantities shared by HELIOS source readers."""

from __future__ import annotations

import numpy as np


def plasma_pressure(ion_pressure, electron_pressure) -> np.ndarray:
    """Return plasma pressure, excluding radiation pressure."""
    return np.asarray(ion_pressure, dtype=np.float64) + np.asarray(electron_pressure, dtype=np.float64)


def total_pressure(ion_pressure, electron_pressure, radiation_pressure) -> np.ndarray:
    """Return total pressure as ion, electron, and radiation components."""
    return plasma_pressure(ion_pressure, electron_pressure) + np.asarray(radiation_pressure, dtype=np.float64)
