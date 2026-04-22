"""Simple line-of-sight helpers for derived spectroscopy estimates."""

from __future__ import annotations


def los_velocity(velocity_cm_s: float, *, cosine: float = 1.0) -> float:
    """Project a scalar velocity onto the assumed line of sight."""

    return float(velocity_cm_s) * float(cosine)
