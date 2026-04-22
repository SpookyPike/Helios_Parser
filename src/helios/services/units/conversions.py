"""Shared unit helpers for derived-analysis services.

This module intentionally stays small and explicit. It only contains the
conversions needed by the current quick-look derived tools and avoids trying to
become a general-purpose units framework.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from helios.services.constants import CM_PER_ANGSTROM, CM_PER_NM, EV_TO_K, WAVELENGTH_ANGSTROM_KEV


def ev_to_kelvin(value_ev: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value_ev, dtype=np.float64) * EV_TO_K


def photon_energy_kev_to_wavelength_angstrom(energy_kev: float) -> float:
    if energy_kev <= 0.0:
        raise ValueError("Photon energy must be positive in keV.")
    return WAVELENGTH_ANGSTROM_KEV / float(energy_kev)


def photon_energy_kev_to_wavelength_cm(energy_kev: float) -> float:
    return photon_energy_kev_to_wavelength_angstrom(energy_kev) * CM_PER_ANGSTROM


def wavelength_nm_to_cm(wavelength_nm: float) -> float:
    return float(wavelength_nm) * CM_PER_NM


def wavelength_cm_to_nm(wavelength_cm: float) -> float:
    return float(wavelength_cm) / CM_PER_NM


def photon_energy_ev_from_wavelength_nm(wavelength_nm: float) -> float:
    """Return the photon energy in eV for a line wavelength expressed in nm."""

    wavelength_angstrom = float(wavelength_nm) * 10.0
    if wavelength_angstrom <= 0.0:
        raise ValueError("Wavelength must be positive in nm.")
    return (1000.0 * WAVELENGTH_ANGSTROM_KEV) / wavelength_angstrom


def wavelength_shift_nm_to_energy_ev(shift_nm: float | np.ndarray, line_wavelength_nm: float) -> float | np.ndarray:
    """Convert a wavelength shift to an energy shift using ``DeltaE/E = -DeltaLambda/Lambda``."""

    reference_energy_ev = photon_energy_ev_from_wavelength_nm(float(line_wavelength_nm))
    scale = -reference_energy_ev / float(line_wavelength_nm)
    return np.asarray(shift_nm, dtype=np.float64) * scale


def weighted_mean(values: Iterable[float] | np.ndarray, weights: Iterable[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(array) & np.isfinite(weight_array) & (weight_array > 0.0)
    if not np.any(finite):
        return float("nan")
    return float(np.average(array[finite], weights=weight_array[finite]))


def safe_ratio(numerator: float, denominator: float, *, default: float = float("nan")) -> float:
    if not math.isfinite(float(denominator)) or abs(float(denominator)) <= 0.0:
        return float(default)
    return float(numerator) / float(denominator)
