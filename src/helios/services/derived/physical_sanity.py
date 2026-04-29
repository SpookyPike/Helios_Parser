"""Lightweight physical sanity checks for release-visible derived outputs."""

from __future__ import annotations

import math

import numpy as np


_MAX_REASONABLE_SPEED_CM_S = 1.0e10


def _finite_scalar(value: object, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} is not finite.")
    return numeric


def _finite_array(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def validate_shock_result(result: object) -> None:
    """Reject shock tracks with non-finite or superluminal-scale speeds."""

    position = np.asarray(getattr(result, "position_cm"), dtype=np.float64)
    speed = np.asarray(getattr(result, "speed_magnitude_cm_s"), dtype=np.float64)
    finite_position = position[np.isfinite(position)]
    finite_speed = speed[np.isfinite(speed)]
    if finite_position.size == 0 or finite_speed.size == 0:
        raise ValueError("Shock result contains no finite track samples.")
    if np.nanmax(np.abs(finite_speed)) > _MAX_REASONABLE_SPEED_CM_S:
        raise ValueError("Shock speed exceeds the v1.0 physical sanity limit.")


def validate_xrd_result(result: object) -> None:
    """Reject XRD quick-look estimates with impossible compression/Q values."""

    wavelength = _finite_scalar(getattr(result, "wavelength_angstrom"), "XRD wavelength")
    if wavelength <= 0.0:
        raise ValueError("XRD wavelength must be positive.")
    for layer in getattr(result, "layers"):
        compressed_density = _finite_scalar(getattr(layer, "compressed_density_g_cm3"), "XRD compressed density")
        compression_ratio = _finite_scalar(getattr(layer, "compression_ratio"), "XRD compression ratio")
        d_over_d0 = _finite_scalar(getattr(layer, "d_over_d0"), "XRD d/d0")
        q0 = _finite_scalar(getattr(layer, "q0_inv_angstrom"), "XRD Q0")
        q = _finite_scalar(getattr(layer, "q_compressed_inv_angstrom"), "XRD Q")
        if compressed_density <= 0.0:
            raise ValueError("XRD compressed density must be positive.")
        if not (1.0e-3 <= compression_ratio <= 1.0e3):
            raise ValueError("XRD compression ratio is outside the v1.0 sanity range.")
        if not (1.0e-3 <= d_over_d0 <= 1.0e3):
            raise ValueError("XRD lattice-spacing ratio is outside the v1.0 sanity range.")
        if q0 <= 0.0 or q <= 0.0:
            raise ValueError("XRD scattering vector values must be positive.")


def validate_spectroscopy_result(result: object) -> None:
    """Reject Doppler/broadening quick-look outputs that violate basic bounds."""

    wavelength_nm = _finite_scalar(getattr(result, "line_wavelength_nm"), "Spectroscopy line wavelength")
    if wavelength_nm <= 0.0:
        raise ValueError("Spectroscopy line wavelength must be positive.")
    scalars = (
        float(getattr(result, "bulk_velocity_cm_s")),
        float(getattr(result, "los_velocity_cm_s")),
        float(getattr(result, "thermal_width_fraction")),
        float(getattr(result, "ion_temperature_ev")),
        float(getattr(result, "ion_mass_mu")),
    )
    if not any(math.isfinite(value) for value in scalars):
        return
    bulk_velocity = _finite_scalar(getattr(result, "bulk_velocity_cm_s"), "Spectroscopy bulk velocity")
    los_velocity = _finite_scalar(getattr(result, "los_velocity_cm_s"), "Spectroscopy LOS velocity")
    thermal_fraction = _finite_scalar(getattr(result, "thermal_width_fraction"), "Spectroscopy thermal width fraction")
    ion_temperature = _finite_scalar(getattr(result, "ion_temperature_ev"), "Spectroscopy ion temperature")
    ion_mass = _finite_scalar(getattr(result, "ion_mass_mu"), "Spectroscopy ion mass")
    if abs(bulk_velocity) > _MAX_REASONABLE_SPEED_CM_S or abs(los_velocity) > _MAX_REASONABLE_SPEED_CM_S:
        raise ValueError("Spectroscopy velocity exceeds the v1.0 physical sanity limit.")
    if thermal_fraction < 0.0 or thermal_fraction > 1.0:
        raise ValueError("Spectroscopy thermal width fraction is outside the v1.0 sanity range.")
    if ion_temperature < 0.0:
        raise ValueError("Spectroscopy ion temperature must be non-negative.")
    if ion_mass <= 0.0:
        raise ValueError("Spectroscopy ion mass must be positive.")
