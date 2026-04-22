"""Curated display-unit conversion registry for the HELIOS viewer.

All conversions in this module are display-only. Source HDF5 values remain in
their native parsed units and are never rewritten. The viewer and unified shell
use this registry to keep labels, colorbars, probe readouts, and plot data
numerically consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EV_TO_K = 11604.518121550082

TIME_FACTORS = {
    "s": 1.0,
    "ms": 1.0e3,
    "us": 1.0e6,
    "ns": 1.0e9,
    "ps": 1.0e12,
    "fs": 1.0e15,
}

LENGTH_FACTORS = {
    "cm": 1.0,
    "mm": 10.0,
    "um": 1.0e4,
    "nm": 1.0e7,
}

PRESSURE_FACTORS = {
    "J/cm3": 1.0,
    "GPa": 1.0e-3,
    "Mbar": 1.0e-5,
}

DENSITY_FACTORS = {
    "g/cm3": 1.0,
    "kg/m3": 1.0e3,
}

TEMPERATURE_FACTORS = {
    "eV": 1.0,
    "K": EV_TO_K,
}

VELOCITY_FACTORS = {
    "cm/s": 1.0,
    "m/s": 1.0e-2,
    "km/s": 1.0e-5,
}

SPECIFIC_ENERGY_FACTORS = {
    "J/g": 1.0,
    "kJ/g": 1.0e-3,
    "MJ/kg": 1.0e-3,
}

RATE_FACTORS = {
    "J/g/s": 1.0,
    "TW/kg": 1.0e-9,
}

HEAT_CAPACITY_FACTORS = {
    "J/g/eV": 1.0,
    "J/kg/eV": 1.0e3,
    "J/g/K": 1.0 / EV_TO_K,
    "J/kg/K": 1.0e3 / EV_TO_K,
}

NUMBER_DENSITY_FACTORS = {
    "1/cm3": 1.0,
    "1/m3": 1.0e6,
}


PRESSURE_FIELDS = {"pressure", "pressure_e", "pressure_i", "pressure_radiation", "artificial_viscosity"}
DENSITY_FIELDS = {"density"}
TEMPERATURE_FIELDS = {"temperature_e", "temperature_i", "temperature_radiation"}
LENGTH_FIELDS = {"radius", "zone_width"}
VELOCITY_FIELDS = {"velocity"}
SPECIFIC_ENERGY_FIELDS = {"electron_energy", "ion_energy", "radiation_energy", "kinetic_energy", "laser_source"}
RATE_FIELDS = {"radiation_heating", "radiation_cooling", "radiation_sink", "radiation_net_heating", "laser_deposition"}
HEAT_CAPACITY_FIELDS = {"electron_heat_capacity", "ion_heat_capacity"}
NUMBER_DENSITY_FIELDS = {"electron_density"}


@dataclass(frozen=True)
class DisplayUnitChoices:
    """User-selected display units grouped by field family."""

    time_unit: str
    length_unit: str
    pressure_unit: str
    density_unit: str
    temperature_unit: str
    velocity_unit: str
    specific_energy_unit: str
    rate_unit: str
    heat_capacity_unit: str
    number_density_unit: str
    angle_unit: str = "deg"
    photon_unit: str = "eV"


def convert_time_values(values: np.ndarray, unit: str) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * TIME_FACTORS.get(unit, 1.0)


def convert_length_values(values: np.ndarray, unit: str) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * LENGTH_FACTORS.get(unit, 1.0)


def _field_unit_family(field_name: str, native_unit: str) -> str:
    if field_name in PRESSURE_FIELDS and native_unit == "J/cm3":
        return "pressure"
    if field_name in DENSITY_FIELDS and native_unit == "g/cm3":
        return "density"
    if field_name in TEMPERATURE_FIELDS and native_unit == "eV":
        return "temperature"
    if field_name in LENGTH_FIELDS and native_unit == "cm":
        return "length"
    if field_name in VELOCITY_FIELDS and native_unit == "cm/s":
        return "velocity"
    if field_name in SPECIFIC_ENERGY_FIELDS and native_unit == "J/g":
        return "specific_energy"
    if field_name in RATE_FIELDS and native_unit == "J/g/s":
        return "rate"
    if field_name in HEAT_CAPACITY_FIELDS and native_unit == "J/g/eV":
        return "heat_capacity"
    if field_name in NUMBER_DENSITY_FIELDS and native_unit == "1/cm3":
        return "number_density"
    if native_unit in {"", "rho/rho0"}:
        return "dimensionless"
    return "native"


def unit_options_for_field(field_name: str, native_unit: str) -> tuple[str, ...]:
    """Return curated display-unit choices for a given field family."""
    family = _field_unit_family(field_name, native_unit)
    if family == "pressure":
        return tuple(PRESSURE_FACTORS)
    if family == "density":
        return tuple(DENSITY_FACTORS)
    if family == "temperature":
        return tuple(TEMPERATURE_FACTORS)
    if family == "length":
        return tuple(LENGTH_FACTORS)
    if family == "velocity":
        return tuple(VELOCITY_FACTORS)
    if family == "specific_energy":
        return tuple(SPECIFIC_ENERGY_FACTORS)
    if family == "rate":
        return tuple(RATE_FACTORS)
    if family == "heat_capacity":
        return tuple(HEAT_CAPACITY_FACTORS)
    if family == "number_density":
        return tuple(NUMBER_DENSITY_FACTORS)
    if native_unit:
        return (native_unit,)
    return ("",)


def display_unit_for_field(field_name: str, native_unit: str, units: DisplayUnitChoices) -> str:
    family = _field_unit_family(field_name, native_unit)
    if family == "pressure":
        return units.pressure_unit
    if family == "density":
        return units.density_unit
    if family == "temperature":
        return units.temperature_unit
    if family == "length":
        return units.length_unit
    if family == "velocity":
        return units.velocity_unit
    if family == "specific_energy":
        return units.specific_energy_unit
    if family == "rate":
        return units.rate_unit
    if family == "heat_capacity":
        return units.heat_capacity_unit
    if family == "number_density":
        return units.number_density_unit
    return native_unit


def convert_field_values(field_name: str, values: np.ndarray, native_unit: str, units: DisplayUnitChoices) -> tuple[np.ndarray, str]:
    """Convert field values to the current display unit, if supported."""
    array = np.asarray(values, dtype=np.float64)
    family = _field_unit_family(field_name, native_unit)
    if family == "pressure":
        return array * PRESSURE_FACTORS.get(units.pressure_unit, 1.0), units.pressure_unit
    if family == "density":
        return array * DENSITY_FACTORS.get(units.density_unit, 1.0), units.density_unit
    if family == "temperature":
        return array * TEMPERATURE_FACTORS.get(units.temperature_unit, 1.0), units.temperature_unit
    if family == "length":
        return array * LENGTH_FACTORS.get(units.length_unit, 1.0), units.length_unit
    if family == "velocity":
        return array * VELOCITY_FACTORS.get(units.velocity_unit, 1.0), units.velocity_unit
    if family == "specific_energy":
        return array * SPECIFIC_ENERGY_FACTORS.get(units.specific_energy_unit, 1.0), units.specific_energy_unit
    if family == "rate":
        return array * RATE_FACTORS.get(units.rate_unit, 1.0), units.rate_unit
    if family == "heat_capacity":
        return array * HEAT_CAPACITY_FACTORS.get(units.heat_capacity_unit, 1.0), units.heat_capacity_unit
    if family == "number_density":
        return array * NUMBER_DENSITY_FACTORS.get(units.number_density_unit, 1.0), units.number_density_unit
    return array, native_unit
