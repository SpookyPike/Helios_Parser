"""Units for the simulation-setup parameters, and the honesty rule that governs them.

HELIOS stores deck parameters as bare numbers: ``laser_source/wavelength`` is
``0.515`` with nothing saying what that is.  Guessing is the failure mode this
repository guards against, because a wrong unit on a laser wavelength produces a
confident, plausible, wrong answer.

So a unit is emitted only when the repository itself already declares that unit
somewhere a reader can check: the time axis is in seconds, the temperature fields
are in eV, and the photon group edges are in eV.  Everything else is ``null``
with a note.  ``CONVENTIONS`` records the customary unit as prose for the human
reader, never as data.
"""

from __future__ import annotations

from typing import Any

# Grounded in units the converted file already declares on its own datasets:
# time/* is "s", fields/temperature_* is "eV", radiation/group_edges is "eV".
_TABLE: dict[str, str] = {
    "time_control/initial_time_step": "s",
    "time_control/min_time_step": "s",
    "time_control/max_time_step": "s",
    "time_control/max_simulation_time": "s",
    "time_control/max_cycle_count": "1",
    "hydro/boundary_temperature_rmin": "eV",
    "hydro/boundary_temperature_rmax": "eV",
    "hydro/quiet_start_temperature": "eV",
    # The log column header says TW and prints watts; the converter records W, and
    # inspect has to say so or the number is uninterpretable.
    "laser_source/power_table/power": "W",
    "laser_source/wavelength": "um",
    "laser_source/incident_angle_cosine": "1",
    "laser_source/origin_zone_index": "1",
}

# Prefix matches for the sections whose every leaf shares a unit.
_PREFIX_TABLE: tuple[tuple[str, str], ...] = (
    ("time_control/time_step_controls/", "1"),
    ("laser_source/power_table/time", "s"),
)

# Suffix matches: the frequency grid mixes photon energies with group counts and
# section ids, so a prefix rule would label a count as an energy.
_SUFFIX_TABLE: tuple[tuple[str, str], ...] = (
    ("photon_energy", "eV"),
)

# Explicit dimensionless entries, so the suffix rule cannot reach them.
_DIMENSIONLESS: frozenset[str] = frozenset(
    {
        "radiative_transfer/frequency_gridding/number_of_groups",
        "radiative_transfer/frequency_gridding/section",
        "radiative_transfer/number_of_frequency_groups",
    }
)

# A conventional unit the file does not record. Reported as prose, never as data.
CONVENTIONS: dict[str, str] = {}


def unit_for(path: str) -> tuple[str | None, str | None]:
    """Return ``(unit, unit_origin)``; ``(None, None)`` when nothing is grounded."""
    if path in _DIMENSIONLESS:
        return "1", "table"
    if path in _TABLE:
        return _TABLE[path], "table"
    for prefix, unit in _PREFIX_TABLE:
        if path.startswith(prefix):
            return unit, "table"
    for suffix, unit in _SUFFIX_TABLE:
        if path.endswith(suffix):
            return unit, "table"
    return None, None


def annotate(path: str, value: Any) -> dict[str, Any]:
    """Wrap a raw parameter with its unit, the origin of that unit, and any caveat."""
    unit, origin = unit_for(path)
    entry: dict[str, Any] = {"value": value, "unit": unit, "unit_origin": origin}
    if unit is None:
        caveat = CONVENTIONS.get(path, "the deck records no unit for this parameter")
        entry["note"] = caveat
    return entry