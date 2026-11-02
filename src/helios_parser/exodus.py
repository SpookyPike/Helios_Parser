"""Exodus II (HDF5) plot-output reader.

WHY THIS EXISTS.  HELIOS writes its plot output as an Exodus II file, and in
12.0.0 that is the **default** (`Plot file type = 0` in `[Output Control Data]`);
`Plot file type = 1` is the BPF format this package already reads.  A user who
upgrades from 11.0.0 and leaves the output control alone therefore gets an
`.exo` file and no BPF, so the plot/derived workflow needs this path.

WHAT IT IS.  An Exodus II file written by HELIOS is a NetCDF-4 container, i.e.
plain HDF5, with the hydrodynamic fields as top-level datasets.  This module
reads the core fields and maps them onto the same names and units the log and
BPF paths emit, so everything downstream is unchanged.

THE ONE NON-OBVIOUS MAPPING.  `fluid_velocity` is stored at the **nodes**
(n_zones + 1 entries) while every other field is zone-centred.  The value the
log prints for a zone is `fluid_velocity[:, 1:]` -- the *upper* node of that
zone -- not the mean of its two nodes.  Verified against the same run's log:
`[:, 1:]` agrees to 1.6e-4 relative (the log's five printed digits) while the
two-node mean is off by 27 %.  Averaging would put a systematic error into the
rear-face velocity, which is the quantity breakout times are read from.

LIMITS.  The Exodus container does not record the HELIOS version, so
`helios_version` is left empty rather than guessed.  The frequency-resolved
radiation arrays are present but only when frequency-dependent output is on, and
they are not mapped here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

LOGGER = logging.getLogger(__name__)

# Exodus dataset name -> (parser field name, unit).  Units are HELIOS's own, as
# printed in the log header, so an Exodus-derived file is dimensionally
# indistinguishable from a log-derived one.
# Field names are the LOG path's names, not the Exodus ones, so an
# Exodus-derived file is interchangeable with a log-derived one downstream.
EXODUS_ZONE_FIELDS: dict[str, tuple[str, str]] = {
    "mass_density": ("density", "g/cm3"),
    "ion_temperature": ("temperature_i", "eV"),
    "elec_temperature": ("temperature_e", "eV"),
    "radiation_temperature": ("temperature_radiation", "eV"),
    "ion_pressure": ("pressure_i", "J/cm3"),
    "elec_pressure": ("pressure_e", "J/cm3"),
    "rad_pressure": ("pressure_radiation", "J/cm3"),
    "ion_density": ("ion_density", "1/cm3"),
    "elec_density": ("electron_density", "1/cm3"),
}

TIME_DATASET = "time_whole"
CYCLE_DATASET = "cycle_num"
TIMESTEP_DATASET = "Time step size [sec]"
EDGE_DATASET = "zone_boundaries"
VELOCITY_DATASET = "fluid_velocity"

GEOMETRY_ATTR = "Geometry ID (0 = planar, 1 = cylindrical, 2 = spherical)"
TEMPERATURE_MODEL_ATTR = "Temperature model (1 = 1-temperature, 2 = 2-temperature)"

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class ExodusFormatError(ValueError):
    """The file is HDF5 but is not a HELIOS Exodus II plot file."""


@dataclass(slots=True)
class ExodusRun:
    """The core contents of one HELIOS Exodus II plot file."""

    source: Path
    time_s: np.ndarray
    cycle: np.ndarray
    time_step_s: np.ndarray | None
    coordinate_edge_cm: np.ndarray
    coordinate_center_cm: np.ndarray
    zone_width_cm: np.ndarray
    fields: dict[str, np.ndarray]
    field_units: dict[str, str]
    raw_field_map: dict[str, str]
    coordinate_name: str
    n_zones: int
    n_snapshots: int
    temperature_model: int | None
    zone_mass_g: np.ndarray | None = None

    @property
    def geometry(self) -> str:
        return "planar" if self.coordinate_name == "x" else self.coordinate_name


def looks_like_exodus(path: str | Path) -> bool:
    """True when the path is an HDF5 container holding HELIOS Exodus fields."""
    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        with candidate.open("rb") as handle:
            if handle.read(8) != HDF5_MAGIC:
                return False
    except OSError:
        return False
    try:
        with h5py.File(candidate, "r") as handle:
            return TIME_DATASET in handle and EDGE_DATASET in handle
    except OSError:
        return False


def _scalar_attr(handle: h5py.File, name: str) -> int | None:
    value = handle.attrs.get(name)
    if value is None:
        return None
    array = np.asarray(value).ravel()
    if array.size == 0:
        return None
    try:
        return int(array[0])
    except (TypeError, ValueError):
        return None


def _coordinate_name(handle: h5py.File) -> str:
    geometry_id = _scalar_attr(handle, GEOMETRY_ATTR)
    if geometry_id == 1:
        return "r"
    if geometry_id == 2:
        return "r"
    if geometry_id == 0:
        return "x"
    # Older files may only carry the text attribute.
    geometry = handle.attrs.get("geometry", b"")
    text = geometry.decode("utf-8", "replace").lower() if isinstance(geometry, bytes) else str(geometry).lower()
    return "r" if ("cyl" in text or "spher" in text) else "x"


def read_exodus(path: str | Path) -> ExodusRun:
    """Read the core hydrodynamic fields from a HELIOS Exodus II plot file."""
    source = Path(path)
    with h5py.File(source, "r") as handle:
        missing = [name for name in (TIME_DATASET, EDGE_DATASET) if name not in handle]
        if missing:
            raise ExodusFormatError(
                f"{source} is HDF5 but not a HELIOS Exodus II plot file: missing {', '.join(missing)}."
            )
        time_s = np.asarray(handle[TIME_DATASET][()], dtype=np.float64).ravel()
        edge = np.asarray(handle[EDGE_DATASET][()], dtype=np.float64)
        if edge.ndim == 1:
            edge = edge[None, :]
        n_snapshots = int(time_s.size)
        if edge.shape[0] != n_snapshots:
            raise ExodusFormatError(
                f"{source}: {edge.shape[0]} coordinate records for {n_snapshots} time steps."
            )
        n_zones = int(edge.shape[1] - 1)

        cycle = (
            np.asarray(handle[CYCLE_DATASET][()], dtype=np.int64).ravel()
            if CYCLE_DATASET in handle
            else np.arange(1, n_snapshots + 1, dtype=np.int64)
        )
        time_step = (
            np.asarray(handle[TIMESTEP_DATASET][()], dtype=np.float64).ravel()
            if TIMESTEP_DATASET in handle
            else None
        )

        fields: dict[str, np.ndarray] = {}
        units: dict[str, str] = {}
        raw_map: dict[str, str] = {}
        for dataset_name, (field_name, unit) in EXODUS_ZONE_FIELDS.items():
            if dataset_name not in handle:
                LOGGER.warning("Exodus file %s has no %s; %s will be absent.", source.name, dataset_name, field_name)
                continue
            values = np.asarray(handle[dataset_name][()], dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != n_zones:
                raise ExodusFormatError(
                    f"{source}: {dataset_name} has shape {values.shape}, expected ({n_snapshots}, {n_zones})."
                )
            fields[field_name] = values
            units[field_name] = unit
            raw_map[field_name] = dataset_name

        if VELOCITY_DATASET in handle:
            nodal = np.asarray(handle[VELOCITY_DATASET][()], dtype=np.float64)
            if nodal.ndim != 2 or nodal.shape[1] != n_zones + 1:
                raise ExodusFormatError(
                    f"{source}: {VELOCITY_DATASET} has shape {nodal.shape}, expected ({n_snapshots}, {n_zones + 1})."
                )
            # The log prints the zone velocity at the zone's UPPER node.
            fields["velocity"] = nodal[:, 1:]
            units["velocity"] = "cm/s"
            raw_map["velocity"] = f"{VELOCITY_DATASET}[:, 1:]"
        else:
            LOGGER.warning("Exodus file %s has no %s; velocity will be absent.", source.name, VELOCITY_DATASET)

        zone_mass = (
            np.asarray(handle["zone_mass"][()], dtype=np.float64)
            if "zone_mass" in handle
            else None
        )

        temperature_model = _scalar_attr(handle, TEMPERATURE_MODEL_ATTR)
        coordinate_name = _coordinate_name(handle)

    if "density" not in fields:
        raise ExodusFormatError(f"{source}: no mass_density dataset, so this is not a HELIOS plot file.")

    # Derived pressures, exactly as the log path defines them, so a consumer that
    # asks for total pressure gets the same quantity from either source.
    if "pressure_i" in fields and "pressure_e" in fields:
        plasma = fields["pressure_i"] + fields["pressure_e"]
        fields["plasma_pressure_j_cm3"] = plasma
        units["plasma_pressure_j_cm3"] = "J/cm3"
        raw_map["plasma_pressure_j_cm3"] = "ion_pressure + elec_pressure"
        total = plasma + fields.get("pressure_radiation", 0.0)
        fields["total_pressure_j_cm3"] = total
        units["total_pressure_j_cm3"] = "J/cm3"
        raw_map["total_pressure_j_cm3"] = "ion_pressure + elec_pressure + rad_pressure"
        fields["pressure"] = total
        units["pressure"] = "J/cm3"
        raw_map["pressure"] = "alias of total_pressure_j_cm3"

    coordinate_edge = edge
    coordinate_center = 0.5 * (coordinate_edge[:, :-1] + coordinate_edge[:, 1:])
    zone_width = np.diff(coordinate_edge, axis=1)
    return ExodusRun(
        source=source,
        time_s=time_s,
        cycle=cycle,
        time_step_s=time_step,
        coordinate_edge_cm=coordinate_edge,
        coordinate_center_cm=coordinate_center,
        zone_width_cm=zone_width,
        fields=fields,
        field_units=units,
        raw_field_map=raw_map,
        coordinate_name=coordinate_name,
        n_zones=n_zones,
        n_snapshots=n_snapshots,
        temperature_model=temperature_model,
        zone_mass_g=zone_mass,
    )