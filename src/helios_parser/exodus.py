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

THE SPECTRAL PATH.  `Write freq-dependent radiation output = 1` adds 18 datasets.
They split over **two different time axes**, and that split is the whole reason
they get two HDF5 groups instead of one: the four node-centred arrays are sampled
on `time_freq`, while the boundary and interface fluxes and every integrated
scalar are sampled on `time_whole`.  `time_freq` is a subset of `time_whole` by
value, never by index, so `radiation/hydro_index` records the pairing.

THE `n_zones + 2` NODE DIMENSION.  The node-centred arrays hold two more rows than
`zone_boundaries`.  Measured on all three fixtures, every dump: rows 0 and
n_zones+1 are exactly zero and the interior rows are the zone centres in Rmin to
Rmax order.  Two independent identities say so: the group sum of
`energy_density` equals `rad_energy_density[interior]` exactly, and
`rad_energy_density[interior]` equals 3 * `rad_pressure`.  The padding rows are
dropped, so the spectral arrays index the same zones as every other field.  A file
whose padding rows are not empty is not that layout, and this module raises rather
than discarding data.

GROUP INTEGRATION.  `sum(FdIntgPowerLossRmin * dE)` reproduces
`FreqIntgRadEnLossRmin` to 1e-16 relative on all three fixtures, i.e. the vendor
integrates with rectangle weights over the bin width.  Trapezoid weights are off
by up to 1.5 % on the Rmin boundary.  `helios.services.derived.radiation` follows
the vendor convention.

LIMITS.  The Exodus container does not record the HELIOS version, so
`helios_version` is left empty rather than guessed.  `Freq. resolved incident flux
at Rmin` and `at Rmax` are identically zero in all three fixtures, so they are
mapped but carry nothing there.  Nothing here was verified for
collisional-radiative runs.
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

# --- Radiation -------------------------------------------------------------

RADIATION_TIME_DATASET = "time_freq"
RADIATION_CYCLE_DATASET = "cycle_num_freq"
GROUP_BOUNDS_DATASET = "photon_energy_group_bounds"
GROUP_BOUNDS_STATIC_DATASET = "photon_energy_group_bounds_singleArrayNonCR"
INTERFACE_INDEX_DATASET = "Indices at region interfaces"

# Node-centred spectral arrays, on the `time_freq` axis.  The node dimension is
# n_zones + 2; see the module docstring.
SPECTRAL_NODE_FIELDS: dict[str, tuple[str, str]] = {
    "Freq. resolved radiation energy density for nodes [J per (cm3.group)]": ("energy_density", "J/cm3/group"),
    "Freq. resolved heating rate for nodes [J per (g.sec.group)]": ("heating_rate", "J/(g s group)"),
    "Freq. resolved cooling rate for nodes [J per (g.sec.group)]": ("cooling_rate", "J/(g s group)"),
    "Freq. resolved Rosseland opacity for nodes [cm2 per g]": ("rosseland_opacity", "cm2/g"),
}

# Boundary and interface spectra, on the `time_whole` axis.  These are per-group
# BIN CONTENTS, not densities: the plain group sum of `net_interface` reproduces
# `TotalFluxRmin`/`Rmax` exactly, while weighting by the group width is off by
# 1e4.  `FdIntgPowerLoss*` is the opposite: it needs the group width and its plain
# sum is off by 1e15.  The container labels both "per eV", which is why the
# normalization is written into the dataset attributes instead of inferred.
FLUX_FIELDS: dict[str, tuple[str, str]] = {
    "Freq. resolved incident flux at Rmax [J per (cm2.sec.eV)]": ("incident_rmax", "J/(cm2 s eV)"),
    "Freq. resolved incident flux at Rmin [J per (cm2.sec.eV)]": ("incident_rmin", "J/(cm2 s eV)"),
    "Freq. resolved net flux at region interfaces [J per (cm2.sec.eV)]": ("net_interface", "J/(cm2 s eV)"),
    "FdPowerLossRmax": ("power_loss_rmax", ""),
    "FdPowerLossRmin": ("power_loss_rmin", ""),
    "FdIntgPowerLossRmax": ("power_loss_rmax_integrated", ""),
    "FdIntgPowerLossRmin": ("power_loss_rmin_integrated", ""),
}

# Per-group normalization, so a consumer cannot integrate a bin content as if it
# were a density.  Measured on all three fixtures; see the module docstring.
FLUX_NORMALIZATION: dict[str, str] = {
    "incident_rmax": "per-group bin content; the group sum is the boundary total (zero in every reference fixture)",
    "incident_rmin": "per-group bin content; the group sum is the boundary total (zero in every reference fixture)",
    "net_interface": "per-group bin content; the group sum reproduces TotalFluxRmin/TotalFluxRmax exactly",
    "power_loss_rmax": "per-group rate",
    "power_loss_rmin": "per-group rate",
    "power_loss_rmax_integrated": "per-group density; weight by the group width (FdIntg * dE = FreqIntgRadEnLoss)",
    "power_loss_rmin_integrated": "per-group density; weight by the group width (FdIntg * dE = FreqIntgRadEnLoss)",
}

# One row per hydro dump.  Units are set only where the container dataset name
# states one; the bookkeeping arrays carry no unit there, so they get "" rather
# than a guess.
INTEGRATED_FIELDS: dict[str, tuple[str, str]] = {
    "EnTotRadiation": ("total_energy", ""),
    "FreqIntgRadEnLossRmax": ("loss_rmax", ""),
    "FreqIntgRadEnLossRmin": ("loss_rmin", ""),
    "TotalFluxRmax": ("total_flux_rmax", "J/(cm2 s)"),
    "TotalFluxRmin": ("total_flux_rmin", "J/(cm2 s)"),
    "Radiation losses at boundaries": ("loss_at_boundaries", ""),
    "TimeIntRadiationLossAtBds": ("loss_at_boundaries_time_integral", ""),
    "RadNetCoolingRateRegion": ("net_cooling_by_region", ""),
    "Fluxes at region interfaces [J per (sec.cm2)]": ("interface_flux", "J/(cm2 s)"),
    "EnExchEleToRadTimeIntg": ("electron_to_radiation_integral", ""),
}

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
    radiation: "RadiationOutput | None" = None

    @property
    def geometry(self) -> str:
        return "planar" if self.coordinate_name == "x" else self.coordinate_name


@dataclass(slots=True)
class SpectralRadiation:
    """The node-centred spectral field, on the `time_freq` axis."""

    time_s: np.ndarray
    cycle: np.ndarray
    hydro_index: np.ndarray
    group_edges_ev: np.ndarray
    group_centre_ev: np.ndarray
    node_axis_cm: np.ndarray
    fields: dict[str, np.ndarray]
    field_units: dict[str, str]


@dataclass(slots=True)
class BoundaryFlux:
    """Boundary and interface spectra, on the `time_whole` axis."""

    time_s: np.ndarray
    cycle: np.ndarray
    group_edges_ev: np.ndarray
    region_boundary_node_index: np.ndarray
    fields: dict[str, np.ndarray]
    field_units: dict[str, str]


@dataclass(slots=True)
class RadiationOutput:
    """Everything the frequency-resolved radiation output carries."""

    spectral: SpectralRadiation
    flux: BoundaryFlux
    integrated: dict[str, np.ndarray]
    integrated_units: dict[str, str]

    @property
    def n_groups(self) -> int:
        return int(self.spectral.group_edges_ev.shape[1] - 1)

    @property
    def n_freq_snapshots(self) -> int:
        return int(self.spectral.time_s.size)

    @property
    def n_boundaries(self) -> int:
        net_interface = self.flux.fields.get("net_interface")
        if net_interface is not None:
            return int(net_interface.shape[1])
        interface_flux = self.integrated.get("interface_flux")
        if interface_flux is not None:
            return int(interface_flux.shape[1])
        return int(self.flux.region_boundary_node_index.size + 1)

    @property
    def n_regions(self) -> int:
        return int(self.flux.region_boundary_node_index.size)


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


def _optional_array(handle: h5py.File, name: str) -> np.ndarray | None:
    if name not in handle:
        return None
    return np.asarray(handle[name][()], dtype=np.float64)


def _nearest_rows(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of the nearest entry of `axis` for every value, or zeros when empty.

    `axis` is ascending (a HELIOS time axis is), so a searchsorted pair beats an
    (n_freq_t, n_hydro_t) distance matrix on long runs.
    """
    if axis.size == 0 or values.size == 0:
        return np.zeros(values.size, dtype=np.int64)
    if axis.size == 1:
        return np.zeros(values.size, dtype=np.int64)
    upper = np.clip(np.searchsorted(axis, values), 1, axis.size - 1)
    lower = upper - 1
    nearest = np.where(np.abs(values - axis[lower]) <= np.abs(axis[upper] - values), lower, upper)
    return nearest.astype(np.int64)


def _group_edges(
    source_rows: np.ndarray | None,
    n_rows: int,
    *,
    bounds: np.ndarray | None,
    static: np.ndarray | None,
    source: Path,
) -> np.ndarray:
    """The photon energy grid for `n_rows` dumps, preferring the per-dump array.

    `source_rows` maps each output row onto a row of the per-dump array.  Falls
    back to the static non-CR grid, then to a single repeated row, because a
    regrouping run can move the edges and a run with one dump cannot tell them
    apart.
    """
    if bounds is not None:
        if bounds.ndim == 1:
            return np.repeat(bounds[None, :], n_rows, axis=0)
        if source_rows is not None and source_rows.size == n_rows and int(source_rows.max(initial=0)) < bounds.shape[0]:
            return bounds[source_rows]
        if bounds.shape[0] == n_rows:
            return bounds
        return np.repeat(bounds[:1], n_rows, axis=0)
    if static is not None:
        return np.repeat(static[None, :], n_rows, axis=0)
    raise ExodusFormatError(
        f"{source}: no {GROUP_BOUNDS_DATASET} and no {GROUP_BOUNDS_STATIC_DATASET}, "
        "so the photon energy grid cannot be resolved."
    )


def _trim_node_padding(source: Path, name: str, values: np.ndarray, n_zones: int) -> np.ndarray:
    """Drop the two empty node rows and return the zone-centred interior."""
    if values.shape[1] != n_zones + 2:
        raise ExodusFormatError(
            f"{source}: {name} has {values.shape[1]} node rows, expected {n_zones + 2} "
            f"({n_zones} zones + 2). Reading it against the zone axis would silently misalign it."
        )
    if np.any(values[:, 0, :]) or np.any(values[:, -1, :]):
        raise ExodusFormatError(
            f"{source}: {name} has non-zero values in the node padding rows, so the "
            "n_zones + 2 layout assumed here does not hold for this file."
        )
    return values[:, 1:-1, :]


def _read_radiation(
    handle: h5py.File,
    *,
    source: Path,
    n_zones: int,
    coordinate_center_cm: np.ndarray,
    time_whole_s: np.ndarray,
    cycle_whole: np.ndarray,
) -> RadiationOutput | None:
    """Read the frequency-resolved radiation block, or None when it is off."""
    time_freq = _optional_array(handle, RADIATION_TIME_DATASET)
    present = [name for name in SPECTRAL_NODE_FIELDS if name in handle]
    if time_freq is None or not present:
        return None
    time_freq = time_freq.ravel()
    n_freq_t = int(time_freq.size)
    hydro_index = _nearest_rows(time_whole_s, time_freq)
    cycle_freq = _optional_array(handle, RADIATION_CYCLE_DATASET)
    cycle = cycle_freq.ravel().astype(np.int64) if cycle_freq is not None and cycle_freq.size == n_freq_t else np.zeros(n_freq_t, dtype=np.int64)

    spectral_fields: dict[str, np.ndarray] = {}
    spectral_units: dict[str, str] = {}
    n_groups: int | None = None
    for dataset_name, (field_name, unit) in SPECTRAL_NODE_FIELDS.items():
        if dataset_name not in handle:
            LOGGER.warning("Exodus file %s has no %s; %s will be absent.", source.name, dataset_name, field_name)
            continue
        values = _optional_array(handle, dataset_name)
        if values.ndim != 3 or values.shape[0] != n_freq_t:
            raise ExodusFormatError(
                f"{source}: {dataset_name} has shape {values.shape}, expected "
                f"({n_freq_t}, {n_zones + 2}, n_groups) on the {RADIATION_TIME_DATASET} axis."
            )
        if n_groups is None:
            n_groups = int(values.shape[2])
        elif values.shape[2] != n_groups:
            raise ExodusFormatError(
                f"{source}: {dataset_name} has {values.shape[2]} groups, earlier spectral arrays have {n_groups}."
            )
        spectral_fields[field_name] = _trim_node_padding(source, dataset_name, values, n_zones)
        spectral_units[field_name] = unit

    if n_groups is None:
        return None

    bounds = _optional_array(handle, GROUP_BOUNDS_DATASET)
    static = _optional_array(handle, GROUP_BOUNDS_STATIC_DATASET)
    spectral_edges = _group_edges(hydro_index, n_freq_t, bounds=bounds, static=static, source=source)
    if spectral_edges.shape[1] != n_groups + 1:
        raise ExodusFormatError(
            f"{source}: the photon energy grid has {spectral_edges.shape[1] - 1} groups, "
            f"the spectral arrays have {n_groups}."
        )
    spectral = SpectralRadiation(
        time_s=time_freq,
        cycle=cycle,
        hydro_index=hydro_index,
        group_edges_ev=spectral_edges,
        group_centre_ev=np.sqrt(spectral_edges[:, :-1] * spectral_edges[:, 1:]),
        node_axis_cm=np.asarray(coordinate_center_cm[0], dtype=np.float64),
        fields=spectral_fields,
        field_units=spectral_units,
    )

    n_hydro_t = int(time_whole_s.size)
    flux_fields: dict[str, np.ndarray] = {}
    flux_units: dict[str, str] = {}
    for dataset_name, (field_name, unit) in FLUX_FIELDS.items():
        if dataset_name not in handle:
            LOGGER.warning("Exodus file %s has no %s; %s will be absent.", source.name, dataset_name, field_name)
            continue
        values = _optional_array(handle, dataset_name)
        if values.ndim < 2 or values.shape[0] != n_hydro_t:
            raise ExodusFormatError(
                f"{source}: {dataset_name} has shape {values.shape}, expected "
                f"({n_hydro_t}, n_groups) on the {TIME_DATASET} axis."
            )
        if values.shape[-1] != n_groups:
            raise ExodusFormatError(
                f"{source}: {dataset_name} has {values.shape[-1]} groups, the spectral arrays have {n_groups}."
            )
        flux_fields[field_name] = values
        flux_units[field_name] = unit

    interface_index = _optional_array(handle, INTERFACE_INDEX_DATASET)
    if interface_index is None:
        # No region boundaries: one region, so the only boundary above Rmin is Rmax.
        region_boundary = np.asarray([n_zones], dtype=np.int64)
    else:
        first = interface_index.ravel() if interface_index.ndim == 1 else interface_index[0]
        # HELIOS counts nodes 1-based up to and including each region's upper
        # boundary; 0-based makes the values index grid/coordinate_edge directly.
        region_boundary = np.asarray(first, dtype=np.int64) - 1

    flux = BoundaryFlux(
        time_s=time_whole_s,
        cycle=np.asarray(cycle_whole, dtype=np.int64),
        group_edges_ev=_group_edges(
            np.arange(n_hydro_t, dtype=np.int64),
            n_hydro_t,
            bounds=bounds,
            static=static,
            source=source,
        ),
        region_boundary_node_index=region_boundary,
        fields=flux_fields,
        field_units=flux_units,
    )

    integrated: dict[str, np.ndarray] = {}
    integrated_units: dict[str, str] = {}
    for dataset_name, (field_name, unit) in INTEGRATED_FIELDS.items():
        if dataset_name not in handle:
            LOGGER.warning("Exodus file %s has no %s; %s will be absent.", source.name, dataset_name, field_name)
            continue
        values = _optional_array(handle, dataset_name)
        if values.shape[0] != n_hydro_t:
            raise ExodusFormatError(
                f"{source}: {dataset_name} has {values.shape[0]} rows, expected {n_hydro_t}."
            )
        integrated[field_name] = values
        integrated_units[field_name] = unit

    return RadiationOutput(
        spectral=spectral,
        flux=flux,
        integrated=integrated,
        integrated_units=integrated_units,
    )


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
        radiation = _read_radiation(
            handle,
            source=source,
            n_zones=n_zones,
            coordinate_center_cm=0.5 * (edge[:, :-1] + edge[:, 1:]),
            time_whole_s=time_s,
            cycle_whole=cycle,
        )

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
        radiation=radiation,
    )