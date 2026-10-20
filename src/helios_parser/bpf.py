from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np

from .quantities import plasma_pressure, total_pressure


@dataclass(frozen=True, slots=True)
class BpfRecordSpec:
    rel: int
    key: str
    dtype: str
    shape: str
    status: str
    unit: str = ""
    label: str = ""
    axes: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class BpfLayout:
    path: Path
    run_date: str
    run_clock: str
    record_count: int
    n_nodes: int
    n_zones: int
    n_freq_bins: int
    block_records: int
    n_snapshots: int
    trailing_records: int
    trailing_record_lengths: tuple[int, ...]
    recognized_zero_trailer: bool
    incomplete_final_snapshot: bool
    partial_final_record_bytes: int
    first_time_s: float
    last_time_s: float
    first_cycle: int
    last_cycle: int


@dataclass(frozen=True, slots=True)
class BpfSnapshot:
    index: int
    time_s: float
    cycle: int
    fields: dict[str, np.ndarray]


BPF_FIXED_RECORD_SPECS: tuple[BpfRecordSpec, ...] = (
    BpfRecordSpec(0, "snapshot_header", "f64", "30", "mapped", "", "BPF snapshot header", ("header_value",), "Header vector; validated entries include time and cycle."),
    BpfRecordSpec(1, "global_radiation_energy_summary", "f64", "70", "mapped", "", "Global radiation energy summary", ("summary_value",), "Mixed global diagnostics; preserved without per-entry labels."),
    BpfRecordSpec(2, "global_source_summary", "f64", "n_freq_bins", "mapped", "", "Global source summary", ("summary_value",), "Mixed source vector; preserved without per-entry labels."),
    BpfRecordSpec(3, "bpf_record_03", "f64", "50", "unknown_bpf_record", "", "BPF record 03", ("bpf_record_value",)),
    BpfRecordSpec(4, "node_position_cm", "f64", "n_nodes", "validated", "cm", "Node position", ("node",), "Dynamic node coordinate."),
    BpfRecordSpec(5, "zone_mass", "f64", "n_zones", "validated", "g/cm**X", "Zone mass", ("zone",)),
    BpfRecordSpec(6, "ion_temperature_eV", "f64", "n_zones", "validated", "eV", "Ion temperature", ("zone",)),
    BpfRecordSpec(7, "radiation_temperature_eV", "f64", "n_zones", "validated", "eV", "Radiation temperature", ("zone",)),
    BpfRecordSpec(8, "mass_density_g_cm3", "f64", "n_zones", "validated", "g/cm3", "Mass density", ("zone",)),
    BpfRecordSpec(9, "mean_charge", "f64", "n_zones", "validated", "", "Mean charge", ("zone",), "Charge-state weighted mean charge."),
    BpfRecordSpec(10, "ion_pressure_j_cm3", "f64", "n_zones", "validated", "J/cm3", "Ion pressure", ("zone",)),
    BpfRecordSpec(11, "radiation_cooling_j_g_s", "f64", "n_zones", "validated", "J/g/s", "Radiation cooling", ("zone",)),
    BpfRecordSpec(12, "radiation_heating_j_g_s", "f64", "n_zones", "validated", "J/g/s", "Radiation heating", ("zone",)),
    BpfRecordSpec(13, "ion_energy_j_g", "f64", "n_zones", "validated", "J/g", "Ion energy", ("zone",)),
    BpfRecordSpec(14, "bpf_record_14", "f64", "n_zones+2", "validated", "J/cm3", "Radiation energy density padded", ("bpf_record_value",), "Padded radiation energy density vector: entries 1..n_zones equal zone radiation energy density; endpoints are boundary padding. Interior equals 3*LOG radiation pressure and density*LOG radiation energy."),
    BpfRecordSpec(15, "bpf_record_15", "f64", "n_nodes", "partially_characterized", "", "BPF node auxiliary 15", ("node",), "Positive node-centered auxiliary. It follows the expanding right-going front and correlates with interface velocity/front position, but no stable unit or physics label is validated."),
    BpfRecordSpec(16, "interface_velocity_cm_s", "f64", "n_nodes", "validated", "cm/s", "Interface velocity", ("node",), "Interface velocity; zone-centered average matches EXO fluid velocity."),
    BpfRecordSpec(17, "laser_deposition_j_g_s", "f64", "n_zones", "validated", "J/g/s", "Laser deposition", ("zone",)),
    BpfRecordSpec(18, "bpf_record_18", "f64", "n_nodes", "partially_characterized", "", "BPF node auxiliary 18", ("node",), "Node-centered radiation-flux-like auxiliary. Right-boundary values correlate with boundary radiation flux, but cross-run scaling is not stable enough for a validated flux label."),
    BpfRecordSpec(19, "bpf_record_19", "f64", "n_nodes", "partially_characterized", "", "BPF node mask/weight record 19", ("node",), "Constant one-valued node vector in available samples; likely an enabled mask/weight, exact semantics unresolved."),
    BpfRecordSpec(20, "artificial_viscosity_j_cm3", "f64", "n_zones", "validated", "J/cm3", "Artificial viscosity", ("zone",)),
    BpfRecordSpec(21, "bpf_record_21", "f64", "n_nodes", "partially_characterized", "", "Inactive BPF node record 21", ("node",), "All-zero node vector in available samples; optional/inactive physics channel, exact semantics unresolved."),
    BpfRecordSpec(22, "bpf_record_22", "f64", "n_nodes", "partially_characterized", "", "Inactive BPF node record 22", ("node",), "All-zero node vector in available samples; optional/inactive physics channel, exact semantics unresolved."),
    BpfRecordSpec(23, "bpf_record_23", "f64", "n_nodes", "partially_characterized", "", "Inactive BPF node record 23", ("node",), "All-zero node vector in available samples; optional/inactive physics channel, exact semantics unresolved."),
    BpfRecordSpec(24, "bpf_record_24", "f64", "n_nodes", "partially_characterized", "", "Inactive BPF node record 24", ("node",), "All-zero node vector in available samples; optional/inactive physics channel, exact semantics unresolved."),
    BpfRecordSpec(25, "bpf_record_25", "f64", "n_nodes", "partially_characterized", "", "Inactive BPF node record 25", ("node",), "All-zero node vector in available samples; optional/inactive physics channel, exact semantics unresolved."),
    BpfRecordSpec(26, "bpf_record_26", "f64", "n_nodes", "partially_characterized", "", "Inactive BPF node record 26", ("node",), "All-zero node vector in available samples; optional/inactive physics channel, exact semantics unresolved."),
    BpfRecordSpec(27, "bpf_record_27", "f64", "n_nodes", "partially_characterized", "", "Inactive BPF node record 27", ("node",), "All-zero node vector in available samples; optional/inactive physics channel, exact semantics unresolved."),
    BpfRecordSpec(28, "bpf_record_28", "f64", "n_nodes", "partially_characterized", "", "Inactive BPF node record 28", ("node",), "All-zero node vector in available samples; optional/inactive physics channel, exact semantics unresolved."),
    BpfRecordSpec(29, "bpf_record_29", "f64", "n_zones", "unknown_bpf_record", "", "BPF record 29", ("zone",), "Preserved raw zone vector; direct LOG comparison shows this is not the cumulative LaserSrc field."),
    BpfRecordSpec(30, "planck_mean_opacity_absorption_cm2_g", "f64", "n_zones", "mapped", "cm2/g", "Planck mean opacity (absorption)", ("zone",), "Mapped from HydroPLOT opacity entries and sample-data behavior; not directly present in LOG tables."),
    BpfRecordSpec(31, "planck_mean_opacity_emission_cm2_g", "f64", "n_zones", "mapped", "cm2/g", "Planck mean opacity (emission)", ("zone",), "Mapped from HydroPLOT opacity entries and sample-data behavior; not directly present in LOG tables."),
    BpfRecordSpec(32, "rosseland_mean_opacity_cm2_g", "f64", "n_zones", "mapped", "cm2/g", "Mean Rosseland opacity", ("zone",), "Mapped from HydroPLOT opacity entries and sample-data behavior; not directly present in LOG tables."),
    BpfRecordSpec(33, "ion_density_cm3", "f64", "n_zones", "validated", "1/cm3", "Ion density", ("zone",)),
    BpfRecordSpec(34, "boundary_net_flux_pair_j_s_cm2", "f64", "2", "validated", "J/s/cm2", "Boundary net flux pair", ("boundary",), "Signed Rmin/Rmax boundary flux pair."),
    BpfRecordSpec(35, "bpf_record_35", "i32", "1", "mapped", "count", "Node count repeat", ("bpf_record_value",), "Scalar repeat of n_nodes in available samples."),
    BpfRecordSpec(36, "bpf_record_36", "f64", "2", "partially_characterized", "", "Inactive BPF pair record 36", ("bpf_record_value",), "All-zero two-value record in available samples; exact semantics unresolved."),
    BpfRecordSpec(37, "region_net_cooling_rate_j_s_cm2", "f64", "1", "validated", "J/s/cm2", "Region net cooling rate", ("region",)),
    BpfRecordSpec(38, "region_ion_flux_boundary_j_s_cm2", "f64", "1", "validated", "J/s/cm2", "Region ion flux boundary", ("region",)),
    BpfRecordSpec(39, "region_electron_flux_boundary_j_s_cm2", "f64", "1", "validated", "J/s/cm2", "Region electron flux boundary", ("region",)),
    BpfRecordSpec(40, "bpf_record_40", "f64", "1", "partially_characterized", "", "Inactive BPF scalar record 40", ("bpf_record_value",), "All-zero scalar in available samples; exact semantics unresolved."),
    BpfRecordSpec(41, "electron_temperature_eV", "f64", "n_zones", "validated", "eV", "Electron temperature", ("zone",)),
    BpfRecordSpec(42, "electron_pressure_j_cm3", "f64", "n_zones", "validated", "J/cm3", "Electron pressure", ("zone",)),
    BpfRecordSpec(43, "electron_density_cm3", "f64", "n_zones", "validated", "1/cm3", "Electron density", ("zone",)),
    BpfRecordSpec(44, "bpf_record_44", "f64", "n_zones", "unknown_bpf_record", "", "BPF record 44", ("zone",), "Zone field with inconsistent cross-run correlations to energy-like quantities; no stable LOG/EXO/manual mapping found."),
    BpfRecordSpec(45, "frequency_group_boundaries_eV", "f64", "n_freq_bins+1", "validated", "eV", "Frequency group boundaries", ("frequency_edge",)),
    BpfRecordSpec(46, "radiation_source_escaping_rmax_j_s_cm2_eV", "f64", "n_freq_bins", "mapped", "J/s/cm2/eV", "Spectral radiation power loss Rmax", ("frequency",), "HydroPLOT labels this family as radiation power loss; retained under the established field key for compatibility."),
    BpfRecordSpec(47, "radiation_source_escaping_rmin_j_s_cm2_eV", "f64", "n_freq_bins", "mapped", "J/s/cm2/eV", "Spectral radiation power loss Rmin", ("frequency",), "HydroPLOT labels this family as radiation power loss; retained under the established field key for compatibility."),
    BpfRecordSpec(48, "radiation_loss_rmax_j_cm2_eV", "f64", "n_freq_bins", "validated", "J/cm2/eV", "Time-integrated spectral radiation power loss Rmax", ("frequency",)),
    BpfRecordSpec(49, "radiation_loss_rmin_j_cm2_eV", "f64", "n_freq_bins", "validated", "J/cm2/eV", "Time-integrated spectral radiation power loss Rmin", ("frequency",)),
    BpfRecordSpec(50, "radiation_net_flux_rmin_j_s_cm2_eV", "f64", "n_freq_bins", "validated", "J/s/cm2/eV", "Spectral net radiation flux Rmin", ("frequency",), "Positive in +R direction."),
    BpfRecordSpec(51, "radiation_net_flux_rmax_j_s_cm2_eV", "f64", "n_freq_bins", "validated", "J/s/cm2/eV", "Spectral net radiation flux Rmax", ("frequency",), "Positive in +R direction."),
    BpfRecordSpec(52, "external_rad_source_rmin", "f64", "n_freq_bins", "validated", "", "External radiation source Rmin", ("frequency",)),
    BpfRecordSpec(53, "external_rad_source_rmax", "f64", "n_freq_bins", "validated", "", "External radiation source Rmax", ("frequency",)),
    BpfRecordSpec(54, "bpf_record_54", "i32", "1", "partially_characterized", "", "BPF scalar flag 54", ("bpf_record_value",), "Constant value 1 in available samples; likely a control flag, exact semantics unresolved."),
    BpfRecordSpec(55, "bpf_record_55", "f64", "1", "partially_characterized", "", "Inactive BPF scalar record 55", ("bpf_record_value",), "All-zero scalar in available samples; exact semantics unresolved."),
    BpfRecordSpec(56, "bpf_record_56", "i32", "1", "partially_characterized", "", "Inactive BPF scalar record 56", ("bpf_record_value",), "All-zero integer scalar in available samples; exact semantics unresolved."),
    BpfRecordSpec(57, "region_index_by_zone", "i32", "n_zones", "validated", "", "Region index by zone", ("zone",)),
    BpfRecordSpec(58, "bpf_record_58", "i32", "1", "partially_characterized", "", "BPF scalar flag 58", ("bpf_record_value",), "Constant value 1 in available samples; likely a control flag, exact semantics unresolved."),
    BpfRecordSpec(59, "bpf_record_59", "i32", "1", "partially_characterized", "", "BPF scalar flag 59", ("bpf_record_value",), "Constant value 1 in available samples; likely a control flag, exact semantics unresolved."),
    BpfRecordSpec(60, "bpf_record_60", "i32", "1", "partially_characterized", "", "Inactive BPF scalar record 60", ("bpf_record_value",), "All-zero integer scalar in available samples; exact semantics unresolved."),
    BpfRecordSpec(61, "bpf_record_61", "i32", "1", "partially_characterized", "", "Inactive BPF scalar record 61", ("bpf_record_value",), "All-zero integer scalar in available samples; exact semantics unresolved."),
    BpfRecordSpec(62, "ionization_fraction_width", "i32", "1", "validated", "count", "Ionization fraction width", ("bpf_record_value",), "Z+1 charge-state count."),
    BpfRecordSpec(63, "bpf_record_63", "i32", "1", "partially_characterized", "", "Inactive BPF scalar record 63", ("bpf_record_value",), "All-zero integer scalar in available samples; exact semantics unresolved."),
    BpfRecordSpec(64, "atomic_number_z", "i32", "1", "validated", "count", "Atomic number Z", ("bpf_record_value",)),
    BpfRecordSpec(65, "bpf_record_65", "i32", "1", "partially_characterized", "", "BPF scalar flag 65", ("bpf_record_value",), "Constant value -1 in available samples; likely a control flag, exact semantics unresolved."),
    BpfRecordSpec(66, "bpf_record_66", "f64", "1", "partially_characterized", "", "BPF scalar flag 66", ("bpf_record_value",), "Constant value 1 in available samples; likely a control flag, exact semantics unresolved."),
    BpfRecordSpec(67, "bpf_record_67", "i32", "1", "partially_characterized", "", "BPF scalar flag 67", ("bpf_record_value",), "Constant value 1 in available samples; likely a control flag, exact semantics unresolved."),
    BpfRecordSpec(68, "ionization_fraction_width_repeat", "i32", "1", "validated", "count", "Ionization fraction width repeat", ("bpf_record_value",)),
)


DERIVED_BPF_FIELD_METADATA: dict[str, dict[str, Any]] = {
    "zone_center_cm": {
        "unit": "cm",
        "quantity": "length",
        "label": "Zone center position",
        "axes": ("zone",),
        "status": "derived",
        "description": "Derived from adjacent BPF node positions.",
    },
    "zone_width_cm": {
        "unit": "cm",
        "quantity": "length",
        "label": "Zone width",
        "axes": ("zone",),
        "status": "derived",
        "description": "Derived from adjacent BPF node positions.",
    },
    "fluid_velocity_cm_s": {
        "unit": "cm/s",
        "quantity": "velocity",
        "label": "Fluid velocity",
        "axes": ("zone",),
        "status": "derived",
        "description": "Zone-centered average of BPF interface velocity; this is the EXO-compatible velocity convention.",
    },
    "zone_outer_velocity_cm_s": {
        "unit": "cm/s",
        "quantity": "velocity",
        "label": "Zone outer interface velocity",
        "axes": ("zone",),
        "status": "derived",
        "description": "Right/outer interface velocity for each zone; direct LOG comparison shows this matches the LOG Velocity column.",
    },
    "pressure_j_cm3": {
        "unit": "J/cm3",
        "label": "Plasma pressure",
        "axes": ("zone",),
        "status": "derived",
        "description": "Plasma pressure derived as ion pressure plus electron pressure. It excludes radiation pressure.",
        "quantity": "plasma_pressure",
    },
    "total_pressure_j_cm3": {
        "unit": "J/cm3",
        "label": "Total pressure",
        "axes": ("zone",),
        "status": "derived",
        "description": "Total pressure derived as ion plus electron plus radiation pressure.",
        "quantity": "total_pressure",
    },
    "radiation_net_heating_j_g_s": {
        "unit": "J/g/s",
        "label": "Net radiation heating",
        "axes": ("zone",),
        "status": "derived",
        "description": "Derived as radiation heating minus radiation cooling; matches LOG net radiation heating within LOG text precision.",
    },
    "radiation_energy_density_j_cm3": {
        "unit": "J/cm3",
        "label": "Radiation energy density",
        "axes": ("zone",),
        "status": "validated",
        "description": "Interior of BPF record 14. Validated by equality to 3*LOG radiation pressure and density*LOG radiation energy.",
    },
    "radiation_pressure_j_cm3": {
        "unit": "J/cm3",
        "label": "Radiation pressure",
        "axes": ("zone",),
        "status": "validated",
        "description": "Derived as one third of BPF radiation energy density; matches LOG radiation pressure.",
        "quantity": "radiation_pressure",
    },
    "radiation_energy_j_g": {
        "unit": "J/g",
        "quantity": "specific_energy",
        "label": "Radiation energy",
        "axes": ("zone",),
        "status": "validated",
        "description": "Derived as radiation energy density divided by mass density; matches LOG radiation energy within LOG text precision.",
    },
    "laser_source_bpf_sampled_j_g": {
        "unit": "J/g",
        "label": "BPF sampled laser source",
        "axes": ("zone",),
        "status": "estimated",
        "description": "Trapezoidal integration of exported BPF laser-deposition samples, accumulated from zero at the first exported snapshot; it cannot account for pre-export deposition or internal unsampled timesteps.",
        "quantity": "cumulative_specific_energy",
    },
    "laser_source_j_g": {
        "unit": "J/g",
        "quantity": "cumulative_specific_energy",
        "label": "Laser source",
        "axes": ("zone",),
        "status": "derived",
        "description": "Cumulative laser source. With an aligned LOG companion this is copied from LOG LaserSrc; BPF-only files use sampled trapezoidal integration of laser deposition.",
    },
    "ionization_fraction_flags_by_zone": {
        "source": "bpf",
        "unit": "",
        "label": "Ionization fraction flags",
        "axes": ("zone", "bpf_record_value"),
        "status": "mapped",
    },
    "ionization_fractions_by_zone_charge": {
        "source": "bpf",
        "unit": "",
        "label": "Ionization fraction",
        "axes": ("zone", "charge_state"),
        "status": "validated",
        "description": "Charge-state fractions ordered by q=0,1,...; rows sum to one.",
    },
    "charge_state": {
        "unit": "",
        "label": "Charge state",
        "axes": ("charge_state",),
        "status": "derived",
    },
    "dominant_charge_state": {
        "unit": "",
        "label": "Dominant charge state",
        "axes": ("zone",),
        "status": "derived",
    },
}


LOG_COMPATIBLE_BPF_ALIASES: dict[str, str] = {
    "zone_center_cm": "radius",
    "zone_width_cm": "zone_width",
    "mass_density_g_cm3": "density",
    "zone_outer_velocity_cm_s": "velocity",
    "radiation_temperature_eV": "temperature_radiation",
    "ion_temperature_eV": "temperature_i",
    "electron_temperature_eV": "temperature_e",
    "ion_pressure_j_cm3": "pressure_i",
    "electron_pressure_j_cm3": "pressure_e",
    "electron_density_cm3": "electron_density",
    "ion_density_cm3": "ion_density",
    "artificial_viscosity_j_cm3": "artificial_viscosity",
    "ion_energy_j_g": "ion_energy",
    "radiation_heating_j_g_s": "radiation_heating",
    "radiation_cooling_j_g_s": "radiation_cooling",
    "radiation_net_heating_j_g_s": "radiation_net_heating",
    "radiation_pressure_j_cm3": "pressure_radiation",
    "radiation_energy_j_g": "radiation_energy",
    "pressure_j_cm3": "plasma_pressure_j_cm3",
    "total_pressure_j_cm3": "pressure",
    "laser_source_j_g": "laser_source",
    "laser_deposition_j_g_s": "laser_deposition",
}


_SPEC_BY_KEY = {spec.key: spec for spec in BPF_FIXED_RECORD_SPECS}
BPF_QUANTITY_BY_KEY = {
    "node_position_cm": "length",
    "ion_temperature_eV": "temperature",
    "radiation_temperature_eV": "temperature",
    "mass_density_g_cm3": "density",
    "ion_pressure_j_cm3": "pressure",
    "radiation_cooling_j_g_s": "rate",
    "radiation_heating_j_g_s": "rate",
    "ion_energy_j_g": "specific_energy",
    "interface_velocity_cm_s": "velocity",
    "laser_deposition_j_g_s": "rate",
    "artificial_viscosity_j_cm3": "pressure",
    "ion_density_cm3": "number_density",
    "electron_temperature_eV": "temperature",
    "electron_pressure_j_cm3": "pressure",
    "electron_density_cm3": "number_density",
}
_DTYPES = {"f64": np.dtype("<f8"), "i32": np.dtype("<i4")}


def field_metadata_for_bpf_key(key: str) -> dict[str, Any]:
    spec = _SPEC_BY_KEY.get(key)
    if spec is not None:
        metadata = {
            "field_name": spec.key,
            "source": "bpf",
            "dimensions": ("time",) + spec.axes,
            "unit": spec.unit,
            "label": spec.label or spec.key,
            "status": spec.status,
            "description": spec.description,
        }
        quantity = BPF_QUANTITY_BY_KEY.get(spec.key)
        if quantity is not None:
            metadata["quantity"] = quantity
        return metadata
    metadata = dict(DERIVED_BPF_FIELD_METADATA.get(key, {}))
    if metadata:
        metadata.setdefault("field_name", key)
        metadata.setdefault("source", "derived")
        metadata["dimensions"] = ("time",) + tuple(metadata.pop("axes", ()))
        return metadata
    return {
        "field_name": key,
        "source": "bpf",
        "dimensions": ("time",),
        "unit": "",
        "label": key,
        "status": "mapped",
        "description": "",
    }


def alias_metadata(alias: str, canonical: str) -> dict[str, Any]:
    metadata = field_metadata_for_bpf_key(canonical)
    metadata = dict(metadata)
    metadata["field_name"] = alias
    metadata["alias_of"] = canonical
    metadata["source"] = metadata.get("source", "bpf")
    return metadata


def _header_integer(value: float, label: str, path: Path, snapshot_index: int | None = None) -> int:
    numeric = float(value)
    location = f" snapshot {snapshot_index}" if snapshot_index is not None else ""
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{path}{location} has invalid {label}: {numeric!r}.")
    return int(numeric)


def _shape_width(shape: str, layout: BpfLayout) -> int:
    if shape == "n_zones":
        return layout.n_zones
    if shape == "n_nodes":
        return layout.n_nodes
    if shape == "n_zones+2":
        return layout.n_zones + 2
    if shape == "n_freq_bins":
        return layout.n_freq_bins
    if shape == "n_freq_bins+1":
        return layout.n_freq_bins + 1
    return int(shape)


def _decode_ascii(payload: memoryview) -> str:
    return bytes(payload).decode("ascii", errors="replace").strip("\x00 ")


class BpfFile:
    """Memory-mapped HELIOS binary plot file with validated snapshot extraction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._payload_offsets: list[int] = []
        self._payload_lengths: list[int] = []
        self._partial_tail = b""
        try:
            self._scan_records()
            self.layout = self._infer_layout()
        except Exception:
            try:
                self._mmap.close()
            except BufferError:
                # NumPy views may still be referenced by the active traceback;
                # releasing this object after unwinding closes the mapping.
                pass
            self._file.close()
            raise

    def close(self) -> None:
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "BpfFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _scan_records(self) -> None:
        offset = 0
        index = 0
        size = len(self._mmap)
        while offset < size:
            if offset + 4 > size:
                self._partial_tail = self._mmap[offset:size]
                break
            length = struct.unpack_from("<i", self._mmap, offset)[0]
            payload_start = offset + 4
            payload_end = payload_start + length
            if length < 0:
                raise ValueError(f"Invalid BPF record length {length} at record {index}, byte {offset}.")
            if payload_end + 4 > size:
                self._partial_tail = self._mmap[offset:size]
                break
            closing_length = struct.unpack_from("<i", self._mmap, payload_end)[0]
            if closing_length != length:
                raise ValueError(f"BPF Fortran record marker mismatch at record {index}: {length} != {closing_length}.")
            self._payload_offsets.append(payload_start)
            self._payload_lengths.append(length)
            offset = payload_end + 4
            index += 1

    @property
    def record_count(self) -> int:
        return len(self._payload_offsets)

    def payload(self, record_index: int) -> memoryview:
        start = self._payload_offsets[record_index]
        stop = start + self._payload_lengths[record_index]
        return memoryview(self._mmap)[start:stop]

    def array(self, record_index: int, dtype_key: str) -> np.ndarray:
        dtype = _DTYPES[dtype_key]
        length = self._payload_lengths[record_index]
        if length % dtype.itemsize != 0:
            raise ValueError(f"BPF record {record_index} is not aligned for {dtype_key}.")
        return np.frombuffer(self.payload(record_index), dtype=dtype)

    def _infer_layout(self) -> BpfLayout:
        if self.record_count < 3:
            raise ValueError(f"{self.path} has too few records to be a HELIOS BPF file.")
        header = self.array(2, "f64")
        if header.size < 7:
            raise ValueError(f"{self.path} first BPF snapshot header is too short.")
        n_nodes = _header_integer(header[4], "node count", self.path)
        n_zones = n_nodes - 1
        n_freq_bins = _header_integer(header[6], "frequency-bin count", self.path)
        if n_nodes <= 1 or n_freq_bins <= 0:
            raise ValueError(f"{self.path} has implausible BPF layout values: nodes={n_nodes}, freq_bins={n_freq_bins}.")
        block_records = 69 + 2 * n_zones
        next_record = 2
        n_snapshots = 0
        while next_record + block_records <= self.record_count:
            current_header = self.array(next_record, "f64")
            if current_header.size < 7:
                break
            current_nodes = _header_integer(current_header[4], "node count", self.path)
            current_freq = _header_integer(current_header[6], "frequency-bin count", self.path)
            if current_nodes != n_nodes or current_freq != n_freq_bins:
                break
            n_snapshots += 1
            next_record += block_records
        if n_snapshots == 0:
            raise ValueError(f"{self.path} does not contain a complete BPF snapshot block.")
        trailing_count = self.record_count - next_record
        trailer_lengths = tuple(self._payload_lengths[next_record:])
        trailer_payloads = tuple(bytes(self.payload(index)) for index in range(next_record, self.record_count))
        recognized_zero_trailer = trailer_lengths == (240,) and trailer_payloads == (bytes(240),) and not self._partial_tail
        if trailer_lengths and not recognized_zero_trailer:
            if trailing_count >= block_records or not self._valid_incomplete_snapshot_prefix(
                next_record, n_nodes, n_zones, n_freq_bins, block_records
            ):
                raise ValueError(
                    f"{self.path} has unrecognized trailing BPF records after {n_snapshots} complete snapshots: "
                    f"lengths={trailer_lengths}."
                )
        if self._partial_tail and not self._valid_partial_final_record(
            next_record, trailing_count, n_nodes, n_zones, n_freq_bins, block_records
        ):
            raise ValueError(f"{self.path} has corrupt partial BPF data after {n_snapshots} complete snapshots.")
        incomplete_final_snapshot = bool((trailing_count or self._partial_tail) and not recognized_zero_trailer)
        first_header = self.array(2, "f64")
        last_header = self.array(self.snapshot_start(n_snapshots - 1, block_records), "f64")
        return BpfLayout(
            path=self.path,
            run_date=_decode_ascii(self.payload(0)),
            run_clock=_decode_ascii(self.payload(1)),
            record_count=self.record_count,
            n_nodes=n_nodes,
            n_zones=n_zones,
            n_freq_bins=n_freq_bins,
            block_records=block_records,
            n_snapshots=n_snapshots,
            trailing_records=self.record_count - next_record,
            trailing_record_lengths=trailer_lengths,
            recognized_zero_trailer=recognized_zero_trailer,
            incomplete_final_snapshot=incomplete_final_snapshot,
            partial_final_record_bytes=len(self._partial_tail),
            first_time_s=float(first_header[1]),
            last_time_s=float(last_header[1]),
            first_cycle=_header_integer(first_header[2], "cycle", self.path, 0),
            last_cycle=_header_integer(last_header[2], "cycle", self.path, n_snapshots - 1),
        )

    def _expected_record_bytes(self, relative_index: int, n_nodes: int, n_zones: int, n_freq_bins: int) -> int:
        spec = next((item for item in BPF_FIXED_RECORD_SPECS if item.rel == relative_index), None)
        if spec is not None:
            layout = SimpleNamespace(n_nodes=n_nodes, n_zones=n_zones, n_freq_bins=n_freq_bins)
            return _shape_width(spec.shape, layout) * _DTYPES[spec.dtype].itemsize
        if relative_index >= 69:
            ion_index = relative_index - 69
            return 4 if ion_index % 2 == 0 else self._payload_lengths[69 + 1] if len(self._payload_lengths) > 70 else 0
        return 0

    def _valid_incomplete_snapshot_prefix(
        self, start: int, n_nodes: int, n_zones: int, n_freq_bins: int, block_records: int
    ) -> bool:
        prefix_count = self.record_count - start
        if prefix_count >= block_records:
            return False
        fraction_width = None
        for relative in range(prefix_count):
            record_index = start + relative
            expected = self._expected_record_bytes(relative, n_nodes, n_zones, n_freq_bins)
            if relative == 70:
                fraction_width = self._payload_lengths[record_index]
                expected = fraction_width
            elif relative >= 69 and (relative - 69) % 2 == 1:
                expected = fraction_width or 0
            if expected <= 0 or self._payload_lengths[record_index] != expected:
                return False
            if relative == 0:
                header = self.array(record_index, "f64")
                try:
                    if _header_integer(header[4], "node count", self.path) != n_nodes or _header_integer(header[6], "frequency-bin count", self.path) != n_freq_bins:
                        return False
                except (IndexError, ValueError):
                    return False
        return True

    def _valid_partial_final_record(
        self, start: int, prefix_count: int, n_nodes: int, n_zones: int, n_freq_bins: int, block_records: int
    ) -> bool:
        if prefix_count >= block_records or not self._valid_incomplete_snapshot_prefix(
            start, n_nodes, n_zones, n_freq_bins, block_records
        ):
            return False
        relative = prefix_count
        expected = self._expected_record_bytes(relative, n_nodes, n_zones, n_freq_bins)
        if relative >= 69 and (relative - 69) % 2 == 1:
            fraction_relative = relative - 2 if relative > 70 else relative
            fraction_index = start + fraction_relative
            if fraction_index >= self.record_count and start >= block_records:
                fraction_index -= block_records
            expected = self._payload_lengths[fraction_index] if 0 <= fraction_index < self.record_count else 0
        if expected <= 0:
            return False
        tail = self._partial_tail
        marker = struct.pack("<i", expected)
        if len(tail) < 4:
            return tail == marker[: len(tail)]
        declared = struct.unpack_from("<i", tail, 0)[0]
        if declared != expected:
            return False
        closing_offset = expected + 4
        if len(tail) > closing_offset:
            closing_prefix = tail[closing_offset:]
            if closing_prefix != marker[: len(closing_prefix)]:
                return False
        return len(tail) < expected + 8

    @staticmethod
    def snapshot_start(snapshot_index: int, block_records: int) -> int:
        return 2 + snapshot_index * block_records

    def _snapshot_record_start(self, snapshot_index: int) -> int:
        if snapshot_index < 0:
            snapshot_index += self.layout.n_snapshots
        if snapshot_index < 0 or snapshot_index >= self.layout.n_snapshots:
            raise IndexError(f"BPF snapshot index {snapshot_index} outside 0..{self.layout.n_snapshots - 1}.")
        return self.snapshot_start(snapshot_index, self.layout.block_records)

    def extract_snapshot(self, snapshot_index: int) -> BpfSnapshot:
        start = self._snapshot_record_start(snapshot_index)
        layout = self.layout
        fields: dict[str, np.ndarray] = {}
        for spec in BPF_FIXED_RECORD_SPECS:
            values = self.array(start + spec.rel, spec.dtype)
            expected = _shape_width(spec.shape, layout)
            if values.size != expected:
                raise ValueError(
                    f"{self.path} snapshot {snapshot_index} record {spec.rel} ({spec.key}) "
                    f"has width {values.size}, expected {expected}."
                )
            fields[spec.key] = values.copy()

        node_position = fields["node_position_cm"]
        fields["zone_center_cm"] = 0.5 * (node_position[:-1] + node_position[1:])
        fields["zone_width_cm"] = np.diff(node_position)
        interface_velocity = fields["interface_velocity_cm_s"]
        fields["fluid_velocity_cm_s"] = 0.5 * (interface_velocity[:-1] + interface_velocity[1:])
        fields["zone_outer_velocity_cm_s"] = interface_velocity[1:].copy()
        fields["pressure_j_cm3"] = plasma_pressure(fields["ion_pressure_j_cm3"], fields["electron_pressure_j_cm3"])
        fields["radiation_net_heating_j_g_s"] = fields["radiation_heating_j_g_s"] - fields["radiation_cooling_j_g_s"]
        radiation_energy_density = fields["bpf_record_14"][1:-1].copy()
        fields["radiation_energy_density_j_cm3"] = radiation_energy_density
        fields["radiation_pressure_j_cm3"] = radiation_energy_density / 3.0
        fields["total_pressure_j_cm3"] = total_pressure(
            fields["ion_pressure_j_cm3"], fields["electron_pressure_j_cm3"], fields["radiation_pressure_j_cm3"]
        )
        fields["radiation_energy_j_g"] = radiation_energy_density / np.where(
            fields["mass_density_g_cm3"] != 0.0,
            fields["mass_density_g_cm3"],
            np.nan,
        )

        ion_start = start + 69
        first_fraction = self.array(ion_start + 1, "f64")
        charge_width = int(first_fraction.size)
        if charge_width <= 0:
            raise ValueError(f"{self.path} snapshot {snapshot_index} has empty ionization fraction records.")
        flags = np.empty((layout.n_zones, 1), dtype=np.int32)
        fractions = np.empty((layout.n_zones, charge_width), dtype=np.float64)
        for zone_index in range(layout.n_zones):
            flag_values = self.array(ion_start + 2 * zone_index, "i32")
            fraction_values = self.array(ion_start + 2 * zone_index + 1, "f64")
            if flag_values.size != 1:
                raise ValueError(f"{self.path} snapshot {snapshot_index} zone {zone_index} has {flag_values.size} ion flags.")
            if fraction_values.size != charge_width:
                raise ValueError(
                    f"{self.path} snapshot {snapshot_index} zone {zone_index} ionization width "
                    f"{fraction_values.size} != {charge_width}."
                )
            flags[zone_index, 0] = int(flag_values[0])
            fractions[zone_index, :] = fraction_values
        self._validate_snapshot(snapshot_index, fields, fractions)
        fields["ionization_fraction_flags_by_zone"] = flags
        fields["ionization_fractions_by_zone_charge"] = fractions
        charge_states = np.arange(charge_width, dtype=np.float64)
        fields["charge_state"] = charge_states
        fields["dominant_charge_state"] = np.argmax(fractions, axis=1).astype(np.float64)

        header = fields["snapshot_header"]
        cycle = _header_integer(header[2], "cycle", self.path, snapshot_index)
        return BpfSnapshot(
            index=snapshot_index,
            time_s=float(header[1]),
            cycle=cycle,
            fields=fields,
        )

    def _validate_snapshot(self, snapshot_index: int, fields: dict[str, np.ndarray], fractions: np.ndarray) -> None:
        header = fields["snapshot_header"]
        time_s = float(header[1])
        if not np.isfinite(time_s):
            raise ValueError(f"{self.path} snapshot {snapshot_index} has non-finite time.")
        freq_edges = fields["frequency_group_boundaries_eV"]
        if not np.all(np.isfinite(freq_edges)) or not np.all(np.diff(freq_edges) > 0.0):
            raise ValueError(f"{self.path} snapshot {snapshot_index} has invalid frequency group boundaries.")
        for key in (
            "node_position_cm",
            "mass_density_g_cm3",
            "ion_temperature_eV",
            "electron_temperature_eV",
            "interface_velocity_cm_s",
        ):
            values = fields[key]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{self.path} snapshot {snapshot_index} field {key} contains non-finite values.")
        if not np.all(np.diff(fields["node_position_cm"]) > 0.0):
            raise ValueError(f"{self.path} snapshot {snapshot_index} has non-positive cell widths.")
        if np.any(fields["mass_density_g_cm3"] < 0.0):
            raise ValueError(f"{self.path} snapshot {snapshot_index} has negative mass density.")
        if not np.all(np.isfinite(fractions)):
            raise ValueError(f"{self.path} snapshot {snapshot_index} contains non-finite ionization fractions.")
        if np.nanmin(fractions) < -1.0e-8:
            raise ValueError(f"{self.path} snapshot {snapshot_index} contains negative ionization fractions.")
        row_sums = fractions.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1.0e-6, atol=1.0e-8):
            raise ValueError(
                f"{self.path} snapshot {snapshot_index} ionization fractions do not sum to one "
                f"(min={float(row_sums.min()):.6g}, max={float(row_sums.max()):.6g})."
            )

    def iter_snapshot_headers(self) -> Iterator[tuple[float, int]]:
        """Yield validated (time, cycle) pairs without decoding field records."""
        previous_time = -np.inf
        previous_cycle = -1
        for index in range(self.layout.n_snapshots):
            header = self.array(self.snapshot_start(index, self.layout.block_records), "f64")
            if header.size < 3:
                raise ValueError(f"{self.path} snapshot {index} has a short header.")
            time_s = float(header[1])
            if not np.isfinite(time_s):
                raise ValueError(f"{self.path} snapshot {index} has non-finite time.")
            cycle = _header_integer(header[2], "cycle", self.path, index)
            if time_s < previous_time or cycle < previous_cycle:
                raise ValueError(f"{self.path} snapshot {index} has decreasing time or cycle metadata.")
            previous_time = time_s
            previous_cycle = cycle
            yield time_s, cycle

    def iter_snapshots(self) -> Iterator[BpfSnapshot]:
        previous_time = -np.inf
        previous_cycle = -1
        for index in range(self.layout.n_snapshots):
            snapshot = self.extract_snapshot(index)
            if snapshot.time_s < previous_time:
                raise ValueError(f"{self.path} BPF snapshot times are not nondecreasing at snapshot {index}.")
            if snapshot.cycle < previous_cycle:
                raise ValueError(f"{self.path} BPF cycles are not nondecreasing at snapshot {index}.")
            previous_time = snapshot.time_s
            previous_cycle = snapshot.cycle
            yield snapshot


def open_bpf(path: str | Path) -> BpfFile:
    return BpfFile(path)
