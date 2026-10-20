from __future__ import annotations

import json
import logging
import re
import tempfile
import os
import secrets
import stat
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from helios.instrumentation import record_duration
from .bpf import LOG_COMPATIBLE_BPF_ALIASES, BpfFile, alias_metadata, field_metadata_for_bpf_key
from .document import diagnostic_value_width, flatten_snapshot_diagnostics, normalize_diagnostic_value, reconcile_diagnostic_width
from .parser import DEFAULT_TOKENIZER, HeliosParser

LOGGER = logging.getLogger(__name__)
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
H5D_SCHEMA_VERSION = "2.1"
SUPPORTED_INPUT_SUFFIXES = {".log", ".bpf"}


@dataclass(frozen=True, slots=True)
class WriteProgress:
    """Progress payload emitted during HDF5 conversion."""

    stage: str
    current: int
    total: int
    fraction: float
    message: str


@dataclass(frozen=True, slots=True)
class ParseSourceSelection:
    requested_path: Path
    primary_path: Path
    bpf_path: Path | None
    log_path: Path | None
    mode: str
    source_precedence: str


def _set_unit_attrs(dataset: h5py.Dataset, unit: str) -> None:
    dataset.attrs["unit"] = unit
    dataset.attrs["units"] = unit


def _json_attr(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _set_field_metadata(dataset: h5py.Dataset, metadata: dict[str, Any]) -> None:
    unit = str(metadata.get("unit", metadata.get("units", "")) or "")
    _set_unit_attrs(dataset, unit)
    for key in ("field_name", "source", "label", "status", "description", "alias_of", "quantity"):
        if key in metadata and metadata[key] is not None:
            dataset.attrs[key] = str(metadata[key])
    dimensions = metadata.get("dimensions")
    if dimensions is not None:
        dataset.attrs["dimensions"] = _json_attr(tuple(str(value) for value in dimensions))
    plotting_hints = metadata.get("plotting_hints")
    if plotting_hints is not None:
        dataset.attrs["plotting_hints"] = _json_attr(plotting_hints)


def _write_field_metadata_group(field_metadata_group: h5py.Group, name: str, metadata: dict[str, Any]) -> None:
    if name in field_metadata_group:
        del field_metadata_group[name]
    group = field_metadata_group.create_group(name)
    stored = dict(metadata)
    stored.setdefault("field_name", name)
    for key, value in stored.items():
        if isinstance(value, (dict, list, tuple)):
            group.attrs[key] = _json_attr(value)
        elif value is None:
            group.attrs[key] = ""
        else:
            group.attrs[key] = str(value)


def _create_1d(group: h5py.Group, name: str, shape: int, dtype, unit: str, *, maxshape=None, chunks=None) -> h5py.Dataset:
    dataset = group.create_dataset(name, shape=(shape,), dtype=dtype, maxshape=maxshape, chunks=chunks)
    _set_unit_attrs(dataset, unit)
    return dataset


def _field_chunk_shape(n_snapshots: int, n_zones: int) -> tuple[int, int]:
    # Viewer and analysis access are snapshot-major first: single-snapshot reads
    # and short sequential scrubs dominate. Keep each chunk row-complete and
    # only span a small number of snapshots so snapshot reads touch one chunk
    # while zone-column access stays acceptable as a secondary pattern.
    time_chunk = min(max(1, n_snapshots), 16)
    zone_chunk = max(1, n_zones)
    return (time_chunk, zone_chunk)


def _time_chunk_shape(capacity: int) -> tuple[int]:
    return (min(max(1, capacity), 1024),)


def _create_field_dataset(
    group: h5py.Group,
    name: str,
    capacity: int,
    n_zones: int,
    unit: str,
    compression: str | None,
    *,
    metadata: dict[str, Any] | None = None,
    field_metadata_group: h5py.Group | None = None,
) -> h5py.Dataset:
    dataset = group.create_dataset(
        name,
        shape=(capacity, n_zones),
        maxshape=(None, n_zones),
        dtype=np.float64,
        fillvalue=np.nan,
        compression=compression,
        chunks=_field_chunk_shape(capacity, n_zones),
    )
    if metadata is None:
        _set_unit_attrs(dataset, unit)
    else:
        metadata = dict(metadata)
        metadata.setdefault("field_name", name)
        metadata.setdefault("unit", unit)
        metadata.setdefault("dimensions", ("time", "zone"))
        _set_field_metadata(dataset, metadata)
        if field_metadata_group is not None:
            _write_field_metadata_group(field_metadata_group, name, metadata)
    return dataset


def _create_dynamic_dataset(
    group: h5py.Group,
    name: str,
    capacity: int,
    width: int | None,
    unit: str,
    compression: str | None,
) -> h5py.Dataset:
    shape = (capacity,) if width is None else (capacity, width)
    maxshape = (None,) if width is None else (None, None)
    chunks = _time_chunk_shape(capacity) if width is None else _field_chunk_shape(capacity, width)
    dataset = group.create_dataset(
        name,
        shape=shape,
        maxshape=maxshape,
        dtype=np.float64,
        fillvalue=np.nan,
        compression=compression,
        chunks=chunks,
    )
    _set_unit_attrs(dataset, unit)
    return dataset


def _write_scalar_dataset(group: h5py.Group, name: str, value: Any, unit: str) -> None:
    if isinstance(value, (dict, list, tuple)):
        dataset = group.create_dataset(name, data=json.dumps(value), dtype=STRING_DTYPE)
    elif isinstance(value, str):
        dataset = group.create_dataset(name, data=value, dtype=STRING_DTYPE)
    elif value is None:
        dataset = group.create_dataset(name, data="", dtype=STRING_DTYPE)
    else:
        dataset = group.create_dataset(name, data=value)
    _set_unit_attrs(dataset, unit)


def _write_array_dataset(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
    unit: str,
    compression: str | None = None,
) -> None:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U"}:
        dataset = group.create_dataset(name, data=array.astype(STRING_DTYPE), dtype=STRING_DTYPE)
    else:
        dataset = group.create_dataset(name, data=array, compression=compression)
    _set_unit_attrs(dataset, unit)


def _replace_group(group: h5py.Group, name: str) -> h5py.Group:
    if name in group:
        del group[name]
    return group.create_group(name)


def _annotate_coordinate_dataset(
    dataset: h5py.Dataset,
    *,
    coordinate_name: str,
    location: str,
    dynamic: bool,
    legacy_alias: bool = False,
    role: str = "coordinate",
) -> None:
    dataset.attrs["coordinate_name"] = str(coordinate_name)
    dataset.attrs["coordinate_location"] = str(location)
    dataset.attrs["coordinate_dynamic"] = bool(dynamic)
    dataset.attrs["coordinate_role"] = str(role)
    dataset.attrs["legacy_coordinate_alias"] = bool(legacy_alias)


def _ensure_group(parent: h5py.Group, path: tuple[str, ...]) -> h5py.Group:
    group = parent
    for part in path:
        group = group.require_group(part)
    return group


def _write_nested_mapping(
    group: h5py.Group,
    mapping: dict[str, Any],
    *,
    unit_resolver,
    compression: str | None = None,
    path: tuple[str, ...] = (),
) -> None:
    for name, value in mapping.items():
        current_path = path + (name,)
        if isinstance(value, dict):
            _write_nested_mapping(
                group.require_group(name),
                value,
                unit_resolver=unit_resolver,
                compression=compression,
                path=current_path,
            )
            continue
        unit = unit_resolver(current_path)
        if isinstance(value, np.ndarray):
            _write_array_dataset(group, name, value, unit, compression=compression)
        else:
            _write_scalar_dataset(group, name, value, unit)


def _diagnostic_unit(path: tuple[str, ...], diagnostic_units: dict[str, str]) -> str:
    if not path:
        return ""
    if path[0] == "radiation_boundary_fluxes":
        return diagnostic_units.get("radiation_boundary_fluxes", "J/s/cm2")
    if path[0] in {"energy_summary", "energy_exchange", "energy_balance"}:
        return diagnostic_units.get("energy", "")
    return ""


def _input_parameter_unit(path: tuple[str, ...]) -> str:
    units = {
        ("hydro", "boundary_temperature_rmin"): "eV",
        ("hydro", "boundary_temperature_rmax"): "eV",
        ("hydro", "quiet_start_temperature"): "eV",
        ("radiative_transfer", "photon_energy_grid"): "eV",
        ("radiative_transfer", "frequency_gridding", "min_photon_energy"): "eV",
        ("radiative_transfer", "frequency_gridding", "max_photon_energy"): "eV",
        ("laser_source", "wavelength"): "microns",
        ("laser_source", "power_table", "time"): "s",
        ("laser_source", "power_table", "power"): "TW",
        ("time_control", "max_simulation_time"): "s",
        ("time_control", "initial_time_step"): "s",
        ("time_control", "min_time_step"): "s",
        ("time_control", "max_time_step"): "s",
    }
    if path[:2] == ("time_control", "time_step_controls"):
        return ""
    return units.get(path, "")


def _find_bpf_companion(source: Path) -> Path | None:
    if source.suffix.lower() == ".bpf":
        return source if source.exists() else None
    candidate = source.with_suffix(".bpf")
    return candidate if candidate.exists() else None


def _find_log_companion(source: Path) -> Path | None:
    if source.suffix.lower() == ".log":
        return source if source.exists() else None
    candidate = source.with_suffix(".log")
    return candidate if candidate.exists() else None


def resolve_parse_sources(input_path: str | Path) -> ParseSourceSelection:
    requested = Path(input_path)
    suffix = requested.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(f"Unsupported HELIOS input type '{requested.suffix}'. Expected one of: .log, .bpf.")
    if not requested.exists():
        raise FileNotFoundError(requested)

    if suffix == ".bpf":
        log_path = _find_log_companion(requested)
        mode = "bpf_with_log_metadata" if log_path is not None else "bpf_only"
        return ParseSourceSelection(
            requested_path=requested,
            primary_path=requested,
            bpf_path=requested,
            log_path=log_path,
            mode=mode,
            source_precedence="bpf_primary_log_metadata_exo_optional" if log_path is not None else "bpf_primary",
        )

    bpf_path = _find_bpf_companion(requested)
    if bpf_path is not None:
        return ParseSourceSelection(
            requested_path=requested,
            primary_path=bpf_path,
            bpf_path=bpf_path,
            log_path=requested,
            mode="log_input_bpf_primary",
            source_precedence="bpf_primary_log_metadata_exo_optional",
        )
    return ParseSourceSelection(
        requested_path=requested,
        primary_path=requested,
        bpf_path=None,
        log_path=requested,
        mode="log_only",
        source_precedence="log_only",
    )


def _bpf_chunk_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    time_chunk = min(max(1, shape[0]), 16)
    if len(shape) == 1:
        return (time_chunk,)
    return (time_chunk,) + tuple(max(1, width) for width in shape[1:])


def _create_bpf_field_dataset(
    fields_group: h5py.Group,
    field_metadata_group: h5py.Group,
    name: str,
    sample: np.ndarray,
    n_snapshots: int,
    compression: str | None,
    metadata: dict[str, Any],
) -> h5py.Dataset:
    array = np.asarray(sample)
    shape = (int(n_snapshots),) + tuple(int(value) for value in array.shape)
    dataset = fields_group.create_dataset(
        name,
        shape=shape,
        dtype=array.dtype,
        compression=compression if array.dtype.kind == "f" else None,
        chunks=_bpf_chunk_shape(shape),
    )
    _set_field_metadata(dataset, metadata)
    _write_field_metadata_group(field_metadata_group, name, metadata)
    return dataset


def _bpf_minimal_regions(region_index_by_zone: np.ndarray, n_zones: int) -> dict[str, np.ndarray]:
    region_values = np.asarray(region_index_by_zone, dtype=np.int32)
    if region_values.size != n_zones:
        region_values = np.ones(n_zones, dtype=np.int32)
    ids: list[int] = []
    starts: list[int] = []
    stops: list[int] = []
    start = 0
    for index in range(1, n_zones + 1):
        if index == n_zones or region_values[index] != region_values[start]:
            ids.append(int(region_values[start]))
            starts.append(start + 1)
            stops.append(index)
            start = index
    return {
        "region_index": np.asarray(ids, dtype=np.int32),
        "min_zone_index": np.asarray(starts, dtype=np.int32),
        "max_zone_index": np.asarray(stops, dtype=np.int32),
        "material_index": np.asarray(ids, dtype=np.int32),
        "material_table_index": np.asarray([abs(value) for value in ids], dtype=np.int32),
    }


def _write_minimal_bpf_header_groups(
    handle: h5py.File,
    first_fields: dict[str, np.ndarray],
    layout,
    compression: str | None,
) -> None:
    node_position = np.asarray(first_fields["node_position_cm"], dtype=np.float64)
    zone_width = np.diff(node_position)
    zone_center = 0.5 * (node_position[:-1] + node_position[1:])
    region_index_by_zone = np.asarray(first_fields.get("region_index_by_zone", np.ones(layout.n_zones)), dtype=np.int32)

    grid_group = handle.create_group("grid")
    for name, values, unit in (
        ("coordinate_center", zone_center, "cm"),
        ("coordinate_edge", node_position, "cm"),
        ("x", zone_center, "cm"),
        ("zone_id", np.arange(1, layout.n_zones + 1, dtype=np.int32), ""),
        ("zone_width", zone_width, "cm"),
        ("zone_mass", np.asarray(first_fields["zone_mass"], dtype=np.float64), "g/cm**X"),
        ("zone_region_id", region_index_by_zone, ""),
        ("zone_material_index", region_index_by_zone, ""),
    ):
        _write_array_dataset(grid_group, name, np.asarray(values), unit, compression=compression)
    for dataset_name, location, dynamic, legacy_alias, role in (
        ("coordinate_center", "center", False, False, "coordinate"),
        ("coordinate_edge", "edge", False, False, "coordinate"),
        ("x", "center", False, True, "coordinate"),
        ("zone_width", "cell", False, False, "width"),
    ):
        _annotate_coordinate_dataset(
            grid_group[dataset_name],
            coordinate_name="x",
            location=location,
            dynamic=dynamic,
            legacy_alias=legacy_alias,
            role=role,
        )

    regions = _bpf_minimal_regions(region_index_by_zone, layout.n_zones)
    regions_group = handle.create_group("regions")
    for name, values in regions.items():
        _write_array_dataset(regions_group, name, values, "", compression=compression)

    materials_group = handle.create_group("materials")
    material_ids = np.unique(np.abs(region_index_by_zone)).astype(np.int32)
    if material_ids.size == 0:
        material_ids = np.asarray([1], dtype=np.int32)
    _write_array_dataset(materials_group, "index", material_ids, "", compression=None)
    _write_array_dataset(materials_group, "eos_model", np.asarray([""] * material_ids.size, dtype=object), "", compression=None)
    _write_array_dataset(materials_group, "opacity_model", np.asarray([""] * material_ids.size, dtype=object), "", compression=None)


def _write_log_header_groups(
    handle: h5py.File,
    header,
    source: Path,
    compression: str | None,
) -> None:
    coordinate_model = dict(header.metadata.get("coordinate_model", {})) if isinstance(header.metadata.get("coordinate_model", {}), dict) else {}
    coordinate_name = str(coordinate_model.get("coordinate_name", "x"))
    grid_group = handle.create_group("grid")
    for name, values in header.grid.items():
        _write_array_dataset(grid_group, name, np.asarray(values), header.grid_units.get(name, ""), compression=compression)
    for dataset_name, location, dynamic, legacy_alias, role in (
        ("coordinate_center", "center", False, False, "coordinate"),
        ("coordinate_edge", "edge", False, False, "coordinate"),
        ("x", "center", False, True, "coordinate"),
        ("zone_width", "cell", False, False, "width"),
    ):
        if dataset_name in grid_group:
            _annotate_coordinate_dataset(
                grid_group[dataset_name],
                coordinate_name=coordinate_name,
                location=location,
                dynamic=dynamic,
                legacy_alias=legacy_alias,
                role=role,
            )
    regions_group = handle.create_group("regions")
    for name, values in header.regions.items():
        _write_array_dataset(regions_group, name, np.asarray(values), header.region_units.get(name, ""), compression=compression)
    if header.metadata.get("total_mass") is not None:
        _write_scalar_dataset(regions_group, "total_mass", header.metadata["total_mass"], "g/cm**X")
    materials_group = handle.create_group("materials")
    for name, values in header.materials.items():
        _write_array_dataset(materials_group, name, np.asarray(values), header.material_units.get(name, ""), compression=compression)

    metadata_group = handle.create_group("metadata")
    metadata = dict(header.metadata)
    metadata["simulation_name"] = header.simulation_name
    metadata["source_file"] = str(source)
    eos_model = ",".join(dict.fromkeys(str(value) for value in header.materials.get("eos_model", []) if str(value)))
    for key, value in {
        "schema_version": H5D_SCHEMA_VERSION,
        "geometry": metadata.get("geometry"),
        "simulation_name": header.simulation_name,
        "n_zones": header.n_zones,
        "n_regions": header.n_regions,
        "n_materials": header.n_materials,
        "eos_model": eos_model,
        "helios_version": header.code_version,
        "calculation_datetime": header.calculation_datetime,
        "openmp_note": metadata.get("openmp_note"),
        "block_delimiter": header.block_delimiter,
        "source_file": str(source),
    }.items():
        _write_scalar_dataset(metadata_group, key, value, "")
    metadata_group.create_dataset("header_sections", data=np.asarray(header.header_sections, dtype=STRING_DTYPE))
    _set_unit_attrs(metadata_group["header_sections"], "")
    if coordinate_model:
        coordinate_group = metadata_group.create_group("coordinate_model")
        _write_nested_mapping(coordinate_group, coordinate_model, unit_resolver=lambda path: "", compression=None)
    input_group = metadata_group.create_group("input_parameters")
    _write_nested_mapping(input_group, header.input_parameters, unit_resolver=_input_parameter_unit, compression=compression)


def _write_minimal_bpf_metadata(handle: h5py.File, source: Path, bpf: BpfFile) -> h5py.Group:
    metadata_group = handle.create_group("metadata")
    layout = bpf.layout
    for key, value in {
        "schema_version": H5D_SCHEMA_VERSION,
        "geometry": None,
        "simulation_name": source.stem,
        "n_zones": layout.n_zones,
        "n_regions": 0,
        "n_materials": 0,
        "helios_version": None,
        "calculation_datetime": f"{layout.run_date} {layout.run_clock}".strip(),
        "source_file": str(source),
    }.items():
        _write_scalar_dataset(metadata_group, key, value, "")
    metadata_group.create_dataset("header_sections", data=np.asarray([], dtype=STRING_DTYPE))
    _set_unit_attrs(metadata_group["header_sections"], "")
    coordinate_group = metadata_group.create_group("coordinate_model")
    _write_nested_mapping(
        coordinate_group,
        {
            "coordinate_name": "x",
            "static_center_dataset": "coordinate_center",
            "static_edge_dataset": "coordinate_edge",
            "dynamic_center_dataset": "dynamic_coordinate_center",
            "dynamic_edge_dataset": "dynamic_coordinate_edge",
            "width_dataset": "zone_width",
            "legacy_static_center_alias": "x",
            "legacy_dynamic_center_alias": "radius",
        },
        unit_resolver=lambda path: "",
    )
    metadata_group.create_group("input_parameters")
    return metadata_group


def _log_cycle_time_tokens(block_text: str) -> tuple[str, str]:
    lines = block_text.splitlines()
    header = next((index for index, line in enumerate(lines) if re.match(r"^\s*Cycle\s+Time\s+\(s\)", line)), None)
    if header is None or header + 1 >= len(lines):
        raise ValueError("LOG snapshot is missing cycle/time metadata.")
    tokens = lines[header + 1].split()
    if len(tokens) < 2:
        raise ValueError("LOG snapshot has malformed cycle/time metadata.")
    return tokens[0], tokens[1]


def _decimal_token(token: str) -> Decimal:
    normalized = DEFAULT_TOKENIZER.normalize_number_token(token).replace("D", "E").replace("d", "E")
    try:
        value = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric token {token!r}.") from exc
    if not value.is_finite():
        raise ValueError(f"Non-finite numeric token {token!r}.")
    return value


def _log_time_matches_bpf(token: str, bpf_time_s: float) -> bool:
    decimal_time = _decimal_token(token)
    half_quantum = Decimal(1).scaleb(decimal_time.as_tuple().exponent) / Decimal(2)
    center = float(decimal_time)
    roundoff = 4.0 * abs(float(np.spacing(max(abs(center), abs(float(bpf_time_s)), 1.0e-300))))
    return abs(float(bpf_time_s) - center) <= float(half_quantum) + roundoff


def _stage_aligned_log_companion(
    source: Path,
    bpf_path: Path,
    log_path: Path | None,
    staging_path: Path,
    *,
    parser: HeliosParser,
    logger: logging.Logger,
) -> tuple[Any | None, bool]:
    if log_path is None or not log_path.exists():
        return None, False
    explicitly_requested = source.suffix.lower() == ".log"
    try:
        with BpfFile(bpf_path) as bpf, parser.open_document(log_path) as document:
            layout = bpf.layout
            log_header = document.inspect()
            if int(log_header.n_zones) != int(layout.n_zones):
                raise ValueError(f"LOG/BPF zone-count mismatch: log={log_header.n_zones}, bpf={layout.n_zones}.")
            bpf_headers = list(bpf.iter_snapshot_headers())
            blocks = document.index.snapshot_blocks
            aligned = len(blocks) == layout.n_snapshots and len(bpf_headers) == layout.n_snapshots
            snapshot_count = 0
            previous_time: float | None = None
            previous_cycle: int | None = None
            with h5py.File(staging_path, "w") as stage:
                candidate = stage.create_dataset(
                    "laser_source_j_g",
                    shape=(layout.n_snapshots, layout.n_zones),
                    dtype=np.float64,
                    chunks=_field_chunk_shape(layout.n_snapshots, layout.n_zones),
                    fillvalue=np.nan,
                )
                for index, block in enumerate(blocks):
                    block_text = document.buffer.slice(block.span)
                    try:
                        cycle, snapshot_time, values = document.snapshot_parser.parse_laser_source_companion(
                            block_text, layout.n_zones
                        )
                    except Exception:
                        # Match the general streaming parser's recoverable-tail
                        # contract: a damaged final frame invalidates the
                        # companion, while corruption in an earlier frame is an
                        # input error (or optional-companion fallback).
                        if index == len(blocks) - 1 and snapshot_count > 0:
                            aligned = False
                            break
                        raise
                    if previous_time is not None and snapshot_time < previous_time:
                        raise ValueError(
                            f"Snapshot times must be nondecreasing; block {int(block.index)} has "
                            f"{snapshot_time} after {previous_time}."
                        )
                    if previous_cycle is not None and cycle < previous_cycle:
                        raise ValueError(
                            f"Snapshot cycles must be nondecreasing; block {int(block.index)} has "
                            f"{cycle} after {previous_cycle}."
                        )
                    previous_time, previous_cycle = snapshot_time, cycle
                    snapshot_count += 1
                    if index >= layout.n_snapshots:
                        aligned = False
                        continue
                    bpf_time, bpf_cycle = bpf_headers[index]
                    cycle_token, time_token = _log_cycle_time_tokens(block_text)
                    if _decimal_token(cycle_token) != Decimal(bpf_cycle) or cycle != bpf_cycle:
                        aligned = False
                    if not _log_time_matches_bpf(time_token, bpf_time):
                        aligned = False
                    candidate[index] = values
            if snapshot_count != layout.n_snapshots:
                aligned = False
            if not aligned:
                logger.warning("Ignoring unaligned LOG companion %s for BPF %s; BPF metadata and sampled source will be used.", log_path, bpf_path)
                return None, False
            return log_header, True
    except Exception:
        if explicitly_requested:
            raise
        logger.warning("Could not validate optional LOG companion %s; falling back to standalone BPF metadata.", log_path, exc_info=True)
        return None, False


def _write_hdf5_from_bpf(
    source: Path,
    bpf_path: Path,
    target: Path,
    *,
    log_path: Path | None,
    source_precedence: str,
    compression: str | None,
    parser: HeliosParser,
    logger: logging.Logger,
    progress_callback: Callable[[WriteProgress], None] | None,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="helios-log-stage-") as staging_dir:
        stage_path = Path(staging_dir) / "companion.h5"
        log_header, source_aligned = _stage_aligned_log_companion(
            source, bpf_path, log_path, stage_path, parser=parser, logger=logger
        )
        if not stage_path.exists():
            with h5py.File(stage_path, "w"):
                pass
        with h5py.File(stage_path, "r") as staged:
            source_dataset = staged["laser_source_j_g"] if log_header is not None and source_aligned else None
            return _write_hdf5_from_bpf_validated(
                source,
                bpf_path,
                target,
                log_path=log_path if log_header is not None else None,
                source_precedence=source_precedence if log_header is not None else "bpf_primary",
                compression=compression,
                parser=parser,
                logger=logger,
                progress_callback=progress_callback,
                log_header=log_header,
                log_source_dataset=source_dataset,
            )


def _write_hdf5_from_bpf_validated(
    source: Path,
    bpf_path: Path,
    target: Path,
    *,
    log_path: Path | None,
    source_precedence: str,
    compression: str | None,
    parser: HeliosParser,
    logger: logging.Logger,
    progress_callback: Callable[[WriteProgress], None] | None,
    log_header,
    log_source_dataset: h5py.Dataset | None,
) -> Path:
    def emit_progress(stage: str, current: int, total: int, fraction: float, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(WriteProgress(stage, int(current), max(1, int(total)), max(0.0, min(1.0, float(fraction))), message))

    started_write = time.perf_counter()
    emit_progress("prepare", 0, 1, 0.02, f"Opening {bpf_path.name}")
    with BpfFile(bpf_path) as bpf:
        layout = bpf.layout
        snapshots = iter(bpf.iter_snapshots())
        first_snapshot = next(snapshots)
        emit_progress("prepare", 0, layout.n_snapshots, 0.08, f"Preparing BPF layout for {layout.n_snapshots} snapshots")
        with h5py.File(target, "w") as handle:
            handle.attrs["source_file"] = str(source)
            handle.attrs["bpf_source_file"] = str(bpf_path)
            handle.attrs["format"] = "HELIOS H5D/HDF5"
            handle.attrs["schema_version"] = H5D_SCHEMA_VERSION
            if log_header is not None:
                handle.attrs["helios_version"] = log_header.code_version or ""
                _write_log_header_groups(handle, log_header, source, compression)
                metadata_group = handle["metadata"]
                coordinate_model = log_header.metadata.get("coordinate_model", {})
                coordinate_name = str(coordinate_model.get("coordinate_name", "x")) if isinstance(coordinate_model, dict) else "x"
            else:
                handle.attrs["helios_version"] = ""
                _write_minimal_bpf_header_groups(handle, first_snapshot.fields, layout, compression)
                metadata_group = _write_minimal_bpf_metadata(handle, source, bpf)
                coordinate_name = "x"

            source_group = metadata_group.create_group("source_files")
            _write_scalar_dataset(source_group, "bpf", str(bpf_path), "")
            if log_header is not None:
                _write_scalar_dataset(source_group, "log", str(log_path), "")
            _write_scalar_dataset(source_group, "requested", str(source), "")
            _write_scalar_dataset(metadata_group, "source_precedence", source_precedence, "")
            _write_scalar_dataset(metadata_group, "parse_mode", "bpf_primary", "")
            _write_scalar_dataset(metadata_group, "bpf_run_date", layout.run_date, "")
            _write_scalar_dataset(metadata_group, "bpf_run_clock", layout.run_clock, "")
            _write_scalar_dataset(metadata_group, "bpf_record_count", layout.record_count, "")
            _write_scalar_dataset(metadata_group, "bpf_trailing_records", layout.trailing_records, "")
            _write_scalar_dataset(metadata_group, "bpf_recognized_zero_trailer", layout.recognized_zero_trailer, "")
            _write_scalar_dataset(metadata_group, "bpf_incomplete_final_snapshot", layout.incomplete_final_snapshot, "")
            _write_scalar_dataset(metadata_group, "bpf_partial_final_record_bytes", layout.partial_final_record_bytes, "")

            time_group = handle.create_group("time")
            time_dataset = _create_1d(time_group, "time", layout.n_snapshots, np.float64, "s", chunks=_time_chunk_shape(layout.n_snapshots))
            cycle_dataset = _create_1d(time_group, "cycle", layout.n_snapshots, np.int64, "", chunks=_time_chunk_shape(layout.n_snapshots))

            fields_group = handle.create_group("fields")
            field_metadata_group = handle.create_group("field_metadata")
            diagnostics_group = handle.create_group("diagnostics")

            grid_group = handle["grid"]
            dynamic_center = grid_group.create_dataset(
                "dynamic_coordinate_center",
                shape=(layout.n_snapshots, layout.n_zones),
                dtype=np.float64,
                compression=compression,
                chunks=_field_chunk_shape(layout.n_snapshots, layout.n_zones),
            )
            _set_unit_attrs(dynamic_center, "cm")
            _annotate_coordinate_dataset(dynamic_center, coordinate_name=coordinate_name, location="center", dynamic=True)
            dynamic_edge = grid_group.create_dataset(
                "dynamic_coordinate_edge",
                shape=(layout.n_snapshots, layout.n_nodes),
                dtype=np.float64,
                compression=compression,
                chunks=_field_chunk_shape(layout.n_snapshots, layout.n_nodes),
            )
            _set_unit_attrs(dynamic_edge, "cm")
            _annotate_coordinate_dataset(dynamic_edge, coordinate_name=coordinate_name, location="edge", dynamic=True)
            if "photon_energy_boundaries_eV" not in grid_group:
                _write_array_dataset(
                    grid_group,
                    "photon_energy_boundaries_eV",
                    np.asarray(first_snapshot.fields["frequency_group_boundaries_eV"], dtype=np.float64),
                    "eV",
                    compression=compression,
                )

            field_datasets: dict[str, h5py.Dataset] = {}
            for name, values in first_snapshot.fields.items():
                metadata = field_metadata_for_bpf_key(name)
                field_datasets[name] = _create_bpf_field_dataset(
                    fields_group,
                    field_metadata_group,
                    name,
                    np.asarray(values),
                    layout.n_snapshots,
                    compression,
                    metadata,
                )
            laser_source_metadata = field_metadata_for_bpf_key("laser_source_j_g")
            if log_source_dataset is not None:
                laser_source_metadata = dict(laser_source_metadata)
                laser_source_metadata["source"] = "log"
                laser_source_metadata["status"] = "validated"
                laser_source_metadata["description"] = (
                    "Copied from aligned LOG LaserSrc because HELIOS integrates this cumulative source "
                    "on internal timesteps; BPF-only files fall back to sampled trapezoidal integration."
                )
            laser_source_dataset = _create_bpf_field_dataset(
                fields_group,
                field_metadata_group,
                "laser_source_j_g",
                np.zeros(layout.n_zones, dtype=np.float64),
                layout.n_snapshots,
                compression,
                laser_source_metadata,
            )
            bpf_sampled_source_dataset = _create_bpf_field_dataset(
                fields_group,
                field_metadata_group,
                "laser_source_bpf_sampled_j_g",
                np.zeros(layout.n_zones, dtype=np.float64),
                layout.n_snapshots,
                compression,
                field_metadata_for_bpf_key("laser_source_bpf_sampled_j_g"),
            )

            def snapshot_iter():
                yield first_snapshot
                yield from snapshots

            laser_cumulative = np.zeros(layout.n_zones, dtype=np.float64)
            previous_laser_deposition: np.ndarray | None = None
            previous_time_s: float | None = None
            for written_index, snapshot in enumerate(snapshot_iter()):
                time_dataset[written_index] = snapshot.time_s
                cycle_dataset[written_index] = snapshot.cycle
                dynamic_center[written_index, :] = snapshot.fields["zone_center_cm"]
                dynamic_edge[written_index, :] = snapshot.fields["node_position_cm"]
                for name, dataset in field_datasets.items():
                    dataset[written_index, ...] = snapshot.fields[name]
                laser_deposition = np.asarray(snapshot.fields["laser_deposition_j_g_s"], dtype=np.float64)
                if previous_time_s is not None and previous_laser_deposition is not None:
                    dt = max(0.0, float(snapshot.time_s) - float(previous_time_s))
                    laser_cumulative += 0.5 * (previous_laser_deposition + laser_deposition) * dt
                bpf_sampled_source_dataset[written_index, :] = laser_cumulative
                if log_source_dataset is not None:
                    laser_source_dataset[written_index, :] = log_source_dataset[written_index]
                else:
                    laser_source_dataset[written_index, :] = laser_cumulative
                previous_time_s = float(snapshot.time_s)
                previous_laser_deposition = laser_deposition.copy()
                emit_progress(
                    "snapshots",
                    written_index + 1,
                    layout.n_snapshots,
                    0.10 + 0.84 * ((written_index + 1) / max(1, layout.n_snapshots)),
                    f"Writing BPF snapshots ({written_index + 1}/{layout.n_snapshots})",
                )

            for canonical, alias in LOG_COMPATIBLE_BPF_ALIASES.items():
                if canonical in fields_group and alias not in fields_group:
                    fields_group[alias] = fields_group[canonical]
                    if canonical in field_metadata_group:
                        metadata = {key: value for key, value in field_metadata_group[canonical].attrs.items()}
                        metadata["field_name"] = alias
                        metadata["alias_of"] = canonical
                    else:
                        metadata = alias_metadata(alias, canonical)
                    _write_field_metadata_group(field_metadata_group, alias, metadata)

            available_fields = sorted(fields_group.keys())
            metadata_group.create_dataset("available_fields", data=np.asarray(available_fields, dtype=STRING_DTYPE))
            _set_unit_attrs(metadata_group["available_fields"], "")
            _write_scalar_dataset(metadata_group, "n_snapshots", layout.n_snapshots, "")
            run_status_group = _replace_group(metadata_group, "run_status")
            bpf_status = "truncated" if layout.incomplete_final_snapshot else "complete"
            _write_nested_mapping(
                run_status_group,
                {
                    "state": bpf_status,
                    "source": "bpf_partial_final_snapshot" if layout.incomplete_final_snapshot else "bpf",
                    "indexed_snapshot_count": layout.n_snapshots + int(layout.incomplete_final_snapshot),
                    "valid_snapshot_count": layout.n_snapshots,
                    "last_valid_snapshot_time_s": layout.last_time_s,
                    "dropped_partial_final_block": layout.incomplete_final_snapshot,
                    "notes": ("Incomplete final BPF snapshot dropped; complete snapshots retained.",)
                    if layout.incomplete_final_snapshot
                    else ("BPF primary parse",),
                },
                unit_resolver=lambda path: "s" if path[-1] == "last_valid_snapshot_time_s" else "",
            )
            emit_progress("finalize", layout.n_snapshots, layout.n_snapshots, 0.98, f"Finalizing {target.name}")
            logger.info("Wrote %s BPF snapshots to %s", layout.n_snapshots, target)
            del snapshot, first_snapshot
    record_duration("hdf5.write", time.perf_counter() - started_write)
    return target


def _write_hdf5_unpublished(
    input_path: str | Path,
    output_path: str | Path,
    *,
    compression: str | None = None,
    overwrite: bool = False,
    parser: HeliosParser | None = None,
    logger: logging.Logger | None = None,
    progress_callback: Callable[[WriteProgress], None] | None = None,
) -> Path:
    active_logger = logger or LOGGER
    active_parser = parser or HeliosParser(active_logger)
    source = Path(input_path)
    target = Path(output_path)
    selection = resolve_parse_sources(source)
    if selection.bpf_path is not None:
        return _write_hdf5_from_bpf(
            selection.requested_path,
            selection.bpf_path,
            target,
            log_path=selection.log_path,
            source_precedence=selection.source_precedence,
            compression=compression,
            parser=active_parser,
            logger=active_logger,
            progress_callback=progress_callback,
        )

    def emit_progress(stage: str, current: int, total: int, fraction: float, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(
            WriteProgress(
                stage=stage,
                current=int(current),
                total=max(1, int(total)),
                fraction=max(0.0, min(1.0, float(fraction))),
                message=message,
            )
        )

    emit_progress("prepare", 0, 1, 0.02, f"Opening {source.name}")
    started_write = time.perf_counter()
    with active_parser.open_document(source) as document, h5py.File(target, "w") as handle:
        header = document.inspect()
        coordinate_model = dict(header.metadata.get("coordinate_model", {})) if isinstance(header.metadata.get("coordinate_model", {}), dict) else {}
        coordinate_name = str(coordinate_model.get("coordinate_name", "x"))
        emit_progress("prepare", 0, 1, 0.08, f"Inspecting {source.name}")
        handle.attrs["source_file"] = str(source)
        handle.attrs["helios_version"] = header.code_version or ""
        handle.attrs["format"] = "HELIOS log converted to HDF5"
        handle.attrs["schema_version"] = H5D_SCHEMA_VERSION

        grid_group = handle.create_group("grid")
        for name, values in header.grid.items():
            _write_array_dataset(grid_group, name, np.asarray(values), header.grid_units.get(name, ""), compression=compression)
        for dataset_name, location, dynamic, legacy_alias, role in (
            ("coordinate_center", "center", False, False, "coordinate"),
            ("coordinate_edge", "edge", False, False, "coordinate"),
            ("x", "center", False, True, "coordinate"),
            ("zone_width", "cell", False, False, "width"),
        ):
            if dataset_name in grid_group:
                _annotate_coordinate_dataset(
                    grid_group[dataset_name],
                    coordinate_name=coordinate_name,
                    location=location,
                    dynamic=dynamic,
                    legacy_alias=legacy_alias,
                    role=role,
                )

        regions_group = handle.create_group("regions")
        for name, values in header.regions.items():
            _write_array_dataset(regions_group, name, np.asarray(values), header.region_units.get(name, ""), compression=compression)
        if header.metadata.get("total_mass") is not None:
            _write_scalar_dataset(regions_group, "total_mass", header.metadata["total_mass"], "g/cm**X")

        materials_group = handle.create_group("materials")
        for name, values in header.materials.items():
            _write_array_dataset(materials_group, name, np.asarray(values), header.material_units.get(name, ""), compression=compression)

        metadata_group = handle.create_group("metadata")
        metadata = dict(header.metadata)
        metadata["simulation_name"] = header.simulation_name
        metadata["source_file"] = str(source)
        eos_model = ",".join(dict.fromkeys(str(value) for value in header.materials.get("eos_model", []) if str(value)))
        for key, value in {
            "schema_version": H5D_SCHEMA_VERSION,
            "geometry": metadata.get("geometry"),
            "simulation_name": header.simulation_name,
            "n_zones": header.n_zones,
            "n_regions": header.n_regions,
            "n_materials": header.n_materials,
            "eos_model": eos_model,
            "helios_version": header.code_version,
            "calculation_datetime": header.calculation_datetime,
            "openmp_note": metadata.get("openmp_note"),
            "block_delimiter": header.block_delimiter,
            "source_file": str(source),
        }.items():
            _write_scalar_dataset(metadata_group, key, value, "")
        source_group = metadata_group.create_group("source_files")
        _write_scalar_dataset(source_group, "log", str(source), "")
        _write_scalar_dataset(source_group, "requested", str(source), "")
        _write_scalar_dataset(metadata_group, "source_precedence", "log_only", "")
        _write_scalar_dataset(metadata_group, "parse_mode", "log_only", "")
        metadata_group.create_dataset("header_sections", data=np.asarray(header.header_sections, dtype=STRING_DTYPE))
        _set_unit_attrs(metadata_group["header_sections"], "")
        if coordinate_model:
            coordinate_group = metadata_group.create_group("coordinate_model")
            _write_nested_mapping(
                coordinate_group,
                coordinate_model,
                unit_resolver=lambda path: "",
                compression=None,
            )
        input_group = metadata_group.create_group("input_parameters")
        _write_nested_mapping(
            input_group,
            header.input_parameters,
            unit_resolver=_input_parameter_unit,
            compression=compression,
        )

        time_group = handle.create_group("time")
        iterator = document.iter_snapshots_streaming(header=header)
        first_snapshot = next(iterator, None)
        if first_snapshot is None:
            raise ValueError(f"No HELIOS snapshots found in {source}.")

        snapshot_capacity = max(1, int(document.index.snapshot_count))
        emit_progress("prepare", 0, snapshot_capacity, 0.12, f"Preparing up to {snapshot_capacity} snapshots")
        time_dataset = _create_1d(time_group, "time", snapshot_capacity, np.float64, "s", chunks=_time_chunk_shape(snapshot_capacity), maxshape=(None,))
        cycle_dataset = _create_1d(time_group, "cycle", snapshot_capacity, np.int64, "", chunks=_time_chunk_shape(snapshot_capacity), maxshape=(None,))
        timestep_dataset = _create_1d(
            time_group,
            "time_step",
            snapshot_capacity,
            np.float64,
            "s",
            chunks=_time_chunk_shape(snapshot_capacity),
            maxshape=(None,),
        )
        timestep_control_dataset = time_group.create_dataset(
            "time_step_control",
            shape=(snapshot_capacity,),
            maxshape=(None,),
            dtype=STRING_DTYPE,
            chunks=_time_chunk_shape(snapshot_capacity),
        )
        _set_unit_attrs(timestep_control_dataset, "")

        fields_group = handle.create_group("fields")
        field_metadata_group = handle.create_group("field_metadata")
        diagnostics_group = handle.create_group("diagnostics")
        available_fields: list[str] = []
        batch_size = min(max(1, snapshot_capacity), 32)
        dynamic_coordinate_center_dataset = (
            _create_dynamic_dataset(grid_group, "dynamic_coordinate_center", snapshot_capacity, header.n_zones, "cm", compression)
            if first_snapshot.coordinate_center is not None
            else None
        )
        dynamic_coordinate_edge_dataset = (
            _create_dynamic_dataset(grid_group, "dynamic_coordinate_edge", snapshot_capacity, header.n_zones + 1, "cm", compression)
            if first_snapshot.coordinate_edge is not None
            else None
        )
        if dynamic_coordinate_center_dataset is not None:
            _annotate_coordinate_dataset(
                dynamic_coordinate_center_dataset,
                coordinate_name=coordinate_name,
                location="center",
                dynamic=True,
            )
        if dynamic_coordinate_edge_dataset is not None:
            _annotate_coordinate_dataset(
                dynamic_coordinate_edge_dataset,
                coordinate_name=coordinate_name,
                location="edge",
                dynamic=True,
            )
        dynamic_coordinate_center_buffer = (
            np.full((batch_size, header.n_zones), np.nan, dtype=np.float64) if dynamic_coordinate_center_dataset is not None else None
        )
        dynamic_coordinate_edge_buffer = (
            np.full((batch_size, header.n_zones + 1), np.nan, dtype=np.float64) if dynamic_coordinate_edge_dataset is not None else None
        )
        field_datasets: dict[str, h5py.Dataset] = {}
        field_batch_buffers: dict[str, np.ndarray] = {}
        diagnostic_datasets: dict[tuple[str, ...], h5py.Dataset] = {}
        diagnostic_batch_buffers: dict[tuple[str, ...], np.ndarray] = {}
        diagnostic_groups: dict[tuple[str, ...], h5py.Group] = {}
        diagnostic_schema: dict[tuple[str, ...], int | None] = {}
        diagnostic_schema_notes: list[str] = []
        diagnostic_units = dict(first_snapshot.diagnostics.get("units", {})) if isinstance(first_snapshot.diagnostics.get("units", {}), dict) else {}
        time_batch = np.empty(batch_size, dtype=np.float64)
        cycle_batch = np.empty(batch_size, dtype=np.int64)
        timestep_batch = np.empty(batch_size, dtype=np.float64)
        timestep_control_batch = np.empty(batch_size, dtype=object)
        active_field_names: set[str] = set()
        active_diagnostic_keys: set[tuple[str, ...]] = set()

        def get_diagnostic_group(path: tuple[str, ...]) -> h5py.Group:
            group = diagnostic_groups.get(path)
            if group is None:
                group = _ensure_group(diagnostics_group, path)
                diagnostic_groups[path] = group
            return group

        def ensure_field_dataset(name: str, unit: str, *, label: str | None = None) -> h5py.Dataset:
            dataset = field_datasets.get(name)
            if dataset is None:
                metadata = {
                    "field_name": name,
                    "source": "log",
                    "dimensions": ("time", "zone"),
                    "unit": unit,
                    "label": label or name,
                    "status": "validated",
                }
                if name == "plasma_pressure_j_cm3":
                    metadata.update(quantity="plasma_pressure", label="Plasma pressure", description="Ion pressure plus electron pressure; excludes radiation pressure.")
                elif name == "total_pressure_j_cm3":
                    metadata.update(quantity="total_pressure", label="Total pressure", description="Ion plus electron plus radiation pressure.")
                elif name == "pressure":
                    metadata.update(quantity="total_pressure", label="Total pressure", status="alias", alias_of="total_pressure_j_cm3", description="Compatibility alias of total_pressure_j_cm3.")
                dataset = _create_field_dataset(
                    fields_group,
                    name,
                    snapshot_capacity,
                    header.n_zones,
                    unit,
                    compression,
                    metadata=metadata,
                    field_metadata_group=field_metadata_group,
                )
                if name == "radius":
                    _annotate_coordinate_dataset(
                        dataset,
                        coordinate_name=coordinate_name,
                        location="center",
                        dynamic=True,
                        legacy_alias=True,
                    )
                elif name == "zone_width":
                    _annotate_coordinate_dataset(
                        dataset,
                        coordinate_name=coordinate_name,
                        location="cell",
                        dynamic=True,
                        role="width",
                    )
                field_datasets[name] = dataset
                field_batch_buffers[name] = np.full((batch_size, header.n_zones), np.nan, dtype=np.float64)
                available_fields.append(name)
            return dataset

        def ensure_diagnostic_dataset(path_key: tuple[str, ...], value) -> h5py.Dataset:
            dataset = diagnostic_datasets.get(path_key)
            if dataset is None:
                width = diagnostic_schema.get(path_key)
                if path_key not in diagnostic_schema:
                    width = diagnostic_value_width(value)
                    diagnostic_schema[path_key] = width
                dataset = _create_dynamic_dataset(
                    get_diagnostic_group(path_key[:-1]),
                    path_key[-1],
                    snapshot_capacity,
                    width,
                    _diagnostic_unit(path_key, diagnostic_units),
                    compression,
                )
                diagnostic_datasets[path_key] = dataset
                diagnostic_batch_buffers[path_key] = (
                    np.full((batch_size, width), np.nan, dtype=np.float64)
                    if width is not None
                    else np.full(batch_size, np.nan, dtype=np.float64)
                )
            else:
                current_width = diagnostic_schema.get(path_key)
                resolved_width, widened, reason = reconcile_diagnostic_width(current_width, value)
                if widened and current_width is None and reason == "scalar_to_vector":
                    raise ValueError(
                        f"Diagnostic {'/'.join(path_key)} changed from scalar to vector width {int(resolved_width)} after streaming schema lock."
                    )
                if widened and current_width is not None and resolved_width is not None and int(resolved_width) > int(current_width):
                    dataset.resize((dataset.shape[0], int(resolved_width)))
                    upgraded = np.full((batch_size, int(resolved_width)), np.nan, dtype=np.float64)
                    upgraded[:, : int(current_width)] = np.asarray(diagnostic_batch_buffers[path_key], dtype=np.float64)
                    diagnostic_batch_buffers[path_key] = upgraded
                    diagnostic_schema[path_key] = int(resolved_width)
                    diagnostic_schema_notes.append(
                        f"Diagnostic {'/'.join(path_key)} widened from width {int(current_width)} to {int(resolved_width)} during HDF5 write."
                    )
            return dataset

        for name, unit in first_snapshot.field_units.items():
            ensure_field_dataset(name, unit, label=first_snapshot.raw_field_map.get(name, name))

        def flush_batch(batch_start: int, batch_count: int) -> None:
            if batch_count == 0:
                return
            batch_stop = batch_start + batch_count
            time_dataset[batch_start:batch_stop] = time_batch[:batch_count]
            cycle_dataset[batch_start:batch_stop] = cycle_batch[:batch_count]
            timestep_dataset[batch_start:batch_stop] = timestep_batch[:batch_count]
            timestep_control_dataset[batch_start:batch_stop] = timestep_control_batch[:batch_count]
            if dynamic_coordinate_center_dataset is not None and dynamic_coordinate_center_buffer is not None:
                dynamic_coordinate_center_dataset[batch_start:batch_stop, :] = dynamic_coordinate_center_buffer[:batch_count, :]
                dynamic_coordinate_center_buffer[:batch_count, :] = np.nan
            if dynamic_coordinate_edge_dataset is not None and dynamic_coordinate_edge_buffer is not None:
                dynamic_coordinate_edge_dataset[batch_start:batch_stop, :] = dynamic_coordinate_edge_buffer[:batch_count, :]
                dynamic_coordinate_edge_buffer[:batch_count, :] = np.nan
            for name in active_field_names:
                field_datasets[name][batch_start:batch_stop, :] = field_batch_buffers[name][:batch_count, :]
                field_batch_buffers[name][:batch_count, :] = np.nan
            for path_key in active_diagnostic_keys:
                dataset = diagnostic_datasets[path_key]
                buffer = diagnostic_batch_buffers[path_key]
                if dataset.ndim == 1:
                    dataset[batch_start:batch_stop] = buffer[:batch_count]
                    buffer[:batch_count] = np.nan
                else:
                    dataset[batch_start:batch_stop, :] = buffer[:batch_count, :]
                    buffer[:batch_count, :] = np.nan
            active_field_names.clear()
            active_diagnostic_keys.clear()

        def buffer_snapshot(local_index: int, snapshot) -> None:
            nonlocal diagnostic_units
            time_batch[local_index] = snapshot.time
            cycle_batch[local_index] = snapshot.cycle
            timestep_batch[local_index] = snapshot.time_step
            timestep_control_batch[local_index] = snapshot.time_step_control
            if dynamic_coordinate_center_buffer is not None and snapshot.coordinate_center is not None:
                dynamic_coordinate_center_buffer[local_index, :] = np.asarray(snapshot.coordinate_center, dtype=np.float64)
            if dynamic_coordinate_edge_buffer is not None and snapshot.coordinate_edge is not None:
                dynamic_coordinate_edge_buffer[local_index, :] = np.asarray(snapshot.coordinate_edge, dtype=np.float64)

            for name, values in snapshot.fields.items():
                unit = snapshot.field_units.get(name, "")
                ensure_field_dataset(name, unit, label=snapshot.raw_field_map.get(name, name))
                field_batch_buffers[name][local_index, :] = values
                active_field_names.add(name)

            snapshot_units = snapshot.diagnostics.get("units", {})
            if isinstance(snapshot_units, dict):
                diagnostic_units.update(snapshot_units)
            if "radiation_boundary_fluxes" in snapshot.diagnostics:
                rad_group = diagnostics_group.require_group("radiation_boundary_fluxes")
                if "region_index" not in rad_group and "region_index" in header.regions:
                    _write_array_dataset(rad_group, "region_index", np.asarray(header.regions["region_index"]), "", compression=None)
            for path_key, value in flatten_snapshot_diagnostics(snapshot.diagnostics):
                dataset = ensure_diagnostic_dataset(path_key, value)
                buffer = diagnostic_batch_buffers[path_key]
                width = diagnostic_schema[path_key]
                normalized = normalize_diagnostic_value(value, width)
                if width is not None:
                    buffer[local_index, :] = normalized
                else:
                    buffer[local_index] = float(normalized)
                active_diagnostic_keys.add(path_key)

        buffer_snapshot(0, first_snapshot)
        flush_batch(0, 1)
        emit_progress(
            "snapshots",
            1,
            snapshot_capacity,
            0.12 + 0.82 * (1 / max(1, snapshot_capacity)),
            f"Writing snapshots (1/{snapshot_capacity})",
        )
        step = 1
        batch_start = 1
        batch_count = 0
        for snapshot in iterator:
            local_index = batch_count
            buffer_snapshot(local_index, snapshot)
            batch_count += 1
            step += 1
            if batch_count == batch_size:
                flush_batch(batch_start, batch_count)
                batch_start += batch_count
                batch_count = 0
            emit_progress(
                "snapshots",
                step,
                snapshot_capacity,
                0.12 + 0.82 * (step / max(1, snapshot_capacity)),
                f"Writing snapshots ({step}/{snapshot_capacity})",
            )
        flush_batch(batch_start, batch_count)

        time_dataset.resize((step,))
        cycle_dataset.resize((step,))
        timestep_dataset.resize((step,))
        timestep_control_dataset.resize((step,))
        if dynamic_coordinate_center_dataset is not None:
            dynamic_coordinate_center_dataset.resize((step, dynamic_coordinate_center_dataset.shape[1]))
        if dynamic_coordinate_edge_dataset is not None:
            dynamic_coordinate_edge_dataset.resize((step, dynamic_coordinate_edge_dataset.shape[1]))
        for dataset in field_datasets.values():
            dataset.resize((step, dataset.shape[1]))
        for dataset in diagnostic_datasets.values():
            if dataset.ndim == 1:
                dataset.resize((step,))
            else:
                dataset.resize((step, dataset.shape[1]))

        if "pressure" in fields_group and "total_pressure_j_cm3" in fields_group:
            del fields_group["pressure"]
            fields_group["pressure"] = fields_group["total_pressure_j_cm3"]
            if "pressure" in field_metadata_group:
                del field_metadata_group["pressure"]
            _write_field_metadata_group(
                field_metadata_group,
                "pressure",
                {
                    "field_name": "pressure",
                    "source": "log",
                    "dimensions": ("time", "zone"),
                    "unit": "J/cm3",
                    "label": "Total pressure",
                    "status": "alias",
                    "quantity": "total_pressure",
                    "alias_of": "total_pressure_j_cm3",
                    "description": "Compatibility alias of total_pressure_j_cm3.",
                },
            )

        stream_status = iterator.run_status or header.run_status
        if stream_status is None:
            raise RuntimeError("Streaming conversion finished without a finalized run status.")
        header.metadata["run_status"] = dict(header.metadata.get("run_status", {}))

        _write_scalar_dataset(metadata_group, "n_snapshots", int(step), "")
        run_status_group = _replace_group(metadata_group, "run_status")
        _write_nested_mapping(
            run_status_group,
            dict(header.metadata.get("run_status", {})),
            unit_resolver=lambda path: "s" if path[-1] in {"intended_end_time_s", "last_valid_snapshot_time_s"} else "",
            compression=None,
        )

        metadata_group.create_dataset("available_fields", data=np.asarray(available_fields, dtype=STRING_DTYPE))
        _set_unit_attrs(metadata_group["available_fields"], "")
        if diagnostic_schema_notes:
            metadata_group.create_dataset("diagnostic_schema_notes", data=np.asarray(diagnostic_schema_notes, dtype=STRING_DTYPE))
            _set_unit_attrs(metadata_group["diagnostic_schema_notes"], "")
        emit_progress("finalize", step, snapshot_capacity, 0.98, f"Finalizing {target.name}")
        active_logger.info("Wrote %s snapshots to %s (%s)", step, target, stream_status.state)
    elapsed_write = time.perf_counter() - started_write
    record_duration("hdf5.write", elapsed_write)
    if active_logger.isEnabledFor(logging.DEBUG):
        active_logger.debug("hdf5.write took %.3f ms (%s)", elapsed_write * 1.0e3, source.name)

    return target


def _selected_input_paths(selection: ParseSourceSelection) -> tuple[Path, ...]:
    paths = [selection.requested_path, selection.primary_path]
    if selection.bpf_path is not None:
        paths.append(selection.bpf_path)
    if selection.log_path is not None:
        paths.append(selection.log_path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return tuple(unique)


def _paths_alias(left: Path, right: Path) -> bool:
    if left.resolve(strict=False) == right.resolve(strict=False):
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _create_sibling_temp(target: Path) -> tuple[Path, int]:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    for _ in range(100):
        candidate = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
        try:
            fd = os.open(candidate, flags, 0o666)
            return candidate, fd
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate a temporary output beside {target}.")


def _validate_completed_hdf5(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        required = ("grid", "time/time", "fields", "metadata/n_snapshots", "metadata/run_status/state")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"Generated HDF5 is missing required entries: {', '.join(missing)}.")
        count = int(handle["metadata/n_snapshots"][()])
        if count <= 0 or int(handle["time/time"].shape[0]) != count:
            raise ValueError("Generated HDF5 has an invalid snapshot count.")
        state = handle["metadata/run_status/state"][()]
        if isinstance(state, bytes):
            state = state.decode("utf-8", errors="replace")
        if not str(state):
            raise ValueError("Generated HDF5 has no run status.")


def write_hdf5(
    input_path: str | Path,
    output_path: str | Path,
    *,
    compression: str | None = None,
    overwrite: bool = False,
    parser: HeliosParser | None = None,
    logger: logging.Logger | None = None,
    progress_callback: Callable[[WriteProgress], None] | None = None,
) -> Path:
    """Convert HELIOS input and atomically publish a validated HDF5 file.

    Existing symlink outputs are replaced as directory entries; their referents
    are never overwritten. Output paths may not alias the selected input files.
    """
    active_logger = logger or LOGGER
    active_parser = parser or HeliosParser(active_logger)
    source = Path(input_path)
    target = Path(output_path)
    selection = resolve_parse_sources(source)
    inputs = _selected_input_paths(selection)
    if any(_paths_alias(target, item) for item in inputs):
        raise ValueError(f"Output path {target} aliases a selected input file.")

    exists = os.path.lexists(target)
    if exists and not overwrite:
        raise FileExistsError(f"{target} already exists. Use --overwrite to replace it.")
    if target.parent and not target.parent.exists():
        raise FileNotFoundError(target.parent)

    temp_path, fd = _create_sibling_temp(target)
    os.close(fd)
    published = False
    try:
        def report(event: WriteProgress) -> None:
            if progress_callback is None:
                return
            message = event.message.replace(temp_path.name, target.name)
            progress_callback(WriteProgress(event.stage, event.current, event.total, event.fraction, message))

        _write_hdf5_unpublished(
            source,
            temp_path,
            compression=compression,
            overwrite=False,
            parser=active_parser,
            logger=active_logger,
            progress_callback=report if progress_callback is not None else None,
        )
        _validate_completed_hdf5(temp_path)
        if exists and not target.is_symlink():
            os.chmod(temp_path, stat.S_IMODE(target.stat().st_mode))
        if exists or overwrite:
            os.replace(temp_path, target)
        else:
            # link() is an atomic no-clobber publish on the same filesystem.
            os.link(temp_path, target)
            temp_path.unlink()
        published = True
        if progress_callback is not None:
            try:
                progress_callback(WriteProgress("done", 1, 1, 1.0, f"Wrote {target.name}"))
            except Exception:
                active_logger.exception("HDF5 output was published, but the completion callback failed.")
        return target
    finally:
        if not published:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                active_logger.warning("Could not remove temporary HDF5 output %s", temp_path, exc_info=True)
