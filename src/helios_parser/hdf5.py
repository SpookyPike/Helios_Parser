from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from helios.instrumentation import record_duration
from .document import diagnostic_value_width, flatten_snapshot_diagnostics, normalize_diagnostic_value, reconcile_diagnostic_width
from .parser import HeliosParser

LOGGER = logging.getLogger(__name__)
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class WriteProgress:
    """Progress payload emitted during HDF5 conversion."""

    stage: str
    current: int
    total: int
    fraction: float
    message: str


def _set_unit_attrs(dataset: h5py.Dataset, unit: str) -> None:
    dataset.attrs["unit"] = unit
    dataset.attrs["units"] = unit


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
    _set_unit_attrs(dataset, unit)
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
    active_logger = logger or LOGGER
    active_parser = parser or HeliosParser(active_logger)
    source = Path(input_path)
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists. Use --overwrite to replace it.")

    if target.exists():
        target.unlink()

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

        def ensure_field_dataset(name: str, unit: str) -> h5py.Dataset:
            dataset = field_datasets.get(name)
            if dataset is None:
                dataset = _create_field_dataset(fields_group, name, snapshot_capacity, header.n_zones, unit, compression)
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
            ensure_field_dataset(name, unit)

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
                ensure_field_dataset(name, unit)
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
    emit_progress("done", 1, 1, 1.0, f"Wrote {target.name}")

    return target
