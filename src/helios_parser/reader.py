"""Lazy HDF5 reader for stabilized HELIOS datasets.

``HeliosRun`` is the supported lightweight consumption layer for downstream
analysis scripts, the viewer, and the unified shell. It hides the raw HDF5 tree
behind field/grid/diagnostic accessors while preserving lazy slicing semantics.
Coordinate getters default to zone centers for backward safety, but the reader
also exposes explicit edge-coordinate access plus run-status and VISAR-readiness
metadata surfaces needed by the current application.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from helios.instrumentation import timed_block
from .coordinates import (
    build_coordinate_edge_array,
    build_coordinate_edge_grid,
    centers_from_edge_grid_and_widths,
    centers_from_edges_and_widths,
    coordinate_name_for_geometry,
)

LOGGER = logging.getLogger(__name__)


REQUIRED_GROUPS = ("grid", "time", "fields", "regions", "materials", "diagnostics", "metadata")


@dataclass(frozen=True, slots=True)
class VisarBoundaryCandidate:
    label: str
    kind: str
    boundary_zone: int
    left_region: int | None = None
    right_region: int | None = None


@dataclass(frozen=True, slots=True)
class VisarSupportMetadata:
    velocity_field_name: str | None
    time_axis_name: str
    static_coordinate_name: str | None
    dynamic_coordinate_field_name: str | None
    boundary_indexing_consistent: bool
    candidate_boundaries: tuple[VisarBoundaryCandidate, ...]
    event_timing_source: str | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VisarReadinessStatus:
    ready: bool
    reasons: tuple[str, ...]
    support: VisarSupportMetadata


@dataclass(frozen=True, slots=True)
class FieldMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    source: str
    dimensions: tuple[str, ...]
    unit: str
    label: str
    status: str
    description: str = ""
    plotting_hints: dict[str, Any] | None = None
    alias_of: str | None = None


def _normalize_loaded_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_loaded_value(value[()])
        if value.dtype.kind in {"S", "O"}:
            return np.asarray([_normalize_loaded_value(item) for item in value], dtype=object)
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_dataset(dataset: h5py.Dataset) -> Any:
    if h5py.check_string_dtype(dataset.dtype) is not None:
        value = dataset.asstr()[()]
    else:
        value = dataset[()]
    return _normalize_loaded_value(value)


def _read_group(group: h5py.Group) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            values[name] = _read_group(item)
        else:
            values[name] = _read_dataset(item)
    return values


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_attr(value: Any, default: Any) -> Any:
    value = _decode_attr(value)
    if value is None or value == "":
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


class HeliosRun:
    """Open a stabilized HELIOS HDF5 file with lazy dataset access."""

    def __init__(self, path: str | Path, *, mode: str = "r") -> None:
        self.path = Path(path)
        with timed_block("hdf5.read.open_run", logger=LOGGER, details=self.path.name):
            self._handle = h5py.File(self.path, mode)
            self._groups = {name: self._handle[name] for name in REQUIRED_GROUPS}
            self._grid_datasets = {name: dataset for name, dataset in self._groups["grid"].items()}
            self._time_datasets = {name: dataset for name, dataset in self._groups["time"].items()}
            self._field_datasets = {name: dataset for name, dataset in self._groups["fields"].items()}
            self._diagnostic_datasets = self._index_diagnostics(self._groups["diagnostics"])
            self._field_metadata_group = self._handle.get("field_metadata")
        self._metadata_cache: dict[str, Any] | None = None
        self._regions_cache: dict[str, Any] | None = None
        self._materials_cache: dict[str, Any] | None = None
        self._coordinate_model_cache: dict[str, Any] | None = None
        self._static_coordinate_center_cache: np.ndarray | None = None
        self._static_coordinate_edge_cache: np.ndarray | None = None
        self._dynamic_coordinate_center_cache: np.ndarray | None = None
        self._dynamic_coordinate_edge_cache: np.ndarray | None = None
        self._zone_region_ids: np.ndarray | None = None
        self._zone_material_ids: np.ndarray | None = None
        self._field_names = self._load_field_names()
        self._field_metadata_cache: dict[str, FieldMetadata] | None = None

    def _index_diagnostics(self, group: h5py.Group) -> dict[str, h5py.Dataset]:
        datasets: dict[str, h5py.Dataset] = {}

        def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if isinstance(item, h5py.Dataset):
                datasets[name] = item

        group.visititems(visitor)
        return datasets

    def _load_field_names(self) -> tuple[str, ...]:
        metadata_group = self._groups["metadata"]
        if "available_fields" in metadata_group:
            values = _read_dataset(metadata_group["available_fields"])
            if isinstance(values, np.ndarray):
                return tuple(str(value) for value in values.tolist() if str(value) in self._field_datasets)
        return tuple(self._field_datasets)

    def _metadata_from_attrs(self, name: str, dataset: h5py.Dataset) -> FieldMetadata:
        attr_source: dict[str, Any] = {}
        attr_source.update({key: _decode_attr(value) for key, value in dataset.attrs.items()})
        field_metadata_group = getattr(self, "_field_metadata_group", None)
        if isinstance(field_metadata_group, h5py.Group) and name in field_metadata_group:
            attr_source.update({key: _decode_attr(value) for key, value in field_metadata_group[name].attrs.items()})
        dimensions = _json_attr(attr_source.get("dimensions"), None)
        dataset_shape = tuple(int(value) for value in dataset.shape)
        dataset_ndim = len(dataset_shape)
        if dimensions is None:
            if dataset_ndim == 2 and dataset_shape[0] == self.n_snapshots and dataset_shape[1] == self.n_zones:
                dimensions = ("time", "zone")
            elif dataset_ndim >= 1 and dataset_shape[0] == self.n_snapshots:
                dimensions = ("time",) + tuple(f"axis_{index}" for index in range(1, dataset_ndim))
            else:
                dimensions = tuple(f"axis_{index}" for index in range(dataset_ndim))
        plotting_hints = _json_attr(attr_source.get("plotting_hints"), None)
        dtype = getattr(dataset, "dtype", None)
        if dtype is None and hasattr(dataset, "values"):
            dtype = np.asarray(dataset.values).dtype
        return FieldMetadata(
            name=str(attr_source.get("field_name", name) or name),
            shape=dataset_shape,
            dtype=str(dtype or ""),
            source=str(attr_source.get("source", "unknown") or "unknown"),
            dimensions=tuple(str(value) for value in dimensions),
            unit=str(attr_source.get("units", attr_source.get("unit", "")) or ""),
            label=str(attr_source.get("label", name) or name),
            status=str(attr_source.get("status", "legacy") or "legacy"),
            description=str(attr_source.get("description", "") or ""),
            plotting_hints=plotting_hints if isinstance(plotting_hints, dict) else None,
            alias_of=str(attr_source["alias_of"]) if attr_source.get("alias_of") else None,
        )

    def _load_field_metadata(self) -> dict[str, FieldMetadata]:
        if getattr(self, "_field_metadata_cache", None) is None:
            self._field_metadata_cache = {
                name: self._metadata_from_attrs(name, dataset)
                for name, dataset in self._field_datasets.items()
            }
        return self._field_metadata_cache

    def _coordinate_model(self) -> dict[str, Any]:
        if self._coordinate_model_cache is not None:
            return self._coordinate_model_cache
        metadata_group = self._groups["metadata"]
        if "coordinate_model" in metadata_group and isinstance(metadata_group["coordinate_model"], h5py.Group):
            raw_model = _read_group(metadata_group["coordinate_model"])
            if isinstance(raw_model, dict):
                self._coordinate_model_cache = raw_model
                return raw_model
        inferred = {
            "coordinate_name": coordinate_name_for_geometry(self.get_metadata().get("geometry")),
            "static_center_dataset": "coordinate_center",
            "static_edge_dataset": "coordinate_edge",
            "dynamic_center_dataset": "dynamic_coordinate_center",
            "dynamic_edge_dataset": "dynamic_coordinate_edge",
            "width_dataset": "zone_width",
            "legacy_static_center_alias": "x",
            "legacy_dynamic_center_alias": "radius",
        }
        self._coordinate_model_cache = inferred
        return inferred

    def _coordinate_name(self) -> str:
        return str(self._coordinate_model().get("coordinate_name", coordinate_name_for_geometry(self.get_metadata().get("geometry"))))

    def _static_widths(self) -> np.ndarray:
        if "zone_width" in self._grid_datasets:
            return np.asarray(self._grid_datasets["zone_width"][()], dtype=np.float64)
        raise KeyError("Static zone-width grid is missing.")

    @staticmethod
    def _edges_from_centers(values: np.ndarray) -> np.ndarray:
        centers = np.asarray(values, dtype=np.float64)
        if centers.size == 0:
            return np.array([], dtype=np.float64)
        if centers.size == 1:
            delta = max(abs(float(centers[0])) * 0.5, 0.5)
            return np.asarray([float(centers[0]) - delta, float(centers[0]) + delta], dtype=np.float64)
        edges = np.empty(centers.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - (edges[1] - centers[0])
        edges[-1] = centers[-1] + (centers[-1] - edges[-2])
        return edges

    def _dynamic_widths(self) -> np.ndarray:
        if "zone_width" in self._field_datasets:
            values = np.asarray(self._field_datasets["zone_width"][()], dtype=np.float64)
            if values.shape == (self.n_snapshots, self.n_zones):
                return values
        widths = self._static_widths()
        return np.broadcast_to(widths[None, :], (self.n_snapshots, self.n_zones))

    def _dynamic_widths_snapshot(self, snapshot_index: int) -> np.ndarray:
        if "zone_width" in self._field_datasets:
            dataset = self._field_datasets["zone_width"]
            if tuple(dataset.shape) == (self.n_snapshots, self.n_zones):
                return np.asarray(dataset[self._normalize_snapshot_index(snapshot_index), :], dtype=np.float64)
        return self._static_widths()

    @staticmethod
    def _slice_array(values: np.ndarray, selection: slice | int | None) -> np.ndarray:
        index = slice(None) if selection is None else selection
        return np.asarray(values[index], dtype=np.float64)

    def has_dynamic_coordinate(self) -> bool:
        model = self._coordinate_model()
        return (
            str(model.get("dynamic_center_dataset", "")) in self._grid_datasets
            or str(model.get("dynamic_edge_dataset", "")) in self._grid_datasets
            or str(model.get("legacy_dynamic_center_alias", "radius")) in self._field_datasets
        )

    def get_static_coordinate(self, *, location: str = "center") -> np.ndarray:
        normalized = str(location).strip().lower()
        model = self._coordinate_model()
        if normalized == "center":
            if self._static_coordinate_center_cache is None:
                dataset_name = str(model.get("static_center_dataset", "coordinate_center"))
                if dataset_name in self._grid_datasets:
                    self._static_coordinate_center_cache = np.asarray(self._grid_datasets[dataset_name][()], dtype=np.float64)
                elif "x" in self._grid_datasets:
                    raw = np.asarray(self._grid_datasets["x"][()], dtype=np.float64)
                    attr_location = str(self._grid_datasets["x"].attrs.get("coordinate_location", "")).strip().lower()
                    if attr_location == "center":
                        self._static_coordinate_center_cache = raw
                    elif raw.size == self.n_zones + 1:
                        try:
                            widths = self._static_widths()
                        except KeyError:
                            self._static_coordinate_center_cache = 0.5 * (raw[:-1] + raw[1:])
                        else:
                            self._static_coordinate_center_cache = centers_from_edges_and_widths(raw, widths)
                    else:
                        try:
                            widths = self._static_widths()
                        except KeyError:
                            self._static_coordinate_center_cache = raw
                        else:
                            edge_values = build_coordinate_edge_array(raw, widths, geometry=self.get_metadata().get("geometry"))
                            self._static_coordinate_center_cache = centers_from_edges_and_widths(edge_values, widths)
                else:
                    raise KeyError("No static coordinate dataset is available.")
            return np.asarray(self._static_coordinate_center_cache, dtype=np.float64)
        if normalized == "edge":
            if self._static_coordinate_edge_cache is None:
                dataset_name = str(model.get("static_edge_dataset", "coordinate_edge"))
                if dataset_name in self._grid_datasets:
                    self._static_coordinate_edge_cache = np.asarray(self._grid_datasets[dataset_name][()], dtype=np.float64)
                elif "x" in self._grid_datasets:
                    raw = np.asarray(self._grid_datasets["x"][()], dtype=np.float64)
                    attr_location = str(self._grid_datasets["x"].attrs.get("coordinate_location", "")).strip().lower()
                    if attr_location == "center":
                        try:
                            widths = self._static_widths()
                        except KeyError:
                            self._static_coordinate_edge_cache = self._edges_from_centers(raw)
                        else:
                            outer_edges = raw + 0.5 * widths
                            self._static_coordinate_edge_cache = build_coordinate_edge_array(
                                outer_edges,
                                widths,
                                geometry=self.get_metadata().get("geometry"),
                            )
                    elif raw.size == self.n_zones + 1:
                        self._static_coordinate_edge_cache = raw
                    else:
                        try:
                            widths = self._static_widths()
                        except KeyError:
                            self._static_coordinate_edge_cache = self._edges_from_centers(raw)
                        else:
                            self._static_coordinate_edge_cache = build_coordinate_edge_array(
                                raw,
                                widths,
                                geometry=self.get_metadata().get("geometry"),
                            )
                else:
                    raise KeyError("No static coordinate dataset is available.")
            return np.asarray(self._static_coordinate_edge_cache, dtype=np.float64)
        raise ValueError(f"Unknown coordinate location {location!r}; expected 'center' or 'edge'.")

    def get_dynamic_coordinate(
        self,
        snapshot_index: int | None = None,
        *,
        location: str = "center",
    ) -> np.ndarray | None:
        if not self.has_dynamic_coordinate():
            return None
        normalized = str(location).strip().lower()
        if snapshot_index is not None:
            normalized_snapshot = self._normalize_snapshot_index(snapshot_index)
            model = self._coordinate_model()
            if normalized == "center":
                dataset_name = str(model.get("dynamic_center_dataset", "dynamic_coordinate_center"))
                if dataset_name in self._grid_datasets:
                    return np.asarray(self._grid_datasets[dataset_name][normalized_snapshot, :], dtype=np.float64)
                if "radius" in self._field_datasets:
                    radius_dataset = self._field_datasets["radius"]
                    raw = np.asarray(radius_dataset[normalized_snapshot, :], dtype=np.float64)
                    attr_location = str(radius_dataset.attrs.get("coordinate_location", "")).strip().lower()
                    if attr_location == "center":
                        return raw
                    widths = self._dynamic_widths_snapshot(normalized_snapshot)
                    if raw.ndim == 1 and raw.shape[0] == self.n_zones + 1:
                        return centers_from_edges_and_widths(raw, widths)
                    edge_values = build_coordinate_edge_array(raw, widths, geometry=self.get_metadata().get("geometry"))
                    return centers_from_edges_and_widths(edge_values, widths)
                return None
            if normalized == "edge":
                dataset_name = str(model.get("dynamic_edge_dataset", "dynamic_coordinate_edge"))
                if dataset_name in self._grid_datasets:
                    return np.asarray(self._grid_datasets[dataset_name][normalized_snapshot, :], dtype=np.float64)
                if "radius" in self._field_datasets:
                    radius_dataset = self._field_datasets["radius"]
                    raw = np.asarray(radius_dataset[normalized_snapshot, :], dtype=np.float64)
                    attr_location = str(radius_dataset.attrs.get("coordinate_location", "")).strip().lower()
                    widths = self._dynamic_widths_snapshot(normalized_snapshot)
                    if attr_location == "center":
                        outer_edges = raw + 0.5 * widths
                        return build_coordinate_edge_array(
                            outer_edges,
                            widths,
                            geometry=self.get_metadata().get("geometry"),
                        )
                    if raw.ndim == 1 and raw.shape[0] == self.n_zones + 1:
                        return raw
                    return build_coordinate_edge_array(
                        raw,
                        widths,
                        geometry=self.get_metadata().get("geometry"),
                    )
                return None
            raise ValueError(f"Unknown coordinate location {location!r}; expected 'center' or 'edge'.")
        model = self._coordinate_model()
        if normalized == "center":
            if self._dynamic_coordinate_center_cache is None:
                dataset_name = str(model.get("dynamic_center_dataset", "dynamic_coordinate_center"))
                if dataset_name in self._grid_datasets:
                    self._dynamic_coordinate_center_cache = np.asarray(self._grid_datasets[dataset_name][()], dtype=np.float64)
                elif "radius" in self._field_datasets:
                    raw = np.asarray(self._field_datasets["radius"][()], dtype=np.float64)
                    attr_location = str(self._field_datasets["radius"].attrs.get("coordinate_location", "")).strip().lower()
                    if attr_location == "center":
                        self._dynamic_coordinate_center_cache = raw
                    elif raw.ndim == 2 and raw.shape[1] == self.n_zones + 1:
                        widths = self._dynamic_widths()
                        self._dynamic_coordinate_center_cache = centers_from_edge_grid_and_widths(raw, widths)
                    else:
                        widths = self._dynamic_widths()
                        edge_values = build_coordinate_edge_grid(raw, widths, geometry=self.get_metadata().get("geometry"))
                        self._dynamic_coordinate_center_cache = centers_from_edge_grid_and_widths(edge_values, widths)
                else:
                    return None
            values = self._dynamic_coordinate_center_cache
        elif normalized == "edge":
            if self._dynamic_coordinate_edge_cache is None:
                dataset_name = str(model.get("dynamic_edge_dataset", "dynamic_coordinate_edge"))
                if dataset_name in self._grid_datasets:
                    self._dynamic_coordinate_edge_cache = np.asarray(self._grid_datasets[dataset_name][()], dtype=np.float64)
                elif "radius" in self._field_datasets:
                    raw = np.asarray(self._field_datasets["radius"][()], dtype=np.float64)
                    widths = self._dynamic_widths()
                    attr_location = str(self._field_datasets["radius"].attrs.get("coordinate_location", "")).strip().lower()
                    if attr_location == "center":
                        outer_edges = raw + 0.5 * widths
                        self._dynamic_coordinate_edge_cache = build_coordinate_edge_grid(
                            outer_edges,
                            widths,
                            geometry=self.get_metadata().get("geometry"),
                        )
                    elif raw.ndim == 2 and raw.shape[1] == self.n_zones + 1:
                        self._dynamic_coordinate_edge_cache = raw
                    else:
                        self._dynamic_coordinate_edge_cache = build_coordinate_edge_grid(
                            raw,
                            widths,
                            geometry=self.get_metadata().get("geometry"),
                        )
                else:
                    return None
            values = self._dynamic_coordinate_edge_cache
        else:
            raise ValueError(f"Unknown coordinate location {location!r}; expected 'center' or 'edge'.")
        return np.asarray(values, dtype=np.float64)

    def get_coordinate(
        self,
        *,
        snapshot_index: int | None = None,
        location: str = "center",
        prefer_dynamic: bool = True,
    ) -> np.ndarray:
        if prefer_dynamic:
            dynamic = self.get_dynamic_coordinate(snapshot_index, location=location)
            if dynamic is not None:
                return dynamic
        return self.get_static_coordinate(location=location)

    def _normalize_snapshot_index(self, snapshot_index: int) -> int:
        normalized = snapshot_index
        if normalized < 0:
            normalized += self.n_snapshots
        if normalized < 0 or normalized >= self.n_snapshots:
            raise IndexError(f"snapshot_index {snapshot_index} is out of range for {self.n_snapshots} snapshots.")
        return normalized

    def _get_field_dataset(self, field_name: str) -> h5py.Dataset:
        try:
            return self._field_datasets[field_name]
        except KeyError as exc:
            raise KeyError(f"Unknown field {field_name!r}. Available fields: {', '.join(self._field_names)}") from exc

    def _get_diagnostic_dataset(self, path: str) -> h5py.Dataset:
        normalized = path.strip("/")
        try:
            return self._diagnostic_datasets[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._diagnostic_datasets))
            raise KeyError(f"Unknown diagnostic {path!r}. Available diagnostics: {available}") from exc

    @property
    def n_zones(self) -> int:
        return int(self._grid_datasets["zone_id"].shape[0])

    @property
    def n_snapshots(self) -> int:
        return int(self._time_datasets["time"].shape[0])

    @property
    def n_regions(self) -> int:
        return int(self._groups["regions"]["region_index"].shape[0])

    @property
    def n_materials(self) -> int:
        return int(self._groups["materials"]["index"].shape[0])

    def summary(self) -> dict[str, Any]:
        """Return a compact run summary suitable for UIs and quick inspection."""
        metadata = self.get_metadata()
        run_status = self.get_run_status()
        return {
            "path": str(self.path),
            "simulation_name": metadata.get("simulation_name", self.path.stem),
            "geometry": metadata.get("geometry"),
            "helios_version": metadata.get("helios_version"),
            "calculation_datetime": metadata.get("calculation_datetime"),
            "n_zones": self.n_zones,
            "n_snapshots": self.n_snapshots,
            "n_regions": self.n_regions,
            "n_materials": self.n_materials,
            "available_fields": list(self._field_names),
            "run_status": run_status,
            "metadata": {
                key: metadata.get(key)
                for key in ("simulation_name", "geometry", "eos_model", "helios_version", "calculation_datetime", "source_file")
                if key in metadata
            },
        }

    def get_run_status(self) -> dict[str, Any]:
        metadata = self.get_metadata()
        raw = metadata.get("run_status")
        if isinstance(raw, dict):
            status = dict(raw)
        else:
            status = {}
        status.setdefault("state", "unknown")
        status.setdefault("source", "legacy_hdf5")
        status.setdefault("footer_message", None)
        status.setdefault("footer_datetime", None)
        status.setdefault("intended_end_time_s", None)
        status.setdefault("last_valid_snapshot_time_s", float(self.get_time()[-1]) if self.n_snapshots else None)
        status.setdefault("indexed_snapshot_count", int(self.n_snapshots))
        status.setdefault("valid_snapshot_count", int(self.n_snapshots))
        status.setdefault("dropped_partial_final_block", False)
        status.setdefault("damaged_final_block_reason", None)
        notes = status.get("notes", ())
        if isinstance(notes, np.ndarray):
            notes = tuple(str(value) for value in notes.tolist())
        elif isinstance(notes, list):
            notes = tuple(str(value) for value in notes)
        elif isinstance(notes, tuple):
            notes = tuple(str(value) for value in notes)
        elif notes in {None, ""}:
            notes = ()
        elif isinstance(notes, str) and notes.startswith("["):
            try:
                parsed = json.loads(notes)
            except json.JSONDecodeError:
                notes = (notes,)
            else:
                if isinstance(parsed, list):
                    notes = tuple(str(value) for value in parsed)
                else:
                    notes = (str(parsed),)
        else:
            notes = (str(notes),)
        status["notes"] = notes
        return status

    def _visar_boundary_candidates(self) -> tuple[tuple[VisarBoundaryCandidate, ...], list[str]]:
        issues: list[str] = []
        try:
            regions = self.get_regions()
        except Exception as exc:
            return (), [f"region/interface metadata is unavailable: {exc}"]
        required = ("region_index", "min_zone_index", "max_zone_index")
        if any(name not in regions for name in required):
            return (), ["region/interface metadata is incomplete."]
        region_ids = np.asarray(regions["region_index"], dtype=np.int32)
        min_zone = np.asarray(regions["min_zone_index"], dtype=np.int32)
        max_zone = np.asarray(regions["max_zone_index"], dtype=np.int32)
        if not (region_ids.size == min_zone.size == max_zone.size):
            return (), ["region/interface indexing arrays do not have matching lengths."]
        candidates: list[VisarBoundaryCandidate] = [
            VisarBoundaryCandidate("Low-index surface", "surface_low", 1),
            VisarBoundaryCandidate("High-index surface", "surface_high", self.n_zones),
        ]
        if region_ids.size == 0:
            issues.append("region/interface metadata contains no regions.")
            return tuple(candidates), issues
        ordered = bool(np.all(np.diff(min_zone) >= 0) and np.all(np.diff(max_zone) >= 0))
        contiguous = bool(int(min_zone[0]) == 1 and int(max_zone[-1]) == self.n_zones)
        for index in range(region_ids.size - 1):
            expected_next = int(max_zone[index]) + 1
            if int(min_zone[index + 1]) != expected_next:
                contiguous = False
            candidates.append(
                VisarBoundaryCandidate(
                    label=f"Interface: region {int(region_ids[index])} -> {int(region_ids[index + 1])}",
                    kind="interface",
                    boundary_zone=int(max_zone[index]),
                    left_region=int(region_ids[index]),
                    right_region=int(region_ids[index + 1]),
                )
            )
        if not ordered:
            issues.append("region/interface indexing is not ordered monotonically.")
        if not contiguous:
            issues.append("region/interface indexing is not contiguous across zones.")
        return tuple(candidates), issues

    def get_visar_support_metadata(self) -> VisarSupportMetadata:
        """Return the compact metadata surface needed for future VISAR work."""

        candidates, issues = self._visar_boundary_candidates()
        velocity_field_name = "velocity" if "velocity" in self._field_datasets else None
        static_coordinate_name = self._coordinate_name() if self.get_static_coordinate(location="center").shape[0] == self.n_zones else None
        dynamic_coordinate_field_name = "radius" if self.has_dynamic_coordinate() else None
        if dynamic_coordinate_field_name is not None:
            dynamic_center = self.get_dynamic_coordinate(location="center")
            if dynamic_center is None or tuple(dynamic_center.shape) != (self.n_snapshots, self.n_zones):
                issues.append("dynamic coordinate field shape does not match snapshot/zone dimensions.")
        return VisarSupportMetadata(
            velocity_field_name=velocity_field_name,
            time_axis_name="time",
            static_coordinate_name=static_coordinate_name,
            dynamic_coordinate_field_name=dynamic_coordinate_field_name,
            boundary_indexing_consistent=not issues,
            candidate_boundaries=candidates,
            event_timing_source="derived.shock_tracking.track_shock_front",
            notes=tuple(issues),
        )

    def check_visar_readiness(self) -> VisarReadinessStatus:
        """Report whether the dataset has the minimum structure for future VISAR use."""

        support = self.get_visar_support_metadata()
        reasons: list[str] = []
        if support.velocity_field_name is None:
            reasons.append("velocity field is missing.")
        else:
            velocity_shape = tuple(self._field_datasets[support.velocity_field_name].shape)
            if velocity_shape != (self.n_snapshots, self.n_zones):
                reasons.append("velocity field shape does not match snapshot/zone dimensions.")
        time_dataset = self._time_datasets.get(support.time_axis_name)
        if time_dataset is None:
            reasons.append("time axis is missing.")
        else:
            time_values = np.asarray(time_dataset[()], dtype=np.float64)
            if time_values.ndim != 1 or time_values.size != self.n_snapshots:
                reasons.append("time axis shape is inconsistent with snapshot count.")
            elif time_values.size and not np.all(np.isfinite(time_values)):
                reasons.append("time axis contains non-finite values.")
        if support.static_coordinate_name is None and support.dynamic_coordinate_field_name is None:
            reasons.append("no coordinate source is available for boundary selection.")
        reasons.extend(str(note) for note in support.notes)
        if len(support.candidate_boundaries) == 0:
            reasons.append("no observable surfaces or interfaces were identified.")
        return VisarReadinessStatus(
            ready=len(reasons) == 0,
            reasons=tuple(reasons),
            support=support,
        )

    def list_fields(self) -> list[str]:
        return list(self._field_names)

    def list_field_metadata(self) -> dict[str, FieldMetadata]:
        return {name: self._load_field_metadata()[name] for name in self._field_names if name in self._load_field_metadata()}

    def get_field_metadata(self, field_name: str) -> FieldMetadata:
        self._get_field_dataset(field_name)
        return self._load_field_metadata()[field_name]

    def has_field(self, field_name: str) -> bool:
        return field_name in self._field_datasets

    def get_field_axes(self, field_name: str) -> tuple[str, ...]:
        return self.get_field_metadata(field_name).dimensions

    def get_field_label(self, field_name: str) -> str:
        return self.get_field_metadata(field_name).label

    def get_field_plotting_hints(self, field_name: str) -> dict[str, Any]:
        return dict(self.get_field_metadata(field_name).plotting_hints or {})

    def plotting_modes_for_field(self, field_name: str) -> tuple[str, ...]:
        axes = self.get_field_axes(field_name)
        if axes == ("time", "zone"):
            return ("time_map", "snapshot_profile", "zone_trace")
        if axes == ("time", "node"):
            return ("node_time_map", "snapshot_node_profile", "node_trace")
        if axes == ("time", "frequency"):
            return ("spectral_evolution", "spectrum", "frequency_trace")
        if axes == ("time", "frequency_edge"):
            return ("axis_values",)
        if axes == ("time", "boundary"):
            return ("boundary_trace",)
        if axes == ("time", "zone", "charge_state"):
            return ("ionization_fraction", "charge_state_profile", "dominant_charge_summary")
        if axes == ("time", "summary_value"):
            return ("summary_vector", "time_series")
        if axes == ("time", "header_value"):
            return ("header_vector", "time_series")
        if axes == ("time", "bpf_record_value"):
            return ("raw_vector", "time_series")
        if axes and axes[0] == "time":
            return ("time_series",)
        return ("array",)

    def list_diagnostics(self) -> list[str]:
        return sorted(self._diagnostic_datasets)

    def list_region_ids(self) -> list[int]:
        return [int(value) for value in self.get_regions()["region_index"]]

    def list_material_ids(self) -> list[int]:
        return [int(value) for value in self.get_materials()["index"]]

    def get_field_unit(self, field_name: str) -> str:
        if field_name == "radius":
            if "radius" in self._field_datasets:
                return self.get_field_metadata("radius").unit
            return self.get_grid_unit("x")
        return self.get_field_metadata(field_name).unit

    def get_grid_unit(self, name: str = "x") -> str:
        if name in {"x", "coordinate_center"}:
            model = self._coordinate_model()
            dataset_name = str(model.get("static_center_dataset", "coordinate_center"))
            if dataset_name in self._grid_datasets:
                return str(self._grid_datasets[dataset_name].attrs.get("units", ""))
            if "x" in self._grid_datasets:
                return str(self._grid_datasets["x"].attrs.get("units", ""))
        if name == "coordinate_edge":
            model = self._coordinate_model()
            dataset_name = str(model.get("static_edge_dataset", "coordinate_edge"))
            if dataset_name in self._grid_datasets:
                return str(self._grid_datasets[dataset_name].attrs.get("units", ""))
        return str(self._grid_datasets[name].attrs.get("units", ""))

    def get_time_unit(self, name: str = "time") -> str:
        return str(self._time_datasets[name].attrs.get("units", ""))

    def get_diagnostic_unit(self, path: str) -> str:
        return str(self._get_diagnostic_dataset(path).attrs.get("units", ""))

    def get_field(
        self,
        field_name: str,
        *,
        time_slice: slice | int | None = None,
        zone_slice: slice | int | None = None,
        selection: Any = None,
    ) -> np.ndarray:
        """Return a field array with schema-aware legacy time/zone slicing."""
        if selection is not None:
            return self._get_field_dataset(field_name)[selection]
        if field_name == "radius":
            time_index = slice(None) if time_slice is None else time_slice
            zone_index = slice(None) if zone_slice is None else zone_slice
            if isinstance(time_index, int):
                values = self.get_dynamic_coordinate(time_index, location="center")
                if values is None:
                    raise KeyError("Dynamic coordinate field is unavailable.")
                return np.asarray(values[zone_index], dtype=np.float64)
            values = self.get_dynamic_coordinate(location="center")
            if values is None:
                raise KeyError("Dynamic coordinate field is unavailable.")
            return np.asarray(values[time_index, zone_index], dtype=np.float64)
        dataset = self._get_field_dataset(field_name)
        time_index = slice(None) if time_slice is None else time_slice
        dataset_ndim = len(tuple(dataset.shape))
        if dataset_ndim == 0:
            return dataset[()]
        axes = self.get_field_axes(field_name)
        index: list[Any] = [slice(None)] * dataset_ndim
        if axes and axes[0] == "time":
            index[0] = time_index
        elif time_slice is not None:
            index[0] = time_index
        if zone_slice is not None:
            try:
                zone_axis = axes.index("zone")
            except ValueError:
                if dataset_ndim < 2:
                    raise KeyError(f"Field {field_name!r} has no zone axis.")
                zone_axis = 1
            index[zone_axis] = zone_slice
        return dataset[tuple(index)]

    def get_snapshot_field(self, field_name: str, snapshot_index: int) -> np.ndarray:
        return self.get_field(field_name, time_slice=self._normalize_snapshot_index(snapshot_index))

    def get_lineout(self, field_name: str, snapshot_index: int) -> np.ndarray:
        return self.get_snapshot_field(field_name, snapshot_index)

    def get_time_trace(
        self,
        field_name: str,
        zone_index: int,
        *,
        time_slice: slice | int | None = None,
    ) -> np.ndarray:
        normalized_zone = int(zone_index)
        if normalized_zone < 0:
            normalized_zone += self.n_zones
        if normalized_zone < 0 or normalized_zone >= self.n_zones:
            raise IndexError(f"zone_index {zone_index} is out of range for {self.n_zones} zones.")
        if field_name == "radius":
            time_index = slice(None) if time_slice is None else time_slice
            model = self._coordinate_model()
            dataset_name = str(model.get("dynamic_center_dataset", "dynamic_coordinate_center"))
            if dataset_name in self._grid_datasets:
                return np.asarray(self._grid_datasets[dataset_name][time_index, normalized_zone], dtype=np.float64)
            if "radius" in self._field_datasets:
                radius_dataset = self._field_datasets["radius"]
                attr_location = str(radius_dataset.attrs.get("coordinate_location", "")).strip().lower()
                if attr_location == "center":
                    return np.asarray(radius_dataset[time_index, normalized_zone], dtype=np.float64)
        axes = self.get_field_axes(field_name)
        if "zone" not in axes:
            raise KeyError(f"Field {field_name!r} has no zone axis. Axes: {axes}.")
        return np.asarray(self.get_field(field_name, time_slice=time_slice, zone_slice=normalized_zone), dtype=np.float64)

    def get_time(self, name: str = "time", selection: slice | int | None = None) -> np.ndarray:
        dataset = self._time_datasets[name]
        index = slice(None) if selection is None else selection
        return dataset[index]

    def get_grid(self, name: str = "x", selection: slice | int | None = None) -> np.ndarray:
        if name in {"x", "coordinate_center"}:
            return self._slice_array(self.get_static_coordinate(location="center"), selection)
        if name == "coordinate_edge":
            return self._slice_array(self.get_static_coordinate(location="edge"), selection)
        dataset = self._grid_datasets[name]
        index = slice(None) if selection is None else selection
        return dataset[index]

    def get_radius(self, snapshot_index: int | None = None) -> np.ndarray:
        if snapshot_index is not None:
            dynamic = self.get_dynamic_coordinate(snapshot_index, location="center")
            if dynamic is not None:
                return dynamic
        return self.get_static_coordinate(location="center")

    def get_regions(self) -> dict[str, Any]:
        if self._regions_cache is None:
            self._regions_cache = _read_group(self._groups["regions"])
        return self._regions_cache

    def get_materials(self) -> dict[str, Any]:
        if self._materials_cache is None:
            self._materials_cache = _read_group(self._groups["materials"])
        return self._materials_cache

    def get_metadata(self) -> dict[str, Any]:
        if self._metadata_cache is None:
            self._metadata_cache = _read_group(self._groups["metadata"])
        return self._metadata_cache

    def get_region_mask(self, region_id: int) -> np.ndarray:
        if self._zone_region_ids is None:
            self._zone_region_ids = np.asarray(self._grid_datasets["zone_region_id"][:], dtype=np.int32)
        return self._zone_region_ids == int(region_id)

    def get_material_mask(self, material_id: int) -> np.ndarray:
        if self._zone_material_ids is None:
            self._zone_material_ids = np.asarray(self._grid_datasets["zone_material_index"][:], dtype=np.int32)
        return np.abs(self._zone_material_ids) == abs(int(material_id))

    def get_diagnostic(self, path: str, selection: Any = None) -> Any:
        """Return a diagnostic dataset or diagnostic slice by logical path."""
        dataset = self._get_diagnostic_dataset(path)
        index = () if selection is None else selection
        if index == ():
            return _normalize_loaded_value(dataset[()])
        return dataset[index]

    def close(self) -> None:
        if self._handle.id.valid:
            self._handle.close()

    def __enter__(self) -> "HeliosRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"HeliosRun(path={str(self.path)!r}, n_snapshots={self.n_snapshots}, "
            f"n_zones={self.n_zones}, n_regions={self.n_regions}, n_materials={self.n_materials})"
        )
