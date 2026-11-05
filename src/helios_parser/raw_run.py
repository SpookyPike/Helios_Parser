"""Read-only run view over a parsed HELIOS deck.

`HeliosRun` reads a converted HDF5 file. The command line also has to answer
questions about a deck that was never converted, so this adapter exposes the same
accessor surface over a `Simulation` that `HeliosDocument.parse_full` produced.
It exists for the `helios_diag` commands and for `load_raw_run_data`; nothing in
the desktop application uses it.

The contract is deliberately narrow: only the methods those callers use. Units
come from the deck's own header and are never inferred. A raw deck carries no
`schema_version`, no writer contract and no VISAR structure, so those answers say
so instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .model import Simulation
from .reader import FieldMetadata, VisarSupportMetadata

_RAW_FORMAT_LABELS = {
    "log": "HELIOS log",
    "bpf": "HELIOS BPF",
    "exodus": "HELIOS Exodus II",
}


@dataclass(slots=True)
class RawRun:
    """A parsed deck behind the `HeliosRun` accessor surface."""

    simulation: Simulation
    format: str
    source_precedence: str = ""

    # ------------------------------------------------------------------ shape

    @property
    def path(self) -> Path:
        return Path(self.simulation.source)

    @property
    def n_zones(self) -> int:
        zone_id = self.simulation.grid.get("zone_id")
        if zone_id is not None:
            return int(np.asarray(zone_id).size)
        if self.simulation.fields:
            return int(next(iter(self.simulation.fields.values())).shape[-1])
        return 0

    @property
    def n_snapshots(self) -> int:
        return int(np.asarray(self.simulation.time["time"]).size)

    @property
    def n_regions(self) -> int:
        return int(np.asarray(self.simulation.regions.get("region_index", ())).size)

    @property
    def n_materials(self) -> int:
        return int(np.asarray(self.simulation.materials.get("index", ())).size)

    # ----------------------------------------------------------------- fields

    def list_fields(self) -> list[str]:
        return list(self.simulation.fields)

    def has_field(self, field_name: str) -> bool:
        return field_name in self.simulation.fields

    def has_grid(self, name: str) -> bool:
        return name in self.simulation.grid

    def get_field_metadata(self, field_name: str) -> FieldMetadata:
        values = self.simulation.fields[field_name]
        unit = self.simulation.field_units.get(field_name, "")
        label = self.simulation.raw_field_map.get(field_name, field_name)
        return FieldMetadata(
            name=field_name,
            shape=tuple(int(size) for size in values.shape),
            dtype=str(values.dtype),
            source=self.format,
            dimensions=("time", "zone"),
            unit=unit,
            label=label,
            status="validated" if unit else "legacy",
        )

    def get_field_unit(self, field_name: str) -> str:
        return self.simulation.field_units.get(field_name, "")

    def get_field(
        self,
        field_name: str,
        *,
        time_slice: slice | int | None = None,
        zone_slice: slice | int | None = None,
        selection: Any = None,
    ) -> np.ndarray:
        values = self.simulation.fields[field_name]
        if selection is not None:
            return np.asarray(values[selection])
        time_index = slice(None) if time_slice is None else time_slice
        zone_index = slice(None) if zone_slice is None else zone_slice
        return np.asarray(values[time_index, zone_index], dtype=np.float64)

    def get_snapshot_field(self, field_name: str, snapshot_index: int) -> np.ndarray:
        return self.get_field(field_name, time_slice=self._normalize_snapshot_index(snapshot_index))

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
        return np.asarray(
            self.get_field(field_name, time_slice=time_slice, zone_slice=normalized_zone),
            dtype=np.float64,
        )

    # ------------------------------------------------------------------- time

    def get_time(self, name: str = "time", selection: slice | int | None = None) -> np.ndarray:
        if name not in self.simulation.time:
            raise KeyError(f"Time axis {name!r} is not present in a raw deck.")
        values = self.simulation.time[name]
        index = slice(None) if selection is None else selection
        return values[index]

    def _normalize_snapshot_index(self, snapshot_index: int) -> int:
        index = int(snapshot_index)
        if index < 0:
            index += self.n_snapshots
        if index < 0 or index >= self.n_snapshots:
            raise IndexError(f"snapshot index {snapshot_index} is out of range for {self.n_snapshots} snapshots.")
        return index

    # ------------------------------------------------------------------- grid

    def get_grid(self, name: str = "x", selection: slice | int | None = None) -> np.ndarray:
        if name in {"x", "coordinate_center"}:
            values = self.get_static_coordinate(location="center")
        elif name == "coordinate_edge":
            values = self.get_static_coordinate(location="edge")
        elif name in self.simulation.grid:
            values = np.asarray(self.simulation.grid[name])
        else:
            raise KeyError(f"Grid array {name!r} is not present in a raw deck.")
        if selection is None:
            return values
        return values[selection]

    def get_grid_unit(self, name: str = "x") -> str:
        if name == "coordinate_edge":
            return self.simulation.grid_units.get("coordinate_edge", "")
        if name in {"x", "coordinate_center"}:
            return self.simulation.grid_units.get("coordinate_center", "")
        return self.simulation.grid_units.get(name, "")

    def get_static_coordinate(self, *, location: str = "center") -> np.ndarray:
        key = "coordinate_edge" if location == "edge" else "coordinate_center"
        if key in self.simulation.grid:
            return np.asarray(self.simulation.grid[key], dtype=np.float64)
        if key == "coordinate_center" and "x" in self.simulation.grid:
            return np.asarray(self.simulation.grid["x"], dtype=np.float64)
        raise KeyError(f"Static coordinate {key!r} is not present in a raw deck.")

    def has_dynamic_coordinate(self) -> bool:
        return "dynamic_coordinate_center" in self.simulation.grid

    def get_dynamic_coordinate(self, snapshot_index: int | None = None, *, location: str = "center") -> np.ndarray | None:
        key = "dynamic_coordinate_edge" if location == "edge" else "dynamic_coordinate_center"
        values = self.simulation.grid.get(key)
        if values is None:
            return None
        if snapshot_index is None:
            return np.asarray(values, dtype=np.float64)
        return np.asarray(values[self._normalize_snapshot_index(snapshot_index)], dtype=np.float64)

    # --------------------------------------------------------- regions, masks

    def list_region_ids(self) -> list[int]:
        return [int(value) for value in self.simulation.regions.get("region_index", ())]

    def list_material_ids(self) -> list[int]:
        return [int(value) for value in self.simulation.materials.get("index", ())]

    def get_region_mask(self, region_id: int) -> np.ndarray:
        values = np.asarray(self.simulation.grid["zone_region_id"], dtype=np.int32)
        return values == int(region_id)

    def get_material_mask(self, material_id: int) -> np.ndarray:
        values = np.asarray(self.simulation.grid["zone_material_index"], dtype=np.int32)
        return np.abs(values) == abs(int(material_id))

    # -------------------------------------------------------------- metadata

    def get_regions(self) -> dict[str, Any]:
        return {name: np.asarray(values) for name, values in self.simulation.regions.items()}

    def get_materials(self) -> dict[str, Any]:
        return {name: np.asarray(values) for name, values in self.simulation.materials.items()}

    def summary(self) -> dict[str, Any]:
        metadata = self.get_metadata()
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
            "available_fields": list(self.simulation.fields),
            "run_status": self.get_run_status(),
            "metadata": {
                key: metadata.get(key)
                for key in (
                    "simulation_name",
                    "geometry",
                    "eos_model",
                    "helios_version",
                    "calculation_datetime",
                    "source_file",
                )
                if key in metadata
            },
        }

    def get_visar_support_metadata(self) -> VisarSupportMetadata:
        return VisarSupportMetadata(
            velocity_field_name="velocity" if "velocity" in self.simulation.fields else None,
            time_axis_name="time",
            static_coordinate_name="x" if "x" in self.simulation.grid else None,
            dynamic_coordinate_field_name="radius" if self.has_dynamic_coordinate() else None,
            boundary_indexing_consistent=False,
            candidate_boundaries=(),
            event_timing_source=None,
            notes=("raw deck: no VISAR structure is recorded",),
        )

    def get_metadata(self) -> dict[str, Any]:
        metadata = dict(self.simulation.metadata)
        metadata["input_parameters"] = self.simulation.input_parameters
        metadata.setdefault("n_zones", self.n_zones)
        metadata.setdefault("n_snapshots", self.n_snapshots)
        metadata.setdefault("available_fields", list(self.simulation.fields))
        metadata.setdefault("raw_field_map", dict(self.simulation.raw_field_map))
        metadata.setdefault("parse_mode", f"raw_{self.format}")
        return metadata

    def get_file_attributes(self) -> dict[str, Any]:
        metadata = self.simulation.metadata
        return {
            "format": self.format,
            "geometry": metadata.get("geometry"),
            "helios_version": metadata.get("helios_version"),
            "source_file": str(self.path),
            "parse_mode": self.source_precedence or f"raw_{self.format}",
        }

    def get_run_status(self) -> dict[str, Any]:
        raw = self.simulation.metadata.get("run_status")
        status = dict(raw) if isinstance(raw, dict) else {}
        status.setdefault("state", "unknown")
        status.setdefault("source", f"raw_{self.format}")
        status.setdefault("footer_message", None)
        status.setdefault("footer_datetime", None)
        status.setdefault("intended_end_time_s", None)
        status.setdefault(
            "last_valid_snapshot_time_s",
            float(self.get_time()[-1]) if self.n_snapshots else None,
        )
        status.setdefault("indexed_snapshot_count", int(self.n_snapshots))
        status.setdefault("valid_snapshot_count", int(self.n_snapshots))
        status.setdefault("dropped_partial_final_block", False)
        status.setdefault("damaged_final_block_reason", None)
        status.setdefault("notes", ())
        return status

    def check_visar_readiness(self):
        raise LookupError(
            "A raw HELIOS deck carries no VISAR structure; convert it first to inspect VISAR readiness."
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> "RawRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------ description

    def describe_format(self) -> str:
        return _RAW_FORMAT_LABELS.get(self.format, self.format)