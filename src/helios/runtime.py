"""Run-scoped state shared across HELIOS Analyzer workspaces.

`RunContext` centralizes the viewer-facing state that would otherwise be copied
through ad hoc widget fields: the active file, grid/time shape, field registry,
coordinate choices, subset selections, and current snapshot index.

This is deliberately a small data carrier. Scientific data still lives in the
reader payloads; `RunContext` only tracks the currently selected view of a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from helios_viewer.models import OpenRunPayload


def _shared_read_only_array(values: np.ndarray | list[float] | tuple[float, ...], dtype) -> np.ndarray:
    """Return a read-only array view suitable for sharing across requests.

    `RunContext` metadata arrays are immutable run descriptors. Sharing them
    avoids repeated control-path copies when the controller snapshots context
    for derived-analysis requests, while the read-only flag keeps accidental
    mutation bugs obvious.
    """

    array = np.asarray(values, dtype=dtype)
    view = array.view()
    view.setflags(write=False)
    return view


@dataclass(slots=True)
class RunContext:
    """Current run-scoped UI state for HELIOS Analyzer.

    The context is owned by the viewer and shell layers. It intentionally does
    not duplicate full HDF5 arrays; it stores metadata, active coordinate
    semantics, subset selections, and snapshot/time navigation state.
    """

    path: Path | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    fields: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    time_values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    static_x_values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    zone_region_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    zone_material_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    has_dynamic_radius: bool = False
    snapshot_index: int = 0
    map_coordinate: str = "static_x"
    slice_coordinate: str = "zone"
    selected_region_ids: tuple[int, ...] = ()
    selected_material_ids: tuple[int, ...] = ()

    @classmethod
    def empty(cls) -> "RunContext":
        """Return an empty context used before a run is loaded."""

        return cls()

    @classmethod
    def from_payload(cls, payload: "OpenRunPayload") -> "RunContext":
        """Build a context from the lightweight run-open payload."""

        summary = dict(payload.summary)
        n_snapshots = int(summary.get("n_snapshots", len(payload.time)))
        default_map = "moving_radius" if payload.has_dynamic_radius else "static_x"
        default_slice = "zone"
        return cls(
            path=payload.path,
            summary=summary,
            metadata=dict(payload.metadata),
            fields=tuple(payload.fields),
            diagnostics=tuple(payload.diagnostics),
            time_values=_shared_read_only_array(payload.time, np.float64),
            static_x_values=_shared_read_only_array(payload.static_x, np.float64),
            zone_region_id=_shared_read_only_array(payload.zone_region_id, np.int32),
            zone_material_index=_shared_read_only_array(payload.zone_material_index, np.int32),
            has_dynamic_radius=bool(payload.has_dynamic_radius),
            snapshot_index=0 if n_snapshots <= 0 else 0,
            map_coordinate=default_map,
            slice_coordinate=default_slice,
        )

    @property
    def has_run(self) -> bool:
        """Whether the context currently points at a real loaded run."""

        return self.path is not None

    @property
    def n_zones(self) -> int:
        return int(self.summary.get("n_zones", self.static_x_values.size))

    @property
    def n_snapshots(self) -> int:
        return int(self.summary.get("n_snapshots", self.time_values.size))

    @property
    def context_key(self) -> tuple[object, ...]:
        """Stable key used for task invalidation and cache scoping."""

        return (
            str(self.path) if self.path is not None else None,
            int(self.n_snapshots),
            int(self.n_zones),
            bool(self.has_dynamic_radius),
        )

    def set_snapshot_index(self, index: int) -> None:
        """Clamp and store the active snapshot index."""

        if self.n_snapshots <= 0:
            self.snapshot_index = 0
            return
        self.snapshot_index = max(0, min(int(index), self.n_snapshots - 1))

    def set_coordinate_modes(self, *, map_coordinate: str, slice_coordinate: str) -> None:
        """Update the active coordinate semantics for map and slice views."""

        self.map_coordinate = str(map_coordinate)
        self.slice_coordinate = str(slice_coordinate)

    def set_subset(self, *, region_ids: tuple[int, ...], material_ids: tuple[int, ...]) -> None:
        """Store the active region/material subset selection."""

        self.selected_region_ids = tuple(int(value) for value in region_ids)
        self.selected_material_ids = tuple(int(value) for value in material_ids)

    def copy(self) -> "RunContext":
        """Return a detached copy safe to hand to background tasks.

        The dictionaries/tuples are copied normally, while immutable
        run-descriptor arrays are rewrapped as read-only shared views to avoid
        repeated control-path copies.
        """

        return replace(
            self,
            summary=dict(self.summary),
            metadata=dict(self.metadata),
            fields=tuple(self.fields),
            diagnostics=tuple(self.diagnostics),
            time_values=_shared_read_only_array(self.time_values, np.float64),
            static_x_values=_shared_read_only_array(self.static_x_values, np.float64),
            zone_region_id=_shared_read_only_array(self.zone_region_id, np.int32),
            zone_material_index=_shared_read_only_array(self.zone_material_index, np.int32),
            selected_region_ids=tuple(self.selected_region_ids),
            selected_material_ids=tuple(self.selected_material_ids),
        )
