from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class SnapshotBlock:
    index: int
    span: TextSpan
    cycle_header_span: TextSpan
    diagnostics_span: TextSpan | None = None


@dataclass(frozen=True, slots=True)
class HeliosBlockIndex:
    header_span: TextSpan
    snapshot_blocks: tuple[SnapshotBlock, ...]

    @property
    def snapshot_count(self) -> int:
        return len(self.snapshot_blocks)


@dataclass(frozen=True, slots=True)
class RunStatusInfo:
    state: str
    source: str
    footer_message: str | None
    footer_datetime: str | None
    intended_end_time_s: float | None
    last_valid_snapshot_time_s: float | None
    indexed_snapshot_count: int
    valid_snapshot_count: int
    dropped_partial_final_block: bool = False
    damaged_final_block_reason: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class HeliosHeader:
    source: Path
    simulation_name: str
    code_version: str | None
    calculation_datetime: str | None
    header_sections: tuple[str, ...]
    block_delimiter: str
    n_regions: int
    n_materials: int
    n_zones: int
    grid: dict[str, np.ndarray]
    grid_units: dict[str, str]
    regions: dict[str, np.ndarray]
    region_units: dict[str, str]
    materials: dict[str, np.ndarray]
    material_units: dict[str, str]
    input_parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    run_status: RunStatusInfo | None = None


@dataclass(slots=True)
class HeliosPreview:
    source: Path
    header: HeliosHeader
    snapshot: Snapshot | None


@dataclass(slots=True)
class Snapshot:
    cycle: int
    time: float
    time_step: float
    time_step_control: str
    fields: dict[str, np.ndarray]
    field_units: dict[str, str]
    raw_field_map: dict[str, str]
    coordinate_name: str | None = None
    coordinate_center: np.ndarray | None = None
    coordinate_edge: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Simulation:
    source: Path
    grid: dict[str, np.ndarray]
    grid_units: dict[str, str]
    regions: dict[str, np.ndarray]
    region_units: dict[str, str]
    materials: dict[str, np.ndarray]
    material_units: dict[str, str]
    time: dict[str, np.ndarray]
    time_units: dict[str, str]
    fields: dict[str, np.ndarray]
    field_units: dict[str, str]
    diagnostics: dict[str, Any]
    input_parameters: dict[str, Any]
    metadata: dict[str, Any]
    raw_field_map: dict[str, str]
    run_status: RunStatusInfo | None = None
