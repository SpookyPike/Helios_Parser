from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class OpenRunPayload:
    run_generation: int
    path: Path
    summary: dict[str, Any]
    metadata: dict[str, Any]
    fields: list[str]
    field_units: dict[str, str]
    diagnostics: list[str]
    diagnostic_units: dict[str, str]
    regions: dict[str, Any]
    materials: dict[str, Any]
    time: np.ndarray
    time_unit: str
    # Legacy field name kept for backward compatibility with the viewer. This
    # array now always carries static zone-center coordinates; physical edges
    # are exposed separately through `static_x_edges`.
    static_x: np.ndarray
    static_x_edges: np.ndarray
    static_x_unit: str
    zone_region_id: np.ndarray
    zone_material_index: np.ndarray
    has_dynamic_radius: bool
    radius_unit: str
    run_status: dict[str, Any] | None = None
    visar_support_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class FieldPayload:
    run_generation: int
    field_name: str
    unit: str
    data: np.ndarray
    # For coordinate fields such as HELIOS' dynamic Radius column, `data`
    # carries zone centers while `edge_data` carries the explicit edge grid.
    edge_data: np.ndarray | None = None


@dataclass(slots=True)
class DiagnosticPayload:
    run_generation: int
    path: str
    unit: str
    data: np.ndarray
