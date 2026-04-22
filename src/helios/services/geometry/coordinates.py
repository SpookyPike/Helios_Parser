"""Coordinate and run-geometry helpers shared by viewer and derived services."""

from __future__ import annotations

from typing import Any

import numpy as np


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Convert monotonic zone-center coordinates into edge coordinates."""

    array = np.asarray(centers, dtype=np.float64)
    if array.size == 0:
        return np.empty(0, dtype=np.float64)
    if array.size == 1:
        half_width = 0.5
        return np.asarray([array[0] - half_width, array[0] + half_width], dtype=np.float64)
    edges = np.empty(array.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (array[:-1] + array[1:])
    edges[0] = array[0] - 0.5 * (array[1] - array[0])
    edges[-1] = array[-1] + 0.5 * (array[-1] - array[-2])
    return edges


def build_zone_property_from_regions(regions: dict[str, Any], values: np.ndarray, n_zones: int) -> np.ndarray:
    """Broadcast a region-scoped property to a per-zone array."""

    zone_values = np.full(int(n_zones), np.nan, dtype=np.float64)
    min_zone = np.asarray(regions["min_zone_index"], dtype=np.int32)
    max_zone = np.asarray(regions["max_zone_index"], dtype=np.int32)
    source = np.asarray(values, dtype=np.float64)
    for index, region_min in enumerate(min_zone):
        start = max(0, int(region_min) - 1)
        stop = min(int(n_zones), int(max_zone[index]))
        zone_values[start:stop] = float(source[index])
    return zone_values


def subset_mask(
    *,
    zone_region_id: np.ndarray,
    zone_material_index: np.ndarray,
    selected_region_ids: tuple[int, ...] | list[int],
    selected_material_ids: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Return the active zone mask using the same semantics as the viewer."""

    mask = np.ones(np.asarray(zone_region_id).shape[0], dtype=bool)
    region_ids = tuple(int(value) for value in selected_region_ids)
    material_ids = tuple(int(value) for value in selected_material_ids)
    if region_ids:
        mask &= np.isin(np.asarray(zone_region_id, dtype=np.int32), np.asarray(region_ids, dtype=np.int32))
    else:
        mask &= False
    if material_ids:
        mask &= np.isin(np.abs(np.asarray(zone_material_index, dtype=np.int32)), np.abs(np.asarray(material_ids, dtype=np.int32)))
    else:
        mask &= False
    return mask


def infer_laser_entry(
    *,
    metadata: dict[str, Any],
    n_zones: int,
    zone_region_id: np.ndarray,
    regions: dict[str, Any],
) -> dict[str, Any] | None:
    """Infer laser-entry provenance from the stored input metadata.

    This mirrors the logic used by the viewer summary panel and keeps the
    boundary interpretation consistent for derived analyses.
    """

    laser = metadata.get("input_parameters", {}).get("laser_source", {})
    if not isinstance(laser, dict):
        return None
    origin = laser.get("origin_zone_index")
    direction = laser.get("propagation_direction")
    if origin is None or direction is None:
        return None
    try:
        origin_zone = int(origin)
    except (TypeError, ValueError):
        return None
    direction_text = str(direction).strip()
    if direction_text == "Rmin":
        propagation_text = "toward Rmin"
        if origin_zone > int(n_zones):
            incident_boundary = "high-index boundary"
            first_zone = int(n_zones)
            boundary_kind = "high"
        elif 1 <= origin_zone <= int(n_zones):
            incident_boundary = "internal launch"
            first_zone = int(origin_zone)
            boundary_kind = "internal"
        else:
            return None
    elif direction_text == "Rmax":
        propagation_text = "toward Rmax"
        if origin_zone < 1:
            incident_boundary = "low-index boundary"
            first_zone = 1
            boundary_kind = "low"
        elif 1 <= origin_zone <= int(n_zones):
            incident_boundary = "internal launch"
            first_zone = int(origin_zone)
            boundary_kind = "internal"
        else:
            return None
    else:
        return None
    if first_zone < 1 or first_zone > np.asarray(zone_region_id).size:
        return None
    incident_region = int(np.asarray(zone_region_id, dtype=np.int32)[first_zone - 1])
    region_ids = np.asarray(regions["region_index"], dtype=np.int32)
    min_zones = np.asarray(regions["min_zone_index"], dtype=np.int32)
    max_zones = np.asarray(regions["max_zone_index"], dtype=np.int32)
    match = np.flatnonzero(region_ids == incident_region)
    if match.size == 0:
        return None
    region_offset = int(match[0])
    if boundary_kind == "low":
        boundary_label = f"Region {incident_region} low-index boundary"
    elif boundary_kind == "high":
        boundary_label = f"Region {incident_region} high-index boundary"
    else:
        boundary_label = (
            f"Region {incident_region}, launched inside zones "
            f"{int(min_zones[region_offset])}-{int(max_zones[region_offset])}"
        )
    return {
        "origin_zone_index": int(origin_zone),
        "propagation_direction": direction_text,
        "propagation_direction_text": propagation_text,
        "incident_boundary": incident_boundary,
        "first_physical_zone": int(first_zone),
        "incident_region": incident_region,
        "incident_region_boundary": boundary_label,
        "boundary_kind": boundary_kind,
    }


def region_interface_boundaries(regions: dict[str, Any]) -> list[tuple[int, int, int]]:
    """Return material-interface boundaries as ``(left_region, right_region, boundary_zone)``."""

    region_ids = np.asarray(regions["region_index"], dtype=np.int32)
    max_zones = np.asarray(regions["max_zone_index"], dtype=np.int32)
    boundaries: list[tuple[int, int, int]] = []
    for index in range(len(region_ids) - 1):
        boundaries.append((int(region_ids[index]), int(region_ids[index + 1]), int(max_zones[index])))
    return boundaries
