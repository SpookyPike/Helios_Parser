from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CoordinateValidationIssue:
    code: str
    message: str
    severity: str = "warning"


def coordinate_name_for_geometry(geometry: str | None) -> str:
    normalized = str(geometry or "").strip().upper()
    if normalized in {"CYLINDRICAL", "SPHERICAL"}:
        return "radius"
    return "x"


def stabilize_boundary_edge(boundary_edge: float, widths: np.ndarray) -> float:
    del widths
    return float(boundary_edge)


def _positive_widths(widths: np.ndarray) -> np.ndarray:
    width_array = np.asarray(widths, dtype=np.float64).reshape(-1)
    if width_array.size and (not np.all(np.isfinite(width_array)) or not np.all(width_array > 0.0)):
        raise ValueError("Zone widths must be finite and strictly positive.")
    return width_array


def validate_coordinate_edges(edges: np.ndarray) -> np.ndarray:
    edge_array = np.asarray(edges, dtype=np.float64).reshape(-1)
    if edge_array.size <= 1:
        return edge_array
    if not np.all(np.isfinite(edge_array)):
        raise ValueError("Coordinate edges must be finite.")
    if not np.all(np.diff(edge_array) > 0.0):
        raise ValueError("Coordinate edges must be strictly increasing.")
    return edge_array


def _cumulative_edges_from_widths(boundary_edge: float, widths: np.ndarray) -> np.ndarray:
    width_array = _positive_widths(widths)
    edges = np.empty(width_array.size + 1, dtype=np.float64)
    edges[0] = float(boundary_edge)
    if width_array.size:
        edges[1:] = edges[0] + np.cumsum(width_array, dtype=np.float64)
    return edges


def _width_consistency_tolerance(widths: np.ndarray) -> float:
    width_array = _positive_widths(widths)
    if width_array.size == 0:
        return 1.0e-12
    # HELIOS edge columns are sometimes rounded more coarsely than the zone
    # width column. A 0.5% absolute-width tolerance is conservative enough to
    # absorb file-precision mismatch without silently treating materially
    # different geometries as width-consistent.
    return max(float(np.nanmax(np.abs(width_array))) * 5.0e-3, 1.0e-12)


def _edges_match_widths(edges: np.ndarray, widths: np.ndarray) -> bool:
    edge_array = np.asarray(edges, dtype=np.float64).reshape(-1)
    width_array = _positive_widths(widths)
    if edge_array.size != width_array.size + 1:
        return False
    return bool(np.allclose(edge_array[1:] - edge_array[:-1], width_array, rtol=0.0, atol=_width_consistency_tolerance(width_array)))


def _edges_close_to_cumulative(parsed_edges: np.ndarray, cumulative_edges: np.ndarray) -> bool:
    parsed = np.asarray(parsed_edges, dtype=np.float64).reshape(-1)
    cumulative = np.asarray(cumulative_edges, dtype=np.float64).reshape(-1)
    if parsed.shape != cumulative.shape:
        return False
    scale = max(float(np.nanmax(np.abs(cumulative))) if cumulative.size else 0.0, 1.0)
    return bool(np.allclose(parsed, cumulative, rtol=0.0, atol=max(scale * 1.0e-3, 1.0e-12)))


def _boundary_noise_tolerance(widths: np.ndarray) -> float:
    width_array = _positive_widths(widths)
    if width_array.size == 0:
        return 1.0e-18
    scale = float(np.nanmax(np.abs(width_array)))
    return max(scale * 1.0e-9, 1.0e-18)


def _stabilize_cylindrical_boundary(
    boundary_edge: float,
    widths: np.ndarray,
    *,
    geometry: str | None,
    issues: list[CoordinateValidationIssue] | None,
) -> float:
    normalized = str(geometry or "").strip().upper()
    if normalized != "CYLINDRICAL":
        return float(boundary_edge)
    value = float(boundary_edge)
    tolerance = _boundary_noise_tolerance(widths)
    if value < 0.0 and abs(value) <= tolerance:
        if issues is not None:
            issues.append(
                CoordinateValidationIssue(
                    code="cylindrical_boundary_noise_clipped",
                    message=(
                        "A tiny negative cylindrical inner-edge value was clipped to 0 because it is within "
                        "floating-point noise tolerance."
                    ),
                    severity="warning",
                )
            )
        return 0.0
    if value < 0.0:
        raise ValueError("Cylindrical coordinate edges must not cross below r=0.")
    return value


def midpoint_centers_from_edges(edges: np.ndarray) -> np.ndarray:
    edge_array = validate_coordinate_edges(edges)
    if edge_array.size <= 1:
        return np.empty(0, dtype=np.float64)
    return np.asarray(0.5 * (edge_array[:-1] + edge_array[1:]), dtype=np.float64)


def build_coordinate_edge_array(
    outer_edges: np.ndarray,
    zone_widths: np.ndarray,
    *,
    boundary_edge: float | None = None,
    geometry: str | None = None,
    issues: list[CoordinateValidationIssue] | None = None,
) -> np.ndarray:
    outer = np.asarray(outer_edges, dtype=np.float64).reshape(-1)
    widths = _positive_widths(zone_widths)
    if outer.size != widths.size:
        raise ValueError("Outer-edge coordinates and zone widths must have matching lengths.")
    if outer.size == 0:
        return np.empty(0, dtype=np.float64)
    parsed_edges = np.empty(outer.size + 1, dtype=np.float64)
    parsed_edges[1:] = outer
    if boundary_edge is None or not np.isfinite(boundary_edge):
        inferred = float(outer[0]) - float(widths[0])
    else:
        inferred = float(boundary_edge)
    parsed_edges[0] = _stabilize_cylindrical_boundary(
        stabilize_boundary_edge(inferred, widths),
        widths,
        geometry=geometry,
        issues=issues,
    )
    cumulative_edges = _cumulative_edges_from_widths(parsed_edges[0], widths)
    try:
        validated = validate_coordinate_edges(parsed_edges)
    except ValueError:
        # Some HELIOS tables round the edge/Radius column enough to introduce
        # repeated outer-edge values late in the target. If the preserved
        # boundary plus widths still defines a consistent edge grid close to the
        # parsed values, prefer that exact-width reconstruction instead of
        # letting rounded plateaus break the coordinate model. This fallback is
        # intentionally conservative: it is only used for non-monotonic parsed
        # edges, not for merely width-inconsistent but still monotonic grids.
        if _edges_close_to_cumulative(parsed_edges, cumulative_edges):
            if issues is not None:
                issues.append(
                    CoordinateValidationIssue(
                        code="rounded_edge_fallback",
                        message=(
                            "Parsed coordinate edges were non-monotonic but matched the boundary+width cumulative "
                            "grid within file-precision tolerance, so cumulative-width reconstruction was used."
                        ),
                        severity="warning",
                    )
                )
            return validate_coordinate_edges(cumulative_edges)
        raise
    if _edges_match_widths(validated, widths):
        return validated
    if issues is not None:
        issues.append(
            CoordinateValidationIssue(
                code="edge_width_mismatch",
                message=(
                    "Parsed coordinate edges are monotonic but materially inconsistent with the zone-width-implied "
                    "edge grid. The parsed edges were preserved and this mismatch should be treated as a validation issue."
                ),
                severity="warning",
            )
        )
    return validated


def centers_from_edges_and_widths(edges: np.ndarray, zone_widths: np.ndarray) -> np.ndarray:
    edge_array = np.asarray(edges, dtype=np.float64).reshape(-1)
    widths = np.asarray(zone_widths, dtype=np.float64).reshape(-1)
    if edge_array.size != widths.size + 1:
        raise ValueError("Edge array must be one element longer than the zone-width array.")
    return midpoint_centers_from_edges(edge_array)


def build_coordinate_model(
    outer_edges: np.ndarray,
    zone_widths: np.ndarray,
    *,
    boundary_edge: float | None = None,
    geometry: str | None = None,
    issues: list[CoordinateValidationIssue] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    edges = build_coordinate_edge_array(
        outer_edges,
        zone_widths,
        boundary_edge=boundary_edge,
        geometry=geometry,
        issues=issues,
    )
    centers = centers_from_edges_and_widths(edges, zone_widths)
    return edges, centers


def build_coordinate_edge_grid(
    outer_edges: np.ndarray,
    zone_widths: np.ndarray,
    *,
    boundary_edges: np.ndarray | None = None,
    geometry: str | None = None,
    issues: list[CoordinateValidationIssue] | None = None,
) -> np.ndarray:
    outer = np.asarray(outer_edges, dtype=np.float64)
    widths = np.asarray(zone_widths, dtype=np.float64)
    if outer.ndim != 2 or widths.ndim != 2:
        raise ValueError("Dynamic coordinate reconstruction expects 2D snapshot-zone arrays.")
    if outer.shape != widths.shape:
        raise ValueError("Dynamic outer-edge coordinates and widths must have matching shapes.")
    width_rows = _positive_widths(widths.reshape(-1)).reshape(widths.shape)
    parsed_edges = np.empty((outer.shape[0], outer.shape[1] + 1), dtype=np.float64)
    parsed_edges[:, 1:] = outer
    if boundary_edges is None:
        inferred = outer[:, 0] - width_rows[:, 0]
    else:
        inferred = np.asarray(boundary_edges, dtype=np.float64).reshape(-1)
        if inferred.shape[0] != outer.shape[0]:
            raise ValueError("Dynamic boundary-edge vector length must match snapshot count.")
    for row_index, boundary in enumerate(inferred):
        parsed_edges[row_index, 0] = _stabilize_cylindrical_boundary(
            float(boundary),
            width_rows[row_index],
            geometry=geometry,
            issues=issues,
        )
    cumulative_edges = np.empty_like(parsed_edges)
    cumulative_edges[:, 0] = parsed_edges[:, 0]
    cumulative_edges[:, 1:] = parsed_edges[:, 0][:, None] + np.cumsum(width_rows, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(parsed_edges)):
        raise ValueError("Dynamic coordinate edges must be finite.")
    result = parsed_edges.copy()
    diffs = np.diff(parsed_edges, axis=1)
    invalid_rows = np.any(diffs <= 0.0, axis=1)
    for row_index in np.flatnonzero(invalid_rows):
        if _edges_close_to_cumulative(parsed_edges[row_index], cumulative_edges[row_index]):
            if issues is not None:
                issues.append(
                    CoordinateValidationIssue(
                        code="rounded_dynamic_edge_fallback",
                        message=(
                            f"Snapshot {int(row_index)} had non-monotonic parsed coordinate edges, so the boundary+width "
                            "cumulative reconstruction was used because it matched within file-precision tolerance."
                        ),
                        severity="warning",
                    )
                )
            result[row_index] = cumulative_edges[row_index]
        else:
            raise ValueError("Dynamic coordinate edges must be strictly increasing within each snapshot.")
    width_mismatch = np.max(np.abs(np.diff(result, axis=1) - width_rows), axis=1)
    tolerance = np.asarray([_width_consistency_tolerance(row) for row in width_rows], dtype=np.float64)
    for row_index in np.flatnonzero(width_mismatch > tolerance):
        if issues is not None:
            issues.append(
                CoordinateValidationIssue(
                    code="dynamic_edge_width_mismatch",
                    message=(
                        f"Snapshot {int(row_index)} has monotonic parsed coordinate edges that are materially "
                        "inconsistent with the zone-width-implied cumulative grid. The parsed edges were preserved."
                    ),
                    severity="warning",
                )
            )
    if not np.all(np.diff(result, axis=1) > 0.0):
        raise ValueError("Dynamic coordinate edges must be strictly increasing within each snapshot.")
    return result


def centers_from_edge_grid_and_widths(edges: np.ndarray, zone_widths: np.ndarray) -> np.ndarray:
    edge_array = np.asarray(edges, dtype=np.float64)
    widths = np.asarray(zone_widths, dtype=np.float64)
    if edge_array.ndim != 2 or widths.ndim != 2:
        raise ValueError("Dynamic coordinate reconstruction expects 2D snapshot-zone arrays.")
    if edge_array.shape != (widths.shape[0], widths.shape[1] + 1):
        raise ValueError("Dynamic edge grid must be one column wider than the zone-width grid.")
    if edge_array.shape[1] <= 1:
        return np.empty((edge_array.shape[0], 0), dtype=np.float64)
    if not np.all(np.isfinite(edge_array)):
        raise ValueError("Dynamic coordinate edges must be finite.")
    return np.asarray(0.5 * (edge_array[:, :-1] + edge_array[:, 1:]), dtype=np.float64)
