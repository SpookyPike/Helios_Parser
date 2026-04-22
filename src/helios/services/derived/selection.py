"""Shared geometry, subset, and weighting helpers for Derived mode.

This module centralizes the analysis-selection semantics introduced in Phase
4.1. Derived services call into it so that geometry, filtering, and weighting
stay consistent across XRD, plasmon, transmission, and spectroscopy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, TypeVar

import numpy as np

from helios.instrumentation import increment_counter
from helios.runtime import RunContext
from helios.services.derived.models import AnalysisGeometryMetadata, AnalysisSelectionMetadata, DerivedRunData, DerivedWarning
from helios.services.geometry.coordinates import subset_mask


WEIGHTING_SIMPLE_MEAN = "simple_mean"
WEIGHTING_WIDTH = "width"
WEIGHTING_MASS = "mass"
WEIGHTING_ELECTRON_DENSITY = "electron_density"
WEIGHTING_ELECTRON_COLUMN = "electron_column"
_CachedPayloadT = TypeVar("_CachedPayloadT")

PRIMARY_WEIGHTING_OPTIONS = (
    WEIGHTING_ELECTRON_COLUMN,
    WEIGHTING_ELECTRON_DENSITY,
    WEIGHTING_MASS,
    WEIGHTING_WIDTH,
    WEIGHTING_SIMPLE_MEAN,
)


def coordinate_name(dataset: DerivedRunData) -> str:
    model = dataset.metadata.get("coordinate_model", {})
    if isinstance(model, dict):
        value = str(model.get("coordinate_name", "")).strip().lower()
        if value:
            return value
    geometry = str(dataset.metadata.get("geometry", "")).strip().upper()
    return "radius" if geometry in {"CYLINDRICAL", "SPHERICAL"} else "x"


def geometry_kind(dataset: DerivedRunData) -> str:
    return str(dataset.metadata.get("geometry", "")).strip().upper() or "PLANAR"


def static_coordinate_label(dataset: DerivedRunData, *, capitalize: bool = False) -> str:
    name = coordinate_name(dataset)
    if name == "radius":
        return "Radius" if capitalize else "radius"
    return "Static x" if capitalize else "static x"


def moving_coordinate_label(dataset: DerivedRunData, *, capitalize: bool = False) -> str:
    name = coordinate_name(dataset)
    if name == "radius":
        return "Moving-mesh radius" if capitalize else "moving-mesh radius"
    return "Moving-mesh x" if capitalize else "moving-mesh x"


def profile_coordinate_edges(dataset: DerivedRunData, snapshot_index: int, mode: str) -> np.ndarray | None:
    resolved = resolve_profile_coordinate_mode(dataset, RunContext.empty(), mode)
    if resolved == "moving_radius" and dataset.radius_edge_cm is not None:
        return np.asarray(dataset.radius_edge_cm[int(snapshot_index)], dtype=np.float64) * 1.0e4
    if resolved == "static_x":
        return np.asarray(dataset.static_x_edge_cm, dtype=np.float64) * 1.0e4
    if resolved == "zone":
        n_zones = int(dataset.summary["n_zones"])
        return np.linspace(0.5, float(n_zones) + 0.5, n_zones + 1, dtype=np.float64)
    return None


def _cylindrical_shell_edges_cm(dataset: DerivedRunData, snapshot_index: int) -> np.ndarray | None:
    if dataset.radius_edge_cm is not None:
        return np.asarray(dataset.radius_edge_cm[int(snapshot_index)], dtype=np.float64)
    static_edges = np.asarray(dataset.static_x_edge_cm, dtype=np.float64)
    return static_edges if static_edges.ndim == 1 and static_edges.size >= 2 else None


def cylindrical_shell_path_length_cm(shell_edges_cm: np.ndarray, impact_parameter_cm: float = 0.0) -> np.ndarray:
    """Return LOS shell lengths for a straight ray through concentric cylinders.

    The shell grid is defined by monotonically increasing radial edges. For an
    impact parameter ``b`` relative to the cylinder axis, the LOS length through
    shell ``[r_i, r_{i+1}]`` is the difference between the outer and inner chord
    lengths. Small negative radicands from floating-point noise are clipped to
    zero before the square root is taken.
    """

    edges = np.asarray(shell_edges_cm, dtype=np.float64).reshape(-1)
    if edges.size < 2:
        return np.zeros(0, dtype=np.float64)
    radii = np.maximum(edges, 0.0)
    impact = abs(float(impact_parameter_cm))
    outer = radii[1:]
    inner = radii[:-1]
    outer_term = np.sqrt(np.maximum(outer * outer - impact * impact, 0.0))
    inner_term = np.sqrt(np.maximum(inner * inner - impact * impact, 0.0))
    lengths = 2.0 * (outer_term - inner_term)
    return np.where(np.isfinite(lengths) & (lengths > 0.0), lengths, 0.0)


def cylindrical_path_note(module_name: str, dataset: DerivedRunData, geometry: AnalysisGeometryMetadata) -> DerivedWarning | None:
    if geometry_kind(dataset) != "CYLINDRICAL":
        return None
    impact_um = abs(float(geometry.impact_parameter_cm)) * 1.0e4
    if geometry.path_length_mode.startswith("cylindrical-shell"):
        chord_label = "centerline chord" if impact_um <= 1.0e-9 else f"impact-parameter chord (b={impact_um:.3f} um)"
        return DerivedWarning(
            module_name,
            f"Cylindrical LOS geometry uses shell intersections for a {chord_label}; the module physics kernel remains a quick-look approximation rather than full cylindrical transport.",
            severity="info",
        )
    return DerivedWarning(
        module_name,
        "Cylindrical geometry is loaded correctly, but LOS/path integration fell back to the older slab-like sec(theta) approximation because a usable shell-edge grid was unavailable.",
        severity="warning",
    )


def path_geometry_summary(dataset: DerivedRunData, geometry: AnalysisGeometryMetadata) -> str:
    if geometry_kind(dataset) != "CYLINDRICAL":
        return "planar LOS: exact (unchanged)"
    impact_um = abs(float(geometry.impact_parameter_cm)) * 1.0e4
    if geometry.path_length_mode == "cylindrical-shell-centerline":
        return "cylindrical shell LOS applied (centerline chord, b=0.000 um)"
    if geometry.path_length_mode == "cylindrical-shell-impact-parameter":
        return f"cylindrical shell LOS applied (b={impact_um:.3f} um)"
    return "cylindrical LOS: slab-like fallback"


def cylindrical_shell_factor_cm2(dataset: DerivedRunData, snapshot_index: int) -> np.ndarray | None:
    if geometry_kind(dataset) != "CYLINDRICAL":
        return None
    edge_grid = dataset.radius_edge_cm
    if edge_grid is not None:
        edges = np.asarray(edge_grid[int(snapshot_index)], dtype=np.float64)
    else:
        edges = np.asarray(dataset.static_x_edge_cm, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        return None
    shell_factor = np.asarray(edges[1:] ** 2 - edges[:-1] ** 2, dtype=np.float64)
    return np.where(np.isfinite(shell_factor) & (shell_factor > 0.0), shell_factor, 0.0)


@dataclass(slots=True)
class AnalysisStateCache:
    """Per-analysis-run cache for expensive per-snapshot selection state.

    The cache is intentionally local to a single analysis execution. It stores
    snapshot masks, LOS path lengths, and base weighting vectors so multiple
    derived modules can reuse the same arrays without re-running the filter and
    weighting setup repeatedly.
    """

    mask_cache: dict[tuple[object, ...], tuple[np.ndarray, AnalysisSelectionMetadata, tuple[DerivedWarning, ...]]] = field(default_factory=dict)
    path_cache: dict[tuple[object, ...], np.ndarray] = field(default_factory=dict)
    weight_cache: dict[tuple[object, ...], np.ndarray] = field(default_factory=dict)
    time_series_cache: dict[tuple[object, ...], object] = field(default_factory=dict)
    mask_hits: int = 0
    mask_misses: int = 0
    path_hits: int = 0
    path_misses: int = 0
    weight_hits: int = 0
    weight_misses: int = 0
    time_series_hits: int = 0
    time_series_misses: int = 0

    def stats(self) -> dict[str, int]:
        return {
            "mask_hits": int(self.mask_hits),
            "mask_misses": int(self.mask_misses),
            "path_hits": int(self.path_hits),
            "path_misses": int(self.path_misses),
            "weight_hits": int(self.weight_hits),
            "weight_misses": int(self.weight_misses),
            "time_series_hits": int(self.time_series_hits),
            "time_series_misses": int(self.time_series_misses),
        }


def cached_time_series_payload(
    cache_key: tuple[object, ...],
    *,
    analysis_cache: AnalysisStateCache | None,
    builder: Callable[[], _CachedPayloadT],
) -> _CachedPayloadT:
    """Return a per-analysis-run cached time-series payload."""

    if analysis_cache is None:
        return builder()
    cached = analysis_cache.time_series_cache.get(cache_key)
    if cached is not None:
        analysis_cache.time_series_hits += 1
        increment_counter("derived.cache.time_series.hit")
        return cached  # type: ignore[return-value]
    analysis_cache.time_series_misses += 1
    increment_counter("derived.cache.time_series.miss")
    payload = builder()
    analysis_cache.time_series_cache[cache_key] = payload
    return payload


def _readonly_mask_view(mask: np.ndarray) -> np.ndarray:
    view = np.asarray(mask, dtype=bool).view()
    view.setflags(write=False)
    return view


def _normalized_id_tuple(values: Iterable[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(sorted({int(value) for value in values}))


def _geometry_cache_key(geometry: AnalysisGeometryMetadata) -> tuple[object, ...]:
    return (
        str(geometry.observation_side),
        str(geometry.observation_boundary),
        round(float(geometry.line_of_sight_angle_deg), 12),
        round(float(geometry.line_of_sight_cosine), 12),
        round(float(geometry.impact_parameter_cm), 12),
        str(geometry.profile_coordinate_mode),
        str(geometry.path_length_mode),
        str(geometry.propagation_direction),
    )


def _mask_signature(mask: np.ndarray) -> bytes:
    array = np.asarray(mask, dtype=np.uint8)
    return np.packbits(array, bitorder="little").tobytes()


def selection_cache_key(selection: AnalysisSelectionMetadata) -> tuple[object, ...]:
    """Return the logical identity of a selection without scanning the mask."""

    return (
        bool(selection.reuse_viewer_subset),
        tuple(int(value) for value in selection.viewer_region_ids),
        tuple(int(value) for value in selection.viewer_material_ids),
        tuple(int(value) for value in selection.derived_region_ids),
        tuple(int(value) for value in selection.derived_material_ids),
        bool(selection.exclude_entry_region),
        bool(selection.exclude_low_density),
        None if selection.min_density_g_cm3 is None else round(float(selection.min_density_g_cm3), 12),
        bool(selection.exclude_opposite_velocity),
        selection.zone_index_lower,
        selection.zone_index_upper,
        str(selection.weighting_mode),
        int(selection.selected_zone_count),
    )


def selection_request_cache_key(
    context: RunContext,
    *,
    reuse_viewer_subset: bool,
    derived_region_ids: Iterable[int] | None,
    derived_material_ids: Iterable[int] | None,
    exclude_entry_region: bool,
    exclude_low_density: bool,
    min_density_g_cm3: float,
    exclude_opposite_velocity: bool,
    zone_index_lower: int | None,
    zone_index_upper: int | None,
    weighting_mode: str,
) -> tuple[object, ...]:
    """Return a snapshot-independent logical key for selection settings."""

    normalized_region_ids = _normalized_id_tuple(derived_region_ids)
    normalized_material_ids = _normalized_id_tuple(derived_material_ids)
    return (
        bool(reuse_viewer_subset and context.has_run),
        tuple(int(value) for value in context.selected_region_ids) if reuse_viewer_subset and context.has_run else (),
        tuple(int(value) for value in context.selected_material_ids) if reuse_viewer_subset and context.has_run else (),
        normalized_region_ids,
        normalized_material_ids,
        bool(exclude_entry_region),
        bool(exclude_low_density),
        float(max(0.0, float(min_density_g_cm3))),
        bool(exclude_opposite_velocity),
        zone_index_lower,
        zone_index_upper,
        str(weighting_mode),
    )


def shared_time_series_selection_state(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    geometry: AnalysisGeometryMetadata,
    weighting_mode: str,
    reuse_viewer_subset: bool,
    derived_region_ids: Iterable[int] | None,
    derived_material_ids: Iterable[int] | None,
    exclude_entry_region: bool,
    exclude_low_density: bool,
    min_density_g_cm3: float,
    exclude_opposite_velocity: bool,
    zone_index_lower: int | None,
    zone_index_upper: int | None,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> tuple[np.ndarray, tuple[tuple[object, ...], ...]]:
    """Return cached per-snapshot masks and logical selection identities.

    Several time-plot modules need the same subset semantics over every
    snapshot. Materializing that state once per logical request keeps the hot
    loops local to the backend and avoids repeating mask selection plumbing in
    every module.
    """

    request_key = selection_request_cache_key(
        context,
        reuse_viewer_subset=reuse_viewer_subset,
        derived_region_ids=derived_region_ids,
        derived_material_ids=derived_material_ids,
        exclude_entry_region=exclude_entry_region,
        exclude_low_density=exclude_low_density,
        min_density_g_cm3=min_density_g_cm3,
        exclude_opposite_velocity=exclude_opposite_velocity,
        zone_index_lower=zone_index_lower,
        zone_index_upper=zone_index_upper,
        weighting_mode=weighting_mode,
    )
    cache_key = (
        "selection_state",
        _geometry_cache_key(geometry),
        request_key,
    )

    def _build() -> tuple[np.ndarray, tuple[tuple[object, ...], ...]]:
        n_times = int(dataset.time_s.size)
        n_zones = int(dataset.summary["n_zones"])
        mask_matrix = np.zeros((n_times, n_zones), dtype=bool)
        selection_keys: list[tuple[object, ...]] = []
        for time_index in range(n_times):
            if progress_check is not None and (time_index % 8 == 0):
                progress_check()
            time_mask, time_selection, _ = build_analysis_mask(
                dataset,
                context,
                snapshot_index=time_index,
                geometry=geometry,
                reuse_viewer_subset=reuse_viewer_subset,
                derived_region_ids=derived_region_ids,
                derived_material_ids=derived_material_ids,
                exclude_entry_region=exclude_entry_region,
                exclude_low_density=exclude_low_density,
                min_density_g_cm3=min_density_g_cm3,
                exclude_opposite_velocity=exclude_opposite_velocity,
                zone_index_lower=zone_index_lower,
                zone_index_upper=zone_index_upper,
                weighting_mode=weighting_mode,
                analysis_cache=analysis_cache,
            )
            mask_matrix[time_index] = np.asarray(time_mask, dtype=bool)
            selection_keys.append(selection_cache_key(time_selection))
        mask_matrix.setflags(write=False)
        return mask_matrix, tuple(selection_keys)

    return cached_time_series_payload(
        cache_key,
        analysis_cache=analysis_cache,
        builder=_build,
    )


def shared_time_series_weighted_means(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    geometry: AnalysisGeometryMetadata,
    weighting_mode: str,
    field_series: tuple[tuple[str, np.ndarray], ...],
    reuse_viewer_subset: bool,
    derived_region_ids: Iterable[int] | None,
    derived_material_ids: Iterable[int] | None,
    exclude_entry_region: bool,
    exclude_low_density: bool,
    min_density_g_cm3: float,
    exclude_opposite_velocity: bool,
    zone_index_lower: int | None,
    zone_index_upper: int | None,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> dict[str, np.ndarray]:
    """Compute reusable weighted time-series means once per logical request."""

    prepared_fields = tuple((str(name), np.asarray(values, dtype=np.float64)) for name, values in field_series)
    request_key = selection_request_cache_key(
        context,
        reuse_viewer_subset=reuse_viewer_subset,
        derived_region_ids=derived_region_ids,
        derived_material_ids=derived_material_ids,
        exclude_entry_region=exclude_entry_region,
        exclude_low_density=exclude_low_density,
        min_density_g_cm3=min_density_g_cm3,
        exclude_opposite_velocity=exclude_opposite_velocity,
        zone_index_lower=zone_index_lower,
        zone_index_upper=zone_index_upper,
        weighting_mode=weighting_mode,
    )
    cache_key = (
        "weighted_means",
        _geometry_cache_key(geometry),
        request_key,
        tuple(name for name, _ in prepared_fields),
    )

    def _build() -> dict[str, np.ndarray]:
        n_times = int(dataset.time_s.size)
        if not prepared_fields or n_times <= 0:
            return {name: np.zeros(0, dtype=np.float64) for name, _ in prepared_fields}
        n_fields = len(prepared_fields)
        n_zones = int(dataset.summary["n_zones"])
        series_matrix = np.full((n_fields, n_times), np.nan, dtype=np.float64)
        snapshot_stack = np.empty((n_fields, n_zones), dtype=np.float64)
        mask_matrix, selection_keys = shared_time_series_selection_state(
            dataset,
            context,
            geometry=geometry,
            weighting_mode=weighting_mode,
            reuse_viewer_subset=reuse_viewer_subset,
            derived_region_ids=derived_region_ids,
            derived_material_ids=derived_material_ids,
            exclude_entry_region=exclude_entry_region,
            exclude_low_density=exclude_low_density,
            min_density_g_cm3=min_density_g_cm3,
            exclude_opposite_velocity=exclude_opposite_velocity,
            zone_index_lower=zone_index_lower,
            zone_index_upper=zone_index_upper,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        for time_index in range(n_times):
            if progress_check is not None and (time_index % 8 == 0):
                progress_check()
            time_mask = np.asarray(mask_matrix[int(time_index)], dtype=bool)
            for field_index, (_name, values) in enumerate(prepared_fields):
                snapshot_stack[field_index] = values[int(time_index)] if values.ndim == 2 else values
            series_matrix[:, time_index] = weighted_means(
                snapshot_stack,
                dataset,
                time_index,
                time_mask,
                mode=weighting_mode,
                geometry=geometry,
                selection_key=selection_keys[int(time_index)],
                analysis_cache=analysis_cache,
            )
        return {name: series_matrix[index].copy() for index, (name, _values) in enumerate(prepared_fields)}

    return cached_time_series_payload(cache_key, analysis_cache=analysis_cache, builder=_build)


def _base_weight_array(
    dataset: DerivedRunData,
    snapshot_index: int,
    mask: np.ndarray,
    *,
    mode: str,
    geometry: AnalysisGeometryMetadata,
    selection_key: tuple[object, ...] | None = None,
    analysis_cache: AnalysisStateCache | None = None,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    resolved_mode = resolve_weighting_mode(mode, module_name="generic")
    cache_key = None
    if analysis_cache is not None:
        cache_key = (
            int(snapshot_index),
            str(resolved_mode),
            _geometry_cache_key(geometry),
            selection_key if selection_key is not None else ("mask-fallback", _mask_signature(mask)),
        )
        cached = analysis_cache.weight_cache.get(cache_key)
        if cached is not None:
            analysis_cache.weight_hits += 1
            increment_counter("derived.cache.weight.hit")
            return cached
        analysis_cache.weight_misses += 1
        increment_counter("derived.cache.weight.miss")

    path = path_length_cm(dataset, snapshot_index, geometry, analysis_cache=analysis_cache)
    if resolved_mode == WEIGHTING_SIMPLE_MEAN:
        weights = np.ones(mask.shape[0], dtype=np.float64)
    elif resolved_mode == WEIGHTING_MASS:
        density = np.asarray(dataset.density_g_cm3[int(snapshot_index)], dtype=np.float64)
        shell_factor = cylindrical_shell_factor_cm2(dataset, snapshot_index)
        if shell_factor is not None and shell_factor.shape == density.shape:
            # For 1D cylindrical runs, shell mass per unit axial length scales as
            # rho * (r_outer^2 - r_inner^2). This exact local geometry factor is
            # available from the explicit edge grid and avoids falling back to a
            # planar rho*dr approximation for mass-weighted averages.
            weights = density * shell_factor
        else:
            weights = density * path
    elif resolved_mode == WEIGHTING_ELECTRON_DENSITY:
        weights = np.asarray(dataset.electron_density_cm3[int(snapshot_index)], dtype=np.float64)
    elif resolved_mode == WEIGHTING_ELECTRON_COLUMN:
        weights = np.asarray(dataset.electron_density_cm3[int(snapshot_index)], dtype=np.float64) * path
    else:
        weights = path
    weights = np.where(mask & np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if analysis_cache is not None and cache_key is not None:
        analysis_cache.weight_cache[cache_key] = weights
    return weights


def infer_primary_propagation_direction(dataset: DerivedRunData, context: RunContext) -> tuple[str, float]:
    """Infer the dominant front direction and expected velocity sign.

    Returns ``("high_to_low" | "low_to_high", expected_velocity_sign)`` where
    the velocity sign is expressed in zone-index coordinate order.
    """

    entry = dataset.laser_entry
    if entry is not None:
        if str(entry.get("boundary_kind")) == "high":
            return "high_to_low", -1.0
        if str(entry.get("boundary_kind")) == "low":
            return "low_to_high", 1.0
    velocity = np.asarray(dataset.velocity_cm_s, dtype=np.float64)
    finite = velocity[np.isfinite(velocity)]
    if finite.size == 0:
        return "high_to_low", -1.0
    median_velocity = float(np.nanmedian(finite))
    if median_velocity < 0.0:
        return "high_to_low", -1.0
    return "low_to_high", 1.0


def resolve_observation_boundary(
    dataset: DerivedRunData,
    *,
    observation_side: str,
    propagation_direction: str,
) -> str:
    """Return ``low`` or ``high`` for the chosen observation side.

    ``front`` is defined as the same side as the inferred/declared laser entry.
    ``back`` is the opposite side.
    """

    side = str(observation_side).strip().lower()
    if side not in {"front", "back"}:
        side = "front"
    if propagation_direction == "high_to_low":
        return "high" if side == "front" else "low"
    return "low" if side == "front" else "high"


def line_of_sight_cosine_from_angle(angle_deg: float) -> float:
    angle = np.deg2rad(float(angle_deg))
    return float(np.cos(angle))


def observation_axis_cosine(
    dataset: DerivedRunData,
    *,
    observation_side: str,
    los_cosine: float,
    propagation_direction: str,
) -> float:
    """Project LOS cosine into the stored coordinate sign convention."""

    boundary = resolve_observation_boundary(dataset, observation_side=observation_side, propagation_direction=propagation_direction)
    sign = 1.0 if boundary == "low" else -1.0
    return sign * float(los_cosine)


def path_length_scale(los_cosine: float) -> float:
    cosine = abs(float(los_cosine))
    return 1.0 / max(cosine, 1.0e-6)


def resolve_profile_coordinate_mode(dataset: DerivedRunData, context: RunContext, requested: str) -> str:
    """Resolve the profile/display coordinate mode for Derived plots."""

    mode = str(requested or "viewer").strip().lower()
    if mode in {"viewer", "run_default"}:
        candidate = str(context.slice_coordinate or "").strip().lower()
        if candidate == "radius":
            mode = "moving_radius"
        elif candidate in {"moving_radius", "static_x", "zone"}:
            mode = candidate
        else:
            mode = "zone" if dataset.summary.get("n_zones", 0) else ("moving_radius" if dataset.radius_cm is not None else "static_x")
    if mode == "moving_radius" and dataset.radius_cm is None:
        mode = "static_x"
    if mode not in {"moving_radius", "static_x", "zone"}:
        mode = "moving_radius" if dataset.radius_cm is not None else "static_x"
    return mode


def profile_coordinate_values(dataset: DerivedRunData, snapshot_index: int, mode: str) -> tuple[np.ndarray, str]:
    resolved = resolve_profile_coordinate_mode(dataset, RunContext.empty(), mode)
    if resolved == "moving_radius" and dataset.radius_cm is not None:
        return (
            np.asarray(dataset.radius_cm[int(snapshot_index)], dtype=np.float64) * 1.0e4,
            f"{moving_coordinate_label(dataset, capitalize=True)} [um]",
        )
    if resolved == "static_x":
        return np.asarray(dataset.static_x_cm, dtype=np.float64) * 1.0e4, f"{static_coordinate_label(dataset, capitalize=True)} [um]"
    return np.arange(1, int(dataset.summary["n_zones"]) + 1, dtype=np.float64), "Zone index"


def profile_boundary_positions(dataset: DerivedRunData, snapshot_index: int, mode: str) -> tuple[float, ...]:
    resolved = resolve_profile_coordinate_mode(dataset, RunContext.empty(), mode)
    boundaries = np.asarray(dataset.regions["max_zone_index"], dtype=np.int32)[:-1]
    if boundaries.size == 0:
        return ()
    if resolved == "zone":
        return tuple(float(boundary) + 0.5 for boundary in boundaries)
    coordinate_edges = profile_coordinate_edges(dataset, snapshot_index, resolved)
    if coordinate_edges is None:
        return ()
    positions: list[float] = []
    for boundary in boundaries:
        edge_index = int(boundary)
        if 0 < edge_index < coordinate_edges.size:
            positions.append(float(coordinate_edges[edge_index]))
    return tuple(positions)


def build_analysis_geometry(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    observation_side: str,
    line_of_sight_angle_deg: float,
    line_of_sight_impact_parameter_cm: float = 0.0,
    profile_coordinate_mode: str,
) -> AnalysisGeometryMetadata:
    propagation_direction, _ = infer_primary_propagation_direction(dataset, context)
    los_cosine = line_of_sight_cosine_from_angle(line_of_sight_angle_deg)
    observation_boundary = resolve_observation_boundary(
        dataset,
        observation_side=observation_side,
        propagation_direction=propagation_direction,
    )
    geometry_type = geometry_kind(dataset)
    impact_parameter_cm = max(abs(float(line_of_sight_impact_parameter_cm)), 0.0)
    if geometry_type == "CYLINDRICAL":
        has_shell_edges = _cylindrical_shell_edges_cm(dataset, 0) is not None
        if has_shell_edges:
            path_length_mode = "cylindrical-shell-centerline" if impact_parameter_cm <= 0.0 else "cylindrical-shell-impact-parameter"
        else:
            path_length_mode = "cylindrical-shell-unavailable-fallback-slab"
    else:
        path_length_mode = "normal-incidence" if abs(los_cosine) >= 0.999 else "oblique-sec(theta)"
    return AnalysisGeometryMetadata(
        observation_side=str(observation_side),
        observation_boundary=observation_boundary,
        line_of_sight_angle_deg=float(line_of_sight_angle_deg),
        line_of_sight_cosine=float(los_cosine),
        profile_coordinate_mode=resolve_profile_coordinate_mode(dataset, context, profile_coordinate_mode),
        path_length_mode=path_length_mode,
        propagation_direction=propagation_direction,
        impact_parameter_cm=impact_parameter_cm,
    )


def build_analysis_mask(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    geometry: AnalysisGeometryMetadata,
    reuse_viewer_subset: bool,
    derived_region_ids: Iterable[int] | None,
    derived_material_ids: Iterable[int] | None,
    exclude_entry_region: bool,
    exclude_low_density: bool,
    min_density_g_cm3: float,
    exclude_opposite_velocity: bool,
    zone_index_lower: int | None,
    zone_index_upper: int | None,
    weighting_mode: str,
    analysis_cache: AnalysisStateCache | None = None,
) -> tuple[np.ndarray, AnalysisSelectionMetadata, tuple[DerivedWarning, ...]]:
    """Construct the active zone mask and structured metadata."""

    normalized_region_ids = _normalized_id_tuple(derived_region_ids)
    normalized_material_ids = _normalized_id_tuple(derived_material_ids)
    cache_key = None
    if analysis_cache is not None:
        cache_key = (
            int(snapshot_index),
            _geometry_cache_key(geometry),
            bool(reuse_viewer_subset and context.has_run),
            tuple(int(value) for value in context.selected_region_ids) if reuse_viewer_subset and context.has_run else (),
            tuple(int(value) for value in context.selected_material_ids) if reuse_viewer_subset and context.has_run else (),
            normalized_region_ids,
            normalized_material_ids,
            bool(exclude_entry_region),
            bool(exclude_low_density),
            float(max(0.0, float(min_density_g_cm3))),
            bool(exclude_opposite_velocity),
            zone_index_lower,
            zone_index_upper,
            str(weighting_mode),
        )
        cached = analysis_cache.mask_cache.get(cache_key)
        if cached is not None:
            analysis_cache.mask_hits += 1
            increment_counter("derived.cache.mask.hit")
            cached_mask, cached_selection, cached_warnings = cached
            return _readonly_mask_view(cached_mask), cached_selection, cached_warnings
        analysis_cache.mask_misses += 1
        increment_counter("derived.cache.mask.miss")

    warnings: list[DerivedWarning] = []
    n_zones = int(dataset.summary["n_zones"])
    if reuse_viewer_subset and context.has_run:
        mask = subset_mask(
            zone_region_id=dataset.zone_region_id,
            zone_material_index=dataset.zone_material_index,
            selected_region_ids=context.selected_region_ids,
            selected_material_ids=context.selected_material_ids,
        )
    else:
        mask = np.ones(n_zones, dtype=bool)

    notes: list[str] = []

    if normalized_region_ids is not None:
        before = int(np.count_nonzero(mask))
        mask &= np.isin(dataset.zone_region_id, np.asarray(normalized_region_ids, dtype=np.int32))
        removed = before - int(np.count_nonzero(mask))
        notes.append(
            "Derived regions: "
            + (", ".join(str(value) for value in normalized_region_ids) if normalized_region_ids else "none selected")
        )
        if removed == 0:
            notes.append("Derived region filter removed 0 zones")
    if normalized_material_ids is not None:
        before = int(np.count_nonzero(mask))
        mask &= np.isin(np.abs(dataset.zone_material_index), np.abs(np.asarray(normalized_material_ids, dtype=np.int32)))
        removed = before - int(np.count_nonzero(mask))
        notes.append(
            "Derived materials: "
            + (", ".join(str(value) for value in normalized_material_ids) if normalized_material_ids else "none selected")
        )
        if removed == 0:
            notes.append("Derived material filter removed 0 zones")

    lower = int(zone_index_lower) if zone_index_lower is not None else 1
    upper = int(zone_index_upper) if zone_index_upper not in {None, 0} else n_zones
    lower = max(1, min(lower, n_zones))
    upper = max(lower, min(upper, n_zones))
    if lower > 1 or upper < n_zones:
        zone_selector = np.zeros(n_zones, dtype=bool)
        zone_selector[lower - 1 : upper] = True
        mask &= zone_selector
        notes.append(f"Zone clip: {lower}-{upper}")

    if exclude_entry_region and dataset.laser_entry is not None and dataset.laser_entry.get("incident_region") is not None:
        entry_region = int(dataset.laser_entry["incident_region"])
        mask &= dataset.zone_region_id != entry_region
        notes.append(f"Excluded entry region {entry_region}")

    threshold = max(0.0, float(min_density_g_cm3))
    if exclude_low_density or threshold > 0.0:
        density = np.asarray(dataset.density_g_cm3[int(snapshot_index)], dtype=np.float64)
        mask &= np.isfinite(density) & (density >= threshold)
        notes.append(f"rho >= {threshold:.4g} g/cm3")

    if exclude_opposite_velocity:
        _, expected_velocity_sign = infer_primary_propagation_direction(dataset, context)
        velocity = np.asarray(dataset.velocity_cm_s[int(snapshot_index)], dtype=np.float64)
        mask &= np.isfinite(velocity) & (expected_velocity_sign * velocity >= 0.0)
        notes.append("Excluded opposite-velocity zones")

    selected_zone_count = int(np.count_nonzero(mask))
    if selected_zone_count == 0:
        warnings.append(
            DerivedWarning(
                "selection",
                "No active zones remain after applying the Derived geometry/filter selection.",
                severity="error",
            )
        )

    selection = AnalysisSelectionMetadata(
        reuse_viewer_subset=bool(reuse_viewer_subset),
        viewer_region_ids=tuple(int(value) for value in context.selected_region_ids),
        viewer_material_ids=tuple(int(value) for value in context.selected_material_ids),
        derived_region_ids=() if normalized_region_ids is None else normalized_region_ids,
        derived_material_ids=() if normalized_material_ids is None else normalized_material_ids,
        exclude_entry_region=bool(exclude_entry_region),
        exclude_low_density=bool(exclude_low_density or threshold > 0.0),
        min_density_g_cm3=(threshold if threshold > 0.0 else None),
        exclude_opposite_velocity=bool(exclude_opposite_velocity),
        zone_index_lower=lower,
        zone_index_upper=upper,
        weighting_mode=str(weighting_mode),
        selected_zone_count=selected_zone_count,
        notes=tuple(notes),
    )
    warning_tuple = tuple(warnings)
    if analysis_cache is not None and cache_key is not None:
        canonical_mask = np.asarray(mask, dtype=bool).copy()
        canonical_mask.setflags(write=False)
        analysis_cache.mask_cache[cache_key] = (canonical_mask, selection, warning_tuple)
        return _readonly_mask_view(canonical_mask), selection, warning_tuple
    return mask, selection, warning_tuple


def resolve_weighting_mode(requested: str, *, module_name: str) -> str:
    """Resolve ``auto`` into a module-specific default weighting mode."""

    mode = str(requested or "auto").strip().lower()
    if mode != "auto":
        if mode in PRIMARY_WEIGHTING_OPTIONS:
            return mode
        return WEIGHTING_WIDTH
    if module_name == "plasmon":
        return WEIGHTING_ELECTRON_COLUMN
    if module_name == "spectroscopy":
        return WEIGHTING_MASS
    if module_name == "xrd":
        return WEIGHTING_WIDTH
    return WEIGHTING_WIDTH


def path_length_cm(
    dataset: DerivedRunData,
    snapshot_index: int,
    geometry: AnalysisGeometryMetadata,
    *,
    analysis_cache: AnalysisStateCache | None = None,
) -> np.ndarray:
    """Return the current LOS path length per zone.

    Planar runs retain the existing slab path model exactly. Cylindrical runs
    use shell intersections through the explicit edge grid when available.
    """

    cache_key = None
    if analysis_cache is not None:
        cache_key = (int(snapshot_index), _geometry_cache_key(geometry))
        cached = analysis_cache.path_cache.get(cache_key)
        if cached is not None:
            analysis_cache.path_hits += 1
            increment_counter("derived.cache.path.hit")
            return cached
        analysis_cache.path_misses += 1
        increment_counter("derived.cache.path.miss")
    if geometry_kind(dataset) == "CYLINDRICAL" and geometry.path_length_mode.startswith("cylindrical-shell"):
        shell_edges = _cylindrical_shell_edges_cm(dataset, snapshot_index)
        if shell_edges is not None:
            result = cylindrical_shell_path_length_cm(shell_edges, geometry.impact_parameter_cm)
        else:
            widths = np.asarray(dataset.zone_width_cm[int(snapshot_index)], dtype=np.float64)
            result = widths * path_length_scale(geometry.line_of_sight_cosine)
    else:
        widths = np.asarray(dataset.zone_width_cm[int(snapshot_index)], dtype=np.float64)
        result = widths * path_length_scale(geometry.line_of_sight_cosine)
    if analysis_cache is not None and cache_key is not None:
        analysis_cache.path_cache[cache_key] = result
    return result


def weight_array(
    values: np.ndarray,
    dataset: DerivedRunData,
    snapshot_index: int,
    mask: np.ndarray,
    *,
    mode: str,
    geometry: AnalysisGeometryMetadata,
    selection_key: tuple[object, ...] | None = None,
    analysis_cache: AnalysisStateCache | None = None,
) -> np.ndarray:
    """Return non-negative weights for a snapshot weighting mode."""

    mask = np.asarray(mask, dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    weights = _base_weight_array(
        dataset,
        snapshot_index,
        mask,
        mode=mode,
        geometry=geometry,
        selection_key=selection_key,
        analysis_cache=analysis_cache,
    )
    weights = np.where(np.isfinite(values) & np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    return weights


def weighted_means(
    values: np.ndarray,
    dataset: DerivedRunData,
    snapshot_index: int,
    mask: np.ndarray,
    *,
    mode: str,
    geometry: AnalysisGeometryMetadata,
    selection_key: tuple[object, ...] | None = None,
    analysis_cache: AnalysisStateCache | None = None,
) -> np.ndarray:
    """Compute one or more weighted means in a single vectorized reduction."""

    arrays = np.asarray(values, dtype=np.float64)
    one_dimensional = arrays.ndim == 1
    if one_dimensional:
        arrays = arrays[np.newaxis, :]
    base_weights = _base_weight_array(
        dataset,
        snapshot_index,
        mask,
        mode=mode,
        geometry=geometry,
        selection_key=selection_key,
        analysis_cache=analysis_cache,
    )
    valid = np.isfinite(arrays) & np.isfinite(base_weights)[np.newaxis, :] & (base_weights[np.newaxis, :] > 0.0)
    weighted_values = np.where(valid, arrays * base_weights[np.newaxis, :], 0.0)
    numerator = np.sum(weighted_values, axis=1, dtype=np.float64)
    denominator = np.sum(np.where(valid, base_weights[np.newaxis, :], 0.0), axis=1, dtype=np.float64)
    means = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    return means[0] if one_dimensional else means


def weighted_average(
    values: np.ndarray,
    dataset: DerivedRunData,
    snapshot_index: int,
    mask: np.ndarray,
    *,
    mode: str,
    geometry: AnalysisGeometryMetadata,
    selection_key: tuple[object, ...] | None = None,
    analysis_cache: AnalysisStateCache | None = None,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(
        weighted_means(
            array,
            dataset,
            snapshot_index,
            mask,
            mode=mode,
            geometry=geometry,
            selection_key=selection_key,
            analysis_cache=analysis_cache,
        )
    )
