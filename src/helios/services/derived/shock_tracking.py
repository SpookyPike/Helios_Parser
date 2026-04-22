"""Wave and shock tracking helpers for HELIOS Derived mode."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from helios.runtime import RunContext
from helios.services.derived.models import (
    AnalysisGeometryMetadata,
    DerivedRunData,
    DerivedWarning,
    InterfaceEventRecord,
    InterfaceEventsResult,
    ShockInterfaceCrossing,
    ShockTrackingResult,
    WaveBranchSummary,
    WaveEvidenceFormulaHook,
    WaveEvidenceMap,
    WaveFrontCandidate,
    WaveFrontFitSeed,
    WaveLocalStateSummary,
    WaveTrackingResult,
)
from helios.services.derived.selection import (
    AnalysisStateCache,
    build_analysis_geometry,
    cached_time_series_payload,
    infer_primary_propagation_direction,
    moving_coordinate_label,
    selection_request_cache_key,
    shared_time_series_selection_state,
    static_coordinate_label,
)
from helios.services.geometry.coordinates import region_interface_boundaries

if TYPE_CHECKING:
    from helios.services.derived.analysis import DerivedAnalysisParameters


@dataclass(frozen=True, slots=True)
class _ShockEventScaffold:
    """Internal interface scaffold shared by shock and wave tracking."""

    time_s: np.ndarray
    interface_position_grid: np.ndarray
    interface_boundaries: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class _WaveTrackingConfig:
    """Numerical controls for the 1D multi-branch wave tracker."""

    max_candidates_per_family: int = 2
    max_candidates_per_frame: int = 5
    shock_threshold_quantile: float = 92.0
    release_threshold_quantile: float = 92.0
    contact_threshold_quantile: float = 88.0
    minimum_score: float = 0.18
    merge_distance_cells: float = 1.5
    merge_score_ratio: float = 1.6
    association_position_scale_cm: float = 2.0e-4
    association_velocity_scale_cm_s: float = 1.0e7
    association_score_scale: float = 0.75
    association_miss_cost: float = 1.5
    association_birth_cost: float = 1.0
    association_max_cost: float = 6.5
    lambda_x: float = 1.0
    lambda_v: float = 0.1
    lambda_s: float = 0.35
    lambda_t: float = 1.25
    lambda_i: float = 0.35
    provisional_sample_limit: int = 2
    tracked_branch_min_samples: int = 5
    display_branch_limit: int = 12


@dataclass(frozen=True, slots=True)
class _LocalizedFit:
    position_cm: float
    width_cm: float
    fit_quality: float
    confidence: float
    fit_seed: WaveFrontFitSeed


@dataclass(frozen=True, slots=True)
class _FrameCandidateSeed:
    snapshot_index: int
    family: str
    family_score: float
    interface_index: float
    direction: str | None
    direction_sign: float | None
    candidate_type: str
    ambiguous: bool
    upstream_state: WaveLocalStateSummary | None
    downstream_state: WaveLocalStateSummary | None
    fit: _LocalizedFit
    notes: tuple[str, ...] = ()

    @property
    def position_cm(self) -> float:
        return float(self.fit.position_cm)

    @property
    def width_cm(self) -> float:
        return float(self.fit.width_cm)

    @property
    def fit_quality(self) -> float:
        return float(self.fit.fit_quality)

    @property
    def confidence(self) -> float:
        return float(self.fit.confidence)


@dataclass(slots=True)
class _ActiveBranch:
    branch_id: str
    family: str
    branch_type: str
    primary: bool
    candidates: list[_FrameCandidateSeed]
    missed_frames: int = 0


@dataclass(frozen=True, slots=True)
class _EvidencePayload:
    formula_hooks: tuple[WaveEvidenceFormulaHook, ...]
    interface_position_grid: np.ndarray
    family_score: dict[str, np.ndarray]
    family_direction: dict[str, np.ndarray]
    candidate_thresholds: dict[str, float]


_WAVE_CONFIG = _WaveTrackingConfig()


def _smooth_profile(values: np.ndarray, passes: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 3:
        return array.copy()
    result = array.copy()
    for _ in range(max(1, int(passes))):
        updated = result.copy()
        updated[1:-1] = 0.25 * result[:-2] + 0.5 * result[1:-1] + 0.25 * result[2:]
        result = updated
    return result


def _smooth_track(track: np.ndarray, *, direction: str) -> np.ndarray:
    valid = np.asarray(track, dtype=np.float64)
    finite = np.isfinite(valid)
    if np.count_nonzero(finite) < 3:
        return valid
    indices = np.flatnonzero(finite)
    smoothed = valid.copy()
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= np.sum(kernel)
    segment = np.asarray(valid[indices], dtype=np.float64)
    padded = np.pad(segment, (2, 2), mode="edge")
    filtered = np.convolve(padded, kernel, mode="valid")
    filtered[0] = segment[0]
    filtered[-1] = segment[-1]
    if direction == "high_to_low":
        smoothed[indices] = np.minimum.accumulate(filtered)
    else:
        smoothed[indices] = np.maximum.accumulate(filtered)
    return smoothed


def _build_interface_position_grid(dataset: DerivedRunData) -> np.ndarray:
    n_snapshots = int(dataset.time_s.size)
    n_interfaces = max(0, int(dataset.summary["n_zones"]) - 1)
    if n_snapshots == 0 or n_interfaces == 0:
        return np.full((n_snapshots, n_interfaces), np.nan, dtype=np.float64)
    if dataset.radius_edge_cm is not None:
        radius_edges = np.asarray(dataset.radius_edge_cm, dtype=np.float64)
        grid = np.full((n_snapshots, n_interfaces), np.nan, dtype=np.float64)
        if radius_edges.ndim == 2 and radius_edges.shape[1] >= 3:
            rows = min(n_snapshots, radius_edges.shape[0])
            cols = min(n_interfaces, radius_edges.shape[1] - 2)
            if cols > 0:
                grid[:rows, :cols] = radius_edges[:rows, 1 : 1 + cols]
        return grid
    static_edges = np.asarray(dataset.static_x_edge_cm, dtype=np.float64)
    static_positions = static_edges[1:-1] if static_edges.size >= 3 else np.empty(0, dtype=np.float64)
    if static_positions.size == 0:
        return np.full((n_snapshots, n_interfaces), np.nan, dtype=np.float64)
    return np.broadcast_to(static_positions, (n_snapshots, static_positions.size))


def _position_from_interface_index(interface_positions: np.ndarray, interface_index: float) -> float:
    if interface_positions.size == 0 or not np.isfinite(interface_index):
        return float("nan")
    grid = np.arange(interface_positions.size, dtype=np.float64)
    return float(np.interp(float(interface_index), grid, interface_positions))


def _interpolate_threshold_crossing(
    time_s: np.ndarray,
    track_index: np.ndarray,
    *,
    threshold: float,
    direction: str,
) -> tuple[int | None, float | None]:
    valid = np.isfinite(track_index) & np.isfinite(time_s)
    indices = np.flatnonzero(valid)
    if indices.size < 2:
        return None, None
    for left, right in zip(indices[:-1], indices[1:]):
        x0 = float(track_index[left])
        x1 = float(track_index[right])
        if direction == "high_to_low":
            crossed = x0 > threshold >= x1
        else:
            crossed = x0 < threshold <= x1
        if not crossed:
            continue
        if x1 == x0:
            return int(right), float(time_s[right])
        fraction = (threshold - x0) / (x1 - x0)
        crossing_time = float(time_s[left] + fraction * (float(time_s[right]) - float(time_s[left])))
        return int(right), crossing_time
    return None, None


def _activation_threshold(scores: np.ndarray) -> float:
    finite_positive = np.asarray(scores, dtype=np.float64)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size == 0:
        return 0.025
    return max(0.025, float(np.percentile(finite_positive, 95.0)) * 0.15)


def _coordinate_label(dataset: DerivedRunData) -> str:
    return moving_coordinate_label(dataset) if dataset.radius_cm is not None else static_coordinate_label(dataset)


def _zone_centers_cm(dataset: DerivedRunData, snapshot_index: int) -> np.ndarray:
    if dataset.radius_cm is not None:
        return np.asarray(dataset.radius_cm[int(snapshot_index)], dtype=np.float64)
    return np.asarray(dataset.static_x_cm, dtype=np.float64)


def _zone_edges_cm(dataset: DerivedRunData, snapshot_index: int) -> np.ndarray:
    if dataset.radius_edge_cm is not None:
        return np.asarray(dataset.radius_edge_cm[int(snapshot_index)], dtype=np.float64)
    return np.asarray(dataset.static_x_edge_cm, dtype=np.float64)


def _interface_spacing_cm(dataset: DerivedRunData, snapshot_index: int) -> np.ndarray:
    centers = _zone_centers_cm(dataset, snapshot_index)
    if centers.size < 2:
        return np.zeros(0, dtype=np.float64)
    spacing = np.diff(centers)
    positive = spacing[np.isfinite(spacing) & (spacing > 0.0)]
    minimum = float(np.nanmin(positive)) if positive.size else 1.0
    return np.where(np.isfinite(spacing) & (spacing > 0.0), spacing, minimum)


def _safe_positive_log(values: np.ndarray, *, floor: float = 1.0e-30) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    clipped = np.where(np.isfinite(array) & (array > floor), array, floor)
    return np.log(clipped)


def _positive_activation(values: np.ndarray, scale: float) -> np.ndarray:
    safe_scale = max(float(scale), 1.0e-12)
    normalized = np.maximum(np.asarray(values, dtype=np.float64), 0.0) / safe_scale
    return np.clip(normalized, 0.0, 2.5)


def _robust_scale(values: np.ndarray, fallback: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = np.abs(array[np.isfinite(array)])
    if finite.size == 0:
        return float(fallback)
    return max(float(np.percentile(finite, 80.0)), float(fallback))


def _total_pressure_snapshot(dataset: DerivedRunData, snapshot_index: int) -> np.ndarray | None:
    if dataset.pressure_total_j_cm3 is not None:
        return np.asarray(dataset.pressure_total_j_cm3[int(snapshot_index)], dtype=np.float64)
    components: list[np.ndarray] = []
    if dataset.pressure_i_j_cm3 is not None:
        components.append(np.asarray(dataset.pressure_i_j_cm3[int(snapshot_index)], dtype=np.float64))
    if dataset.pressure_e_j_cm3 is not None:
        components.append(np.asarray(dataset.pressure_e_j_cm3[int(snapshot_index)], dtype=np.float64))
    if dataset.pressure_radiation_j_cm3 is not None:
        components.append(np.asarray(dataset.pressure_radiation_j_cm3[int(snapshot_index)], dtype=np.float64))
    if not components:
        return None
    return np.nansum(np.asarray(components, dtype=np.float64), axis=0)


def _state_summary(dataset: DerivedRunData, snapshot_index: int, zone_index: int | None) -> WaveLocalStateSummary | None:
    if zone_index is None:
        return None
    n_zones = int(dataset.summary["n_zones"])
    zone = max(0, min(int(zone_index), n_zones - 1))
    total_pressure = _total_pressure_snapshot(dataset, snapshot_index)
    return WaveLocalStateSummary(
        density_g_cm3=float(dataset.density_g_cm3[snapshot_index, zone]),
        velocity_cm_s=float(dataset.velocity_cm_s[snapshot_index, zone]),
        pressure_total_j_cm3=(None if total_pressure is None else float(total_pressure[zone])),
        temperature_e_ev=float(dataset.temperature_e_ev[snapshot_index, zone]),
        temperature_i_ev=float(dataset.temperature_i_ev[snapshot_index, zone]),
        temperature_radiation_ev=(
            None if dataset.temperature_radiation_ev is None else float(dataset.temperature_radiation_ev[snapshot_index, zone])
        ),
        mean_charge=float(dataset.mean_charge[snapshot_index, zone]),
        material_index=int(dataset.zone_material_index[zone]),
        region_id=int(dataset.zone_region_id[zone]),
    )


def _upstream_downstream_zone_indices(interface_index: int, direction: str | None, n_zones: int) -> tuple[int | None, int | None]:
    low_zone = max(0, min(interface_index, n_zones - 1))
    high_zone = max(0, min(interface_index + 1, n_zones - 1))
    if direction == "low_to_high":
        return high_zone, low_zone
    if direction == "high_to_low":
        return low_zone, high_zone
    return low_zone, high_zone


def _local_maxima_indices(values: np.ndarray, threshold: float) -> np.ndarray:
    row = np.asarray(values, dtype=np.float64)
    if row.size == 0:
        return np.empty(0, dtype=np.int32)
    finite = np.where(np.isfinite(row), row, -np.inf)
    maxima: list[int] = []
    for index in range(row.size):
        value = finite[index]
        if not np.isfinite(value) or value < threshold:
            continue
        left = finite[index - 1] if index > 0 else -np.inf
        right = finite[index + 1] if index + 1 < row.size else -np.inf
        if value >= left and value >= right:
            maxima.append(index)
    return np.asarray(maxima, dtype=np.int32)


def _family_threshold(score: np.ndarray, *, quantile: float, minimum: float) -> float:
    finite = np.asarray(score, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return float(minimum)
    return max(float(minimum), float(np.percentile(finite, float(quantile))))


def _branch_type_from_candidate(
    *,
    family: str,
    direction: str | None,
    primary_direction: str,
    interface_crossed: bool,
    near_rear_boundary: bool,
    ambiguous: bool,
) -> str:
    if ambiguous or family == "merged_unresolved":
        return "merged_unresolved_front"
    if family == "contact_like":
        return "contact_transition"
    if family == "release_like":
        if near_rear_boundary and direction is not None and direction != primary_direction:
            return "rear_rarefaction"
        return "release_rarefaction"
    if family != "shock_like":
        return family
    if direction is not None and direction != primary_direction:
        return "reflected_shock"
    if interface_crossed:
        return "transmitted_shock"
    return "compressive_shock"


def shock_tracking_formula_hooks(dataset: DerivedRunData) -> tuple[WaveEvidenceFormulaHook, ...]:
    """Return the evidence families used by the multi-branch wave tracker."""

    pressure_field = "pressure" if dataset.pressure_total_j_cm3 is not None else "pressure_components"
    q_art_field = "artificial_viscosity" if dataset.artificial_viscosity_j_cm3 is not None else "unavailable"
    return (
        WaveEvidenceFormulaHook(
            family="shock_like",
            expression_label="w_rho*Phi(+/-Delta ln rho) + w_P*Phi(+/-Delta ln P_tot) + w_u*Phi(-du/dx) + w_q*Phi(q_art/(P_tot+eps)) + w_T*Phi(+/-Delta ln Te) - artifact_penalty",
            configurable_terms=("w_rho", "w_P", "w_u", "w_q", "w_T", "artifact_penalty", "Phi"),
            required_fields=("density", "velocity"),
            optional_fields=(pressure_field, q_art_field, "temperature_e"),
            notes=("Both propagation directions are evaluated and the stronger compressive evidence is retained.",),
        ),
        WaveEvidenceFormulaHook(
            family="release_like",
            expression_label="w_rho*Phi(-/+Delta ln rho) + w_P*Phi(-/+Delta ln P_tot) + w_u*Phi(+du/dx) + w_T*Phi(-/+Delta ln Ti) - shocklike_penalty",
            configurable_terms=("w_rho", "w_P", "w_u", "w_T", "shocklike_penalty", "Phi"),
            required_fields=("density", "velocity"),
            optional_fields=(pressure_field, "temperature_i"),
            notes=("Both propagation directions are evaluated and the stronger release evidence is retained.",),
        ),
        WaveEvidenceFormulaHook(
            family="contact_like",
            expression_label="w_m*1(material jump) + w_rho*abs(Delta ln rho) - w_P*abs(Delta ln P_tot) - w_u*abs(du/dx)",
            configurable_terms=("w_m", "w_rho", "w_P", "w_u"),
            required_fields=("density", "velocity"),
            optional_fields=(pressure_field, "material_jump"),
            notes=("Contact evidence prefers material or density jumps without strong compressive support.",),
        ),
    )


def _build_detector_score(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    geometry: AnalysisGeometryMetadata,
    parameters: "DerivedAnalysisParameters",
    initial_density: np.ndarray,
    sign: float,
    propagation_direction: str,
    expected_velocity_sign: float,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> np.ndarray:
    """Build the legacy primary-shock detector matrix once for a logical request."""

    n_snapshots = int(dataset.time_s.size)
    n_interfaces = max(0, int(dataset.summary["n_zones"]) - 1)
    detector_score = np.full((n_snapshots, n_interfaces), 0.0, dtype=np.float64)
    if n_snapshots == 0 or n_interfaces <= 0:
        return detector_score
    mask_matrix, _selection_keys = shared_time_series_selection_state(
        dataset,
        context,
        geometry=geometry,
        weighting_mode=parameters.weighting_mode,
        reuse_viewer_subset=parameters.reuse_viewer_subset,
        derived_region_ids=parameters.derived_region_ids,
        derived_material_ids=parameters.derived_material_ids,
        exclude_entry_region=parameters.exclude_entry_region,
        exclude_low_density=parameters.exclude_low_density,
        min_density_g_cm3=parameters.min_density_g_cm3,
        exclude_opposite_velocity=parameters.exclude_opposite_velocity,
        zone_index_lower=parameters.zone_index_lower,
        zone_index_upper=parameters.zone_index_upper,
        analysis_cache=analysis_cache,
        progress_check=progress_check,
    )
    for snapshot_index in range(n_snapshots):
        if progress_check is not None and (snapshot_index % 8 == 0):
            progress_check()
        mask = np.asarray(mask_matrix[int(snapshot_index)], dtype=bool)
        density = _smooth_profile(
            np.asarray(dataset.density_g_cm3[snapshot_index], dtype=np.float64) / initial_density,
            passes=2,
        )
        velocity = np.asarray(dataset.velocity_cm_s[snapshot_index], dtype=np.float64)
        directional_gradient = sign * np.diff(density)
        compressed_side = density[1:] if propagation_direction == "high_to_low" else density[:-1]
        interface_velocity = 0.5 * (velocity[:-1] + velocity[1:])
        interface_mask = np.asarray(mask[:-1] & mask[1:], dtype=bool)

        score = np.maximum(directional_gradient, 0.0)
        score *= np.clip((compressed_side - 1.0) / 0.08, 0.0, 1.0)
        score *= np.where(np.abs(interface_velocity) >= 2.0e4, 1.0, 0.35)
        score *= np.where(expected_velocity_sign * interface_velocity >= 0.0, 1.0, 0.35)
        detector_score[snapshot_index] = np.where(interface_mask & np.isfinite(score), score, 0.0)
    return detector_score


def _build_wave_evidence_payload(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    geometry: AnalysisGeometryMetadata,
    parameters: "DerivedAnalysisParameters",
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> _EvidencePayload:
    formula_hooks = shock_tracking_formula_hooks(dataset)
    selection_request_key = selection_request_cache_key(
        context,
        reuse_viewer_subset=parameters.reuse_viewer_subset,
        derived_region_ids=parameters.derived_region_ids,
        derived_material_ids=parameters.derived_material_ids,
        exclude_entry_region=parameters.exclude_entry_region,
        exclude_low_density=parameters.exclude_low_density,
        min_density_g_cm3=parameters.min_density_g_cm3,
        exclude_opposite_velocity=parameters.exclude_opposite_velocity,
        zone_index_lower=parameters.zone_index_lower,
        zone_index_upper=parameters.zone_index_upper,
        weighting_mode=parameters.weighting_mode,
    )
    cache_key = (
        "wave.evidence",
        geometry.observation_side,
        geometry.observation_boundary,
        round(float(geometry.line_of_sight_angle_deg), 12),
        round(float(geometry.impact_parameter_cm), 12),
        geometry.profile_coordinate_mode,
        geometry.propagation_direction,
        selection_request_key,
    )

    def _build() -> _EvidencePayload:
        n_snapshots = int(dataset.time_s.size)
        n_interfaces = max(0, int(dataset.summary["n_zones"]) - 1)
        shock_score = np.zeros((n_snapshots, n_interfaces), dtype=np.float64)
        release_score = np.zeros((n_snapshots, n_interfaces), dtype=np.float64)
        contact_score = np.zeros((n_snapshots, n_interfaces), dtype=np.float64)
        shock_dir = np.zeros((n_snapshots, n_interfaces), dtype=np.float64)
        release_dir = np.zeros((n_snapshots, n_interfaces), dtype=np.float64)
        contact_dir = np.zeros((n_snapshots, n_interfaces), dtype=np.float64)

        mask_matrix, _selection_keys = shared_time_series_selection_state(
            dataset,
            context,
            geometry=geometry,
            weighting_mode=parameters.weighting_mode,
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        for snapshot_index in range(n_snapshots):
            if progress_check is not None and (snapshot_index % 8 == 0):
                progress_check()
            mask = np.asarray(mask_matrix[int(snapshot_index)], dtype=bool)
            interface_mask = mask[:-1] & mask[1:]
            if not np.any(interface_mask):
                continue

            rho = _smooth_profile(np.asarray(dataset.density_g_cm3[snapshot_index], dtype=np.float64), passes=1)
            u = _smooth_profile(np.asarray(dataset.velocity_cm_s[snapshot_index], dtype=np.float64), passes=1)
            te = _smooth_profile(np.asarray(dataset.temperature_e_ev[snapshot_index], dtype=np.float64), passes=1)
            ti = _smooth_profile(np.asarray(dataset.temperature_i_ev[snapshot_index], dtype=np.float64), passes=1)
            p_total = _total_pressure_snapshot(dataset, snapshot_index)
            q_art = (
                None
                if dataset.artificial_viscosity_j_cm3 is None
                else _smooth_profile(np.asarray(dataset.artificial_viscosity_j_cm3[snapshot_index], dtype=np.float64), passes=1)
            )

            dlnrho = np.diff(_safe_positive_log(rho))
            spacing = _interface_spacing_cm(dataset, snapshot_index)
            du_dx = np.divide(np.diff(u), spacing, out=np.zeros_like(dlnrho), where=np.isfinite(spacing) & (spacing > 0.0))
            dlnte = np.diff(_safe_positive_log(te))
            dlnti = np.diff(_safe_positive_log(ti))
            dlnp = np.zeros_like(dlnrho)
            if p_total is not None:
                dlnp = np.diff(_safe_positive_log(p_total))
            q_ratio = np.zeros_like(dlnrho)
            if q_art is not None and p_total is not None:
                p_interface = 0.5 * (p_total[:-1] + p_total[1:])
                q_ratio = np.divide(
                    0.5 * (q_art[:-1] + q_art[1:]),
                    np.maximum(p_interface, 1.0e-30),
                    out=np.zeros_like(dlnrho),
                    where=np.isfinite(p_interface),
                )

            region_jump = np.diff(np.asarray(dataset.zone_region_id, dtype=np.int32)) != 0
            material_jump = np.diff(np.asarray(dataset.zone_material_index, dtype=np.int32)) != 0
            jump_indicator = np.where(region_jump | material_jump, 1.0, 0.0)

            rho_scale = _robust_scale(dlnrho[interface_mask], 0.05)
            p_scale = _robust_scale(dlnp[interface_mask], 0.05)
            u_scale = _robust_scale(du_dx[interface_mask], 5.0e10)
            q_scale = _robust_scale(q_ratio[interface_mask], 0.05)
            te_scale = _robust_scale(dlnte[interface_mask], 0.05)
            ti_scale = _robust_scale(dlnti[interface_mask], 0.05)

            shock_hl = 0.33 * _positive_activation(+dlnrho, rho_scale)
            shock_lh = 0.33 * _positive_activation(-dlnrho, rho_scale)
            if p_total is not None:
                shock_hl += 0.25 * _positive_activation(+dlnp, p_scale)
                shock_lh += 0.25 * _positive_activation(-dlnp, p_scale)
            shock_hl += 0.20 * _positive_activation(-du_dx, u_scale)
            shock_lh += 0.20 * _positive_activation(-du_dx, u_scale)
            if q_art is not None and p_total is not None:
                shock_hl += 0.12 * _positive_activation(q_ratio, q_scale)
                shock_lh += 0.12 * _positive_activation(q_ratio, q_scale)
            shock_hl += 0.10 * _positive_activation(+dlnte, te_scale)
            shock_lh += 0.10 * _positive_activation(-dlnte, te_scale)
            artifact_penalty = 0.18 * _positive_activation(np.abs(dlnrho) - np.maximum(np.abs(dlnp), 0.5 * q_ratio), rho_scale)
            shock_combined = np.maximum(shock_hl, shock_lh) - artifact_penalty

            release_hl = 0.38 * _positive_activation(-dlnrho, rho_scale)
            release_lh = 0.38 * _positive_activation(+dlnrho, rho_scale)
            if p_total is not None:
                release_hl += 0.28 * _positive_activation(-dlnp, p_scale)
                release_lh += 0.28 * _positive_activation(+dlnp, p_scale)
            release_hl += 0.24 * _positive_activation(+du_dx, u_scale)
            release_lh += 0.24 * _positive_activation(+du_dx, u_scale)
            release_hl += 0.10 * _positive_activation(-dlnti, ti_scale)
            release_lh += 0.10 * _positive_activation(+dlnti, ti_scale)
            release_combined = np.maximum(release_hl, release_lh) - 0.35 * np.maximum(shock_hl, shock_lh)

            contact_component = 0.65 * jump_indicator
            contact_component += 0.28 * np.clip(np.abs(dlnrho) / max(rho_scale, 1.0e-12), 0.0, 2.5)
            if p_total is not None:
                contact_component -= 0.18 * np.clip(np.abs(dlnp) / max(p_scale, 1.0e-12), 0.0, 2.5)
            contact_component -= 0.12 * np.clip(np.abs(du_dx) / max(u_scale, 1.0e-12), 0.0, 2.5)

            shock_score[snapshot_index] = np.where(interface_mask, np.maximum(shock_combined, 0.0), 0.0)
            release_score[snapshot_index] = np.where(interface_mask, np.maximum(release_combined, 0.0), 0.0)
            contact_score[snapshot_index] = np.where(interface_mask, np.maximum(contact_component, 0.0), 0.0)
            shock_dir[snapshot_index] = np.where(interface_mask, np.where(shock_lh >= shock_hl, 1.0, -1.0), 0.0)
            release_dir[snapshot_index] = np.where(interface_mask, np.where(release_lh >= release_hl, 1.0, -1.0), 0.0)
            contact_dir[snapshot_index] = np.where(interface_mask, np.sign(0.5 * (u[:-1] + u[1:])), 0.0)

        return _EvidencePayload(
            formula_hooks=formula_hooks,
            interface_position_grid=_build_interface_position_grid(dataset),
            family_score={
                "shock_like": shock_score,
                "release_like": release_score,
                "contact_like": contact_score,
            },
            family_direction={
                "shock_like": shock_dir,
                "release_like": release_dir,
                "contact_like": contact_dir,
            },
            candidate_thresholds={
                "shock_like": _family_threshold(shock_score, quantile=_WAVE_CONFIG.shock_threshold_quantile, minimum=_WAVE_CONFIG.minimum_score),
                "release_like": _family_threshold(release_score, quantile=_WAVE_CONFIG.release_threshold_quantile, minimum=_WAVE_CONFIG.minimum_score),
                "contact_like": _family_threshold(contact_score, quantile=_WAVE_CONFIG.contact_threshold_quantile, minimum=max(0.08, 0.75 * _WAVE_CONFIG.minimum_score)),
            },
        )

    return cached_time_series_payload(cache_key, analysis_cache=analysis_cache, builder=_build)


def _candidate_window_positions(dataset: DerivedRunData, snapshot_index: int, interface_index: int) -> tuple[np.ndarray, np.ndarray]:
    n_zones = int(dataset.summary["n_zones"])
    zone_lo = max(0, interface_index - 2)
    zone_hi = min(n_zones - 1, interface_index + 3)
    zone_slice = slice(zone_lo, zone_hi + 1)
    centers = _zone_centers_cm(dataset, snapshot_index)[zone_slice]
    return centers, _zone_edges_cm(dataset, snapshot_index)[zone_lo : zone_hi + 2]


def _least_squares_tanh_fit(
    x: np.ndarray,
    values: tuple[np.ndarray, ...],
    initial_xf: float,
    initial_width: float,
) -> _LocalizedFit | None:
    valid_fields = [np.asarray(field, dtype=np.float64) for field in values if field.size == x.size and np.count_nonzero(np.isfinite(field)) >= 3]
    if x.size < 4 or not valid_fields:
        return None
    finite_dx = np.diff(np.asarray(x, dtype=np.float64))
    positive_dx = finite_dx[np.isfinite(finite_dx) & (finite_dx > 0.0)]
    cell_width = float(np.nanmedian(positive_dx)) if positive_dx.size else max(float(initial_width), 1.0e-8)
    width_candidates = np.linspace(max(cell_width * 0.75, 1.0e-8), max(cell_width * 4.0, cell_width), 4)
    x_candidates = np.linspace(float(initial_xf - cell_width), float(initial_xf + cell_width), 5)
    best_error = math.inf
    best_xf = float(initial_xf)
    best_width = max(float(initial_width), cell_width)
    for x_f in x_candidates:
        for width in width_candidates:
            basis = np.column_stack((np.ones(x.size, dtype=np.float64), np.tanh((x - float(x_f)) / max(float(width), 1.0e-12))))
            total_error = 0.0
            for field in valid_fields:
                finite = np.isfinite(field) & np.all(np.isfinite(basis), axis=1)
                if np.count_nonzero(finite) < 3:
                    continue
                solution, *_ = np.linalg.lstsq(basis[finite], field[finite], rcond=None)
                residual = field[finite] - basis[finite] @ solution
                amplitude = max(float(np.nanstd(field[finite])), 1.0e-12)
                total_error += float(np.sqrt(np.mean((residual / amplitude) ** 2)))
            if total_error < best_error:
                best_error = total_error
                best_xf = float(x_f)
                best_width = float(width)
    if not np.isfinite(best_error):
        return None
    fit_quality = float(1.0 / (1.0 + best_error))
    confidence = float(np.clip(0.3 + 0.7 * fit_quality, 0.0, 1.0))
    return _LocalizedFit(
        position_cm=best_xf,
        width_cm=best_width,
        fit_quality=fit_quality,
        confidence=confidence,
        fit_seed=WaveFrontFitSeed(
            model_name="local-joint-tanh",
            front_position_cm=best_xf,
            effective_width_cm=best_width,
            fit_quality=fit_quality,
            confidence=confidence,
            fitted_fields=tuple(f"f{index}" for index in range(len(valid_fields))),
            notes=("Joint tanh fit over the local candidate window.",),
        ),
    )


def _localize_candidate(
    dataset: DerivedRunData,
    snapshot_index: int,
    interface_index: int,
    family_score_row: np.ndarray,
    total_pressure: np.ndarray | None,
    *,
    use_tanh_fit: bool,
) -> _LocalizedFit:
    centers, edges = _candidate_window_positions(dataset, snapshot_index, interface_index)
    interface_positions = _build_interface_position_grid(dataset)[snapshot_index]
    initial_xf = _position_from_interface_index(interface_positions, float(interface_index))
    widths = np.diff(edges)
    positive_widths = widths[np.isfinite(widths) & (widths > 0.0)]
    initial_width = float(np.nanmedian(positive_widths)) if positive_widths.size else 1.0e-8
    zone_lo = max(0, interface_index - 2)
    zone_hi = min(int(dataset.summary["n_zones"]) - 1, interface_index + 3)
    zone_slice = slice(zone_lo, zone_hi + 1)
    fit_values: list[np.ndarray] = [_safe_positive_log(dataset.density_g_cm3[snapshot_index, zone_slice])]
    if total_pressure is not None:
        fit_values.append(_safe_positive_log(total_pressure[zone_slice]))
    fit_values.append(np.asarray(dataset.velocity_cm_s[snapshot_index, zone_slice], dtype=np.float64))
    if use_tanh_fit:
        fit = _least_squares_tanh_fit(np.asarray(centers, dtype=np.float64), tuple(fit_values), initial_xf, initial_width)
        if fit is not None:
            return fit
    row = np.asarray(family_score_row, dtype=np.float64)
    left = max(0, interface_index - 1)
    right = min(row.size - 1, interface_index + 1)
    local_idx = np.arange(left, right + 1, dtype=np.float64)
    local_score = np.maximum(row[left : right + 1], 0.0)
    if np.any(local_score > 0.0):
        centroid_index = float(np.sum(local_idx * local_score) / np.sum(local_score))
        centroid = _position_from_interface_index(interface_positions, centroid_index)
        width = max(float(np.sqrt(np.sum(((local_idx - centroid_index) ** 2) * local_score) / np.sum(local_score))) * max(initial_width, 1.0e-8), initial_width)
    else:
        centroid = initial_xf
        width = initial_width
    return _LocalizedFit(
        position_cm=float(centroid),
        width_cm=float(width),
        fit_quality=0.35,
        confidence=0.3,
        fit_seed=WaveFrontFitSeed(
            model_name="weighted-centroid",
            front_position_cm=float(centroid),
            effective_width_cm=float(width),
            fit_quality=0.35,
            confidence=0.3,
            fitted_fields=(),
            notes=("Fallback localization used because the local tanh fit was underconstrained.",),
        ),
    )


def _collapse_ambiguous_candidates(seeds: list[_FrameCandidateSeed]) -> list[_FrameCandidateSeed]:
    if len(seeds) < 2:
        return seeds
    consumed = [False] * len(seeds)
    collapsed: list[_FrameCandidateSeed] = []
    for index, seed in enumerate(seeds):
        if consumed[index]:
            continue
        merged = [seed]
        consumed[index] = True
        for other_index in range(index + 1, len(seeds)):
            other = seeds[other_index]
            if consumed[other_index]:
                continue
            close = abs(other.interface_index - seed.interface_index) <= _WAVE_CONFIG.merge_distance_cells
            score_ratio = max(seed.family_score, other.family_score) / max(min(seed.family_score, other.family_score), 1.0e-12)
            if other.snapshot_index == seed.snapshot_index and close and score_ratio <= _WAVE_CONFIG.merge_score_ratio:
                merged.append(other)
                consumed[other_index] = True
        if len(merged) == 1:
            collapsed.append(seed)
            continue
        best = max(merged, key=lambda item: item.family_score)
        collapsed.append(
            _FrameCandidateSeed(
                snapshot_index=best.snapshot_index,
                family="merged_unresolved",
                family_score=float(np.mean([item.family_score for item in merged])),
                interface_index=float(np.mean([item.interface_index for item in merged])),
                direction=best.direction,
                direction_sign=best.direction_sign,
                candidate_type="merged_unresolved_front",
                ambiguous=True,
                upstream_state=best.upstream_state,
                downstream_state=best.downstream_state,
                fit=best.fit,
                notes=("Merged unresolved structure from overlapping evidence families.",),
            )
        )
    return collapsed


def _extract_frame_candidates(dataset: DerivedRunData, snapshot_index: int, payload: _EvidencePayload, *, primary_direction: str) -> list[_FrameCandidateSeed]:
    total_pressure = _total_pressure_snapshot(dataset, snapshot_index)
    seeds: list[_FrameCandidateSeed] = []
    n_zones = int(dataset.summary["n_zones"])
    for family in ("shock_like", "release_like", "contact_like"):
        score_row = np.asarray(payload.family_score[family][snapshot_index], dtype=np.float64)
        direction_row = np.asarray(payload.family_direction[family][snapshot_index], dtype=np.float64)
        maxima = _local_maxima_indices(score_row, float(payload.candidate_thresholds[family]))
        row_peak = float(np.nanmax(score_row)) if np.any(np.isfinite(score_row)) else 0.0
        ranked = sorted((int(index) for index in maxima), key=lambda idx: float(score_row[idx]), reverse=True)
        for family_rank, interface_index in enumerate(ranked[: _WAVE_CONFIG.max_candidates_per_family]):
            score_value = float(score_row[interface_index])
            if score_value < max(float(payload.candidate_thresholds[family]), 0.55 * row_peak):
                continue
            direction_sign = float(direction_row[interface_index]) if np.isfinite(direction_row[interface_index]) else 0.0
            direction = None if (family == "contact_like" and abs(direction_sign) < 0.5) else ("low_to_high" if direction_sign >= 0.0 else "high_to_low")
            upstream_zone, downstream_zone = _upstream_downstream_zone_indices(interface_index, direction, n_zones)
            fit = _localize_candidate(
                dataset,
                snapshot_index,
                interface_index,
                score_row,
                total_pressure,
                use_tanh_fit=(family != "contact_like" and family_rank == 0 and score_value >= 1.25 * float(payload.candidate_thresholds[family])),
            )
            confidence = float(np.clip(0.55 * min(score_value / max(float(payload.candidate_thresholds[family]), 1.0e-12), 1.5) / 1.5 + 0.45 * fit.confidence, 0.0, 1.0))
            interface_crossed = bool(np.any(np.diff(dataset.zone_region_id[: interface_index + 2]) != 0)) if interface_index + 1 < dataset.zone_region_id.size else False
            near_rear = bool((direction == "high_to_low" and interface_index >= max(0, score_row.size - 3)) or (direction == "low_to_high" and interface_index <= 2))
            seeds.append(
                _FrameCandidateSeed(
                    snapshot_index=snapshot_index,
                    family=family,
                    family_score=score_value,
                    interface_index=float(interface_index),
                    direction=direction,
                    direction_sign=(None if direction is None else float(1.0 if direction == "low_to_high" else -1.0)),
                    candidate_type=_branch_type_from_candidate(
                        family=family,
                        direction=direction,
                        primary_direction=primary_direction,
                        interface_crossed=interface_crossed,
                        near_rear_boundary=near_rear,
                        ambiguous=False,
                    ),
                    ambiguous=False,
                    upstream_state=_state_summary(dataset, snapshot_index, upstream_zone),
                    downstream_state=_state_summary(dataset, snapshot_index, downstream_zone),
                    fit=_LocalizedFit(
                        position_cm=fit.position_cm,
                        width_cm=fit.width_cm,
                        fit_quality=fit.fit_quality,
                        confidence=confidence,
                        fit_seed=WaveFrontFitSeed(
                            model_name=fit.fit_seed.model_name,
                            front_position_cm=fit.fit_seed.front_position_cm,
                            effective_width_cm=fit.fit_seed.effective_width_cm,
                            fit_quality=fit.fit_quality,
                            confidence=confidence,
                            fitted_fields=fit.fit_seed.fitted_fields,
                            notes=fit.fit_seed.notes,
                        ),
                    ),
                )
            )
    seeds = _collapse_ambiguous_candidates(sorted(seeds, key=lambda item: item.family_score, reverse=True))
    seeds.sort(key=lambda item: item.family_score, reverse=True)
    return seeds[: _WAVE_CONFIG.max_candidates_per_frame]


def _count_interfaces_crossed(event_scaffold: _ShockEventScaffold, left_position_cm: float, right_position_cm: float, snapshot_index: int) -> int:
    if not (np.isfinite(left_position_cm) and np.isfinite(right_position_cm)):
        return 0
    positions = event_scaffold.interface_position_grid[min(max(snapshot_index, 0), event_scaffold.interface_position_grid.shape[0] - 1)]
    low = min(float(left_position_cm), float(right_position_cm))
    high = max(float(left_position_cm), float(right_position_cm))
    return int(np.count_nonzero(np.isfinite(positions) & (positions > low) & (positions <= high)))


def _candidate_velocity_proxy(candidate: _FrameCandidateSeed) -> float | None:
    velocities = [
        value
        for value in (
            None if candidate.upstream_state is None else candidate.upstream_state.velocity_cm_s,
            None if candidate.downstream_state is None else candidate.downstream_state.velocity_cm_s,
        )
        if value is not None and np.isfinite(value)
    ]
    if not velocities:
        return None
    return float(np.mean(velocities))


def _association_cost(branch: _ActiveBranch, candidate: _FrameCandidateSeed, *, time_s: np.ndarray, event_scaffold: _ShockEventScaffold) -> float:
    previous = branch.candidates[-1]
    dt = float(time_s[candidate.snapshot_index] - time_s[previous.snapshot_index])
    previous_velocity = _candidate_velocity_proxy(previous)
    predicted = float(previous.position_cm if previous_velocity is None or not np.isfinite(previous_velocity) else previous.position_cm + previous_velocity * dt)
    position_scale = max(_WAVE_CONFIG.association_position_scale_cm, 8.0 * max(previous.width_cm, candidate.width_cm, 1.0e-8))
    cost_x = _WAVE_CONFIG.lambda_x * abs(float(candidate.position_cm) - predicted) / position_scale
    candidate_velocity = _candidate_velocity_proxy(candidate)
    if previous_velocity is None or candidate_velocity is None or not np.isfinite(previous_velocity) or not np.isfinite(candidate_velocity):
        cost_v = 0.15
    else:
        cost_v = _WAVE_CONFIG.lambda_v * abs(candidate_velocity - previous_velocity) / _WAVE_CONFIG.association_velocity_scale_cm_s
    cost_s = _WAVE_CONFIG.lambda_s * abs(float(candidate.family_score) - float(previous.family_score)) / _WAVE_CONFIG.association_score_scale
    type_penalty = 0.0
    if branch.family != candidate.family and "merged_unresolved" not in {branch.family, candidate.family}:
        type_penalty = _WAVE_CONFIG.lambda_t
    illegal_penalty = _WAVE_CONFIG.lambda_i * max(0, _count_interfaces_crossed(event_scaffold, previous.position_cm, candidate.position_cm, candidate.snapshot_index) - 1)
    if previous.direction is not None and candidate.direction is not None and previous.direction != candidate.direction and branch.family == candidate.family:
        illegal_penalty += 0.35
    return float(cost_x + cost_v + cost_s + type_penalty + illegal_penalty)


def _optimal_assignment(cost_matrix: np.ndarray, *, miss_cost: float, birth_cost: float, max_cost: float) -> tuple[list[tuple[int, int]], set[int]]:
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0:
        return [], set(range(n_cols))

    @lru_cache(maxsize=None)
    def _solve(row_index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if row_index >= n_rows:
            unused = n_cols - int(bin(used_mask).count("1"))
            return float(unused) * float(birth_cost), ()
        base_tail, base_assignments = _solve(row_index + 1, used_mask)
        best_cost = float(miss_cost) + base_tail
        best_assignments = base_assignments
        for col_index in range(n_cols):
            bit = 1 << col_index
            if used_mask & bit:
                continue
            match_cost = float(cost_matrix[row_index, col_index])
            if not np.isfinite(match_cost) or match_cost > max_cost:
                continue
            tail_cost, tail_assignments = _solve(row_index + 1, used_mask | bit)
            total_cost = match_cost + tail_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_assignments = ((row_index, col_index),) + tail_assignments
        return best_cost, best_assignments

    _total, assignments = _solve(0, 0)
    matched_cols = {col for _row, col in assignments}
    return list(assignments), set(range(n_cols)) - matched_cols


def _build_wave_branch_summaries(
    dataset: DerivedRunData,
    branches: list[_ActiveBranch],
    *,
    primary_direction: str,
    event_scaffold: _ShockEventScaffold,
) -> tuple[WaveBranchSummary, ...]:
    summaries: list[WaveBranchSummary] = []
    for branch in branches:
        candidates = branch.candidates
        snapshot_indices = np.asarray([candidate.snapshot_index for candidate in candidates], dtype=np.int32)
        interface_index = np.asarray([candidate.interface_index for candidate in candidates], dtype=np.float64)
        position_cm = np.asarray([candidate.position_cm for candidate in candidates], dtype=np.float64)
        score = np.asarray([candidate.family_score for candidate in candidates], dtype=np.float64)
        width_cm = np.asarray([candidate.width_cm for candidate in candidates], dtype=np.float64)
        if position_cm.size >= 2:
            velocity_cm_s = np.gradient(position_cm, dataset.time_s[snapshot_indices.astype(int)])
        else:
            velocity_cm_s = np.full(position_cm.shape, np.nan, dtype=np.float64)
        ambiguous = bool(any(candidate.ambiguous for candidate in candidates))
        confidence = float(np.nanmean([candidate.confidence for candidate in candidates])) if candidates else None
        sample_count = int(snapshot_indices.size)
        support_class = _classify_branch_support(sample_count)
        duration_s = None
        continuity_fraction = None
        if sample_count >= 2:
            start_index = int(snapshot_indices[0])
            end_index = int(snapshot_indices[-1])
            duration_s = float(dataset.time_s[end_index] - dataset.time_s[start_index])
            frame_span = max(1, end_index - start_index + 1)
            continuity_fraction = float(sample_count / frame_span)
        positive_score = np.clip(np.asarray(score, dtype=np.float64), 0.0, None)
        integrated_score = float(np.nansum(positive_score)) if positive_score.size else 0.0
        finite_positions = np.asarray(position_cm[np.isfinite(position_cm)], dtype=np.float64)
        position_span_cm = float(np.nanmax(finite_positions) - np.nanmin(finite_positions)) if finite_positions.size >= 2 else 0.0
        finite_widths = np.asarray(width_cm[np.isfinite(width_cm) & (width_cm > 0.0)], dtype=np.float64)
        span_cells = position_span_cm / max(float(np.nanmedian(finite_widths)) if finite_widths.size else 1.0e-8, 1.0e-8)
        significance = integrated_score * (1.0 + 0.45 * math.log1p(sample_count))
        if duration_s is not None and math.isfinite(duration_s):
            significance *= 1.0 + 0.25 * math.log1p(max(duration_s, 0.0) / max(float(np.nanmedian(np.diff(dataset.time_s))) if dataset.time_s.size >= 2 else 1.0e-12, 1.0e-12))
        if continuity_fraction is not None and math.isfinite(continuity_fraction):
            significance *= 0.55 + 0.45 * float(np.clip(continuity_fraction, 0.0, 1.0))
        significance *= 1.0 + 0.06 * min(max(span_cells, 0.0), 10.0)
        if support_class == "provisional":
            significance *= 0.2
        elif support_class == "short_weak":
            significance *= 0.65
        if ambiguous:
            significance *= 0.9
        notes: list[str] = []
        if ambiguous:
            notes.append("Branch contains unresolved overlapping candidates.")
        if branch.family == "shock_like" and candidates[-1].direction is not None and candidates[-1].direction != primary_direction:
            notes.append("Shock-like branch propagates opposite the primary laser-entry direction.")
        if support_class == "provisional":
            notes.append("Support classification: provisional candidate (fewer than 3 supporting samples).")
        elif support_class == "short_weak":
            notes.append(
                f"Support classification: short/weak branch ({sample_count} samples; tracked threshold is {_WAVE_CONFIG.tracked_branch_min_samples})."
            )
        summaries.append(
            WaveBranchSummary(
                branch_id=branch.branch_id,
                family=branch.family,
                branch_type=branch.branch_type,
                snapshot_indices=snapshot_indices,
                interface_index=interface_index,
                position_cm=position_cm,
                velocity_cm_s=np.asarray(velocity_cm_s, dtype=np.float64),
                score=score,
                width_cm=width_cm,
                confidence=confidence,
                ambiguous=ambiguous,
                propagation_direction=candidates[-1].direction,
                breakout_time_s=None,
                support_class=support_class,
                sample_count=sample_count,
                duration_s=duration_s,
                integrated_score=integrated_score,
                position_span_cm=position_span_cm,
                significance=float(significance),
                continuity_fraction=continuity_fraction,
                upstream_state=candidates[-1].upstream_state,
                downstream_state=candidates[-1].downstream_state,
                primary=branch.primary,
                notes=tuple(notes),
            )
        )
    return tuple(summaries)


def _classify_branch_support(sample_count: int) -> str:
    if int(sample_count) <= _WAVE_CONFIG.provisional_sample_limit:
        return "provisional"
    if int(sample_count) < _WAVE_CONFIG.tracked_branch_min_samples:
        return "short_weak"
    return "tracked"


def _branch_support_priority(branch: WaveBranchSummary) -> int:
    order = {"tracked": 0, "short_weak": 1, "provisional": 2}
    return int(order.get(str(branch.support_class), 3))


def _branch_family_priority(branch: WaveBranchSummary) -> tuple[int, float]:
    if branch.branch_type in {"compressive_shock", "transmitted_shock"}:
        return (0, -float(np.nanmax(branch.score)))
    if branch.branch_type == "reflected_shock":
        return (1, -float(np.nanmax(branch.score)))
    if branch.branch_type in {"release_rarefaction", "rear_rarefaction"}:
        return (2, -float(np.nanmax(branch.score)))
    if branch.branch_type == "contact_transition":
        return (3, -float(np.nanmax(branch.score)))
    return (4, -float(np.nanmax(branch.score)))


def _branch_sort_key(branch: WaveBranchSummary) -> tuple[int, float, int, float, str]:
    family_rank, family_peak = _branch_family_priority(branch)
    significance = 0.0 if branch.significance is None or not np.isfinite(float(branch.significance)) else float(branch.significance)
    return (
        _branch_support_priority(branch),
        -significance,
        family_rank,
        family_peak,
        str(branch.branch_id),
    )


def track_wave_branches(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    parameters: "DerivedAnalysisParameters | None" = None,
    geometry: AnalysisGeometryMetadata | None = None,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> WaveTrackingResult:
    """Track multiple 1D wave branches using multi-cue evidence and assignment."""

    if parameters is None:
        from helios.services.derived.analysis import DerivedAnalysisParameters as _DerivedAnalysisParameters

        parameters = _DerivedAnalysisParameters()
    if geometry is None:
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )

    event_scaffold = _ShockEventScaffold(
        time_s=np.asarray(dataset.time_s, dtype=np.float64),
        interface_position_grid=_build_interface_position_grid(dataset),
        interface_boundaries=tuple(region_interface_boundaries(dataset.regions)),
    )
    payload = _build_wave_evidence_payload(
        dataset,
        context,
        geometry=geometry,
        parameters=parameters,
        analysis_cache=analysis_cache,
        progress_check=progress_check,
    )
    warnings: list[DerivedWarning] = []
    if dataset.pressure_total_j_cm3 is None and not dataset.field_capabilities.pressure_components_available:
        warnings.append(DerivedWarning("wave_tracking", "Total pressure was unavailable, so wave evidence used density/velocity/temperature cues only.", severity="info"))
    primary_direction, _expected_velocity_sign = infer_primary_propagation_direction(dataset, context)

    active: list[_ActiveBranch] = []
    finished: list[_ActiveBranch] = []
    next_branch_index = 1
    for snapshot_index in range(int(dataset.time_s.size)):
        if progress_check is not None and (snapshot_index % 4 == 0):
            progress_check()
        frame_candidates = _extract_frame_candidates(dataset, snapshot_index, payload, primary_direction=primary_direction)
        if not active:
            for seed in frame_candidates:
                active.append(_ActiveBranch(branch_id=f"branch-{next_branch_index}", family=seed.family, branch_type=seed.candidate_type, primary=False, candidates=[seed]))
                next_branch_index += 1
            continue
        cost_matrix = np.full((len(active), len(frame_candidates)), np.inf, dtype=np.float64)
        for row_index, branch in enumerate(active):
            for col_index, candidate in enumerate(frame_candidates):
                cost_matrix[row_index, col_index] = _association_cost(branch, candidate, time_s=np.asarray(dataset.time_s, dtype=np.float64), event_scaffold=event_scaffold)
        assignments, unmatched_candidates = _optimal_assignment(
            cost_matrix,
            miss_cost=_WAVE_CONFIG.association_miss_cost,
            birth_cost=_WAVE_CONFIG.association_birth_cost,
            max_cost=_WAVE_CONFIG.association_max_cost,
        )
        matched_rows = set()
        for row_index, col_index in assignments:
            matched_rows.add(row_index)
            branch = active[row_index]
            candidate = frame_candidates[col_index]
            branch.candidates.append(candidate)
            branch.family = candidate.family if candidate.family != "merged_unresolved" else branch.family
            branch.branch_type = candidate.candidate_type
            branch.missed_frames = 0
        survivors: list[_ActiveBranch] = []
        for row_index, branch in enumerate(active):
            if row_index in matched_rows:
                survivors.append(branch)
                continue
            branch.missed_frames += 1
            if branch.missed_frames <= 1:
                survivors.append(branch)
            else:
                finished.append(branch)
        active = survivors
        for col_index in sorted(unmatched_candidates):
            seed = frame_candidates[col_index]
            active.append(_ActiveBranch(branch_id=f"branch-{next_branch_index}", family=seed.family, branch_type=seed.candidate_type, primary=False, candidates=[seed]))
            next_branch_index += 1
    finished.extend(active)

    finished = [
        branch
        for branch in finished
        if len(branch.candidates) >= 2
        or max((candidate.family_score for candidate in branch.candidates), default=0.0) >= (2.0 * _WAVE_CONFIG.minimum_score)
    ]
    branch_summaries = list(_build_wave_branch_summaries(dataset, finished, primary_direction=primary_direction, event_scaffold=event_scaffold))
    provisional_branch_count = sum(1 for branch in branch_summaries if branch.support_class == "provisional")
    short_branch_count = sum(1 for branch in branch_summaries if branch.support_class == "short_weak")
    tracked_branch_count = sum(1 for branch in branch_summaries if branch.support_class == "tracked")
    branch_summaries.sort(key=_branch_sort_key)
    suppressed_branch_count = max(0, len(branch_summaries) - _WAVE_CONFIG.display_branch_limit)
    branch_summaries = branch_summaries[: _WAVE_CONFIG.display_branch_limit]
    if branch_summaries:
        primary = branch_summaries[0]
        branch_summaries[0] = WaveBranchSummary(
            branch_id=primary.branch_id,
            family=primary.family,
            branch_type=primary.branch_type,
            snapshot_indices=primary.snapshot_indices,
            interface_index=primary.interface_index,
            position_cm=primary.position_cm,
            velocity_cm_s=primary.velocity_cm_s,
            score=primary.score,
            width_cm=primary.width_cm,
            confidence=primary.confidence,
            ambiguous=primary.ambiguous,
            propagation_direction=primary.propagation_direction,
            breakout_time_s=primary.breakout_time_s,
            support_class=primary.support_class,
            sample_count=primary.sample_count,
            duration_s=primary.duration_s,
            integrated_score=primary.integrated_score,
            position_span_cm=primary.position_span_cm,
            significance=primary.significance,
            continuity_fraction=primary.continuity_fraction,
            upstream_state=primary.upstream_state,
            downstream_state=primary.downstream_state,
            primary=True,
            notes=primary.notes,
        )
    if len(branch_summaries) <= 1:
        warnings.append(DerivedWarning("wave_tracking", "The wave tracker found only one coherent branch in this run window; reflected or rear-release structure may simply be absent.", severity="info"))
    elif any(branch.branch_type == "merged_unresolved_front" for branch in branch_summaries):
        warnings.append(DerivedWarning("wave_tracking", "Some fronts remained unresolved and were labeled as merged/ambiguous structures.", severity="caution"))
    if suppressed_branch_count > 0:
        warnings.append(DerivedWarning("wave_tracking", f"Suppressed {suppressed_branch_count} weaker branch summaries from the default payload to keep the result compact.", severity="info"))
    if provisional_branch_count > 0:
        warnings.append(
            DerivedWarning(
                "wave_tracking",
                f"{provisional_branch_count} provisional detections had fewer than 3 supporting samples and are hidden from the default WaveFront view.",
                severity="info",
            )
        )

    candidates: list[WaveFrontCandidate] = []
    kept_branch_ids = {branch.branch_id for branch in branch_summaries}
    for branch in finished:
        if branch.branch_id not in kept_branch_ids:
            continue
        for seed in branch.candidates:
            candidates.append(
                WaveFrontCandidate(
                    snapshot_index=int(seed.snapshot_index),
                    family=str(seed.family),
                    candidate_type=str(seed.candidate_type),
                    coordinate_label=_coordinate_label(dataset),
                    interface_index=float(seed.interface_index),
                    position_cm=float(seed.position_cm),
                    width_cm=float(seed.width_cm),
                    score=float(seed.family_score),
                    propagation_direction=seed.direction,
                    direction_sign=seed.direction_sign,
                    fit_quality=float(seed.fit_quality),
                    confidence=float(seed.confidence),
                    ambiguous=bool(seed.ambiguous),
                    branch_hint=branch.branch_id,
                    upstream_state=seed.upstream_state,
                    downstream_state=seed.downstream_state,
                    fit_seed=seed.fit.fit_seed,
                    notes=seed.notes,
                )
            )

    evidence_maps = tuple(
        WaveEvidenceMap(
            family=family,
            coordinate_label=_coordinate_label(dataset),
            time_s=np.asarray(dataset.time_s, dtype=np.float64),
            interface_position_cm=np.asarray(payload.interface_position_grid, dtype=np.float64),
            score=np.asarray(payload.family_score[family], dtype=np.float64),
            formula_hook=next(hook for hook in payload.formula_hooks if hook.family == family),
            notes=("The strongest direction-resolved family score is stored per interface.",),
        )
        for family in ("shock_like", "release_like", "contact_like")
    )

    return WaveTrackingResult(
        method="multi-cue multi-branch wave tracker",
        coordinate_label=_coordinate_label(dataset),
        supported_formula_hooks=payload.formula_hooks,
        evidence_maps=evidence_maps,
        candidates=tuple(candidates),
        branches=tuple(branch_summaries),
        primary_branch_id=(None if not branch_summaries else branch_summaries[0].branch_id),
        candidate_count=len(candidates),
        tracked_branch_count=tracked_branch_count,
        short_branch_count=short_branch_count,
        provisional_branch_count=provisional_branch_count,
        suppressed_branch_count=suppressed_branch_count,
        compatibility_source=None,
        warnings=tuple(warnings),
    )


def _branch_time_axis(dataset: DerivedRunData, branch: WaveBranchSummary) -> np.ndarray:
    return np.asarray(dataset.time_s[np.asarray(branch.snapshot_indices, dtype=np.int32)], dtype=np.float64)


def _branch_series_at_time(
    dataset: DerivedRunData,
    branch: WaveBranchSummary,
    values: np.ndarray,
    target_time_s: float,
) -> float | None:
    branch_times = _branch_time_axis(dataset, branch)
    series = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(branch_times) & np.isfinite(series)
    if np.count_nonzero(finite) == 0:
        return None
    if np.count_nonzero(finite) == 1:
        return float(series[finite][0])
    return float(np.interp(float(target_time_s), branch_times[finite], series[finite]))


def _branch_position_distance_cells(branch: WaveBranchSummary, boundary_index: float) -> float:
    if branch.interface_index is None:
        return float("inf")
    indices = np.asarray(branch.interface_index, dtype=np.float64)
    finite = indices[np.isfinite(indices)]
    if finite.size == 0:
        return float("inf")
    return float(np.nanmin(np.abs(finite - float(boundary_index))))


def _local_time_window(dataset: DerivedRunData, snapshot_index: int, radius: int = 2) -> np.ndarray:
    start = max(0, int(snapshot_index) - int(radius))
    stop = min(int(dataset.time_s.size), int(snapshot_index) + int(radius) + 1)
    return np.arange(start, stop, dtype=np.int32)


def _safe_integral(time_values_s: np.ndarray, signal: np.ndarray) -> float | None:
    time_values = np.asarray(time_values_s, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(time_values) & np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return None
    return float(np.trapz(values[finite], time_values[finite]))


def _series_from_optional(field: np.ndarray | None, indices: np.ndarray, zone_index: int) -> np.ndarray | None:
    if field is None:
        return None
    array = np.asarray(field, dtype=np.float64)
    return np.asarray(array[np.asarray(indices, dtype=np.int32), int(zone_index)], dtype=np.float64)


def _pressure_series(dataset: DerivedRunData, indices: np.ndarray, zone_index: int) -> np.ndarray | None:
    series: list[np.ndarray] = []
    for snapshot_index in np.asarray(indices, dtype=np.int32):
        pressure = _total_pressure_snapshot(dataset, int(snapshot_index))
        if pressure is None:
            return None
        series.append(np.asarray(pressure, dtype=np.float64))
    stacked = np.asarray(series, dtype=np.float64)
    return np.asarray(stacked[:, int(zone_index)], dtype=np.float64)


def _density_series(dataset: DerivedRunData, indices: np.ndarray, zone_index: int) -> np.ndarray:
    return np.asarray(dataset.density_g_cm3[np.asarray(indices, dtype=np.int32), int(zone_index)], dtype=np.float64)


def _velocity_series(dataset: DerivedRunData, indices: np.ndarray, zone_index: int) -> np.ndarray:
    return np.asarray(dataset.velocity_cm_s[np.asarray(indices, dtype=np.int32), int(zone_index)], dtype=np.float64)


def _energy_flux_diagnostics(
    dataset: DerivedRunData,
    indices: np.ndarray,
    zone_index: int,
) -> tuple[dict[str, np.ndarray] | None, tuple[str, ...]]:
    pressure = _pressure_series(dataset, indices, zone_index)
    ion_energy = _series_from_optional(dataset.ion_energy_j_g, indices, zone_index)
    electron_energy = _series_from_optional(dataset.electron_energy_j_g, indices, zone_index)
    radiation_energy = _series_from_optional(dataset.radiation_energy_j_g, indices, zone_index)
    kinetic_energy = _series_from_optional(dataset.kinetic_energy_j_g, indices, zone_index)
    if pressure is None or ion_energy is None or electron_energy is None or radiation_energy is None or kinetic_energy is None:
        return None, ("Diagnostic energy-flux channels require P_tot, e_i, e_e, e_r, and e_k.",)
    density = _density_series(dataset, indices, zone_index)
    velocity = _velocity_series(dataset, indices, zone_index)
    e_int = ion_energy + electron_energy + radiation_energy
    e_tot = e_int + kinetic_energy
    e_tot_vol = density * e_tot
    return (
        {
            "pressure": pressure,
            "density": density,
            "velocity": velocity,
            "e_int": e_int,
            "e_k": kinetic_energy,
            "F_E": velocity * (e_tot_vol + pressure),
            "F_int": velocity * density * e_int,
            "F_k": velocity * density * kinetic_energy,
            "F_P": velocity * pressure,
        },
        (),
    )


def _pressure_impulse(pressure: np.ndarray | None, time_values_s: np.ndarray) -> float | None:
    if pressure is None:
        return None
    values = np.asarray(pressure, dtype=np.float64)
    if values.size == 0:
        return None
    baseline = float(values[0])
    return _safe_integral(time_values_s, np.maximum(values - baseline, 0.0))


def _compression_ratio(density: np.ndarray, *, allow_rarefaction: bool) -> float | None:
    values = np.asarray(density, dtype=np.float64)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return None
    baseline = float(finite[0])
    if baseline <= 0.0:
        return None
    if allow_rarefaction:
        return float(np.nanmin(finite) / baseline)
    return float(np.nanmax(finite) / baseline)


def _dominant_channel(channels: dict[str, float | None]) -> tuple[str | None, float | None, float | None, float | None]:
    finite = {
        key: float(value)
        for key, value in channels.items()
        if value is not None and np.isfinite(float(value)) and float(value) > 0.0
    }
    if not finite:
        return None, None, None, None
    total = float(sum(finite.values()))
    dominant_key = max(finite, key=finite.get)
    label = {
        "internal": "mostly internal",
        "kinetic": "mostly kinetic",
        "pressure_work": "mostly pressure-work",
    }.get(dominant_key, dominant_key)
    return (
        label,
        float(finite.get("internal", 0.0) / total),
        float(finite.get("kinetic", 0.0) / total),
        float(finite.get("pressure_work", 0.0) / total),
    )


def _impedance_preview(
    *,
    pressure_up: float | None,
    density_up: float | None,
    e_int_up: float | None,
    pressure_down: float | None,
    density_down: float | None,
    e_int_down: float | None,
) -> tuple[bool | None, float | None, float | None, float | None, float | None, tuple[str, ...]]:
    values = (pressure_up, density_up, e_int_up, pressure_down, density_down, e_int_down)
    if any(value is None or not np.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        return None, None, None, None, None, ("Impedance preview was unavailable because pressure or internal-energy baselines were incomplete.",)

    def _proxy_impedance(pressure: float, density: float, e_int: float) -> float:
        gamma_eff = 1.0 + float(pressure) / max(float(density) * float(e_int), 1.0e-30)
        sound_speed_sq = max(gamma_eff * float(pressure) / max(float(density), 1.0e-30), 0.0) * 1.0e7
        return float(density) * math.sqrt(sound_speed_sq)

    z_up = _proxy_impedance(float(pressure_up), float(density_up), float(e_int_up))
    z_down = _proxy_impedance(float(pressure_down), float(density_down), float(e_int_down))
    if not np.isfinite(z_up) or not np.isfinite(z_down) or (z_up + z_down) <= 0.0:
        return None, None, None, None, None, ("Impedance preview could not be evaluated from the local proxy state.",)
    reflection = ((z_down - z_up) / (z_down + z_up)) ** 2
    transmission = 4.0 * z_up * z_down / (z_down + z_up) ** 2
    return True, z_up, z_down, float(reflection), float(transmission), (
        "Impedance preview uses a gamma_eff and sound-speed proxy; treat it as an interpretation aid only.",
    )


def _associated_branch_candidates(
    dataset: DerivedRunData,
    incident_branch: WaveBranchSummary,
    *,
    all_branches: tuple[WaveBranchSummary, ...],
    boundary_index: float,
    crossing_time_s: float,
    time_tolerance_s: float,
    expected_direction: str | None,
    allowed_types: tuple[str, ...],
) -> list[WaveBranchSummary]:
    matches: list[tuple[tuple[int, int, float, float, float, str], WaveBranchSummary]] = []
    for branch in all_branches:
        if branch.branch_id == incident_branch.branch_id or branch.support_class == "provisional":
            continue
        if allowed_types and str(branch.branch_type) not in allowed_types:
            continue
        branch_times = _branch_time_axis(dataset, branch)
        if branch_times.size == 0:
            continue
        nearest_time_delta = float(np.nanmin(np.abs(branch_times - float(crossing_time_s))))
        if nearest_time_delta > float(time_tolerance_s):
            continue
        interface_distance = _branch_position_distance_cells(branch, boundary_index)
        if not np.isfinite(interface_distance) or interface_distance > 2.5:
            continue
        direction_penalty = 0
        if expected_direction is not None and str(branch.propagation_direction or "") != str(expected_direction):
            direction_penalty = 1
        significance = 0.0 if branch.significance is None or not np.isfinite(float(branch.significance)) else float(branch.significance)
        matches.append(
            (
                (
                    _branch_support_priority(branch),
                    direction_penalty,
                    nearest_time_delta,
                    interface_distance,
                    -significance,
                    str(branch.branch_id),
                ),
                branch,
            )
        )
    matches.sort(key=lambda item: item[0])
    return [branch for _, branch in matches]


def _classification_for_event(
    incident_branch: WaveBranchSummary,
    *,
    transmitted: WaveBranchSummary | None,
    reflected_shock: WaveBranchSummary | None,
    reflected_release: WaveBranchSummary | None,
    contact_branch: WaveBranchSummary | None,
) -> tuple[str, bool, tuple[str, ...]]:
    notes: list[str] = []
    outcomes = sum(
        branch is not None
        for branch in (transmitted, reflected_shock, reflected_release, contact_branch)
    )
    if incident_branch.branch_type == "contact_transition":
        return "contact_continuation", False, ()
    if outcomes >= 2:
        notes.append("Multiple post-interface branch families were active in the same gating window.")
        return "unresolved_ambiguous_split", True, tuple(notes)
    if transmitted is not None or incident_branch.branch_type == "transmitted_shock":
        return "transmitted_shock", False, tuple(notes)
    if reflected_shock is not None or incident_branch.branch_type == "reflected_shock":
        return "reflected_shock", False, tuple(notes)
    if reflected_release is not None or incident_branch.branch_type in {"release_rarefaction", "rear_rarefaction"}:
        return "reflected_release", False, tuple(notes)
    if contact_branch is not None:
        return "contact_continuation", False, tuple(notes)
    notes.append("No clean transmitted or reflected companion branch was isolated near the interface.")
    return "unresolved_ambiguous_split", True, tuple(notes)


def build_interface_events_from_wave_tracking(dataset: DerivedRunData, wave_tracking: WaveTrackingResult) -> InterfaceEventsResult:
    """Build interface-event records from tracked wave branches."""

    if wave_tracking is None or not wave_tracking.branches:
        return InterfaceEventsResult(
            available=False,
            supported=True,
            events=(),
            notes=("No tracked branches were available for interface-event association.",),
            warnings=tuple(),
        )
    interface_grid = _build_interface_position_grid(dataset)
    boundaries = tuple(region_interface_boundaries(dataset.regions))
    if not boundaries:
        return InterfaceEventsResult(
            available=False,
            supported=True,
            events=(),
            notes=("No material or region interfaces were available for interface-event association.",),
            warnings=tuple(),
        )

    dt_median = float(np.nanmedian(np.diff(np.asarray(dataset.time_s, dtype=np.float64)))) if dataset.time_s.size >= 2 else 0.0
    time_tolerance_s = max(3.0 * dt_median, 1.0e-12)
    classification_counts: dict[str, int] = {}
    available_metrics: set[str] = {"classification", "timing", "support"}
    warnings: list[DerivedWarning] = []
    notes: list[str] = ["Interface events were associated from the multi-branch wave tracker."]
    events: list[InterfaceEventRecord] = []
    tracked_event_count = 0
    weak_event_count = 0
    suppressed_event_count = 0

    meaningful_branches = tuple(branch for branch in wave_tracking.branches if branch.support_class != "provisional")
    for branch in wave_tracking.branches:
        if branch.interface_index is None or branch.snapshot_indices.size < 2:
            continue
        branch_times = _branch_time_axis(dataset, branch)
        branch_interface = np.asarray(branch.interface_index, dtype=np.float64)
        if branch_times.size < 2 or branch_interface.size < 2:
            continue
        for left_region, right_region, boundary_zone in boundaries:
            boundary_index = float(int(boundary_zone) - 1)
            direction = "high_to_low" if branch.propagation_direction == "high_to_low" else "low_to_high"
            crossing_snapshot, crossing_time = _interpolate_threshold_crossing(
                branch_times,
                branch_interface,
                threshold=boundary_index,
                direction=direction,
            )
            if crossing_time is None or crossing_snapshot is None:
                continue
            if branch.support_class == "provisional":
                suppressed_event_count += 1
                continue

            snapshot = int(branch.snapshot_indices[min(crossing_snapshot, branch.snapshot_indices.size - 1)])
            interface_position = _position_from_interface_index(interface_grid[snapshot], boundary_index)
            interface_label = f"Region {left_region} -> {right_region}"
            opposite_direction = "low_to_high" if branch.propagation_direction == "high_to_low" else "high_to_low"
            transmitted_matches = _associated_branch_candidates(
                dataset,
                branch,
                all_branches=meaningful_branches,
                boundary_index=boundary_index,
                crossing_time_s=float(crossing_time),
                time_tolerance_s=time_tolerance_s,
                expected_direction=branch.propagation_direction,
                allowed_types=("transmitted_shock", "compressive_shock"),
            )
            reflected_shock_matches = _associated_branch_candidates(
                dataset,
                branch,
                all_branches=meaningful_branches,
                boundary_index=boundary_index,
                crossing_time_s=float(crossing_time),
                time_tolerance_s=time_tolerance_s,
                expected_direction=opposite_direction,
                allowed_types=("reflected_shock",),
            )
            reflected_release_matches = _associated_branch_candidates(
                dataset,
                branch,
                all_branches=meaningful_branches,
                boundary_index=boundary_index,
                crossing_time_s=float(crossing_time),
                time_tolerance_s=time_tolerance_s,
                expected_direction=opposite_direction,
                allowed_types=("release_rarefaction", "rear_rarefaction"),
            )
            contact_matches = _associated_branch_candidates(
                dataset,
                branch,
                all_branches=meaningful_branches,
                boundary_index=boundary_index,
                crossing_time_s=float(crossing_time),
                time_tolerance_s=time_tolerance_s,
                expected_direction=None,
                allowed_types=("contact_transition",),
            )

            transmitted = branch if branch.branch_type == "transmitted_shock" else (transmitted_matches[0] if transmitted_matches else None)
            reflected_shock = reflected_shock_matches[0] if reflected_shock_matches else None
            reflected_release = reflected_release_matches[0] if reflected_release_matches else None
            contact_branch = branch if branch.branch_type == "contact_transition" else (contact_matches[0] if contact_matches else None)

            event_classification, ambiguous, classification_notes = _classification_for_event(
                branch,
                transmitted=transmitted,
                reflected_shock=reflected_shock,
                reflected_release=reflected_release,
                contact_branch=contact_branch,
            )
            classification_counts[event_classification] = classification_counts.get(event_classification, 0) + 1
            if branch.support_class == "tracked":
                tracked_event_count += 1
            else:
                weak_event_count += 1

            time_window = _local_time_window(dataset, snapshot, radius=2)
            local_times = np.asarray(dataset.time_s[time_window], dtype=np.float64)
            boundary_interface_index = int(boundary_zone) - 1
            upstream_zone, downstream_zone = _upstream_downstream_zone_indices(
                boundary_interface_index,
                branch.propagation_direction,
                int(dataset.summary["n_zones"]),
            )
            upstream_state = _state_summary(dataset, snapshot, upstream_zone)
            downstream_state = _state_summary(dataset, snapshot, downstream_zone)
            event_notes: list[str] = [f"Incident branch {branch.branch_id} ({branch.branch_type})."]
            event_notes.extend(classification_notes)

            upstream_density = np.empty(0, dtype=np.float64) if upstream_zone is None else _density_series(dataset, time_window, upstream_zone)
            downstream_density = np.empty(0, dtype=np.float64) if downstream_zone is None else _density_series(dataset, time_window, downstream_zone)
            upstream_pressure = None if upstream_zone is None else _pressure_series(dataset, time_window, upstream_zone)
            downstream_pressure = None if downstream_zone is None else _pressure_series(dataset, time_window, downstream_zone)
            pressure_impulse_upstream = _pressure_impulse(upstream_pressure, local_times)
            pressure_impulse_downstream = _pressure_impulse(downstream_pressure, local_times)
            incident_peak_pressure = (
                None if upstream_pressure is None or not np.any(np.isfinite(upstream_pressure)) else float(np.nanmax(upstream_pressure))
            )
            transmitted_peak_pressure = (
                None if downstream_pressure is None or not np.any(np.isfinite(downstream_pressure)) else float(np.nanmax(downstream_pressure))
            )
            if reflected_release is not None and upstream_pressure is not None and np.any(np.isfinite(upstream_pressure)):
                reflected_peak_pressure = float(np.nanmin(upstream_pressure))
            elif reflected_shock is not None and upstream_pressure is not None and np.any(np.isfinite(upstream_pressure)):
                reflected_peak_pressure = float(np.nanmax(upstream_pressure))
            else:
                reflected_peak_pressure = None

            incident_compression_ratio = _compression_ratio(upstream_density, allow_rarefaction=False)
            transmitted_compression_ratio = _compression_ratio(
                downstream_density,
                allow_rarefaction=(event_classification == "reflected_release"),
            )
            if reflected_release is not None:
                reflected_compression_ratio = _compression_ratio(upstream_density, allow_rarefaction=True)
            elif reflected_shock is not None:
                reflected_compression_ratio = _compression_ratio(upstream_density, allow_rarefaction=False)
            else:
                reflected_compression_ratio = None

            incident_speed = _branch_series_at_time(dataset, branch, np.asarray(branch.velocity_cm_s, dtype=np.float64), float(crossing_time))
            transmitted_speed = (
                None
                if transmitted is None
                else _branch_series_at_time(dataset, transmitted, np.asarray(transmitted.velocity_cm_s, dtype=np.float64), float(crossing_time))
            )
            reflected_branch = reflected_shock if reflected_shock is not None else reflected_release
            reflected_speed = (
                None
                if reflected_branch is None
                else _branch_series_at_time(dataset, reflected_branch, np.asarray(reflected_branch.velocity_cm_s, dtype=np.float64), float(crossing_time))
            )

            incident_energy = None
            transmitted_energy = None
            reflected_energy = None
            transfer_fraction = None
            reflection_fraction = None
            transfer_overlap_flag = False
            dominant_channel = None
            channel_fraction_internal = None
            channel_fraction_kinetic = None
            channel_fraction_pressure_work = None
            impedance_supported = None
            impedance_upstream = None
            impedance_downstream = None
            impedance_reflection_preview = None
            impedance_transmission_preview = None
            if pressure_impulse_upstream is not None:
                available_metrics.add("pressure_impulse")
            if upstream_zone is not None and downstream_zone is not None:
                upstream_fluxes, upstream_flux_notes = _energy_flux_diagnostics(dataset, time_window, upstream_zone)
                downstream_fluxes, downstream_flux_notes = _energy_flux_diagnostics(dataset, time_window, downstream_zone)
                event_notes.extend(upstream_flux_notes)
                event_notes.extend(downstream_flux_notes)
                if upstream_fluxes is not None and downstream_fluxes is not None:
                    center_offset = int(np.clip(np.searchsorted(time_window, snapshot), 0, max(time_window.size - 1, 0)))
                    incident_sign = -1.0 if branch.propagation_direction == "high_to_low" else 1.0
                    baseline_up = {key: float(np.asarray(values, dtype=np.float64)[0]) for key, values in upstream_fluxes.items()}
                    baseline_down = {key: float(np.asarray(values, dtype=np.float64)[0]) for key, values in downstream_fluxes.items()}
                    incident_flux = incident_sign * (np.asarray(upstream_fluxes["F_E"], dtype=np.float64) - baseline_up["F_E"])
                    transmitted_flux = incident_sign * (np.asarray(downstream_fluxes["F_E"], dtype=np.float64) - baseline_down["F_E"])
                    reflected_flux = -incident_sign * (np.asarray(upstream_fluxes["F_E"], dtype=np.float64) - baseline_up["F_E"])
                    incident_energy = _safe_integral(local_times, np.maximum(incident_flux, 0.0))
                    transmitted_energy = _safe_integral(local_times[center_offset:], np.maximum(transmitted_flux[center_offset:], 0.0))
                    reflected_energy = _safe_integral(local_times[center_offset:], np.maximum(reflected_flux[center_offset:], 0.0))

                    downstream_internal = _safe_integral(local_times[center_offset:], np.maximum(incident_sign * (np.asarray(downstream_fluxes["F_int"], dtype=np.float64)[center_offset:] - baseline_down["F_int"]), 0.0))
                    downstream_kinetic = _safe_integral(local_times[center_offset:], np.maximum(incident_sign * (np.asarray(downstream_fluxes["F_k"], dtype=np.float64)[center_offset:] - baseline_down["F_k"]), 0.0))
                    downstream_pressure_work = _safe_integral(local_times[center_offset:], np.maximum(incident_sign * (np.asarray(downstream_fluxes["F_P"], dtype=np.float64)[center_offset:] - baseline_down["F_P"]), 0.0))
                    dominant_channel, channel_fraction_internal, channel_fraction_kinetic, channel_fraction_pressure_work = _dominant_channel(
                        {
                            "internal": downstream_internal,
                            "kinetic": downstream_kinetic,
                            "pressure_work": downstream_pressure_work,
                        }
                    )
                    if dominant_channel is not None:
                        available_metrics.add("transfer_channels")
                    if incident_energy is not None and incident_energy > 0.0 and not ambiguous and branch.support_class == "tracked":
                        transfer_fraction = None if transmitted_energy is None else float(np.clip(transmitted_energy / incident_energy, 0.0, 5.0))
                        reflection_fraction = None if reflected_energy is None else float(np.clip(reflected_energy / incident_energy, 0.0, 5.0))
                        available_metrics.add("transfer_fractions")
                        if (
                            (transfer_fraction is not None and transfer_fraction > 1.05)
                            or (reflection_fraction is not None and reflection_fraction > 1.05)
                            or (
                                transfer_fraction is not None
                                and reflection_fraction is not None
                                and (transfer_fraction + reflection_fraction) > 1.10
                            )
                        ):
                            transfer_overlap_flag = True
                            event_notes.append(
                                "Diagnostic transfer fractions exceeded unity in the gated window; treat them as overlap-sensitive diagnostics rather than conservative coefficients."
                            )
                    elif incident_energy is not None and incident_energy > 0.0:
                        event_notes.append("Transfer fractions were suppressed because the event is ambiguous or only weakly supported.")

                    impedance_supported, impedance_upstream, impedance_downstream, impedance_reflection_preview, impedance_transmission_preview, impedance_notes = _impedance_preview(
                        pressure_up=None if upstream_pressure is None else float(upstream_pressure[0]),
                        density_up=None if upstream_density.size == 0 else float(upstream_density[0]),
                        e_int_up=float(upstream_fluxes["e_int"][0]) if np.isfinite(upstream_fluxes["e_int"][0]) else None,
                        pressure_down=None if downstream_pressure is None else float(downstream_pressure[0]),
                        density_down=None if downstream_density.size == 0 else float(downstream_density[0]),
                        e_int_down=float(downstream_fluxes["e_int"][0]) if np.isfinite(downstream_fluxes["e_int"][0]) else None,
                    )
                    event_notes.extend(impedance_notes)
                    if impedance_supported:
                        available_metrics.add("impedance_preview")

            event_significance = 0.0 if branch.significance is None or not np.isfinite(float(branch.significance)) else float(branch.significance)
            if transmitted is not None and transmitted is not branch and transmitted.significance is not None and np.isfinite(float(transmitted.significance)):
                event_significance += 0.25 * float(transmitted.significance)
            if reflected_branch is not None and reflected_branch.significance is not None and np.isfinite(float(reflected_branch.significance)):
                event_significance += 0.2 * float(reflected_branch.significance)
            confidence_inputs = [
                value
                for value in (
                    branch.confidence,
                    None if transmitted is None else transmitted.confidence,
                    None if reflected_branch is None else reflected_branch.confidence,
                )
                if value is not None and np.isfinite(float(value))
            ]
            event_confidence = float(np.nanmean(confidence_inputs)) if confidence_inputs else None
            if event_confidence is not None and (ambiguous or transfer_overlap_flag):
                event_confidence *= 0.75

            events.append(
                InterfaceEventRecord(
                    event_kind="wave_interface_event",
                    interface_label=interface_label,
                    boundary_zone=int(boundary_zone),
                    snapshot_index=snapshot,
                    time_s=float(crossing_time),
                    position_cm=(None if not np.isfinite(interface_position) else float(interface_position)),
                    branch_id=branch.branch_id,
                    event_classification=event_classification,
                    support_class=str(branch.support_class),
                    significance=float(event_significance),
                    confidence=event_confidence,
                    ambiguous=bool(ambiguous or branch.ambiguous or transfer_overlap_flag),
                    incident_branch_type=str(branch.branch_type),
                    incident_arrival_time_s=float(crossing_time),
                    transmitted_branch_id=(None if transmitted is None else str(transmitted.branch_id)),
                    transmitted_branch_type=(None if transmitted is None else str(transmitted.branch_type)),
                    transmitted_time_s=(
                        None
                        if transmitted is None
                        else float(crossing_time)
                        if transmitted.branch_id == branch.branch_id
                        else float(_branch_time_axis(dataset, transmitted)[0])
                    ),
                    reflected_branch_id=(None if reflected_branch is None else str(reflected_branch.branch_id)),
                    reflected_branch_type=(None if reflected_branch is None else str(reflected_branch.branch_type)),
                    reflected_time_s=(None if reflected_branch is None else float(_branch_time_axis(dataset, reflected_branch)[0])),
                    incident_peak_pressure_j_cm3=incident_peak_pressure,
                    transmitted_peak_pressure_j_cm3=transmitted_peak_pressure,
                    reflected_peak_pressure_j_cm3=reflected_peak_pressure,
                    incident_compression_ratio=incident_compression_ratio,
                    transmitted_compression_ratio=transmitted_compression_ratio,
                    reflected_compression_ratio=reflected_compression_ratio,
                    incident_speed_cm_s=incident_speed,
                    transmitted_speed_cm_s=transmitted_speed,
                    reflected_speed_cm_s=reflected_speed,
                    pressure_impulse_upstream_j_s_cm3=pressure_impulse_upstream,
                    pressure_impulse_downstream_j_s_cm3=pressure_impulse_downstream,
                    incident_energy_j_cm2=incident_energy,
                    transmitted_energy_j_cm2=transmitted_energy,
                    reflected_energy_j_cm2=reflected_energy,
                    transfer_fraction=transfer_fraction,
                    reflection_fraction=reflection_fraction,
                    dominant_transfer_channel=dominant_channel,
                    channel_fraction_internal=channel_fraction_internal,
                    channel_fraction_kinetic=channel_fraction_kinetic,
                    channel_fraction_pressure_work=channel_fraction_pressure_work,
                    impedance_preview_supported=impedance_supported,
                    impedance_upstream=impedance_upstream,
                    impedance_downstream=impedance_downstream,
                    impedance_reflection_preview=impedance_reflection_preview,
                    impedance_transmission_preview=impedance_transmission_preview,
                    upstream_state=upstream_state,
                    downstream_state=downstream_state,
                    legal_behavior=(None if event_classification == "unresolved_ambiguous_split" else True),
                    notes=tuple(event_notes),
                )
            )

    if suppressed_event_count > 0:
        notes.append(f"{suppressed_event_count} interface-event candidates were suppressed because they were supported only by provisional branches.")
    if weak_event_count > 0:
        notes.append(f"{weak_event_count} interface-event records remain marked as short / weak because the incident branch support is below the tracked threshold.")
    if not events:
        if suppressed_event_count > 0:
            notes.append("Only provisional branch crossings were detected, so no default interface events were surfaced.")
        return InterfaceEventsResult(
            available=False,
            supported=True,
            events=(),
            tracked_event_count=0,
            weak_event_count=0,
            suppressed_event_count=suppressed_event_count,
            classification_counts=(),
            available_metrics=tuple(sorted(available_metrics)),
            notes=tuple(notes),
            warnings=tuple(warnings),
        )
    events.sort(
        key=lambda event: (
            0 if str(event.support_class) == "tracked" else 1,
            -(0.0 if event.significance is None or not np.isfinite(float(event.significance)) else float(event.significance)),
            float("inf") if event.time_s is None or not np.isfinite(float(event.time_s)) else float(event.time_s),
            str(event.interface_label),
        )
    )
    notes.append("Interface transfer metrics are diagnostic flux and impulse measures, not conserved single-shock invariants.")
    return InterfaceEventsResult(
        available=True,
        supported=True,
        events=tuple(events),
        tracked_event_count=tracked_event_count,
        weak_event_count=weak_event_count,
        suppressed_event_count=suppressed_event_count,
        classification_counts=tuple(sorted(classification_counts.items())),
        available_metrics=tuple(sorted(available_metrics)),
        notes=tuple(notes),
        warnings=tuple(warnings),
    )


def build_shock_tracking_compatibility_result(dataset: DerivedRunData, wave_tracking: WaveTrackingResult) -> ShockTrackingResult:
    """Adapt the dominant compressive branch from wave tracking to legacy shock output."""

    n_snapshots = int(dataset.time_s.size)
    raw_position = np.full(n_snapshots, np.nan, dtype=np.float64)
    smoothed_position = np.full(n_snapshots, np.nan, dtype=np.float64)
    raw_zone_index = np.full(n_snapshots, -1, dtype=np.int32)
    smoothed_zone_index = np.full(n_snapshots, np.nan, dtype=np.float64)
    detector_score = np.full(n_snapshots, np.nan, dtype=np.float64)
    if not wave_tracking.branches:
        empty = np.full(n_snapshots, np.nan, dtype=np.float64)
        empty_i = np.full(n_snapshots, -1, dtype=np.int32)
        return ShockTrackingResult(
            method="wave-tracking compatibility",
            coordinate_label=_coordinate_label(dataset),
            time_s=np.asarray(dataset.time_s, dtype=np.float64),
            position_cm=empty,
            zone_index=empty_i,
            velocity_cm_s=empty,
            speed_magnitude_cm_s=empty,
            detector_score=empty,
            smoothed_position_cm=empty,
            smoothed_zone_index=empty,
            activation_snapshot_index=None,
            propagation_direction=infer_primary_propagation_direction(dataset, RunContext.empty())[0],
            breakout_time_s=None,
            interface_crossings=(),
            warnings=(DerivedWarning("shock", "Wave-tracking compatibility found no compressive branch.", severity="warning"),),
        )
    primary = next((branch for branch in wave_tracking.branches if branch.primary), min(wave_tracking.branches, key=_branch_family_priority))
    indices = primary.snapshot_indices.astype(int)
    if primary.interface_index is not None:
        smoothed_zone_index[indices] = np.asarray(primary.interface_index, dtype=np.float64)
        raw_zone_index[indices] = np.rint(np.asarray(primary.interface_index, dtype=np.float64)).astype(np.int32)
    raw_position[indices] = np.asarray(primary.position_cm, dtype=np.float64)
    smoothed_position[indices] = np.asarray(primary.position_cm, dtype=np.float64)
    detector_score[indices] = np.asarray(primary.score, dtype=np.float64)
    velocity_cm_s = np.full(n_snapshots, np.nan, dtype=np.float64)
    speed_magnitude_cm_s = np.full(n_snapshots, np.nan, dtype=np.float64)
    velocity_cm_s[indices] = np.asarray(primary.velocity_cm_s, dtype=np.float64)
    speed_magnitude_cm_s[indices] = np.abs(np.asarray(primary.velocity_cm_s, dtype=np.float64))
    interface_events = build_interface_events_from_wave_tracking(dataset, wave_tracking)
    interface_crossings = tuple(
        ShockInterfaceCrossing(
            interface_label=event.interface_label,
            boundary_zone=int(event.boundary_zone),
            crossing_snapshot=event.snapshot_index,
            crossing_time_s=event.time_s,
            crossing_position_cm=event.position_cm,
        )
        for event in interface_events.events
        if event.branch_id == primary.branch_id
    )
    return ShockTrackingResult(
        method="wave-tracking compatibility",
        coordinate_label=_coordinate_label(dataset),
        time_s=np.asarray(dataset.time_s, dtype=np.float64),
        position_cm=raw_position,
        zone_index=raw_zone_index,
        velocity_cm_s=velocity_cm_s,
        speed_magnitude_cm_s=speed_magnitude_cm_s,
        detector_score=detector_score,
        smoothed_position_cm=smoothed_position,
        smoothed_zone_index=smoothed_zone_index,
        activation_snapshot_index=(None if indices.size == 0 else int(indices[0])),
        propagation_direction=str(primary.propagation_direction or infer_primary_propagation_direction(dataset, RunContext.empty())[0]),
        breakout_time_s=primary.breakout_time_s,
        interface_crossings=interface_crossings,
        warnings=tuple(wave_tracking.warnings),
    )


def build_wave_tracking_compatibility_result(dataset: DerivedRunData, shock: ShockTrackingResult) -> WaveTrackingResult:
    """Adapt the legacy single-front tracker into the wave-tracking seam."""

    n_snapshots = int(np.asarray(shock.time_s, dtype=np.float64).size)
    interface_axis = np.full((n_snapshots, 1), np.nan, dtype=np.float64)
    if n_snapshots:
        interface_axis[:, 0] = np.asarray(shock.smoothed_position_cm, dtype=np.float64)
    shock_scores = np.asarray(shock.detector_score, dtype=np.float64).reshape(-1, 1) if n_snapshots else np.empty((0, 1), dtype=np.float64)
    fit_seed = WaveFrontFitSeed(
        model_name="a+b*tanh((x-x_f)/w)",
        front_position_cm=(float(shock.smoothed_position_cm[shock.activation_snapshot_index]) if shock.activation_snapshot_index is not None and np.isfinite(shock.smoothed_position_cm[shock.activation_snapshot_index]) else None),
        effective_width_cm=None,
        fitted_fields=("density", "velocity"),
        notes=("Compatibility-only placeholder; no sub-cell fit was performed.",),
    )
    finite = np.isfinite(np.asarray(shock.smoothed_zone_index, dtype=np.float64))
    candidates = tuple(
        WaveFrontCandidate(
            snapshot_index=int(snapshot_index),
            family="shock_like",
            candidate_type="compressive_shock",
            coordinate_label=str(shock.coordinate_label),
            interface_index=float(shock.smoothed_zone_index[snapshot_index]),
            position_cm=None if not np.isfinite(shock.smoothed_position_cm[snapshot_index]) else float(shock.smoothed_position_cm[snapshot_index]),
            width_cm=None,
            score=0.0 if not np.isfinite(shock.detector_score[snapshot_index]) else float(shock.detector_score[snapshot_index]),
            propagation_direction=str(shock.propagation_direction),
            direction_sign=(1.0 if str(shock.propagation_direction) == "low_to_high" else -1.0),
            ambiguous=False,
            branch_hint="primary",
            fit_seed=(fit_seed if snapshot_index == (shock.activation_snapshot_index or -1) else None),
        )
        for snapshot_index in range(n_snapshots)
        if finite[snapshot_index]
    )
    sample_count = int(np.count_nonzero(finite))
    support_class = _classify_branch_support(sample_count)
    duration_s = None
    if sample_count >= 2:
        active_indices = np.flatnonzero(finite).astype(np.int32, copy=False)
        duration_s = float(dataset.time_s[int(active_indices[-1])] - dataset.time_s[int(active_indices[0])])
    integrated_score = float(np.nansum(np.clip(np.asarray(shock.detector_score[finite], dtype=np.float64), 0.0, None))) if sample_count else 0.0
    finite_positions = np.asarray(shock.smoothed_position_cm[finite], dtype=np.float64)
    position_span_cm = float(np.nanmax(finite_positions) - np.nanmin(finite_positions)) if finite_positions.size >= 2 else 0.0
    primary_branch = WaveBranchSummary(
        branch_id="primary",
        family="shock_like",
        branch_type="compressive_shock",
        snapshot_indices=np.flatnonzero(finite).astype(np.int32, copy=False),
        interface_index=np.asarray(shock.smoothed_zone_index[finite], dtype=np.float64),
        position_cm=np.asarray(shock.smoothed_position_cm[finite], dtype=np.float64),
        velocity_cm_s=np.asarray(shock.velocity_cm_s[finite], dtype=np.float64),
        score=np.asarray(shock.detector_score[finite], dtype=np.float64),
        width_cm=None,
        propagation_direction=str(shock.propagation_direction),
        breakout_time_s=shock.breakout_time_s,
        support_class=support_class,
        sample_count=sample_count,
        duration_s=duration_s,
        integrated_score=integrated_score,
        position_span_cm=position_span_cm,
        significance=(None if sample_count == 0 else integrated_score * (1.0 + 0.45 * math.log1p(sample_count))),
        continuity_fraction=1.0 if sample_count else None,
        primary=True,
        notes=("Compatibility branch generated from the legacy single-front tracker.",),
    )
    formula_hooks = shock_tracking_formula_hooks(dataset)
    evidence_map = WaveEvidenceMap(
        family="shock_like",
        coordinate_label=str(shock.coordinate_label),
        time_s=np.asarray(shock.time_s, dtype=np.float64),
        interface_position_cm=interface_axis,
        score=shock_scores,
        formula_hook=formula_hooks[0],
        notes=("Compatibility-only map storing the tracked primary shock evidence trace.",),
    )
    return WaveTrackingResult(
        method="compatibility-primary-front",
        coordinate_label=str(shock.coordinate_label),
        supported_formula_hooks=formula_hooks,
        evidence_maps=(evidence_map,),
        candidates=candidates,
        branches=(primary_branch,),
        primary_branch_id="primary",
        candidate_count=len(candidates),
        tracked_branch_count=(1 if support_class == "tracked" else 0),
        short_branch_count=(1 if support_class == "short_weak" else 0),
        provisional_branch_count=(1 if support_class == "provisional" else 0),
        suppressed_branch_count=0,
        compatibility_source="ShockTrackingResult",
        warnings=tuple(shock.warnings),
    )


def build_interface_event_compatibility_result(shock: ShockTrackingResult) -> InterfaceEventsResult:
    """Adapt legacy shock interface crossings into the future event seam."""

    events = tuple(
        InterfaceEventRecord(
            event_kind="primary_shock_crossing",
            interface_label=str(crossing.interface_label),
            boundary_zone=int(crossing.boundary_zone),
            snapshot_index=(None if crossing.crossing_snapshot is None else int(crossing.crossing_snapshot)),
            time_s=(None if crossing.crossing_time_s is None else float(crossing.crossing_time_s)),
            position_cm=(None if crossing.crossing_position_cm is None else float(crossing.crossing_position_cm)),
            branch_id="primary",
            legal_behavior=None,
            notes=("Compatibility event promoted from legacy shock interface crossings.",),
        )
        for crossing in shock.interface_crossings
    )
    return InterfaceEventsResult(
        available=bool(events),
        supported=True,
        events=events,
        notes=("Compatibility-only event view derived from the legacy single-front tracker.",),
        warnings=tuple(),
    )


def track_shock_front(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    parameters: DerivedAnalysisParameters | None = None,
    geometry: AnalysisGeometryMetadata | None = None,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> ShockTrackingResult:
    """Track the primary shock front through time using continuity constraints."""

    if parameters is None:
        from helios.services.derived.analysis import DerivedAnalysisParameters as _DerivedAnalysisParameters

        parameters = _DerivedAnalysisParameters()
    if geometry is None:
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )

    warnings: list[DerivedWarning] = []
    propagation_direction, expected_velocity_sign = infer_primary_propagation_direction(dataset, context)
    n_snapshots = int(dataset.time_s.size)
    n_interfaces = int(dataset.summary["n_zones"]) - 1
    if n_snapshots == 0 or n_interfaces <= 0:
        warnings.append(DerivedWarning("shock", "Shock tracking is unavailable because the run has no usable snapshots.", severity="error"))
        empty = np.full(n_snapshots, np.nan, dtype=np.float64)
        empty_i = np.full(n_snapshots, -1, dtype=np.int32)
        return ShockTrackingResult(
            method="continuity-constrained signed d(rho/rho0)",
            coordinate_label=_coordinate_label(dataset),
            time_s=np.asarray(dataset.time_s, dtype=np.float64),
            position_cm=empty,
            zone_index=empty_i,
            velocity_cm_s=empty,
            speed_magnitude_cm_s=empty,
            detector_score=empty,
            smoothed_position_cm=empty,
            smoothed_zone_index=empty,
            activation_snapshot_index=None,
            propagation_direction=propagation_direction,
            breakout_time_s=None,
            interface_crossings=(),
            warnings=tuple(warnings),
        )

    sign = 1.0 if propagation_direction == "high_to_low" else -1.0
    initial_density = np.asarray(dataset.zone_initial_density_g_cm3, dtype=np.float64)
    fallback_density = np.asarray(dataset.density_g_cm3[0], dtype=np.float64)
    initial_density = np.where(np.isfinite(initial_density) & (initial_density > 0.0), initial_density, fallback_density)
    initial_density = np.where(initial_density > 0.0, initial_density, 1.0)

    event_scaffold = _ShockEventScaffold(
        time_s=np.asarray(dataset.time_s, dtype=np.float64),
        interface_position_grid=_build_interface_position_grid(dataset),
        interface_boundaries=tuple(region_interface_boundaries(dataset.regions)),
    )
    selection_request_key = selection_request_cache_key(
        context,
        reuse_viewer_subset=parameters.reuse_viewer_subset,
        derived_region_ids=parameters.derived_region_ids,
        derived_material_ids=parameters.derived_material_ids,
        exclude_entry_region=parameters.exclude_entry_region,
        exclude_low_density=parameters.exclude_low_density,
        min_density_g_cm3=parameters.min_density_g_cm3,
        exclude_opposite_velocity=parameters.exclude_opposite_velocity,
        zone_index_lower=parameters.zone_index_lower,
        zone_index_upper=parameters.zone_index_upper,
        weighting_mode=parameters.weighting_mode,
    )
    detector_score = cached_time_series_payload(
        (
            "shock.detector_score",
            geometry.observation_side,
            geometry.observation_boundary,
            round(float(geometry.line_of_sight_angle_deg), 12),
            round(float(geometry.impact_parameter_cm), 12),
            geometry.profile_coordinate_mode,
            propagation_direction,
            selection_request_key,
        ),
        analysis_cache=analysis_cache,
        builder=lambda: _build_detector_score(
            dataset,
            context,
            geometry=geometry,
            parameters=parameters,
            initial_density=initial_density,
            sign=sign,
            propagation_direction=propagation_direction,
            expected_velocity_sign=expected_velocity_sign,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        ),
    )

    activation_threshold = _activation_threshold(detector_score)
    hold_threshold = max(0.005, activation_threshold * 0.25)
    activation_snapshot: int | None = None
    activation_index: int | None = None
    for snapshot_index in range(n_snapshots):
        score_row = detector_score[snapshot_index]
        candidate = int(np.argmax(score_row))
        if float(score_row[candidate]) >= activation_threshold:
            activation_snapshot = snapshot_index
            activation_index = candidate
            break

    track_index = np.full(n_snapshots, np.nan, dtype=np.float64)
    raw_zone_index = np.full(n_snapshots, -1, dtype=np.int32)
    detector_trace = np.full(n_snapshots, np.nan, dtype=np.float64)
    if activation_snapshot is None or activation_index is None:
        warnings.append(
            DerivedWarning(
                "shock",
                "Crossing time unavailable because no stable propagating front was detected above the activation threshold.",
                severity="warning",
            )
        )
    else:
        track_index[activation_snapshot] = float(activation_index)
        raw_zone_index[activation_snapshot] = int(activation_index)
        detector_trace[activation_snapshot] = float(detector_score[activation_snapshot, activation_index])
        previous = activation_index
        max_forward_jump = 48
        max_backtrack = 4
        jump_penalty = activation_threshold * 0.015
        for snapshot_index in range(activation_snapshot + 1, n_snapshots):
            score_row = detector_score[snapshot_index]
            if propagation_direction == "high_to_low":
                lower = max(0, previous - max_forward_jump)
                upper = min(n_interfaces - 1, previous + max_backtrack)
            else:
                lower = max(0, previous - max_backtrack)
                upper = min(n_interfaces - 1, previous + max_forward_jump)
            search_indices = np.arange(lower, upper + 1, dtype=np.int32)
            penalized = score_row[search_indices] - jump_penalty * np.abs(search_indices - previous)
            best_offset = int(np.argmax(penalized))
            candidate = int(search_indices[best_offset])
            candidate_score = float(score_row[candidate])
            if candidate_score < hold_threshold:
                candidate = previous
                candidate_score = float(score_row[candidate])
            track_index[snapshot_index] = float(candidate)
            raw_zone_index[snapshot_index] = candidate
            detector_trace[snapshot_index] = candidate_score
            previous = candidate

    smoothed_track = _smooth_track(track_index, direction=propagation_direction)
    raw_position = np.full(n_snapshots, np.nan, dtype=np.float64)
    smoothed_position = np.full(n_snapshots, np.nan, dtype=np.float64)
    for snapshot_index in range(n_snapshots):
        interface_positions = event_scaffold.interface_position_grid[snapshot_index]
        raw_position[snapshot_index] = _position_from_interface_index(interface_positions, track_index[snapshot_index])
        smoothed_position[snapshot_index] = _position_from_interface_index(interface_positions, smoothed_track[snapshot_index])

    finite_position = np.isfinite(smoothed_position)
    signed_velocity = np.full(n_snapshots, np.nan, dtype=np.float64)
    if np.count_nonzero(finite_position) >= 2:
        signed_velocity[finite_position] = np.gradient(smoothed_position[finite_position], dataset.time_s[finite_position])
    speed_magnitude = np.abs(signed_velocity)

    interface_crossings: list[ShockInterfaceCrossing] = []
    for left_region, right_region, boundary_zone in event_scaffold.interface_boundaries:
        boundary_index = int(boundary_zone) - 1
        crossing_snapshot, crossing_time = _interpolate_threshold_crossing(
            event_scaffold.time_s,
            smoothed_track,
            threshold=float(boundary_index),
            direction=propagation_direction,
        )
        crossing_position = None
        if crossing_time is not None and crossing_snapshot is not None:
            crossing_position = _position_from_interface_index(event_scaffold.interface_position_grid[crossing_snapshot], float(boundary_index))
        interface_crossings.append(
            ShockInterfaceCrossing(
                interface_label=f"Region {left_region} -> {right_region}",
                boundary_zone=int(boundary_zone),
                crossing_snapshot=crossing_snapshot,
                crossing_time_s=crossing_time,
                crossing_position_cm=crossing_position,
            )
        )

    if propagation_direction == "high_to_low":
        _, breakout_time = _interpolate_threshold_crossing(event_scaffold.time_s, smoothed_track, threshold=0.5, direction=propagation_direction)
    else:
        _, breakout_time = _interpolate_threshold_crossing(event_scaffold.time_s, smoothed_track, threshold=float(max(0, n_interfaces - 1.5)), direction=propagation_direction)

    if activation_snapshot is not None and np.isfinite(detector_trace[activation_snapshot]):
        warnings.append(DerivedWarning("shock", f"Primary front activated at snapshot {activation_snapshot} with detector score {detector_trace[activation_snapshot]:.3g}.", severity="info"))
    if breakout_time is None:
        warnings.append(DerivedWarning("shock", "No breakout time was inferred from the tracked primary front within the active time window.", severity="info"))
    if not [item for item in interface_crossings if item.crossing_time_s is not None] and len(interface_crossings) > 0:
        warnings.append(DerivedWarning("shock", "Interface crossing times are unavailable because the tracked front did not cross the material boundaries within the active time window.", severity="caution"))

    return ShockTrackingResult(
        method="continuity-constrained signed d(rho/rho0)",
        coordinate_label=_coordinate_label(dataset),
        time_s=np.asarray(dataset.time_s, dtype=np.float64),
        position_cm=raw_position,
        zone_index=raw_zone_index,
        velocity_cm_s=signed_velocity,
        speed_magnitude_cm_s=speed_magnitude,
        detector_score=detector_trace,
        smoothed_position_cm=smoothed_position,
        smoothed_zone_index=smoothed_track,
        activation_snapshot_index=activation_snapshot,
        propagation_direction=propagation_direction,
        breakout_time_s=breakout_time,
        interface_crossings=tuple(interface_crossings),
        warnings=tuple(warnings),
    )
