"""Preheat diagnostics layered on top of advanced wave tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from helios.services.derived.models import (
    DerivedPlotBundle,
    DerivedRunData,
    DerivedWarning,
    InterfaceEventRecord,
    InterfaceEventsResult,
    PreheatBudgetRow,
    PreheatOnsetMarker,
    PreheatProfileField,
    PreheatStateMetric,
    PreheatSummary,
    PreheatThresholds,
    WaveBranchSummary,
    WaveTrackingResult,
)


@dataclass(frozen=True, slots=True)
class _PreheatConfig:
    max_density_ratio: float = 1.10
    max_relative_pressure: float = 0.25
    min_delta_temperature_e_ev: float = 1.0
    min_delta_mean_charge: float = 0.05
    min_delta_electron_energy_j_g: float = 1.0e7
    min_radiation_net_heating_j_g_s: float = 1.0e10
    min_laser_deposition_j_g_s: float = 1.0e10
    severity_moderate_fraction: float = 0.10
    severity_severe_fraction: float = 0.35
    severity_moderate_delta_te_ev: float = 3.0
    severity_severe_delta_te_ev: float = 10.0


_PREHEAT_CONFIG = _PreheatConfig()
_COMPRESSIVE_TYPES = frozenset({"compressive_shock", "transmitted_shock", "reflected_shock"})


def _finite_or_none(value: float | np.ndarray | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _optional_array(dataset: DerivedRunData, attr_name: str) -> np.ndarray | None:
    values = getattr(dataset, attr_name)
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)


def _total_pressure_array(dataset: DerivedRunData) -> np.ndarray | None:
    if dataset.pressure_total_j_cm3 is not None:
        return np.asarray(dataset.pressure_total_j_cm3, dtype=np.float64)
    components: list[np.ndarray] = []
    if dataset.pressure_i_j_cm3 is not None:
        components.append(np.asarray(dataset.pressure_i_j_cm3, dtype=np.float64))
    if dataset.pressure_e_j_cm3 is not None:
        components.append(np.asarray(dataset.pressure_e_j_cm3, dtype=np.float64))
    if dataset.pressure_radiation_j_cm3 is not None:
        components.append(np.asarray(dataset.pressure_radiation_j_cm3, dtype=np.float64))
    if not components:
        return None
    return np.nansum(np.asarray(components, dtype=np.float64), axis=0)


def _internal_energy_array(dataset: DerivedRunData) -> np.ndarray | None:
    components: list[np.ndarray] = []
    if dataset.ion_energy_j_g is not None:
        components.append(np.asarray(dataset.ion_energy_j_g, dtype=np.float64))
    if dataset.electron_energy_j_g is not None:
        components.append(np.asarray(dataset.electron_energy_j_g, dtype=np.float64))
    if dataset.radiation_energy_j_g is not None:
        components.append(np.asarray(dataset.radiation_energy_j_g, dtype=np.float64))
    if not components:
        return None
    return np.nansum(np.asarray(components, dtype=np.float64), axis=0)


def _material_label(dataset: DerivedRunData, material_table_index: int | None) -> str | None:
    if material_table_index is None:
        return None
    material_indices = np.asarray(dataset.materials.get("index", ()), dtype=np.int32)
    if material_indices.size == 0:
        return None
    matches = np.flatnonzero(material_indices == int(material_table_index))
    if matches.size == 0:
        return None
    offset = int(matches[0])
    eos_paths = dataset.materials.get("eos_file_path")
    if eos_paths is None:
        return None
    try:
        return Path(str(eos_paths[offset])).stem
    except (TypeError, ValueError, IndexError):
        return None


def _region_records(dataset: DerivedRunData) -> list[dict[str, object]]:
    region_ids = np.asarray(dataset.regions.get("region_index", ()), dtype=np.int32)
    min_zones = np.asarray(dataset.regions.get("min_zone_index", ()), dtype=np.int32)
    max_zones = np.asarray(dataset.regions.get("max_zone_index", ()), dtype=np.int32)
    material_table = np.asarray(dataset.regions.get("material_table_index", ()), dtype=np.int32)
    records: list[dict[str, object]] = []
    for index, region_id in enumerate(region_ids):
        zone_start = max(0, int(min_zones[index]) - 1)
        zone_stop = min(int(dataset.summary["n_zones"]), int(max_zones[index]))
        material_index = None if material_table.size <= index else int(material_table[index])
        material_label = _material_label(dataset, material_index)
        label = f"Region {int(region_id)}"
        if material_label:
            label += f" ({material_label})"
        records.append(
            {
                "region_id": int(region_id),
                "zone_start": int(zone_start),
                "zone_stop": int(zone_stop),
                "boundary_low_zone": int(min_zones[index]) - 1,
                "boundary_high_zone": int(max_zones[index]),
                "material_index": material_index,
                "label": label,
            }
        )
    return records


def _primary_compressive_branch(wave_tracking: WaveTrackingResult | None) -> WaveBranchSummary | None:
    if wave_tracking is None:
        return None
    compressive = [
        branch
        for branch in wave_tracking.branches
        if str(branch.branch_type) in _COMPRESSIVE_TYPES and str(branch.support_class) != "provisional"
    ]
    if not compressive:
        return None
    primary_id = str(wave_tracking.primary_branch_id or "")
    if primary_id:
        match = next((branch for branch in compressive if str(branch.branch_id) == primary_id), None)
        if match is not None:
            return match
    primary = next((branch for branch in compressive if bool(branch.primary)), None)
    if primary is not None:
        return primary
    return max(
        compressive,
        key=lambda branch: (
            0 if str(branch.support_class) == "tracked" else 1,
            -float(branch.significance or 0.0),
            -int(branch.sample_count),
            str(branch.branch_id),
        ),
    )


def _crossing_time_for_branch(
    dataset: DerivedRunData,
    branch: WaveBranchSummary,
    *,
    boundary_interface_index: float,
) -> tuple[int | None, float | None]:
    indices = np.asarray(branch.snapshot_indices, dtype=np.int32)
    interface_index = np.asarray(branch.interface_index, dtype=np.float64) if branch.interface_index is not None else np.empty(0, dtype=np.float64)
    if indices.size < 2 or interface_index.size != indices.size:
        return None, None
    direction = str(branch.propagation_direction or "")
    times = np.asarray(dataset.time_s[indices], dtype=np.float64)
    for left in range(indices.size - 1):
        x0 = float(interface_index[left])
        x1 = float(interface_index[left + 1])
        if not (math.isfinite(x0) and math.isfinite(x1)):
            continue
        if direction == "high_to_low":
            crossed = x0 > boundary_interface_index >= x1
        else:
            crossed = x0 < boundary_interface_index <= x1
        if not crossed:
            continue
        if x1 == x0:
            return int(indices[left + 1]), float(times[left + 1])
        fraction = (boundary_interface_index - x0) / (x1 - x0)
        crossing_time = float(times[left] + fraction * (times[left + 1] - times[left]))
        return int(indices[left + 1]), crossing_time
    return None, None


def _event_for_primary_branch(
    interface_events: InterfaceEventsResult | None,
    *,
    primary_branch_id: str,
    boundary_zone: int,
) -> InterfaceEventRecord | None:
    if interface_events is None:
        return None
    for event in interface_events.events:
        if int(event.boundary_zone) != int(boundary_zone):
            continue
        if str(event.support_class or "") == "provisional":
            continue
        if str(event.branch_id or "") == primary_branch_id:
            return event
    return None


def _first_branch_sample_time(dataset: DerivedRunData, branch: WaveBranchSummary) -> tuple[int | None, float | None]:
    indices = np.asarray(branch.snapshot_indices, dtype=np.int32)
    if indices.size == 0:
        return None, None
    first_snapshot = int(indices[0])
    if not (0 <= first_snapshot < int(dataset.time_s.size)):
        return first_snapshot, None
    return first_snapshot, float(dataset.time_s[first_snapshot])


def _region_initial_metrics(dataset: DerivedRunData, record: dict[str, object]) -> tuple[float, float]:
    zone_slice = slice(int(record["zone_start"]), int(record["zone_stop"]))
    density0 = np.asarray(dataset.density_g_cm3[0, zone_slice], dtype=np.float64)
    width0 = np.asarray(dataset.zone_width_cm[0, zone_slice], dtype=np.float64)
    width_cm = float(np.nansum(width0)) if width0.size else 0.0
    areal_mass_g_cm2 = float(np.nansum(density0 * width0)) if density0.size and width0.size else 0.0
    return width_cm, areal_mass_g_cm2


def _target_entry_for_region(
    dataset: DerivedRunData,
    *,
    branch: WaveBranchSummary,
    interface_events: InterfaceEventsResult | None,
    record: dict[str, object],
    incident_region_id: int,
    direction: str,
) -> tuple[float | None, str | None, int | None, tuple[str, ...]]:
    notes: list[str] = []
    region_id = int(record["region_id"])
    if region_id == int(incident_region_id):
        first_snapshot, first_time = _first_branch_sample_time(dataset, branch)
        if first_time is None:
            return None, None, None, ("The primary compressive branch never produced a valid first sample for the incident-side region.",)
        boundary_label = str(dataset.laser_entry.get("incident_region_boundary") if dataset.laser_entry else "Incident boundary")
        notes.append("Using the first tracked primary-branch sample as the entry time for the incident-side region of interest.")
        return float(first_time), boundary_label, None, tuple(notes)

    boundary_zone = int(record["boundary_high_zone"] if direction == "high_to_low" else record["boundary_low_zone"])
    event = _event_for_primary_branch(
        interface_events,
        primary_branch_id=str(branch.branch_id),
        boundary_zone=boundary_zone,
    )
    if event is not None and event.time_s is not None:
        return float(event.time_s), str(event.interface_label), int(boundary_zone), tuple(notes)

    crossing_snapshot, crossing_time = _crossing_time_for_branch(
        dataset,
        branch,
        boundary_interface_index=float(boundary_zone - 1),
    )
    if crossing_time is None:
        return None, None, None, tuple(notes)
    if crossing_snapshot is not None:
        notes.append(f"Target-entry time for {record['label']} was inferred from the tracked branch trajectory.")
    return float(crossing_time), f"Region entry @ zone {boundary_zone}", int(boundary_zone), tuple(notes)


def _select_auto_target_candidate(
    candidates: list[dict[str, object]],
    *,
    incident_region_id: int,
) -> tuple[dict[str, object], str]:
    if len(candidates) == 1:
        only = candidates[0]
        return only, f"Auto target guess uses {only['label']} because it is the only region reached by the primary compressive branch."

    selectable = [candidate for candidate in candidates if int(candidate["region_id"]) != int(incident_region_id)] or list(candidates)
    mass_values = np.asarray(
        [max(float(candidate.get("initial_areal_mass_g_cm2", 0.0)), 0.0) for candidate in selectable],
        dtype=np.float64,
    )
    width_values = np.asarray(
        [max(float(candidate.get("initial_width_cm", 0.0)), 0.0) for candidate in selectable],
        dtype=np.float64,
    )
    max_mass = float(np.nanmax(mass_values)) if mass_values.size and np.any(np.isfinite(mass_values)) else 1.0
    max_width = float(np.nanmax(width_values)) if width_values.size and np.any(np.isfinite(width_values)) else 1.0

    def _score(candidate: dict[str, object]) -> tuple[float, float, float, int]:
        path_rank = int(candidate.get("path_rank", 0))
        path_count = int(candidate.get("path_count", len(candidates)))
        mass_score = 0.0 if max_mass <= 0.0 else float(candidate.get("initial_areal_mass_g_cm2", 0.0)) / max_mass
        width_score = 0.0 if max_width <= 0.0 else float(candidate.get("initial_width_cm", 0.0)) / max_width
        score = 1.10 * mass_score + 0.35 * width_score
        if int(candidate["region_id"]) != int(incident_region_id):
            score += 0.75
        if 0 < path_rank < max(1, path_count - 1):
            score += 0.35
        if path_count > 2 and path_rank == path_count - 1:
            score -= 0.20
        if path_rank >= 2:
            score += 0.10
        return score, mass_score, width_score, -path_rank

    best = max(selectable, key=_score)
    deepest = candidates[-1]
    note = (
        "Auto target guess is a heuristic: it prefers a non-incident, sample-like region with larger initial areal mass/thickness and lightly penalizes the deepest terminal region. "
        + (
            f"It selected {best['label']} instead of blindly using the deepest reached region {deepest['label']}."
            if int(best["region_id"]) != int(deepest["region_id"])
            else f"In this case the heuristic still selected the deepest reached region {best['label']}."
        )
    )
    return best, note


def _select_target_region(
    dataset: DerivedRunData,
    *,
    wave_tracking: WaveTrackingResult,
    interface_events: InterfaceEventsResult | None,
    target_region_id: int | None,
) -> tuple[
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
    float | None,
    str | None,
    int | None,
    str,
    tuple[str, ...],
]:
    notes: list[str] = []
    primary_branch = _primary_compressive_branch(wave_tracking)
    if primary_branch is None:
        return None, None, None, None, None, None, None, "auto", ("No reliable non-provisional compressive branch was available for preheat timing.",)
    records = _region_records(dataset)
    if not records:
        return None, None, None, None, None, None, None, "auto", ("Run metadata does not expose region boundaries, so target-region selection is unavailable.",)

    direction = str(primary_branch.propagation_direction or "high_to_low")
    traversal = list(records if direction == "low_to_high" else reversed(records))
    incident_region_id = None if dataset.laser_entry is None else dataset.laser_entry.get("incident_region")
    if incident_region_id is None:
        incident_region_id = int(traversal[0]["region_id"])
        notes.append("Laser-entry metadata was unavailable; the boundary-facing region is being treated as the incident-side region for target selection.")
    try:
        start_index = next(index for index, record in enumerate(traversal) if int(record["region_id"]) == int(incident_region_id))
    except StopIteration:
        start_index = 0
        incident_region_id = int(traversal[0]["region_id"])
        notes.append("Incident-region metadata did not match the stored region ordering; the boundary-facing region is being treated as incident side.")
    downstream = traversal[start_index:]
    incident_record = downstream[0] if downstream else None

    candidates: list[dict[str, object]] = []
    for path_rank, record in enumerate(downstream):
        entry_time_s, interface_label, boundary_zone, entry_notes = _target_entry_for_region(
            dataset,
            branch=primary_branch,
            interface_events=interface_events,
            record=record,
            incident_region_id=int(incident_region_id),
            direction=direction,
        )
        if entry_time_s is None:
            continue
        width_cm, areal_mass_g_cm2 = _region_initial_metrics(dataset, record)
        candidates.append(
            {
                **record,
                "entry_time_s": float(entry_time_s),
                "interface_label": interface_label,
                "boundary_zone": boundary_zone,
                "path_rank": int(path_rank),
                "path_count": int(len(downstream)),
                "initial_width_cm": float(width_cm),
                "initial_areal_mass_g_cm2": float(areal_mass_g_cm2),
                "entry_notes": tuple(entry_notes),
            }
        )

    if not candidates:
        notes.append("The primary compressive branch never entered a region of interest that could anchor a preheat window safely.")
        return None, None, None, incident_record, None, None, None, "auto", tuple(notes)

    deepest_reached = candidates[-1]
    auto_target, auto_note = _select_auto_target_candidate(candidates, incident_region_id=int(incident_region_id))
    notes.extend(str(note) for note in auto_target.get("entry_notes", ()))
    notes.append(auto_note)
    selection_mode = "auto"
    selected = auto_target
    if target_region_id is not None:
        selection_mode = "user_selected"
        selected = next((candidate for candidate in candidates if int(candidate["region_id"]) == int(target_region_id)), None)
        if selected is None:
            notes.append(
                f"User-selected preheat region Region {int(target_region_id)} was not reached by the primary compressive branch, so no target-entry time could be anchored for that override."
            )
            return None, auto_target, deepest_reached, incident_record, None, None, None, selection_mode, tuple(notes)
        notes.extend(str(note) for note in selected.get("entry_notes", ()))
        if int(selected["region_id"]) == int(auto_target["region_id"]):
            notes.append(f"User-selected preheat region matches the automatic target guess: {selected['label']}.")
        else:
            notes.append(
                f"Using user-selected preheat region {selected['label']} instead of the automatic target guess {auto_target['label']}."
            )
    else:
        notes.append(f"Automatic preheat region of interest: {auto_target['label']}.")

    return (
        selected,
        auto_target,
        deepest_reached,
        incident_record,
        float(selected["entry_time_s"]),
        None if selected.get("interface_label") is None else str(selected["interface_label"]),
        None if selected.get("boundary_zone") is None else int(selected["boundary_zone"]),
        selection_mode,
        tuple(notes),
    )


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(array) & np.isfinite(weight_array) & (weight_array > 0.0)
    if not np.any(valid):
        return None
    return float(np.sum(array[valid] * weight_array[valid]) / np.sum(weight_array[valid]))


def _max_over_mask(values: np.ndarray | None, mask: np.ndarray) -> float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array) & mask
    if not np.any(valid):
        return None
    return float(np.nanmax(array[valid]))


def _first_marker_time(metric: np.ndarray, time_s: np.ndarray, threshold: float) -> tuple[float | None, float | None]:
    array = np.asarray(metric, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(array) & (array >= float(threshold)))
    if valid.size == 0:
        return None, None
    index = int(valid[0])
    return float(time_s[index]), float(array[index])


def _source_term_budget(
    dataset: DerivedRunData,
    *,
    attr_name: str,
    density: np.ndarray,
    zone_width: np.ndarray,
    target_mask: np.ndarray,
    time_indices: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    field = _optional_array(dataset, attr_name)
    if field is None or time_indices.size == 0:
        return None, None
    target_rate = np.nansum(density[:, target_mask] * field[:, target_mask] * zone_width[:, target_mask], axis=1)
    cumulative = np.full(target_rate.shape, np.nan, dtype=np.float64)
    if time_indices.size:
        cumulative[int(time_indices[0])] = 0.0
    if time_indices.size >= 2:
        for position in range(1, time_indices.size):
            left = int(time_indices[position - 1])
            right = int(time_indices[position])
            dt = float(dataset.time_s[right] - dataset.time_s[left])
            cumulative[right] = float(cumulative[left]) + 0.5 * (float(target_rate[left]) + float(target_rate[right])) * dt
    last_value = float(cumulative[int(time_indices[-1])]) if time_indices.size else None
    return cumulative, last_value


def _energy_areal_density(energy_j_g: np.ndarray | None, density: np.ndarray, zone_width: np.ndarray, target_mask: np.ndarray) -> np.ndarray | None:
    if energy_j_g is None:
        return None
    energy = np.asarray(energy_j_g, dtype=np.float64)
    return np.nansum(density[:, target_mask] * energy[:, target_mask] * zone_width[:, target_mask], axis=1)


def _support_warning(primary_branch: WaveBranchSummary) -> tuple[DerivedWarning, ...]:
    if str(primary_branch.support_class) == "tracked":
        return ()
    return (
        DerivedWarning(
            "preheat",
            f"Preheat timing uses branch {primary_branch.branch_id}, but its support class is {primary_branch.support_class}. Interpret the target-entry timing cautiously.",
            severity="warning",
        ),
    )


def evaluate_preheat(
    dataset: DerivedRunData,
    *,
    wave_tracking: WaveTrackingResult | None,
    interface_events: InterfaceEventsResult | None = None,
    target_region_id: int | None = None,
) -> PreheatSummary:
    """Diagnose target pre-modification before the tracked main shock enters it."""

    thresholds = PreheatThresholds(
        max_density_ratio=_PREHEAT_CONFIG.max_density_ratio,
        max_relative_pressure=_PREHEAT_CONFIG.max_relative_pressure,
        min_delta_temperature_e_ev=_PREHEAT_CONFIG.min_delta_temperature_e_ev,
        min_delta_mean_charge=_PREHEAT_CONFIG.min_delta_mean_charge,
        min_delta_electron_energy_j_g=_PREHEAT_CONFIG.min_delta_electron_energy_j_g,
        min_radiation_net_heating_j_g_s=_PREHEAT_CONFIG.min_radiation_net_heating_j_g_s,
        min_laser_deposition_j_g_s=_PREHEAT_CONFIG.min_laser_deposition_j_g_s,
        notes=(
            "Preheat cells must stay below the compressive-loading thresholds while showing a measurable thermal, ionization, or source-term change.",
        ),
    )
    capability_notes = tuple(dataset.wave_physics_capabilities.notes)
    if wave_tracking is None:
        return PreheatSummary(
            available=False,
            supported=bool(dataset.wave_physics_capabilities.preheat_supported),
            method="wave-tracking-gated",
            candidate_metric_names=tuple(),
            scalar_summaries={},
            thresholds=thresholds,
            notes=("Wave tracking has not been requested yet, so preheat diagnostics are unavailable.", *capability_notes),
            warnings=tuple(),
        )
    primary_branch = _primary_compressive_branch(wave_tracking)
    if primary_branch is None:
        return PreheatSummary(
            available=False,
            supported=bool(dataset.wave_physics_capabilities.preheat_supported),
            method="wave-tracking-gated",
            candidate_metric_names=tuple(),
            scalar_summaries={},
            thresholds=thresholds,
            notes=("No reliable non-provisional compressive branch was available, so preheat diagnostics cannot anchor a target-entry time.", *capability_notes),
            warnings=tuple(),
        )
    target_region, auto_target_region, deepest_reached_region, incident_region, target_entry_time_s, interface_label, boundary_zone, target_selection_mode, target_notes = _select_target_region(
        dataset,
        wave_tracking=wave_tracking,
        interface_events=interface_events,
        target_region_id=target_region_id,
    )
    if target_region is None or target_entry_time_s is None:
        return PreheatSummary(
            available=False,
            supported=bool(dataset.wave_physics_capabilities.preheat_supported),
            method="branch-anchored-preheat",
            candidate_metric_names=tuple(),
            scalar_summaries={},
            target_selection_mode=target_selection_mode,
            target_region_id=(
                int(target_region["region_id"])
                if target_region is not None
                else (None if target_selection_mode != "user_selected" or target_region_id is None else int(target_region_id))
            ),
            auto_target_region_id=None if auto_target_region is None else int(auto_target_region["region_id"]),
            incident_region_id=None if incident_region is None else int(incident_region["region_id"]),
            deepest_reached_region_id=None if deepest_reached_region is None else int(deepest_reached_region["region_id"]),
            target_material_index=None if target_region is None or target_region["material_index"] is None else int(target_region["material_index"]),
            target_label=(
                str(target_region["label"])
                if target_region is not None
                else (None if target_selection_mode != "user_selected" or target_region_id is None else f"Region {int(target_region_id)}")
            ),
            auto_target_label=None if auto_target_region is None else str(auto_target_region["label"]),
            incident_region_label=None if incident_region is None else str(incident_region["label"]),
            deepest_reached_label=None if deepest_reached_region is None else str(deepest_reached_region["label"]),
            primary_branch_id=str(primary_branch.branch_id),
            primary_branch_support_class=str(primary_branch.support_class),
            primary_branch_significance=_finite_or_none(primary_branch.significance),
            thresholds=thresholds,
            notes=(*target_notes, *capability_notes),
            warnings=_support_warning(primary_branch),
        )

    time_s = np.asarray(dataset.time_s, dtype=np.float64)
    preheat_indices = np.flatnonzero(time_s < float(target_entry_time_s))
    if preheat_indices.size == 0:
        return PreheatSummary(
            available=False,
            supported=True,
            method="branch-anchored-preheat",
            candidate_metric_names=tuple(),
            scalar_summaries={"target_entry_time_s": float(target_entry_time_s)},
            target_selection_mode=target_selection_mode,
            target_region_id=int(target_region["region_id"]),
            auto_target_region_id=None if auto_target_region is None else int(auto_target_region["region_id"]),
            incident_region_id=None if incident_region is None else int(incident_region["region_id"]),
            deepest_reached_region_id=None if deepest_reached_region is None else int(deepest_reached_region["region_id"]),
            target_material_index=None if target_region["material_index"] is None else int(target_region["material_index"]),
            target_label=str(target_region["label"]),
            auto_target_label=None if auto_target_region is None else str(auto_target_region["label"]),
            incident_region_label=None if incident_region is None else str(incident_region["label"]),
            deepest_reached_label=None if deepest_reached_region is None else str(deepest_reached_region["label"]),
            primary_branch_id=str(primary_branch.branch_id),
            primary_branch_support_class=str(primary_branch.support_class),
            primary_branch_significance=_finite_or_none(primary_branch.significance),
            target_entry_interface_label=interface_label,
            target_entry_boundary_zone=boundary_zone,
            target_entry_time_s=float(target_entry_time_s),
            preheat_window_end_time_s=float(target_entry_time_s),
            target_zone_count=int(target_region["zone_stop"]) - int(target_region["zone_start"]),
            thresholds=thresholds,
            notes=(
                f"The primary compressive branch reaches {target_region['label']} at {target_entry_time_s:.6e} s, but there are no earlier snapshots to define a preheat window.",
                *target_notes,
                *capability_notes,
            ),
            warnings=_support_warning(primary_branch),
        )

    zone_slice = slice(int(target_region["zone_start"]), int(target_region["zone_stop"]))
    target_mask = np.zeros(int(dataset.summary["n_zones"]), dtype=bool)
    target_mask[zone_slice] = True
    density = np.asarray(dataset.density_g_cm3, dtype=np.float64)
    zone_width = np.asarray(dataset.zone_width_cm, dtype=np.float64)
    density0 = np.asarray(density[0, target_mask], dtype=np.float64)
    pressure_total = _total_pressure_array(dataset)
    pressure0 = None if pressure_total is None else np.asarray(pressure_total[0, target_mask], dtype=np.float64)
    pressure_eps = 1.0e-9
    if pressure0 is not None:
        finite_positive = pressure0[np.isfinite(pressure0) & (pressure0 > 0.0)]
        if finite_positive.size:
            pressure_eps = max(pressure_eps, float(np.nanmedian(finite_positive)) * 0.01)

    target_density_ratio = np.full((time_s.size, int(np.count_nonzero(target_mask))), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        target_density_ratio = np.divide(
            density[:, target_mask],
            density0[np.newaxis, :],
            out=target_density_ratio,
            where=np.isfinite(density0[np.newaxis, :]) & (density0[np.newaxis, :] > 0.0),
        )
    target_relative_pressure = None
    if pressure_total is not None and pressure0 is not None:
        target_relative_pressure = (pressure_total[:, target_mask] - pressure0[np.newaxis, :]) / (pressure0[np.newaxis, :] + pressure_eps)

    delta_te = np.asarray(dataset.temperature_e_ev[:, target_mask], dtype=np.float64) - np.asarray(dataset.temperature_e_ev[0, target_mask], dtype=np.float64)
    delta_z = np.asarray(dataset.mean_charge[:, target_mask], dtype=np.float64) - np.asarray(dataset.mean_charge[0, target_mask], dtype=np.float64)
    electron_energy = _optional_array(dataset, "electron_energy_j_g")
    delta_ee = None if electron_energy is None else np.asarray(electron_energy[:, target_mask], dtype=np.float64) - np.asarray(electron_energy[0, target_mask], dtype=np.float64)
    radiation_net_heating = _optional_array(dataset, "radiation_net_heating_j_g_s")
    laser_deposition = _optional_array(dataset, "laser_deposition_j_g_s")
    laser_source = _optional_array(dataset, "laser_source_j_g_s")
    temperature_r = _optional_array(dataset, "temperature_radiation_ev")
    pressure_i = _optional_array(dataset, "pressure_i_j_cm3")
    pressure_e = _optional_array(dataset, "pressure_e_j_cm3")
    pressure_r = _optional_array(dataset, "pressure_radiation_j_cm3")
    ion_energy = _optional_array(dataset, "ion_energy_j_g")
    radiation_energy = _optional_array(dataset, "radiation_energy_j_g")
    internal_energy = _internal_energy_array(dataset)

    modification = delta_te >= _PREHEAT_CONFIG.min_delta_temperature_e_ev
    modification |= delta_z >= _PREHEAT_CONFIG.min_delta_mean_charge
    if delta_ee is not None:
        modification |= delta_ee >= _PREHEAT_CONFIG.min_delta_electron_energy_j_g
    if radiation_net_heating is not None:
        modification |= np.asarray(radiation_net_heating[:, target_mask], dtype=np.float64) >= _PREHEAT_CONFIG.min_radiation_net_heating_j_g_s
    if laser_deposition is not None:
        modification |= np.asarray(laser_deposition[:, target_mask], dtype=np.float64) >= _PREHEAT_CONFIG.min_laser_deposition_j_g_s

    unshocked = target_density_ratio < _PREHEAT_CONFIG.max_density_ratio
    if target_relative_pressure is not None:
        unshocked &= target_relative_pressure < _PREHEAT_CONFIG.max_relative_pressure
    preheated_unshocked = modification & unshocked
    preheated_unshocked[~np.isfinite(target_density_ratio)] = False
    if target_relative_pressure is not None:
        preheated_unshocked[~np.isfinite(target_relative_pressure)] = False
    preheated_unshocked[np.setdiff1d(np.arange(time_s.size), preheat_indices), :] = False

    target_width = np.asarray(zone_width[:, target_mask], dtype=np.float64)
    target_areal_mass = np.asarray(density[:, target_mask] * target_width, dtype=np.float64)
    target_total_width = np.nansum(target_width, axis=1)
    target_total_mass = np.nansum(target_areal_mass, axis=1)
    affected_width = np.nansum(np.where(preheated_unshocked, target_width, 0.0), axis=1)
    affected_mass = np.nansum(np.where(preheated_unshocked, target_areal_mass, 0.0), axis=1)
    affected_fraction = np.divide(
        affected_width,
        target_total_width,
        out=np.full_like(affected_width, np.nan),
        where=np.isfinite(target_total_width) & (target_total_width > 0.0),
    )
    affected_mass_fraction = np.divide(
        affected_mass,
        target_total_mass,
        out=np.full_like(affected_mass, np.nan),
        where=np.isfinite(target_total_mass) & (target_total_mass > 0.0),
    )

    peak_snapshot = int(preheat_indices[0])
    finite_affected = np.asarray(np.where(np.isfinite(affected_mass_fraction), affected_mass_fraction, -np.inf), dtype=np.float64)
    if np.any(np.isfinite(finite_affected[preheat_indices])):
        peak_snapshot = int(preheat_indices[int(np.argmax(finite_affected[preheat_indices]))])
    peak_mask = np.asarray(preheated_unshocked[peak_snapshot], dtype=bool)
    representative_weights = np.asarray(target_areal_mass[peak_snapshot], dtype=np.float64)

    def _state_metric(key: str, label: str, unit: str, values: np.ndarray | None) -> PreheatStateMetric:
        representative = None
        maximum = None
        if values is not None:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 2:
                return PreheatStateMetric(key=key, label=label, unit=unit)
            if array.shape[1] == int(np.count_nonzero(target_mask)):
                target_values = array
            else:
                target_values = np.asarray(array[:, target_mask], dtype=np.float64)
            representative = _weighted_nanmean(np.asarray(target_values[peak_snapshot], dtype=np.float64)[peak_mask], representative_weights[peak_mask])
            maximum = _max_over_mask(np.asarray(target_values, dtype=np.float64), preheated_unshocked)
        return PreheatStateMetric(key=key, label=label, unit=unit, representative_value=representative, max_value=maximum)

    delta_electron_energy = None if electron_energy is None else np.asarray(electron_energy[:, target_mask], dtype=np.float64) - np.asarray(electron_energy[0, target_mask], dtype=np.float64)
    delta_ion_energy = None if ion_energy is None else np.asarray(ion_energy[:, target_mask], dtype=np.float64) - np.asarray(ion_energy[0, target_mask], dtype=np.float64)
    delta_radiation_energy = None if radiation_energy is None else np.asarray(radiation_energy[:, target_mask], dtype=np.float64) - np.asarray(radiation_energy[0, target_mask], dtype=np.float64)
    delta_internal_energy = None if internal_energy is None else np.asarray(internal_energy[:, target_mask], dtype=np.float64) - np.asarray(internal_energy[0, target_mask], dtype=np.float64)

    state_metrics = tuple(
        metric
        for metric in (
            _state_metric("temperature_e", "Electron temperature", "eV", np.asarray(dataset.temperature_e_ev, dtype=np.float64)),
            _state_metric("temperature_i", "Ion temperature", "eV", np.asarray(dataset.temperature_i_ev, dtype=np.float64)),
            _state_metric("temperature_radiation", "Radiation temperature", "eV", temperature_r),
            _state_metric("mean_charge", "Mean charge", "", np.asarray(dataset.mean_charge, dtype=np.float64)),
            _state_metric("pressure_total", "Total pressure", "J/cm^3", pressure_total),
            _state_metric("pressure_i", "Ion pressure", "J/cm^3", pressure_i),
            _state_metric("pressure_e", "Electron pressure", "J/cm^3", pressure_e),
            _state_metric("pressure_radiation", "Radiation pressure", "J/cm^3", pressure_r),
            _state_metric("delta_electron_energy", "Electron energy change", "J/g", delta_electron_energy),
            _state_metric("delta_ion_energy", "Ion energy change", "J/g", delta_ion_energy),
            _state_metric("delta_radiation_energy", "Radiation energy change", "J/g", delta_radiation_energy),
            _state_metric("delta_internal_energy", "Internal energy change", "J/g", delta_internal_energy),
        )
        if metric.representative_value is not None or metric.max_value is not None
    )

    delta_te_mean = np.full(time_s.shape, np.nan, dtype=np.float64)
    delta_ti_mean = np.full(time_s.shape, np.nan, dtype=np.float64)
    delta_tr_mean = np.full(time_s.shape, np.nan, dtype=np.float64)
    delta_z_mean = np.full(time_s.shape, np.nan, dtype=np.float64)
    radiation_mean = np.full(time_s.shape, np.nan, dtype=np.float64)
    laser_mean = np.full(time_s.shape, np.nan, dtype=np.float64)
    delta_te_peak = np.full(time_s.shape, np.nan, dtype=np.float64)
    delta_z_peak = np.full(time_s.shape, np.nan, dtype=np.float64)
    radiation_peak = np.full(time_s.shape, np.nan, dtype=np.float64)
    laser_peak = np.full(time_s.shape, np.nan, dtype=np.float64)
    for snapshot_index in preheat_indices:
        weights = np.asarray(target_areal_mass[snapshot_index], dtype=np.float64)
        te_mean = _weighted_nanmean(delta_te[snapshot_index], weights)
        delta_te_mean[snapshot_index] = np.nan if te_mean is None else float(te_mean)
        finite_te = np.asarray(delta_te[snapshot_index], dtype=np.float64)
        delta_te_peak[snapshot_index] = float(np.nanmax(finite_te)) if np.any(np.isfinite(finite_te)) else np.nan
        ti_mean = _weighted_nanmean(
            np.asarray(dataset.temperature_i_ev[snapshot_index, target_mask], dtype=np.float64)
            - np.asarray(dataset.temperature_i_ev[0, target_mask], dtype=np.float64),
            weights,
        )
        delta_ti_mean[snapshot_index] = np.nan if ti_mean is None else float(ti_mean)
        z_mean = _weighted_nanmean(delta_z[snapshot_index], weights)
        delta_z_mean[snapshot_index] = np.nan if z_mean is None else float(z_mean)
        finite_z = np.asarray(delta_z[snapshot_index], dtype=np.float64)
        delta_z_peak[snapshot_index] = float(np.nanmax(finite_z)) if np.any(np.isfinite(finite_z)) else np.nan
        if temperature_r is not None:
            tr_mean = _weighted_nanmean(
                np.asarray(temperature_r[snapshot_index, target_mask], dtype=np.float64)
                - np.asarray(temperature_r[0, target_mask], dtype=np.float64),
                weights,
            )
            delta_tr_mean[snapshot_index] = np.nan if tr_mean is None else float(tr_mean)
        if radiation_net_heating is not None:
            rad_mean = _weighted_nanmean(np.asarray(radiation_net_heating[snapshot_index, target_mask], dtype=np.float64), weights)
            radiation_mean[snapshot_index] = np.nan if rad_mean is None else float(rad_mean)
            finite_rad = np.asarray(radiation_net_heating[snapshot_index, target_mask], dtype=np.float64)
            radiation_peak[snapshot_index] = float(np.nanmax(finite_rad)) if np.any(np.isfinite(finite_rad)) else np.nan
        if laser_deposition is not None:
            laser_dep_mean = _weighted_nanmean(np.asarray(laser_deposition[snapshot_index, target_mask], dtype=np.float64), weights)
            laser_mean[snapshot_index] = np.nan if laser_dep_mean is None else float(laser_dep_mean)
            finite_laser = np.asarray(laser_deposition[snapshot_index, target_mask], dtype=np.float64)
            laser_peak[snapshot_index] = float(np.nanmax(finite_laser)) if np.any(np.isfinite(finite_laser)) else np.nan

    laser_cumulative, laser_dep_budget = _source_term_budget(
        dataset,
        attr_name="laser_deposition_j_g_s",
        density=density,
        zone_width=zone_width,
        target_mask=target_mask,
        time_indices=preheat_indices,
    )
    laser_source_cumulative, laser_source_budget = _source_term_budget(
        dataset,
        attr_name="laser_source_j_g_s",
        density=density,
        zone_width=zone_width,
        target_mask=target_mask,
        time_indices=preheat_indices,
    )
    radiation_cumulative, radiation_budget = _source_term_budget(
        dataset,
        attr_name="radiation_net_heating_j_g_s",
        density=density,
        zone_width=zone_width,
        target_mask=target_mask,
        time_indices=preheat_indices,
    )
    observed_internal = _energy_areal_density(internal_energy, density, zone_width, target_mask)
    observed_electron = _energy_areal_density(electron_energy, density, zone_width, target_mask)
    observed_ion = _energy_areal_density(ion_energy, density, zone_width, target_mask)
    observed_radiation = _energy_areal_density(radiation_energy, density, zone_width, target_mask)
    observed_internal_delta = None if observed_internal is None else float(observed_internal[int(preheat_indices[-1])] - observed_internal[int(preheat_indices[0])])
    observed_electron_delta = None if observed_electron is None else float(observed_electron[int(preheat_indices[-1])] - observed_electron[int(preheat_indices[0])])
    observed_ion_delta = None if observed_ion is None else float(observed_ion[int(preheat_indices[-1])] - observed_ion[int(preheat_indices[0])])
    observed_radiation_delta = None if observed_radiation is None else float(observed_radiation[int(preheat_indices[-1])] - observed_radiation[int(preheat_indices[0])])

    accounted_laser = laser_dep_budget if laser_dep_budget is not None else laser_source_budget
    residual_budget = None if observed_internal_delta is None else observed_internal_delta - float((accounted_laser or 0.0) + (radiation_budget or 0.0))
    contributor_strength = {
        "laser deposition": abs(float(accounted_laser)) if accounted_laser is not None else -1.0,
        "net radiation heating": abs(float(radiation_budget)) if radiation_budget is not None else -1.0,
        "residual processes": abs(float(residual_budget)) if residual_budget is not None else -1.0,
    }
    dominant_source = max(contributor_strength.items(), key=lambda item: item[1])[0] if any(value >= 0.0 for value in contributor_strength.values()) else None

    event_energy = None
    if interface_events is not None and interface_label is not None and boundary_zone is not None:
        target_event = _event_for_primary_branch(interface_events, primary_branch_id=str(primary_branch.branch_id), boundary_zone=boundary_zone)
        if target_event is not None and not bool(target_event.ambiguous):
            event_energy = _finite_or_none(target_event.transmitted_energy_j_cm2 or target_event.incident_energy_j_cm2)
    penalty_ratio = None
    if event_energy is not None and observed_internal_delta is not None and event_energy > 0.0:
        penalty_ratio = float(max(observed_internal_delta, 0.0) / event_energy)

    observed_reference = observed_internal_delta if observed_internal_delta not in (None, 0.0) else None
    budget_rows = (
        PreheatBudgetRow(
            key="laser_deposition",
            label="Laser deposition",
            unit="J/cm^2",
            integrated_value=_finite_or_none(laser_dep_budget),
            fraction_of_observed=(None if observed_reference in (None, 0.0) or laser_dep_budget is None else float(laser_dep_budget / observed_reference)),
            available=laser_dep_budget is not None,
            notes=("Preferred laser-source accounting channel when available.",),
        ),
        PreheatBudgetRow(
            key="laser_source",
            label="Laser source",
            unit="J/cm^2",
            integrated_value=_finite_or_none(laser_source_budget),
            fraction_of_observed=(None if observed_reference in (None, 0.0) or laser_source_budget is None else float(laser_source_budget / observed_reference)),
            available=laser_source_budget is not None,
            notes=("Auxiliary source-term channel; may overlap with laser deposition depending on the stored run format.",),
        ),
        PreheatBudgetRow(
            key="radiation_net_heating",
            label="Net radiation heating",
            unit="J/cm^2",
            integrated_value=_finite_or_none(radiation_budget),
            fraction_of_observed=(None if observed_reference in (None, 0.0) or radiation_budget is None else float(radiation_budget / observed_reference)),
            available=radiation_budget is not None,
        ),
        PreheatBudgetRow(
            key="observed_electron_delta",
            label="Observed electron-energy change",
            unit="J/cm^2",
            integrated_value=_finite_or_none(observed_electron_delta),
            fraction_of_observed=(None if observed_reference in (None, 0.0) or observed_electron_delta is None else float(observed_electron_delta / observed_reference)),
            available=observed_electron_delta is not None,
        ),
        PreheatBudgetRow(
            key="observed_ion_delta",
            label="Observed ion-energy change",
            unit="J/cm^2",
            integrated_value=_finite_or_none(observed_ion_delta),
            fraction_of_observed=(None if observed_reference in (None, 0.0) or observed_ion_delta is None else float(observed_ion_delta / observed_reference)),
            available=observed_ion_delta is not None,
        ),
        PreheatBudgetRow(
            key="observed_radiation_delta",
            label="Observed radiation-energy change",
            unit="J/cm^2",
            integrated_value=_finite_or_none(observed_radiation_delta),
            fraction_of_observed=(None if observed_reference in (None, 0.0) or observed_radiation_delta is None else float(observed_radiation_delta / observed_reference)),
            available=observed_radiation_delta is not None,
        ),
        PreheatBudgetRow(
            key="observed_internal_delta",
            label="Observed internal-energy change",
            unit="J/cm^2",
            integrated_value=_finite_or_none(observed_internal_delta),
            fraction_of_observed=1.0 if observed_reference not in (None, 0.0) else None,
            available=observed_internal_delta is not None,
        ),
        PreheatBudgetRow(
            key="residual",
            label="Residual",
            unit="J/cm^2",
            integrated_value=_finite_or_none(residual_budget),
            fraction_of_observed=(None if observed_reference in (None, 0.0) or residual_budget is None else float(residual_budget / observed_reference)),
            available=residual_budget is not None,
            notes=(
                "Residual captures pdV work, advection, conduction-like effects, and any model components not separately exposed in the stored fields.",
            ),
        ),
    )

    onset_markers = (
        PreheatOnsetMarker(
            key="temperature_e",
            label="Electron-temperature onset",
            threshold_value=_PREHEAT_CONFIG.min_delta_temperature_e_ev,
            first_time_s=_first_marker_time(delta_te_peak, time_s, _PREHEAT_CONFIG.min_delta_temperature_e_ev)[0],
            observed_value=_first_marker_time(delta_te_peak, time_s, _PREHEAT_CONFIG.min_delta_temperature_e_ev)[1],
            unit="eV",
            available=True,
        ),
        PreheatOnsetMarker(
            key="mean_charge",
            label="Ionization onset",
            threshold_value=_PREHEAT_CONFIG.min_delta_mean_charge,
            first_time_s=_first_marker_time(delta_z_peak, time_s, _PREHEAT_CONFIG.min_delta_mean_charge)[0],
            observed_value=_first_marker_time(delta_z_peak, time_s, _PREHEAT_CONFIG.min_delta_mean_charge)[1],
            unit="",
            available=True,
        ),
        PreheatOnsetMarker(
            key="radiation",
            label="Radiation-heating onset",
            threshold_value=_PREHEAT_CONFIG.min_radiation_net_heating_j_g_s,
            first_time_s=(None if radiation_net_heating is None else _first_marker_time(radiation_peak, time_s, _PREHEAT_CONFIG.min_radiation_net_heating_j_g_s)[0]),
            observed_value=(None if radiation_net_heating is None else _first_marker_time(radiation_peak, time_s, _PREHEAT_CONFIG.min_radiation_net_heating_j_g_s)[1]),
            unit="J/g/s",
            available=radiation_net_heating is not None,
            notes=(("No radiation-net-heating field is available." if radiation_net_heating is None else ""),),
        ),
        PreheatOnsetMarker(
            key="laser",
            label="Laser-deposition onset",
            threshold_value=_PREHEAT_CONFIG.min_laser_deposition_j_g_s,
            first_time_s=(None if laser_deposition is None else _first_marker_time(laser_peak, time_s, _PREHEAT_CONFIG.min_laser_deposition_j_g_s)[0]),
            observed_value=(None if laser_deposition is None else _first_marker_time(laser_peak, time_s, _PREHEAT_CONFIG.min_laser_deposition_j_g_s)[1]),
            unit="J/g/s",
            available=laser_deposition is not None,
            notes=(("No laser-deposition field is available." if laser_deposition is None else ""),),
        ),
    )
    onset_markers = tuple(
        PreheatOnsetMarker(
            key=item.key,
            label=item.label,
            threshold_value=item.threshold_value,
            first_time_s=item.first_time_s,
            observed_value=item.observed_value,
            unit=item.unit,
            available=item.available,
            notes=tuple(note for note in item.notes if note),
        )
        for item in onset_markers
    )

    candidate_metric_names = tuple(
        metric
        for metric in (
            "temperature_e",
            "mean_charge",
            "electron_energy" if electron_energy is not None else None,
            "radiation_net_heating" if radiation_net_heating is not None else None,
            "laser_deposition" if laser_deposition is not None else None,
            "laser_source" if laser_source is not None else None,
        )
        if metric is not None
    )
    available_fields = tuple(
        name
        for name, available in (
            ("temperature_e", True),
            ("temperature_i", True),
            ("temperature_radiation", temperature_r is not None),
            ("mean_charge", True),
            ("pressure_total", pressure_total is not None),
            ("pressure_i", pressure_i is not None),
            ("pressure_e", pressure_e is not None),
            ("pressure_radiation", pressure_r is not None),
            ("electron_energy", electron_energy is not None),
            ("ion_energy", ion_energy is not None),
            ("radiation_energy", radiation_energy is not None),
            ("radiation_net_heating", radiation_net_heating is not None),
            ("laser_deposition", laser_deposition is not None),
            ("laser_source", laser_source is not None),
        )
        if available
    )
    missing_fields = tuple(
        name
        for name, available in (
            ("temperature_radiation", temperature_r is not None),
            ("pressure_total", pressure_total is not None),
            ("pressure_i", pressure_i is not None),
            ("pressure_e", pressure_e is not None),
            ("pressure_radiation", pressure_r is not None),
            ("electron_energy", electron_energy is not None),
            ("ion_energy", ion_energy is not None),
            ("radiation_energy", radiation_energy is not None),
            ("radiation_net_heating", radiation_net_heating is not None),
            ("laser_deposition", laser_deposition is not None),
            ("laser_source", laser_source is not None),
        )
        if not available
    )

    target_zone_indices = np.asarray(np.arange(int(target_region["zone_start"]), int(target_region["zone_stop"])), dtype=np.float64) + 1.0
    target_static_x_cm = np.asarray(dataset.static_x_cm[target_mask], dtype=np.float64) if dataset.static_x_cm is not None else None
    target_dynamic_coordinate_cm = (
        None
        if dataset.radius_cm is None
        else np.asarray(dataset.radius_cm[:, target_mask], dtype=np.float64)
    )

    observed_internal_trace = np.full(time_s.shape, np.nan, dtype=np.float64)
    if observed_internal is not None:
        observed_internal_trace = observed_internal - float(observed_internal[int(preheat_indices[0])])
    time_plots = (
        DerivedPlotBundle(
            key="preheat_temperature",
            title="Target preheat state before primary-shock entry",
            x_label="Time [s]",
            y_label="Temperature rise [eV]",
            x_values=np.asarray(time_s, dtype=np.float64),
            y_series=(
                np.asarray(delta_te_mean, dtype=np.float64),
                np.asarray(delta_ti_mean, dtype=np.float64),
                np.asarray(delta_tr_mean, dtype=np.float64),
            ),
            curve_names=("Delta Te", "Delta Ti", "Delta Tr"),
        ),
        DerivedPlotBundle(
            key="preheat_extent",
            title="Preheat extent before target entry",
            x_label="Time [s]",
            y_label="Affected fraction",
            x_values=np.asarray(time_s, dtype=np.float64),
            y_series=(
                np.asarray(affected_fraction, dtype=np.float64),
                np.asarray(affected_mass_fraction, dtype=np.float64),
            ),
            curve_names=("Thickness fraction", "Areal-mass fraction"),
        ),
        DerivedPlotBundle(
            key="preheat_budget",
            title="Target-integrated preheat budget before shock entry",
            x_label="Time [s]",
            y_label="Areal budget [J/cm^2]",
            x_values=np.asarray(time_s, dtype=np.float64),
            y_series=(
                np.full(time_s.shape, np.nan, dtype=np.float64) if laser_cumulative is None else np.asarray(laser_cumulative, dtype=np.float64),
                np.full(time_s.shape, np.nan, dtype=np.float64) if radiation_cumulative is None else np.asarray(radiation_cumulative, dtype=np.float64),
                np.asarray(observed_internal_trace, dtype=np.float64),
            ),
            curve_names=("Laser deposition", "Net radiation heating", "Observed internal delta"),
        ),
    )

    profile_fields: list[PreheatProfileField] = [
        PreheatProfileField(
            key="delta_temperature_e",
            label="Delta Te",
            unit="eV",
            values=np.asarray(delta_te, dtype=np.float64),
            notes=("Electron-temperature rise relative to the initial target state.",),
        ),
        PreheatProfileField(
            key="density_ratio",
            label="rho / rho0",
            unit="",
            values=np.asarray(target_density_ratio, dtype=np.float64),
            notes=("Compression proxy; values near 1 indicate the target is still largely unshocked.",),
        ),
        PreheatProfileField(
            key="delta_mean_charge",
            label="Delta Zbar",
            unit="",
            values=np.asarray(delta_z, dtype=np.float64),
        ),
        PreheatProfileField(
            key="preheat_mask",
            label="Preheated & unshocked mask",
            unit="fraction",
            values=np.asarray(preheated_unshocked, dtype=np.float64),
            notes=("1 means the cell met the configured preheated-but-unshocked criteria for that snapshot.",),
        ),
    ]
    if pressure_total is not None:
        profile_fields.append(
            PreheatProfileField(
                key="pressure_total",
                label="Total pressure",
                unit="J/cm^3",
                values=np.asarray(pressure_total[:, target_mask], dtype=np.float64),
            )
        )
    if target_relative_pressure is not None:
        profile_fields.append(
            PreheatProfileField(
                key="relative_pressure",
                label="(P - P0) / (P0 + eps)",
                unit="",
                values=np.asarray(target_relative_pressure, dtype=np.float64),
            )
        )
    if temperature_r is not None:
        profile_fields.append(
            PreheatProfileField(
                key="delta_temperature_radiation",
                label="Delta Tr",
                unit="eV",
                values=np.asarray(temperature_r[:, target_mask], dtype=np.float64) - np.asarray(temperature_r[0, target_mask], dtype=np.float64),
            )
        )
    if radiation_net_heating is not None:
        profile_fields.append(
            PreheatProfileField(
                key="radiation_net_heating",
                label="Net radiation heating",
                unit="J/g/s",
                values=np.asarray(radiation_net_heating[:, target_mask], dtype=np.float64),
            )
        )
    if laser_deposition is not None:
        profile_fields.append(
            PreheatProfileField(
                key="laser_deposition",
                label="Laser deposition",
                unit="J/g/s",
                values=np.asarray(laser_deposition[:, target_mask], dtype=np.float64),
            )
        )
    if laser_source is not None:
        profile_fields.append(
            PreheatProfileField(
                key="laser_source",
                label="Laser source",
                unit="J/g/s",
                values=np.asarray(laser_source[:, target_mask], dtype=np.float64),
            )
        )

    snapshot_scalar_series: dict[str, np.ndarray] = {
        "affected_depth_cm": np.asarray(affected_width, dtype=np.float64),
        "affected_thickness_fraction": np.asarray(affected_fraction, dtype=np.float64),
        "affected_areal_mass_fraction": np.asarray(affected_mass_fraction, dtype=np.float64),
        "delta_temperature_e_mean": np.asarray(delta_te_mean, dtype=np.float64),
        "delta_temperature_e_peak": np.asarray(delta_te_peak, dtype=np.float64),
        "delta_temperature_i_mean": np.asarray(delta_ti_mean, dtype=np.float64),
        "delta_temperature_r_mean": np.asarray(delta_tr_mean, dtype=np.float64),
        "delta_mean_charge_peak": np.asarray(delta_z_peak, dtype=np.float64),
    }
    if pressure_total is not None:
        pressure_peak = np.full(time_s.shape, np.nan, dtype=np.float64)
        for snapshot_index in range(time_s.size):
            finite_values = np.asarray(pressure_total[snapshot_index, target_mask], dtype=np.float64)
            pressure_peak[snapshot_index] = float(np.nanmax(finite_values)) if np.any(np.isfinite(finite_values)) else np.nan
        snapshot_scalar_series["pressure_total_peak_j_cm3"] = pressure_peak
    if radiation_net_heating is not None:
        snapshot_scalar_series["radiation_peak_j_g_s"] = np.asarray(radiation_peak, dtype=np.float64)
    if laser_deposition is not None:
        snapshot_scalar_series["laser_peak_j_g_s"] = np.asarray(laser_peak, dtype=np.float64)

    peak_delta_te = _finite_or_none(np.nanmax(delta_te[preheat_indices])) if preheat_indices.size else None
    affected_fraction_peak = _finite_or_none(np.nanmax(affected_fraction[preheat_indices])) if preheat_indices.size else None
    severity = "negligible"
    if (affected_fraction_peak or 0.0) >= _PREHEAT_CONFIG.severity_severe_fraction or (peak_delta_te or 0.0) >= _PREHEAT_CONFIG.severity_severe_delta_te_ev:
        severity = "severe"
    elif (affected_fraction_peak or 0.0) >= _PREHEAT_CONFIG.severity_moderate_fraction or (peak_delta_te or 0.0) >= _PREHEAT_CONFIG.severity_moderate_delta_te_ev:
        severity = "moderate"
    elif (affected_fraction_peak or 0.0) > 0.0 or (peak_delta_te or 0.0) > 0.0:
        severity = "mild"

    scalar_summaries = {
        "target_entry_time_s": float(target_entry_time_s),
        "affected_depth_cm": _finite_or_none(np.nanmax(affected_width[preheat_indices])) if preheat_indices.size else None,
        "affected_thickness_fraction": affected_fraction_peak,
        "affected_areal_mass_fraction": _finite_or_none(np.nanmax(affected_mass_fraction[preheat_indices])) if preheat_indices.size else None,
        "max_temperature_e_ev": peak_delta_te,
        "max_mean_charge_delta": _finite_or_none(np.nanmax(delta_z[preheat_indices])) if preheat_indices.size else None,
        "preheat_penalty_ratio": penalty_ratio,
    }

    notes = [
        f"Preheat is defined relative to the primary compressive branch entering the selected region of interest {target_region['label']} at {target_entry_time_s:.6e} s.",
        f"Incident-side region={('-' if incident_region is None else incident_region['label'])}; deepest reached region={('-' if deepest_reached_region is None else deepest_reached_region['label'])}; automatic target guess={('-' if auto_target_region is None else auto_target_region['label'])}.",
        (
            f"Target selection mode={target_selection_mode}; using user-selected region {target_region['label']}."
            if target_selection_mode == "user_selected"
            else f"Target selection mode=auto; using heuristic target guess {target_region['label']}."
        ),
        f"Severity={severity}; dominant source={dominant_source or 'unresolved'}; window end={target_entry_time_s:.6e} s.",
        *target_notes,
        *capability_notes,
    ]
    if boundary_zone is not None:
        notes.append(f"Target-entry boundary zone: {boundary_zone}.")
    if penalty_ratio is None:
        notes.append("Preheat penalty ratio is unavailable because no reliable transmitted-loading denominator was available for the target-entry event.")
    if residual_budget is not None:
        notes.append("Residual is explicit by design; it may contain pdV work, advection, conduction-like transport, and model components not separately exposed in the stored fields.")
    if missing_fields:
        notes.append("Missing preheat-support fields: " + ", ".join(missing_fields))

    return PreheatSummary(
        available=bool(np.any(preheated_unshocked[preheat_indices])),
        supported=True,
        method="branch-anchored-target-preheat",
        candidate_metric_names=candidate_metric_names,
        scalar_summaries=scalar_summaries,
        target_selection_mode=target_selection_mode,
        target_region_id=int(target_region["region_id"]),
        auto_target_region_id=None if auto_target_region is None else int(auto_target_region["region_id"]),
        incident_region_id=None if incident_region is None else int(incident_region["region_id"]),
        deepest_reached_region_id=None if deepest_reached_region is None else int(deepest_reached_region["region_id"]),
        target_material_index=None if target_region["material_index"] is None else int(target_region["material_index"]),
        target_label=str(target_region["label"]),
        auto_target_label=None if auto_target_region is None else str(auto_target_region["label"]),
        incident_region_label=None if incident_region is None else str(incident_region["label"]),
        deepest_reached_label=None if deepest_reached_region is None else str(deepest_reached_region["label"]),
        primary_branch_id=str(primary_branch.branch_id),
        primary_branch_support_class=str(primary_branch.support_class),
        primary_branch_significance=_finite_or_none(primary_branch.significance),
        target_entry_interface_label=interface_label,
        target_entry_boundary_zone=boundary_zone,
        target_entry_time_s=float(target_entry_time_s),
        preheat_window_end_time_s=float(target_entry_time_s),
        target_zone_count=int(np.count_nonzero(target_mask)),
        available_fields=available_fields,
        missing_fields=missing_fields,
        thresholds=thresholds,
        state_metrics=state_metrics,
        budget_rows=tuple(row for row in budget_rows if row.available),
        onset_markers=onset_markers,
        time_plots=time_plots,
        profile_plots=tuple(),
        snapshot_indices=np.asarray(np.arange(time_s.size), dtype=np.int32),
        latest_pre_entry_snapshot_index=(None if preheat_indices.size == 0 else int(preheat_indices[-1])),
        peak_snapshot_index=int(peak_snapshot),
        target_zone_indices=target_zone_indices,
        target_static_x_cm=target_static_x_cm,
        target_dynamic_coordinate_cm=target_dynamic_coordinate_cm,
        profile_fields=tuple(profile_fields),
        snapshot_scalar_series=snapshot_scalar_series,
        affected_depth_cm=scalar_summaries["affected_depth_cm"],
        affected_thickness_fraction=scalar_summaries["affected_thickness_fraction"],
        affected_areal_mass_fraction=scalar_summaries["affected_areal_mass_fraction"],
        severity_label=severity,
        preheat_penalty_ratio=penalty_ratio,
        dominant_source=dominant_source,
        notes=tuple(notes),
        warnings=_support_warning(primary_branch),
    )
