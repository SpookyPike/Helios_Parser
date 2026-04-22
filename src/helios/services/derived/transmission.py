"""Quick beam-transmission / extinction estimates for HELIOS Derived mode."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import math
from typing import Callable

import numpy as np

from helios.instrumentation import increment_counter, timed_block
from helios.runtime import RunContext
from helios.services.constants import HBAR_EV_S, THOMSON_CROSS_SECTION_CM2
from helios.services.derived.common import picosecond_drive_warning
from helios.services.derived.models import (
    DerivedPlotBundle,
    DerivedRunData,
    DerivedWarning,
    TransmissionColdMaterialBudget,
    TransmissionColdRefinement,
    TransmissionPartitionSummary,
    TransmissionRegimeSummary,
    TransmissionRegionBudget,
    TransmissionResult,
)
from helios.services.derived.plasmon import coulomb_logarithm_ei, electron_plasma_frequency_rad_s
from helios.services.derived.selection import (
    AnalysisStateCache,
    build_analysis_mask,
    cached_time_series_payload,
    cylindrical_path_note,
    path_geometry_summary,
    path_length_cm,
    shared_time_series_selection_state,
)
from helios.services.derived.xcom_hook import (
    build_cold_attenuation_request,
    cold_attenuation_cache_key,
    describe_optional_cold_backend,
    load_precomputed_cold_backend,
    load_optional_cold_backend,
    normalize_material_key,
    persistent_cold_attenuation_cache,
)

if TYPE_CHECKING:
    from helios.services.derived.analysis import DerivedAnalysisParameters
    from helios.services.derived.models import AnalysisGeometryMetadata


@dataclass(frozen=True, slots=True)
class _TransmissionAutoThresholds:
    xcom_max_temperature_ev: float = 12.0
    xcom_transition_temperature_ev: float = 18.0
    xcom_min_density_ratio: float = 0.72
    xcom_transition_density_ratio: float = 0.55
    plasma_min_temperature_ev: float = 25.0
    plasma_transition_temperature_ev: float = 15.0
    plasma_max_density_ratio: float = 0.55
    plasma_transition_density_ratio: float = 0.75
    optically_negligible_tau: float = 1.0e-8


TRANSMISSION_AUTO_THRESHOLDS = _TransmissionAutoThresholds()


def _zone_region_dense_index(region_ids: np.ndarray, zone_region_id: np.ndarray) -> np.ndarray:
    dense = np.searchsorted(region_ids, zone_region_id)
    valid = (dense >= 0) & (dense < region_ids.size)
    if np.any(valid):
        valid &= region_ids[dense] == zone_region_id
    if not np.all(valid):
        raise ValueError("Zone-to-region mapping contains region identifiers not present in the region table.")
    return dense.astype(np.int32, copy=False)


def _append_warning_once(warnings: tuple[DerivedWarning, ...], warning: DerivedWarning) -> tuple[DerivedWarning, ...]:
    key = (warning.source, warning.message, warning.severity)
    for existing in warnings:
        if (existing.source, existing.message, existing.severity) == key:
            return warnings
    return (*warnings, warning)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return None
    ordered = np.argsort(values[finite])
    sorted_values = values[finite][ordered]
    sorted_weights = weights[finite][ordered]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = min(max(index, 0), sorted_values.size - 1)
    return float(sorted_values[index])


def _cold_refinement_status_message(
    *,
    backend_message: str,
    applicability: str,
    source: str,
    energy_kev: float | None = None,
    cached: bool = False,
    interpolation_note: str | None = None,
    quantity_type: str | None = None,
) -> str:
    if source in {"live_xcom_backend", "precomputed_xcom_table"} and energy_kev is not None:
        source_label = "live XCOM backend" if source == "live_xcom_backend" else "precomputed XCOM table"
        cache_text = " (cached)" if cached else ""
        quantity_text = ""
        if source == "precomputed_xcom_table" and quantity_type == "mu_rho_cm2_g":
            quantity_text = " Uses mu/rho [cm^2/g] with tau=(mu/rho)*Sigma."
        suffix = f" {interpolation_note}" if interpolation_note else ""
        return f"XCOM refinement active at {float(energy_kev):.4g} keV via {source_label}{cache_text}.{quantity_text}{suffix}".strip()
    if applicability == "not_recommended":
        return "XCOM refinement was not run because the selected subset looks too ionized for a cold-opacity baseline."
    if applicability == "borderline":
        return f"{backend_message} The selected subset is borderline for a cold-opacity approximation."
    if applicability == "recommended":
        return f"{backend_message} The selected subset looks cold/weakly ionized enough for an optional XCOM baseline."
    return backend_message


def _density_reference_profile(dataset: DerivedRunData) -> np.ndarray:
    initial_density = np.asarray(dataset.zone_initial_density_g_cm3, dtype=np.float64)
    fallback_density = np.asarray(dataset.density_g_cm3[0], dtype=np.float64)
    if initial_density.shape != fallback_density.shape:
        initial_density = np.resize(initial_density, fallback_density.shape)
    reference = np.where(np.isfinite(initial_density) & (initial_density > 0.0), initial_density, fallback_density)
    return np.where(np.isfinite(reference) & (reference > 0.0), reference, 1.0)


def _safe_density_ratio(current_density_g_cm3: np.ndarray, reference_density_g_cm3: np.ndarray) -> np.ndarray:
    current = np.asarray(current_density_g_cm3, dtype=np.float64)
    reference = np.asarray(reference_density_g_cm3, dtype=np.float64)
    safe_reference = np.where(np.isfinite(reference) & (reference > 0.0), reference, 1.0)
    ratio = np.where(np.isfinite(current), current / safe_reference, np.nan)
    return np.asarray(ratio, dtype=np.float64)


def _cold_zone_metadata(request) -> dict[int, tuple[str | None, str, str, str | None]]:
    metadata: dict[int, tuple[str | None, str, str, str | None]] = {}
    for zone in request.zones:
        metadata[int(zone.zone_index) - 1] = (
            None if zone.material_label is None else str(zone.material_label).strip() or None,
            str(zone.material_display_label or zone.material_label or f"material {int(zone.material_id)}").strip(),
            str(zone.material_resolution_status or "unresolved"),
            None if zone.material_canonical_key is None else str(zone.material_canonical_key).strip() or None,
        )
    return metadata


def _plasma_like_zone(te_ev: float, density_ratio: float) -> bool:
    thresholds = TRANSMISSION_AUTO_THRESHOLDS
    if not math.isfinite(te_ev) or not math.isfinite(density_ratio):
        return False
    if te_ev >= thresholds.plasma_min_temperature_ev:
        return True
    if density_ratio <= thresholds.plasma_max_density_ratio:
        return True
    if te_ev >= thresholds.plasma_transition_temperature_ev and density_ratio <= thresholds.plasma_transition_density_ratio:
        return True
    return False


def _xcom_like_zone(te_ev: float, density_ratio: float, *, material_resolved: bool) -> bool:
    thresholds = TRANSMISSION_AUTO_THRESHOLDS
    if not material_resolved or not math.isfinite(te_ev) or not math.isfinite(density_ratio):
        return False
    if te_ev <= thresholds.xcom_max_temperature_ev and density_ratio >= thresholds.xcom_min_density_ratio:
        return True
    if te_ev <= thresholds.xcom_transition_temperature_ev and density_ratio >= thresholds.xcom_transition_density_ratio:
        return True
    return False


def _assess_xcom_applicability(dataset: DerivedRunData, request) -> tuple[str, float | None, float | None, float | None, str]:
    if not request.zones:
        return "indeterminate", None, None, None, "No active zones are available for XCOM refinement."
    zone_indices = np.asarray([int(zone.zone_index) - 1 for zone in request.zones], dtype=np.int32)
    weights = np.asarray([float(zone.path_length_cm) for zone in request.zones], dtype=np.float64)
    finite_zone_indices = zone_indices[(zone_indices >= 0) & (zone_indices < dataset.zone_material_index.size)]
    if finite_zone_indices.size == 0:
        return "indeterminate", None, None, None, "No valid zone indices are available for XCOM refinement."
    weights = weights[: finite_zone_indices.size]
    te_snapshot = np.asarray(dataset.temperature_e_ev[int(request.snapshot_index)], dtype=np.float64)[finite_zone_indices]
    zbar_snapshot = np.asarray(dataset.mean_charge[int(request.snapshot_index)], dtype=np.float64)[finite_zone_indices]
    density_snapshot = np.asarray(dataset.density_g_cm3[int(request.snapshot_index)], dtype=np.float64)[finite_zone_indices]
    density_reference = _density_reference_profile(dataset)[finite_zone_indices]
    density_ratio = _safe_density_ratio(density_snapshot, density_reference)
    valid = np.isfinite(te_snapshot) & np.isfinite(density_ratio) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return "indeterminate", None, None, None, "Electron temperature or condensed-density reference is unavailable for XCOM applicability screening."
    te_valid = te_snapshot[valid]
    zbar_valid = zbar_snapshot[valid]
    density_ratio_valid = density_ratio[valid]
    weights_valid = weights[valid]
    total_weight = float(np.sum(weights_valid))
    if total_weight <= 0.0:
        return "indeterminate", None, None, None, "LOS weights are not positive for the active subset."
    thresholds = TRANSMISSION_AUTO_THRESHOLDS
    cold_mask = (te_valid <= thresholds.xcom_transition_temperature_ev) & (density_ratio_valid >= thresholds.xcom_transition_density_ratio)
    hot_mask = (te_valid >= thresholds.plasma_min_temperature_ev) | (density_ratio_valid <= thresholds.plasma_max_density_ratio)
    cold_fraction = float(np.sum(weights_valid[cold_mask]) / total_weight)
    hot_fraction = float(np.sum(weights_valid[hot_mask]) / total_weight)
    path_weighted_te = _weighted_median(te_valid, weights_valid)
    path_weighted_zbar = _weighted_median(zbar_valid, weights_valid)
    path_weighted_density_ratio = _weighted_median(density_ratio_valid, weights_valid)
    if cold_fraction >= 0.75 and (path_weighted_te is not None and path_weighted_te <= thresholds.xcom_transition_temperature_ev) and (path_weighted_density_ratio is not None and path_weighted_density_ratio >= thresholds.xcom_transition_density_ratio):
        return "recommended", cold_fraction, path_weighted_te, path_weighted_zbar, "The active subset is mostly cold and condensed enough for an XCOM-like cold-material baseline."
    if hot_fraction > 0.5:
        return "not_recommended", cold_fraction, path_weighted_te, path_weighted_zbar, "The active subset looks plasma-like or expanded over most of the LOS; XCOM is not a good global cold-opacity baseline there."
    return "borderline", cold_fraction, path_weighted_te, path_weighted_zbar, "The active subset mixes condensed and plasma-like zones; XCOM remains local and conditional rather than a global truth."


def _baseline_cold_refinement() -> TransmissionColdRefinement:
    backend = describe_optional_cold_backend()
    return TransmissionColdRefinement(
        backend_status=str(backend.status),
        applicability="indeterminate",
        message=_cold_refinement_status_message(backend_message=str(backend.message), applicability="indeterminate", source="baseline"),
        backend_name=backend.backend_name,
        backend_available=bool(backend.available),
        backend_fingerprint=backend.backend_fingerprint,
        source="baseline",
    )


TRANSMISSION_MODE_THOMSON = "thomson"
TRANSMISSION_MODE_FREE_FREE = "free_free"
TRANSMISSION_MODE_FREE_FREE_THOMSON = "free_free_thomson"
TRANSMISSION_MODE_XCOM = "xcom"
TRANSMISSION_MODE_AUTO = "auto_hybrid"
_TRANSMISSION_MODE_VALUES = {
    TRANSMISSION_MODE_THOMSON,
    TRANSMISSION_MODE_FREE_FREE,
    TRANSMISSION_MODE_FREE_FREE_THOMSON,
    TRANSMISSION_MODE_XCOM,
    TRANSMISSION_MODE_AUTO,
}
_REGIME_XCOM = "xcom"
_REGIME_PLASMA = "free_free_thomson"
_REGIME_THOMSON = "thomson"
_REGIME_THOMSON_FALLBACK = "thomson_fallback"
_REGIME_FREE_FREE = "free_free"


def normalize_transmission_mode(value: str | None) -> str:
    normalized = str(value or TRANSMISSION_MODE_THOMSON).strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "auto": TRANSMISSION_MODE_AUTO,
        "hybrid": TRANSMISSION_MODE_AUTO,
        "auto_hybrid": TRANSMISSION_MODE_AUTO,
        "thomson": TRANSMISSION_MODE_THOMSON,
        "free_free": TRANSMISSION_MODE_FREE_FREE,
        "freefree": TRANSMISSION_MODE_FREE_FREE,
        "free_free_thomson": TRANSMISSION_MODE_FREE_FREE_THOMSON,
        "free_free__thomson": TRANSMISSION_MODE_FREE_FREE_THOMSON,
        "xcom": TRANSMISSION_MODE_XCOM,
    }
    mode = aliases.get(normalized, normalized)
    return mode if mode in _TRANSMISSION_MODE_VALUES else TRANSMISSION_MODE_THOMSON


def transmission_mode_label(mode: str) -> str:
    labels = {
        TRANSMISSION_MODE_AUTO: "Auto hybrid",
        TRANSMISSION_MODE_THOMSON: "Thomson",
        TRANSMISSION_MODE_FREE_FREE: "Free-free",
        TRANSMISSION_MODE_FREE_FREE_THOMSON: "Free-free + Thomson",
        TRANSMISSION_MODE_XCOM: "XCOM",
    }
    return labels.get(normalize_transmission_mode(mode), "Thomson")


def _regime_display_label(regime: str) -> str:
    labels = {
        _REGIME_XCOM: "XCOM",
        _REGIME_PLASMA: "Free-free + Thomson",
        _REGIME_FREE_FREE: "Free-free",
        _REGIME_THOMSON: "Thomson",
        _REGIME_THOMSON_FALLBACK: "Thomson fallback",
    }
    return labels.get(str(regime), str(regime).replace("_", " ").title())


def _significant_regime_fraction_keys(
    fractions: dict[str, float],
    *,
    threshold: float = 0.05,
) -> set[str]:
    return {
        str(key)
        for key, value in fractions.items()
        if math.isfinite(float(value)) and float(value) >= float(threshold)
    }


def _append_warning_message(warnings: tuple[DerivedWarning, ...], message: str, *, severity: str) -> tuple[DerivedWarning, ...]:
    return _append_warning_once(warnings, DerivedWarning("transmission", message, severity=severity))


def _filter_mode_specific_base_warnings(warnings: tuple[DerivedWarning, ...], *, mode: str) -> tuple[DerivedWarning, ...]:
    if normalize_transmission_mode(mode) == TRANSMISSION_MODE_THOMSON:
        return tuple(warnings)
    filtered: list[DerivedWarning] = []
    for warning in warnings:
        message = str(warning.message)
        if message.startswith("Transmission is a Thomson-only quick-look estimate;"):
            continue
        filtered.append(warning)
    return tuple(filtered)


def _mode_supports_energy(mode: str) -> bool:
    return normalize_transmission_mode(mode) in {
        TRANSMISSION_MODE_FREE_FREE,
        TRANSMISSION_MODE_FREE_FREE_THOMSON,
        TRANSMISSION_MODE_XCOM,
        TRANSMISSION_MODE_AUTO,
    }


def _cold_zone_labels(request) -> dict[int, str | None]:
    return {
        int(zone.zone_index) - 1: (
            str(zone.material_canonical_key).strip()
            if zone.material_canonical_key is not None
            else (str(zone.material_label).strip() if zone.material_label is not None else None)
        )
        for zone in request.zones
    }


def _material_resolution_for_request(request) -> tuple[dict[str, float], tuple[str, ...], tuple[str, ...]]:
    grouped: dict[str, float] = {}
    resolved: set[str] = set()
    unresolved: set[str] = set()
    for zone in request.zones:
        label = str(zone.material_canonical_key or zone.material_label or "").strip()
        display = str(zone.material_display_label or zone.material_label or f"material {int(zone.material_id)}").strip()
        status = str(zone.material_resolution_status or "unresolved")
        if not label:
            unresolved.add(f"{display} [{status}]")
            continue
        grouped[label] = float(grouped.get(label, 0.0) + float(zone.density_g_cm3) * float(zone.path_length_cm))
        resolved.add(f"{display} [{status}]")
    return grouped, tuple(sorted(resolved)), tuple(sorted(unresolved))


def _classify_zone_regimes(
    dataset: DerivedRunData,
    *,
    snapshot_index: int,
    active_zone_indices: np.ndarray,
    material_labels: dict[int, str | None],
) -> tuple[np.ndarray, list[str], tuple[str, ...]]:
    te = np.asarray(dataset.temperature_e_ev[int(snapshot_index)], dtype=np.float64)[active_zone_indices]
    zbar = np.asarray(dataset.mean_charge[int(snapshot_index)], dtype=np.float64)[active_zone_indices]
    density = np.asarray(dataset.density_g_cm3[int(snapshot_index)], dtype=np.float64)[active_zone_indices]
    density_reference = _density_reference_profile(dataset)[active_zone_indices]
    density_ratio = _safe_density_ratio(density, density_reference)
    regimes = np.full(active_zone_indices.shape, _REGIME_THOMSON_FALLBACK, dtype=object)
    unresolved_materials: list[str] = []
    notes: list[str] = []
    thresholds = TRANSMISSION_AUTO_THRESHOLDS
    xcom_count = 0
    plasma_count = 0
    fallback_count = 0
    expanded_count = 0
    hot_count = 0
    for offset, zone_index in enumerate(active_zone_indices.tolist()):
        label = material_labels.get(int(zone_index))
        te_value = float(te[offset]) if np.isfinite(te[offset]) else float("nan")
        zbar_value = float(zbar[offset]) if np.isfinite(zbar[offset]) else float("nan")
        density_ratio_value = float(density_ratio[offset]) if np.isfinite(density_ratio[offset]) else float("nan")
        has_xcom_material = bool(label)
        if _xcom_like_zone(te_value, density_ratio_value, material_resolved=has_xcom_material) and not _plasma_like_zone(te_value, density_ratio_value):
            regimes[offset] = _REGIME_XCOM
            xcom_count += 1
            continue
        if _plasma_like_zone(te_value, density_ratio_value):
            regimes[offset] = _REGIME_PLASMA
            plasma_count += 1
            if math.isfinite(te_value) and te_value >= thresholds.plasma_min_temperature_ev:
                hot_count += 1
            if math.isfinite(density_ratio_value) and density_ratio_value <= thresholds.plasma_max_density_ratio:
                expanded_count += 1
            continue
        regimes[offset] = _REGIME_THOMSON_FALLBACK
        fallback_count += 1
        if not has_xcom_material:
            unresolved = f"material {int(abs(dataset.zone_material_index[int(zone_index)]))}"
            if unresolved not in unresolved_materials:
                unresolved_materials.append(unresolved)
    notes.append(
        "Auto hybrid is XCOM-first for cold condensed resolved materials, switches to Free-free + Thomson for clearly plasma-like or expanded zones, and uses Thomson only as a last-resort fallback."
    )
    notes.append(
        "Thresholds: "
        f"XCOM if Te <= {thresholds.xcom_transition_temperature_ev:.1f} eV and rho/rho0 >= {thresholds.xcom_transition_density_ratio:.2f}; "
        f"plasma-like if Te >= {thresholds.plasma_transition_temperature_ev:.1f} eV or rho/rho0 <= {thresholds.plasma_transition_density_ratio:.2f}."
    )
    notes.append(
        f"Current partition: XCOM {xcom_count} zones | plasma-like {plasma_count} zones | Thomson fallback {fallback_count} zones."
    )
    if hot_count or expanded_count:
        notes.append(
            f"Plasma-like triggers: {hot_count} hot zones and {expanded_count} expanded/ablated zones crossed the plasma threshold band."
        )
    if np.any(np.isfinite(zbar)):
        zbar_median = _weighted_median(zbar[np.isfinite(zbar)], np.ones(np.count_nonzero(np.isfinite(zbar)), dtype=np.float64))
        if zbar_median is not None:
            notes.append(
                f"Mean charge is treated as diagnostic only here (pathless median Zbar={float(zbar_median):.3g}); it is not the primary XCOM veto."
            )
    return regimes.astype(object, copy=False), unresolved_materials, tuple(notes)


def _photon_angular_frequency_rad_s(photon_energy_kev: float) -> float:
    if photon_energy_kev <= 0.0:
        return float("nan")
    return float(photon_energy_kev) * 1.0e3 / HBAR_EV_S


def _free_free_absorption_m_inv(te_ev: float, ne_cm3: float, mean_charge: float, photon_energy_kev: float) -> tuple[float, float | None, str | None]:
    """Return weak-coupling inverse-bremsstrahlung alpha [m^-1].

    This uses the approved quick-look form:

        alpha = 3.1e-17 * Z * n_e^2 * lnLambda * T_e^-3/2 * omega^-2
                * (1 - omega_p^2/omega^2)^-1/2

    with n_e in m^-3, T_e in eV, omega in s^-1. It is intentionally limited to
    weak-coupling quasi-free plasma conditions and is not a full opacity model.
    """

    if te_ev <= 0.0 or ne_cm3 <= 0.0 or mean_charge <= 0.0 or photon_energy_kev <= 0.0:
        return float("nan"), None, "Free-free quick look requires positive Te, ne, Zbar, and photon energy."
    coulomb_log = coulomb_logarithm_ei(float(te_ev), float(ne_cm3), max(float(mean_charge), 1.0))
    if not math.isfinite(coulomb_log) or coulomb_log <= 0.0:
        return float("nan"), None, "Coulomb logarithm is not finite for the selected plasma state; free-free absorption is out of domain."
    omega = _photon_angular_frequency_rad_s(float(photon_energy_kev))
    omega_p = electron_plasma_frequency_rad_s(float(ne_cm3))
    if not math.isfinite(omega) or omega <= 0.0 or not math.isfinite(omega_p):
        return float("nan"), coulomb_log, "Photon or plasma frequency is not finite for the selected state."
    cutoff_term = 1.0 - float(omega_p * omega_p) / float(omega * omega)
    if cutoff_term <= 0.0:
        return float("inf"), coulomb_log, "Photon frequency is at or below the local plasma cutoff; the free-free quick look treats the zone as effectively opaque."
    ne_m3 = float(ne_cm3) * 1.0e6
    alpha = 3.1e-17 * float(mean_charge) * ne_m3 * ne_m3 * float(coulomb_log) * float(te_ev) ** (-1.5) * float(omega) ** (-2.0) * float(cutoff_term) ** (-0.5)
    return float(alpha), float(coulomb_log), None


def _subset_cold_request(request, zone_indices: np.ndarray):
    index_set = {int(value) for value in np.asarray(zone_indices, dtype=np.int32).tolist()}
    return replace(
        request,
        zones=tuple(zone for zone in request.zones if (int(zone.zone_index) - 1) in index_set),
    )


def _build_xcom_result_from_payload(
    *,
    payload: dict[str, object],
    backend_name: str | None,
    backend_available: bool,
    backend_fingerprint: str | None,
    source: str,
    energy_kev: float,
    applicability: str,
    message: str,
    resolved_materials: tuple[str, ...],
    unresolved_materials: tuple[str, ...],
    cold_fraction: float | None,
    path_weighted_te: float | None,
    path_weighted_zbar: float | None,
    cached: bool = False,
) -> TransmissionColdRefinement:
    material_budgets = tuple(
        TransmissionColdMaterialBudget(
            label=str(item.get("label", "")),
            areal_density_g_cm2=float(item.get("areal_density_g_cm2", 0.0)),
            optical_depth=tuple(float(value) for value in item.get("optical_depth", ())),
        )
        for item in payload.get("material_budgets", [])
        if isinstance(item, dict)
    )
    return TransmissionColdRefinement(
        backend_status="refined_cached" if cached else ("refined" if source in {"live_xcom_backend", "precomputed_xcom_table"} else "baseline"),
        applicability=applicability,
        message=message,
        backend_name=backend_name,
        backend_available=backend_available,
        backend_fingerprint=backend_fingerprint,
        source=source,
        photon_energies_kev=tuple(float(value) for value in payload.get("energies_kev", (energy_kev,))),
        optical_depth=tuple(float(value) for value in payload.get("optical_depth", ())),
        transmission=tuple(float(value) for value in payload.get("transmission", ())),
        attenuation_mode=str(payload.get("attenuation_mode", "total_with_coherent")),
        resolved_materials=resolved_materials,
        unresolved_materials=unresolved_materials,
        material_budgets=material_budgets,
        cold_fraction=cold_fraction,
        path_weighted_temperature_e_ev=path_weighted_te,
        path_weighted_mean_charge=path_weighted_zbar,
    )


def _payload_coefficients(result_payload: dict[str, object]) -> dict[str, float]:
    coefficients: dict[str, float] = {}
    for item in result_payload.get("material_budgets", []):
        if not isinstance(item, dict):
            continue
        raw_label = str(item.get("label", "")).strip()
        areal_density = float(item.get("areal_density_g_cm2", 0.0))
        optical_depth_values = tuple(float(value) for value in item.get("optical_depth", ()))
        if not raw_label or areal_density <= 0.0 or not optical_depth_values:
            continue
        coefficient = float(optical_depth_values[0]) / areal_density
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            continue
        coefficients[normalize_material_key(raw_label)] = coefficient
    return coefficients


def _cold_backend_payload(
    backend,
    subset,
    *,
    backend_name: str,
    backend_fingerprint: str,
) -> dict[str, object]:
    cold_result = backend.compute_transmission(subset)
    return {
        "energies_kev": [float(value) for value in np.asarray(cold_result.energies_kev, dtype=np.float64).tolist()],
        "transmission": [float(value) for value in np.asarray(cold_result.transmission, dtype=np.float64).tolist()],
        "optical_depth": [float(value) for value in cold_result.metadata.get("optical_depth", [])],
        "attenuation_mode": str(cold_result.metadata.get("attenuation_mode", "total_with_coherent")),
        "backend_name": str(cold_result.metadata.get("backend_name", backend_name)),
        "backend_fingerprint": str(cold_result.metadata.get("backend_fingerprint", backend_fingerprint)),
        "material_budgets": list(cold_result.metadata.get("material_budgets", [])),
        "source": str(cold_result.metadata.get("source", "")),
        "interpolation_mode": str(cold_result.metadata.get("interpolation_mode", "")),
        "interpolation_note": str(cold_result.metadata.get("interpolation_note", "")),
        "quantity": str(cold_result.metadata.get("quantity", "")),
    }


def _xcom_material_tau_coefficients(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    zone_indices: np.ndarray,
) -> tuple[dict[str, float], TransmissionColdRefinement | None, tuple[DerivedWarning, ...], str, str]:
    """Return per-material XCOM tau / areal-density coefficients for selected zones."""

    backend_status = describe_optional_cold_backend(probe_compute=True)
    energy_kev = float(parameters.transmission_photon_energy_kev)
    request = build_cold_attenuation_request(
        dataset,
        context,
        snapshot_index=snapshot_index,
        parameters=parameters,
        geometry=geometry,
        photon_energies_kev=(energy_kev,),
    )
    subset = _subset_cold_request(request, zone_indices)
    applicability, cold_fraction, path_weighted_te, path_weighted_zbar, applicability_message = _assess_xcom_applicability(dataset, subset)
    _grouped_materials, resolved_materials, unresolved_materials = _material_resolution_for_request(subset)
    if not subset.zones:
        return {}, TransmissionColdRefinement(
            backend_status="unused",
            applicability="indeterminate",
            message="No XCOM-classified zones are active for the current snapshot.",
            backend_name=backend_status.backend_name,
            backend_available=bool(backend_status.available),
            backend_fingerprint=backend_status.backend_fingerprint,
            source="baseline",
            photon_energies_kev=(energy_kev,),
        ), (), "baseline", str(backend_status.status)
    if unresolved_materials:
        warning = DerivedWarning("transmission", "XCOM-classified zones include unresolved material labels; those zones fall back to Thomson.", severity="warning")
        refinement = TransmissionColdRefinement(
            backend_status="unresolved_materials",
            applicability=applicability,
            message="Some XCOM-classified zones could not be mapped to material formulas; Thomson fallback was used there.",
            backend_name=backend_status.backend_name,
            backend_available=bool(backend_status.available),
            backend_fingerprint=backend_status.backend_fingerprint,
            source="baseline",
            photon_energies_kev=(energy_kev,),
            resolved_materials=resolved_materials,
            unresolved_materials=unresolved_materials,
            cold_fraction=cold_fraction,
            path_weighted_temperature_e_ev=path_weighted_te,
            path_weighted_mean_charge=path_weighted_zbar,
        )
        return {}, refinement, (warning,), "baseline", "unresolved_materials"
    if applicability == "not_recommended":
        warning = DerivedWarning("transmission", applicability_message, severity="caution")
        refinement = TransmissionColdRefinement(
            backend_status=str(backend_status.status),
            applicability=applicability,
            message=applicability_message,
            backend_name=backend_status.backend_name,
            backend_available=bool(backend_status.available),
            backend_fingerprint=backend_status.backend_fingerprint,
            source="baseline",
            photon_energies_kev=(energy_kev,),
            resolved_materials=resolved_materials,
            unresolved_materials=unresolved_materials,
            cold_fraction=cold_fraction,
            path_weighted_temperature_e_ev=path_weighted_te,
            path_weighted_mean_charge=path_weighted_zbar,
        )
        return {}, refinement, (warning,), "baseline", str(backend_status.status)
    attenuation_mode = "total_with_coherent"
    cache = persistent_cold_attenuation_cache()
    result_payload: dict[str, object] | None = None
    source = "baseline"
    selected_backend_status = str(backend_status.status)
    selected_backend_name = backend_status.backend_name
    selected_backend_available = bool(backend_status.available)
    selected_backend_fingerprint = backend_status.backend_fingerprint
    cached = False
    warning_messages: list[str] = []

    backend_candidates: list[tuple[str, object | None, str | None, str | None, bool]] = []
    live_backend = load_optional_cold_backend(require_compute_ok=True)
    if live_backend is not None and backend_status.available:
        backend_candidates.append(
            (
                "live_xcom_backend",
                live_backend,
                str(getattr(live_backend, "backend_name", None) or backend_status.backend_name or "XCOM"),
                str(getattr(live_backend, "backend_fingerprint", None) or backend_status.backend_fingerprint or "xcom"),
                True,
            )
        )
    table_backend = load_precomputed_cold_backend()
    if table_backend is not None:
        backend_candidates.append(
            (
                "precomputed_xcom_table",
                table_backend,
                str(getattr(table_backend, "backend_name", None) or "XCOM table"),
                str(getattr(table_backend, "backend_fingerprint", None) or "precomputed_xcom_table"),
                True,
            )
        )

    for backend_source, backend, backend_name, backend_fingerprint, backend_available in backend_candidates:
        if backend is None:
            continue
        cache_key, cache_request_payload = cold_attenuation_cache_key(
            subset,
            backend_fingerprint=str(backend_fingerprint),
            attenuation_mode=attenuation_mode,
        )
        cached_entry = cache.get(cache_key)
        if cached_entry is not None:
            increment_counter("derived.cache.xcom.hit")
            result_payload = dict(cached_entry.get("result", {}))
            source = str(result_payload.get("source", backend_source) or backend_source)
            selected_backend_status = "refined"
            selected_backend_name = str(result_payload.get("backend_name", backend_name))
            selected_backend_available = bool(backend_available)
            selected_backend_fingerprint = str(result_payload.get("backend_fingerprint", backend_fingerprint))
            cached = True
            break
        increment_counter("derived.cache.xcom.miss")
        try:
            with timed_block("derived.compute.transmission_xcom"):
                result_payload = _cold_backend_payload(
                    backend,
                    subset,
                    backend_name=str(backend_name),
                    backend_fingerprint=str(backend_fingerprint),
                )
        except Exception as exc:
            warning_messages.append(f"{backend_name} refinement failed: {exc}")
            continue
        source = str(result_payload.get("source", backend_source) or backend_source)
        selected_backend_status = "refined"
        selected_backend_name = str(result_payload.get("backend_name", backend_name))
        selected_backend_available = bool(backend_available)
        selected_backend_fingerprint = str(result_payload.get("backend_fingerprint", backend_fingerprint))
        cache.put(cache_key, request_payload=cache_request_payload, result_payload=result_payload)
        break

    if result_payload is None:
        if warning_messages:
            message_text = " ".join(str(value) for value in warning_messages)
            warning = DerivedWarning("transmission", message_text, severity="warning")
        else:
            message_text = str(backend_status.message)
            warning = DerivedWarning("transmission", message_text, severity="info")
        refinement = TransmissionColdRefinement(
            backend_status=str(selected_backend_status),
            applicability=applicability,
            message=message_text,
            backend_name=selected_backend_name,
            backend_available=bool(selected_backend_available),
            backend_fingerprint=selected_backend_fingerprint,
            source="baseline",
            photon_energies_kev=(energy_kev,),
            resolved_materials=resolved_materials,
            unresolved_materials=unresolved_materials,
            cold_fraction=cold_fraction,
            path_weighted_temperature_e_ev=path_weighted_te,
            path_weighted_mean_charge=path_weighted_zbar,
        )
        return {}, refinement, (warning,), "baseline", str(selected_backend_status)

    coefficients = _payload_coefficients(result_payload)
    interpolation_note = str(result_payload.get("interpolation_note", "") or "").strip()
    message = _cold_refinement_status_message(
        backend_message=applicability_message,
        applicability=applicability,
        source=source,
        energy_kev=energy_kev,
        cached=cached,
        interpolation_note=interpolation_note or None,
        quantity_type=str(result_payload.get("quantity_type", "") or None),
    )
    refinement = _build_xcom_result_from_payload(
        payload=result_payload,
        backend_name=str(result_payload.get("backend_name", selected_backend_name or backend_status.backend_name or "XCOM")),
        backend_available=bool(selected_backend_available),
        backend_fingerprint=str(result_payload.get("backend_fingerprint", selected_backend_fingerprint or "xcom")),
        source=source,
        energy_kev=energy_kev,
        applicability=applicability,
        message=message,
        resolved_materials=resolved_materials,
        unresolved_materials=unresolved_materials,
        cold_fraction=cold_fraction,
        path_weighted_te=path_weighted_te,
        path_weighted_zbar=path_weighted_zbar,
        cached=cached,
    )
    return coefficients, refinement, (), source, str(selected_backend_status)


def _selected_time_plot_bundles(
    time_ns: np.ndarray,
    tau_series: np.ndarray,
    transmission_series: np.ndarray,
    *,
    mode: str,
) -> tuple[DerivedPlotBundle, ...]:
    label = transmission_mode_label(mode)
    return (
        DerivedPlotBundle(
            key="selected_tau",
            title=f"{label} tau vs time",
            x_label="Time [ns]",
            y_label=f"{label} tau",
            x_values=np.asarray(time_ns, dtype=np.float64),
            y_series=(np.asarray(tau_series, dtype=np.float64),),
            curve_names=(f"{label} tau",),
        ),
        DerivedPlotBundle(
            key="selected_transmission",
            title=f"{label} transmission vs time",
            x_label="Time [ns]",
            y_label="Transmission",
            x_values=np.asarray(time_ns, dtype=np.float64),
            y_series=(np.asarray(transmission_series, dtype=np.float64),),
            curve_names=(f"{label} transmission",),
        ),
    )


def _baseline_time_series_key(
    *,
    geometry: "AnalysisGeometryMetadata",
    parameters: "DerivedAnalysisParameters",
) -> tuple[object, ...]:
    return (
        "transmission.time_series",
        geometry.observation_side,
        round(float(geometry.line_of_sight_angle_deg), 12),
        round(float(geometry.impact_parameter_cm), 12),
        parameters.reuse_viewer_subset,
        tuple(parameters.derived_region_ids or ()),
        tuple(parameters.derived_material_ids or ()),
        bool(parameters.exclude_entry_region),
        bool(parameters.exclude_low_density),
        round(float(parameters.min_density_g_cm3), 12),
        bool(parameters.exclude_opposite_velocity),
        parameters.zone_index_lower,
        parameters.zone_index_upper,
    )


def _mode_time_series_key(
    *,
    mode: str,
    geometry: "AnalysisGeometryMetadata",
    parameters: "DerivedAnalysisParameters",
    energy_kev: float | None,
    backend_fingerprint: str | None,
) -> tuple[object, ...]:
    return (
        "transmission.selected_time_series",
        normalize_transmission_mode(mode),
        *_baseline_time_series_key(geometry=geometry, parameters=parameters),
        (None if energy_kev is None else round(float(energy_kev), 12)),
        str(backend_fingerprint or ""),
    )


def _baseline_time_series_payload(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> dict[str, np.ndarray]:
    series_cache_key = _baseline_time_series_key(geometry=geometry, parameters=parameters)

    def _build_time_series() -> dict[str, np.ndarray]:
        n_times = int(dataset.time_s.size)
        areal_series = np.full(n_times, np.nan, dtype=np.float64)
        column_series = np.full(n_times, np.nan, dtype=np.float64)
        tau_series = np.full(n_times, np.nan, dtype=np.float64)
        transmission_series = np.full(n_times, np.nan, dtype=np.float64)
        mask_matrix, _selection_keys = shared_time_series_selection_state(
            dataset,
            context,
            geometry=geometry,
            weighting_mode="path_integrated",
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
        for time_index in range(n_times):
            if progress_check is not None and (time_index % 8 == 0):
                progress_check()
            mask = np.asarray(mask_matrix[int(time_index)], dtype=bool)
            density = np.asarray(dataset.density_g_cm3[int(time_index)], dtype=np.float64)
            path = path_length_cm(dataset, time_index, geometry, analysis_cache=analysis_cache)
            electron_density = np.asarray(dataset.electron_density_cm3[int(time_index)], dtype=np.float64)
            active_density = np.where(mask, density, 0.0)
            active_path = np.where(mask, path, 0.0)
            active_electron_density = np.where(mask, electron_density, 0.0)
            point = {
                "areal_density": float(np.sum(active_density * active_path)),
                "electron_column": float(np.sum(active_electron_density * active_path)),
            }
            point["tau"] = float(THOMSON_CROSS_SECTION_CM2 * point["electron_column"])
            point["transmission"] = (
                float(math.exp(point["tau"] * -1.0)) if math.isfinite(float(point["tau"])) else float("nan")
            )
            areal_series[time_index] = float(point["areal_density"])
            column_series[time_index] = float(point["electron_column"])
            tau_series[time_index] = float(point["tau"])
            transmission_series[time_index] = float(point["transmission"])
        return {
            "areal_density": areal_series,
            "electron_column": column_series,
            "tau": tau_series,
            "transmission": transmission_series,
        }

    return cached_time_series_payload(
        series_cache_key,
        analysis_cache=analysis_cache,
        builder=_build_time_series,
    )


def _mode_aware_time_plots(
    base_result: TransmissionResult,
    dataset: DerivedRunData,
    context: RunContext,
    *,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    mode: str,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> tuple[DerivedPlotBundle, ...]:
    normalized = normalize_transmission_mode(mode)
    time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
    baseline_series = _baseline_time_series_payload(
        dataset,
        context,
        parameters=parameters,
        geometry=geometry,
        analysis_cache=analysis_cache,
        progress_check=progress_check,
    )
    if normalized == TRANSMISSION_MODE_THOMSON:
        return _selected_time_plot_bundles(
            time_ns,
            np.asarray(baseline_series["tau"], dtype=np.float64),
            np.asarray(baseline_series["transmission"], dtype=np.float64),
            mode=normalized,
        )

    backend_fingerprint = None
    energy_kev = float(parameters.transmission_photon_energy_kev) if _mode_supports_energy(normalized) else None
    if normalized in {TRANSMISSION_MODE_XCOM, TRANSMISSION_MODE_AUTO}:
        backend_fingerprint = describe_optional_cold_backend().backend_fingerprint
    series_cache_key = _mode_time_series_key(
        mode=normalized,
        geometry=geometry,
        parameters=parameters,
        energy_kev=energy_kev,
        backend_fingerprint=backend_fingerprint,
    )

    def _build_selected_series() -> dict[str, np.ndarray]:
        n_times = int(dataset.time_s.size)
        tau_series = np.full(n_times, np.nan, dtype=np.float64)
        transmission_series = np.full(n_times, np.nan, dtype=np.float64)
        for time_index in range(n_times):
            if progress_check is not None and (time_index % 2 == 0):
                progress_check()
            snapshot_context = replace(context.copy(), snapshot_index=int(time_index))
            baseline = evaluate_transmission(
                dataset,
                snapshot_context,
                snapshot_index=int(time_index),
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
                analysis_cache=analysis_cache,
                progress_check=progress_check,
            )
            selected = apply_transmission_model(
                baseline,
                dataset,
                snapshot_context,
                snapshot_index=int(time_index),
                parameters=parameters,
                geometry=geometry,
                include_time_plots=False,
                analysis_cache=analysis_cache,
                progress_check=progress_check,
            )
            tau_series[time_index] = float(selected.selected_tau) if selected.selected_tau is not None else float("nan")
            transmission_series[time_index] = (
                float(selected.selected_transmission) if selected.selected_transmission is not None else float("nan")
            )
        return {
            "tau": tau_series,
            "transmission": transmission_series,
        }

    selected_series = cached_time_series_payload(
        series_cache_key,
        analysis_cache=analysis_cache,
        builder=_build_selected_series,
    )
    return _selected_time_plot_bundles(
        time_ns,
        np.asarray(selected_series["tau"], dtype=np.float64),
        np.asarray(selected_series["transmission"], dtype=np.float64),
        mode=normalized,
    )


def apply_transmission_model(
    base_result: TransmissionResult,
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    include_time_plots: bool = True,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> TransmissionResult:
    mode = normalize_transmission_mode(parameters.transmission_mode)
    energy_kev = float(parameters.transmission_photon_energy_kev)
    summary, _baseline_region_budgets, budget_warnings, mask = _snapshot_budget(
        dataset,
        context,
        snapshot_index=snapshot_index,
        parameters=parameters,
        geometry=geometry,
        analysis_cache=analysis_cache,
    )
    warnings = _filter_mode_specific_base_warnings(tuple(base_result.warnings), mode=mode)
    for warning in budget_warnings:
        warnings = _append_warning_once(warnings, warning)

    density = np.asarray(dataset.density_g_cm3[int(snapshot_index)], dtype=np.float64)
    electron_density = np.asarray(dataset.electron_density_cm3[int(snapshot_index)], dtype=np.float64)
    temperature_e = np.asarray(dataset.temperature_e_ev[int(snapshot_index)], dtype=np.float64)
    mean_charge = np.asarray(dataset.mean_charge[int(snapshot_index)], dtype=np.float64)
    density_reference = _density_reference_profile(dataset)
    density_ratio = _safe_density_ratio(density, density_reference)
    path = np.asarray(path_length_cm(dataset, snapshot_index, geometry, analysis_cache=analysis_cache), dtype=np.float64)
    active_zone_indices = np.flatnonzero(mask)

    zone_areal = np.where(mask, density * path, 0.0)
    zone_thomson_tau = np.where(mask, THOMSON_CROSS_SECTION_CM2 * electron_density * path, 0.0)
    zone_free_free_tau = np.zeros_like(zone_thomson_tau, dtype=np.float64)
    zone_xcom_tau = np.zeros_like(zone_thomson_tau, dtype=np.float64)

    cold_request = build_cold_attenuation_request(
        dataset,
        context,
        snapshot_index=snapshot_index,
        parameters=parameters,
        geometry=geometry,
        photon_energies_kev=((energy_kev,) if _mode_supports_energy(mode) else ()),
    )
    zone_material_metadata = _cold_zone_metadata(cold_request)
    material_labels = {zone_index: metadata[0] for zone_index, metadata in zone_material_metadata.items()}
    material_canonical_keys = {zone_index: metadata[3] for zone_index, metadata in zone_material_metadata.items()}
    auto_regimes, unresolved_materials, classification_notes = _classify_zone_regimes(
        dataset,
        snapshot_index=snapshot_index,
        active_zone_indices=active_zone_indices,
        material_labels=material_canonical_keys,
    )
    auto_regime_by_zone = np.full(mask.shape, _REGIME_THOMSON_FALLBACK, dtype=object)
    for offset, zone_index in enumerate(active_zone_indices.tolist()):
        auto_regime_by_zone[int(zone_index)] = str(auto_regimes[offset])

    coulomb_logs: list[float] = []
    free_free_invalid = 0
    free_free_cutoff = 0
    free_free_valid_mask = np.zeros(mask.shape, dtype=bool)
    if mode in {TRANSMISSION_MODE_FREE_FREE, TRANSMISSION_MODE_FREE_FREE_THOMSON, TRANSMISSION_MODE_AUTO}:
        for offset, zone_index in enumerate(active_zone_indices.tolist()):
            use_ff = mode in {TRANSMISSION_MODE_FREE_FREE, TRANSMISSION_MODE_FREE_FREE_THOMSON} or str(auto_regimes[offset]) == _REGIME_PLASMA
            if not use_ff:
                continue
            alpha_ff_m_inv, coulomb_log, note = _free_free_absorption_m_inv(
                float(temperature_e[zone_index]),
                float(electron_density[zone_index]),
                float(mean_charge[zone_index]),
                energy_kev,
            )
            if coulomb_log is not None and math.isfinite(float(coulomb_log)):
                coulomb_logs.append(float(coulomb_log))
            if math.isinf(alpha_ff_m_inv):
                zone_free_free_tau[zone_index] = float("inf")
                free_free_cutoff += 1
                free_free_valid_mask[zone_index] = True
                continue
            if not math.isfinite(alpha_ff_m_inv) or alpha_ff_m_inv < 0.0:
                free_free_invalid += 1
                continue
            zone_free_free_tau[zone_index] = float(alpha_ff_m_inv) * float(path[zone_index]) * 1.0e-2
            free_free_valid_mask[zone_index] = True
        if free_free_invalid:
            warnings = _append_warning_message(
                warnings,
                f"Free-free quick look was unavailable in {int(free_free_invalid)} selected zones; those zones did not contribute a free-free term.",
                severity="caution",
            )
        if free_free_cutoff:
            warnings = _append_warning_message(
                warnings,
                f"{int(free_free_cutoff)} zones are at or below the local plasma cutoff for the selected photon energy; the free-free quick look treats them as effectively opaque.",
                severity="warning",
            )

    cold_refinement = base_result.cold_refinement
    xcom_source = "baseline"
    backend_status = cold_refinement.backend_status if cold_refinement is not None else None
    xcom_zone_indices = np.asarray(
        [int(zone_index) for offset, zone_index in enumerate(active_zone_indices.tolist()) if str(auto_regimes[offset]) == _REGIME_XCOM],
        dtype=np.int32,
    )
    if mode in {TRANSMISSION_MODE_XCOM, TRANSMISSION_MODE_AUTO}:
        coefficients, refinement, xcom_warnings, xcom_source, backend_status = _xcom_material_tau_coefficients(
            dataset,
            context,
            snapshot_index=snapshot_index,
            parameters=parameters,
            geometry=geometry,
            zone_indices=xcom_zone_indices,
        )
        cold_refinement = refinement
        for warning in xcom_warnings:
            warnings = _append_warning_once(warnings, warning)
        for zone_index in xcom_zone_indices.tolist():
            canonical_key = material_canonical_keys.get(int(zone_index))
            if not canonical_key:
                continue
            coefficient = float(coefficients.get(str(canonical_key), 0.0))
            if coefficient <= 0.0:
                continue
            zone_xcom_tau[zone_index] = coefficient * float(zone_areal[zone_index])
    else:
        cold_refinement = None
        backend_status = None

    xcom_requested_mask = np.zeros(mask.shape, dtype=bool)
    xcom_requested_mask[xcom_zone_indices] = True
    xcom_applied_mask = mask & (zone_xcom_tau > 0.0)
    xcom_fallback_mask = xcom_requested_mask & ~xcom_applied_mask

    selected_regime = np.full(mask.shape, "", dtype=object)
    if mode == TRANSMISSION_MODE_THOMSON:
        selected_zone_tau = zone_thomson_tau
        selected_regime[mask] = _REGIME_THOMSON
        status_message = "Thomson quick-look estimate. Electron scattering is treated as an attenuation/loss proxy rather than true absorption."
        source = "baseline"
    elif mode == TRANSMISSION_MODE_FREE_FREE:
        selected_zone_tau = zone_free_free_tau
        selected_regime[mask] = _REGIME_FREE_FREE
        status_message = (
            "Free-free quick look only. Uses a weak-coupling inverse-bremsstrahlung attenuation estimate and omits Thomson scattering from the selected tau."
        )
        source = "baseline"
    elif mode == TRANSMISSION_MODE_FREE_FREE_THOMSON:
        selected_zone_tau = zone_free_free_tau + zone_thomson_tau
        selected_regime[mask] = _REGIME_PLASMA
        status_message = (
            "Free-free + Thomson quick look. Uses a weak-coupling inverse-bremsstrahlung absorption estimate plus Thomson scattering loss."
        )
        source = "baseline"
    elif mode == TRANSMISSION_MODE_XCOM:
        selected_zone_tau = np.where(xcom_applied_mask, zone_xcom_tau, zone_thomson_tau)
        selected_regime[mask] = _REGIME_THOMSON_FALLBACK
        selected_regime[xcom_applied_mask] = _REGIME_XCOM
        status_message = "XCOM cold-material baseline for locally XCOM-qualified zones, with explicit Thomson fallback only where XCOM could not be applied."
        if cold_refinement is not None:
            status_message = str(cold_refinement.message or status_message)
        source = xcom_source
    else:
        plasma_selected_mask = np.asarray(auto_regime_by_zone, dtype=object) == _REGIME_PLASMA
        plasma_applied_mask = plasma_selected_mask & free_free_valid_mask
        selected_zone_tau = np.where(
            xcom_applied_mask,
            zone_xcom_tau,
            np.where(
                plasma_applied_mask,
                zone_free_free_tau + zone_thomson_tau,
                zone_thomson_tau,
            ),
        )
        selected_regime[mask] = _REGIME_THOMSON_FALLBACK
        selected_regime[plasma_applied_mask] = _REGIME_PLASMA
        selected_regime[xcom_applied_mask] = _REGIME_XCOM
        status_message = "Auto hybrid combines XCOM for cold condensed resolved zones, Free-free + Thomson for plasma-like zones, and Thomson only where no stronger local model could be applied."
        source = xcom_source if xcom_source in {"live_xcom_backend", "precomputed_xcom_table"} else "baseline"

    selected_zone_tau = np.where(mask, selected_zone_tau, 0.0)
    selected_tau_total = float(np.sum(selected_zone_tau)) if np.all(np.isfinite(selected_zone_tau) | (selected_zone_tau == float("inf"))) else float("nan")
    selected_transmission = float(math.exp(-selected_tau_total)) if math.isfinite(selected_tau_total) else (0.0 if np.isinf(selected_tau_total) else float("nan"))

    if unresolved_materials and mode in {TRANSMISSION_MODE_XCOM, TRANSMISSION_MODE_AUTO}:
        warnings = _append_warning_message(
            warnings,
            "Some selected materials could not be resolved to XCOM formulas and therefore stayed on Thomson fallback.",
            severity="caution",
        )
    if np.any(xcom_fallback_mask) and mode in {TRANSMISSION_MODE_XCOM, TRANSMISSION_MODE_AUTO}:
        fallback_count = int(np.count_nonzero(xcom_fallback_mask))
        fallback_materials = sorted(
            {
                str(zone_material_metadata.get(int(zone_index), (None, f"material {int(abs(dataset.zone_material_index[int(zone_index)]))}", "unresolved", None))[1])
                for zone_index in np.flatnonzero(xcom_fallback_mask).tolist()
            }
        )
        material_text = f" Affected materials: {', '.join(fallback_materials)}." if fallback_materials else ""
        warnings = _append_warning_message(
            warnings,
            f"{fallback_count} XCOM-classified zones fell back to Thomson because no valid XCOM attenuation coefficient was available for the current request.{material_text}",
            severity="caution",
        )
    if mode == TRANSMISSION_MODE_AUTO:
        plasma_fallback_mask = (np.asarray(auto_regime_by_zone, dtype=object) == _REGIME_PLASMA) & ~free_free_valid_mask & mask
        if np.any(plasma_fallback_mask):
            warnings = _append_warning_message(
                warnings,
                f"{int(np.count_nonzero(plasma_fallback_mask))} plasma-classified zones fell back to Thomson because the free-free quick look was invalid or unavailable locally.",
                severity="caution",
            )

    selected_regime_array = np.asarray(selected_regime, dtype=object)
    plasma_like_selected_mask = mask & np.isin(selected_regime_array, (_REGIME_PLASMA, _REGIME_FREE_FREE))
    thomson_like_selected_mask = mask & np.isin(selected_regime_array, (_REGIME_THOMSON, _REGIME_THOMSON_FALLBACK))
    finite_selected_tau = np.where(np.isfinite(selected_zone_tau), selected_zone_tau, 0.0)
    optically_negligible_mask = (
        mask
        & np.isfinite(finite_selected_tau)
        & (np.abs(finite_selected_tau) <= TRANSMISSION_AUTO_THRESHOLDS.optically_negligible_tau)
    )

    region_ids = np.asarray(dataset.regions["region_index"], dtype=np.int32)
    dense_region_index = _zone_region_dense_index(region_ids, np.asarray(dataset.zone_region_id, dtype=np.int32))
    active_dense = dense_region_index[mask]
    region_areal = np.bincount(active_dense, weights=zone_areal[mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_column = np.bincount(active_dense, weights=(electron_density[mask] * path[mask]), minlength=region_ids.size).astype(np.float64, copy=False)
    region_th = np.bincount(active_dense, weights=zone_thomson_tau[mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_ff = np.bincount(active_dense, weights=np.where(np.isfinite(zone_free_free_tau[mask]), zone_free_free_tau[mask], 0.0), minlength=region_ids.size).astype(np.float64, copy=False)
    region_xcom = np.bincount(active_dense, weights=zone_xcom_tau[mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_total = np.bincount(active_dense, weights=finite_selected_tau[mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_zone_count = np.bincount(active_dense, minlength=region_ids.size).astype(np.int32, copy=False)
    region_path = np.bincount(active_dense, weights=path[mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_xcom_path = np.bincount(dense_region_index[xcom_applied_mask], weights=path[xcom_applied_mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_plasma_path = np.bincount(dense_region_index[plasma_like_selected_mask], weights=path[plasma_like_selected_mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_thomson_path = np.bincount(dense_region_index[thomson_like_selected_mask], weights=path[thomson_like_selected_mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_xcom_selected_tau = np.bincount(dense_region_index[xcom_applied_mask], weights=finite_selected_tau[xcom_applied_mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_plasma_selected_tau = np.bincount(dense_region_index[plasma_like_selected_mask], weights=finite_selected_tau[plasma_like_selected_mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_thomson_selected_tau = np.bincount(dense_region_index[thomson_like_selected_mask], weights=finite_selected_tau[thomson_like_selected_mask], minlength=region_ids.size).astype(np.float64, copy=False)
    region_negligible_path = np.bincount(dense_region_index[optically_negligible_mask], weights=path[optically_negligible_mask], minlength=region_ids.size).astype(np.float64, copy=False)

    region_budgets: list[TransmissionRegionBudget] = []
    for region_offset in np.flatnonzero(region_zone_count > 0):
        region_mask = mask & (dense_region_index == int(region_offset))
        local_regime_array = np.asarray(selected_regime[region_mask], dtype=object)
        local_tau = np.where(np.isfinite(selected_zone_tau[region_mask]), selected_zone_tau[region_mask], 0.0)
        local_path = np.asarray(path[region_mask], dtype=np.float64)
        tau_contributions: dict[str, float] = {}
        path_contributions: dict[str, float] = {}
        for regime_name in {str(value) for value in local_regime_array.tolist() if str(value)}:
            local_mask = local_regime_array == regime_name
            tau_contributions[str(regime_name)] = float(np.sum(local_tau[local_mask]))
            path_contributions[str(regime_name)] = float(np.sum(local_path[local_mask]))
        significant_tau = {
            key: value
            for key, value in tau_contributions.items()
            if float(value) > TRANSMISSION_AUTO_THRESHOLDS.optically_negligible_tau
        }
        if significant_tau:
            dominant_regime = max(
                significant_tau.items(),
                key=lambda item: (float(item[1]), path_contributions.get(str(item[0]), 0.0), str(item[0])),
            )[0]
        elif path_contributions:
            dominant_regime = max(path_contributions.items(), key=lambda item: (float(item[1]), str(item[0])))[0]
        else:
            dominant_regime = _REGIME_THOMSON if mode == TRANSMISSION_MODE_THOMSON else _REGIME_THOMSON_FALLBACK
        region_path_total = float(region_path[region_offset])
        region_tau_total = float(region_total[region_offset])
        notes: list[str] = []
        negligible_fraction = None if region_path_total <= 0.0 else float(region_negligible_path[region_offset]) / region_path_total
        if negligible_fraction is not None and negligible_fraction > 0.0:
            notes.append(f"Optically negligible path fraction: {negligible_fraction * 100.0:.1f}%.")
        path_fraction_components = {
            _REGIME_XCOM: (float(region_xcom_path[region_offset]) / region_path_total if region_path_total > 0.0 else 0.0),
            _REGIME_PLASMA: (float(region_plasma_path[region_offset]) / region_path_total if region_path_total > 0.0 else 0.0),
            _REGIME_THOMSON_FALLBACK: (float(region_thomson_path[region_offset]) / region_path_total if region_path_total > 0.0 else 0.0),
        }
        tau_fraction_components = {
            _REGIME_XCOM: (float(region_xcom_selected_tau[region_offset]) / region_tau_total if region_tau_total > 0.0 else 0.0),
            _REGIME_PLASMA: (float(region_plasma_selected_tau[region_offset]) / region_tau_total if region_tau_total > 0.0 else 0.0),
            _REGIME_THOMSON_FALLBACK: (float(region_thomson_selected_tau[region_offset]) / region_tau_total if region_tau_total > 0.0 else 0.0),
        }
        significant_regimes = _significant_regime_fraction_keys(path_fraction_components) | _significant_regime_fraction_keys(tau_fraction_components)
        if len(significant_regimes) >= 2:
            dominant_path_regime, dominant_path_fraction = max(
                path_fraction_components.items(),
                key=lambda item: (float(item[1]), str(item[0])),
            )
            dominant_tau_regime, dominant_tau_fraction = max(
                tau_fraction_components.items(),
                key=lambda item: (float(item[1]), str(item[0])),
            )
            if dominant_path_regime == dominant_tau_regime:
                notes.append(
                    f"Mixed region: {_regime_display_label(str(dominant_tau_regime))} spans {dominant_path_fraction * 100.0:.1f}% "
                    f"of the path and contributes {dominant_tau_fraction * 100.0:.1f}% of the selected tau."
                )
            else:
                notes.append(
                    f"Mixed region: path is mostly {_regime_display_label(str(dominant_path_regime))} "
                    f"({dominant_path_fraction * 100.0:.1f}%), but selected tau is dominated by "
                    f"{_regime_display_label(str(dominant_tau_regime))} ({dominant_tau_fraction * 100.0:.1f}%)."
                )
        region_budgets.append(
            TransmissionRegionBudget(
                region_id=int(region_ids[region_offset]),
                areal_density_g_cm2=float(region_areal[region_offset]),
                electron_column_cm2=float(region_column[region_offset]),
                thomson_tau=float(region_th[region_offset]),
                free_free_tau=float(region_ff[region_offset]),
                xcom_tau=float(region_xcom[region_offset]),
                total_tau=float(region_total[region_offset]),
                xcom_path_fraction=(float(region_xcom_path[region_offset]) / region_path_total if region_path_total > 0.0 else None),
                free_free_thomson_path_fraction=(float(region_plasma_path[region_offset]) / region_path_total if region_path_total > 0.0 else None),
                thomson_fallback_path_fraction=(float(region_thomson_path[region_offset]) / region_path_total if region_path_total > 0.0 else None),
                xcom_tau_fraction=(float(region_xcom_selected_tau[region_offset]) / region_tau_total if region_tau_total > 0.0 else None),
                free_free_thomson_tau_fraction=(float(region_plasma_selected_tau[region_offset]) / region_tau_total if region_tau_total > 0.0 else None),
                thomson_fallback_tau_fraction=(float(region_thomson_selected_tau[region_offset]) / region_tau_total if region_tau_total > 0.0 else None),
                dominant_regime=str(dominant_regime),
                notes=tuple(notes),
            )
        )

    reverse = geometry.observation_boundary == "high"
    cumulative_depth_um = np.cumsum((path[mask][::-1] if reverse else path[mask])) * 1.0e4
    cumulative_th = np.cumsum((zone_thomson_tau[mask][::-1] if reverse else zone_thomson_tau[mask]))
    cumulative_ff = np.cumsum((zone_free_free_tau[mask][::-1] if reverse else zone_free_free_tau[mask]))
    cumulative_xcom = np.cumsum((zone_xcom_tau[mask][::-1] if reverse else zone_xcom_tau[mask]))
    cumulative_total = np.cumsum((selected_zone_tau[mask][::-1] if reverse else selected_zone_tau[mask]))
    cumulative_transmission = np.exp(-np.clip(cumulative_total, a_min=None, a_max=700.0))

    regime_summaries: list[TransmissionRegimeSummary] = []
    total_path = float(np.sum(path[mask]))
    total_areal = float(np.sum(zone_areal[mask]))
    total_tau_finite = float(np.sum(finite_selected_tau[mask]))
    for regime in (_REGIME_XCOM, _REGIME_PLASMA, _REGIME_FREE_FREE, _REGIME_THOMSON_FALLBACK, _REGIME_THOMSON):
        regime_mask = mask & (selected_regime_array == regime)
        if not np.any(regime_mask):
            continue
        regime_path = float(np.sum(path[regime_mask]))
        regime_areal = float(np.sum(zone_areal[regime_mask]))
        regime_tau = float(np.sum(finite_selected_tau[regime_mask]))
        regime_summaries.append(
            TransmissionRegimeSummary(
                regime=str(regime),
                zone_count=int(np.count_nonzero(regime_mask)),
                path_fraction=(regime_path / total_path if total_path > 0.0 else None),
                areal_density_fraction=(regime_areal / total_areal if total_areal > 0.0 else None),
                tau_fraction=(regime_tau / total_tau_finite if total_tau_finite > 0.0 else None),
            )
        )

    partition_notes = list(classification_notes)
    partition_notes.append(f"Selected mode: {transmission_mode_label(mode)}.")
    xcom_path_fraction = float(np.sum(path[xcom_applied_mask]) / total_path) if total_path > 0.0 else 0.0
    plasma_path_fraction = float(np.sum(path[plasma_like_selected_mask]) / total_path) if total_path > 0.0 else 0.0
    thomson_path_fraction = float(np.sum(path[thomson_like_selected_mask]) / total_path) if total_path > 0.0 else 0.0
    xcom_tau_fraction_total = float(np.sum(finite_selected_tau[xcom_applied_mask]) / total_tau_finite) if total_tau_finite > 0.0 else 0.0
    plasma_tau_fraction_total = float(np.sum(finite_selected_tau[plasma_like_selected_mask]) / total_tau_finite) if total_tau_finite > 0.0 else 0.0
    thomson_tau_fraction_total = float(np.sum(finite_selected_tau[thomson_like_selected_mask]) / total_tau_finite) if total_tau_finite > 0.0 else 0.0
    if _mode_supports_energy(mode):
        partition_notes.append(f"Photon energy: {energy_kev:.4g} keV.")
    partition_notes.append(
        f"Mixture summary: XCOM path {xcom_path_fraction * 100.0:.1f}% / tau {xcom_tau_fraction_total * 100.0:.1f}%, "
        f"plasma path {plasma_path_fraction * 100.0:.1f}% / tau {plasma_tau_fraction_total * 100.0:.1f}%, "
        f"Thomson path {thomson_path_fraction * 100.0:.1f}% / tau {thomson_tau_fraction_total * 100.0:.1f}%."
    )
    if coulomb_logs:
        partition_notes.append(
            f"Free-free quick look used the NRL-style Coulomb logarithm with LOS median lnΛ={float(np.nanmedian(np.asarray(coulomb_logs, dtype=np.float64))):.3g}."
        )
    partition_notes.append("Free-free terms use a weak-coupling quasi-free inverse-bremsstrahlung quick look and should be treated as approximate.")
    if include_time_plots:
        partition_notes.append(
            f"Time traces below show the currently selected {transmission_mode_label(mode)} transmission and tau across snapshots."
        )
    negligible_path_fraction = float(np.sum(path[optically_negligible_mask]) / total_path) if total_path > 0.0 else 0.0
    if negligible_path_fraction > 0.0:
        partition_notes.append(
            f"Optically negligible path: {negligible_path_fraction * 100.0:.1f}% of the selected LOS contributes tau <= {TRANSMISSION_AUTO_THRESHOLDS.optically_negligible_tau:.1e} at this energy."
        )
    if np.any(xcom_fallback_mask) and mode in {TRANSMISSION_MODE_XCOM, TRANSMISSION_MODE_AUTO}:
        partition_notes.append(
            f"XCOM fallback: {int(np.count_nonzero(xcom_fallback_mask))} XCOM-classified zones were carried on Thomson because no valid XCOM attenuation coefficient was available."
        )
    if mode in {TRANSMISSION_MODE_AUTO, TRANSMISSION_MODE_XCOM}:
        secondary_zbar = _weighted_median(mean_charge[mask], path[mask]) if np.any(mask) else None
        if secondary_zbar is not None:
            partition_notes.append(
                f"Mean charge remains diagnostic only for XCOM gating here (path-weighted median Zbar={float(secondary_zbar):.3g})."
            )
    if cold_refinement is not None and cold_refinement.resolved_materials:
        partition_notes.append("Resolved XCOM materials: " + ", ".join(str(value) for value in cold_refinement.resolved_materials))
    if cold_refinement is not None and cold_refinement.unresolved_materials:
        partition_notes.append("Unresolved XCOM materials: " + ", ".join(str(value) for value in cold_refinement.unresolved_materials))
    partition = TransmissionPartitionSummary(
        mode=mode,
        photon_energy_kev=(energy_kev if _mode_supports_energy(mode) else None),
        zone_count=int(np.count_nonzero(mask)),
        backend_status=backend_status,
        approximate=(mode != TRANSMISSION_MODE_THOMSON),
        cached=bool(cold_refinement is not None and cold_refinement.backend_status == "refined_cached"),
        regime_summaries=tuple(regime_summaries),
        unresolved_materials=tuple(unresolved_materials),
        notes=tuple(partition_notes),
    )
    warnings = _append_warning_message(
        warnings,
        "Transmission modes beyond Thomson use quick-look approximations; free-free terms are weak-coupling and XCOM remains a cold-material baseline.",
        severity="info",
    )
    if mode == TRANSMISSION_MODE_AUTO:
        warnings = _append_warning_message(
            warnings,
            "Auto hybrid classification is snapshot-local and partitions zones explicitly into XCOM, Free-free + Thomson, or Thomson fallback using current Te, rho/rho0, and resolved material identity. Mean charge is diagnostic only.",
            severity="info",
        )
    applied_model = mode
    if mode == TRANSMISSION_MODE_XCOM and not np.any(zone_xcom_tau > 0.0):
        applied_model = TRANSMISSION_MODE_THOMSON
    elif mode == TRANSMISSION_MODE_AUTO and not np.any(xcom_applied_mask) and not np.any(plasma_like_selected_mask):
        applied_model = TRANSMISSION_MODE_THOMSON

    profile_plots = (
        DerivedPlotBundle(
            key="cumulative_thomson_tau",
            title="Cumulative Thomson tau through target",
            x_label="Path depth from observation side [um]",
            y_label="Thomson tau",
            x_values=np.asarray(cumulative_depth_um, dtype=np.float64),
            y_series=(np.asarray(cumulative_th, dtype=np.float64),),
            curve_names=("Thomson tau",),
        ),
        DerivedPlotBundle(
            key="cumulative_free_free_tau",
            title="Cumulative Free-free tau through target",
            x_label="Path depth from observation side [um]",
            y_label="Free-free tau",
            x_values=np.asarray(cumulative_depth_um, dtype=np.float64),
            y_series=(np.asarray(cumulative_ff, dtype=np.float64),),
            curve_names=("Free-free tau",),
        ),
        DerivedPlotBundle(
            key="cumulative_xcom_tau",
            title="Cumulative XCOM tau through target",
            x_label="Path depth from observation side [um]",
            y_label="XCOM tau",
            x_values=np.asarray(cumulative_depth_um, dtype=np.float64),
            y_series=(np.asarray(cumulative_xcom, dtype=np.float64),),
            curve_names=("XCOM tau",),
        ),
        DerivedPlotBundle(
            key="cumulative_total_tau",
            title=f"Cumulative {transmission_mode_label(mode)} tau through target",
            x_label="Path depth from observation side [um]",
            y_label="Total tau",
            x_values=np.asarray(cumulative_depth_um, dtype=np.float64),
            y_series=(np.asarray(cumulative_total, dtype=np.float64),),
            curve_names=("Total tau",),
        ),
        DerivedPlotBundle(
            key="cumulative_selected_transmission",
            title=f"Cumulative {transmission_mode_label(mode)} transmission through target",
            x_label="Path depth from observation side [um]",
            y_label="Transmission",
            x_values=np.asarray(cumulative_depth_um, dtype=np.float64),
            y_series=(np.asarray(cumulative_transmission, dtype=np.float64),),
            curve_names=("Transmission",),
        ),
    )
    return replace(
        base_result,
        time_plots=(
            _mode_aware_time_plots(
                base_result,
                dataset,
                context,
                parameters=parameters,
                geometry=geometry,
                mode=mode,
                analysis_cache=analysis_cache,
                progress_check=progress_check,
            )
            if include_time_plots
            else ()
        ),
        profile_plots=profile_plots,
        region_budgets=tuple(region_budgets),
        model_type=applied_model,
        selected_mode=mode,
        photon_energy_kev=(energy_kev if _mode_supports_energy(mode) else None),
        selected_tau=selected_tau_total,
        selected_transmission=selected_transmission,
        source=source,
        status_message=status_message,
        backend_status=backend_status,
        partition=partition,
        cold_refinement=cold_refinement,
        warnings=warnings,
    )


def refine_transmission_with_xcom(
    base_result: TransmissionResult,
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
) -> TransmissionResult:
    override = replace(parameters, transmission_mode=TRANSMISSION_MODE_XCOM)
    return apply_transmission_model(
        base_result,
        dataset,
        context,
        snapshot_index=snapshot_index,
        parameters=override,
        geometry=geometry,
    )


def _snapshot_budget(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    analysis_cache: AnalysisStateCache | None = None,
) -> tuple[dict[str, float], tuple[TransmissionRegionBudget, ...], tuple[DerivedWarning, ...], np.ndarray]:
    warnings: list[DerivedWarning] = []
    mask, _, selection_warnings = build_analysis_mask(
        dataset,
        context,
        snapshot_index=snapshot_index,
        geometry=geometry,
        reuse_viewer_subset=parameters.reuse_viewer_subset,
        derived_region_ids=parameters.derived_region_ids,
        derived_material_ids=parameters.derived_material_ids,
        exclude_entry_region=parameters.exclude_entry_region,
        exclude_low_density=parameters.exclude_low_density,
        min_density_g_cm3=parameters.min_density_g_cm3,
        exclude_opposite_velocity=parameters.exclude_opposite_velocity,
        zone_index_lower=parameters.zone_index_lower,
        zone_index_upper=parameters.zone_index_upper,
        weighting_mode="path_integrated",
        analysis_cache=analysis_cache,
    )
    warnings.extend(selection_warnings)

    density = np.asarray(dataset.density_g_cm3[int(snapshot_index)], dtype=np.float64)
    path = path_length_cm(dataset, snapshot_index, geometry, analysis_cache=analysis_cache)
    electron_density = np.asarray(dataset.electron_density_cm3[int(snapshot_index)], dtype=np.float64)
    active_density = np.where(mask, density, 0.0)
    active_path = np.where(mask, path, 0.0)
    active_electron_density = np.where(mask, electron_density, 0.0)
    areal_density = float(np.sum(active_density * active_path))
    electron_column = float(np.sum(active_electron_density * active_path))
    tau_thomson = float(THOMSON_CROSS_SECTION_CM2 * electron_column)
    transmission = float(math.exp(-tau_thomson)) if math.isfinite(tau_thomson) else float("nan")

    region_ids = np.asarray(dataset.regions["region_index"], dtype=np.int32)
    dense_region_index = _zone_region_dense_index(region_ids, np.asarray(dataset.zone_region_id, dtype=np.int32))
    active_dense = dense_region_index[mask]
    region_areal = np.bincount(
        active_dense,
        weights=(density[mask] * path[mask]),
        minlength=region_ids.size,
    ).astype(np.float64, copy=False)
    region_column = np.bincount(
        active_dense,
        weights=(electron_density[mask] * path[mask]),
        minlength=region_ids.size,
    ).astype(np.float64, copy=False)
    region_zone_count = np.bincount(active_dense, minlength=region_ids.size).astype(np.int32, copy=False)
    active_regions = np.flatnonzero(region_zone_count > 0)
    region_budgets = tuple(
        TransmissionRegionBudget(
            region_id=int(region_ids[region_offset]),
            areal_density_g_cm2=float(region_areal[region_offset]),
            electron_column_cm2=float(region_column[region_offset]),
            thomson_tau=float(THOMSON_CROSS_SECTION_CM2 * region_column[region_offset]),
        )
        for region_offset in active_regions
    )

    if tau_thomson > 3.0:
        warnings.append(DerivedWarning("transmission", "Thomson optical depth is well above unity; the active subset is optically thick to electron scattering.", severity="warning"))
    elif tau_thomson > 1.0:
        warnings.append(DerivedWarning("transmission", "Thomson optical depth exceeds unity; the active subset is optically thick to electron scattering.", severity="caution"))
    if not np.any(mask):
        warnings.append(DerivedWarning("transmission", "No active zones are selected for the transmission estimate.", severity="error"))
    elif float(np.sum(active_path)) <= 0.0:
        warnings.append(DerivedWarning("transmission", "The active geometry produced zero effective path length after filtering.", severity="warning"))
    if geometry.path_length_mode in {"oblique-sec(theta)", "cylindrical-shell-unavailable-fallback-slab"} and abs(float(geometry.line_of_sight_cosine)) < 0.1:
        warnings.append(
            DerivedWarning(
                "transmission",
                "LOS cosine is very small; sec(theta) path-length amplification makes the transmission budget highly geometry-sensitive.",
                severity="warning",
            )
        )

    return (
        {
            "areal_density": areal_density,
            "electron_column": electron_column,
            "tau": tau_thomson,
            "transmission": transmission,
        },
        region_budgets,
        tuple(warnings),
        mask,
    )


def evaluate_transmission(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    include_time_plots: bool = True,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> TransmissionResult:
    """Compute areal density, electron column, and a Thomson optical-depth budget."""

    warnings: list[DerivedWarning] = [
        DerivedWarning(
            "transmission",
            "Transmission is a Thomson-only quick-look estimate; detailed photoabsorption and warm-dense-matter opacity physics are not included.",
            severity="info",
        )
    ]
    ps_warning = picosecond_drive_warning(
        "transmission",
        dataset,
        "The current transmission quick look is only weakly validated for ps-scale drives and should be treated as an approximate baseline there.",
    )
    if ps_warning is not None:
        warnings.append(ps_warning)
    cylindrical_warning = cylindrical_path_note("transmission", dataset, geometry)
    if cylindrical_warning is not None:
        warnings.append(cylindrical_warning)
    summary, region_budgets, budget_warnings, mask = _snapshot_budget(
        dataset,
        context,
        snapshot_index=snapshot_index,
        parameters=parameters,
        geometry=geometry,
        analysis_cache=analysis_cache,
    )
    warnings.extend(budget_warnings)

    time_plots: tuple[DerivedPlotBundle, ...] = ()
    if include_time_plots:
        time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
        time_series = _baseline_time_series_payload(
            dataset,
            context,
            parameters=parameters,
            geometry=geometry,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        time_plots = _selected_time_plot_bundles(
            time_ns,
            np.asarray(time_series["tau"], dtype=np.float64),
            np.asarray(time_series["transmission"], dtype=np.float64),
            mode=TRANSMISSION_MODE_THOMSON,
        )

    density = np.asarray(dataset.density_g_cm3[int(snapshot_index)], dtype=np.float64)
    electron_density = np.asarray(dataset.electron_density_cm3[int(snapshot_index)], dtype=np.float64)
    path = path_length_cm(dataset, snapshot_index, geometry, analysis_cache=analysis_cache)
    active_density = np.where(mask, density, 0.0)
    active_path = np.where(mask, path, 0.0)
    active_column_density = np.where(mask, electron_density, 0.0)

    reverse = geometry.observation_boundary == "high"
    if reverse:
        active_density = active_density[::-1]
        active_path = active_path[::-1]
        active_column_density = active_column_density[::-1]
    cumulative_depth_um = np.cumsum(active_path) * 1.0e4
    cumulative_areal = np.cumsum(active_density * active_path)
    cumulative_tau = np.cumsum(active_column_density * active_path) * THOMSON_CROSS_SECTION_CM2
    cumulative_transmission = np.exp(-cumulative_tau)

    geometry_summary = (
        f"{geometry.observation_side} side ({geometry.observation_boundary}-index boundary) | "
        f"LOS cos={geometry.line_of_sight_cosine:.3f} | {path_geometry_summary(dataset, geometry)}"
    )
    areal_title = "Cumulative areal density through target"
    tau_title = "Cumulative Thomson tau through target"
    transmission_title = "Cumulative transmission through target"
    depth_label = "Path depth from observation side [um]"
    if str(dataset.metadata.get("geometry", "")).strip().upper() == "CYLINDRICAL":
        areal_title = "Cumulative LOS shell areal density through target"
        tau_title = "Cumulative LOS shell Thomson tau through target"
        transmission_title = "Cumulative LOS shell transmission through target"
        depth_label = "Accumulated LOS shell path [um]"

    return TransmissionResult(
        snapshot_index=int(snapshot_index),
        weighting_mode="path_integrated",
        geometry_summary=geometry_summary,
        areal_density_g_cm2=float(summary["areal_density"]),
        electron_column_cm2=float(summary["electron_column"]),
        thomson_tau=float(summary["tau"]),
        thomson_transmission=float(summary["transmission"]),
        time_plots=time_plots,
        profile_plots=(
            DerivedPlotBundle(
                key="cumulative_areal",
                title=areal_title,
                x_label=depth_label,
                y_label="Areal density [g/cm2]",
                x_values=np.asarray(cumulative_depth_um, dtype=np.float64),
                y_series=(np.asarray(cumulative_areal, dtype=np.float64),),
                curve_names=("Cumulative areal density",),
            ),
            DerivedPlotBundle(
                key="cumulative_tau",
                title=tau_title,
                x_label=depth_label,
                y_label="Thomson tau",
                x_values=np.asarray(cumulative_depth_um, dtype=np.float64),
                y_series=(np.asarray(cumulative_tau, dtype=np.float64),),
                curve_names=("Cumulative tau",),
            ),
            DerivedPlotBundle(
                key="cumulative_transmission",
                title=transmission_title,
                x_label=depth_label,
                y_label="Transmission",
                x_values=np.asarray(cumulative_depth_um, dtype=np.float64),
                y_series=(np.asarray(cumulative_transmission, dtype=np.float64),),
                curve_names=("Cumulative transmission",),
            ),
        ),
        region_budgets=tuple(region_budgets),
        model_type="thomson",
        selected_mode=TRANSMISSION_MODE_THOMSON,
        selected_tau=float(summary["tau"]),
        selected_transmission=float(summary["transmission"]),
        source="baseline",
        status_message="Thomson quick-look estimate. Electron scattering is treated as an attenuation/loss proxy rather than true absorption.",
        backend_status=_baseline_cold_refinement().backend_status,
        cold_refinement=_baseline_cold_refinement(),
        warnings=tuple(warnings),
    )
