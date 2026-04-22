"""Quick-look XRTS / plasmon regime estimates with NRL-based unit discipline."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
import math
from time import perf_counter
from typing import Callable

import numpy as np

from helios.runtime import RunContext
from helios.services.derived.common import picosecond_drive_warning
from helios.services.derived.models import DerivedPlotBundle, DerivedRunData, DerivedWarning, PlasmonResult
from helios.services.derived.plasmon_config import (
    PLASMON_AXIS_ANGLE_DEG,
    PLASMON_AXIS_K_ANGSTROM_INV,
    PLASMON_BENCHMARK_PRESET_NONE,
    PLASMON_COLLISION_MODEL_BENCHMARK_DENSE,
    PLASMON_COLLISION_MODEL_MANUAL_CONSTANT,
    PLASMON_COLLISION_MODEL_NRL_CONSTANT,
    PLASMON_COMPARISON_MODEL_CHOICES,
    PLASMON_EXECUTION_MODE_BENCHMARK,
    PLASMON_EXECUTION_MODE_QUICKLOOK,
    PLASMON_INTEGRATION_MODE_LOS_INTEGRATED,
    PLASMON_MODEL_AUTO_BEST,
    PLASMON_MODEL_LINDHARD,
    PLASMON_MODEL_LINDHARD_MERMIN,
    PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC,
    PLASMON_MODEL_LINDHARD_STATIC_LFC,
    PLASMON_MODEL_MERMIN,
    PLASMON_MODEL_MERMIN_STATIC_LFC,
    PLASMON_MODEL_FINITE_T_STLS,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_MODEL_QUICKLOOK,
    PLASMON_MODEL_RPA,
    PLASMON_MODEL_RPA_STATIC_LFC,
    PLASMON_OBSERVABLE_MODE_DIELECTRIC,
    PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
    PLASMON_OBSERVABLE_MODE_XRTS,
    PLASMON_STUDY_MODE_DISPERSION,
    PLASMON_STUDY_MODE_SPECTRUM,
)
from helios.services.derived.plasmon_hydrodynamic import epsilon_quantum_hydrodynamic
from helios.services.derived.plasmon_electron_policy import (
    PLASMON_BENCHMARK_POLICY_COMPARISON,
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
    PLASMON_ELECTRON_POLICY_RAW,
    ElectronPolicyPayload,
    policy_label as plasmon_electron_policy_label,
    resolve_effective_electron_fields,
)
from helios.services.derived.plasmon_lfc import esa_domain_contains, esa_domain_message
from helios.services.derived.plasmon_stls import epsilon_finite_t_stls
from helios.services.derived.plasmon_spectrum import (
    dsf_from_loss,
    energy_axis_ev,
    epsilon_lindhard,
    epsilon_lindhard_mermin,
    epsilon_lindhard_mermin_static_lfc,
    epsilon_lindhard_static_lfc,
    epsilon_mermin,
    epsilon_mermin_static_lfc,
    epsilon_rpa,
    epsilon_rpa_static_lfc,
    estimate_peak_metrics,
    gaussian_convolve,
    loss_function_from_epsilon,
    normalize_spectrum,
)
from helios.services.derived.plasmon_xrts_observable import (
    build_minimal_xrts_components,
    finalize_xrts_observable,
    normalize_observable_mode,
)
from helios.services.derived.plasmon_xrts_material import (
    build_article_native_al_components,
    finalize_article_native_observable,
)
from helios.services.derived.plasmon_units import (
    ELECTRON_MASS_KG,
    HBAR_J_S,
    coulomb_logarithm_ei,
    coulomb_logarithm_ei_nrl_piecewise,
    electron_collision_rate_s,
    electron_density_m3_from_cm3,
    electron_debye_length_cm,
    electron_fermi_energy_ev,
    electron_fermi_wavevector_m_inv,
    electron_plasma_energy_ev,
    electron_plasma_frequency_rad_s,
    electron_thermal_speed_m_s,
    electron_theta_degeneracy,
    electron_wigner_seitz_rs,
    ion_sound_speed_cm_s,
    plasmon_probe_wavelength_angstrom,
    plasmon_probe_wavelength_cm,
    scattering_wavevector_cm_inv,
    scattering_wavevector_m_inv,
)
from helios.services.derived.selection import (
    AnalysisStateCache,
    build_analysis_mask,
    cached_time_series_payload,
    cylindrical_path_note,
    path_geometry_summary,
    profile_boundary_positions,
    profile_coordinate_values,
    resolve_weighting_mode,
    selection_cache_key,
    shared_time_series_weighted_means,
    weight_array,
    weighted_means,
)

if TYPE_CHECKING:
    from helios.services.derived.analysis import DerivedAnalysisParameters
    from helios.services.derived.models import AnalysisGeometryMetadata


_BENCHMARK_STATUS_NOT_APPLICABLE = "not_applicable"
_BENCHMARK_STATUS_VALID = "valid"
_BENCHMARK_STATUS_INVALID = "invalid_for_benchmark"
_BENCHMARK_STATUS_FALLBACK = "fallback_applied"


def _plasmon_execution_mode(parameters: "DerivedAnalysisParameters") -> str:
    value = str(getattr(parameters, "plasmon_execution_mode", PLASMON_EXECUTION_MODE_QUICKLOOK) or PLASMON_EXECUTION_MODE_QUICKLOOK)
    return value if value in {PLASMON_EXECUTION_MODE_QUICKLOOK, PLASMON_EXECUTION_MODE_BENCHMARK} else PLASMON_EXECUTION_MODE_QUICKLOOK


def _plasmon_benchmark_mode(parameters: "DerivedAnalysisParameters") -> bool:
    return _plasmon_execution_mode(parameters) == PLASMON_EXECUTION_MODE_BENCHMARK


def _resolved_energy_points(parameters: "DerivedAnalysisParameters") -> int:
    points = max(int(parameters.plasmon_energy_points), 101)
    if _plasmon_benchmark_mode(parameters):
        model = str(getattr(parameters, "plasmon_model", "") or "")
        benchmark_floor = 1201 if _uses_quantum_correlated_backend(model) else 4001
        points = max(points, benchmark_floor)
    return points + 1 if points % 2 == 0 else points


def _peak_fit_method(parameters: "DerivedAnalysisParameters") -> str:
    return "local_quadratic" if _plasmon_benchmark_mode(parameters) else "quadratic"


def _spectrum_metric_tuple(energy: np.ndarray, observed: np.ndarray, parameters: "DerivedAnalysisParameters") -> tuple[float, float, str]:
    method = _peak_fit_method(parameters)
    peak_energy_ev, peak_fwhm_ev = estimate_peak_metrics(
        energy,
        observed,
        method=method,
        local_half_window_points=(2 if method == "local_quadratic" else 1),
    )
    return float(peak_energy_ev), float(peak_fwhm_ev), str(method)


_DEGENERATE_BASELINE_ISSUE = "degenerate_classical_baseline"
_NONCOLLECTIVE_ISSUE = "noncollective_regime"
_WEAK_COUPLING_ISSUE = "weak_coupling_collision_closure"
_LFC_OUT_OF_DOMAIN_ISSUE = "lfc_out_of_domain"
_INVALID_COLLISION_ISSUE = "invalid_collision_rate"
_HARD_BENCHMARK_ISSUES = (
    _NONCOLLECTIVE_ISSUE,
    _WEAK_COUPLING_ISSUE,
    _LFC_OUT_OF_DOMAIN_ISSUE,
    _INVALID_COLLISION_ISSUE,
)
_TRACKED_BENCHMARK_ISSUES = (_DEGENERATE_BASELINE_ISSUE,) + _HARD_BENCHMARK_ISSUES

_LINDHARD_MODELS = {
    PLASMON_MODEL_LINDHARD,
    PLASMON_MODEL_LINDHARD_MERMIN,
    PLASMON_MODEL_LINDHARD_STATIC_LFC,
    PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC,
}
_STLS_MODELS = {PLASMON_MODEL_FINITE_T_STLS}
_MERMIN_MODELS = {
    PLASMON_MODEL_MERMIN,
    PLASMON_MODEL_MERMIN_STATIC_LFC,
    PLASMON_MODEL_LINDHARD_MERMIN,
    PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC,
}
_QHD_MODELS = {PLASMON_MODEL_QUANTUM_HYDRODYNAMIC}
_COLLISION_MODELS = _MERMIN_MODELS | _QHD_MODELS
_STATIC_LFC_MODELS = {
    PLASMON_MODEL_RPA_STATIC_LFC,
    PLASMON_MODEL_MERMIN_STATIC_LFC,
    PLASMON_MODEL_LINDHARD_STATIC_LFC,
    PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC,
}
_ADVANCED_SPECTRAL_MODELS = {
    PLASMON_MODEL_RPA,
    PLASMON_MODEL_MERMIN,
    PLASMON_MODEL_RPA_STATIC_LFC,
    PLASMON_MODEL_MERMIN_STATIC_LFC,
    PLASMON_MODEL_LINDHARD,
    PLASMON_MODEL_LINDHARD_MERMIN,
    PLASMON_MODEL_LINDHARD_STATIC_LFC,
    PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC,
    PLASMON_MODEL_FINITE_T_STLS,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_MODEL_AUTO_BEST,
}
_COMPARISON_MODELS = tuple(str(value) for _label, value in PLASMON_COMPARISON_MODEL_CHOICES)


def _plasmon_study_mode(parameters: "DerivedAnalysisParameters") -> str:
    value = str(getattr(parameters, "plasmon_study_mode", PLASMON_STUDY_MODE_SPECTRUM) or PLASMON_STUDY_MODE_SPECTRUM)
    return value if value in {PLASMON_STUDY_MODE_SPECTRUM, PLASMON_STUDY_MODE_DISPERSION} else PLASMON_STUDY_MODE_SPECTRUM


def _plasmon_coordinate_axis(parameters: "DerivedAnalysisParameters") -> str:
    value = str(getattr(parameters, "plasmon_coordinate_axis", PLASMON_AXIS_ANGLE_DEG) or PLASMON_AXIS_ANGLE_DEG)
    return value if value in {PLASMON_AXIS_ANGLE_DEG, PLASMON_AXIS_K_ANGSTROM_INV} else PLASMON_AXIS_ANGLE_DEG


def _plasmon_scan_axis(parameters: "DerivedAnalysisParameters") -> str:
    value = str(getattr(parameters, "plasmon_scan_axis", PLASMON_AXIS_ANGLE_DEG) or PLASMON_AXIS_ANGLE_DEG)
    return value if value in {PLASMON_AXIS_ANGLE_DEG, PLASMON_AXIS_K_ANGSTROM_INV} else PLASMON_AXIS_ANGLE_DEG


def _plasmon_compare_models(parameters: "DerivedAnalysisParameters") -> bool:
    return bool(getattr(parameters, "plasmon_compare_models", False))


def _requested_comparison_models(parameters: "DerivedAnalysisParameters") -> tuple[str, ...]:
    requested = getattr(parameters, "plasmon_compare_model_names", None)
    if requested is None:
        return _COMPARISON_MODELS
    normalized: list[str] = []
    for value in requested:
        model = str(value or "").strip()
        if model in _COMPARISON_MODELS and model not in normalized:
            normalized.append(model)
    return tuple(normalized) if normalized else _COMPARISON_MODELS


def _plasmon_compare_policies(parameters: "DerivedAnalysisParameters") -> bool:
    return bool(getattr(parameters, "plasmon_compare_policies", False))


def _plasmon_observable_mode(parameters: "DerivedAnalysisParameters") -> str:
    return normalize_observable_mode(getattr(parameters, "plasmon_observable_mode", PLASMON_OBSERVABLE_MODE_DIELECTRIC))


def _k_angstrom_inv_from_angle_deg(angle_deg: float, energy_kev: float) -> float:
    wavelength_angstrom = float(plasmon_probe_wavelength_angstrom(float(energy_kev)))
    if not math.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0.0:
        return float("nan")
    return float((4.0 * math.pi * math.sin(math.radians(float(angle_deg)) / 2.0)) / wavelength_angstrom)


def _angle_deg_from_k_angstrom_inv(k_angstrom_inv: float, energy_kev: float) -> float:
    wavelength_angstrom = float(plasmon_probe_wavelength_angstrom(float(energy_kev)))
    if not math.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0.0:
        return float("nan")
    argument = float(k_angstrom_inv) * wavelength_angstrom / (4.0 * math.pi)
    if not math.isfinite(argument) or argument < 0.0 or argument > 1.0:
        return float("nan")
    return float(math.degrees(2.0 * math.asin(argument)))


def _plasmon_scan_values(parameters: "DerivedAnalysisParameters") -> np.ndarray:
    start = float(getattr(parameters, "plasmon_scan_start", 10.0))
    stop = float(getattr(parameters, "plasmon_scan_stop", 140.0))
    points = max(int(getattr(parameters, "plasmon_scan_points", 61)), 3)
    if not math.isfinite(start) or not math.isfinite(stop):
        return np.asarray([], dtype=np.float64)
    return np.linspace(start, stop, points, dtype=np.float64)


def _uses_lindhard_backend(model: str) -> bool:
    return str(model) in _LINDHARD_MODELS


def _uses_stls_backend(model: str) -> bool:
    return str(model) in _STLS_MODELS


def _uses_quantum_correlated_backend(model: str) -> bool:
    return _uses_lindhard_backend(model) or _uses_stls_backend(model)


def _uses_collision_branch(model: str) -> bool:
    return str(model) in _COLLISION_MODELS


def _uses_quantum_hydrodynamic_backend(model: str) -> bool:
    return str(model) in _QHD_MODELS


def _uses_static_lfc(model: str) -> bool:
    return str(model) in _STATIC_LFC_MODELS


def _benchmark_rejects_noncollective(model: str) -> bool:
    return not _uses_quantum_correlated_backend(model)


def _empty_benchmark_issue_counts() -> dict[str, int]:
    return {name: 0 for name in _TRACKED_BENCHMARK_ISSUES}


def _benchmark_domain_flags(
    current: dict[str, float],
    parameters: "DerivedAnalysisParameters",
    requested_model: str,
) -> dict[str, bool]:
    model = str(requested_model or PLASMON_MODEL_QUICKLOOK)
    theta = float(current.get("theta_degeneracy", float("nan")))
    k_lambda = float(current.get("k_lambda", float("nan")))
    ln_lambda = float(current.get("coulomb_log", float("nan")))
    rs = float(current.get("wigner_seitz_rs", float("nan")))
    collision = _resolved_plasmon_collision_metadata(current, parameters)
    collision_rate = float(collision["rate_s"])
    del parameters
    flags = {
        _DEGENERATE_BASELINE_ISSUE: (math.isfinite(theta) and theta < 1.0 and not _uses_lindhard_backend(model)),
        _NONCOLLECTIVE_ISSUE: (math.isfinite(k_lambda) and k_lambda >= 1.0 and _benchmark_rejects_noncollective(model)),
        _WEAK_COUPLING_ISSUE: False,
        _LFC_OUT_OF_DOMAIN_ISSUE: False,
        _INVALID_COLLISION_ISSUE: False,
    }
    if _uses_collision_branch(model):
        flags[_WEAK_COUPLING_ISSUE] = (
            bool(collision.get("weak_coupling_required", False))
            and math.isfinite(ln_lambda)
            and ln_lambda <= 2.0
        )
        flags[_INVALID_COLLISION_ISSUE] = not math.isfinite(collision_rate) or collision_rate < 0.0
    if _uses_static_lfc(model):
        flags[_LFC_OUT_OF_DOMAIN_ISSUE] = not esa_domain_contains(rs, theta)
    return flags


def _accumulate_benchmark_issue_counts(counts: dict[str, int], flags: dict[str, bool]) -> dict[str, int]:
    for key, active in flags.items():
        if active:
            counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _benchmark_issue_count(counts: dict[str, int], *, hard_only: bool = False) -> int:
    keys = _HARD_BENCHMARK_ISSUES if hard_only else _TRACKED_BENCHMARK_ISSUES
    return int(sum(int(counts.get(key, 0)) for key in keys))


def _payload_issue_metadata(
    *,
    zone_counts: dict[str, int] | None = None,
    cluster_counts: dict[str, int] | None = None,
    zone_total: int = 0,
    cluster_total: int = 0,
) -> dict[str, int | float]:
    z = zone_counts or _empty_benchmark_issue_counts()
    c = cluster_counts or _empty_benchmark_issue_counts()
    zone_failures = _benchmark_issue_count(z, hard_only=True)
    cluster_failures = _benchmark_issue_count(c, hard_only=True)
    denom = float(cluster_total if cluster_total > 0 else zone_total)
    numer = float(cluster_failures if cluster_total > 0 else zone_failures)
    fraction = numer / denom if denom > 0.0 else 0.0
    return {
        "domain_failure_fraction": float(fraction),
        "degenerate_zone_count": int(z.get(_DEGENERATE_BASELINE_ISSUE, 0)),
        "noncollective_zone_count": int(z.get(_NONCOLLECTIVE_ISSUE, 0)),
        "weak_coupling_zone_count": int(z.get(_WEAK_COUPLING_ISSUE, 0)),
        "lfc_out_of_domain_zone_count": int(z.get(_LFC_OUT_OF_DOMAIN_ISSUE, 0)),
        "invalid_collision_zone_count": int(z.get(_INVALID_COLLISION_ISSUE, 0)),
        "degenerate_cluster_count": int(c.get(_DEGENERATE_BASELINE_ISSUE, 0)),
        "noncollective_cluster_count": int(c.get(_NONCOLLECTIVE_ISSUE, 0)),
        "weak_coupling_cluster_count": int(c.get(_WEAK_COUPLING_ISSUE, 0)),
        "lfc_out_of_domain_cluster_count": int(c.get(_LFC_OUT_OF_DOMAIN_ISSUE, 0)),
        "invalid_collision_cluster_count": int(c.get(_INVALID_COLLISION_ISSUE, 0)),
    }


def _augment_effective_state_domain_metadata(
    payload: dict[str, np.ndarray | float],
    *,
    current: dict[str, float],
    parameters: "DerivedAnalysisParameters",
    requested_model: str,
) -> dict[str, np.ndarray | float]:
    counts = _accumulate_benchmark_issue_counts(_empty_benchmark_issue_counts(), _benchmark_domain_flags(current, parameters, requested_model))
    payload.update(_payload_issue_metadata(zone_counts=counts, cluster_counts=counts, zone_total=1, cluster_total=1))
    return payload


def _plasmon_state_summary(
    te_ev: float,
    ti_ev: float,
    ne_cm3: float,
    zbar: float,
    ion_mass_mu: float,
    parameters: "DerivedAnalysisParameters",
) -> dict[str, float]:
    debye_length = electron_debye_length_cm(te_ev, ne_cm3)
    omega_pe = electron_plasma_frequency_rad_s(ne_cm3)
    ln_lambda = coulomb_logarithm_ei(te_ev, ne_cm3, max(zbar, 1.0))
    collision_rate = electron_collision_rate_s(ne_cm3, te_ev, ln_lambda)
    sound_speed = ion_sound_speed_cm_s(te_ev, max(zbar, 1.0), ion_mass_mu, parameters.plasmon_adiabatic_index)
    scattering_wavevector = scattering_wavevector_cm_inv(parameters.plasmon_photon_energy_kev, parameters.plasmon_scattering_angle_deg)
    k_lambda = scattering_wavevector * debye_length if math.isfinite(debye_length) else float("nan")
    collectivity = (1.0 / k_lambda) if math.isfinite(k_lambda) and k_lambda > 0.0 else float("nan")
    regime = "collective" if math.isfinite(k_lambda) and k_lambda < 1.0 else "non-collective"
    return {
        "te_ev": float(te_ev),
        "ti_ev": float(ti_ev),
        "ne_cm3": float(ne_cm3),
        "zbar": float(zbar),
        "ion_mass_mu": float(ion_mass_mu),
        "debye_length_cm": float(debye_length),
        "omega_pe_rad_s": float(omega_pe),
        "omega_pe_ev": float(electron_plasma_energy_ev(ne_cm3)),
        "collision_rate_s": float(collision_rate),
        "coulomb_log": float(ln_lambda),
        "sound_speed_cm_s": float(sound_speed),
        "probe_wavelength_angstrom": float(plasmon_probe_wavelength_angstrom(parameters.plasmon_photon_energy_kev)),
        "scattering_wavevector_cm_inv": float(scattering_wavevector),
        "scattering_wavevector_m_inv": float(scattering_wavevector_m_inv(parameters.plasmon_photon_energy_kev, parameters.plasmon_scattering_angle_deg)),
        "k_lambda": float(k_lambda),
        "collectivity": float(collectivity),
        "regime": regime,
        "fermi_energy_ev": float(electron_fermi_energy_ev(ne_cm3)),
        "theta_degeneracy": float(electron_theta_degeneracy(te_ev, ne_cm3)),
        "wigner_seitz_rs": float(electron_wigner_seitz_rs(ne_cm3)),
    }


def _summary_values(
    dataset: DerivedRunData,
    context: RunContext,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    *,
    snapshot_index: int,
    weighting_mode: str,
    electron_fields: ElectronPolicyPayload,
    analysis_cache: AnalysisStateCache | None = None,
) -> tuple[dict[str, float], tuple[DerivedWarning, ...], np.ndarray, object]:
    mask, selection, warnings = build_analysis_mask(
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
        weighting_mode=weighting_mode,
        analysis_cache=analysis_cache,
    )
    summary_fields = np.stack(
        (
            np.asarray(dataset.temperature_e_ev[int(snapshot_index)], dtype=np.float64),
            np.asarray(dataset.temperature_i_ev[int(snapshot_index)], dtype=np.float64),
            np.asarray(electron_fields.electron_density_cm3[int(snapshot_index)], dtype=np.float64),
            np.asarray(electron_fields.mean_charge[int(snapshot_index)], dtype=np.float64),
            np.asarray(dataset.zone_atomic_weight, dtype=np.float64),
        ),
        axis=0,
    )
    te_ev, ti_ev, ne_cm3, zbar, ion_mass_mu = weighted_means(
        summary_fields,
        dataset,
        snapshot_index,
        mask,
        mode=weighting_mode,
        geometry=geometry,
        selection_key=selection_cache_key(selection),
        analysis_cache=analysis_cache,
    )
    summary = _plasmon_state_summary(te_ev, ti_ev, ne_cm3, zbar, ion_mass_mu, parameters)
    return summary, warnings, mask, selection


def _summary_values_for_existing_selection(
    dataset: DerivedRunData,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    *,
    snapshot_index: int,
    weighting_mode: str,
    electron_fields: ElectronPolicyPayload,
    mask: np.ndarray,
    selection: object,
    analysis_cache: AnalysisStateCache | None = None,
) -> dict[str, float]:
    summary_fields = np.stack(
        (
            np.asarray(dataset.temperature_e_ev[int(snapshot_index)], dtype=np.float64),
            np.asarray(dataset.temperature_i_ev[int(snapshot_index)], dtype=np.float64),
            np.asarray(electron_fields.electron_density_cm3[int(snapshot_index)], dtype=np.float64),
            np.asarray(electron_fields.mean_charge[int(snapshot_index)], dtype=np.float64),
            np.asarray(dataset.zone_atomic_weight, dtype=np.float64),
        ),
        axis=0,
    )
    te_ev, ti_ev, ne_cm3, zbar, ion_mass_mu = weighted_means(
        summary_fields,
        dataset,
        snapshot_index,
        mask,
        mode=weighting_mode,
        geometry=geometry,
        selection_key=selection_cache_key(selection),
        analysis_cache=analysis_cache,
    )
    return _plasmon_state_summary(te_ev, ti_ev, ne_cm3, zbar, ion_mass_mu, parameters)


def _effective_electron_fields(dataset: DerivedRunData, parameters: "DerivedAnalysisParameters") -> ElectronPolicyPayload:
    driven_response_model = getattr(parameters, "plasmon_driven_response_model", None)
    if driven_response_model is not None and not str(driven_response_model).strip():
        driven_response_model = None
    return resolve_effective_electron_fields(
        dataset,
        getattr(parameters, "plasmon_electron_policy", PLASMON_ELECTRON_POLICY_RAW),
        driven_response_model=driven_response_model,
    )


def _subset_electron_policy_payload(
    dataset: DerivedRunData,
    mask: np.ndarray,
    electron_fields: ElectronPolicyPayload,
) -> ElectronPolicyPayload:
    active_mask = np.asarray(mask, dtype=bool)
    if active_mask.ndim != 1 or active_mask.size != np.asarray(dataset.zone_material_index).size:
        return electron_fields
    active_material_ids = {int(value) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)[active_mask]))}
    if not active_material_ids:
        return electron_fields
    formula_map = dict(getattr(electron_fields, "material_formula_map", {}) or {})
    active_formulas = {str(formula_map[material_id]) for material_id in active_material_ids if material_id in formula_map}

    def _include_material_entry(entry: str) -> bool:
        token = str(entry).strip()
        if not token:
            return False
        if token.lower().startswith("material "):
            try:
                material_id = int(token.split()[1])
            except Exception:
                return False
            return material_id in active_material_ids
        formula = token.split("->", 1)[0].split("@", 1)[0].split("(", 1)[0].strip()
        return formula in active_formulas

    resolved_entries = tuple(str(value) for value in getattr(electron_fields, "resolved_materials", ()) if _include_material_entry(str(value)))
    unresolved_entries = tuple(str(value) for value in getattr(electron_fields, "unresolved_materials", ()) if _include_material_entry(str(value)))
    raw_kept_entries = tuple(str(value) for value in getattr(electron_fields, "raw_kept_materials", ()) if _include_material_entry(str(value)))

    if str(electron_fields.policy) == PLASMON_ELECTRON_POLICY_RAW:
        summary = "Raw HELIOS electron-density and mean-charge fields for the active subset."
    else:
        policy_label = plasmon_electron_policy_label(electron_fields.policy)
        summary = (
            f"{policy_label} active-subset materials: "
            f"{', '.join(resolved_entries) if resolved_entries else 'no resolved benchmark materials'}; "
            f"unresolved [{', '.join(unresolved_entries) if unresolved_entries else 'none'}]; "
            f"raw-kept [{', '.join(raw_kept_entries) if raw_kept_entries else 'none'}]."
        )
    return ElectronPolicyPayload(
        requested_policy=str(electron_fields.requested_policy),
        policy=str(electron_fields.policy),
        electron_density_cm3=electron_fields.electron_density_cm3,
        mean_charge=electron_fields.mean_charge,
        source_label=str(electron_fields.source_label),
        summary=summary,
        resolved_materials=resolved_entries,
        unresolved_materials=unresolved_entries,
        raw_kept_materials=raw_kept_entries,
        material_formula_map=formula_map,
        baseline_mode=str(getattr(electron_fields, "baseline_mode", "")),
        baseline_entries=tuple(str(value) for value in getattr(electron_fields, "baseline_entries", ())),
        baseline_table_source=str(getattr(electron_fields, "baseline_table_source", "")),
        baseline_mean_charge=getattr(electron_fields, "baseline_mean_charge", None),
        increment_mean_charge=getattr(electron_fields, "increment_mean_charge", None),
        increment_mode=str(getattr(electron_fields, "increment_mode", "")),
        increment_entries=tuple(str(value) for value in getattr(electron_fields, "increment_entries", ())),
        driven_response_model=str(getattr(electron_fields, "driven_response_model", "none")),
        driven_response_summary=str(getattr(electron_fields, "driven_response_summary", "")),
        driven_response_weight_mode=str(getattr(electron_fields, "driven_response_weight_mode", "")),
        driven_response_weight_multiplier=getattr(electron_fields, "driven_response_weight_multiplier", None),
        driven_response_shape_mode=str(getattr(electron_fields, "driven_response_shape_mode", "")),
        driven_response_shape_fwhm_ev=getattr(electron_fields, "driven_response_shape_fwhm_ev", None),
        driven_response_ensemble_mode=str(getattr(electron_fields, "driven_response_ensemble_mode", "")),
    )


def _active_material_formulas(
    dataset: DerivedRunData,
    mask: np.ndarray,
    electron_fields: ElectronPolicyPayload,
) -> tuple[str, ...]:
    active_mask = np.asarray(mask, dtype=bool)
    if active_mask.ndim != 1 or active_mask.size != np.asarray(dataset.zone_material_index).size:
        return ()
    formula_map = dict(getattr(electron_fields, "material_formula_map", {}) or {})
    active_material_ids = {int(value) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)[active_mask]))}
    formulas = sorted({str(formula_map[material_id]) for material_id in active_material_ids if material_id in formula_map and str(formula_map[material_id]).strip()})
    return tuple(formulas)


def _observable_material_formula(
    dataset: DerivedRunData,
    mask: np.ndarray,
    electron_fields: ElectronPolicyPayload,
) -> str | None:
    formulas = _active_material_formulas(dataset, mask, electron_fields)
    if len(formulas) == 1:
        return str(formulas[0])
    return None


def _electron_policy_warnings(electron_fields: ElectronPolicyPayload) -> list[DerivedWarning]:
    if str(electron_fields.policy) == PLASMON_ELECTRON_POLICY_RAW:
        return []
    warnings = [
        DerivedWarning(
            "plasmon",
            f"Plasmon electron policy '{plasmon_electron_policy_label(electron_fields.policy)}' is active: {electron_fields.summary} This mode is intended for literature-facing benchmark comparisons, not as a blind replacement for raw hydro ionization.",
            severity="info",
        )
    ]
    if electron_fields.unresolved_materials:
        warnings.append(
            DerivedWarning(
                "plasmon",
                "Some active-subset materials did not resolve to a benchmark valence entry and therefore stayed on raw HELIOS electron fields: " + ", ".join(electron_fields.unresolved_materials),
                severity="caution",
            )
        )
    if electron_fields.raw_kept_materials and str(electron_fields.policy) == PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK:
        warnings.append(
            DerivedWarning(
                "plasmon",
                "Article Al benchmark policy only remapped Al zones. The following active-subset materials stayed on raw HELIOS electron fields: " + ", ".join(electron_fields.raw_kept_materials),
                severity="info",
            )
        )
    return warnings


def _cache_float(value: float, digits: int = 12) -> float | str:
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    return float(f"{value:.{digits}g}")


def _extract_zone_plasmon_states(
    dataset: DerivedRunData,
    *,
    snapshot_index: int,
    mask: np.ndarray,
    selection: object,
    weighting_mode: str,
    geometry: "AnalysisGeometryMetadata",
    electron_fields: ElectronPolicyPayload,
    analysis_cache: AnalysisStateCache | None = None,
) -> dict[str, np.ndarray]:
    selection_key = selection_cache_key(selection)
    base_weights = weight_array(
        np.ones(mask.shape[0], dtype=np.float64),
        dataset,
        snapshot_index,
        mask,
        mode=weighting_mode,
        geometry=geometry,
        selection_key=selection_key,
        analysis_cache=analysis_cache,
    )
    te = np.asarray(dataset.temperature_e_ev[int(snapshot_index)], dtype=np.float64)
    ti = np.asarray(dataset.temperature_i_ev[int(snapshot_index)], dtype=np.float64)
    ne = np.asarray(electron_fields.electron_density_cm3[int(snapshot_index)], dtype=np.float64)
    zbar = np.asarray(electron_fields.mean_charge[int(snapshot_index)], dtype=np.float64)
    ion_mass = np.asarray(dataset.zone_atomic_weight, dtype=np.float64)
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(base_weights)
        & (base_weights > 0.0)
        & np.isfinite(te)
        & (te > 0.0)
        & np.isfinite(ti)
        & np.isfinite(ne)
        & (ne > 0.0)
        & np.isfinite(zbar)
        & (zbar > 0.0)
        & np.isfinite(ion_mass)
        & (ion_mass > 0.0)
    )
    return {
        "te_ev": te[valid],
        "ti_ev": ti[valid],
        "ne_cm3": ne[valid],
        "zbar": zbar[valid],
        "ion_mass_mu": ion_mass[valid],
        "weight": np.asarray(base_weights[valid], dtype=np.float64),
    }


def _cluster_zone_plasmon_states(
    dataset: DerivedRunData,
    *,
    snapshot_index: int,
    selection: object,
    zone_states: dict[str, np.ndarray],
    weighting_mode: str,
    geometry: "AnalysisGeometryMetadata",
    parameters: "DerivedAnalysisParameters",
    electron_fields: ElectronPolicyPayload,
    analysis_cache: AnalysisStateCache | None = None,
) -> tuple[dict[str, np.ndarray], tuple[tuple[object, ...], ...]]:
    weights = np.asarray(zone_states["weight"], dtype=np.float64)
    if weights.size == 0:
        empty = {key: np.asarray([], dtype=np.float64) for key in ("te_ev", "ti_ev", "ne_cm3", "zbar", "ion_mass_mu", "weight")}
        return empty, ()
    geometry_key = (
        str(geometry.observation_side),
        round(float(geometry.line_of_sight_angle_deg), 8),
        round(float(geometry.line_of_sight_cosine), 8),
        round(float(geometry.impact_parameter_cm), 8),
        str(geometry.path_length_mode),
    )
    cache_key = (
        "plasmon.state_clusters",
        int(snapshot_index),
        selection_cache_key(selection),
        str(weighting_mode),
        geometry_key,
        round(float(parameters.plasmon_cluster_log_ne_tol), 8),
        round(float(parameters.plasmon_cluster_log_te_tol), 8),
        round(float(parameters.plasmon_cluster_z_tol), 8),
        str(getattr(electron_fields, "requested_policy", "")),
        str(getattr(electron_fields, "policy", "")),
        str(getattr(electron_fields, "source_label", "")),
        tuple(str(value) for value in getattr(electron_fields, "resolved_materials", ())),
        tuple(str(value) for value in getattr(electron_fields, "unresolved_materials", ())),
        tuple(str(value) for value in getattr(electron_fields, "raw_kept_materials", ())),
        str(getattr(electron_fields, "baseline_mode", "")),
    )

    def _build() -> tuple[dict[str, np.ndarray], tuple[tuple[object, ...], ...]]:
        log_ne_tol = max(float(parameters.plasmon_cluster_log_ne_tol), 0.0)
        log_te_tol = max(float(parameters.plasmon_cluster_log_te_tol), 0.0)
        z_tol = max(float(parameters.plasmon_cluster_z_tol), 0.0)
        if log_ne_tol <= 0.0 and log_te_tol <= 0.0 and z_tol <= 0.0:
            clusters = {key: np.asarray(value, dtype=np.float64) for key, value in zone_states.items()}
            signature = tuple(
                (
                    _cache_float(float(te), 10),
                    _cache_float(float(ti), 10),
                    _cache_float(float(ne), 10),
                    _cache_float(float(z), 10),
                    _cache_float(float(ion_mass), 10),
                    _cache_float(float(weight), 10),
                )
                for te, ti, ne, z, ion_mass, weight in zip(
                    clusters["te_ev"],
                    clusters["ti_ev"],
                    clusters["ne_cm3"],
                    clusters["zbar"],
                    clusters["ion_mass_mu"],
                    clusters["weight"],
                    strict=False,
                )
            )
            return clusters, signature
        log_ne = np.log10(np.clip(zone_states["ne_cm3"], 1.0e-300, None))
        log_te = np.log10(np.clip(zone_states["te_ev"], 1.0e-300, None))
        key_ne = np.rint(log_ne / log_ne_tol).astype(np.int64) if log_ne_tol > 0.0 else np.arange(weights.size, dtype=np.int64)
        key_te = np.rint(log_te / log_te_tol).astype(np.int64) if log_te_tol > 0.0 else np.arange(weights.size, dtype=np.int64)
        key_z = np.rint(zone_states["zbar"] / z_tol).astype(np.int64) if z_tol > 0.0 else np.arange(weights.size, dtype=np.int64)
        cluster_keys = np.stack((key_ne, key_te, key_z), axis=1)
        unique_keys, inverse = np.unique(cluster_keys, axis=0, return_inverse=True)
        order = np.argsort(np.bincount(inverse, weights=weights), kind="stable")[::-1]
        te_clusters = []
        ti_clusters = []
        ne_clusters = []
        z_clusters = []
        ion_mass_clusters = []
        weight_clusters = []
        signature = []
        for ordered_index in order:
            cluster_index = int(ordered_index)
            member_mask = inverse == cluster_index
            member_weights = weights[member_mask]
            total_weight = float(np.sum(member_weights, dtype=np.float64))
            if not math.isfinite(total_weight) or total_weight <= 0.0:
                continue
            normalized = member_weights / total_weight
            te_value = float(np.sum(zone_states["te_ev"][member_mask] * normalized, dtype=np.float64))
            ti_value = float(np.sum(zone_states["ti_ev"][member_mask] * normalized, dtype=np.float64))
            ne_value = float(np.sum(zone_states["ne_cm3"][member_mask] * normalized, dtype=np.float64))
            z_value = float(np.sum(zone_states["zbar"][member_mask] * normalized, dtype=np.float64))
            ion_mass_value = float(np.sum(zone_states["ion_mass_mu"][member_mask] * normalized, dtype=np.float64))
            te_clusters.append(te_value)
            ti_clusters.append(ti_value)
            ne_clusters.append(ne_value)
            z_clusters.append(z_value)
            ion_mass_clusters.append(ion_mass_value)
            weight_clusters.append(total_weight)
            signature.append((
                _cache_float(te_value, 10),
                _cache_float(ti_value, 10),
                _cache_float(ne_value, 10),
                _cache_float(z_value, 10),
                _cache_float(ion_mass_value, 10),
                _cache_float(total_weight, 10),
                tuple(int(v) for v in unique_keys[cluster_index]),
            ))
        return {
            "te_ev": np.asarray(te_clusters, dtype=np.float64),
            "ti_ev": np.asarray(ti_clusters, dtype=np.float64),
            "ne_cm3": np.asarray(ne_clusters, dtype=np.float64),
            "zbar": np.asarray(z_clusters, dtype=np.float64),
            "ion_mass_mu": np.asarray(ion_mass_clusters, dtype=np.float64),
            "weight": np.asarray(weight_clusters, dtype=np.float64),
        }, tuple(signature)

    return cached_time_series_payload(cache_key, analysis_cache=analysis_cache, builder=_build)




def _collision_model_label(value: str) -> str:
    labels = {
        PLASMON_COLLISION_MODEL_NRL_CONSTANT: "NRL constant nu",
        PLASMON_COLLISION_MODEL_BENCHMARK_DENSE: "Benchmark dense nu",
        PLASMON_COLLISION_MODEL_MANUAL_CONSTANT: "Manual constant nu",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def _benchmark_dense_collision_rate_s(current: dict[str, float]) -> tuple[float, str]:
    ne_cm3 = float(current.get("ne_cm3", float("nan")))
    te_ev = float(current.get("te_ev", float("nan")))
    zbar = float(current.get("zbar", float("nan")))
    if not math.isfinite(ne_cm3) or ne_cm3 <= 0.0 or not math.isfinite(te_ev) or te_ev <= 0.0:
        return float("nan"), "benchmark_dense: invalid ne/Te"
    z_eff = max(abs(zbar), 1.0e-6)
    ne_m3 = electron_density_m3_from_cm3(ne_cm3)
    ion_density_m3 = ne_m3 / z_eff if z_eff > 0.0 else float("nan")
    if not math.isfinite(ion_density_m3) or ion_density_m3 <= 0.0:
        return float("nan"), "benchmark_dense: invalid ion density"
    ion_sphere_m = (3.0 / (4.0 * math.pi * ion_density_m3)) ** (1.0 / 3.0)
    if not math.isfinite(ion_sphere_m) or ion_sphere_m <= 0.0:
        return float("nan"), "benchmark_dense: invalid ion-sphere spacing"
    thermal_speed = electron_thermal_speed_m_s(te_ev)
    kf = electron_fermi_wavevector_m_inv(ne_cm3)
    fermi_speed = float("nan")
    if math.isfinite(kf) and kf > 0.0:
        fermi_speed = (HBAR_J_S * kf) / ELECTRON_MASS_KG
    v_eff = math.sqrt(max(thermal_speed, 0.0) ** 2 + max(fermi_speed if math.isfinite(fermi_speed) else 0.0, 0.0) ** 2)
    if not math.isfinite(v_eff) or v_eff <= 0.0:
        return float("nan"), "benchmark_dense: invalid effective electron speed"
    dense_floor = 0.05 * v_eff / ion_sphere_m
    return float(dense_floor), (
        f"benchmark_dense floor from v_eff/a_i with v_eff={v_eff:.3e} m/s, "
        f"a_i={ion_sphere_m:.3e} m"
    )


def _resolved_plasmon_collision_metadata(current: dict[str, float], parameters: "DerivedAnalysisParameters") -> dict[str, object]:
    model = str(parameters.plasmon_collision_model or PLASMON_COLLISION_MODEL_NRL_CONSTANT)
    if model == PLASMON_COLLISION_MODEL_MANUAL_CONSTANT:
        value = float(parameters.plasmon_manual_collision_rate_s)
        rate = value if math.isfinite(value) and value >= 0.0 else float("nan")
        return {
            "rate_s": float(rate),
            "source": PLASMON_COLLISION_MODEL_MANUAL_CONSTANT,
            "summary": f"manual constant nu = {rate:.3e} 1/s" if math.isfinite(rate) else "manual constant nu is invalid",
            "weak_coupling_required": False,
        }
    base = float(current.get("collision_rate_s", float("nan")))
    ln_lambda = float(current.get("coulomb_log", float("nan")))
    if not math.isfinite(base) or base < 0.0:
        ln_lambda = coulomb_logarithm_ei_nrl_piecewise(
            float(current.get("te_ev", float("nan"))),
            float(current.get("ne_cm3", float("nan"))),
            float(current.get("zbar", float("nan"))),
        )
        base = electron_collision_rate_s(
            float(current.get("ne_cm3", float("nan"))),
            float(current.get("te_ev", float("nan"))),
            ln_lambda,
        )
    scale = float(parameters.plasmon_collision_scale)
    if model == PLASMON_COLLISION_MODEL_BENCHMARK_DENSE:
        dense_floor, dense_summary = _benchmark_dense_collision_rate_s(current)
        components = []
        if math.isfinite(base) and base >= 0.0:
            components.append(float(base))
        if math.isfinite(dense_floor) and dense_floor >= 0.0:
            components.append(float(dense_floor))
        rate = max(components) if components else float("nan")
        if math.isfinite(rate) and math.isfinite(scale) and scale >= 0.0:
            rate *= scale
        else:
            rate = float("nan")
        return {
            "rate_s": float(rate),
            "source": PLASMON_COLLISION_MODEL_BENCHMARK_DENSE,
            "summary": (
                f"benchmark_dense uses max(NRL={base:.3e}, dense_floor={dense_floor:.3e}) * scale={scale:.3g}; "
                f"{dense_summary}"
                if math.isfinite(rate)
                else f"benchmark_dense failed: NRL={base:.3e}, dense_floor={dense_floor:.3e}; {dense_summary}"
            ),
            "weak_coupling_required": False,
        }
    if not math.isfinite(base) or base < 0.0 or not math.isfinite(scale) or scale < 0.0:
        rate = float("nan")
    else:
        rate = base * scale
    return {
        "rate_s": float(rate),
        "source": PLASMON_COLLISION_MODEL_NRL_CONSTANT,
        "summary": (
            f"nrl_constant uses nu={base:.3e} 1/s * scale={scale:.3g}; lnLambda={ln_lambda:.3g}"
            if math.isfinite(rate)
            else "nrl_constant failed to resolve a finite non-negative collision rate"
        ),
        "weak_coupling_required": True,
    }


def _resolved_plasmon_collision_rate_s(current: dict[str, float], parameters: "DerivedAnalysisParameters") -> float:
    return float(_resolved_plasmon_collision_metadata(current, parameters)["rate_s"])


def _spectral_imag_shift_ev(current: dict[str, float], parameters: "DerivedAnalysisParameters") -> float:
    count = _resolved_energy_points(parameters)
    step = (2.0 * max(float(parameters.plasmon_energy_window_ev), 1.0)) / max(count - 1, 1)
    omega_pe_ev = abs(float(current.get("omega_pe_ev", float("nan"))))
    return max(0.01, 0.5 * step, 1.0e-4 * omega_pe_ev if math.isfinite(omega_pe_ev) else 0.0)


def _mark_valid_spectrum_payload(payload: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    payload["benchmark_status"] = _BENCHMARK_STATUS_VALID
    payload["model_executed_fully"] = True
    payload["fallback_fraction"] = 0.0
    payload.setdefault("domain_failure_fraction", 0.0)
    payload.setdefault("stls_converged", False)
    payload.setdefault("stls_iteration_count", 0)
    payload.setdefault("stls_convergence_residual", float("nan"))
    payload.setdefault("stls_convergence_relative_residual", float("nan"))
    payload.setdefault("stls_closure_name", "")
    payload.setdefault("stls_local_field_value", float("nan"))
    payload.setdefault("stls_q_over_qf", float("nan"))
    for key, value in _payload_issue_metadata().items():
        payload.setdefault(key, value)
    return payload


def _invalid_benchmark_spectrum_payload(
    parameters: "DerivedAnalysisParameters",
    *,
    fallback_fraction: float = 1.0,
    collision_rate_s: float = float("nan"),
    collision_source: str = "",
    collision_summary: str = "",
    imag_shift_ev: float = float("nan"),
    static_lfc_value: float = float("nan"),
    q_over_qf: float = float("nan"),
    zone_count_used: int = 0,
    cluster_count_used: int = 0,
    zone_issue_counts: dict[str, int] | None = None,
    cluster_issue_counts: dict[str, int] | None = None,
    resolved_model: str | None = None,
) -> dict[str, np.ndarray | float]:
    empty = np.asarray([], dtype=np.float64)
    fraction = float(fallback_fraction) if math.isfinite(float(fallback_fraction)) else 1.0
    fraction = min(max(fraction, 0.0), 1.0)
    backend_model = str(resolved_model or getattr(parameters, "plasmon_model", PLASMON_MODEL_QUICKLOOK) or PLASMON_MODEL_QUICKLOOK)
    payload = {
        "energy_ev": empty,
        "spectrum": empty,
        "loss": empty,
        "epsilon_real": empty,
        "epsilon_imag": empty,
        "peak_energy_ev": float("nan"),
        "peak_fwhm_ev": float("nan"),
        "peak_fit_method": "none",
        "imag_shift_ev": float(imag_shift_ev),
        "collision_rate_s": float(collision_rate_s),
        "collision_source": str(collision_source),
        "collision_summary": str(collision_summary),
        "static_lfc_value": float(static_lfc_value),
        "q_over_qf": float(q_over_qf),
        "zone_count_used": int(zone_count_used),
        "cluster_count_used": int(cluster_count_used),
        "benchmark_status": _BENCHMARK_STATUS_INVALID,
        "model_executed_fully": False,
        "fallback_fraction": fraction,
        "response_backend": _spectral_backend_tag(backend_model),
        "backend_summary": "",
        "stls_converged": False,
        "stls_iteration_count": 0,
        "stls_convergence_residual": float("nan"),
        "stls_convergence_relative_residual": float("nan"),
        "stls_closure_name": "",
        "stls_local_field_value": float("nan"),
        "stls_q_over_qf": float("nan"),
        "resolved_model_name": backend_model,
    }
    payload.update(_payload_issue_metadata(zone_counts=zone_issue_counts, cluster_counts=cluster_issue_counts, zone_total=int(zone_count_used), cluster_total=int(cluster_count_used)))
    del parameters
    return payload


def _spectral_backend_tag(model: str) -> str:
    if _uses_lindhard_backend(model):
        return "finite_t_lindhard"
    if _uses_stls_backend(model):
        return "finite_t_stls"
    if _uses_quantum_hydrodynamic_backend(model):
        return "quantum_hydrodynamic"
    return "classical_maxwellian"


def _effective_backend_model(current: dict[str, float], parameters: "DerivedAnalysisParameters", requested_model: str) -> str:
    model = str(requested_model or PLASMON_MODEL_QUICKLOOK)
    if model != PLASMON_MODEL_AUTO_BEST:
        return model
    theta = float(current.get("theta_degeneracy", float("nan")))
    collision_rate = float(_resolved_plasmon_collision_metadata(current, parameters)["rate_s"])
    collision_valid = math.isfinite(collision_rate) and collision_rate >= 0.0
    lfc_requested = str(parameters.plasmon_lfc_model or "none") == "esa_static"
    lfc_valid = lfc_requested and esa_domain_contains(float(current.get("wigner_seitz_rs", float("nan"))), theta)
    if collision_valid and lfc_valid:
        return PLASMON_MODEL_MERMIN_STATIC_LFC
    if lfc_valid:
        return PLASMON_MODEL_RPA_STATIC_LFC
    if collision_valid:
        return PLASMON_MODEL_MERMIN
    return PLASMON_MODEL_RPA


def _model_display_name(model: str) -> str:
    labels = {
        PLASMON_MODEL_QUICKLOOK: "Quick look",
        PLASMON_MODEL_RPA: "RPA",
        PLASMON_MODEL_MERMIN: "Mermin",
        PLASMON_MODEL_RPA_STATIC_LFC: "RPA + static LFC",
        PLASMON_MODEL_MERMIN_STATIC_LFC: "Mermin + static LFC",
        PLASMON_MODEL_LINDHARD: "Finite-T Lindhard",
        PLASMON_MODEL_LINDHARD_MERMIN: "Finite-T Lindhard + Mermin",
        PLASMON_MODEL_LINDHARD_STATIC_LFC: "Finite-T Lindhard + static LFC",
        PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC: "Finite-T Lindhard + Mermin + static LFC",
        PLASMON_MODEL_FINITE_T_STLS: "Finite-T STLS",
        PLASMON_MODEL_QUANTUM_HYDRODYNAMIC: "Quantum hydrodynamic",
        PLASMON_MODEL_AUTO_BEST: "Auto best",
    }
    return labels.get(str(model), str(model).replace("_", " ").title())


def _current_state_at_angle(
    current: dict[str, float],
    parameters: "DerivedAnalysisParameters",
    *,
    angle_deg: float,
) -> dict[str, float]:
    effective_parameters = replace(parameters, plasmon_scattering_angle_deg=float(angle_deg))
    return _plasmon_state_summary(
        float(current["te_ev"]),
        float(current["ti_ev"]),
        float(current["ne_cm3"]),
        float(current["zbar"]),
        float(current["ion_mass_mu"]),
        effective_parameters,
    )


def _build_spectrum_payload_generic(
    current: dict[str, float],
    parameters: "DerivedAnalysisParameters",
    *,
    requested_model: str,
    material_formula: str | None = None,
    analysis_cache: AnalysisStateCache | None = None,
) -> dict[str, np.ndarray | float]:
    resolved_model = _effective_backend_model(current, parameters, requested_model)
    collision_metadata = _resolved_plasmon_collision_metadata(current, parameters)
    resolved_collision_rate_s = float(collision_metadata["rate_s"])
    observable_mode = _plasmon_observable_mode(parameters)
    cache_key = (
        "plasmon.spectrum.generic",
        str(resolved_model),
        str(observable_mode),
        str(material_formula or ""),
        round(float(current.get("te_ev", float("nan"))), 10),
        round(float(current.get("ne_cm3", float("nan"))), 3),
        round(float(current.get("scattering_wavevector_m_inv", float("nan"))), 6),
        round(float(current.get("wigner_seitz_rs", float("nan"))), 6),
        round(float(current.get("theta_degeneracy", float("nan"))), 6),
        round(float(parameters.plasmon_energy_window_ev), 8),
        int(parameters.plasmon_energy_points),
        round(float(parameters.plasmon_instrument_fwhm_ev), 8),
        str(parameters.plasmon_normalization),
        str(parameters.plasmon_collision_model),
        round(float(parameters.plasmon_collision_scale), 10),
        round(float(parameters.plasmon_manual_collision_rate_s), 6),
        round(float(resolved_collision_rate_s), 6) if math.isfinite(float(resolved_collision_rate_s)) else "nan",
        str(collision_metadata.get("source", "")),
        str(parameters.plasmon_lfc_model),
        str(parameters.plasmon_execution_mode),
    )

    def _build() -> dict[str, np.ndarray | float]:
        energy = energy_axis_ev(float(parameters.plasmon_energy_window_ev), _resolved_energy_points(parameters))
        imag_shift_ev = _spectral_imag_shift_ev(current, parameters)
        benchmark = _plasmon_benchmark_mode(parameters)
        q_m_inv = float(current.get("scattering_wavevector_m_inv", float("nan")))
        te_ev = float(current.get("te_ev", float("nan")))
        ne_cm3 = float(current.get("ne_cm3", float("nan")))
        rs = float(current.get("wigner_seitz_rs", float("nan")))
        theta = float(current.get("theta_degeneracy", float("nan")))
        backend_summary = ""
        stls_converged = False
        stls_iteration_count = 0
        stls_convergence_residual = float("nan")
        stls_convergence_relative_residual = float("nan")
        stls_closure_name = ""
        stls_local_field_value = float("nan")
        stls_q_over_qf = float("nan")
        if resolved_model == PLASMON_MODEL_RPA:
            chi, epsilon = epsilon_rpa(energy, k_m_inv=q_m_inv, te_ev=te_ev, ne_cm3=ne_cm3, imag_shift_ev=float(imag_shift_ev))
            static_lfc_value = float("nan")
            q_over_qf = float("nan")
            collision_rate_s = float(current.get("collision_rate_s", float("nan")))
        elif resolved_model == PLASMON_MODEL_RPA_STATIC_LFC:
            chi, epsilon, static_lfc_value, q_over_qf = epsilon_rpa_static_lfc(energy, k_m_inv=q_m_inv, te_ev=te_ev, ne_cm3=ne_cm3, imag_shift_ev=float(imag_shift_ev), rs=rs, theta=theta)
            collision_rate_s = float(current.get("collision_rate_s", float("nan")))
        elif resolved_model == PLASMON_MODEL_MERMIN:
            chi, epsilon = epsilon_mermin(energy, k_m_inv=q_m_inv, te_ev=te_ev, ne_cm3=ne_cm3, collision_rate_s=float(resolved_collision_rate_s), imag_shift_ev=float(imag_shift_ev))
            static_lfc_value = float("nan")
            q_over_qf = float("nan")
            collision_rate_s = float(resolved_collision_rate_s)
        elif resolved_model == PLASMON_MODEL_MERMIN_STATIC_LFC:
            chi, epsilon, static_lfc_value, q_over_qf = epsilon_mermin_static_lfc(energy, k_m_inv=q_m_inv, te_ev=te_ev, ne_cm3=ne_cm3, collision_rate_s=float(resolved_collision_rate_s), imag_shift_ev=float(imag_shift_ev), rs=rs, theta=theta)
            collision_rate_s = float(resolved_collision_rate_s)
        elif resolved_model == PLASMON_MODEL_LINDHARD:
            chi, epsilon = epsilon_lindhard(energy, k_m_inv=q_m_inv, te_ev=te_ev, ne_cm3=ne_cm3, imag_shift_ev=float(imag_shift_ev), benchmark=benchmark)
            static_lfc_value = float("nan")
            q_over_qf = float("nan")
            collision_rate_s = float(current.get("collision_rate_s", float("nan")))
        elif resolved_model == PLASMON_MODEL_LINDHARD_STATIC_LFC:
            chi, epsilon, static_lfc_value, q_over_qf = epsilon_lindhard_static_lfc(energy, k_m_inv=q_m_inv, te_ev=te_ev, ne_cm3=ne_cm3, imag_shift_ev=float(imag_shift_ev), rs=rs, theta=theta, benchmark=benchmark)
            collision_rate_s = float(current.get("collision_rate_s", float("nan")))
        elif resolved_model == PLASMON_MODEL_LINDHARD_MERMIN:
            chi, epsilon = epsilon_lindhard_mermin(energy, k_m_inv=q_m_inv, te_ev=te_ev, ne_cm3=ne_cm3, collision_rate_s=float(resolved_collision_rate_s), imag_shift_ev=float(imag_shift_ev), benchmark=benchmark)
            static_lfc_value = float("nan")
            q_over_qf = float("nan")
            collision_rate_s = float(resolved_collision_rate_s)
        elif resolved_model == PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC:
            chi, epsilon, static_lfc_value, q_over_qf = epsilon_lindhard_mermin_static_lfc(energy, k_m_inv=q_m_inv, te_ev=te_ev, ne_cm3=ne_cm3, collision_rate_s=float(resolved_collision_rate_s), imag_shift_ev=float(imag_shift_ev), rs=rs, theta=theta, benchmark=benchmark)
            collision_rate_s = float(resolved_collision_rate_s)
        elif resolved_model == PLASMON_MODEL_FINITE_T_STLS:
            chi, epsilon, stls_metadata = epsilon_finite_t_stls(
                energy,
                k_m_inv=q_m_inv,
                te_ev=te_ev,
                ne_cm3=ne_cm3,
                imag_shift_ev=float(imag_shift_ev),
                benchmark=benchmark,
            )
            static_lfc_value = float("nan")
            q_over_qf = float("nan")
            collision_rate_s = 0.0
            backend_summary = str(stls_metadata.get("backend_summary", ""))
            stls_converged = bool(stls_metadata.get("converged", False))
            stls_iteration_count = int(stls_metadata.get("iterations", 0))
            stls_convergence_residual = float(stls_metadata.get("residual", float("nan")))
            stls_convergence_relative_residual = float(stls_metadata.get("relative_residual", float("nan")))
            stls_closure_name = str(stls_metadata.get("closure_name", ""))
            stls_local_field_value = float(stls_metadata.get("local_field_value", float("nan")))
            stls_q_over_qf = float(stls_metadata.get("q_over_qf", float("nan")))
            if not stls_converged:
                invalid_payload = _invalid_benchmark_spectrum_payload(
                    parameters,
                    fallback_fraction=1.0,
                    collision_rate_s=0.0,
                    collision_source="collisionless_stls",
                    collision_summary="STLS backend is collisionless.",
                    imag_shift_ev=float(imag_shift_ev),
                    zone_count_used=1,
                    cluster_count_used=1,
                    resolved_model=resolved_model,
                )
                invalid_payload.update(
                    {
                        "backend_summary": str(backend_summary),
                        "stls_converged": bool(stls_converged),
                        "stls_iteration_count": int(stls_iteration_count),
                        "stls_convergence_residual": float(stls_convergence_residual),
                        "stls_convergence_relative_residual": float(stls_convergence_relative_residual),
                        "stls_closure_name": str(stls_closure_name),
                        "stls_local_field_value": float(stls_local_field_value),
                        "stls_q_over_qf": float(stls_q_over_qf),
                    }
                )
                return invalid_payload
        elif resolved_model == PLASMON_MODEL_QUANTUM_HYDRODYNAMIC:
            chi, epsilon, hydrodynamic_metadata = epsilon_quantum_hydrodynamic(
                energy,
                k_m_inv=q_m_inv,
                te_ev=te_ev,
                ne_cm3=ne_cm3,
                collision_rate_s=float(resolved_collision_rate_s),
                imag_shift_ev=float(imag_shift_ev),
            )
            static_lfc_value = float("nan")
            q_over_qf = float("nan")
            collision_rate_s = float(resolved_collision_rate_s)
            backend_summary = str(hydrodynamic_metadata.get("backend_summary", ""))
        else:
            raise ValueError(f"Unsupported plasmon spectrum model: {resolved_model}")
        del chi
        loss = loss_function_from_epsilon(epsilon)
        raw_spectrum = dsf_from_loss(loss, energy, te_ev)
        observable_summary = ""
        observable_decomposition_mode = ""
        observable_peak_extraction_mode = "positive_branch"
        observable_elastic_exclusion_ev = 0.0
        observable_free_fraction = float("nan")
        observable_bound_fraction = float("nan")
        observable_elastic_fraction = float("nan")
        observable_comparison_mode = ""
        observable_subtraction_mode = ""
        observable_normalization_mode = ""
        observable_peak_discrete_energy_ev = float("nan")
        observable_peak_fit_energy_ev = float("nan")
        observable_peak_fit_status = ""
        observable_peak_edge_dominated = False
        observable_elastic_form_factor_total = float("nan")
        observable_elastic_form_factor_core = float("nan")
        observable_elastic_screening_form_factor = float("nan")
        observable_ion_structure_factor = float("nan")
        observable_bound_core_mode = ""
        observable_bound_shell_summary = ""
        free_component = np.asarray(raw_spectrum, dtype=np.float64)
        bound_component = np.zeros_like(free_component, dtype=np.float64)
        elastic_component = np.zeros_like(free_component, dtype=np.float64)
        peak_fit_method = _peak_fit_method(parameters)
        if observable_mode == PLASMON_OBSERVABLE_MODE_XRTS:
            components = build_minimal_xrts_components(
                energy,
                raw_spectrum,
                material_formula=material_formula,
                electron_density_cm3=ne_cm3,
                mean_charge=float(current.get("zbar", float("nan"))),
                scattering_wavevector_m_inv=q_m_inv,
                spectrum_window_ev=float(parameters.plasmon_energy_window_ev),
            )
            finalized = finalize_xrts_observable(
                energy,
                components,
                instrument_fwhm_ev=float(parameters.plasmon_instrument_fwhm_ev),
                normalization=str(parameters.plasmon_normalization or "peak"),
                peak_fit_method=peak_fit_method,
            )
            observed = np.asarray(finalized["spectrum"], dtype=np.float64)
            free_component = np.asarray(finalized["free_component"], dtype=np.float64)
            bound_component = np.asarray(finalized["bound_component"], dtype=np.float64)
            elastic_component = np.asarray(finalized["elastic_component"], dtype=np.float64)
            peak_energy_ev = float(finalized["peak_energy_ev"])
            peak_fwhm_ev = float(finalized["peak_fwhm_ev"])
            observable_summary = str(finalized.get("observable_summary", ""))
            observable_decomposition_mode = str(finalized.get("observable_decomposition_mode", ""))
            observable_peak_extraction_mode = str(finalized.get("observable_peak_extraction_mode", "positive_branch"))
            observable_elastic_exclusion_ev = float(finalized.get("observable_elastic_exclusion_ev", 0.0))
            observable_free_fraction = float(finalized.get("observable_free_fraction", float("nan")))
            observable_bound_fraction = float(finalized.get("observable_bound_fraction", float("nan")))
            observable_elastic_fraction = float(finalized.get("observable_elastic_fraction", float("nan")))
        elif observable_mode == PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE:
            components, observable_diagnostics = build_article_native_al_components(
                energy,
                raw_spectrum,
                material_formula=material_formula,
                electron_density_cm3=ne_cm3,
                mean_charge=float(current.get("zbar", float("nan"))),
                scattering_wavevector_m_inv=q_m_inv,
                spectrum_window_ev=float(parameters.plasmon_energy_window_ev),
            )
            finalized = finalize_article_native_observable(
                energy,
                components,
                instrument_fwhm_ev=float(parameters.plasmon_instrument_fwhm_ev),
                normalization=str(parameters.plasmon_normalization or "peak"),
                peak_fit_method=peak_fit_method,
                diagnostics=observable_diagnostics,
            )
            observed = np.asarray(finalized["spectrum"], dtype=np.float64)
            free_component = np.asarray(finalized["free_component"], dtype=np.float64)
            bound_component = np.asarray(finalized["bound_component"], dtype=np.float64)
            elastic_component = np.asarray(finalized["elastic_component"], dtype=np.float64)
            peak_energy_ev = float(finalized["peak_energy_ev"])
            peak_fwhm_ev = float(finalized["peak_fwhm_ev"])
            observable_summary = str(finalized.get("observable_summary", ""))
            observable_decomposition_mode = str(finalized.get("observable_decomposition_mode", ""))
            observable_peak_extraction_mode = str(finalized.get("observable_peak_extraction_mode", "positive_branch"))
            observable_elastic_exclusion_ev = float(finalized.get("observable_elastic_exclusion_ev", 0.0))
            observable_free_fraction = float(finalized.get("observable_free_fraction", float("nan")))
            observable_bound_fraction = float(finalized.get("observable_bound_fraction", float("nan")))
            observable_elastic_fraction = float(finalized.get("observable_elastic_fraction", float("nan")))
            observable_comparison_mode = str(finalized.get("observable_comparison_mode", ""))
            observable_subtraction_mode = str(finalized.get("observable_subtraction_mode", ""))
            observable_normalization_mode = str(finalized.get("observable_normalization_mode", ""))
            observable_peak_discrete_energy_ev = float(finalized.get("observable_peak_discrete_energy_ev", float("nan")))
            observable_peak_fit_energy_ev = float(finalized.get("observable_peak_fit_energy_ev", float("nan")))
            observable_peak_fit_status = str(finalized.get("observable_peak_fit_status", ""))
            observable_peak_edge_dominated = bool(finalized.get("observable_peak_edge_dominated", False))
            observable_elastic_form_factor_total = float(finalized.get("observable_elastic_form_factor_total", float("nan")))
            observable_elastic_form_factor_core = float(finalized.get("observable_elastic_form_factor_core", float("nan")))
            observable_elastic_screening_form_factor = float(finalized.get("observable_elastic_screening_form_factor", float("nan")))
            observable_ion_structure_factor = float(finalized.get("observable_ion_structure_factor", float("nan")))
            observable_bound_core_mode = str(finalized.get("observable_bound_core_mode", ""))
            observable_bound_shell_summary = str(finalized.get("observable_bound_shell_summary", ""))
        else:
            observed = gaussian_convolve(energy, raw_spectrum, float(parameters.plasmon_instrument_fwhm_ev))
            observed = normalize_spectrum(energy, observed, str(parameters.plasmon_normalization or "peak"))
            free_component = np.asarray(observed, dtype=np.float64)
            peak_energy_ev, peak_fwhm_ev, peak_fit_method = _spectrum_metric_tuple(energy, observed, parameters)
        payload = _mark_valid_spectrum_payload({
            "energy_ev": np.asarray(energy, dtype=np.float64),
            "spectrum": np.asarray(observed, dtype=np.float64),
            "free_component": np.asarray(free_component, dtype=np.float64),
            "bound_component": np.asarray(bound_component, dtype=np.float64),
            "elastic_component": np.asarray(elastic_component, dtype=np.float64),
            "loss": np.asarray(loss, dtype=np.float64),
            "epsilon_real": np.asarray(np.real(epsilon), dtype=np.float64),
            "epsilon_imag": np.asarray(np.imag(epsilon), dtype=np.float64),
            "peak_energy_ev": float(peak_energy_ev),
            "peak_fwhm_ev": float(peak_fwhm_ev),
            "peak_fit_method": str(peak_fit_method),
            "observable_mode": str(observable_mode),
            "observable_summary": str(observable_summary),
            "observable_decomposition_mode": str(observable_decomposition_mode),
            "observable_peak_extraction_mode": str(observable_peak_extraction_mode),
            "observable_elastic_exclusion_ev": float(observable_elastic_exclusion_ev),
            "observable_free_fraction": float(observable_free_fraction),
            "observable_bound_fraction": float(observable_bound_fraction),
            "observable_elastic_fraction": float(observable_elastic_fraction),
            "observable_comparison_mode": str(observable_comparison_mode),
            "observable_subtraction_mode": str(observable_subtraction_mode),
            "observable_normalization_mode": str(observable_normalization_mode),
            "observable_peak_discrete_energy_ev": float(observable_peak_discrete_energy_ev),
            "observable_peak_fit_energy_ev": float(observable_peak_fit_energy_ev),
            "observable_peak_fit_status": str(observable_peak_fit_status),
            "observable_peak_edge_dominated": bool(observable_peak_edge_dominated),
            "observable_elastic_form_factor_total": float(observable_elastic_form_factor_total),
            "observable_elastic_form_factor_core": float(observable_elastic_form_factor_core),
            "observable_elastic_screening_form_factor": float(observable_elastic_screening_form_factor),
            "observable_ion_structure_factor": float(observable_ion_structure_factor),
            "observable_bound_core_mode": str(observable_bound_core_mode),
            "observable_bound_shell_summary": str(observable_bound_shell_summary),
            "imag_shift_ev": float(imag_shift_ev),
            "collision_rate_s": float(collision_rate_s),
            "collision_source": ("collisionless_stls" if resolved_model == PLASMON_MODEL_FINITE_T_STLS else str(collision_metadata.get("source", ""))),
            "collision_summary": ("STLS backend is collisionless." if resolved_model == PLASMON_MODEL_FINITE_T_STLS else str(collision_metadata.get("summary", ""))),
            "static_lfc_value": float(static_lfc_value),
            "q_over_qf": float(q_over_qf),
            "response_backend": _spectral_backend_tag(resolved_model),
            "backend_summary": str(backend_summary),
            "stls_converged": bool(stls_converged),
            "stls_iteration_count": int(stls_iteration_count),
            "stls_convergence_residual": float(stls_convergence_residual),
            "stls_convergence_relative_residual": float(stls_convergence_relative_residual),
            "stls_closure_name": str(stls_closure_name),
            "stls_local_field_value": float(stls_local_field_value),
            "stls_q_over_qf": float(stls_q_over_qf),
            "resolved_model_name": str(resolved_model),
        })
        return payload

    return _augment_effective_state_domain_metadata(cached_time_series_payload(cache_key, analysis_cache=analysis_cache, builder=_build), current=current, parameters=parameters, requested_model=resolved_model)


def _build_rpa_spectrum_payload(current: dict[str, float], parameters: "DerivedAnalysisParameters", *, analysis_cache: AnalysisStateCache | None = None) -> dict[str, np.ndarray | float]:
    return _build_spectrum_payload_generic(current, parameters, requested_model=PLASMON_MODEL_RPA, analysis_cache=analysis_cache)


def _build_rpa_static_lfc_payload(current: dict[str, float], parameters: "DerivedAnalysisParameters", *, analysis_cache: AnalysisStateCache | None = None) -> dict[str, np.ndarray | float]:
    return _build_spectrum_payload_generic(current, parameters, requested_model=PLASMON_MODEL_RPA_STATIC_LFC, analysis_cache=analysis_cache)


def _build_mermin_spectrum_payload(current: dict[str, float], parameters: "DerivedAnalysisParameters", *, analysis_cache: AnalysisStateCache | None = None) -> dict[str, np.ndarray | float]:
    return _build_spectrum_payload_generic(current, parameters, requested_model=PLASMON_MODEL_MERMIN, analysis_cache=analysis_cache)


def _build_mermin_static_lfc_payload(current: dict[str, float], parameters: "DerivedAnalysisParameters", *, analysis_cache: AnalysisStateCache | None = None) -> dict[str, np.ndarray | float]:
    return _build_spectrum_payload_generic(current, parameters, requested_model=PLASMON_MODEL_MERMIN_STATIC_LFC, analysis_cache=analysis_cache)


def _build_lindhard_payload(current: dict[str, float], parameters: "DerivedAnalysisParameters", *, analysis_cache: AnalysisStateCache | None = None) -> dict[str, np.ndarray | float]:
    return _build_spectrum_payload_generic(current, parameters, requested_model=PLASMON_MODEL_LINDHARD, analysis_cache=analysis_cache)


def _build_lindhard_static_lfc_payload(current: dict[str, float], parameters: "DerivedAnalysisParameters", *, analysis_cache: AnalysisStateCache | None = None) -> dict[str, np.ndarray | float]:
    return _build_spectrum_payload_generic(current, parameters, requested_model=PLASMON_MODEL_LINDHARD_STATIC_LFC, analysis_cache=analysis_cache)


def _build_lindhard_mermin_payload(current: dict[str, float], parameters: "DerivedAnalysisParameters", *, analysis_cache: AnalysisStateCache | None = None) -> dict[str, np.ndarray | float]:
    return _build_spectrum_payload_generic(current, parameters, requested_model=PLASMON_MODEL_LINDHARD_MERMIN, analysis_cache=analysis_cache)


def _build_lindhard_mermin_static_lfc_payload(current: dict[str, float], parameters: "DerivedAnalysisParameters", *, analysis_cache: AnalysisStateCache | None = None) -> dict[str, np.ndarray | float]:
    return _build_spectrum_payload_generic(current, parameters, requested_model=PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC, analysis_cache=analysis_cache)


def _raw_spectrum_parameters(parameters: "DerivedAnalysisParameters") -> "DerivedAnalysisParameters":
    return replace(
        parameters,
        plasmon_instrument_fwhm_ev=0.0,
        plasmon_normalization="none",
        plasmon_observable_mode=PLASMON_OBSERVABLE_MODE_DIELECTRIC,
        plasmon_integration_mode="effective_state",
    )


def _build_los_integrated_spectrum_payload(
    requested_model: str,
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    current: dict[str, float],
    selection: object,
    mask: np.ndarray,
    weighting_mode: str,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    electron_fields: ElectronPolicyPayload,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> tuple[dict[str, np.ndarray | float], list[DerivedWarning], str]:
    del context
    observable_mode = _plasmon_observable_mode(parameters)
    material_formula = _observable_material_formula(dataset, mask, electron_fields)
    collision_metadata = _resolved_plasmon_collision_metadata(current, parameters)
    zone_states = _extract_zone_plasmon_states(
        dataset,
        snapshot_index=snapshot_index,
        mask=mask,
        selection=selection,
        weighting_mode=weighting_mode,
        geometry=geometry,
        electron_fields=electron_fields,
        analysis_cache=analysis_cache,
    )
    warnings: list[DerivedWarning] = []
    zone_count = int(np.asarray(zone_states["weight"], dtype=np.float64).size)
    zone_issue_counts = _empty_benchmark_issue_counts()
    for idx in range(zone_count):
        if progress_check is not None and (idx % 16 == 0):
            progress_check()
        state = _plasmon_state_summary(
            float(zone_states["te_ev"][idx]),
            float(zone_states["ti_ev"][idx]),
            float(zone_states["ne_cm3"][idx]),
            float(zone_states["zbar"][idx]),
            float(zone_states["ion_mass_mu"][idx]),
            parameters,
        )
        _accumulate_benchmark_issue_counts(zone_issue_counts, _benchmark_domain_flags(state, parameters, _effective_backend_model(state, parameters, requested_model)))
    clustered_states, cluster_signature = _cluster_zone_plasmon_states(
        dataset,
        snapshot_index=snapshot_index,
        selection=selection,
        zone_states=zone_states,
        weighting_mode=weighting_mode,
        geometry=geometry,
        parameters=parameters,
        electron_fields=electron_fields,
        analysis_cache=analysis_cache,
    )
    cluster_count = int(np.asarray(clustered_states["weight"], dtype=np.float64).size)
    cluster_issue_counts = _empty_benchmark_issue_counts()
    for idx in range(cluster_count):
        if progress_check is not None and (idx % 8 == 0):
            progress_check()
        state = _plasmon_state_summary(
            float(clustered_states["te_ev"][idx]),
            float(clustered_states["ti_ev"][idx]),
            float(clustered_states["ne_cm3"][idx]),
            float(clustered_states["zbar"][idx]),
            float(clustered_states["ion_mass_mu"][idx]),
            parameters,
        )
        _accumulate_benchmark_issue_counts(cluster_issue_counts, _benchmark_domain_flags(state, parameters, _effective_backend_model(state, parameters, requested_model)))
    if cluster_count == 0:
        energy = energy_axis_ev(float(parameters.plasmon_energy_window_ev), _resolved_energy_points(parameters))
        empty = np.zeros_like(energy, dtype=np.float64)
        return _invalid_benchmark_spectrum_payload(
            parameters,
            fallback_fraction=1.0,
            collision_rate_s=float("nan"),
            collision_source=str(collision_metadata.get("source", "")),
            collision_summary=str(collision_metadata.get("summary", "")),
            imag_shift_ev=float("nan"),
            static_lfc_value=float("nan"),
            q_over_qf=float("nan"),
            zone_count_used=0,
            cluster_count_used=0,
            zone_issue_counts=zone_issue_counts,
            cluster_issue_counts=cluster_issue_counts,
            resolved_model=requested_model,
        ), warnings, requested_model

    cache_key = (
        "plasmon.spectrum.integrated",
        str(requested_model),
        str(observable_mode),
        str(material_formula or ""),
        round(float(current.get("scattering_wavevector_m_inv", float("nan"))), 6),
        round(float(parameters.plasmon_energy_window_ev), 8),
        int(parameters.plasmon_energy_points),
        round(float(parameters.plasmon_instrument_fwhm_ev), 8),
        str(parameters.plasmon_normalization),
        str(parameters.plasmon_collision_model),
        round(float(parameters.plasmon_collision_scale), 10),
        round(float(parameters.plasmon_manual_collision_rate_s), 6),
        str(parameters.plasmon_lfc_model),
        round(float(parameters.plasmon_cluster_log_ne_tol), 8),
        round(float(parameters.plasmon_cluster_log_te_tol), 8),
        round(float(parameters.plasmon_cluster_z_tol), 8),
        str(getattr(parameters, "plasmon_electron_policy", PLASMON_ELECTRON_POLICY_RAW)),
        cluster_signature,
    )
    raw_parameters = _raw_spectrum_parameters(parameters)
    collision_label = _collision_model_label(str(parameters.plasmon_collision_model or PLASMON_COLLISION_MODEL_NRL_CONSTANT))
    if _plasmon_benchmark_mode(parameters):
        if int(zone_issue_counts.get(_NONCOLLECTIVE_ISSUE, 0)) > 0:
            warnings.append(DerivedWarning("plasmon", f"Benchmark LOS mode rejected the spectrum because {int(zone_issue_counts[_NONCOLLECTIVE_ISSUE])}/{zone_count} selected zones are in a non-collective regime (k*lambda_D >= 1). Exclude those zones in the left panel or lower q before treating the result as a plasmon benchmark.", severity="warning"))
        if _uses_collision_branch(requested_model) and int(zone_issue_counts.get(_WEAK_COUPLING_ISSUE, 0)) > 0:
            warnings.append(DerivedWarning("plasmon", f"Benchmark LOS mode rejected the spectrum because {int(zone_issue_counts[_WEAK_COUPLING_ISSUE])}/{zone_count} selected zones have Coulomb logarithm <= 2, so the chosen collision closure ({collision_label}) is not trustworthy there.", severity="warning"))
        if _uses_static_lfc(requested_model) and int(zone_issue_counts.get(_LFC_OUT_OF_DOMAIN_ISSUE, 0)) > 0:
            warnings.append(DerivedWarning("plasmon", f"Benchmark LOS mode rejected the spectrum because {int(zone_issue_counts[_LFC_OUT_OF_DOMAIN_ISSUE])}/{zone_count} selected zones lie outside the nominal ESA static-LFC domain before any state clustering. This prevents cluster averaging from hiding invalid edge zones.", severity="warning"))
        if _uses_collision_branch(requested_model) and int(zone_issue_counts.get(_INVALID_COLLISION_ISSUE, 0)) > 0:
            warnings.append(DerivedWarning("plasmon", f"Benchmark LOS mode rejected the spectrum because {int(zone_issue_counts[_INVALID_COLLISION_ISSUE])}/{zone_count} selected zones did not resolve a finite non-negative collision rate for the chosen collision closure ({collision_label}).", severity="warning"))
        if _benchmark_issue_count(zone_issue_counts, hard_only=True) > 0 or _benchmark_issue_count(cluster_issue_counts, hard_only=True) > 0:
            return _invalid_benchmark_spectrum_payload(
                parameters,
                fallback_fraction=0.0,
                collision_rate_s=float(collision_metadata.get("rate_s", float("nan"))),
                collision_source=str(collision_metadata.get("source", "")),
                collision_summary=str(collision_metadata.get("summary", "")),
                imag_shift_ev=_spectral_imag_shift_ev(current, parameters),
                static_lfc_value=float("nan"),
                q_over_qf=float("nan"),
                zone_count_used=int(zone_count),
                cluster_count_used=int(cluster_count),
                zone_issue_counts=zone_issue_counts,
                cluster_issue_counts=cluster_issue_counts,
                resolved_model=requested_model,
            ), warnings, requested_model

    def _build() -> dict[str, np.ndarray | float]:
        weights = np.asarray(clustered_states["weight"], dtype=np.float64)
        total_weight = float(np.sum(weights, dtype=np.float64))
        fractions = np.divide(weights, total_weight, out=np.zeros_like(weights), where=total_weight > 0.0)
        observed_raw: np.ndarray | None = None
        free_raw_sum: np.ndarray | None = None
        bound_raw_sum: np.ndarray | None = None
        elastic_raw_sum: np.ndarray | None = None
        loss_average: np.ndarray | None = None
        eps_real_average: np.ndarray | None = None
        eps_imag_average: np.ndarray | None = None
        weighted_collision = 0.0
        weighted_imag_shift = 0.0
        weighted_static_lfc = 0.0
        weighted_q_over_qf = 0.0
        weighted_stls_local_field = 0.0
        weighted_stls_q_over_qf = 0.0
        weighted_stls_residual = 0.0
        weighted_stls_relative_residual = 0.0
        weighted_stls_iterations = 0.0
        any_static_lfc = False
        any_stls = False
        cluster_fallbacks = 0
        cluster_nonconverged = 0
        out_of_domain_clusters = 0
        energy_template: np.ndarray | None = None
        model_counts: dict[str, int] = {}
        stls_closure_names: set[str] = set()
        backend_summaries: set[str] = set()
        observable_summaries: set[str] = set()
        observable_decomposition_modes: set[str] = set()
        observable_free_fraction_weighted = 0.0
        observable_bound_fraction_weighted = 0.0
        observable_elastic_fraction_weighted = 0.0
        observable_fraction_weight_total = 0.0
        for idx in range(cluster_count):
            if progress_check is not None and (idx % 4 == 0):
                progress_check()
            state = _plasmon_state_summary(
                float(clustered_states["te_ev"][idx]),
                float(clustered_states["ti_ev"][idx]),
                float(clustered_states["ne_cm3"][idx]),
                float(clustered_states["zbar"][idx]),
                float(clustered_states["ion_mass_mu"][idx]),
                parameters,
            )
            weight = float(weights[idx])
            local_model = _effective_backend_model(state, parameters, requested_model)
            model_counts[local_model] = int(model_counts.get(local_model, 0)) + 1
            if _uses_static_lfc(local_model) and not esa_domain_contains(float(state.get("wigner_seitz_rs", float("nan"))), float(state.get("theta_degeneracy", float("nan")))):
                out_of_domain_clusters += 1
            if _uses_collision_branch(local_model):
                collision_rate = _resolved_plasmon_collision_rate_s(state, parameters)
                if not math.isfinite(float(collision_rate)) or float(collision_rate) < 0.0:
                    cluster_fallbacks += 1
                    continue
            payload = _build_spectrum_payload_generic(
                state,
                raw_parameters,
                requested_model=local_model,
                material_formula=material_formula,
                analysis_cache=analysis_cache,
            )
            if _uses_stls_backend(local_model):
                any_stls = True
                if not bool(payload.get("stls_converged", False)):
                    cluster_nonconverged += 1
                    continue
                if math.isfinite(float(payload.get("stls_local_field_value", float("nan")))):
                    weighted_stls_local_field += weight * float(payload.get("stls_local_field_value", float("nan")))
                if math.isfinite(float(payload.get("stls_q_over_qf", float("nan")))):
                    weighted_stls_q_over_qf += weight * float(payload.get("stls_q_over_qf", float("nan")))
                if math.isfinite(float(payload.get("stls_convergence_residual", float("nan")))):
                    weighted_stls_residual += fractions[idx] * float(payload.get("stls_convergence_residual", float("nan")))
                if math.isfinite(float(payload.get("stls_convergence_relative_residual", float("nan")))):
                    weighted_stls_relative_residual += fractions[idx] * float(payload.get("stls_convergence_relative_residual", float("nan")))
                if math.isfinite(float(payload.get("stls_iteration_count", float("nan")))):
                    weighted_stls_iterations += fractions[idx] * float(payload.get("stls_iteration_count", float("nan")))
                if str(payload.get("stls_closure_name", "")).strip():
                    stls_closure_names.add(str(payload.get("stls_closure_name", "")))
            if str(payload.get("backend_summary", "")).strip():
                backend_summaries.add(str(payload.get("backend_summary", "")))
            if _uses_static_lfc(local_model):
                any_static_lfc = True
                if math.isfinite(float(payload.get("static_lfc_value", float("nan")))):
                    weighted_static_lfc += weight * float(payload.get("static_lfc_value", float("nan")))
                if math.isfinite(float(payload.get("q_over_qf", float("nan")))):
                    weighted_q_over_qf += weight * float(payload.get("q_over_qf", float("nan")))
            if math.isfinite(float(payload.get("collision_rate_s", float("nan")))):
                weighted_collision += weight * float(payload.get("collision_rate_s", float("nan")))
            energy = np.asarray(payload["energy_ev"], dtype=np.float64)
            if energy_template is None:
                energy_template = energy
                observed_raw = np.zeros_like(energy_template, dtype=np.float64)
                free_raw_sum = np.zeros_like(energy_template, dtype=np.float64)
                bound_raw_sum = np.zeros_like(energy_template, dtype=np.float64)
                elastic_raw_sum = np.zeros_like(energy_template, dtype=np.float64)
                loss_average = np.zeros_like(energy_template, dtype=np.float64)
                eps_real_average = np.zeros_like(energy_template, dtype=np.float64)
                eps_imag_average = np.zeros_like(energy_template, dtype=np.float64)
            cluster_spectrum = np.asarray(payload["spectrum"], dtype=np.float64)
            assert observed_raw is not None and free_raw_sum is not None and bound_raw_sum is not None and elastic_raw_sum is not None and loss_average is not None and eps_real_average is not None and eps_imag_average is not None
            if observable_mode == PLASMON_OBSERVABLE_MODE_XRTS:
                components = build_minimal_xrts_components(
                    energy,
                    cluster_spectrum,
                    material_formula=material_formula,
                    electron_density_cm3=float(state.get("ne_cm3", float("nan"))),
                    mean_charge=float(state.get("zbar", float("nan"))),
                    scattering_wavevector_m_inv=float(state.get("scattering_wavevector_m_inv", float("nan"))),
                    spectrum_window_ev=float(parameters.plasmon_energy_window_ev),
                )
                free_raw_sum += weight * np.asarray(components.free_raw, dtype=np.float64)
                bound_raw_sum += weight * np.asarray(components.bound_raw, dtype=np.float64)
                elastic_raw_sum += weight * np.asarray(components.elastic_raw, dtype=np.float64)
                observed_raw += weight * np.asarray(components.total_raw, dtype=np.float64)
                if str(components.summary).strip():
                    observable_summaries.add(str(components.summary))
                if str(components.decomposition_mode).strip():
                    observable_decomposition_modes.add(str(components.decomposition_mode))
                if math.isfinite(float(components.free_fraction)):
                    observable_free_fraction_weighted += fractions[idx] * float(components.free_fraction)
                    observable_fraction_weight_total += fractions[idx]
                if math.isfinite(float(components.bound_fraction)):
                    observable_bound_fraction_weighted += fractions[idx] * float(components.bound_fraction)
                if math.isfinite(float(components.elastic_fraction)):
                    observable_elastic_fraction_weighted += fractions[idx] * float(components.elastic_fraction)
            else:
                observed_raw += weight * cluster_spectrum
                free_raw_sum += weight * cluster_spectrum
            loss_average += fractions[idx] * np.asarray(payload["loss"], dtype=np.float64)
            eps_real_average += fractions[idx] * np.asarray(payload["epsilon_real"], dtype=np.float64)
            eps_imag_average += fractions[idx] * np.asarray(payload["epsilon_imag"], dtype=np.float64)
            if math.isfinite(float(payload.get("imag_shift_ev", float("nan")))):
                weighted_imag_shift += fractions[idx] * float(payload.get("imag_shift_ev", float("nan")))
        invalid_cluster_fraction = float(cluster_fallbacks + cluster_nonconverged)
        fallback_fraction = (invalid_cluster_fraction / float(cluster_count)) if cluster_count > 0 else 1.0
        if cluster_fallbacks > 0 or cluster_nonconverged > 0:
            return _invalid_benchmark_spectrum_payload(
                parameters,
                fallback_fraction=fallback_fraction,
                collision_rate_s=(float(weighted_collision / total_weight) if total_weight > 0.0 else float("nan")),
                collision_source=("collisionless_stls" if requested_model == PLASMON_MODEL_FINITE_T_STLS else str(collision_metadata.get("source", ""))),
                collision_summary=("STLS backend is collisionless." if requested_model == PLASMON_MODEL_FINITE_T_STLS else str(collision_metadata.get("summary", ""))),
                imag_shift_ev=float(weighted_imag_shift),
                static_lfc_value=(float(weighted_static_lfc / total_weight) if any_static_lfc and total_weight > 0.0 else float("nan")),
                q_over_qf=(float(weighted_q_over_qf / total_weight) if any_static_lfc and total_weight > 0.0 else float("nan")),
                zone_count_used=int(zone_count),
                cluster_count_used=int(cluster_count),
                zone_issue_counts=zone_issue_counts,
                cluster_issue_counts=cluster_issue_counts,
                resolved_model=("mixed" if len(model_counts) > 1 else next(iter(model_counts), requested_model)),
            ) | {
                "cluster_fallback_count": int(cluster_fallbacks),
                "stls_nonconverged_cluster_count": int(cluster_nonconverged),
                "out_of_domain_cluster_count": int(out_of_domain_clusters),
                "total_weight": float(total_weight),
                "auto_model_summary": ", ".join(f"{name}:{count}" for name, count in sorted(model_counts.items())),
                "response_backend": ("mixed" if len(model_counts) > 1 else _spectral_backend_tag(next(iter(model_counts), requested_model))),
                "backend_summary": " || ".join(sorted(backend_summaries)),
                "stls_converged": False,
                "stls_iteration_count": int(round(weighted_stls_iterations)),
                "stls_convergence_residual": float(weighted_stls_residual),
                "stls_convergence_relative_residual": float(weighted_stls_relative_residual),
                "stls_closure_name": " || ".join(sorted(stls_closure_names)),
                "stls_local_field_value": (float(weighted_stls_local_field / total_weight) if any_stls and total_weight > 0.0 else float("nan")),
                "stls_q_over_qf": (float(weighted_stls_q_over_qf / total_weight) if any_stls and total_weight > 0.0 else float("nan")),
                "resolved_model_name": ("mixed" if len(model_counts) > 1 else next(iter(model_counts), requested_model)),
            }
        assert energy_template is not None and observed_raw is not None and loss_average is not None and eps_real_average is not None and eps_imag_average is not None
        observable_summary = ""
        observable_decomposition_mode = ""
        observable_peak_extraction_mode = "positive_branch"
        observable_elastic_exclusion_ev = 0.0
        observable_free_fraction = float("nan")
        observable_bound_fraction = float("nan")
        observable_elastic_fraction = float("nan")
        observable_comparison_mode = ""
        observable_subtraction_mode = ""
        observable_normalization_mode = ""
        observable_peak_discrete_energy_ev = float("nan")
        observable_peak_fit_energy_ev = float("nan")
        observable_peak_fit_status = ""
        observable_peak_edge_dominated = False
        observable_elastic_form_factor_total = float("nan")
        observable_elastic_form_factor_core = float("nan")
        observable_elastic_screening_form_factor = float("nan")
        observable_ion_structure_factor = float("nan")
        observable_bound_core_mode = ""
        observable_bound_shell_summary = ""
        peak_fit_method = _peak_fit_method(parameters)
        if observable_mode == PLASMON_OBSERVABLE_MODE_XRTS:
            assert free_raw_sum is not None and bound_raw_sum is not None and elastic_raw_sum is not None
            components = build_minimal_xrts_components(
                energy_template,
                np.asarray(free_raw_sum, dtype=np.float64),
                material_formula=material_formula,
                electron_density_cm3=float(current.get("ne_cm3", float("nan"))),
                mean_charge=float(current.get("zbar", float("nan"))),
                scattering_wavevector_m_inv=float(current.get("scattering_wavevector_m_inv", float("nan"))),
                spectrum_window_ev=float(parameters.plasmon_energy_window_ev),
            )
            components = components.__class__(
                mode=components.mode,
                decomposition_mode=components.decomposition_mode,
                summary=components.summary,
                provenance=components.provenance,
                free_raw=np.asarray(free_raw_sum, dtype=np.float64),
                bound_raw=np.asarray(bound_raw_sum, dtype=np.float64),
                elastic_raw=np.asarray(elastic_raw_sum, dtype=np.float64),
                total_raw=np.asarray(free_raw_sum, dtype=np.float64) + np.asarray(bound_raw_sum, dtype=np.float64) + np.asarray(elastic_raw_sum, dtype=np.float64),
                free_fraction=components.free_fraction,
                bound_fraction=components.bound_fraction,
                elastic_fraction=components.elastic_fraction,
            )
            finalized = finalize_xrts_observable(
                energy_template,
                components,
                instrument_fwhm_ev=float(parameters.plasmon_instrument_fwhm_ev),
                normalization=str(parameters.plasmon_normalization or "peak"),
                peak_fit_method=peak_fit_method,
            )
            observed = np.asarray(finalized["spectrum"], dtype=np.float64)
            free_component = np.asarray(finalized["free_component"], dtype=np.float64)
            bound_component = np.asarray(finalized["bound_component"], dtype=np.float64)
            elastic_component = np.asarray(finalized["elastic_component"], dtype=np.float64)
            peak_energy_ev = float(finalized["peak_energy_ev"])
            peak_fwhm_ev = float(finalized["peak_fwhm_ev"])
            observable_summary = " || ".join(sorted(observable_summaries)) if observable_summaries else str(finalized.get("observable_summary", ""))
            observable_decomposition_mode = " || ".join(sorted(observable_decomposition_modes)) if observable_decomposition_modes else str(finalized.get("observable_decomposition_mode", ""))
            observable_peak_extraction_mode = str(finalized.get("observable_peak_extraction_mode", "positive_branch"))
            observable_elastic_exclusion_ev = float(finalized.get("observable_elastic_exclusion_ev", 0.0))
            if observable_fraction_weight_total > 0.0:
                observable_free_fraction = float(observable_free_fraction_weighted / observable_fraction_weight_total)
                observable_bound_fraction = float(observable_bound_fraction_weighted / observable_fraction_weight_total)
                observable_elastic_fraction = float(observable_elastic_fraction_weighted / observable_fraction_weight_total)
            else:
                observable_free_fraction = float(finalized.get("observable_free_fraction", float("nan")))
                observable_bound_fraction = float(finalized.get("observable_bound_fraction", float("nan")))
                observable_elastic_fraction = float(finalized.get("observable_elastic_fraction", float("nan")))
        elif observable_mode == PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE:
            assert free_raw_sum is not None and bound_raw_sum is not None and elastic_raw_sum is not None
            components, observable_diagnostics = build_article_native_al_components(
                energy_template,
                np.asarray(free_raw_sum, dtype=np.float64),
                material_formula=material_formula,
                electron_density_cm3=float(current.get("ne_cm3", float("nan"))),
                mean_charge=float(current.get("zbar", float("nan"))),
                scattering_wavevector_m_inv=float(current.get("scattering_wavevector_m_inv", float("nan"))),
                spectrum_window_ev=float(parameters.plasmon_energy_window_ev),
            )
            components = components.__class__(
                mode=components.mode,
                decomposition_mode=components.decomposition_mode,
                summary=components.summary,
                provenance=components.provenance,
                free_raw=np.asarray(free_raw_sum, dtype=np.float64),
                bound_raw=np.asarray(bound_raw_sum, dtype=np.float64),
                elastic_raw=np.asarray(elastic_raw_sum, dtype=np.float64),
                total_raw=np.asarray(free_raw_sum, dtype=np.float64) + np.asarray(bound_raw_sum, dtype=np.float64) + np.asarray(elastic_raw_sum, dtype=np.float64),
                free_fraction=components.free_fraction,
                bound_fraction=components.bound_fraction,
                elastic_fraction=components.elastic_fraction,
            )
            finalized = finalize_article_native_observable(
                energy_template,
                components,
                instrument_fwhm_ev=float(parameters.plasmon_instrument_fwhm_ev),
                normalization=str(parameters.plasmon_normalization or "peak"),
                peak_fit_method=peak_fit_method,
                diagnostics=observable_diagnostics,
            )
            observed = np.asarray(finalized["spectrum"], dtype=np.float64)
            free_component = np.asarray(finalized["free_component"], dtype=np.float64)
            bound_component = np.asarray(finalized["bound_component"], dtype=np.float64)
            elastic_component = np.asarray(finalized["elastic_component"], dtype=np.float64)
            peak_energy_ev = float(finalized["peak_energy_ev"])
            peak_fwhm_ev = float(finalized["peak_fwhm_ev"])
            observable_summary = " || ".join(sorted(observable_summaries)) if observable_summaries else str(finalized.get("observable_summary", ""))
            observable_decomposition_mode = " || ".join(sorted(observable_decomposition_modes)) if observable_decomposition_modes else str(finalized.get("observable_decomposition_mode", ""))
            observable_peak_extraction_mode = str(finalized.get("observable_peak_extraction_mode", "positive_branch"))
            observable_elastic_exclusion_ev = float(finalized.get("observable_elastic_exclusion_ev", 0.0))
            if observable_fraction_weight_total > 0.0:
                observable_free_fraction = float(observable_free_fraction_weighted / observable_fraction_weight_total)
                observable_bound_fraction = float(observable_bound_fraction_weighted / observable_fraction_weight_total)
                observable_elastic_fraction = float(observable_elastic_fraction_weighted / observable_fraction_weight_total)
            else:
                observable_free_fraction = float(finalized.get("observable_free_fraction", float("nan")))
                observable_bound_fraction = float(finalized.get("observable_bound_fraction", float("nan")))
                observable_elastic_fraction = float(finalized.get("observable_elastic_fraction", float("nan")))
            observable_comparison_mode = str(finalized.get("observable_comparison_mode", ""))
            observable_subtraction_mode = str(finalized.get("observable_subtraction_mode", ""))
            observable_normalization_mode = str(finalized.get("observable_normalization_mode", ""))
            observable_peak_discrete_energy_ev = float(finalized.get("observable_peak_discrete_energy_ev", float("nan")))
            observable_peak_fit_energy_ev = float(finalized.get("observable_peak_fit_energy_ev", float("nan")))
            observable_peak_fit_status = str(finalized.get("observable_peak_fit_status", ""))
            observable_peak_edge_dominated = bool(finalized.get("observable_peak_edge_dominated", False))
            observable_elastic_form_factor_total = float(finalized.get("observable_elastic_form_factor_total", float("nan")))
            observable_elastic_form_factor_core = float(finalized.get("observable_elastic_form_factor_core", float("nan")))
            observable_elastic_screening_form_factor = float(finalized.get("observable_elastic_screening_form_factor", float("nan")))
            observable_ion_structure_factor = float(finalized.get("observable_ion_structure_factor", float("nan")))
            observable_bound_core_mode = str(finalized.get("observable_bound_core_mode", ""))
            observable_bound_shell_summary = str(finalized.get("observable_bound_shell_summary", ""))
        else:
            observed = gaussian_convolve(energy_template, observed_raw, float(parameters.plasmon_instrument_fwhm_ev))
            observed = normalize_spectrum(energy_template, observed, str(parameters.plasmon_normalization or "peak"))
            free_component = np.asarray(observed, dtype=np.float64)
            bound_component = np.zeros_like(free_component, dtype=np.float64)
            elastic_component = np.zeros_like(free_component, dtype=np.float64)
            peak_energy_ev, peak_fwhm_ev, peak_fit_method = _spectrum_metric_tuple(energy_template, observed, parameters)
        payload = _mark_valid_spectrum_payload({
            "energy_ev": np.asarray(energy_template, dtype=np.float64),
            "spectrum": np.asarray(observed, dtype=np.float64),
            "free_component": np.asarray(free_component, dtype=np.float64),
            "bound_component": np.asarray(bound_component, dtype=np.float64),
            "elastic_component": np.asarray(elastic_component, dtype=np.float64),
            "loss": np.asarray(loss_average, dtype=np.float64),
            "epsilon_real": np.asarray(eps_real_average, dtype=np.float64),
            "epsilon_imag": np.asarray(eps_imag_average, dtype=np.float64),
            "peak_energy_ev": float(peak_energy_ev),
            "peak_fwhm_ev": float(peak_fwhm_ev),
            "peak_fit_method": str(peak_fit_method),
            "observable_mode": str(observable_mode),
            "observable_summary": str(observable_summary),
            "observable_decomposition_mode": str(observable_decomposition_mode),
            "observable_peak_extraction_mode": str(observable_peak_extraction_mode),
            "observable_elastic_exclusion_ev": float(observable_elastic_exclusion_ev),
            "observable_free_fraction": float(observable_free_fraction),
            "observable_bound_fraction": float(observable_bound_fraction),
            "observable_elastic_fraction": float(observable_elastic_fraction),
            "observable_comparison_mode": str(observable_comparison_mode),
            "observable_subtraction_mode": str(observable_subtraction_mode),
            "observable_normalization_mode": str(observable_normalization_mode),
            "observable_peak_discrete_energy_ev": float(observable_peak_discrete_energy_ev),
            "observable_peak_fit_energy_ev": float(observable_peak_fit_energy_ev),
            "observable_peak_fit_status": str(observable_peak_fit_status),
            "observable_peak_edge_dominated": bool(observable_peak_edge_dominated),
            "observable_elastic_form_factor_total": float(observable_elastic_form_factor_total),
            "observable_elastic_form_factor_core": float(observable_elastic_form_factor_core),
            "observable_elastic_screening_form_factor": float(observable_elastic_screening_form_factor),
            "observable_ion_structure_factor": float(observable_ion_structure_factor),
            "observable_bound_core_mode": str(observable_bound_core_mode),
            "observable_bound_shell_summary": str(observable_bound_shell_summary),
            "imag_shift_ev": float(weighted_imag_shift),
            "collision_rate_s": (float(weighted_collision / total_weight) if total_weight > 0.0 else float("nan")),
            "collision_source": str(collision_metadata.get("source", "")),
            "collision_summary": str(collision_metadata.get("summary", "")),
            "static_lfc_value": (float(weighted_static_lfc / total_weight) if any_static_lfc and total_weight > 0.0 else float("nan")),
            "q_over_qf": (float(weighted_q_over_qf / total_weight) if any_static_lfc and total_weight > 0.0 else float("nan")),
            "zone_count_used": int(zone_count),
            "cluster_count_used": int(cluster_count),
            "cluster_fallback_count": int(cluster_fallbacks),
            "stls_nonconverged_cluster_count": int(cluster_nonconverged),
            "out_of_domain_cluster_count": int(out_of_domain_clusters),
            "total_weight": float(total_weight),
            "auto_model_summary": ", ".join(f"{name}:{count}" for name, count in sorted(model_counts.items())),
            "response_backend": ("mixed" if len(model_counts) > 1 else _spectral_backend_tag(next(iter(model_counts), requested_model))),
            "backend_summary": (
                "LOS-integrated QHD response averaged clustered damped quantum-hydrodynamic dielectric states with beta_eff^2 = 3 v_th^2 + 3/5 v_F^2 and Bohm recoil retained."
                if requested_model == PLASMON_MODEL_QUANTUM_HYDRODYNAMIC and len(model_counts) == 1
                else " || ".join(sorted(backend_summaries))
            ),
            "stls_converged": (bool(any_stls) and cluster_nonconverged == 0),
            "stls_iteration_count": int(round(weighted_stls_iterations)),
            "stls_convergence_residual": float(weighted_stls_residual),
            "stls_convergence_relative_residual": float(weighted_stls_relative_residual),
            "stls_closure_name": " || ".join(sorted(stls_closure_names)),
            "stls_local_field_value": (float(weighted_stls_local_field / total_weight) if any_stls and total_weight > 0.0 else float("nan")),
            "stls_q_over_qf": (float(weighted_stls_q_over_qf / total_weight) if any_stls and total_weight > 0.0 else float("nan")),
            "resolved_model_name": ("mixed" if len(model_counts) > 1 else next(iter(model_counts), requested_model)),
        })
        payload.update(_payload_issue_metadata(zone_counts=zone_issue_counts, cluster_counts=cluster_issue_counts, zone_total=int(zone_count), cluster_total=int(cluster_count)))
        return payload

    payload = cached_time_series_payload(cache_key, analysis_cache=analysis_cache, builder=_build)
    applied_model = requested_model
    collision_label = _collision_model_label(str(parameters.plasmon_collision_model or PLASMON_COLLISION_MODEL_NRL_CONSTANT))
    if requested_model == PLASMON_MODEL_MERMIN and int(payload.get("cluster_fallback_count", 0)) > 0:
        warnings.append(
            DerivedWarning(
                "plasmon",
                f"LOS-integrated Mermin spectrum did not execute fully; benchmark status is invalid because {int(payload['cluster_fallback_count'])}/{cluster_count} clusters did not resolve a finite non-negative collision rate for the selected collision closure ({collision_label}).",
                severity="caution",
            )
        )
    if requested_model == PLASMON_MODEL_MERMIN_STATIC_LFC and int(payload.get("cluster_fallback_count", 0)) > 0:
        warnings.append(
            DerivedWarning(
                "plasmon",
                f"LOS-integrated Mermin + static LFC spectrum did not execute fully; benchmark status is invalid because {int(payload['cluster_fallback_count'])}/{cluster_count} clusters did not resolve a finite non-negative collision rate for the selected collision closure ({collision_label}).",
                severity="caution",
            )
        )
    if requested_model == PLASMON_MODEL_LINDHARD_MERMIN and int(payload.get("cluster_fallback_count", 0)) > 0:
        warnings.append(
            DerivedWarning(
                "plasmon",
                f"LOS-integrated finite-T Lindhard + Mermin spectrum did not execute fully; benchmark status is invalid because {int(payload['cluster_fallback_count'])}/{cluster_count} clusters did not resolve a finite non-negative collision rate for the selected Mermin closure.",
                severity="caution",
            )
        )
    if requested_model == PLASMON_MODEL_LINDHARD_MERMIN_STATIC_LFC and int(payload.get("cluster_fallback_count", 0)) > 0:
        warnings.append(
            DerivedWarning(
                "plasmon",
                f"LOS-integrated finite-T Lindhard + Mermin + static LFC spectrum did not execute fully; benchmark status is invalid because {int(payload['cluster_fallback_count'])}/{cluster_count} clusters did not resolve a finite non-negative collision rate for the selected Mermin closure.",
                severity="caution",
            )
        )
    if requested_model == PLASMON_MODEL_QUANTUM_HYDRODYNAMIC and int(payload.get("cluster_fallback_count", 0)) > 0:
        warnings.append(
            DerivedWarning(
                "plasmon",
                f"LOS-integrated quantum-hydrodynamic spectrum did not execute fully; benchmark status is invalid because {int(payload['cluster_fallback_count'])}/{cluster_count} clusters did not resolve a finite non-negative collision rate for the selected collision closure ({collision_label}).",
                severity="caution",
            )
        )
    if requested_model == PLASMON_MODEL_FINITE_T_STLS and int(payload.get("stls_nonconverged_cluster_count", 0)) > 0:
        warnings.append(
            DerivedWarning(
                "plasmon",
                f"LOS-integrated finite-T STLS spectrum did not execute fully; benchmark status is invalid because {int(payload['stls_nonconverged_cluster_count'])}/{cluster_count} clusters did not converge the self-consistent STLS closure.",
                severity="caution",
            )
        )
    if requested_model in {PLASMON_MODEL_RPA_STATIC_LFC, PLASMON_MODEL_MERMIN_STATIC_LFC} and int(payload.get("out_of_domain_cluster_count", 0)) > 0:
        warnings.append(
            DerivedWarning(
                "plasmon",
                f"ESA static-LFC domain check flagged {int(payload['out_of_domain_cluster_count'])}/{cluster_count} LOS clusters outside the nominal validated (r_s, Theta) range; quicklook mode still uses the compact ESA-style surrogate there, while benchmark mode rejects those selections before or after clustering.",
                severity="caution",
            )
        )
    warnings.append(
        DerivedWarning(
            "plasmon",
            f"LOS-integrated spectral mode sums {zone_count} active zones through {cluster_count} cached plasmon state clusters before the instrument-response convolution.",
            severity="info",
        )
    )
    return payload, warnings, applied_model


def _resolve_spectrum_payload(
    requested_model: str,
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    current: dict[str, float],
    selection: object,
    mask: np.ndarray,
    weighting_mode: str,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    electron_fields: ElectronPolicyPayload,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> tuple[dict[str, np.ndarray | float] | None, list[DerivedWarning], str]:
    warnings: list[DerivedWarning] = []
    applied_model = PLASMON_MODEL_QUICKLOOK
    spectrum_payload: dict[str, np.ndarray | float] | None = None
    material_formula = _observable_material_formula(dataset, mask, electron_fields)
    if requested_model not in _ADVANCED_SPECTRAL_MODELS:
        return None, warnings, applied_model
    integration_mode = str(parameters.plasmon_integration_mode or "effective_state")
    resolved_effective_model = _effective_backend_model(current, parameters, requested_model)
    collision_metadata = _resolved_plasmon_collision_metadata(current, parameters)
    collision_label = _collision_model_label(str(parameters.plasmon_collision_model or PLASMON_COLLISION_MODEL_NRL_CONSTANT))
    effective_issue_counts = _accumulate_benchmark_issue_counts(
        _empty_benchmark_issue_counts(),
        _benchmark_domain_flags(current, parameters, resolved_effective_model),
    )
    if integration_mode == PLASMON_INTEGRATION_MODE_LOS_INTEGRATED:
        spectrum_payload, integration_warnings, _ = _build_los_integrated_spectrum_payload(
            requested_model,
            dataset,
            context,
            snapshot_index=snapshot_index,
            current=current,
            selection=selection,
            mask=mask,
            weighting_mode=weighting_mode,
            parameters=parameters,
            geometry=geometry,
            electron_fields=electron_fields,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        warnings.extend(integration_warnings)
        return spectrum_payload, warnings, requested_model
    if _plasmon_benchmark_mode(parameters) and _benchmark_rejects_noncollective(resolved_effective_model) and int(effective_issue_counts.get(_NONCOLLECTIVE_ISSUE, 0)) > 0:
        warnings.append(DerivedWarning("plasmon", "Benchmark mode rejected the effective-state spectrum because the selected state is non-collective (k*lambda_D >= 1). Narrow the active zone range or lower q before treating this as a plasmon benchmark.", severity="warning"))
        spectrum_payload = _invalid_benchmark_spectrum_payload(parameters, fallback_fraction=0.0, collision_rate_s=float(collision_metadata["rate_s"]), collision_source=str(collision_metadata["source"]), collision_summary=str(collision_metadata["summary"]), imag_shift_ev=_spectral_imag_shift_ev(current, parameters), zone_count_used=int(np.count_nonzero(mask)), cluster_count_used=1 if np.any(mask) else 0, zone_issue_counts=effective_issue_counts, cluster_issue_counts=effective_issue_counts)
        return spectrum_payload, warnings, requested_model
    if _plasmon_benchmark_mode(parameters) and _uses_collision_branch(resolved_effective_model) and int(effective_issue_counts.get(_WEAK_COUPLING_ISSUE, 0)) > 0:
        warnings.append(DerivedWarning("plasmon", f"Benchmark mode rejected the effective-state collision-aware spectrum because Coulomb logarithm <= 2 for the selected state, so the chosen collision closure ({collision_label}) is not trustworthy there.", severity="warning"))
        spectrum_payload = _invalid_benchmark_spectrum_payload(parameters, fallback_fraction=0.0, collision_rate_s=float(collision_metadata["rate_s"]), collision_source=str(collision_metadata["source"]), collision_summary=str(collision_metadata["summary"]), imag_shift_ev=_spectral_imag_shift_ev(current, parameters), zone_count_used=int(np.count_nonzero(mask)), cluster_count_used=1 if np.any(mask) else 0, zone_issue_counts=effective_issue_counts, cluster_issue_counts=effective_issue_counts)
        return spectrum_payload, warnings, requested_model
    if _uses_static_lfc(resolved_effective_model) and str(parameters.plasmon_lfc_model or "none") != "esa_static":
        warnings.append(DerivedWarning("plasmon", "The selected static-LFC branch requires the ESA static-LFC backend selection. The request is marked invalid_for_benchmark instead of silently falling back.", severity="warning"))
        spectrum_payload = _invalid_benchmark_spectrum_payload(parameters, fallback_fraction=1.0, collision_rate_s=float(collision_metadata["rate_s"]), collision_source=str(collision_metadata["source"]), collision_summary=str(collision_metadata["summary"]), imag_shift_ev=_spectral_imag_shift_ev(current, parameters), zone_count_used=int(np.count_nonzero(mask)), cluster_count_used=1 if np.any(mask) else 0, zone_issue_counts=effective_issue_counts, cluster_issue_counts=effective_issue_counts)
        return spectrum_payload, warnings, requested_model
    if _uses_static_lfc(resolved_effective_model) and _plasmon_benchmark_mode(parameters) and int(effective_issue_counts.get(_LFC_OUT_OF_DOMAIN_ISSUE, 0)) > 0:
        warnings.append(DerivedWarning("plasmon", "Benchmark mode rejected the effective-state static-LFC spectrum because the selected state lies outside the nominal ESA surrogate domain.", severity="warning"))
        spectrum_payload = _invalid_benchmark_spectrum_payload(parameters, fallback_fraction=0.0, collision_rate_s=float(collision_metadata["rate_s"]), collision_source=str(collision_metadata["source"]), collision_summary=str(collision_metadata["summary"]), imag_shift_ev=_spectral_imag_shift_ev(current, parameters), zone_count_used=int(np.count_nonzero(mask)), cluster_count_used=1 if np.any(mask) else 0, zone_issue_counts=effective_issue_counts, cluster_issue_counts=effective_issue_counts)
        return spectrum_payload, warnings, requested_model
    if _uses_collision_branch(resolved_effective_model) and (not math.isfinite(float(collision_metadata["rate_s"])) or float(collision_metadata["rate_s"]) < 0.0):
        warnings.append(DerivedWarning("plasmon", f"The selected collision closure ({collision_label}) did not resolve to a finite non-negative rate. The result is therefore marked invalid_for_benchmark instead of silently falling back.", severity="warning"))
        spectrum_payload = _invalid_benchmark_spectrum_payload(parameters, fallback_fraction=1.0, collision_rate_s=float(collision_metadata["rate_s"]), collision_source=str(collision_metadata["source"]), collision_summary=str(collision_metadata["summary"]), imag_shift_ev=_spectral_imag_shift_ev(current, parameters), zone_count_used=int(np.count_nonzero(mask)), cluster_count_used=1 if np.any(mask) else 0, zone_issue_counts=effective_issue_counts, cluster_issue_counts=effective_issue_counts)
        return spectrum_payload, warnings, requested_model
    spectrum_payload = _build_spectrum_payload_generic(
        current,
        parameters,
        requested_model=resolved_effective_model,
        material_formula=material_formula,
        analysis_cache=analysis_cache,
    )
    if resolved_effective_model == PLASMON_MODEL_FINITE_T_STLS and not bool(spectrum_payload.get("stls_converged", False)):
        warnings.append(
            DerivedWarning(
                "plasmon",
                "Finite-T STLS did not converge the self-consistent closure for the selected state, so the result is marked invalid_for_benchmark instead of silently falling back.",
                severity="warning",
            )
        )
    if requested_model == PLASMON_MODEL_AUTO_BEST:
        spectrum_payload = dict(spectrum_payload)
        spectrum_payload.setdefault("auto_model_summary", str(spectrum_payload.get("resolved_model_name", resolved_effective_model)))
    return spectrum_payload, warnings, requested_model


def _quicklook_peak_position_ev(current: dict[str, float], parameters: "DerivedAnalysisParameters") -> float:
    plasma_energy = float(current.get("omega_pe_ev", float("nan")))
    k_lambda = float(current.get("k_lambda", float("nan")))
    gamma = float(getattr(parameters, "plasmon_adiabatic_index", 1.0))
    if not math.isfinite(plasma_energy) or plasma_energy <= 0.0:
        return float("nan")
    correction = 1.0
    if math.isfinite(k_lambda):
        correction = max(0.0, 1.0 + gamma * (k_lambda ** 2))
    return float(plasma_energy * math.sqrt(correction))


def _scan_angle_values(parameters: "DerivedAnalysisParameters") -> tuple[np.ndarray, np.ndarray, str]:
    axis = _plasmon_scan_axis(parameters)
    scan_values = _plasmon_scan_values(parameters)
    if scan_values.size == 0:
        return scan_values, scan_values, "Scattering angle [deg]"
    if axis == PLASMON_AXIS_K_ANGSTROM_INV:
        angle_values = np.asarray([_angle_deg_from_k_angstrom_inv(value, float(parameters.plasmon_photon_energy_kev)) for value in scan_values], dtype=np.float64)
        return scan_values, angle_values, "k [1/A]"
    return scan_values, np.asarray(scan_values, dtype=np.float64), "Scattering angle [deg]"


def _build_spectrum_compare_bundle(
    requested_model: str,
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    current: dict[str, float],
    selection: object,
    mask: np.ndarray,
    weighting_mode: str,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    electron_fields: ElectronPolicyPayload,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> tuple[DerivedPlotBundle | None, list[str]]:
    if not _plasmon_compare_models(parameters):
        return None, []
    curve_names: list[str] = []
    y_series: list[np.ndarray] = []
    energy_axis: np.ndarray | None = None
    skipped: list[str] = []
    for model in _requested_comparison_models(parameters):
        if progress_check is not None:
            progress_check()
        payload, _, _ = _resolve_spectrum_payload(
            model,
            dataset,
            context,
            snapshot_index=snapshot_index,
            current=current,
            selection=selection,
            mask=mask,
            weighting_mode=weighting_mode,
            parameters=parameters,
            geometry=geometry,
            electron_fields=electron_fields,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        if payload is None:
            skipped.append(_model_display_name(model))
            continue
        energy = np.asarray(payload.get("energy_ev", np.asarray([], dtype=np.float64)), dtype=np.float64)
        spectrum = np.asarray(payload.get("spectrum", np.asarray([], dtype=np.float64)), dtype=np.float64)
        if energy.size == 0 or spectrum.size != energy.size:
            skipped.append(_model_display_name(model))
            continue
        if energy_axis is None:
            energy_axis = energy
        elif energy_axis.shape != energy.shape or not np.allclose(energy_axis, energy, equal_nan=True):
            skipped.append(_model_display_name(model))
            continue
        curve_names.append(_model_display_name(model))
        y_series.append(spectrum)
    if energy_axis is None or not y_series:
        return None, skipped
    return DerivedPlotBundle(
        key="spectrum_compare_models",
        title="Model-comparison spectra at fixed geometry",
        x_label="Energy transfer [eV]",
        y_label=("Normalized intensity [arb. u.]" if str(parameters.plasmon_normalization or "peak") != "none" else "Intensity [arb. u.]"),
        x_values=np.asarray(energy_axis, dtype=np.float64),
        y_series=tuple(np.asarray(series, dtype=np.float64) for series in y_series),
        curve_names=tuple(curve_names),
    ), skipped


def _build_spectrum_policy_compare_bundle(
    requested_model: str,
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    weighting_mode: str,
    selection: object,
    mask: np.ndarray,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> tuple[DerivedPlotBundle | None, list[str], tuple[str, ...]]:
    if not _plasmon_compare_policies(parameters):
        return None, [], ()
    if str(requested_model) not in _ADVANCED_SPECTRAL_MODELS or str(requested_model) == PLASMON_MODEL_AUTO_BEST:
        return None, [], ()
    curve_names: list[str] = []
    y_series: list[np.ndarray] = []
    energy_axis: np.ndarray | None = None
    skipped: list[str] = []
    used_policies: list[str] = []
    for policy in PLASMON_BENCHMARK_POLICY_COMPARISON:
        if progress_check is not None:
            progress_check()
        policy_fields = _subset_electron_policy_payload(dataset, mask, resolve_effective_electron_fields(dataset, policy))
        point_current = _summary_values_for_existing_selection(
            dataset,
            parameters,
            geometry,
            snapshot_index=snapshot_index,
            weighting_mode=weighting_mode,
            electron_fields=policy_fields,
            mask=mask,
            selection=selection,
            analysis_cache=analysis_cache,
        )
        payload, _, _ = _resolve_spectrum_payload(
            requested_model,
            dataset,
            context,
            snapshot_index=snapshot_index,
            current=point_current,
            selection=selection,
            mask=mask,
            weighting_mode=weighting_mode,
            parameters=replace(parameters, plasmon_electron_policy=str(policy)),
            geometry=geometry,
            electron_fields=policy_fields,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        if payload is None:
            skipped.append(plasmon_electron_policy_label(str(policy)))
            continue
        energy = np.asarray(payload.get("energy_ev", np.asarray([], dtype=np.float64)), dtype=np.float64)
        spectrum = np.asarray(payload.get("spectrum", np.asarray([], dtype=np.float64)), dtype=np.float64)
        if energy.size == 0 or spectrum.size != energy.size:
            skipped.append(plasmon_electron_policy_label(str(policy)))
            continue
        if energy_axis is None:
            energy_axis = energy
        elif energy_axis.shape != energy.shape or not np.allclose(energy_axis, energy, equal_nan=True):
            skipped.append(plasmon_electron_policy_label(str(policy)))
            continue
        curve_names.append(plasmon_electron_policy_label(str(policy)))
        y_series.append(spectrum)
        used_policies.append(str(policy))
    if energy_axis is None or not y_series:
        return None, skipped, tuple(used_policies)
    return DerivedPlotBundle(
        key="spectrum_compare_policies",
        title=f"{_model_display_name(requested_model)} electron-policy comparison spectrum",
        x_label="Energy transfer [eV]",
        y_label=("Normalized intensity [arb. u.]" if str(parameters.plasmon_normalization or "peak") != "none" else "Intensity [arb. u.]"),
        x_values=np.asarray(energy_axis, dtype=np.float64),
        y_series=tuple(np.asarray(series, dtype=np.float64) for series in y_series),
        curve_names=tuple(curve_names),
    ), skipped, tuple(used_policies)


def _build_dispersion_bundles(
    requested_model: str,
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    current: dict[str, float],
    selection: object,
    mask: np.ndarray,
    weighting_mode: str,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    electron_fields: ElectronPolicyPayload,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> tuple[list[DerivedPlotBundle], list[str]]:
    scan_values, angle_values, x_label = _scan_angle_values(parameters)
    if scan_values.size == 0:
        return [], []
    scan_label = "k" if str(_plasmon_scan_axis(parameters)) == PLASMON_AXIS_K_ANGSTROM_INV else "scattering angle"
    bundles: list[DerivedPlotBundle] = []
    invalid_notes: list[str] = []
    if requested_model == PLASMON_MODEL_QUICKLOOK:
        quicklook_peak = np.asarray([
            _quicklook_peak_position_ev(_current_state_at_angle(current, parameters, angle_deg=float(angle)), parameters)
            if math.isfinite(float(angle)) else float("nan")
            for angle in angle_values
        ], dtype=np.float64)
        bundles.append(
            DerivedPlotBundle(
                key="dispersion_selected_model",
                title="Quick-look plasmon dispersion",
                x_label=x_label,
                y_label="Peak position [eV]",
                x_values=np.asarray(scan_values, dtype=np.float64),
                y_series=(quicklook_peak,),
                curve_names=("Quick look",),
            )
        )
        return bundles, invalid_notes
    selected_peak = np.full(scan_values.shape, np.nan, dtype=np.float64)
    selected_width = np.full(scan_values.shape, np.nan, dtype=np.float64)
    invalid_count = 0
    for index, angle in enumerate(angle_values):
        if progress_check is not None and (index % 4 == 0):
            progress_check()
        if not math.isfinite(float(angle)):
            invalid_count += 1
            continue
        point_parameters = replace(parameters, plasmon_scattering_angle_deg=float(angle))
        point_current = _current_state_at_angle(current, point_parameters, angle_deg=float(angle))
        payload, _, _ = _resolve_spectrum_payload(
            requested_model,
            dataset,
            context,
            snapshot_index=snapshot_index,
            current=point_current,
            selection=selection,
            mask=mask,
            weighting_mode=weighting_mode,
            parameters=point_parameters,
            geometry=geometry,
            electron_fields=electron_fields,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        if payload is None:
            invalid_count += 1
            continue
        selected_peak[index] = float(payload.get("peak_energy_ev", float("nan")))
        selected_width[index] = float(payload.get("peak_fwhm_ev", float("nan")))
    if invalid_count > 0:
        invalid_notes.append(f"{invalid_count}/{scan_values.size} scan points did not produce a valid selected-model spectrum.")
    bundles.append(
            DerivedPlotBundle(
                key="dispersion_selected_model",
                title=f"{_model_display_name(requested_model)} peak position vs {scan_label}",
                x_label=x_label,
                y_label="Peak position [eV]",
            x_values=np.asarray(scan_values, dtype=np.float64),
            y_series=(selected_peak,),
            curve_names=(_model_display_name(requested_model),),
        )
    )
    bundles.append(
            DerivedPlotBundle(
                key="dispersion_selected_width",
                title=f"{_model_display_name(requested_model)} peak FWHM vs {scan_label}",
                x_label=x_label,
                y_label="Peak FWHM [eV]",
            x_values=np.asarray(scan_values, dtype=np.float64),
            y_series=(selected_width,),
            curve_names=(_model_display_name(requested_model),),
        )
    )
    if _plasmon_compare_models(parameters):
        comparison_names: list[str] = []
        comparison_series: list[np.ndarray] = []
        width_names: list[str] = []
        width_series: list[np.ndarray] = []
        skipped_models: list[str] = []
        for model in _requested_comparison_models(parameters):
            if progress_check is not None:
                progress_check()
            peak_values = np.full(scan_values.shape, np.nan, dtype=np.float64)
            width_values = np.full(scan_values.shape, np.nan, dtype=np.float64)
            valid_peak = False
            valid_width = False
            for index, angle in enumerate(angle_values):
                if progress_check is not None and (index % 4 == 0):
                    progress_check()
                if not math.isfinite(float(angle)):
                    continue
                point_parameters = replace(parameters, plasmon_scattering_angle_deg=float(angle))
                point_current = _current_state_at_angle(current, point_parameters, angle_deg=float(angle))
                payload, _, _ = _resolve_spectrum_payload(
                    model,
                    dataset,
                    context,
                    snapshot_index=snapshot_index,
                    current=point_current,
                    selection=selection,
                    mask=mask,
                    weighting_mode=weighting_mode,
                    parameters=point_parameters,
                    geometry=geometry,
                    electron_fields=electron_fields,
                    analysis_cache=analysis_cache,
                    progress_check=progress_check,
                )
                if payload is None:
                    continue
                peak_value = float(payload.get("peak_energy_ev", float("nan")))
                if math.isfinite(peak_value):
                    peak_values[index] = peak_value
                    valid_peak = True
                width_value = float(payload.get("peak_fwhm_ev", float("nan")))
                if math.isfinite(width_value):
                    width_values[index] = width_value
                    valid_width = True
            if valid_peak:
                comparison_names.append(_model_display_name(model))
                comparison_series.append(peak_values)
            if valid_width:
                width_names.append(_model_display_name(model))
                width_series.append(width_values)
            if not valid_peak and not valid_width:
                skipped_models.append(_model_display_name(model))
        if comparison_series:
            bundles.append(
                DerivedPlotBundle(
                    key="dispersion_compare_models",
                    title=f"Peak-position comparison vs {scan_label}",
                    x_label=x_label,
                    y_label="Peak position [eV]",
                    x_values=np.asarray(scan_values, dtype=np.float64),
                    y_series=tuple(np.asarray(series, dtype=np.float64) for series in comparison_series),
                    curve_names=tuple(comparison_names),
                )
            )
        if width_series:
            bundles.append(
                DerivedPlotBundle(
                    key="dispersion_compare_width_models",
                    title=f"Peak-FWHM comparison vs {scan_label}",
                    x_label=x_label,
                    y_label="Peak FWHM [eV]",
                    x_values=np.asarray(scan_values, dtype=np.float64),
                    y_series=tuple(np.asarray(series, dtype=np.float64) for series in width_series),
                    curve_names=tuple(width_names),
                )
            )
        invalid_notes.extend([f"Comparison skipped for {name}." for name in skipped_models])
    if _plasmon_compare_policies(parameters) and requested_model in _ADVANCED_SPECTRAL_MODELS and requested_model != PLASMON_MODEL_AUTO_BEST:
        comparison_names: list[str] = []
        comparison_series: list[np.ndarray] = []
        skipped_policies: list[str] = []
        for policy in PLASMON_BENCHMARK_POLICY_COMPARISON:
            if progress_check is not None:
                progress_check()
            policy_fields = _subset_electron_policy_payload(dataset, mask, resolve_effective_electron_fields(dataset, policy))
            policy_peak_values = np.full(scan_values.shape, np.nan, dtype=np.float64)
            valid = False
            for index, angle in enumerate(angle_values):
                if progress_check is not None and (index % 4 == 0):
                    progress_check()
                if not math.isfinite(float(angle)):
                    continue
                point_parameters = replace(parameters, plasmon_scattering_angle_deg=float(angle), plasmon_electron_policy=str(policy))
                point_current = _summary_values_for_existing_selection(
                    dataset,
                    point_parameters,
                    geometry,
                    snapshot_index=snapshot_index,
                    weighting_mode=weighting_mode,
                    electron_fields=policy_fields,
                    mask=mask,
                    selection=selection,
                    analysis_cache=analysis_cache,
                )
                payload, _, _ = _resolve_spectrum_payload(
                    requested_model,
                    dataset,
                    context,
                    snapshot_index=snapshot_index,
                    current=point_current,
                    selection=selection,
                    mask=mask,
                    weighting_mode=weighting_mode,
                    parameters=point_parameters,
                    geometry=geometry,
                    electron_fields=policy_fields,
                    analysis_cache=analysis_cache,
                    progress_check=progress_check,
                )
                if payload is None:
                    continue
                peak_value = float(payload.get("peak_energy_ev", float("nan")))
                if math.isfinite(peak_value):
                    policy_peak_values[index] = peak_value
                    valid = True
            if valid:
                comparison_names.append(plasmon_electron_policy_label(str(policy)))
                comparison_series.append(policy_peak_values)
            else:
                skipped_policies.append(plasmon_electron_policy_label(str(policy)))
        if comparison_series:
            bundles.append(
                DerivedPlotBundle(
                    key="dispersion_compare_policies",
                    title=f"{_model_display_name(requested_model)} electron-policy comparison",
                    x_label=x_label,
                    y_label="Peak position [eV]",
                    x_values=np.asarray(scan_values, dtype=np.float64),
                    y_series=tuple(np.asarray(series, dtype=np.float64) for series in comparison_series),
                    curve_names=tuple(comparison_names),
                )
            )
        invalid_notes.extend([f"Electron-policy comparison skipped for {name}." for name in skipped_policies])
    return bundles, invalid_notes


def evaluate_plasmon_regime(
    dataset: DerivedRunData,
    context: RunContext,
    *,
    snapshot_index: int,
    photon_energy_kev: float,
    scattering_angle_deg: float,
    adiabatic_index: float,
    parameters: "DerivedAnalysisParameters",
    geometry: "AnalysisGeometryMetadata",
    include_time_plots: bool = True,
    analysis_cache: AnalysisStateCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> PlasmonResult:
    """Estimate whether the selected plasma is in a collective XRTS regime."""

    del photon_energy_kev, scattering_angle_deg, adiabatic_index
    total_start = perf_counter()
    spectrum_runtime_s = 0.0
    comparison_runtime_s = 0.0
    dispersion_runtime_s = 0.0
    time_series_runtime_s = 0.0
    warnings: list[DerivedWarning] = []
    ps_warning = picosecond_drive_warning(
        "plasmon",
        dataset,
        "The plasmon/XRTS effective-state interpretation is less validated for ps-scale drives and may be more sensitive to transient nonequilibrium structure.",
    )
    if ps_warning is not None:
        warnings.append(ps_warning)
    cylindrical_warning = cylindrical_path_note("plasmon", dataset, geometry)
    if cylindrical_warning is not None:
        warnings.append(cylindrical_warning)
    weighting_mode = resolve_weighting_mode(parameters.weighting_mode, module_name="plasmon")
    electron_fields = _effective_electron_fields(dataset, parameters)
    current, selection_warnings, mask, selection = _summary_values(
        dataset,
        context,
        parameters,
        geometry,
        snapshot_index=snapshot_index,
        weighting_mode=weighting_mode,
        electron_fields=electron_fields,
        analysis_cache=analysis_cache,
    )
    electron_fields = _subset_electron_policy_payload(dataset, mask, electron_fields)
    warnings.extend(_electron_policy_warnings(electron_fields))
    warnings.extend(selection_warnings)

    if not np.any(mask):
        warnings.append(DerivedWarning("plasmon", "No active zones are selected for the plasmon estimate.", severity="error"))
    elif not all(
        math.isfinite(float(current[key]))
        for key in ("te_ev", "ti_ev", "ne_cm3", "zbar", "debye_length_cm", "k_lambda")
    ):
        warnings.append(
            DerivedWarning(
                "plasmon",
                "The active weighting/filter selection did not produce a finite effective plasma state for this snapshot.",
                severity="warning",
            )
        )
    requested_model = str(parameters.plasmon_model or PLASMON_MODEL_QUICKLOOK)
    applied_model = PLASMON_MODEL_QUICKLOOK
    spectrum_payload: dict[str, np.ndarray | float] | None = None
    if requested_model in _ADVANCED_SPECTRAL_MODELS:
        resolved_effective_model = _effective_backend_model(current, parameters, requested_model)
        if _plasmon_benchmark_mode(parameters):
            warnings.append(
                DerivedWarning(
                    "plasmon",
                    f"Benchmark spectral mode uses a refined odd energy grid (>= {_resolved_energy_points(parameters)} points) and local quadratic peak extraction; advanced invalid states remain hard failures instead of fallbacks.",
                    severity="info",
                )
            )
        else:
            warnings.append(
                DerivedWarning(
                    "plasmon",
                    "Quicklook spectral mode preserves the lightweight exploratory grid and staged-kernel behavior. Use Benchmark mode for validation runs.",
                    severity="info",
                )
            )
        if _uses_quantum_hydrodynamic_backend(resolved_effective_model):
            warnings.append(DerivedWarning("plasmon", "The selected backend uses a damped quantum-hydrodynamic dielectric with thermal pressure, Fermi pressure, and Bohm-recoil terms. Treat it as an experimental collective-fluid proxy rather than an article-native calculation branch.", severity="info"))
        elif math.isfinite(float(current.get("theta_degeneracy", float("nan")))) and float(current["theta_degeneracy"]) < 1.0 and not _uses_lindhard_backend(resolved_effective_model):
            warnings.append(DerivedWarning("plasmon", "Theta = Te/Ef < 1, so the selected state is at least partially degenerate; the current spectral branch still uses the classical finite-temperature susceptibility baseline and should be treated as approximate in degenerate WDM conditions.", severity="warning"))
        elif _uses_lindhard_backend(resolved_effective_model):
            warnings.append(DerivedWarning("plasmon", "The selected backend uses a numerical finite-T Lindhard susceptibility, so Landau damping and continuum broadening enter through Im[chi(q, omega)] rather than a classical Maxwellian surrogate.", severity="info"))
        selected_lfc_model = str(parameters.plasmon_lfc_model or "none")
        if selected_lfc_model != "none" and not _uses_static_lfc(resolved_effective_model):
            warnings.append(DerivedWarning("plasmon", f"The selected non-LFC spectral branch ignores the requested static-LFC backend setting ({selected_lfc_model}). Choose a dedicated static-LFC plasmon model if you want correlation corrections applied.", severity="info"))
        if _uses_static_lfc(resolved_effective_model):
            domain_message = esa_domain_message(float(current.get("wigner_seitz_rs", float("nan"))), float(current.get("theta_degeneracy", float("nan"))))
            if domain_message:
                warnings.append(DerivedWarning("plasmon", domain_message, severity="caution"))
        spectrum_start = perf_counter()
        spectrum_payload, spectrum_warnings, applied_model = _resolve_spectrum_payload(
            requested_model,
            dataset,
            context,
            snapshot_index=snapshot_index,
            current=current,
            selection=selection,
            mask=mask,
            weighting_mode=weighting_mode,
            parameters=parameters,
            geometry=geometry,
            electron_fields=electron_fields,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        spectrum_runtime_s = perf_counter() - spectrum_start
        warnings.extend(spectrum_warnings)
    elif requested_model != PLASMON_MODEL_QUICKLOOK:
        warnings.append(
            DerivedWarning(
                "plasmon",
                f"Plasmon model {requested_model!r} is not implemented yet in this build; falling back to the existing quick-look regime estimate.",
                severity="info",
            )
        )
    elif _plasmon_benchmark_mode(parameters):
        warnings.append(
            DerivedWarning(
                "plasmon",
                "Quick-look plasmon mode is a heuristic regime/dispersion estimate. It does not provide a benchmark-grade spectral line shape or fitted peak, so benchmark_status stays not_applicable for that branch.",
                severity="info",
            )
        )

    if geometry.path_length_mode in {"oblique-sec(theta)", "cylindrical-shell-unavailable-fallback-slab"} and abs(float(geometry.line_of_sight_cosine)) < 0.1:
        warnings.append(
            DerivedWarning(
                "plasmon",
                "LOS cosine is very small; sec(theta) path-length amplification may exaggerate rarefied edge contributions.",
                severity="warning",
            )
        )

    k_lambda = float(current["k_lambda"])
    collision_rate = float(current["collision_rate_s"])
    omega_pe = float(current["omega_pe_rad_s"])
    ln_lambda = float(current["coulomb_log"])
    if math.isfinite(k_lambda) and k_lambda >= 1.0:
        severity = "warning" if k_lambda >= 2.0 else "caution"
        warnings.append(DerivedWarning("plasmon", "k*lambda_D >= 1, so the selected state is in a non-collective scattering regime.", severity=severity))
    if math.isfinite(ln_lambda) and ln_lambda <= 2.0:
        warnings.append(DerivedWarning("plasmon", "Coulomb logarithm is small; weak-coupling transport estimates are becoming unreliable.", severity="warning"))
    if math.isfinite(collision_rate) and math.isfinite(omega_pe) and omega_pe > 0.0 and collision_rate / omega_pe > 0.1:
        warnings.append(DerivedWarning("plasmon", "Collision rate is a sizable fraction of the plasma frequency; strong damping is likely.", severity="caution"))

    time_plots: tuple[DerivedPlotBundle, ...] = ()
    if include_time_plots:
        time_series_start = perf_counter()
        time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
        mean_series = shared_time_series_weighted_means(
            dataset,
            context,
            geometry=geometry,
            weighting_mode=weighting_mode,
            field_series=(
                ("te_ev", np.asarray(dataset.temperature_e_ev, dtype=np.float64)),
                ("ti_ev", np.asarray(dataset.temperature_i_ev, dtype=np.float64)),
                ("ne_cm3", np.asarray(electron_fields.electron_density_cm3, dtype=np.float64)),
                ("zbar", np.asarray(electron_fields.mean_charge, dtype=np.float64)),
                ("ion_mass_mu", np.asarray(dataset.zone_atomic_weight, dtype=np.float64)),
            ),
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
        series_cache_key = (
            "plasmon.derived_series",
            geometry.observation_side,
            round(float(geometry.line_of_sight_angle_deg), 12),
            round(float(parameters.plasmon_photon_energy_kev), 12),
            round(float(parameters.plasmon_scattering_angle_deg), 12),
            round(float(parameters.plasmon_adiabatic_index), 12),
            weighting_mode,
            parameters.reuse_viewer_subset,
            tuple(parameters.derived_region_ids or ()),
            tuple(parameters.derived_material_ids or ()),
            bool(parameters.exclude_entry_region),
            bool(parameters.exclude_low_density),
            round(float(parameters.min_density_g_cm3), 12),
            bool(parameters.exclude_opposite_velocity),
            parameters.zone_index_lower,
            parameters.zone_index_upper,
            str(getattr(parameters, "plasmon_electron_policy", PLASMON_ELECTRON_POLICY_RAW)),
        )

        def _build_derived_series() -> dict[str, np.ndarray]:
            te_series = np.asarray(mean_series["te_ev"], dtype=np.float64)
            ti_series = np.asarray(mean_series["ti_ev"], dtype=np.float64)
            ne_series = np.asarray(mean_series["ne_cm3"], dtype=np.float64)
            zbar_series = np.asarray(mean_series["zbar"], dtype=np.float64)
            ion_mass_series = np.asarray(mean_series["ion_mass_mu"], dtype=np.float64)
            debye_series = np.full(te_series.shape, np.nan, dtype=np.float64)
            omega_series = np.full(te_series.shape, np.nan, dtype=np.float64)
            collision_series = np.full(te_series.shape, np.nan, dtype=np.float64)
            klambda_series = np.full(te_series.shape, np.nan, dtype=np.float64)
            for time_index in range(te_series.size):
                if progress_check is not None and (time_index % 16 == 0):
                    progress_check()
                te_value = float(te_series[time_index])
                ne_value = float(ne_series[time_index])
                zbar_value = float(zbar_series[time_index])
                ion_mass_value = float(ion_mass_series[time_index])
                debye_cm = electron_debye_length_cm(te_value, ne_value)
                omega_pe = electron_plasma_frequency_rad_s(ne_value)
                ln_lambda = coulomb_logarithm_ei(te_value, ne_value, max(zbar_value, 1.0))
                collision = electron_collision_rate_s(ne_value, te_value, ln_lambda)
                _ = ion_sound_speed_cm_s(te_value, max(zbar_value, 1.0), ion_mass_value, parameters.plasmon_adiabatic_index)
                debye_series[time_index] = debye_cm * 1.0e4 if math.isfinite(debye_cm) else np.nan
                omega_series[time_index] = electron_plasma_energy_ev(ne_value)
                collision_series[time_index] = float(collision)
                klambda_series[time_index] = float(current["scattering_wavevector_cm_inv"]) * debye_cm if math.isfinite(debye_cm) else np.nan
            return {
                "debye_length_um": debye_series,
                "plasma_frequency_ev": omega_series,
                "collision_rate_s": collision_series,
                "k_lambda": klambda_series,
            }

        derived_series = cached_time_series_payload(
            series_cache_key,
            analysis_cache=analysis_cache,
            builder=_build_derived_series,
        )
        te_series = np.asarray(mean_series["te_ev"], dtype=np.float64)
        ti_series = np.asarray(mean_series["ti_ev"], dtype=np.float64)
        ne_series = np.asarray(mean_series["ne_cm3"], dtype=np.float64)
        zbar_series = np.asarray(mean_series["zbar"], dtype=np.float64)
        debye_series = np.asarray(derived_series["debye_length_um"], dtype=np.float64)
        omega_series = np.asarray(derived_series["plasma_frequency_ev"], dtype=np.float64)
        collision_series = np.asarray(derived_series["collision_rate_s"], dtype=np.float64)
        klambda_series = np.asarray(derived_series["k_lambda"], dtype=np.float64)
        time_plots = (
            DerivedPlotBundle(
                key="electron_temperature",
                title="Electron temperature vs time",
                x_label="Time [ns]",
                y_label="Temperature [eV]",
                x_values=time_ns,
                y_series=(te_series,),
                curve_names=("Te",),
            ),
            DerivedPlotBundle(
                key="ion_temperature",
                title="Ion temperature vs time",
                x_label="Time [ns]",
                y_label="Temperature [eV]",
                x_values=time_ns,
                y_series=(ti_series,),
                curve_names=("Ti",),
            ),
            DerivedPlotBundle(
                key="temperatures",
                title="Electron and ion temperature vs time",
                x_label="Time [ns]",
                y_label="Temperature [eV]",
                x_values=time_ns,
                y_series=(te_series, ti_series),
                curve_names=("Te", "Ti"),
            ),
            DerivedPlotBundle(
                key="electron_density",
                title="Electron density vs time",
                x_label="Time [ns]",
                y_label="Electron density [1/cm3]",
                x_values=time_ns,
                y_series=(ne_series,),
                curve_names=("ne",),
            ),
            DerivedPlotBundle(
                key="mean_charge",
                title="Mean charge vs time",
                x_label="Time [ns]",
                y_label="Mean charge",
                x_values=time_ns,
                y_series=(zbar_series,),
                curve_names=("Zbar",),
            ),
            DerivedPlotBundle(
                key="debye_length",
                title="Debye length vs time",
                x_label="Time [ns]",
                y_label="Debye length [um]",
                x_values=time_ns,
                y_series=(debye_series,),
                curve_names=("lambda_D",),
            ),
            DerivedPlotBundle(
                key="plasma_frequency",
                title="Plasma frequency vs time",
                x_label="Time [ns]",
                y_label="hbar*omega_pe [eV]",
                x_values=time_ns,
                y_series=(omega_series,),
                curve_names=("hbar*omega_pe",),
            ),
            DerivedPlotBundle(
                key="collision_rate",
                title="Electron collision rate vs time",
                x_label="Time [ns]",
                y_label="Collision rate [1/s]",
                x_values=time_ns,
                y_series=(collision_series,),
                curve_names=("nu_e",),
            ),
            DerivedPlotBundle(
                key="k_lambda",
                title="k*lambda_D vs time",
                x_label="Time [ns]",
                y_label="k*lambda_D",
                x_values=time_ns,
                y_series=(klambda_series,),
                curve_names=("k lambda_D",),
            ),
        )
        time_series_runtime_s = perf_counter() - time_series_start

    coordinate_values, coordinate_label = profile_coordinate_values(dataset, snapshot_index, geometry.profile_coordinate_mode)
    boundary_positions = profile_boundary_positions(dataset, snapshot_index, geometry.profile_coordinate_mode)
    te_profile = np.where(mask, np.asarray(dataset.temperature_e_ev[int(snapshot_index)], dtype=np.float64), np.nan)
    ti_profile = np.where(mask, np.asarray(dataset.temperature_i_ev[int(snapshot_index)], dtype=np.float64), np.nan)
    ne_profile = np.where(mask, np.asarray(electron_fields.electron_density_cm3[int(snapshot_index)], dtype=np.float64) * 1.0e6, np.nan)
    zbar_profile = np.where(mask, np.asarray(electron_fields.mean_charge[int(snapshot_index)], dtype=np.float64), np.nan)
    local_density = np.asarray(electron_fields.electron_density_cm3[int(snapshot_index)], dtype=np.float64)
    local_debye = np.full(mask.shape, np.nan, dtype=np.float64)
    valid_local = mask & np.isfinite(te_profile) & np.isfinite(local_density) & (te_profile > 0.0) & (local_density > 0.0)
    local_debye[valid_local] = 7.43e2 * np.sqrt(te_profile[valid_local] / local_density[valid_local])
    local_klambda = current["scattering_wavevector_cm_inv"] * local_debye

    study_mode = _plasmon_study_mode(parameters)
    angle_scan = np.linspace(10.0, 140.0, 67, dtype=np.float64)
    wavelength_cm = plasmon_probe_wavelength_cm(parameters.plasmon_photon_energy_kev)
    scattering_k = 4.0 * math.pi * np.sin(np.deg2rad(angle_scan) / 2.0) / wavelength_cm
    angle_scan_klambda = (
        scattering_k * float(current["debye_length_cm"])
        if np.isfinite(float(current["debye_length_cm"]))
        else np.full(angle_scan.shape, np.nan, dtype=np.float64)
    )
    dispersion_bundles: list[DerivedPlotBundle] = []
    comparison_bundle: DerivedPlotBundle | None = None
    policy_comparison_bundle: DerivedPlotBundle | None = None
    policy_comparison_policies: tuple[str, ...] = ()
    if study_mode == PLASMON_STUDY_MODE_DISPERSION:
        dispersion_start = perf_counter()
        dispersion_bundles, dispersion_notes = _build_dispersion_bundles(
            requested_model,
            dataset,
            context,
            snapshot_index=snapshot_index,
            current=current,
            selection=selection,
            mask=mask,
            weighting_mode=weighting_mode,
            parameters=parameters,
            geometry=geometry,
            electron_fields=electron_fields,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        dispersion_runtime_s = perf_counter() - dispersion_start
        if dispersion_notes:
            warnings.extend(DerivedWarning("plasmon", note, severity="info") for note in dispersion_notes)
        if _plasmon_compare_policies(parameters) and any(bundle.key == "dispersion_compare_policies" for bundle in dispersion_bundles):
            policy_comparison_policies = tuple(str(value) for value in PLASMON_BENCHMARK_POLICY_COMPARISON)
    if study_mode == PLASMON_STUDY_MODE_SPECTRUM:
        comparison_start = perf_counter()
        comparison_bundle, comparison_skips = _build_spectrum_compare_bundle(
            requested_model,
            dataset,
            context,
            snapshot_index=snapshot_index,
            current=current,
            selection=selection,
            mask=mask,
            weighting_mode=weighting_mode,
            parameters=parameters,
            geometry=geometry,
            electron_fields=electron_fields,
            analysis_cache=analysis_cache,
            progress_check=progress_check,
        )
        if comparison_skips:
            warnings.append(
                DerivedWarning(
                    "plasmon",
                    "Model comparison skipped invalid spectral branches: " + ", ".join(comparison_skips),
                    severity="info",
                )
            )
        if _plasmon_compare_policies(parameters):
            policy_comparison_bundle, policy_skips, policy_comparison_policies = _build_spectrum_policy_compare_bundle(
                requested_model,
                dataset,
                context,
                snapshot_index=snapshot_index,
                weighting_mode=weighting_mode,
                selection=selection,
                mask=mask,
                parameters=parameters,
                geometry=geometry,
                analysis_cache=analysis_cache,
                progress_check=progress_check,
            )
            if policy_skips:
                warnings.append(
                    DerivedWarning(
                        "plasmon",
                        "Electron-policy comparison skipped invalid benchmark policy branches: " + ", ".join(policy_skips),
                        severity="info",
                    )
                )
        comparison_runtime_s = perf_counter() - comparison_start

    geometry_summary = (
        f"{geometry.observation_side} side | LOS cos={geometry.line_of_sight_cosine:.3f} | "
        f"profile={geometry.profile_coordinate_mode} | {path_geometry_summary(dataset, geometry)}"
    )

    profile_bundles: list[DerivedPlotBundle] = []
    if spectrum_payload is not None:
        profile_bundles.append(
            DerivedPlotBundle(
                key="spectrum_observed",
                title=("LOS-integrated plasmon spectrum" if str(parameters.plasmon_integration_mode) == PLASMON_INTEGRATION_MODE_LOS_INTEGRATED and applied_model != PLASMON_MODEL_QUICKLOOK else "One-state plasmon spectrum"),
                x_label="Energy transfer [eV]",
                y_label=("Normalized intensity [arb. u.]" if str(parameters.plasmon_normalization or "peak") != "none" else "Intensity [arb. u.]"),
                x_values=np.asarray(spectrum_payload["energy_ev"], dtype=np.float64),
                y_series=(np.asarray(spectrum_payload["spectrum"], dtype=np.float64),),
                curve_names=(("Observed" if float(parameters.plasmon_instrument_fwhm_ev) > 0.0 else ("Mermin baseline" if applied_model == PLASMON_MODEL_MERMIN else "RPA baseline")),),
            )
        )
    if comparison_bundle is not None:
        profile_bundles.append(comparison_bundle)
    if policy_comparison_bundle is not None:
        profile_bundles.append(policy_comparison_bundle)
    profile_bundles.extend(dispersion_bundles)
    profile_bundles.extend(
        (
            DerivedPlotBundle(
                key="electron_temperature_profile",
                title="Electron temperature profile",
                x_label=coordinate_label,
                y_label="Temperature [eV]",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(te_profile,),
                curve_names=("Te",),
                boundary_positions=boundary_positions,
            ),
            DerivedPlotBundle(
                key="ion_temperature_profile",
                title="Ion temperature profile",
                x_label=coordinate_label,
                y_label="Temperature [eV]",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(ti_profile,),
                curve_names=("Ti",),
                boundary_positions=boundary_positions,
            ),
            DerivedPlotBundle(
                key="temperature_profile",
                title="Electron and ion temperature profile",
                x_label=coordinate_label,
                y_label="Temperature [eV]",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(te_profile, ti_profile),
                curve_names=("Te", "Ti"),
                boundary_positions=boundary_positions,
            ),
            DerivedPlotBundle(
                key="electron_density_profile",
                title="Electron density profile",
                x_label=coordinate_label,
                y_label="Electron density [1/m3]",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(ne_profile,),
                curve_names=("ne",),
                boundary_positions=boundary_positions,
            ),
            DerivedPlotBundle(
                key="mean_charge_profile",
                title="Mean charge profile",
                x_label=coordinate_label,
                y_label="Mean charge",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(zbar_profile,),
                curve_names=("Zbar",),
                boundary_positions=boundary_positions,
            ),
            DerivedPlotBundle(
                key="local_k_lambda_profile",
                title="Local k*lambda_D profile",
                x_label=coordinate_label,
                y_label="k*lambda_D",
                x_values=np.asarray(coordinate_values, dtype=np.float64),
                y_series=(np.asarray(local_klambda, dtype=np.float64),),
                curve_names=("k lambda_D",),
                boundary_positions=boundary_positions,
            ),
            DerivedPlotBundle(
                key="angle_scan",
                title="k*lambda_D scan vs scattering angle",
                x_label="Scattering angle [deg]",
                y_label="k*lambda_D",
                x_values=np.asarray(angle_scan, dtype=np.float64),
                y_series=(np.asarray(angle_scan_klambda, dtype=np.float64),),
                curve_names=("Current snapshot",),
            ),
        )
    )

    # The plasmon tab uses a generic two-plot panel, so expose the study outputs
    # in the primary bundle list as well. This keeps the top plot focused on the
    # selected spectrum/dispersion workflow instead of defaulting to legacy state
    # traces, while still leaving the state traces available in the selector.
    preferred_time_keys = (
        ("dispersion_compare_policies", "dispersion_compare_models", "dispersion_selected_model", "dispersion_selected_width", "spectrum_compare_policies", "spectrum_compare_models", "spectrum_observed")
        if study_mode == PLASMON_STUDY_MODE_DISPERSION
        else ("spectrum_compare_policies", "spectrum_compare_models", "spectrum_observed", "dispersion_compare_policies", "dispersion_compare_models", "dispersion_selected_model", "dispersion_selected_width")
    )
    seen_study_keys: set[str] = set()
    study_time_bundles: list[DerivedPlotBundle] = []
    for key in preferred_time_keys:
        for bundle in profile_bundles:
            if bundle.key != key or key in seen_study_keys:
                continue
            study_time_bundles.append(bundle)
            seen_study_keys.add(key)
            break
    if study_time_bundles:
        time_plots = tuple(study_time_bundles) + tuple(time_plots)

    total_runtime_s = perf_counter() - total_start
    if total_runtime_s > 2.0:
        warnings.append(
            DerivedWarning(
                "plasmon",
                "Slow plasmon execution: "
                f"total={total_runtime_s:.2f}s | "
                f"spectrum={spectrum_runtime_s:.2f}s | "
                f"comparison={comparison_runtime_s:.2f}s | "
                f"dispersion={dispersion_runtime_s:.2f}s | "
                f"time_series={time_series_runtime_s:.2f}s.",
                severity="info",
            )
        )

    collision_metadata = _resolved_plasmon_collision_metadata(current, parameters)
    collision_rate_s = float(spectrum_payload["collision_rate_s"]) if spectrum_payload is not None and "collision_rate_s" in spectrum_payload else float(collision_metadata["rate_s"])
    return PlasmonResult(
        snapshot_index=int(snapshot_index),
        weighting_mode=weighting_mode,
        geometry_summary=geometry_summary,
        photon_energy_kev=float(parameters.plasmon_photon_energy_kev),
        scattering_angle_deg=float(parameters.plasmon_scattering_angle_deg),
        adiabatic_index=float(parameters.plasmon_adiabatic_index),
        electron_density_cm3=float(current["ne_cm3"]),
        electron_temperature_ev=float(current["te_ev"]),
        ion_temperature_ev=float(current["ti_ev"]),
        mean_charge=float(current["zbar"]),
        ion_mass_mu=float(current["ion_mass_mu"]),
        debye_length_cm=float(current["debye_length_cm"]),
        plasma_frequency_rad_s=float(current["omega_pe_rad_s"]),
        plasma_frequency_ev=float(current["omega_pe_ev"]),
        electron_collision_rate_s=float(collision_rate_s),
        coulomb_logarithm=float(current["coulomb_log"]),
        ion_sound_speed_cm_s=float(current["sound_speed_cm_s"]),
        probe_wavelength_angstrom=float(current["probe_wavelength_angstrom"]),
        scattering_wavevector_cm_inv=float(current["scattering_wavevector_cm_inv"]),
        scattering_wavevector_m_inv=float(current["scattering_wavevector_m_inv"]),
        k_lambda_debye=float(current["k_lambda"]),
        collectivity_parameter=float(current["collectivity"]),
        regime_label=str(current["regime"]),
        fermi_energy_ev=float(current["fermi_energy_ev"]),
        theta_degeneracy=float(current["theta_degeneracy"]),
        wigner_seitz_rs=float(current["wigner_seitz_rs"]),
        model_name=applied_model,
        requested_model_name=str(parameters.plasmon_model or PLASMON_MODEL_QUICKLOOK),
        execution_mode=_plasmon_execution_mode(parameters),
        integration_mode=(str(parameters.plasmon_integration_mode) if spectrum_payload is not None and str(parameters.plasmon_integration_mode) == PLASMON_INTEGRATION_MODE_LOS_INTEGRATED else "effective_state"),
        collision_model=str(parameters.plasmon_collision_model),
        collision_scale=float(parameters.plasmon_collision_scale),
        manual_collision_rate_s=float(parameters.plasmon_manual_collision_rate_s),
        lfc_model=str(parameters.plasmon_lfc_model),
        normalization=str(parameters.plasmon_normalization),
        observable_mode=(_plasmon_observable_mode(parameters) if spectrum_payload is None else str(spectrum_payload.get("observable_mode", _plasmon_observable_mode(parameters)))),
        observable_summary=(str(spectrum_payload.get("observable_summary", "")) if spectrum_payload is not None else ""),
        observable_decomposition_mode=(str(spectrum_payload.get("observable_decomposition_mode", "")) if spectrum_payload is not None else ""),
        observable_peak_extraction_mode=(str(spectrum_payload.get("observable_peak_extraction_mode", "positive_branch")) if spectrum_payload is not None else "positive_branch"),
        observable_elastic_exclusion_ev=(float(spectrum_payload.get("observable_elastic_exclusion_ev", 0.0)) if spectrum_payload is not None else 0.0),
        observable_free_fraction=(float(spectrum_payload.get("observable_free_fraction", float("nan"))) if spectrum_payload is not None else float("nan")),
        observable_bound_fraction=(float(spectrum_payload.get("observable_bound_fraction", float("nan"))) if spectrum_payload is not None else float("nan")),
        observable_elastic_fraction=(float(spectrum_payload.get("observable_elastic_fraction", float("nan"))) if spectrum_payload is not None else float("nan")),
        observable_comparison_mode=(str(spectrum_payload.get("observable_comparison_mode", "")) if spectrum_payload is not None else ""),
        observable_subtraction_mode=(str(spectrum_payload.get("observable_subtraction_mode", "")) if spectrum_payload is not None else ""),
        observable_normalization_mode=(str(spectrum_payload.get("observable_normalization_mode", "")) if spectrum_payload is not None else ""),
        observable_peak_discrete_energy_ev=(float(spectrum_payload.get("observable_peak_discrete_energy_ev", float("nan"))) if spectrum_payload is not None else float("nan")),
        observable_peak_fit_energy_ev=(float(spectrum_payload.get("observable_peak_fit_energy_ev", float("nan"))) if spectrum_payload is not None else float("nan")),
        observable_peak_fit_status=(str(spectrum_payload.get("observable_peak_fit_status", "")) if spectrum_payload is not None else ""),
        observable_peak_edge_dominated=(bool(spectrum_payload.get("observable_peak_edge_dominated", False)) if spectrum_payload is not None else False),
        observable_elastic_form_factor_total=(float(spectrum_payload.get("observable_elastic_form_factor_total", float("nan"))) if spectrum_payload is not None else float("nan")),
        observable_elastic_form_factor_core=(float(spectrum_payload.get("observable_elastic_form_factor_core", float("nan"))) if spectrum_payload is not None else float("nan")),
        observable_elastic_screening_form_factor=(float(spectrum_payload.get("observable_elastic_screening_form_factor", float("nan"))) if spectrum_payload is not None else float("nan")),
        observable_ion_structure_factor=(float(spectrum_payload.get("observable_ion_structure_factor", float("nan"))) if spectrum_payload is not None else float("nan")),
        observable_bound_core_mode=(str(spectrum_payload.get("observable_bound_core_mode", "")) if spectrum_payload is not None else ""),
        observable_bound_shell_summary=(str(spectrum_payload.get("observable_bound_shell_summary", "")) if spectrum_payload is not None else ""),
        spectrum_window_ev=float(parameters.plasmon_energy_window_ev),
        spectrum_points=(int(spectrum_payload.get("energy_ev", np.asarray([], dtype=np.float64)).size) if spectrum_payload is not None and np.asarray(spectrum_payload.get("energy_ev", np.asarray([], dtype=np.float64))).size > 0 else _resolved_energy_points(parameters)),
        instrument_fwhm_ev=float(parameters.plasmon_instrument_fwhm_ev),
        spectral_imag_shift_ev=(float(spectrum_payload["imag_shift_ev"]) if spectrum_payload is not None else 0.0),
        peak_fit_method=(str(spectrum_payload.get("peak_fit_method", "none")) if spectrum_payload is not None else "none"),
        peak_energy_ev=(float(spectrum_payload["peak_energy_ev"]) if spectrum_payload is not None else float("nan")),
        peak_fwhm_ev=(float(spectrum_payload["peak_fwhm_ev"]) if spectrum_payload is not None else float("nan")),
        static_lfc_value=(float(spectrum_payload.get("static_lfc_value", float("nan"))) if spectrum_payload is not None else float("nan")),
        q_over_qf=(float(spectrum_payload.get("q_over_qf", float("nan"))) if spectrum_payload is not None else float("nan")),
        response_backend=(str(spectrum_payload.get("response_backend", "classical_maxwellian")) if spectrum_payload is not None else "classical_maxwellian"),
        backend_summary=(str(spectrum_payload.get("backend_summary", "")) if spectrum_payload is not None else ""),
        stls_converged=(bool(spectrum_payload.get("stls_converged", False)) if spectrum_payload is not None else False),
        stls_iteration_count=(int(spectrum_payload.get("stls_iteration_count", 0)) if spectrum_payload is not None else 0),
        stls_convergence_residual=(float(spectrum_payload.get("stls_convergence_residual", float("nan"))) if spectrum_payload is not None else float("nan")),
        stls_convergence_relative_residual=(float(spectrum_payload.get("stls_convergence_relative_residual", float("nan"))) if spectrum_payload is not None else float("nan")),
        stls_closure_name=(str(spectrum_payload.get("stls_closure_name", "")) if spectrum_payload is not None else ""),
        stls_local_field_value=(float(spectrum_payload.get("stls_local_field_value", float("nan"))) if spectrum_payload is not None else float("nan")),
        stls_q_over_qf=(float(spectrum_payload.get("stls_q_over_qf", float("nan"))) if spectrum_payload is not None else float("nan")),
        auto_model_summary=(str(spectrum_payload.get("auto_model_summary", "")) if spectrum_payload is not None else ""),
        benchmark_preset=str(getattr(parameters, "plasmon_benchmark_preset", PLASMON_BENCHMARK_PRESET_NONE) or PLASMON_BENCHMARK_PRESET_NONE),
        requested_electron_policy=str(getattr(parameters, "plasmon_electron_policy", PLASMON_ELECTRON_POLICY_RAW) or PLASMON_ELECTRON_POLICY_RAW),
        electron_policy=str(electron_fields.policy),
        driven_response_model=str(getattr(electron_fields, "driven_response_model", "none")),
        driven_response_summary=str(getattr(electron_fields, "driven_response_summary", "")),
        driven_response_ensemble_mode=str(getattr(electron_fields, "driven_response_ensemble_mode", "")),
        electron_density_source=str(electron_fields.source_label),
        material_policy_summary=str(electron_fields.summary),
        resolved_materials=tuple(str(value) for value in getattr(electron_fields, "resolved_materials", ())),
        unresolved_materials=tuple(str(value) for value in getattr(electron_fields, "unresolved_materials", ())),
        raw_kept_materials=tuple(str(value) for value in getattr(electron_fields, "raw_kept_materials", ())),
        collision_source=(str(spectrum_payload.get("collision_source", collision_metadata.get("source", ""))) if spectrum_payload is not None else str(collision_metadata.get("source", ""))),
        collision_summary=(str(spectrum_payload.get("collision_summary", collision_metadata.get("summary", ""))) if spectrum_payload is not None else str(collision_metadata.get("summary", ""))),
        cluster_log_ne_tol=float(parameters.plasmon_cluster_log_ne_tol),
        cluster_log_te_tol=float(parameters.plasmon_cluster_log_te_tol),
        cluster_z_tol=float(parameters.plasmon_cluster_z_tol),
        study_mode=_plasmon_study_mode(parameters),
        coordinate_axis=_plasmon_coordinate_axis(parameters),
        coordinate_value=float(getattr(parameters, "plasmon_coordinate_value", parameters.plasmon_scattering_angle_deg)),
        scan_axis=_plasmon_scan_axis(parameters),
        scan_start=float(getattr(parameters, "plasmon_scan_start", 10.0)),
        scan_stop=float(getattr(parameters, "plasmon_scan_stop", 140.0)),
        scan_points=int(getattr(parameters, "plasmon_scan_points", 61)),
        compare_models=_plasmon_compare_models(parameters),
        comparison_models=tuple(str(value) for value in _requested_comparison_models(parameters)),
        compare_policies=_plasmon_compare_policies(parameters),
        policy_comparison_policies=tuple(str(value) for value in policy_comparison_policies),
        zone_count_used=(int(spectrum_payload.get("zone_count_used", int(np.count_nonzero(mask)))) if spectrum_payload is not None else int(np.count_nonzero(mask))),
        cluster_count_used=(int(spectrum_payload.get("cluster_count_used", 1 if np.any(mask) else 0)) if spectrum_payload is not None else (1 if np.any(mask) else 0)),
        benchmark_status=(str(spectrum_payload.get("benchmark_status", _BENCHMARK_STATUS_VALID)) if spectrum_payload is not None else _BENCHMARK_STATUS_NOT_APPLICABLE),
        model_executed_fully=(bool(spectrum_payload.get("model_executed_fully", True)) if spectrum_payload is not None else True),
        fallback_fraction=(float(spectrum_payload.get("fallback_fraction", 0.0)) if spectrum_payload is not None else 0.0),
        domain_failure_fraction=(float(spectrum_payload.get("domain_failure_fraction", 0.0)) if spectrum_payload is not None else 0.0),
        degenerate_zone_count=(int(spectrum_payload.get("degenerate_zone_count", 0)) if spectrum_payload is not None else 0),
        noncollective_zone_count=(int(spectrum_payload.get("noncollective_zone_count", 0)) if spectrum_payload is not None else 0),
        weak_coupling_zone_count=(int(spectrum_payload.get("weak_coupling_zone_count", 0)) if spectrum_payload is not None else 0),
        lfc_out_of_domain_zone_count=(int(spectrum_payload.get("lfc_out_of_domain_zone_count", 0)) if spectrum_payload is not None else 0),
        invalid_collision_zone_count=(int(spectrum_payload.get("invalid_collision_zone_count", 0)) if spectrum_payload is not None else 0),
        degenerate_cluster_count=(int(spectrum_payload.get("degenerate_cluster_count", 0)) if spectrum_payload is not None else 0),
        noncollective_cluster_count=(int(spectrum_payload.get("noncollective_cluster_count", 0)) if spectrum_payload is not None else 0),
        weak_coupling_cluster_count=(int(spectrum_payload.get("weak_coupling_cluster_count", 0)) if spectrum_payload is not None else 0),
        lfc_out_of_domain_cluster_count=(int(spectrum_payload.get("lfc_out_of_domain_cluster_count", 0)) if spectrum_payload is not None else 0),
        invalid_collision_cluster_count=(int(spectrum_payload.get("invalid_collision_cluster_count", 0)) if spectrum_payload is not None else 0),
        advanced_model_available=True,
        total_runtime_s=float(total_runtime_s),
        spectrum_runtime_s=float(spectrum_runtime_s),
        comparison_runtime_s=float(comparison_runtime_s),
        dispersion_runtime_s=float(dispersion_runtime_s),
        time_series_runtime_s=float(time_series_runtime_s),
        spectrum_energy_ev=(np.asarray(spectrum_payload["energy_ev"], dtype=np.float64) if spectrum_payload is not None else np.asarray([], dtype=np.float64)),
        spectrum_intensity=(np.asarray(spectrum_payload["spectrum"], dtype=np.float64) if spectrum_payload is not None else np.asarray([], dtype=np.float64)),
        spectrum_free_component=(np.asarray(spectrum_payload.get("free_component", np.asarray([], dtype=np.float64)), dtype=np.float64) if spectrum_payload is not None else np.asarray([], dtype=np.float64)),
        spectrum_bound_component=(np.asarray(spectrum_payload.get("bound_component", np.asarray([], dtype=np.float64)), dtype=np.float64) if spectrum_payload is not None else np.asarray([], dtype=np.float64)),
        spectrum_elastic_component=(np.asarray(spectrum_payload.get("elastic_component", np.asarray([], dtype=np.float64)), dtype=np.float64) if spectrum_payload is not None else np.asarray([], dtype=np.float64)),
        dielectric_real=(np.asarray(spectrum_payload["epsilon_real"], dtype=np.float64) if spectrum_payload is not None else np.asarray([], dtype=np.float64)),
        dielectric_imag=(np.asarray(spectrum_payload["epsilon_imag"], dtype=np.float64) if spectrum_payload is not None else np.asarray([], dtype=np.float64)),
        loss_function=(np.asarray(spectrum_payload["loss"], dtype=np.float64) if spectrum_payload is not None else np.asarray([], dtype=np.float64)),
        time_plots=time_plots,
        profile_plots=tuple(profile_bundles),
        warnings=tuple(warnings),
    )
