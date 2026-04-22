"""Material-specific Al XRTS observable construction.

This module upgrades the minimal XRTS observable seam with an explicitly
material-specific Al decomposition:

- free-electron inelastic response from the selected dielectric backend
- elastic / ion feature from Al atomic form factors
- shell-resolved bound/core bookkeeping with explicit zero below threshold in
  the narrow article benchmark window

It does not attempt full atomic-physics fidelity. The goal is an honest,
article-facing Al observable layer that remains explicit about its limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from helios.services.derived.plasmon_config import PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE
from helios.services.derived.plasmon_spectrum import estimate_peak_metrics, gaussian_convolve
from helios.services.derived.plasmon_xrts_observable import (
    XrtsObservableComponents,
    _gaussian_area_profile,
    _integral_area,
)


@dataclass(frozen=True, slots=True)
class _CromerMannCoefficients:
    a: tuple[float, float, float, float]
    b: tuple[float, float, float, float]
    c: float


# Neutral-Al coefficients from the IUCr Cromer-Mann tables
# (International Tables for Crystallography / CIF core dictionary).
_AL_NEUTRAL_COEFFS = _CromerMannCoefficients(
    a=(6.4202, 1.9002, 1.5936, 1.9646),
    b=(3.0387, 0.7426, 31.5472, 85.0886),
    c=1.1151,
)

# Al3+ coefficients from the same Cromer-Mann tables. This is used as the
# compact bound/core form-factor proxy for the Al ion after the nominal three
# valence electrons are promoted into the free-electron response.
_AL_3PLUS_COEFFS = _CromerMannCoefficients(
    a=(4.17448, 3.3876, 1.20296, 0.528137),
    b=(1.93816, 4.14553, 0.228753, 8.28524),
    c=0.706786,
)

_AL_L_SHELL_ONSET_EV = 72.6
_AL_K_SHELL_ONSET_EV = 1559.6
_AL_CORE_ELECTRONS = 10.0
_AL_VALENCE_ELECTRONS = 3.0
_DEFAULT_ELASTIC_INTRINSIC_FWHM_EV = 0.18


def _q_angstrom_inv(scattering_wavevector_m_inv: float) -> float:
    return float(scattering_wavevector_m_inv) * 1.0e-10


def _cromer_mann_form_factor(q_ang_inv: float, coeffs: _CromerMannCoefficients) -> float:
    if not math.isfinite(q_ang_inv) or q_ang_inv < 0.0:
        return float("nan")
    s = float(q_ang_inv) / (4.0 * math.pi)
    s2 = s * s
    return float(
        sum(float(a_i) * math.exp(-float(b_i) * s2) for a_i, b_i in zip(coeffs.a, coeffs.b))
        + float(coeffs.c)
    )


def build_article_native_al_components(
    energy_ev: np.ndarray,
    free_raw_spectrum: np.ndarray,
    *,
    material_formula: str | None,
    electron_density_cm3: float,
    mean_charge: float,
    scattering_wavevector_m_inv: float,
    spectrum_window_ev: float,
) -> tuple[XrtsObservableComponents, dict[str, object]]:
    energy = np.asarray(energy_ev, dtype=np.float64)
    free_raw = np.asarray(free_raw_spectrum, dtype=np.float64)
    if energy.shape != free_raw.shape:
        empty = np.zeros_like(energy, dtype=np.float64)
        components = XrtsObservableComponents(
            mode=PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
            decomposition_mode="invalid_shape",
            summary="Article-native Al observable construction failed because the backend spectrum and energy grid were misaligned.",
            provenance=("invalid_shape",),
            free_raw=empty,
            bound_raw=empty,
            elastic_raw=empty,
            total_raw=empty,
            free_fraction=float("nan"),
            bound_fraction=float("nan"),
            elastic_fraction=float("nan"),
        )
        diagnostics = {
            "observable_comparison_mode": "invalid_shape",
            "observable_subtraction_mode": "invalid_shape",
            "observable_normalization_mode": "explicit_not_hidden",
            "observable_peak_discrete_energy_ev": float("nan"),
            "observable_peak_fit_energy_ev": float("nan"),
            "observable_peak_fit_status": "invalid_shape",
            "observable_peak_edge_dominated": False,
            "observable_elastic_form_factor_total": float("nan"),
            "observable_elastic_form_factor_core": float("nan"),
            "observable_elastic_screening_form_factor": float("nan"),
            "observable_ion_structure_factor": float("nan"),
            "observable_bound_core_mode": "invalid_shape",
            "observable_bound_shell_summary": "",
        }
        return components, diagnostics

    formula = str(material_formula or "").strip()
    if formula != "Al":
        total_area = _integral_area(energy, free_raw)
        components = XrtsObservableComponents(
            mode=PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
            decomposition_mode="free_only_passthrough",
            summary=(
                "Article-native Al observable mode fell back to the backend free-electron spectrum because the active subset is not a single Al material."
            ),
            provenance=("free_only_passthrough", f"material={formula or 'unknown'}"),
            free_raw=np.asarray(free_raw, dtype=np.float64),
            bound_raw=np.zeros_like(free_raw, dtype=np.float64),
            elastic_raw=np.zeros_like(free_raw, dtype=np.float64),
            total_raw=np.asarray(free_raw, dtype=np.float64),
            free_fraction=(1.0 if total_area > 0.0 else float("nan")),
            bound_fraction=(0.0 if total_area > 0.0 else float("nan")),
            elastic_fraction=(0.0 if total_area > 0.0 else float("nan")),
        )
        diagnostics = {
            "observable_comparison_mode": "free_only_passthrough",
            "observable_subtraction_mode": "none",
            "observable_normalization_mode": "explicit_not_hidden",
            "observable_peak_discrete_energy_ev": float("nan"),
            "observable_peak_fit_energy_ev": float("nan"),
            "observable_peak_fit_status": "free_only_passthrough",
            "observable_peak_edge_dominated": False,
            "observable_elastic_form_factor_total": float("nan"),
            "observable_elastic_form_factor_core": float("nan"),
            "observable_elastic_screening_form_factor": float("nan"),
            "observable_ion_structure_factor": float("nan"),
            "observable_bound_core_mode": "unsupported_material",
            "observable_bound_shell_summary": "",
        }
        return components, diagnostics

    z_free = max(float(mean_charge), 0.0) if math.isfinite(float(mean_charge)) else _AL_VALENCE_ELECTRONS
    free_component = np.asarray(free_raw, dtype=np.float64) * max(z_free, 0.0)
    bound_component = np.zeros_like(free_component, dtype=np.float64)

    q_ang_inv = _q_angstrom_inv(float(scattering_wavevector_m_inv))
    neutral_form_factor = _cromer_mann_form_factor(q_ang_inv, _AL_NEUTRAL_COEFFS)
    core_form_factor = _cromer_mann_form_factor(q_ang_inv, _AL_3PLUS_COEFFS)
    screening_form_factor = (
        float(neutral_form_factor - core_form_factor)
        if math.isfinite(neutral_form_factor) and math.isfinite(core_form_factor)
        else float("nan")
    )
    ion_structure_factor = 1.0
    elastic_area = (
        max(float(neutral_form_factor * neutral_form_factor * ion_structure_factor), 0.0)
        if math.isfinite(neutral_form_factor)
        else 0.0
    )
    elastic_component = _gaussian_area_profile(
        energy,
        center_ev=0.0,
        fwhm_ev=_DEFAULT_ELASTIC_INTRINSIC_FWHM_EV,
        area=elastic_area,
    )

    window_limit = float(np.nanmax(np.abs(energy))) if energy.size else float(spectrum_window_ev)
    bound_core_mode = (
        "shell_thresholded_zero_below_al_l_shell"
        if math.isfinite(window_limit) and window_limit < _AL_L_SHELL_ONSET_EV
        else "al_bound_core_not_yet_modeled_above_threshold"
    )
    bound_shell_summary = (
        f"K(2@{_AL_K_SHELL_ONSET_EV:.1f}eV)=inactive; "
        f"L(8@{_AL_L_SHELL_ONSET_EV:.1f}eV)=inactive; "
        f"core_electrons={_AL_CORE_ELECTRONS:.1f}; nominal_valence={_AL_VALENCE_ELECTRONS:.1f}"
    )

    total_raw = free_component + bound_component + elastic_component
    free_area = _integral_area(energy, free_component)
    bound_area = _integral_area(energy, bound_component)
    elastic_area_measured = _integral_area(energy, elastic_component)
    total_area = free_area + bound_area + elastic_area_measured
    if not math.isfinite(total_area) or total_area <= 0.0:
        free_fraction = bound_fraction = elastic_fraction = float("nan")
    else:
        free_fraction = float(free_area / total_area)
        bound_fraction = float(bound_area / total_area)
        elastic_fraction = float(elastic_area_measured / total_area)

    summary = (
        "Article-native Al observable: backend free-electron DSF plus an Al-specific elastic/ion feature built from "
        "Cromer-Mann atomic form factors. The bound/core inelastic channel is shell-resolved and stays zero in the "
        f"current benchmark window because |omega| < {_AL_L_SHELL_ONSET_EV:.1f} eV remains below the first Al L-shell onset."
    )
    components = XrtsObservableComponents(
        mode=PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
        decomposition_mode="article_native_al_chihara_like",
        summary=summary,
        provenance=(
            "free=backend_dsf",
            "elastic=al_cromer_mann_neutral_form_factor",
            "bound_core=shell_thresholded_zero_below_l_shell",
            "ion_structure_factor=unity_assumption",
            "comparison=inelastic_branch_after_explicit_elastic_subtraction",
            "normalization=explicit_not_hidden",
        ),
        free_raw=np.asarray(free_component, dtype=np.float64),
        bound_raw=np.asarray(bound_component, dtype=np.float64),
        elastic_raw=np.asarray(elastic_component, dtype=np.float64),
        total_raw=np.asarray(total_raw, dtype=np.float64),
        free_fraction=float(free_fraction),
        bound_fraction=float(bound_fraction),
        elastic_fraction=float(elastic_fraction),
    )
    diagnostics = {
        "observable_comparison_mode": "inelastic_branch_after_explicit_elastic_subtraction",
        "observable_subtraction_mode": "explicit_elastic_component_removed_before_peak_fit",
        "observable_normalization_mode": "explicit_not_hidden",
        "observable_peak_discrete_energy_ev": float("nan"),
        "observable_peak_fit_energy_ev": float("nan"),
        "observable_peak_fit_status": "",
        "observable_peak_edge_dominated": False,
        "observable_elastic_form_factor_total": float(neutral_form_factor),
        "observable_elastic_form_factor_core": float(core_form_factor),
        "observable_elastic_screening_form_factor": float(screening_form_factor),
        "observable_ion_structure_factor": float(ion_structure_factor),
        "observable_bound_core_mode": str(bound_core_mode),
        "observable_bound_shell_summary": str(bound_shell_summary),
        "observable_effective_ne_cm3": float(electron_density_cm3),
        "observable_effective_z_free": float(z_free),
    }
    return components, diagnostics


def _extract_article_native_peak(
    energy_ev: np.ndarray,
    inelastic_spectrum: np.ndarray,
    *,
    elastic_exclusion_ev: float,
    peak_fit_method: str,
) -> dict[str, object]:
    energy = np.asarray(energy_ev, dtype=np.float64)
    values = np.asarray(inelastic_spectrum, dtype=np.float64)
    mask = np.isfinite(energy) & np.isfinite(values) & (energy >= float(elastic_exclusion_ev))
    if np.count_nonzero(mask) < 3:
        return {
            "peak_energy_ev": float("nan"),
            "peak_fwhm_ev": float("nan"),
            "observable_peak_discrete_energy_ev": float("nan"),
            "observable_peak_fit_energy_ev": float("nan"),
            "observable_peak_fit_status": "no_valid_inelastic_branch",
            "observable_peak_edge_dominated": False,
            "observable_peak_extraction_mode": "no_valid_inelastic_branch",
        }
    active_energy = np.asarray(energy[mask], dtype=np.float64)
    active_values = np.asarray(values[mask], dtype=np.float64)
    discrete_index = int(np.nanargmax(active_values))
    discrete_energy = float(active_energy[discrete_index])
    edge_dominated = bool(discrete_index < 2 or discrete_index >= active_values.size - 2)
    fit_mode = str(peak_fit_method or "local_quadratic")
    fit_energy = float("nan")
    fit_status = "discrete"
    if fit_mode.lower() in {"quadratic", "local_quadratic", "publication"} and not edge_dominated:
        fit_energy, fit_fwhm = estimate_peak_metrics(
            active_energy,
            active_values,
            method=fit_mode,
            local_half_window_points=2 if fit_mode.lower() == "local_quadratic" else 1,
        )
        grid_step = float(np.median(np.diff(active_energy))) if active_energy.size >= 2 else 0.0
        allowed_shift = max(1.0, 8.0 * abs(grid_step))
        if (
            math.isfinite(fit_energy)
            and fit_energy >= float(elastic_exclusion_ev)
            and abs(float(fit_energy) - discrete_energy) <= allowed_shift
        ):
            return {
                "peak_energy_ev": float(fit_energy),
                "peak_fwhm_ev": float(fit_fwhm),
                "observable_peak_discrete_energy_ev": float(discrete_energy),
                "observable_peak_fit_energy_ev": float(fit_energy),
                "observable_peak_fit_status": "accepted_local_quadratic",
                "observable_peak_edge_dominated": bool(edge_dominated),
                "observable_peak_extraction_mode": "inelastic_branch_after_elastic_subtraction",
            }
        fit_status = "fallback_from_unstable_local_quadratic"
    peak_energy_ev, peak_fwhm_ev = estimate_peak_metrics(active_energy, active_values, method="discrete")
    return {
        "peak_energy_ev": float(peak_energy_ev),
        "peak_fwhm_ev": float(peak_fwhm_ev),
        "observable_peak_discrete_energy_ev": float(discrete_energy),
        "observable_peak_fit_energy_ev": float(fit_energy),
        "observable_peak_fit_status": str("edge_discrete_fallback" if edge_dominated else fit_status),
        "observable_peak_edge_dominated": bool(edge_dominated),
        "observable_peak_extraction_mode": "inelastic_branch_after_elastic_subtraction",
    }


def finalize_article_native_observable(
    energy_ev: np.ndarray,
    components: XrtsObservableComponents,
    *,
    instrument_fwhm_ev: float,
    normalization: str,
    peak_fit_method: str,
    diagnostics: dict[str, object],
    elastic_exclusion_ev: float | None = None,
) -> dict[str, object]:
    energy = np.asarray(energy_ev, dtype=np.float64)
    free = gaussian_convolve(energy, np.asarray(components.free_raw, dtype=np.float64), float(instrument_fwhm_ev))
    bound = gaussian_convolve(energy, np.asarray(components.bound_raw, dtype=np.float64), float(instrument_fwhm_ev))
    elastic = gaussian_convolve(energy, np.asarray(components.elastic_raw, dtype=np.float64), float(instrument_fwhm_ev))
    total = free + bound + elastic

    scale = 1.0
    normalized = str(normalization or "peak").strip().lower()
    if normalized == "area":
        area = _integral_area(energy, total)
        scale = (1.0 / area) if math.isfinite(area) and area > 0.0 else 1.0
    elif normalized == "peak":
        finite = np.isfinite(total)
        peak = float(np.nanmax(total[finite])) if np.any(finite) else float("nan")
        scale = (1.0 / peak) if math.isfinite(peak) and peak > 0.0 else 1.0
    free *= scale
    bound *= scale
    elastic *= scale
    total *= scale

    exclusion = float(elastic_exclusion_ev) if elastic_exclusion_ev is not None else max(2.0, 1.5 * float(instrument_fwhm_ev))
    inelastic = free + bound
    extracted = _extract_article_native_peak(
        energy,
        inelastic,
        elastic_exclusion_ev=float(exclusion),
        peak_fit_method=str(peak_fit_method or "local_quadratic"),
    )
    return {
        "spectrum": np.asarray(total, dtype=np.float64),
        "free_component": np.asarray(free, dtype=np.float64),
        "bound_component": np.asarray(bound, dtype=np.float64),
        "elastic_component": np.asarray(elastic, dtype=np.float64),
        "observable_mode": PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
        "observable_summary": str(components.summary),
        "observable_decomposition_mode": str(components.decomposition_mode),
        "observable_peak_extraction_mode": str(extracted["observable_peak_extraction_mode"]),
        "observable_elastic_exclusion_ev": float(exclusion),
        "observable_free_fraction": float(components.free_fraction),
        "observable_bound_fraction": float(components.bound_fraction),
        "observable_elastic_fraction": float(components.elastic_fraction),
        "peak_energy_ev": float(extracted["peak_energy_ev"]),
        "peak_fwhm_ev": float(extracted["peak_fwhm_ev"]),
        "observable_provenance": tuple(str(value) for value in components.provenance),
        "observable_comparison_mode": str(diagnostics.get("observable_comparison_mode", "")),
        "observable_subtraction_mode": str(diagnostics.get("observable_subtraction_mode", "")),
        "observable_normalization_mode": str(normalization or diagnostics.get("observable_normalization_mode", "")),
        "observable_peak_discrete_energy_ev": float(extracted["observable_peak_discrete_energy_ev"]),
        "observable_peak_fit_energy_ev": float(extracted["observable_peak_fit_energy_ev"]),
        "observable_peak_fit_status": str(extracted["observable_peak_fit_status"]),
        "observable_peak_edge_dominated": bool(extracted["observable_peak_edge_dominated"]),
        "observable_elastic_form_factor_total": float(diagnostics.get("observable_elastic_form_factor_total", float("nan"))),
        "observable_elastic_form_factor_core": float(diagnostics.get("observable_elastic_form_factor_core", float("nan"))),
        "observable_elastic_screening_form_factor": float(diagnostics.get("observable_elastic_screening_form_factor", float("nan"))),
        "observable_ion_structure_factor": float(diagnostics.get("observable_ion_structure_factor", float("nan"))),
        "observable_bound_core_mode": str(diagnostics.get("observable_bound_core_mode", "")),
        "observable_bound_shell_summary": str(diagnostics.get("observable_bound_shell_summary", "")),
    }
