"""Material-specific XRTS observable construction for plasmon benchmarks.

This layer sits above the dielectric backend. Existing backends still provide
the free-electron loss / DSF response, while this module constructs a minimal
experiment-facing observable using an explicit Chihara-like decomposition:

- free-electron inelastic term from the selected backend
- elastic / ion-feature proxy centered at zero energy transfer
- bound/core inelastic term, kept explicit even when it is negligible in the
  narrow Al article benchmark window

The first implementation is intentionally conservative and material-specific:
it is designed for article-facing aluminium XRTS. Unknown or mixed-material
subsets fall back to the dielectric/free-electron spectrum with explicit
provenance instead of inventing hidden atomic physics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from helios.services.derived.plasmon_config import (
    PLASMON_OBSERVABLE_MODE_DIELECTRIC,
    PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
    PLASMON_OBSERVABLE_MODE_XRTS,
)
from helios.services.derived.plasmon_electron_policy import material_atomic_number
from helios.services.derived.plasmon_spectrum import estimate_peak_metrics, gaussian_convolve


_AL_L_SHELL_ONSET_EV = 72.0
_DEFAULT_ELASTIC_INTRINSIC_FWHM_EV = 0.18


@dataclass(frozen=True, slots=True)
class XrtsObservableComponents:
    mode: str
    decomposition_mode: str
    summary: str
    provenance: tuple[str, ...]
    free_raw: np.ndarray
    bound_raw: np.ndarray
    elastic_raw: np.ndarray
    total_raw: np.ndarray
    free_fraction: float
    bound_fraction: float
    elastic_fraction: float


def normalize_observable_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE:
        return PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE
    if normalized == PLASMON_OBSERVABLE_MODE_XRTS:
        return PLASMON_OBSERVABLE_MODE_XRTS
    return PLASMON_OBSERVABLE_MODE_DIELECTRIC


def observable_mode_label(value: str | None) -> str:
    normalized = normalize_observable_mode(value)
    if normalized == PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE:
        return "XRTS article-native Al"
    if normalized == PLASMON_OBSERVABLE_MODE_XRTS:
        return "XRTS observable"
    return "Dielectric-only"


def _integral_area(energy_ev: np.ndarray, values: np.ndarray) -> float:
    energy = np.asarray(energy_ev, dtype=np.float64)
    spectrum = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(energy) & np.isfinite(spectrum)
    if np.count_nonzero(finite) < 2:
        return 0.0
    x = energy[finite]
    y = np.clip(spectrum[finite], a_min=0.0, a_max=None)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _normalization_scale(energy_ev: np.ndarray, values: np.ndarray, mode: str) -> float:
    normalized = str(mode or "peak").strip().lower()
    spectrum = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(spectrum)
    if normalized == "none" or not np.any(finite):
        return 1.0
    if normalized == "area":
        area = _integral_area(energy_ev, spectrum)
        return (1.0 / area) if math.isfinite(area) and area > 0.0 else 1.0
    peak = float(np.nanmax(spectrum[finite]))
    return (1.0 / peak) if math.isfinite(peak) and peak > 0.0 else 1.0


def _gaussian_area_profile(
    energy_ev: np.ndarray,
    *,
    center_ev: float,
    fwhm_ev: float,
    area: float,
) -> np.ndarray:
    energy = np.asarray(energy_ev, dtype=np.float64)
    if energy.size == 0:
        return np.asarray([], dtype=np.float64)
    if not math.isfinite(float(area)) or float(area) <= 0.0:
        return np.zeros_like(energy, dtype=np.float64)
    width = float(fwhm_ev)
    if not math.isfinite(width) or width <= 0.0:
        width = _DEFAULT_ELASTIC_INTRINSIC_FWHM_EV
    sigma = width / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    values = np.exp(-((energy - float(center_ev)) ** 2) / (2.0 * sigma * sigma), dtype=np.float64)
    current_area = _integral_area(energy, values)
    if not math.isfinite(current_area) or current_area <= 0.0:
        return np.zeros_like(energy, dtype=np.float64)
    return values * (float(area) / current_area)


def _resolve_atomic_number(material_formula: str | None) -> int | None:
    formula = str(material_formula or "").strip()
    atomic_number = material_atomic_number(formula)
    if atomic_number is not None:
        return int(atomic_number)
    if formula == "Al":
        return 13
    return None


def _al_bound_form_factor(
    *,
    bound_electrons: float,
    scattering_wavevector_m_inv: float,
    ion_density_cm3: float,
) -> float:
    if not math.isfinite(bound_electrons) or bound_electrons <= 0.0:
        return 0.0
    if not math.isfinite(scattering_wavevector_m_inv) or scattering_wavevector_m_inv <= 0.0:
        return float(bound_electrons)
    ion_density_m3 = float(ion_density_cm3) * 1.0e6
    if not math.isfinite(ion_density_m3) or ion_density_m3 <= 0.0:
        return float(bound_electrons)
    ion_sphere_radius_m = (3.0 / (4.0 * math.pi * ion_density_m3)) ** (1.0 / 3.0)
    qa = float(scattering_wavevector_m_inv) * ion_sphere_radius_m
    return float(bound_electrons * math.exp(-(qa * qa) / 6.0))


def build_minimal_xrts_components(
    energy_ev: np.ndarray,
    free_raw_spectrum: np.ndarray,
    *,
    material_formula: str | None,
    electron_density_cm3: float,
    mean_charge: float,
    scattering_wavevector_m_inv: float,
    spectrum_window_ev: float,
) -> XrtsObservableComponents:
    energy = np.asarray(energy_ev, dtype=np.float64)
    free_raw = np.asarray(free_raw_spectrum, dtype=np.float64)
    if energy.shape != free_raw.shape:
        empty = np.zeros_like(energy, dtype=np.float64)
        return XrtsObservableComponents(
            mode=PLASMON_OBSERVABLE_MODE_XRTS,
            decomposition_mode="invalid_shape",
            summary="Observable construction failed because the free-electron spectrum and energy grid were misaligned.",
            provenance=("invalid_shape",),
            free_raw=empty,
            bound_raw=empty,
            elastic_raw=empty,
            total_raw=empty,
            free_fraction=float("nan"),
            bound_fraction=float("nan"),
            elastic_fraction=float("nan"),
        )

    atomic_number = _resolve_atomic_number(material_formula)
    formula = str(material_formula or "").strip()
    if atomic_number is None or formula != "Al":
        total_area = _integral_area(energy, free_raw)
        return XrtsObservableComponents(
            mode=PLASMON_OBSERVABLE_MODE_XRTS,
            decomposition_mode="free_only_passthrough",
            summary=(
                "XRTS observable mode fell back to the backend free-electron spectrum because the active subset is not a single supported Al material with a known atomic-number entry."
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

    z_free = min(max(float(mean_charge), 0.0), float(atomic_number))
    z_bound = max(float(atomic_number) - z_free, 0.0)
    ion_density_cm3 = (float(electron_density_cm3) / z_free) if math.isfinite(z_free) and z_free > 0.0 else float("nan")
    free_component = np.asarray(free_raw, dtype=np.float64) * max(z_free, 0.0)
    bound_component = np.zeros_like(free_component, dtype=np.float64)

    form_factor = _al_bound_form_factor(
        bound_electrons=z_bound,
        scattering_wavevector_m_inv=float(scattering_wavevector_m_inv),
        ion_density_cm3=float(ion_density_cm3),
    )
    elastic_area = max(form_factor * form_factor, 0.0)
    elastic_component = _gaussian_area_profile(
        energy,
        center_ev=0.0,
        fwhm_ev=_DEFAULT_ELASTIC_INTRINSIC_FWHM_EV,
        area=elastic_area,
    )

    window_limit = float(np.nanmax(np.abs(energy))) if energy.size else float(spectrum_window_ev)
    bound_mode = "suppressed_below_L_shell_threshold"
    bound_note = (
        "Bound/core inelastic term was set to zero because the narrow Al benchmark window stays below the first L-shell onset, so this layer currently keeps only the free-electron term and the central elastic/ion-feature proxy."
        if math.isfinite(window_limit) and window_limit < _AL_L_SHELL_ONSET_EV
        else "Bound/core inelastic term is not yet modeled explicitly for Al outside the narrow benchmark window, so the current observable keeps it at zero."
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
        "Minimal Chihara-like Al observable: backend free-electron DSF plus an explicit central elastic/ion-feature proxy from the bound-electron form factor; bound/core inelastic term is kept explicit and currently zero in the narrow article benchmark window. "
        + bound_note
    )
    return XrtsObservableComponents(
        mode=PLASMON_OBSERVABLE_MODE_XRTS,
        decomposition_mode="minimal_chihara_like_al",
        summary=summary,
        provenance=(
            "free=backend_dsf",
            "elastic=bound_form_factor_proxy",
            f"bound_mode={bound_mode}",
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


def finalize_xrts_observable(
    energy_ev: np.ndarray,
    components: XrtsObservableComponents,
    *,
    instrument_fwhm_ev: float,
    normalization: str,
    peak_fit_method: str,
    elastic_exclusion_ev: float | None = None,
) -> dict[str, object]:
    energy = np.asarray(energy_ev, dtype=np.float64)
    free = gaussian_convolve(energy, np.asarray(components.free_raw, dtype=np.float64), float(instrument_fwhm_ev))
    bound = gaussian_convolve(energy, np.asarray(components.bound_raw, dtype=np.float64), float(instrument_fwhm_ev))
    elastic = gaussian_convolve(energy, np.asarray(components.elastic_raw, dtype=np.float64), float(instrument_fwhm_ev))
    total = free + bound + elastic
    scale = _normalization_scale(energy, total, normalization)
    free *= scale
    bound *= scale
    elastic *= scale
    total *= scale

    exclusion = float(elastic_exclusion_ev) if elastic_exclusion_ev is not None else max(2.0, 1.5 * float(instrument_fwhm_ev))
    mask = np.isfinite(energy) & np.isfinite(total) & (energy >= exclusion)
    if np.count_nonzero(mask) >= 3:
        fit_mode = str(peak_fit_method or "quadratic")
        half_window = 2 if fit_mode == "local_quadratic" else 1
        peak_energy_ev, peak_fwhm_ev = estimate_peak_metrics(
            energy[mask],
            total[mask],
            method=fit_mode,
            local_half_window_points=half_window,
        )
        peak_extraction_mode = "positive_branch_excluding_elastic_core"
    else:
        peak_energy_ev, peak_fwhm_ev = float("nan"), float("nan")
        peak_extraction_mode = "no_valid_inelastic_branch"

    return {
        "spectrum": np.asarray(total, dtype=np.float64),
        "free_component": np.asarray(free, dtype=np.float64),
        "bound_component": np.asarray(bound, dtype=np.float64),
        "elastic_component": np.asarray(elastic, dtype=np.float64),
        "observable_mode": PLASMON_OBSERVABLE_MODE_XRTS,
        "observable_summary": str(components.summary),
        "observable_decomposition_mode": str(components.decomposition_mode),
        "observable_peak_extraction_mode": str(peak_extraction_mode),
        "observable_elastic_exclusion_ev": float(exclusion),
        "observable_free_fraction": float(components.free_fraction),
        "observable_bound_fraction": float(components.bound_fraction),
        "observable_elastic_fraction": float(components.elastic_fraction),
        "peak_energy_ev": float(peak_energy_ev),
        "peak_fwhm_ev": float(peak_fwhm_ev),
        "observable_provenance": tuple(str(value) for value in components.provenance),
    }
