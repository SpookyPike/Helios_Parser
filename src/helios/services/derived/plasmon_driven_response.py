"""Driven-response models for article-facing plasmon benchmarking.

This layer is intentionally separate from:

- cold electronic baseline resolution;
- hydro/slab selection;
- plot/report generation.

It consumes an already selected state plus the JSON-backed cold baseline and
returns effective electron fields for response evaluation. The first production
goal is exact reproduction of the current scalar driven-increment control path
through an explicit abstraction, so future response-level models can be added
without hiding new benchmark heuristics inside the electron-policy code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from helios.services.derived.plasmon_units import (
    ELECTRON_MASS_KG,
    HBAR_EV_S,
    HBAR_J_S,
    coulomb_logarithm_ei_nrl_piecewise,
    electron_density_m3_from_cm3,
    electron_collision_rate_s,
    electron_fermi_wavevector_m_inv,
    electron_thermal_speed_m_s,
)


PLASMON_DRIVEN_RESPONSE_MODEL_NONE = "none"
PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL = "scalar_increment_control"
PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED = "electron_column_weighted_control"
PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE = "collision_shape_broadened_experimental"
PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE = "response_function_ensemble_experimental"

_SUPPORTED_DRIVEN_RESPONSE_MODELS = (
    PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
    PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
    PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED,
    PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE,
    PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
)

_ARTICLE_AL_COLD_RHO_G_CM3 = 2.70
_ARTICLE_AL_INCREMENT_TE_START_EV = 0.25
_ARTICLE_AL_INCREMENT_TE_SCALE_EV = 0.45
_ARTICLE_AL_INCREMENT_COMPRESSION_SCALE = 0.65
_ARTICLE_AL_INCREMENT_MAX = 0.55


@dataclass(frozen=True, slots=True)
class DrivenElectronResponseState:
    material_formula: str
    density_g_cm3: np.ndarray
    electron_temperature_ev: np.ndarray
    ion_temperature_ev: np.ndarray
    ion_density_cm3: np.ndarray
    raw_electron_density_cm3: np.ndarray
    raw_mean_charge: np.ndarray
    baseline_mean_charge: np.ndarray
    baseline_entry: str = ""
    baseline_table_source: str = ""
    state_origin: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DrivenElectronResponseResult:
    model: str
    effective_electron_density_cm3: np.ndarray
    effective_mean_charge: np.ndarray
    baseline_mean_charge: np.ndarray
    increment_mean_charge: np.ndarray
    mode: str
    summary: str
    provenance: tuple[str, ...] = ()
    response_modifiers: dict[str, Any] | None = None


def supported_driven_response_models() -> tuple[str, ...]:
    return _SUPPORTED_DRIVEN_RESPONSE_MODELS


def normalize_driven_response_model(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _SUPPORTED_DRIVEN_RESPONSE_MODELS:
        return normalized
    return PLASMON_DRIVEN_RESPONSE_MODEL_NONE


def default_driven_response_model_for_policy(policy: str | None) -> str:
    normalized_policy = str(policy or "").strip().lower()
    if normalized_policy == "article_al_driven_increment":
        return PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL
    return PLASMON_DRIVEN_RESPONSE_MODEL_NONE


def driven_response_model_label(value: str | None) -> str:
    normalized = normalize_driven_response_model(value)
    labels = {
        PLASMON_DRIVEN_RESPONSE_MODEL_NONE: "No driven response correction",
        PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL: "Scalar driven increment control",
        PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED: "Electron-column-weighted control (experimental)",
        PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE: "Collision-shape broadening (experimental)",
        PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE: "Response-function ensemble average (experimental)",
    }
    return labels.get(normalized, normalized.replace("_", " ").title())


def article_al_scalar_increment(
    density_g_cm3: np.ndarray | float,
    te_ev: np.ndarray | float,
    *,
    baseline_mean_charge: np.ndarray | float,
    raw_mean_charge: np.ndarray | float,
) -> np.ndarray:
    """Return the existing bounded scalar increment for article-facing Al.

    This reproduces the current benchmark control path exactly. It is kept here
    so the scalar policy becomes one explicit driven-response model rather than
    a special case buried inside electron-policy resolution.
    """

    density = np.asarray(density_g_cm3, dtype=np.float64)
    te = np.asarray(te_ev, dtype=np.float64)
    baseline = np.asarray(baseline_mean_charge, dtype=np.float64)
    raw = np.asarray(raw_mean_charge, dtype=np.float64)
    compression_excess = np.clip(density / _ARTICLE_AL_COLD_RHO_G_CM3 - 1.0, 0.0, None)
    thermal_factor = np.clip((te - _ARTICLE_AL_INCREMENT_TE_START_EV) / _ARTICLE_AL_INCREMENT_TE_SCALE_EV, 0.0, 1.0)
    compression_factor = np.clip(compression_excess / _ARTICLE_AL_INCREMENT_COMPRESSION_SCALE, 0.0, 1.0)
    heuristic_increment = _ARTICLE_AL_INCREMENT_MAX * thermal_factor * compression_factor
    raw_increment = np.clip(raw - baseline, 0.0, _ARTICLE_AL_INCREMENT_MAX)
    return np.clip(np.maximum(heuristic_increment, raw_increment), 0.0, _ARTICLE_AL_INCREMENT_MAX)


def article_al_effective_electron_column_weight(
    ion_density_cm3: np.ndarray | float,
    effective_mean_charge: np.ndarray | float,
) -> np.ndarray:
    """Return an explicit ensemble-weighting proxy for driven Al states.

    The synthetic article-driven benchmark averages spectra from several
    uniform-Al density nodes. A plain arithmetic mean treats every density node
    as if it contributed the same free-electron column, which is not physically
    neutral because the current DSF proxy is derived from ``Im[-1/eps]`` and
    does not carry a separate explicit ``n_e`` prefactor into the final
    benchmark-side averaging step. The first experimental non-scalar model
    therefore keeps the existing scalar-control electron fields but weights the
    benchmark ensemble by an effective free-electron column proxy:

        weight ~ n_i * Z_eff = n_e,eff

    This is still an approximation, but it is state-structured and density
    sensitive rather than another scalar ``Z_eff`` knob.
    """

    ion_density = np.asarray(ion_density_cm3, dtype=np.float64)
    effective_z = np.asarray(effective_mean_charge, dtype=np.float64)
    weight = ion_density * effective_z
    return np.clip(weight, 1.0e-300, None)


def article_al_collision_shape_broadening_fwhm_ev(
    te_ev: np.ndarray | float,
    effective_ne_cm3: np.ndarray | float,
    effective_mean_charge: np.ndarray | float,
) -> np.ndarray:
    """Return an hbar*nu line-shape broadening proxy for driven Al states.

    This is the first response-shape experiment in the scaffold. It keeps the
    scalar-control effective electron fields unchanged, but derives a per-state
    extra Gaussian broadening from the local collision-energy scale:

        delta_E ~ hbar * nu_ei

    For dense driven Al, the plain NRL Coulomb-log estimate can go invalid.
    Instead of silently zeroing the modifier, this experiment uses the same
    dense-state collision floor logic already accepted in the benchmark Mermin
    closure:

        nu_eff = max(nu_nrl, 0.05 * v_eff / a_i)

    and applies ``hbar * nu_eff`` as an extra Gaussian broadening proxy. The
    intent is to test whether the residual driven mismatch is more sensitive to
    state-resolved shape broadening than to further scalar ``Z_eff`` tuning or
    ensemble reweighting.
    """

    te = np.asarray(te_ev, dtype=np.float64)
    ne = np.asarray(effective_ne_cm3, dtype=np.float64)
    z_eff = np.asarray(effective_mean_charge, dtype=np.float64)
    broadening = np.zeros_like(te, dtype=np.float64)
    if broadening.size == 0:
        return broadening
    for index in np.ndindex(te.shape):
        te_value = float(te[index])
        ne_value = float(ne[index])
        z_value = float(z_eff[index])
        if not math.isfinite(te_value) or te_value <= 0.0 or not math.isfinite(ne_value) or ne_value <= 0.0:
            continue
        z_for_log = float(z_value) if math.isfinite(z_value) and z_value > 0.0 else 1.0
        coulomb_log = coulomb_logarithm_ei_nrl_piecewise(te_value, ne_value, z_for_log)
        collision_components: list[float] = []
        if math.isfinite(coulomb_log):
            base_collision = electron_collision_rate_s(ne_value, te_value, coulomb_log)
            if math.isfinite(base_collision) and base_collision > 0.0:
                collision_components.append(float(base_collision))
        ne_m3 = electron_density_m3_from_cm3(ne_value)
        ion_density_m3 = ne_m3 / z_for_log if z_for_log > 0.0 else float("nan")
        if math.isfinite(ion_density_m3) and ion_density_m3 > 0.0:
            ion_sphere_m = (3.0 / (4.0 * math.pi * ion_density_m3)) ** (1.0 / 3.0)
            thermal_speed = electron_thermal_speed_m_s(te_value)
            kf = electron_fermi_wavevector_m_inv(ne_value)
            fermi_speed = float("nan")
            if math.isfinite(kf) and kf > 0.0:
                fermi_speed = (HBAR_J_S * kf) / ELECTRON_MASS_KG
            v_eff = math.sqrt(max(thermal_speed, 0.0) ** 2 + max(fermi_speed if math.isfinite(fermi_speed) else 0.0, 0.0) ** 2)
            dense_floor = 0.05 * v_eff / ion_sphere_m if math.isfinite(v_eff) and v_eff > 0.0 and math.isfinite(ion_sphere_m) and ion_sphere_m > 0.0 else float("nan")
            if math.isfinite(dense_floor) and dense_floor > 0.0:
                collision_components.append(float(dense_floor))
        if not collision_components:
            continue
        collision_rate = max(collision_components)
        broadening[index] = max(float(HBAR_EV_S * collision_rate), 0.0)
    return broadening


def apply_driven_response_model(
    state: DrivenElectronResponseState,
    model: str | None,
) -> DrivenElectronResponseResult:
    normalized = normalize_driven_response_model(model)
    density = np.asarray(state.density_g_cm3, dtype=np.float64)
    te = np.asarray(state.electron_temperature_ev, dtype=np.float64)
    ion_density = np.asarray(state.ion_density_cm3, dtype=np.float64)
    baseline = np.asarray(state.baseline_mean_charge, dtype=np.float64)
    raw_zbar = np.asarray(state.raw_mean_charge, dtype=np.float64)
    material_formula = str(state.material_formula or "").strip()

    if normalized in {
        PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED,
        PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE,
        PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
    } and material_formula == "Al":
        increment = article_al_scalar_increment(
            density,
            te,
            baseline_mean_charge=baseline,
            raw_mean_charge=raw_zbar,
        )
        modifiers = {
            "compression_factor_mean": float(np.nanmean(np.clip(density / _ARTICLE_AL_COLD_RHO_G_CM3 - 1.0, 0.0, None))),
            "thermal_drive_mean_ev": float(np.nanmean(np.maximum(te - _ARTICLE_AL_INCREMENT_TE_START_EV, 0.0))),
        }
        mode = "cold_baseline_plus_bounded_increment"
    else:
        increment = np.zeros_like(baseline, dtype=np.float64)
        mode = "cold_baseline_only"
        modifiers = None

    effective_z = baseline + increment
    effective_ne = ion_density * effective_z
    if normalized == PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED and material_formula == "Al":
        ensemble_weight = article_al_effective_electron_column_weight(ion_density, effective_z)
        modifiers = dict(modifiers or {})
        modifiers.update(
            {
                "ensemble_weight_mode": "effective_electron_column",
                "ensemble_weight_multiplier": np.asarray(ensemble_weight, dtype=np.float64),
                "effective_electron_density_mean_cm3": float(np.nanmean(effective_ne)),
            }
        )
        mode = "cold_baseline_plus_bounded_increment_and_electron_column_weighting"
    elif normalized == PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE and material_formula == "Al":
        shape_broadening = article_al_collision_shape_broadening_fwhm_ev(
            te,
            effective_ne,
            effective_z,
        )
        modifiers = dict(modifiers or {})
        modifiers.update(
            {
                "shape_modifier_mode": "collision_hbar_nu_dense_gaussian",
                "shape_modifier_fwhm_ev": np.asarray(shape_broadening, dtype=np.float64),
                "shape_modifier_basis": "hbar_times_max_of_nrl_and_dense_floor_collision",
            }
        )
        mode = "cold_baseline_plus_bounded_increment_and_collision_shape_broadening"
    elif normalized == PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE and material_formula == "Al":
        modifiers = dict(modifiers or {})
        modifiers.update(
            {
                "ensemble_response_mode": "epsilon_average_before_loss",
                "ensemble_response_basis": "statewise_epsilon_average_uniform_grid",
            }
        )
        mode = "cold_baseline_plus_bounded_increment_and_response_function_ensemble"
    mean_baseline = float(np.nanmean(baseline)) if baseline.size else float("nan")
    mean_increment = float(np.nanmean(increment)) if increment.size else float("nan")
    mean_effective = float(np.nanmean(effective_z)) if effective_z.size else float("nan")
    summary = (
        f"{driven_response_model_label(normalized)} on {material_formula or 'unknown'}"
        f": baseline {mean_baseline:.3f} + increment {mean_increment:.3f} = {mean_effective:.3f}"
    )
    if normalized == PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED and modifiers is not None:
        summary = (
            f"{summary}; ensemble-weight="
            f"{str(modifiers.get('ensemble_weight_mode', 'unknown'))}"
            f" with mean {float(np.nanmean(np.asarray(modifiers.get('ensemble_weight_multiplier', np.asarray([float('nan')], dtype=np.float64)), dtype=np.float64))):.4e}"
        )
    if normalized == PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE and modifiers is not None:
        summary = (
            f"{summary}; shape-modifier="
            f"{str(modifiers.get('shape_modifier_mode', 'unknown'))}"
            f" with mean FWHM {float(np.nanmean(np.asarray(modifiers.get('shape_modifier_fwhm_ev', np.asarray([float('nan')], dtype=np.float64)), dtype=np.float64))):.4f} eV"
        )
    if normalized == PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE and modifiers is not None:
        summary = (
            f"{summary}; ensemble-response="
            f"{str(modifiers.get('ensemble_response_mode', 'unknown'))}"
        )
    provenance = tuple(
        value
        for value in (
            f"baseline_entry={state.baseline_entry}" if state.baseline_entry else "",
            f"baseline_source={state.baseline_table_source}" if state.baseline_table_source else "",
            f"state_origin={state.state_origin}" if state.state_origin else "",
            (
                f"ensemble_weight_mode={str(modifiers.get('ensemble_weight_mode', ''))}"
                if modifiers is not None and str(modifiers.get("ensemble_weight_mode", "")).strip()
                else ""
            ),
            (
                f"shape_modifier_mode={str(modifiers.get('shape_modifier_mode', ''))}"
                if modifiers is not None and str(modifiers.get("shape_modifier_mode", "")).strip()
                else ""
            ),
            (
                f"ensemble_response_mode={str(modifiers.get('ensemble_response_mode', ''))}"
                if modifiers is not None and str(modifiers.get("ensemble_response_mode", "")).strip()
                else ""
            ),
            *tuple(str(note) for note in state.notes if str(note).strip()),
        )
        if value
    )
    return DrivenElectronResponseResult(
        model=normalized,
        effective_electron_density_cm3=np.asarray(effective_ne, dtype=np.float64),
        effective_mean_charge=np.asarray(effective_z, dtype=np.float64),
        baseline_mean_charge=np.asarray(baseline, dtype=np.float64),
        increment_mean_charge=np.asarray(increment, dtype=np.float64),
        mode=mode,
        summary=summary,
        provenance=provenance,
        response_modifiers=modifiers,
    )


def response_model_fields_equal(
    left: DrivenElectronResponseResult,
    right: DrivenElectronResponseResult,
    *,
    atol: float = 1.0e-12,
    rtol: float = 0.0,
) -> bool:
    return (
        np.allclose(np.asarray(left.effective_mean_charge, dtype=np.float64), np.asarray(right.effective_mean_charge, dtype=np.float64), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(left.increment_mean_charge, dtype=np.float64), np.asarray(right.increment_mean_charge, dtype=np.float64), atol=atol, rtol=rtol)
        and np.allclose(np.asarray(left.effective_electron_density_cm3, dtype=np.float64), np.asarray(right.effective_electron_density_cm3, dtype=np.float64), atol=atol, rtol=rtol)
    )
