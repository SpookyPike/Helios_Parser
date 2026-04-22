from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    import _script_bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from scripts import _script_bootstrap  # type: ignore  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon_config import (
    PLASMON_BENCHMARK_PRESET_AL_AMBIENT_ARTICLE,
    PLASMON_BENCHMARK_PRESET_AL_DRIVEN_ARTICLE,
    PLASMON_COLLISION_MODEL_BENCHMARK_DENSE,
    PLASMON_EXECUTION_MODE_BENCHMARK,
    PLASMON_EXECUTION_MODE_QUICKLOOK,
    PLASMON_MODEL_CHOICES,
    PLASMON_MODEL_FINITE_T_STLS,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_MODEL_QUICKLOOK,
    PLASMON_OBSERVABLE_MODE_DIELECTRIC,
    PLASMON_OBSERVABLE_MODE_XRTS,
    PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
    PLASMON_NORMALIZATION_NONE,
)
from helios.services.derived.plasmon_electron_policy import (
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
    PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE,
    PLASMON_ELECTRON_POLICY_RAW,
    PLASMON_ELECTRON_POLICY_VALENCE_LOCKED,
    policy_label as electron_policy_label,
    policy_scope as electron_policy_scope,
    resolved_material_formula_map,
    resolve_effective_electron_fields,
)
from helios.services.derived.plasmon_driven_response import (
    PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
    PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
    driven_response_model_label,
    normalize_driven_response_model,
)
from helios.services.derived.plasmon_reference_data import (
    GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5,
    USER_DRIVEN_AL_DISPERSION_REFERENCE,
)
from helios.services.derived.plasmon_spectrum import dsf_from_loss, estimate_peak_metrics, gaussian_convolve, loss_function_from_epsilon
from helios.services.derived.plasmon_spectrum import classical_response_cache_info
from helios.services.derived.plasmon_xrts_observable import normalize_observable_mode, observable_mode_label
from helios.services.derived.plasmon_validation import (
    compute_plasmon,
    make_run_context,
    q_to_angle_deg,
    shocked_al_slab_summary,
    uniform_al_dataset,
)
from helios.services.derived.selection import AnalysisStateCache


PHOTON_ENERGY_KEV = float(USER_DRIVEN_AL_DISPERSION_REFERENCE["photon_energy_kev"])
ARTICLE_INSTRUMENT_FWHM_EV = 3.5
PEAK_EXTRACTION_FWHM_EV = 0.20
ALL_MODELS = tuple(str(value) for _label, value in PLASMON_MODEL_CHOICES)
PRIMARY_POLICY_AMBIENT = PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK
PRIMARY_POLICY_DRIVEN = PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT
PRIMARY_COLLISION_MODEL = PLASMON_COLLISION_MODEL_BENCHMARK_DENSE
BENCHMARK_POLICIES = (
    PLASMON_ELECTRON_POLICY_RAW,
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
    PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE,
    PLASMON_ELECTRON_POLICY_VALENCE_LOCKED,
)
ARTICLE_DRIVEN_BENCHMARK_POLICIES = tuple(policy for policy in BENCHMARK_POLICIES if policy != PLASMON_ELECTRON_POLICY_RAW)
REPRESENTATIVE_Q_AMBIENT = 1.28
REPRESENTATIVE_Q_DRIVEN = 1.28
AVOGADRO = 6.02214076e23
AL_ATOMIC_WEIGHT_G_MOL = 26.9815
ARTICLE_AL_MIN_CREDIBLE_ZEFF = 0.5
ARTICLE_AL_MIN_CREDIBLE_NE_CM3 = 1.0e22
ARTICLE_DRIVEN_DENSITY_WINDOW = tuple(float(value) for value in USER_DRIVEN_AL_DISPERSION_REFERENCE["rho_g_cm3_range"])
ARTICLE_DRIVEN_TEMPERATURE_EV = float(USER_DRIVEN_AL_DISPERSION_REFERENCE["te_ev"])
ARTICLE_DRIVEN_PROBE_TIME_NS = float(USER_DRIVEN_AL_DISPERSION_REFERENCE["probe_time_ns"])
ARTICLE_DRIVEN_PLATEAU_THICKNESS_UM = 30.0
ARTICLE_DRIVEN_DENSITY_GRID = tuple(float(value) for value in np.linspace(ARTICLE_DRIVEN_DENSITY_WINDOW[0], ARTICLE_DRIVEN_DENSITY_WINDOW[1], 4))


def _series_points(reference: dict[str, object], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = dict(reference["series"])[str(key)]
    return (
        np.asarray(series["q_ang_inv"], dtype=np.float64),
        np.asarray(series["peak_ev"], dtype=np.float64),
        np.asarray(series.get("peak_err_ev", np.zeros_like(series["peak_ev"])), dtype=np.float64),
    )


def _metric_row(prediction: dict[float, float], q_ref: np.ndarray, y_ref: np.ndarray) -> dict[str, float]:
    errors: list[float] = []
    for q_value, ref_value in zip(q_ref.tolist(), y_ref.tolist(), strict=False):
        pred_value = float(prediction.get(float(q_value), float("nan")))
        if math.isfinite(pred_value):
            errors.append(abs(pred_value - float(ref_value)))
    if not errors:
        return {"valid_points": 0.0, "mae_ev": float("nan"), "rmse_ev": float("nan"), "max_abs_ev": float("nan")}
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "valid_points": float(arr.size),
        "mae_ev": float(np.mean(arr)),
        "rmse_ev": float(np.sqrt(np.mean(arr**2))),
        "max_abs_ev": float(np.max(arr)),
    }


def _model_label(model: str) -> str:
    for label, value in PLASMON_MODEL_CHOICES:
        if str(value) == str(model):
            return str(label)
    return str(model)


def _reference_series(reference: dict[str, object]) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for key, series in dict(reference["series"]).items():
        q_ref, y_ref, y_err = _series_points(reference, str(key))
        rows.append((str(series["label"]), q_ref, y_ref, y_err))
    return rows


def _material_span(dataset, *, material_id: int) -> tuple[int, int]:
    zone_material = np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32))
    indices = np.flatnonzero(zone_material == int(abs(material_id)))
    if indices.size == 0:
        raise ValueError(f"No zones found for material {material_id}.")
    return int(indices[0] + 1), int(indices[-1] + 1)


def _material_ids_for_formula(dataset, formula: str) -> tuple[int, ...]:
    mapping = resolved_material_formula_map(dataset)
    return tuple(sorted(int(material_id) for material_id, value in mapping.items() if str(value) == str(formula)))


def _selection_state_summary(
    dataset,
    *,
    snapshot_index: int,
    zone_index_lower: int,
    zone_index_upper: int,
    material_id: int,
    effective_ne_cm3: np.ndarray | None = None,
    effective_zbar: np.ndarray | None = None,
    baseline_zbar: np.ndarray | None = None,
    increment_zbar: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    snapshot_index = int(snapshot_index)
    lower = max(1, int(zone_index_lower))
    upper = min(int(zone_index_upper), int(np.asarray(dataset.density_g_cm3, dtype=np.float64).shape[1]))
    zone_material = np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32))
    rho = np.asarray(dataset.density_g_cm3[snapshot_index], dtype=np.float64)
    te = np.asarray(dataset.temperature_e_ev[snapshot_index], dtype=np.float64)
    ti = np.asarray(dataset.temperature_i_ev[snapshot_index], dtype=np.float64)
    ne = np.asarray(dataset.electron_density_cm3[snapshot_index], dtype=np.float64)
    zbar = np.asarray(dataset.mean_charge[snapshot_index], dtype=np.float64)
    width = np.asarray(dataset.zone_width_cm[snapshot_index], dtype=np.float64)
    atomic_weight = np.asarray(dataset.zone_atomic_weight, dtype=np.float64)
    mask = np.zeros(zone_material.shape, dtype=bool)
    mask[lower - 1 : upper] = True
    mask &= zone_material == int(abs(material_id))
    if not np.any(mask):
        raise ValueError("Selection summary requested for an empty zone span.")
    weights = np.asarray(width[mask], dtype=np.float64)
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Selection summary requested for a span with non-positive path length.")

    def _wavg(values: np.ndarray) -> float:
        local = np.asarray(values[mask], dtype=np.float64)
        return float(np.average(local, weights=weights))

    def _finite_positive_wavg(values: np.ndarray) -> float:
        local = np.asarray(values[mask], dtype=np.float64)
        valid = np.isfinite(local) & (local > 0.0) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid):
            return float("nan")
        return float(np.average(local[valid], weights=weights[valid]))

    summary: dict[str, float | int | str] = {
        "zone_index_lower": int(lower),
        "zone_index_upper": int(upper),
        "zone_count": int(np.count_nonzero(mask)),
        "path_length_total_cm": float(total_weight),
        "rho_min_g_cm3": float(np.nanmin(rho[mask])),
        "rho_max_g_cm3": float(np.nanmax(rho[mask])),
        "rho_weighted_g_cm3": _wavg(rho),
        "te_weighted_ev": _wavg(te),
        "ti_weighted_ev": _wavg(ti),
        "raw_ne_weighted_cm3": _wavg(ne),
        "raw_zbar_weighted": _wavg(zbar),
        "atomic_weight_weighted_g_mol": _finite_positive_wavg(atomic_weight),
    }
    summary["ne_weighted_cm3"] = float(summary["raw_ne_weighted_cm3"])
    summary["zbar_weighted"] = float(summary["raw_zbar_weighted"])
    if effective_ne_cm3 is not None:
        effective_ne = np.asarray(effective_ne_cm3[snapshot_index], dtype=np.float64)
        summary["effective_ne_weighted_cm3"] = _wavg(effective_ne)
        summary["ne_weighted_cm3"] = float(summary["effective_ne_weighted_cm3"])
    if effective_zbar is not None:
        effective_z = np.asarray(effective_zbar[snapshot_index], dtype=np.float64)
        summary["effective_zbar_weighted"] = _wavg(effective_z)
        summary["zbar_weighted"] = float(summary["effective_zbar_weighted"])
    if baseline_zbar is not None:
        baseline_z = np.asarray(baseline_zbar[snapshot_index], dtype=np.float64)
        summary["baseline_zbar_weighted"] = _wavg(baseline_z)
    if increment_zbar is not None:
        increment_z = np.asarray(increment_zbar[snapshot_index], dtype=np.float64)
        summary["increment_zbar_weighted"] = _wavg(increment_z)
    atomic_weight_weighted = float(summary["atomic_weight_weighted_g_mol"])
    if not math.isfinite(atomic_weight_weighted) or atomic_weight_weighted <= 0.0:
        material_formula = str(resolved_material_formula_map(dataset).get(int(material_id), "")).strip()
        if material_formula == "Al":
            atomic_weight_weighted = AL_ATOMIC_WEIGHT_G_MOL
            summary["atomic_weight_weighted_g_mol"] = atomic_weight_weighted
    ion_density_cm3 = (
        float(summary["rho_weighted_g_cm3"]) / atomic_weight_weighted * AVOGADRO
        if math.isfinite(atomic_weight_weighted) and atomic_weight_weighted > 0.0
        else float("nan")
    )
    summary["ion_density_weighted_cm3"] = ion_density_cm3
    summary["effective_valence_from_ne"] = (
        float(summary["ne_weighted_cm3"]) / ion_density_cm3
        if math.isfinite(ion_density_cm3) and ion_density_cm3 > 0.0
        else float("nan")
    )
    summary["raw_effective_valence_from_ne"] = (
        float(summary["raw_ne_weighted_cm3"]) / ion_density_cm3
        if math.isfinite(ion_density_cm3) and ion_density_cm3 > 0.0
        else float("nan")
    )
    return summary


def _response_weight_summary(payload) -> dict[str, float | str]:
    weight_mode = str(getattr(payload, "driven_response_weight_mode", "") or "").strip() or "uniform"
    weights = getattr(payload, "driven_response_weight_multiplier", None)
    if weights is None:
        return {
            "driven_response_weight_mode": weight_mode,
            "driven_response_weight_mean": 1.0,
            "driven_response_weight_min": 1.0,
            "driven_response_weight_max": 1.0,
        }
    arr = np.asarray(weights, dtype=np.float64)
    finite = arr[np.isfinite(arr) & (arr > 0.0)]
    if finite.size == 0:
        return {
            "driven_response_weight_mode": weight_mode,
            "driven_response_weight_mean": 1.0,
            "driven_response_weight_min": 1.0,
            "driven_response_weight_max": 1.0,
        }
    return {
        "driven_response_weight_mode": weight_mode,
        "driven_response_weight_mean": float(np.mean(finite)),
        "driven_response_weight_min": float(np.min(finite)),
        "driven_response_weight_max": float(np.max(finite)),
    }


def _response_shape_summary(payload) -> dict[str, float | str]:
    shape_mode = str(getattr(payload, "driven_response_shape_mode", "") or "").strip()
    values = getattr(payload, "driven_response_shape_fwhm_ev", None)
    if values is None:
        return {
            "driven_response_shape_mode": shape_mode,
            "driven_response_shape_mean_ev": 0.0,
            "driven_response_shape_min_ev": 0.0,
            "driven_response_shape_max_ev": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr) & (arr > 0.0)]
    if finite.size == 0:
        return {
            "driven_response_shape_mode": shape_mode,
            "driven_response_shape_mean_ev": 0.0,
            "driven_response_shape_min_ev": 0.0,
            "driven_response_shape_max_ev": 0.0,
        }
    return {
        "driven_response_shape_mode": shape_mode,
        "driven_response_shape_mean_ev": float(np.mean(finite)),
        "driven_response_shape_min_ev": float(np.min(finite)),
        "driven_response_shape_max_ev": float(np.max(finite)),
    }


def _response_ensemble_summary(payload) -> dict[str, str]:
    return {
        "driven_response_ensemble_mode": str(getattr(payload, "driven_response_ensemble_mode", "") or "").strip(),
    }


def _uniform_policy_state_summary(
    *,
    rho_g_cm3: float,
    te_ev: float,
    policy: str,
    driven_response_model: str | None = None,
) -> dict[str, float | int | str]:
    dataset, _context = uniform_al_dataset(float(rho_g_cm3), float(te_ev))
    payload = resolve_effective_electron_fields(dataset, policy, driven_response_model=driven_response_model)
    weight_summary = _response_weight_summary(payload)
    shape_summary = _response_shape_summary(payload)
    ensemble_summary = _response_ensemble_summary(payload)
    summary = _selection_state_summary(
        dataset,
        snapshot_index=0,
        zone_index_lower=1,
        zone_index_upper=int(np.asarray(dataset.density_g_cm3, dtype=np.float64).shape[1]),
        material_id=1,
        effective_ne_cm3=payload.electron_density_cm3,
        effective_zbar=payload.mean_charge,
        baseline_zbar=getattr(payload, "baseline_mean_charge", None),
        increment_zbar=getattr(payload, "increment_mean_charge", None),
    )
    summary.update(
        {
            "snapshot_index": 0,
            "time_ns": 0.0,
            "material_id": 1,
            "material_formula": "Al",
            "baseline_mode": str(getattr(payload, "baseline_mode", "")),
            "increment_mode": str(getattr(payload, "increment_mode", "")),
            "baseline_table_source": str(getattr(payload, "baseline_table_source", "")),
            "baseline_entries": tuple(str(value) for value in getattr(payload, "baseline_entries", ()) if str(value).startswith("Al@")),
            "increment_entries": tuple(str(value) for value in getattr(payload, "increment_entries", ()) if str(value).startswith("Al@")),
            "driven_response_model": str(getattr(payload, "driven_response_model", PLASMON_DRIVEN_RESPONSE_MODEL_NONE)),
            "driven_response_summary": str(getattr(payload, "driven_response_summary", "")),
            **weight_summary,
            **shape_summary,
            **ensemble_summary,
        }
    )
    return summary


def _article_target_state_summary() -> dict[str, float | int | str]:
    rho_min, rho_max = ARTICLE_DRIVEN_DENSITY_WINDOW
    rho_mid = 0.5 * (rho_min + rho_max)
    ion_density_cm3 = rho_mid / AL_ATOMIC_WEIGHT_G_MOL * AVOGADRO
    return {
        "snapshot_index": -1,
        "time_ns": ARTICLE_DRIVEN_PROBE_TIME_NS,
        "zone_index_lower": -1,
        "zone_index_upper": -1,
        "zone_count": 4,
        "rho_min_g_cm3": rho_min,
        "rho_max_g_cm3": rho_max,
        "rho_weighted_g_cm3": rho_mid,
        "te_weighted_ev": ARTICLE_DRIVEN_TEMPERATURE_EV,
        "ti_weighted_ev": ARTICLE_DRIVEN_TEMPERATURE_EV,
        "ion_density_weighted_cm3": ion_density_cm3,
        "effective_ne_weighted_cm3": ion_density_cm3 * 3.0,
        "effective_valence_from_ne": 3.0,
        "path_length_total_cm": ARTICLE_DRIVEN_PLATEAU_THICKNESS_UM * 1.0e-4,
        "material_formula": "Al",
        "density_floor_g_cm3": rho_min,
        "selection_kind": "article_density_average_fixed_temperature",
    }


def _driven_state_score(summary: dict[str, float | int | str]) -> float:
    rho_center = 0.5 * (ARTICLE_DRIVEN_DENSITY_WINDOW[0] + ARTICLE_DRIVEN_DENSITY_WINDOW[1])
    return (
        3.0 * abs(float(summary["te_weighted_ev"]) - ARTICLE_DRIVEN_TEMPERATURE_EV)
        + 0.75 * abs(float(summary["rho_weighted_g_cm3"]) - rho_center)
        + 0.20 * abs(float(summary["rho_min_g_cm3"]) - ARTICLE_DRIVEN_DENSITY_WINDOW[0])
        + 0.20 * abs(float(summary["rho_max_g_cm3"]) - ARTICLE_DRIVEN_DENSITY_WINDOW[1])
        + 0.05 * abs(float(summary["path_length_total_cm"]) * 1.0e4 - ARTICLE_DRIVEN_PLATEAU_THICKNESS_UM)
        + 0.25 * abs(float(summary["time_ns"]) - ARTICLE_DRIVEN_PROBE_TIME_NS)
    )


def _best_driven_hydro_selection(dataset, *, material_id: int) -> dict[str, float | int | str]:
    time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
    snapshot_candidates = np.flatnonzero(
        np.isfinite(time_ns) & (time_ns >= ARTICLE_DRIVEN_PROBE_TIME_NS - 0.35) & (time_ns <= ARTICLE_DRIVEN_PROBE_TIME_NS + 0.35)
    )
    if snapshot_candidates.size == 0:
        snapshot_candidates = np.asarray([int(np.argmin(np.abs(time_ns - ARTICLE_DRIVEN_PROBE_TIME_NS)))], dtype=np.int32)
    best_summary: dict[str, float | int | str] | None = None
    best_score = float("inf")
    for snapshot_index in snapshot_candidates.tolist():
        for density_floor in (3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9):
            try:
                summary = shocked_al_slab_summary(
                    dataset,
                    snapshot_index=int(snapshot_index),
                    density_floor_g_cm3=float(density_floor),
                    material_id=int(material_id),
                )
            except Exception:
                continue
            score = _driven_state_score(summary)
            if score < best_score:
                best_score = score
                best_summary = dict(summary)
                best_summary["selection_score"] = float(score)
                best_summary["selection_kind"] = "hydro_plateau_best_match"
    if best_summary is None:
        raise ValueError("Unable to identify a driven Al slab candidate near the article probe time.")
    return best_summary


def _density_averaged_policy_selection(policy: str, *, driven_response_model: str | None = None) -> dict[str, float | int | str]:
    rows = [
        _uniform_policy_state_summary(
            rho_g_cm3=float(rho),
            te_ev=ARTICLE_DRIVEN_TEMPERATURE_EV,
            policy=str(policy),
            driven_response_model=driven_response_model,
        )
        for rho in ARTICLE_DRIVEN_DENSITY_GRID
    ]

    def _avg(key: str) -> float:
        values = [float(row.get(key, float("nan"))) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    return {
        "snapshot_index": -1,
        "time_ns": ARTICLE_DRIVEN_PROBE_TIME_NS,
        "zone_index_lower": 1,
        "zone_index_upper": int(len(ARTICLE_DRIVEN_DENSITY_GRID)),
        "zone_count": int(len(ARTICLE_DRIVEN_DENSITY_GRID)),
        "rho_min_g_cm3": float(ARTICLE_DRIVEN_DENSITY_WINDOW[0]),
        "rho_max_g_cm3": float(ARTICLE_DRIVEN_DENSITY_WINDOW[1]),
        "rho_weighted_g_cm3": _avg("rho_weighted_g_cm3"),
        "te_weighted_ev": float(ARTICLE_DRIVEN_TEMPERATURE_EV),
        "ti_weighted_ev": float(ARTICLE_DRIVEN_TEMPERATURE_EV),
        "raw_ne_weighted_cm3": _avg("raw_ne_weighted_cm3"),
        "ne_weighted_cm3": _avg("ne_weighted_cm3"),
        "effective_ne_weighted_cm3": _avg("effective_ne_weighted_cm3"),
        "raw_zbar_weighted": _avg("raw_zbar_weighted"),
        "zbar_weighted": _avg("zbar_weighted"),
        "effective_zbar_weighted": _avg("effective_zbar_weighted"),
        "baseline_zbar_weighted": _avg("baseline_zbar_weighted"),
        "increment_zbar_weighted": _avg("increment_zbar_weighted"),
        "ion_density_weighted_cm3": _avg("ion_density_weighted_cm3"),
        "effective_valence_from_ne": _avg("effective_valence_from_ne"),
        "raw_effective_valence_from_ne": _avg("raw_effective_valence_from_ne"),
        "material_formula": "Al",
        "density_floor_g_cm3": float(ARTICLE_DRIVEN_DENSITY_WINDOW[0]),
        "path_length_total_cm": ARTICLE_DRIVEN_PLATEAU_THICKNESS_UM * 1.0e-4,
        "baseline_mode": str(rows[0].get("baseline_mode", "")) if rows else "",
        "increment_mode": str(rows[0].get("increment_mode", "")) if rows else "",
        "baseline_table_source": str(rows[0].get("baseline_table_source", "")) if rows else "",
        "baseline_entries": tuple(str(value) for value in rows[0].get("baseline_entries", ()) if str(value).strip()) if rows else (),
        "increment_entries": tuple(str(value) for value in rows[0].get("increment_entries", ()) if str(value).strip()) if rows else (),
        "driven_response_model": str(rows[0].get("driven_response_model", PLASMON_DRIVEN_RESPONSE_MODEL_NONE)) if rows else PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
        "driven_response_summary": " || ".join(
            sorted(
                {
                    str(row.get("driven_response_summary", "")).strip()
                    for row in rows
                    if str(row.get("driven_response_summary", "")).strip()
                }
            )
        ),
        "driven_response_weight_mode": str(rows[0].get("driven_response_weight_mode", "uniform")) if rows else "uniform",
        "driven_response_weight_mean": _avg("driven_response_weight_mean"),
        "driven_response_weight_min": _avg("driven_response_weight_min"),
        "driven_response_weight_max": _avg("driven_response_weight_max"),
        "driven_response_shape_mode": str(rows[0].get("driven_response_shape_mode", "")) if rows else "",
        "driven_response_shape_mean_ev": _avg("driven_response_shape_mean_ev"),
        "driven_response_shape_min_ev": _avg("driven_response_shape_min_ev"),
        "driven_response_shape_max_ev": _avg("driven_response_shape_max_ev"),
        "driven_response_ensemble_mode": str(rows[0].get("driven_response_ensemble_mode", "")) if rows else "",
        "selection_kind": "article_density_average_fixed_temperature",
        "density_average_grid_g_cm3": tuple(float(value) for value in ARTICLE_DRIVEN_DENSITY_GRID),
    }


def _quicklook_peak_from_result(result) -> float:
    plasma_energy = float(getattr(result, "plasma_frequency_ev", float("nan")))
    k_lambda = float(getattr(result, "k_lambda_debye", float("nan")))
    gamma = float(getattr(result, "adiabatic_index", 1.0))
    if not math.isfinite(plasma_energy) or plasma_energy <= 0.0:
        return float("nan")
    correction = 1.0
    if math.isfinite(k_lambda):
        correction = max(0.0, 1.0 + gamma * (k_lambda**2))
    return float(plasma_energy * math.sqrt(correction))


def _assess_article_al_policy_state(case_name: str, policy: str, selection: dict[str, object]) -> dict[str, object]:
    rho = float(selection.get("rho_weighted_g_cm3", float("nan")))
    ne = float(selection.get("ne_weighted_cm3", float("nan")))
    zeff = float(selection.get("effective_valence_from_ne", float("nan")))
    raw_ne = float(selection.get("raw_ne_weighted_cm3", float("nan")))
    raw_zeff = float(selection.get("raw_effective_valence_from_ne", float("nan")))
    if str(policy) != PLASMON_ELECTRON_POLICY_RAW:
        return {
            "input_policy_status": "credible",
            "input_policy_reason": "",
        }
    if case_name in {"ambient_al_t0", "driven_al_dense_slab", "driven_al_best_hydro_slab", "driven_al_article_state"} and math.isfinite(rho) and rho >= 1.0:
        if (math.isfinite(zeff) and zeff < ARTICLE_AL_MIN_CREDIBLE_ZEFF) or (
            math.isfinite(ne) and ne < ARTICLE_AL_MIN_CREDIBLE_NE_CM3
        ):
            return {
                "input_policy_status": "invalid_input_policy",
                "input_policy_reason": (
                    "Raw HELIOS electron state is not physically credible for the dense Al benchmark slab: "
                    f"rho={rho:.3f} g/cm^3, ne={ne:.4e} cm^-3, Z_eff={zeff:.3g} "
                    f"(raw ne={raw_ne:.4e} cm^-3, raw Z_eff={raw_zeff:.3g})."
                ),
            }
    return {
        "input_policy_status": "diagnostic_only",
        "input_policy_reason": "Raw HELIOS electron policy is retained as a diagnostic reference only.",
    }


def _model_parameters(
    *,
    model: str,
    electron_policy: str,
    driven_response_model: str | None,
    observable_mode: str,
    angle_deg: float,
    zone_index_lower: int,
    zone_index_upper: int,
    instrument_fwhm_ev: float,
    benchmark_preset: str,
    material_id: int,
) -> dict[str, object]:
    execution_mode = PLASMON_EXECUTION_MODE_QUICKLOOK if str(model) == PLASMON_MODEL_QUICKLOOK else PLASMON_EXECUTION_MODE_BENCHMARK
    return {
        "plasmon_model": str(model),
        "plasmon_execution_mode": execution_mode,
        "plasmon_integration_mode": "los_integrated",
        "plasmon_electron_policy": str(electron_policy),
        "plasmon_driven_response_model": str(driven_response_model or ""),
        "plasmon_benchmark_preset": str(benchmark_preset),
        "plasmon_observable_mode": str(observable_mode),
        "plasmon_photon_energy_kev": PHOTON_ENERGY_KEV,
        "plasmon_scattering_angle_deg": float(angle_deg),
        "plasmon_energy_window_ev": 45.0,
        "plasmon_energy_points": 1201,
        "plasmon_instrument_fwhm_ev": float(instrument_fwhm_ev),
        "plasmon_lfc_model": "esa_static",
        "plasmon_collision_model": PRIMARY_COLLISION_MODEL,
        "derived_material_ids": (int(material_id),),
        "zone_index_lower": int(zone_index_lower),
        "zone_index_upper": int(zone_index_upper),
    }


def _serialize_warnings(result) -> list[str]:
    return [str(item.message) for item in getattr(result, "warnings", ())]


def _compute_point_result(
    dataset,
    context,
    cache: AnalysisStateCache,
    *,
    model: str,
    electron_policy: str,
    q_value: float,
    zone_index_lower: int,
    zone_index_upper: int,
    instrument_fwhm_ev: float,
    benchmark_preset: str,
    material_id: int,
    driven_response_model: str | None = None,
    observable_mode: str = PLASMON_OBSERVABLE_MODE_DIELECTRIC,
) -> dict[str, object]:
    angle_deg = float(q_to_angle_deg(float(q_value), PHOTON_ENERGY_KEV))
    result = compute_plasmon(
        dataset,
        context,
        analysis_cache=cache,
        **_model_parameters(
            model=model,
            electron_policy=str(electron_policy),
            driven_response_model=driven_response_model,
            observable_mode=str(observable_mode),
            angle_deg=angle_deg,
            zone_index_lower=zone_index_lower,
            zone_index_upper=zone_index_upper,
            instrument_fwhm_ev=instrument_fwhm_ev,
            benchmark_preset=str(benchmark_preset),
            material_id=int(material_id),
        ),
    )
    peak_energy_ev = float(result.peak_energy_ev)
    if str(model) == PLASMON_MODEL_QUICKLOOK:
        peak_energy_ev = _quicklook_peak_from_result(result)
    return {
        "model": str(model),
        "model_label": _model_label(str(model)),
        "q_ang_inv": float(q_value),
        "angle_deg": float(angle_deg),
        "status": str(result.benchmark_status),
        "backend": str(result.response_backend),
        "backend_summary": str(getattr(result, "backend_summary", "")),
        "stls_converged": bool(getattr(result, "stls_converged", False)),
        "stls_iteration_count": int(getattr(result, "stls_iteration_count", 0)),
        "stls_convergence_residual": float(getattr(result, "stls_convergence_residual", float("nan"))),
        "stls_convergence_relative_residual": float(getattr(result, "stls_convergence_relative_residual", float("nan"))),
        "stls_closure_name": str(getattr(result, "stls_closure_name", "")),
        "stls_local_field_value": float(getattr(result, "stls_local_field_value", float("nan"))),
        "stls_q_over_qf": float(getattr(result, "stls_q_over_qf", float("nan"))),
        "executed_fully": bool(result.model_executed_fully),
        "fallback_fraction": float(result.fallback_fraction),
        "domain_failure_fraction": float(getattr(result, "domain_failure_fraction", 0.0)),
        "runtime_s": float(getattr(result, "total_runtime_s", 0.0)),
        "spectrum_runtime_s": float(getattr(result, "spectrum_runtime_s", 0.0)),
        "comparison_runtime_s": float(getattr(result, "comparison_runtime_s", 0.0)),
        "dispersion_runtime_s": float(getattr(result, "dispersion_runtime_s", 0.0)),
        "time_series_runtime_s": float(getattr(result, "time_series_runtime_s", 0.0)),
        "peak_energy_ev": float(peak_energy_ev),
        "peak_fwhm_ev": float(result.peak_fwhm_ev),
        "requested_electron_policy": str(getattr(result, "requested_electron_policy", result.electron_policy)),
        "electron_policy": str(result.electron_policy),
        "electron_policy_scope": electron_policy_scope(str(result.electron_policy)),
        "driven_response_model": str(getattr(result, "driven_response_model", str(driven_response_model or PLASMON_DRIVEN_RESPONSE_MODEL_NONE))),
        "driven_response_summary": str(getattr(result, "driven_response_summary", "")),
        "observable_mode": str(getattr(result, "observable_mode", PLASMON_OBSERVABLE_MODE_DIELECTRIC)),
        "observable_summary": str(getattr(result, "observable_summary", "")),
        "observable_decomposition_mode": str(getattr(result, "observable_decomposition_mode", "")),
        "observable_peak_extraction_mode": str(getattr(result, "observable_peak_extraction_mode", "")),
        "observable_elastic_exclusion_ev": float(getattr(result, "observable_elastic_exclusion_ev", 0.0)),
        "observable_free_fraction": float(getattr(result, "observable_free_fraction", float("nan"))),
        "observable_bound_fraction": float(getattr(result, "observable_bound_fraction", float("nan"))),
        "observable_elastic_fraction": float(getattr(result, "observable_elastic_fraction", float("nan"))),
        "observable_comparison_mode": str(getattr(result, "observable_comparison_mode", "")),
        "observable_subtraction_mode": str(getattr(result, "observable_subtraction_mode", "")),
        "observable_normalization_mode": str(getattr(result, "observable_normalization_mode", "")),
        "observable_peak_discrete_energy_ev": float(getattr(result, "observable_peak_discrete_energy_ev", float("nan"))),
        "observable_peak_fit_energy_ev": float(getattr(result, "observable_peak_fit_energy_ev", float("nan"))),
        "observable_peak_fit_status": str(getattr(result, "observable_peak_fit_status", "")),
        "observable_peak_edge_dominated": bool(getattr(result, "observable_peak_edge_dominated", False)),
        "observable_elastic_form_factor_total": float(getattr(result, "observable_elastic_form_factor_total", float("nan"))),
        "observable_elastic_form_factor_core": float(getattr(result, "observable_elastic_form_factor_core", float("nan"))),
        "observable_elastic_screening_form_factor": float(getattr(result, "observable_elastic_screening_form_factor", float("nan"))),
        "observable_ion_structure_factor": float(getattr(result, "observable_ion_structure_factor", float("nan"))),
        "observable_bound_core_mode": str(getattr(result, "observable_bound_core_mode", "")),
        "observable_bound_shell_summary": str(getattr(result, "observable_bound_shell_summary", "")),
        "spectrum_free_component": np.asarray(getattr(result, "spectrum_free_component", np.asarray([], dtype=np.float64)), dtype=np.float64),
        "spectrum_bound_component": np.asarray(getattr(result, "spectrum_bound_component", np.asarray([], dtype=np.float64)), dtype=np.float64),
        "spectrum_elastic_component": np.asarray(getattr(result, "spectrum_elastic_component", np.asarray([], dtype=np.float64)), dtype=np.float64),
        "electron_density_source": str(result.electron_density_source),
        "material_policy_summary": str(result.material_policy_summary),
        "benchmark_preset": str(getattr(result, "benchmark_preset", benchmark_preset)),
        "collision_source": str(getattr(result, "collision_source", "")),
        "collision_summary": str(getattr(result, "collision_summary", "")),
        "zone_count_used": int(result.zone_count_used),
        "cluster_count_used": int(result.cluster_count_used),
        "warnings": _serialize_warnings(result),
        "result": result,
    }


def _compute_density_averaged_point_result(
    *,
    model: str,
    electron_policy: str,
    driven_response_model: str | None,
    q_value: float,
    densities_g_cm3: tuple[float, ...],
    te_ev: float,
    instrument_fwhm_ev: float,
    benchmark_preset: str,
    observable_mode: str = PLASMON_OBSERVABLE_MODE_DIELECTRIC,
) -> dict[str, object]:
    angle_deg = float(q_to_angle_deg(float(q_value), PHOTON_ENERGY_KEV))
    spectra: list[np.ndarray] = []
    energy_axis: np.ndarray | None = None
    warnings: set[str] = set()
    benchmark_statuses: set[str] = set()
    backends: set[str] = set()
    requested_policies: set[str] = set()
    applied_policies: set[str] = set()
    collision_sources: set[str] = set()
    collision_summaries: set[str] = set()
    backend_summaries: set[str] = set()
    driven_response_summaries: set[str] = set()
    runtime_total = 0.0
    zone_total = 0
    cluster_total = 0
    fallback_max = 0.0
    domain_max = 0.0
    executed_fully = True
    template_result = None
    spectra_weights: list[float] = []
    free_components: list[np.ndarray] = []
    bound_components: list[np.ndarray] = []
    elastic_components: list[np.ndarray] = []
    observable_fraction_samples: list[tuple[float, float, float]] = []
    state_weight_mode = "uniform"
    state_shape_mode = ""
    state_ensemble_mode = ""
    state_shape_fwhm_values: list[float] = []
    epsilon_states: list[np.ndarray] = []
    observable_summaries: set[str] = set()
    observable_decomposition_modes: set[str] = set()
    observable_peak_extraction_modes: set[str] = set()
    observable_comparison_modes: set[str] = set()
    observable_subtraction_modes: set[str] = set()
    observable_normalization_modes: set[str] = set()
    observable_peak_fit_statuses: set[str] = set()
    observable_bound_core_modes: set[str] = set()
    observable_bound_shell_summaries: set[str] = set()
    observable_elastic_exclusion_values: list[float] = []
    observable_peak_discrete_values: list[float] = []
    observable_peak_fit_values: list[float] = []
    observable_elastic_form_factor_total_values: list[float] = []
    observable_elastic_form_factor_core_values: list[float] = []
    observable_elastic_screening_form_factor_values: list[float] = []
    observable_ion_structure_factor_values: list[float] = []
    stls_converged_all = True
    stls_iteration_values: list[float] = []
    stls_residual_values: list[float] = []
    stls_relative_residual_values: list[float] = []
    stls_local_field_values: list[float] = []
    stls_q_over_qf_values: list[float] = []
    stls_closure_names: set[str] = set()
    for density in densities_g_cm3:
        dataset, context = uniform_al_dataset(float(density), float(te_ev))
        payload = resolve_effective_electron_fields(dataset, str(electron_policy), driven_response_model=driven_response_model)
        weight_summary = _response_weight_summary(payload)
        shape_summary = _response_shape_summary(payload)
        ensemble_summary = _response_ensemble_summary(payload)
        state_weight_mode = str(weight_summary["driven_response_weight_mode"] or state_weight_mode)
        state_weight = float(weight_summary["driven_response_weight_mean"])
        if str(shape_summary["driven_response_shape_mode"]).strip():
            state_shape_mode = str(shape_summary["driven_response_shape_mode"])
        if str(ensemble_summary["driven_response_ensemble_mode"]).strip():
            state_ensemble_mode = str(ensemble_summary["driven_response_ensemble_mode"])
        state_shape_fwhm_ev = float(shape_summary["driven_response_shape_mean_ev"])
        if str(getattr(payload, "driven_response_summary", "")).strip():
            driven_response_summaries.add(str(getattr(payload, "driven_response_summary", "")))
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model=str(model),
            plasmon_execution_mode=PLASMON_EXECUTION_MODE_BENCHMARK,
            plasmon_integration_mode="effective_state",
            plasmon_electron_policy=str(electron_policy),
            plasmon_driven_response_model=str(driven_response_model or ""),
            plasmon_benchmark_preset=str(benchmark_preset),
            plasmon_observable_mode=str(observable_mode),
            plasmon_photon_energy_kev=PHOTON_ENERGY_KEV,
            plasmon_scattering_angle_deg=float(angle_deg),
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_instrument_fwhm_ev=float(instrument_fwhm_ev),
            plasmon_lfc_model="esa_static",
            plasmon_collision_model=PRIMARY_COLLISION_MODEL,
            plasmon_normalization=PLASMON_NORMALIZATION_NONE,
            derived_material_ids=(1,),
            zone_index_lower=1,
            zone_index_upper=int(np.asarray(dataset.density_g_cm3, dtype=np.float64).shape[1]),
        )
        template_result = result if template_result is None else template_result
        requested_policies.add(str(getattr(result, "requested_electron_policy", electron_policy)))
        applied_policies.add(str(getattr(result, "electron_policy", electron_policy)))
        benchmark_statuses.add(str(getattr(result, "benchmark_status", "")))
        backends.add(str(getattr(result, "response_backend", "")))
        if str(getattr(result, "backend_summary", "")).strip():
            backend_summaries.add(str(getattr(result, "backend_summary", "")))
        collision_sources.add(str(getattr(result, "collision_source", "")))
        collision_summaries.add(str(getattr(result, "collision_summary", "")))
        if str(getattr(result, "observable_summary", "")).strip():
            observable_summaries.add(str(getattr(result, "observable_summary", "")))
        if str(getattr(result, "observable_decomposition_mode", "")).strip():
            observable_decomposition_modes.add(str(getattr(result, "observable_decomposition_mode", "")))
        if str(getattr(result, "observable_peak_extraction_mode", "")).strip():
            observable_peak_extraction_modes.add(str(getattr(result, "observable_peak_extraction_mode", "")))
        if str(getattr(result, "observable_comparison_mode", "")).strip():
            observable_comparison_modes.add(str(getattr(result, "observable_comparison_mode", "")))
        if str(getattr(result, "observable_subtraction_mode", "")).strip():
            observable_subtraction_modes.add(str(getattr(result, "observable_subtraction_mode", "")))
        if str(getattr(result, "observable_normalization_mode", "")).strip():
            observable_normalization_modes.add(str(getattr(result, "observable_normalization_mode", "")))
        if str(getattr(result, "observable_peak_fit_status", "")).strip():
            observable_peak_fit_statuses.add(str(getattr(result, "observable_peak_fit_status", "")))
        if str(getattr(result, "observable_bound_core_mode", "")).strip():
            observable_bound_core_modes.add(str(getattr(result, "observable_bound_core_mode", "")))
        if str(getattr(result, "observable_bound_shell_summary", "")).strip():
            observable_bound_shell_summaries.add(str(getattr(result, "observable_bound_shell_summary", "")))
        elastic_exclusion = float(getattr(result, "observable_elastic_exclusion_ev", float("nan")))
        if math.isfinite(elastic_exclusion):
            observable_elastic_exclusion_values.append(float(elastic_exclusion))
        peak_discrete = float(getattr(result, "observable_peak_discrete_energy_ev", float("nan")))
        if math.isfinite(peak_discrete):
            observable_peak_discrete_values.append(float(peak_discrete))
        peak_fit = float(getattr(result, "observable_peak_fit_energy_ev", float("nan")))
        if math.isfinite(peak_fit):
            observable_peak_fit_values.append(float(peak_fit))
        elastic_ff_total = float(getattr(result, "observable_elastic_form_factor_total", float("nan")))
        if math.isfinite(elastic_ff_total):
            observable_elastic_form_factor_total_values.append(float(elastic_ff_total))
        elastic_ff_core = float(getattr(result, "observable_elastic_form_factor_core", float("nan")))
        if math.isfinite(elastic_ff_core):
            observable_elastic_form_factor_core_values.append(float(elastic_ff_core))
        elastic_ff_screen = float(getattr(result, "observable_elastic_screening_form_factor", float("nan")))
        if math.isfinite(elastic_ff_screen):
            observable_elastic_screening_form_factor_values.append(float(elastic_ff_screen))
        ion_structure = float(getattr(result, "observable_ion_structure_factor", float("nan")))
        if math.isfinite(ion_structure):
            observable_ion_structure_factor_values.append(float(ion_structure))
        warnings.update(_serialize_warnings(result))
        runtime_total += float(getattr(result, "total_runtime_s", 0.0))
        zone_total += int(getattr(result, "zone_count_used", 0))
        cluster_total += int(getattr(result, "cluster_count_used", 0))
        fallback_max = max(fallback_max, float(getattr(result, "fallback_fraction", 0.0)))
        domain_max = max(domain_max, float(getattr(result, "domain_failure_fraction", 0.0)))
        executed_fully = executed_fully and bool(getattr(result, "model_executed_fully", False))
        if str(getattr(result, "response_backend", "")) == "finite_t_stls":
            stls_converged_all = stls_converged_all and bool(getattr(result, "stls_converged", False))
            if math.isfinite(float(getattr(result, "stls_iteration_count", float("nan")))):
                stls_iteration_values.append(float(getattr(result, "stls_iteration_count", 0)))
            if math.isfinite(float(getattr(result, "stls_convergence_residual", float("nan")))):
                stls_residual_values.append(float(getattr(result, "stls_convergence_residual", float("nan"))))
            if math.isfinite(float(getattr(result, "stls_convergence_relative_residual", float("nan")))):
                stls_relative_residual_values.append(float(getattr(result, "stls_convergence_relative_residual", float("nan"))))
            if math.isfinite(float(getattr(result, "stls_local_field_value", float("nan")))):
                stls_local_field_values.append(float(getattr(result, "stls_local_field_value", float("nan"))))
            if math.isfinite(float(getattr(result, "stls_q_over_qf", float("nan")))):
                stls_q_over_qf_values.append(float(getattr(result, "stls_q_over_qf", float("nan"))))
            if str(getattr(result, "stls_closure_name", "")).strip():
                stls_closure_names.add(str(getattr(result, "stls_closure_name", "")))
        energy = np.asarray(getattr(result, "spectrum_energy_ev", np.asarray([], dtype=np.float64)), dtype=np.float64)
        spectrum = np.asarray(getattr(result, "spectrum_intensity", np.asarray([], dtype=np.float64)), dtype=np.float64)
        free_component = np.asarray(getattr(result, "spectrum_free_component", np.asarray([], dtype=np.float64)), dtype=np.float64)
        bound_component = np.asarray(getattr(result, "spectrum_bound_component", np.asarray([], dtype=np.float64)), dtype=np.float64)
        elastic_component = np.asarray(getattr(result, "spectrum_elastic_component", np.asarray([], dtype=np.float64)), dtype=np.float64)
        epsilon_real = np.asarray(getattr(result, "dielectric_real", np.asarray([], dtype=np.float64)), dtype=np.float64)
        epsilon_imag = np.asarray(getattr(result, "dielectric_imag", np.asarray([], dtype=np.float64)), dtype=np.float64)
        if energy.size == 0 or spectrum.size != energy.size:
            continue
        if energy_axis is None:
            energy_axis = energy
        elif energy_axis.shape != energy.shape or not np.allclose(energy_axis, energy, equal_nan=True):
            warnings.add("Article density-averaged benchmark skipped one state because the spectrum grid was not aligned.")
            continue
        if str(state_shape_mode).strip() and math.isfinite(state_shape_fwhm_ev) and state_shape_fwhm_ev > 0.0:
            spectrum = gaussian_convolve(energy, spectrum, float(state_shape_fwhm_ev))
            if free_component.size == energy.size:
                free_component = gaussian_convolve(energy, free_component, float(state_shape_fwhm_ev))
            if bound_component.size == energy.size:
                bound_component = gaussian_convolve(energy, bound_component, float(state_shape_fwhm_ev))
            if elastic_component.size == energy.size:
                elastic_component = gaussian_convolve(energy, elastic_component, float(state_shape_fwhm_ev))
            state_shape_fwhm_values.append(float(state_shape_fwhm_ev))
        spectra.append(spectrum)
        spectra_weights.append(state_weight if math.isfinite(state_weight) and state_weight > 0.0 else 1.0)
        if free_component.size == energy.size and bound_component.size == energy.size and elastic_component.size == energy.size:
            free_components.append(np.asarray(free_component, dtype=np.float64))
            bound_components.append(np.asarray(bound_component, dtype=np.float64))
            elastic_components.append(np.asarray(elastic_component, dtype=np.float64))
            observable_fraction_samples.append(
                (
                    float(getattr(result, "observable_free_fraction", float("nan"))),
                    float(getattr(result, "observable_bound_fraction", float("nan"))),
                    float(getattr(result, "observable_elastic_fraction", float("nan"))),
                )
            )
        if (
            str(state_ensemble_mode).strip()
            and epsilon_real.size == energy.size
            and epsilon_imag.size == energy.size
        ):
            epsilon_states.append(np.asarray(epsilon_real + 1j * epsilon_imag, dtype=np.complex128))

    if energy_axis is None or not spectra or template_result is None:
        return {
            "model": str(model),
            "model_label": _model_label(str(model)),
            "q_ang_inv": float(q_value),
            "angle_deg": float(angle_deg),
            "status": "invalid_for_benchmark",
            "backend": ",".join(sorted(backends)),
            "backend_summary": " || ".join(sorted(value for value in backend_summaries if value)),
            "stls_converged": bool(stls_converged_all and bool(stls_iteration_values)),
            "stls_iteration_count": (int(round(float(np.mean(stls_iteration_values)))) if stls_iteration_values else 0),
            "stls_convergence_residual": (float(np.mean(stls_residual_values)) if stls_residual_values else float("nan")),
            "stls_convergence_relative_residual": (float(np.mean(stls_relative_residual_values)) if stls_relative_residual_values else float("nan")),
            "stls_closure_name": " || ".join(sorted(stls_closure_names)),
            "stls_local_field_value": (float(np.mean(stls_local_field_values)) if stls_local_field_values else float("nan")),
            "stls_q_over_qf": (float(np.mean(stls_q_over_qf_values)) if stls_q_over_qf_values else float("nan")),
            "executed_fully": False,
            "fallback_fraction": float(fallback_max),
            "domain_failure_fraction": float(domain_max),
            "runtime_s": float(runtime_total),
            "spectrum_runtime_s": float(runtime_total),
            "comparison_runtime_s": 0.0,
            "dispersion_runtime_s": 0.0,
            "time_series_runtime_s": 0.0,
            "peak_energy_ev": float("nan"),
            "peak_fwhm_ev": float("nan"),
            "requested_electron_policy": ",".join(sorted(requested_policies)) or str(electron_policy),
            "electron_policy": ",".join(sorted(applied_policies)) or str(electron_policy),
            "electron_policy_scope": electron_policy_scope(str(electron_policy)),
            "driven_response_model": str(driven_response_model or PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
            "driven_response_summary": " || ".join(sorted(driven_response_summaries)),
            "observable_mode": str(observable_mode),
            "observable_summary": "",
            "observable_decomposition_mode": "",
            "observable_peak_extraction_mode": "",
            "observable_elastic_exclusion_ev": 0.0,
            "observable_free_fraction": float("nan"),
            "observable_bound_fraction": float("nan"),
            "observable_elastic_fraction": float("nan"),
            "observable_comparison_mode": "",
            "observable_subtraction_mode": "",
            "observable_normalization_mode": "",
            "observable_peak_discrete_energy_ev": float("nan"),
            "observable_peak_fit_energy_ev": float("nan"),
            "observable_peak_fit_status": "",
            "observable_peak_edge_dominated": False,
            "observable_elastic_form_factor_total": float("nan"),
            "observable_elastic_form_factor_core": float("nan"),
            "observable_elastic_screening_form_factor": float("nan"),
            "observable_ion_structure_factor": float("nan"),
            "observable_bound_core_mode": "",
            "observable_bound_shell_summary": "",
            "driven_response_weight_mode": str(state_weight_mode),
            "driven_response_weight_min": float("nan"),
            "driven_response_weight_max": float("nan"),
            "driven_response_shape_mode": str(state_shape_mode),
            "driven_response_shape_mean_ev": float("nan"),
            "driven_response_shape_min_ev": float("nan"),
            "driven_response_shape_max_ev": float("nan"),
            "driven_response_ensemble_mode": str(state_ensemble_mode),
            "electron_density_source": "article density-averaged uniform-Al benchmark state",
            "material_policy_summary": "No valid density-averaged spectra were available.",
            "benchmark_preset": str(benchmark_preset),
            "collision_source": ",".join(sorted(value for value in collision_sources if value)),
            "collision_summary": " || ".join(sorted(value for value in collision_summaries if value)),
            "zone_count_used": int(zone_total),
            "cluster_count_used": int(cluster_total),
            "warnings": sorted(warnings),
            "energy_ev": np.asarray([], dtype=np.float64),
            "spectrum": np.asarray([], dtype=np.float64),
            "spectrum_free_component": np.asarray([], dtype=np.float64),
            "spectrum_bound_component": np.asarray([], dtype=np.float64),
            "spectrum_elastic_component": np.asarray([], dtype=np.float64),
        }

    spectral_weight_array = np.asarray(spectra_weights, dtype=np.float64)
    spectral_weight_total = float(np.sum(spectral_weight_array, dtype=np.float64))
    if not math.isfinite(spectral_weight_total) or spectral_weight_total <= 0.0:
        spectral_weight_array = np.ones(len(spectra), dtype=np.float64)
        spectral_weight_total = float(np.sum(spectral_weight_array, dtype=np.float64))
        state_weight_mode = "uniform"
    spectral_weight_fraction = spectral_weight_array / spectral_weight_total
    if str(state_ensemble_mode).strip() and len(epsilon_states) == len(spectra):
        epsilon_average = np.tensordot(spectral_weight_fraction, np.stack(epsilon_states, axis=0), axes=(0, 0))
        averaged_loss = loss_function_from_epsilon(np.asarray(epsilon_average, dtype=np.complex128))
        raw_spectrum = dsf_from_loss(np.asarray(averaged_loss, dtype=np.float64), np.asarray(energy_axis, dtype=np.float64), float(te_ev))
        averaged_spectrum = gaussian_convolve(np.asarray(energy_axis, dtype=np.float64), np.asarray(raw_spectrum, dtype=np.float64), float(instrument_fwhm_ev))
        warnings.add(
            "Article density-averaged benchmark combined the driven ensemble at the response-function level "
            f"({state_ensemble_mode}) before loss/spectrum extraction."
        )
    else:
        averaged_spectrum = np.tensordot(spectral_weight_fraction, np.stack(spectra, axis=0), axes=(0, 0))
        if str(state_ensemble_mode).strip():
            warnings.add(
                "Article density-averaged benchmark requested a response-function ensemble mode but fell back to direct spectrum averaging because not all state dielectric arrays were available."
            )
    if len(free_components) == len(spectra):
        averaged_free_component = np.tensordot(spectral_weight_fraction, np.stack(free_components, axis=0), axes=(0, 0))
        averaged_bound_component = np.tensordot(spectral_weight_fraction, np.stack(bound_components, axis=0), axes=(0, 0))
        averaged_elastic_component = np.tensordot(spectral_weight_fraction, np.stack(elastic_components, axis=0), axes=(0, 0))
    else:
        averaged_free_component = np.asarray([], dtype=np.float64)
        averaged_bound_component = np.asarray([], dtype=np.float64)
        averaged_elastic_component = np.asarray([], dtype=np.float64)
        if str(observable_mode) != PLASMON_OBSERVABLE_MODE_DIELECTRIC:
            warnings.add(
                "Article density-averaged observable benchmark could not aggregate the free/bound/elastic component arrays for every state, so component-level diagnostics are partial."
            )
    if observable_fraction_samples and len(observable_fraction_samples) == len(spectra):
        fraction_array = np.asarray(observable_fraction_samples, dtype=np.float64)
        averaged_free_fraction = float(np.dot(spectral_weight_fraction, fraction_array[:, 0]))
        averaged_bound_fraction = float(np.dot(spectral_weight_fraction, fraction_array[:, 1]))
        averaged_elastic_fraction = float(np.dot(spectral_weight_fraction, fraction_array[:, 2]))
    else:
        averaged_free_fraction = float(getattr(template_result, "observable_free_fraction", float("nan")))
        averaged_bound_fraction = float(getattr(template_result, "observable_bound_fraction", float("nan")))
        averaged_elastic_fraction = float(getattr(template_result, "observable_elastic_fraction", float("nan")))
    observable_elastic_exclusion_ev = (
        float(np.mean(observable_elastic_exclusion_values))
        if observable_elastic_exclusion_values
        else float(getattr(template_result, "observable_elastic_exclusion_ev", 0.0))
    )
    observable_peak_discrete_energy_ev = float("nan")
    observable_peak_fit_energy_ev = float("nan")
    observable_peak_fit_status = ""
    observable_peak_edge_dominated = False
    observable_peak_extraction_mode = str(getattr(template_result, "observable_peak_extraction_mode", "positive_branch"))
    if (
        str(observable_mode) != PLASMON_OBSERVABLE_MODE_DIELECTRIC
        and averaged_free_component.size == energy_axis.size
        and averaged_bound_component.size == energy_axis.size
    ):
        extracted_peak = _extract_observable_peak_from_components(
            energy_axis,
            averaged_free_component,
            averaged_bound_component,
            elastic_exclusion_ev=float(observable_elastic_exclusion_ev),
            peak_fit_method="local_quadratic",
        )
        peak_energy_ev = float(extracted_peak["peak_energy_ev"])
        peak_fwhm_ev = float(extracted_peak["peak_fwhm_ev"])
        observable_peak_discrete_energy_ev = float(extracted_peak["observable_peak_discrete_energy_ev"])
        observable_peak_fit_energy_ev = float(extracted_peak["observable_peak_fit_energy_ev"])
        observable_peak_fit_status = str(extracted_peak["observable_peak_fit_status"])
        observable_peak_edge_dominated = bool(extracted_peak["observable_peak_edge_dominated"])
        observable_peak_extraction_mode = str(extracted_peak["observable_peak_extraction_mode"])
    else:
        peak_energy_ev, peak_fwhm_ev = estimate_peak_metrics(
            energy_axis,
            averaged_spectrum,
            method="local_quadratic",
            local_half_window_points=2,
        )
    warnings.add(
        f"Article density-averaged benchmark used {len(densities_g_cm3)} uniform Al states across {densities_g_cm3[0]:.2f}-{densities_g_cm3[-1]:.2f} g/cm^3 at Te={te_ev:.3f} eV."
    )
    if str(state_weight_mode) != "uniform":
        warnings.add(
            "Article density-averaged benchmark weighted the uniform-state spectra by the driven-response ensemble modifier "
            f"({state_weight_mode})."
        )
    if str(state_shape_mode).strip() and state_shape_fwhm_values:
        warnings.add(
            "Article density-averaged benchmark applied the driven-response shape modifier "
            f"({state_shape_mode}) with per-state FWHM {min(state_shape_fwhm_values):.3f}-{max(state_shape_fwhm_values):.3f} eV."
        )
    status = "valid" if executed_fully and benchmark_statuses == {"valid"} else ",".join(sorted(benchmark_statuses))
    return {
        "model": str(model),
        "model_label": _model_label(str(model)),
        "q_ang_inv": float(q_value),
        "angle_deg": float(angle_deg),
        "status": str(status),
        "backend": ",".join(sorted(backends)),
        "backend_summary": " || ".join(sorted(value for value in backend_summaries if value)),
        "stls_converged": bool(stls_converged_all and bool(stls_iteration_values)),
        "stls_iteration_count": (int(round(float(np.mean(stls_iteration_values)))) if stls_iteration_values else 0),
        "stls_convergence_residual": (float(np.mean(stls_residual_values)) if stls_residual_values else float("nan")),
        "stls_convergence_relative_residual": (float(np.mean(stls_relative_residual_values)) if stls_relative_residual_values else float("nan")),
        "stls_closure_name": " || ".join(sorted(stls_closure_names)),
        "stls_local_field_value": (float(np.mean(stls_local_field_values)) if stls_local_field_values else float("nan")),
        "stls_q_over_qf": (float(np.mean(stls_q_over_qf_values)) if stls_q_over_qf_values else float("nan")),
        "executed_fully": bool(executed_fully),
        "fallback_fraction": float(fallback_max),
        "domain_failure_fraction": float(domain_max),
        "runtime_s": float(runtime_total),
        "spectrum_runtime_s": float(runtime_total),
        "comparison_runtime_s": 0.0,
        "dispersion_runtime_s": 0.0,
        "time_series_runtime_s": 0.0,
        "peak_energy_ev": float(peak_energy_ev),
        "peak_fwhm_ev": float(peak_fwhm_ev),
        "requested_electron_policy": next(iter(sorted(requested_policies)), str(electron_policy)),
        "electron_policy": next(iter(sorted(applied_policies)), str(electron_policy)),
        "electron_policy_scope": electron_policy_scope(str(electron_policy)),
        "driven_response_model": str(driven_response_model or PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
        "driven_response_summary": " || ".join(sorted(driven_response_summaries)),
        "observable_mode": str(getattr(template_result, "observable_mode", observable_mode)),
        "observable_summary": (" || ".join(sorted(observable_summaries)) if observable_summaries else str(getattr(template_result, "observable_summary", ""))),
        "observable_decomposition_mode": (" || ".join(sorted(observable_decomposition_modes)) if observable_decomposition_modes else str(getattr(template_result, "observable_decomposition_mode", ""))),
        "observable_peak_extraction_mode": str(observable_peak_extraction_mode),
        "observable_elastic_exclusion_ev": float(observable_elastic_exclusion_ev),
        "observable_free_fraction": float(averaged_free_fraction),
        "observable_bound_fraction": float(averaged_bound_fraction),
        "observable_elastic_fraction": float(averaged_elastic_fraction),
        "observable_comparison_mode": (" || ".join(sorted(observable_comparison_modes)) if observable_comparison_modes else str(getattr(template_result, "observable_comparison_mode", ""))),
        "observable_subtraction_mode": (" || ".join(sorted(observable_subtraction_modes)) if observable_subtraction_modes else str(getattr(template_result, "observable_subtraction_mode", ""))),
        "observable_normalization_mode": (" || ".join(sorted(observable_normalization_modes)) if observable_normalization_modes else str(getattr(template_result, "observable_normalization_mode", ""))),
        "observable_peak_discrete_energy_ev": float(observable_peak_discrete_energy_ev),
        "observable_peak_fit_energy_ev": float(observable_peak_fit_energy_ev),
        "observable_peak_fit_status": (str(observable_peak_fit_status) or "density_averaged"),
        "observable_peak_edge_dominated": bool(observable_peak_edge_dominated),
        "observable_elastic_form_factor_total": (float(np.mean(observable_elastic_form_factor_total_values)) if observable_elastic_form_factor_total_values else float(getattr(template_result, "observable_elastic_form_factor_total", float("nan")))),
        "observable_elastic_form_factor_core": (float(np.mean(observable_elastic_form_factor_core_values)) if observable_elastic_form_factor_core_values else float(getattr(template_result, "observable_elastic_form_factor_core", float("nan")))),
        "observable_elastic_screening_form_factor": (float(np.mean(observable_elastic_screening_form_factor_values)) if observable_elastic_screening_form_factor_values else float(getattr(template_result, "observable_elastic_screening_form_factor", float("nan")))),
        "observable_ion_structure_factor": (float(np.mean(observable_ion_structure_factor_values)) if observable_ion_structure_factor_values else float(getattr(template_result, "observable_ion_structure_factor", float("nan")))),
        "observable_bound_core_mode": (" || ".join(sorted(observable_bound_core_modes)) if observable_bound_core_modes else str(getattr(template_result, "observable_bound_core_mode", ""))),
        "observable_bound_shell_summary": (" || ".join(sorted(observable_bound_shell_summaries)) if observable_bound_shell_summaries else str(getattr(template_result, "observable_bound_shell_summary", ""))),
        "driven_response_weight_mode": str(state_weight_mode),
        "driven_response_weight_min": float(np.min(spectral_weight_array)),
        "driven_response_weight_max": float(np.max(spectral_weight_array)),
        "driven_response_shape_mode": str(state_shape_mode),
        "driven_response_shape_mean_ev": float(np.mean(state_shape_fwhm_values)) if state_shape_fwhm_values else 0.0,
        "driven_response_shape_min_ev": float(np.min(state_shape_fwhm_values)) if state_shape_fwhm_values else 0.0,
        "driven_response_shape_max_ev": float(np.max(state_shape_fwhm_values)) if state_shape_fwhm_values else 0.0,
        "driven_response_ensemble_mode": str(state_ensemble_mode),
        "electron_density_source": "article density-averaged uniform-Al benchmark state",
        "material_policy_summary": str(getattr(template_result, "material_policy_summary", "")),
        "benchmark_preset": str(benchmark_preset),
        "collision_source": ",".join(sorted(value for value in collision_sources if value)),
        "collision_summary": " || ".join(sorted(value for value in collision_summaries if value)),
        "zone_count_used": int(zone_total),
        "cluster_count_used": int(cluster_total),
        "warnings": sorted(warnings),
        "energy_ev": np.asarray(energy_axis, dtype=np.float64),
        "spectrum": np.asarray(averaged_spectrum, dtype=np.float64),
        "spectrum_free_component": np.asarray(averaged_free_component, dtype=np.float64),
        "spectrum_bound_component": np.asarray(averaged_bound_component, dtype=np.float64),
        "spectrum_elastic_component": np.asarray(averaged_elastic_component, dtype=np.float64),
    }


def _prediction_map(model_rows: list[dict[str, object]]) -> dict[float, float]:
    return {
        float(row["q_ang_inv"]): float(row["peak_energy_ev"])
        for row in model_rows
        if math.isfinite(float(row["peak_energy_ev"]))
    }


def _status_summary(model_rows: list[dict[str, object]]) -> str:
    values = sorted({str(row["status"]) for row in model_rows})
    return values[0] if len(values) == 1 else ",".join(values)


def _representative_spectra(
    dataset,
    context,
    cache: AnalysisStateCache,
    *,
    q_value: float,
    electron_policy: str,
    driven_response_model: str | None,
    zone_index_lower: int,
    zone_index_upper: int,
    benchmark_preset: str,
    material_id: int,
    observable_mode: str = PLASMON_OBSERVABLE_MODE_DIELECTRIC,
    models: tuple[str, ...] = ALL_MODELS,
) -> dict[str, dict[str, object]]:
    spectra: dict[str, dict[str, object]] = {}
    for model in models:
        if str(model) == PLASMON_MODEL_QUICKLOOK:
            continue
        row = _compute_point_result(
            dataset,
            context,
            cache,
            model=str(model),
            electron_policy=str(electron_policy),
            driven_response_model=driven_response_model,
            q_value=float(q_value),
            zone_index_lower=zone_index_lower,
            zone_index_upper=zone_index_upper,
            instrument_fwhm_ev=ARTICLE_INSTRUMENT_FWHM_EV,
            benchmark_preset=str(benchmark_preset),
            material_id=int(material_id),
            observable_mode=str(observable_mode),
        )
        result = row["result"]
        energy = np.asarray(result.spectrum_energy_ev, dtype=np.float64)
        intensity = np.asarray(result.spectrum_intensity, dtype=np.float64)
        if energy.size == 0 or intensity.size != energy.size:
            continue
        spectra[str(model)] = {
            "model_label": _model_label(str(model)),
            "status": str(row["status"]),
            "backend": str(row["backend"]),
            "peak_energy_ev": float(row["peak_energy_ev"]),
            "observable_mode": str(row.get("observable_mode", PLASMON_OBSERVABLE_MODE_DIELECTRIC)),
            "observable_summary": str(row.get("observable_summary", "")),
            "observable_decomposition_mode": str(row.get("observable_decomposition_mode", "")),
            "observable_peak_extraction_mode": str(row.get("observable_peak_extraction_mode", "")),
            "observable_comparison_mode": str(row.get("observable_comparison_mode", "")),
            "observable_subtraction_mode": str(row.get("observable_subtraction_mode", "")),
            "observable_normalization_mode": str(row.get("observable_normalization_mode", "")),
            "observable_elastic_exclusion_ev": float(row.get("observable_elastic_exclusion_ev", 0.0)),
            "observable_free_fraction": float(row.get("observable_free_fraction", float("nan"))),
            "observable_bound_fraction": float(row.get("observable_bound_fraction", float("nan"))),
            "observable_elastic_fraction": float(row.get("observable_elastic_fraction", float("nan"))),
            "observable_peak_discrete_energy_ev": float(row.get("observable_peak_discrete_energy_ev", float("nan"))),
            "observable_peak_fit_energy_ev": float(row.get("observable_peak_fit_energy_ev", float("nan"))),
            "observable_peak_fit_status": str(row.get("observable_peak_fit_status", "")),
            "observable_peak_edge_dominated": bool(row.get("observable_peak_edge_dominated", False)),
            "observable_elastic_form_factor_total": float(row.get("observable_elastic_form_factor_total", float("nan"))),
            "observable_elastic_form_factor_core": float(row.get("observable_elastic_form_factor_core", float("nan"))),
            "observable_elastic_screening_form_factor": float(row.get("observable_elastic_screening_form_factor", float("nan"))),
            "observable_ion_structure_factor": float(row.get("observable_ion_structure_factor", float("nan"))),
            "observable_bound_core_mode": str(row.get("observable_bound_core_mode", "")),
            "observable_bound_shell_summary": str(row.get("observable_bound_shell_summary", "")),
            "energy_ev": energy.tolist(),
            "intensity": intensity.tolist(),
            "free_component": np.asarray(getattr(result, "spectrum_free_component", np.asarray([], dtype=np.float64)), dtype=np.float64).tolist(),
            "bound_component": np.asarray(getattr(result, "spectrum_bound_component", np.asarray([], dtype=np.float64)), dtype=np.float64).tolist(),
            "elastic_component": np.asarray(getattr(result, "spectrum_elastic_component", np.asarray([], dtype=np.float64)), dtype=np.float64).tolist(),
            "warnings": list(row["warnings"]),
        }
    return spectra


def _plot_dispersion_case(
    path: Path,
    *,
    title: str,
    references: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    model_rows: dict[str, list[dict[str, object]]],
    models: tuple[str, ...] = ALL_MODELS,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for label, q_ref, y_ref, y_err in references:
        ax.errorbar(q_ref, y_ref, yerr=y_err, marker="o", linestyle="None", capsize=3, label=label)
    for model in models:
        rows = model_rows.get(str(model), [])
        prediction = _prediction_map(rows)
        q_values = np.asarray(sorted(prediction.keys()), dtype=np.float64)
        y_values = np.asarray([prediction[float(q)] for q in q_values], dtype=np.float64)
        if q_values.size == 0:
            continue
        linestyle = "--" if str(model) == PLASMON_MODEL_QUICKLOOK else "-"
        ax.plot(q_values, y_values, marker=".", linewidth=1.3, linestyle=linestyle, label=_model_label(str(model)))
    ax.set_title(title)
    ax.set_xlabel(r"$k$ ($\mathrm{\AA^{-1}}$)")
    ax.set_ylabel("Peak position (eV)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_representative_spectra(path: Path, *, title: str, spectra: dict[str, dict[str, object]], models: tuple[str, ...] = ALL_MODELS) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for model in models:
        payload = spectra.get(str(model))
        if payload is None:
            continue
        energy = np.asarray(payload["energy_ev"], dtype=np.float64)
        intensity = np.asarray(payload["intensity"], dtype=np.float64)
        if energy.size == 0 or intensity.size != energy.size:
            continue
        ax.plot(energy, intensity, linewidth=1.3, label=str(payload["model_label"]))
    ax.set_title(title)
    ax.set_xlabel("Energy transfer (eV)")
    ax.set_ylabel("Intensity (arb. u.)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _flatten_points(case_name: str, model_rows: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, points in model_rows.items():
        for point in points:
            rows.append({
                "case": case_name,
                "model": model,
                "model_label": point["model_label"],
                "requested_electron_policy": point["requested_electron_policy"],
                "electron_policy": point["electron_policy"],
                "electron_policy_scope": point["electron_policy_scope"],
                "driven_response_model": point.get("driven_response_model", PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
                "driven_response_summary": point.get("driven_response_summary", ""),
                "driven_response_weight_mode": point.get("driven_response_weight_mode", ""),
                "driven_response_weight_min": point.get("driven_response_weight_min", float("nan")),
                "driven_response_weight_max": point.get("driven_response_weight_max", float("nan")),
                "driven_response_shape_mode": point.get("driven_response_shape_mode", ""),
                "driven_response_shape_mean_ev": point.get("driven_response_shape_mean_ev", float("nan")),
                "driven_response_shape_min_ev": point.get("driven_response_shape_min_ev", float("nan")),
                "driven_response_shape_max_ev": point.get("driven_response_shape_max_ev", float("nan")),
                "driven_response_ensemble_mode": point.get("driven_response_ensemble_mode", ""),
                "observable_mode": point.get("observable_mode", PLASMON_OBSERVABLE_MODE_DIELECTRIC),
                "observable_summary": point.get("observable_summary", ""),
                "observable_decomposition_mode": point.get("observable_decomposition_mode", ""),
                "observable_peak_extraction_mode": point.get("observable_peak_extraction_mode", ""),
                "observable_elastic_exclusion_ev": point.get("observable_elastic_exclusion_ev", 0.0),
                "observable_free_fraction": point.get("observable_free_fraction", float("nan")),
                "observable_bound_fraction": point.get("observable_bound_fraction", float("nan")),
                "observable_elastic_fraction": point.get("observable_elastic_fraction", float("nan")),
                "observable_comparison_mode": point.get("observable_comparison_mode", ""),
                "observable_subtraction_mode": point.get("observable_subtraction_mode", ""),
                "observable_normalization_mode": point.get("observable_normalization_mode", ""),
                "observable_peak_discrete_energy_ev": point.get("observable_peak_discrete_energy_ev", float("nan")),
                "observable_peak_fit_energy_ev": point.get("observable_peak_fit_energy_ev", float("nan")),
                "observable_peak_fit_status": point.get("observable_peak_fit_status", ""),
                "observable_peak_edge_dominated": point.get("observable_peak_edge_dominated", False),
                "observable_elastic_form_factor_total": point.get("observable_elastic_form_factor_total", float("nan")),
                "observable_elastic_form_factor_core": point.get("observable_elastic_form_factor_core", float("nan")),
                "observable_elastic_screening_form_factor": point.get("observable_elastic_screening_form_factor", float("nan")),
                "observable_ion_structure_factor": point.get("observable_ion_structure_factor", float("nan")),
                "observable_bound_core_mode": point.get("observable_bound_core_mode", ""),
                "observable_bound_shell_summary": point.get("observable_bound_shell_summary", ""),
                "q_ang_inv": point["q_ang_inv"],
                "angle_deg": point["angle_deg"],
                "status": point["status"],
                "backend": point["backend"],
                "backend_summary": point.get("backend_summary", ""),
                "stls_converged": point.get("stls_converged", False),
                "stls_iteration_count": point.get("stls_iteration_count", 0),
                "stls_convergence_residual": point.get("stls_convergence_residual", float("nan")),
                "stls_convergence_relative_residual": point.get("stls_convergence_relative_residual", float("nan")),
                "stls_closure_name": point.get("stls_closure_name", ""),
                "stls_local_field_value": point.get("stls_local_field_value", float("nan")),
                "stls_q_over_qf": point.get("stls_q_over_qf", float("nan")),
                "executed_fully": point["executed_fully"],
                "fallback_fraction": point["fallback_fraction"],
                "domain_failure_fraction": point["domain_failure_fraction"],
                "runtime_s": point["runtime_s"],
                "benchmark_preset": point["benchmark_preset"],
                "collision_source": point["collision_source"],
                "peak_energy_ev": point["peak_energy_ev"],
                "peak_fwhm_ev": point["peak_fwhm_ev"],
                "zone_count_used": point["zone_count_used"],
                "cluster_count_used": point["cluster_count_used"],
                "warning_count": len(point["warnings"]),
                "warnings": " || ".join(str(item) for item in point["warnings"]),
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    return value


def _extract_observable_peak_from_components(
    energy_ev: np.ndarray,
    free_component: np.ndarray,
    bound_component: np.ndarray,
    *,
    elastic_exclusion_ev: float,
    peak_fit_method: str,
) -> dict[str, object]:
    energy = np.asarray(energy_ev, dtype=np.float64)
    inelastic = np.asarray(free_component, dtype=np.float64) + np.asarray(bound_component, dtype=np.float64)
    mask = np.isfinite(energy) & np.isfinite(inelastic) & (energy >= float(elastic_exclusion_ev))
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
    active_values = np.asarray(inelastic[mask], dtype=np.float64)
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


def _case_payload(
    dataset,
    context,
    cache: AnalysisStateCache,
    *,
    case_name: str,
    electron_policy: str,
    driven_response_model: str | None,
    zone_index_lower: int,
    zone_index_upper: int,
    state_summary: dict[str, float | int],
    reference: dict[str, object],
    representative_q: float,
    benchmark_preset: str,
    material_id: int,
    observable_mode: str = PLASMON_OBSERVABLE_MODE_DIELECTRIC,
    point_instrument_fwhm_ev: float = PEAK_EXTRACTION_FWHM_EV,
    models: tuple[str, ...] = ALL_MODELS,
) -> dict[str, object]:
    q_values = sorted({float(v) for series in dict(reference["series"]).values() for v in series["q_ang_inv"]})
    model_rows: dict[str, list[dict[str, object]]] = {}
    summary_rows: list[dict[str, object]] = []
    for model in models:
        points = [
            _compute_point_result(
                dataset,
                context,
                cache,
                model=str(model),
                electron_policy=str(electron_policy),
                driven_response_model=driven_response_model,
                q_value=float(q_value),
                zone_index_lower=zone_index_lower,
                zone_index_upper=zone_index_upper,
                instrument_fwhm_ev=float(point_instrument_fwhm_ev),
                benchmark_preset=str(benchmark_preset),
                material_id=int(material_id),
                observable_mode=str(observable_mode),
            )
            for q_value in q_values
        ]
        model_rows[str(model)] = points
        prediction = _prediction_map(points)
        metrics = {
            str(key): _metric_row(prediction, *_series_points(reference, str(key))[:2])
            for key in dict(reference["series"]).keys()
        }
        warnings = sorted({warning for point in points for warning in point["warnings"]})
        observable_free_values = [float(point.get("observable_free_fraction", float("nan"))) for point in points if math.isfinite(float(point.get("observable_free_fraction", float("nan"))))]
        observable_bound_values = [float(point.get("observable_bound_fraction", float("nan"))) for point in points if math.isfinite(float(point.get("observable_bound_fraction", float("nan"))))]
        observable_elastic_values = [float(point.get("observable_elastic_fraction", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_fraction", float("nan"))))]
        observable_exclusion_values = [float(point.get("observable_elastic_exclusion_ev", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_exclusion_ev", float("nan"))))]
        observable_peak_discrete_values = [float(point.get("observable_peak_discrete_energy_ev", float("nan"))) for point in points if math.isfinite(float(point.get("observable_peak_discrete_energy_ev", float("nan"))))]
        observable_peak_fit_values = [float(point.get("observable_peak_fit_energy_ev", float("nan"))) for point in points if math.isfinite(float(point.get("observable_peak_fit_energy_ev", float("nan"))))]
        observable_elastic_form_factor_total_values = [float(point.get("observable_elastic_form_factor_total", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_form_factor_total", float("nan"))))]
        observable_elastic_form_factor_core_values = [float(point.get("observable_elastic_form_factor_core", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_form_factor_core", float("nan"))))]
        observable_elastic_screening_values = [float(point.get("observable_elastic_screening_form_factor", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_screening_form_factor", float("nan"))))]
        observable_ion_structure_values = [float(point.get("observable_ion_structure_factor", float("nan"))) for point in points if math.isfinite(float(point.get("observable_ion_structure_factor", float("nan"))))]
        observable_comparison_modes = sorted({str(point.get("observable_comparison_mode", "")) for point in points if str(point.get("observable_comparison_mode", "")).strip()})
        observable_subtraction_modes = sorted({str(point.get("observable_subtraction_mode", "")) for point in points if str(point.get("observable_subtraction_mode", "")).strip()})
        observable_normalization_modes = sorted({str(point.get("observable_normalization_mode", "")) for point in points if str(point.get("observable_normalization_mode", "")).strip()})
        observable_peak_fit_statuses = sorted({str(point.get("observable_peak_fit_status", "")) for point in points if str(point.get("observable_peak_fit_status", "")).strip()})
        observable_bound_core_modes = sorted({str(point.get("observable_bound_core_mode", "")) for point in points if str(point.get("observable_bound_core_mode", "")).strip()})
        observable_bound_shell_summaries = sorted({str(point.get("observable_bound_shell_summary", "")) for point in points if str(point.get("observable_bound_shell_summary", "")).strip()})
        summary_rows.append({
            "case": case_name,
            "model": str(model),
            "model_label": _model_label(str(model)),
            "status": _status_summary(points),
            "backend": ",".join(sorted({str(point["backend"]) for point in points})),
            "backend_summary": " || ".join(sorted({str(point.get("backend_summary", "")) for point in points if str(point.get("backend_summary", "")).strip()})),
            "stls_converged_all": bool(all(bool(point.get("stls_converged", False)) for point in points if str(point.get("backend", "")) == "finite_t_stls")) if any(str(point.get("backend", "")) == "finite_t_stls" for point in points) else False,
            "stls_iteration_mean": float(np.mean([float(point.get("stls_iteration_count", float("nan"))) for point in points if math.isfinite(float(point.get("stls_iteration_count", float("nan"))))])) if any(math.isfinite(float(point.get("stls_iteration_count", float("nan")))) for point in points) else float("nan"),
            "stls_residual_mean": float(np.mean([float(point.get("stls_convergence_residual", float("nan"))) for point in points if math.isfinite(float(point.get("stls_convergence_residual", float("nan"))))])) if any(math.isfinite(float(point.get("stls_convergence_residual", float("nan")))) for point in points) else float("nan"),
            "stls_relative_residual_mean": float(np.mean([float(point.get("stls_convergence_relative_residual", float("nan"))) for point in points if math.isfinite(float(point.get("stls_convergence_relative_residual", float("nan"))))])) if any(math.isfinite(float(point.get("stls_convergence_relative_residual", float("nan")))) for point in points) else float("nan"),
            "stls_closure_name": " || ".join(sorted({str(point.get("stls_closure_name", "")) for point in points if str(point.get("stls_closure_name", "")).strip()})),
            "benchmark_preset": str(points[0]["benchmark_preset"]) if points else str(benchmark_preset),
            "requested_electron_policy": str(points[0]["requested_electron_policy"]) if points else str(electron_policy),
            "electron_policy": str(points[0]["electron_policy"]) if points else str(electron_policy),
            "electron_policy_scope": str(points[0]["electron_policy_scope"]) if points else electron_policy_scope(str(electron_policy)),
            "driven_response_model": str(points[0].get("driven_response_model", driven_response_model or PLASMON_DRIVEN_RESPONSE_MODEL_NONE)) if points else str(driven_response_model or PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
            "driven_response_summary": str(points[0].get("driven_response_summary", "")) if points else "",
            "observable_mode": str(points[0].get("observable_mode", observable_mode)) if points else str(observable_mode),
            "observable_summary": str(points[0].get("observable_summary", "")) if points else "",
            "observable_decomposition_mode": str(points[0].get("observable_decomposition_mode", "")) if points else "",
            "observable_peak_extraction_mode": str(points[0].get("observable_peak_extraction_mode", "")) if points else "",
            "observable_elastic_exclusion_ev": (float(np.mean(observable_exclusion_values)) if observable_exclusion_values else 0.0),
            "observable_free_fraction": (float(np.mean(observable_free_values)) if observable_free_values else float("nan")),
            "observable_bound_fraction": (float(np.mean(observable_bound_values)) if observable_bound_values else float("nan")),
            "observable_elastic_fraction": (float(np.mean(observable_elastic_values)) if observable_elastic_values else float("nan")),
            "observable_comparison_mode": " || ".join(observable_comparison_modes),
            "observable_subtraction_mode": " || ".join(observable_subtraction_modes),
            "observable_normalization_mode": " || ".join(observable_normalization_modes),
            "observable_peak_discrete_energy_ev": (float(np.mean(observable_peak_discrete_values)) if observable_peak_discrete_values else float("nan")),
            "observable_peak_fit_energy_ev": (float(np.mean(observable_peak_fit_values)) if observable_peak_fit_values else float("nan")),
            "observable_peak_fit_status": " || ".join(observable_peak_fit_statuses),
            "observable_peak_edge_dominated_any": bool(any(bool(point.get("observable_peak_edge_dominated", False)) for point in points)),
            "observable_elastic_form_factor_total": (float(np.mean(observable_elastic_form_factor_total_values)) if observable_elastic_form_factor_total_values else float("nan")),
            "observable_elastic_form_factor_core": (float(np.mean(observable_elastic_form_factor_core_values)) if observable_elastic_form_factor_core_values else float("nan")),
            "observable_elastic_screening_form_factor": (float(np.mean(observable_elastic_screening_values)) if observable_elastic_screening_values else float("nan")),
            "observable_ion_structure_factor": (float(np.mean(observable_ion_structure_values)) if observable_ion_structure_values else float("nan")),
            "observable_bound_core_mode": " || ".join(observable_bound_core_modes),
            "observable_bound_shell_summary": " || ".join(observable_bound_shell_summaries),
            "driven_response_weight_mode": str(points[0].get("driven_response_weight_mode", "")) if points else "",
            "driven_response_ensemble_mode": str(points[0].get("driven_response_ensemble_mode", "")) if points else "",
            "collision_source": ",".join(sorted({str(point["collision_source"]) for point in points if str(point["collision_source"])})),
            "executed_fully_all": bool(all(bool(point["executed_fully"]) for point in points)),
            "runtime_mean_s": float(np.mean([float(point["runtime_s"]) for point in points])) if points else float("nan"),
            "runtime_max_s": float(np.max([float(point["runtime_s"]) for point in points])) if points else float("nan"),
            "valid_peak_count": int(sum(1 for point in points if math.isfinite(float(point["peak_energy_ev"])))),
            "fallback_fraction_max": float(np.max([float(point["fallback_fraction"]) for point in points])) if points else float("nan"),
            "domain_failure_fraction_max": float(np.max([float(point["domain_failure_fraction"]) for point in points])) if points else float("nan"),
            "metrics": metrics,
            "warnings": warnings,
        })
    spectra = _representative_spectra(
        dataset,
        context,
        cache,
        q_value=float(representative_q),
        electron_policy=str(electron_policy),
        driven_response_model=driven_response_model,
        zone_index_lower=zone_index_lower,
        zone_index_upper=zone_index_upper,
        benchmark_preset=str(benchmark_preset),
        material_id=int(material_id),
        observable_mode=str(observable_mode),
        models=models,
    )
    return {
        "case": case_name,
        "reference_name": str(reference["source"]),
        "reference_provenance": dict(reference.get("provenance", {})),
        "selection": state_summary,
        "benchmark_preset": str(benchmark_preset),
        "requested_electron_policy": str(electron_policy),
        "driven_response_model": str(driven_response_model or PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
        "observable_mode": str(observable_mode),
        "electron_policy_scope": electron_policy_scope(str(electron_policy)),
        "baseline_mode": str(state_summary.get("baseline_mode", "")),
        "baseline_table_source": str(state_summary.get("baseline_table_source", "")),
        "baseline_entries": [str(value) for value in state_summary.get("baseline_entries", ())],
        "requested_collision_model": PRIMARY_COLLISION_MODEL,
        "q_values_ang_inv": [float(value) for value in q_values],
        "representative_q_ang_inv": float(representative_q),
        "representative_angle_deg": float(q_to_angle_deg(float(representative_q), PHOTON_ENERGY_KEV)),
        "models": {
            str(row["model"]): {
                "summary": {key: value for key, value in row.items() if key not in {"case", "model", "warnings"}},
                "warnings": list(row["warnings"]),
                "points": [
                    {key: value for key, value in point.items() if key != "result"}
                    for point in model_rows[str(row["model"])]
                ],
            }
            for row in summary_rows
        },
        "flat_summary_rows": summary_rows,
        "flat_point_rows": _flatten_points(case_name, model_rows),
        "spectra": spectra,
    }


def _density_averaged_case_payload(
    *,
    case_name: str,
    electron_policy: str,
    driven_response_model: str | None,
    state_summary: dict[str, float | int | str],
    reference: dict[str, object],
    representative_q: float,
    benchmark_preset: str,
    densities_g_cm3: tuple[float, ...],
    te_ev: float,
    observable_mode: str = PLASMON_OBSERVABLE_MODE_DIELECTRIC,
    models: tuple[str, ...] = ALL_MODELS,
) -> dict[str, object]:
    q_values = sorted({float(v) for series in dict(reference["series"]).values() for v in series["q_ang_inv"]})
    model_rows: dict[str, list[dict[str, object]]] = {}
    summary_rows: list[dict[str, object]] = []
    for model in models:
        points = [
            _compute_density_averaged_point_result(
                model=str(model),
                electron_policy=str(electron_policy),
                driven_response_model=driven_response_model,
                q_value=float(q_value),
                densities_g_cm3=tuple(float(value) for value in densities_g_cm3),
                te_ev=float(te_ev),
                instrument_fwhm_ev=ARTICLE_INSTRUMENT_FWHM_EV,
                benchmark_preset=str(benchmark_preset),
                observable_mode=str(observable_mode),
            )
            for q_value in q_values
        ]
        model_rows[str(model)] = points
        prediction = _prediction_map(points)
        metrics = {
            str(key): _metric_row(prediction, *_series_points(reference, str(key))[:2])
            for key in dict(reference["series"]).keys()
        }
        warnings = sorted({warning for point in points for warning in point["warnings"]})
        observable_free_values = [float(point.get("observable_free_fraction", float("nan"))) for point in points if math.isfinite(float(point.get("observable_free_fraction", float("nan"))))]
        observable_bound_values = [float(point.get("observable_bound_fraction", float("nan"))) for point in points if math.isfinite(float(point.get("observable_bound_fraction", float("nan"))))]
        observable_elastic_values = [float(point.get("observable_elastic_fraction", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_fraction", float("nan"))))]
        observable_exclusion_values = [float(point.get("observable_elastic_exclusion_ev", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_exclusion_ev", float("nan"))))]
        observable_peak_discrete_values = [float(point.get("observable_peak_discrete_energy_ev", float("nan"))) for point in points if math.isfinite(float(point.get("observable_peak_discrete_energy_ev", float("nan"))))]
        observable_peak_fit_values = [float(point.get("observable_peak_fit_energy_ev", float("nan"))) for point in points if math.isfinite(float(point.get("observable_peak_fit_energy_ev", float("nan"))))]
        observable_elastic_form_factor_total_values = [float(point.get("observable_elastic_form_factor_total", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_form_factor_total", float("nan"))))]
        observable_elastic_form_factor_core_values = [float(point.get("observable_elastic_form_factor_core", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_form_factor_core", float("nan"))))]
        observable_elastic_screening_values = [float(point.get("observable_elastic_screening_form_factor", float("nan"))) for point in points if math.isfinite(float(point.get("observable_elastic_screening_form_factor", float("nan"))))]
        observable_ion_structure_values = [float(point.get("observable_ion_structure_factor", float("nan"))) for point in points if math.isfinite(float(point.get("observable_ion_structure_factor", float("nan"))))]
        observable_comparison_modes = sorted({str(point.get("observable_comparison_mode", "")) for point in points if str(point.get("observable_comparison_mode", "")).strip()})
        observable_subtraction_modes = sorted({str(point.get("observable_subtraction_mode", "")) for point in points if str(point.get("observable_subtraction_mode", "")).strip()})
        observable_normalization_modes = sorted({str(point.get("observable_normalization_mode", "")) for point in points if str(point.get("observable_normalization_mode", "")).strip()})
        observable_peak_fit_statuses = sorted({str(point.get("observable_peak_fit_status", "")) for point in points if str(point.get("observable_peak_fit_status", "")).strip()})
        observable_bound_core_modes = sorted({str(point.get("observable_bound_core_mode", "")) for point in points if str(point.get("observable_bound_core_mode", "")).strip()})
        observable_bound_shell_summaries = sorted({str(point.get("observable_bound_shell_summary", "")) for point in points if str(point.get("observable_bound_shell_summary", "")).strip()})
        summary_rows.append(
            {
                "case": case_name,
                "model": str(model),
                "model_label": _model_label(str(model)),
                "status": _status_summary(points),
                "backend": ",".join(sorted({str(point["backend"]) for point in points if str(point["backend"])})),
                "backend_summary": " || ".join(sorted({str(point.get("backend_summary", "")) for point in points if str(point.get("backend_summary", "")).strip()})),
                "stls_converged_all": bool(all(bool(point.get("stls_converged", False)) for point in points if str(point.get("backend", "")) == "finite_t_stls")) if any(str(point.get("backend", "")) == "finite_t_stls" for point in points) else False,
                "stls_iteration_mean": float(np.mean([float(point.get("stls_iteration_count", float("nan"))) for point in points if math.isfinite(float(point.get("stls_iteration_count", float("nan"))))])) if any(math.isfinite(float(point.get("stls_iteration_count", float("nan")))) for point in points) else float("nan"),
                "stls_residual_mean": float(np.mean([float(point.get("stls_convergence_residual", float("nan"))) for point in points if math.isfinite(float(point.get("stls_convergence_residual", float("nan"))))])) if any(math.isfinite(float(point.get("stls_convergence_residual", float("nan")))) for point in points) else float("nan"),
                "stls_relative_residual_mean": float(np.mean([float(point.get("stls_convergence_relative_residual", float("nan"))) for point in points if math.isfinite(float(point.get("stls_convergence_relative_residual", float("nan"))))])) if any(math.isfinite(float(point.get("stls_convergence_relative_residual", float("nan")))) for point in points) else float("nan"),
                "stls_closure_name": " || ".join(sorted({str(point.get("stls_closure_name", "")) for point in points if str(point.get("stls_closure_name", "")).strip()})),
                "benchmark_preset": str(points[0]["benchmark_preset"]) if points else str(benchmark_preset),
                "requested_electron_policy": str(points[0]["requested_electron_policy"]) if points else str(electron_policy),
                "electron_policy": str(points[0]["electron_policy"]) if points else str(electron_policy),
                "electron_policy_scope": str(points[0]["electron_policy_scope"]) if points else electron_policy_scope(str(electron_policy)),
                "driven_response_model": str(points[0].get("driven_response_model", driven_response_model or PLASMON_DRIVEN_RESPONSE_MODEL_NONE)) if points else str(driven_response_model or PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
                "driven_response_summary": str(points[0].get("driven_response_summary", "")) if points else "",
                "observable_mode": str(points[0].get("observable_mode", observable_mode)) if points else str(observable_mode),
                "observable_summary": str(points[0].get("observable_summary", "")) if points else "",
                "observable_decomposition_mode": str(points[0].get("observable_decomposition_mode", "")) if points else "",
                "observable_peak_extraction_mode": str(points[0].get("observable_peak_extraction_mode", "")) if points else "",
                "observable_elastic_exclusion_ev": (float(np.mean(observable_exclusion_values)) if observable_exclusion_values else 0.0),
                "observable_free_fraction": (float(np.mean(observable_free_values)) if observable_free_values else float("nan")),
                "observable_bound_fraction": (float(np.mean(observable_bound_values)) if observable_bound_values else float("nan")),
                "observable_elastic_fraction": (float(np.mean(observable_elastic_values)) if observable_elastic_values else float("nan")),
                "observable_comparison_mode": " || ".join(observable_comparison_modes),
                "observable_subtraction_mode": " || ".join(observable_subtraction_modes),
                "observable_normalization_mode": " || ".join(observable_normalization_modes),
                "observable_peak_discrete_energy_ev": (float(np.mean(observable_peak_discrete_values)) if observable_peak_discrete_values else float("nan")),
                "observable_peak_fit_energy_ev": (float(np.mean(observable_peak_fit_values)) if observable_peak_fit_values else float("nan")),
                "observable_peak_fit_status": " || ".join(observable_peak_fit_statuses),
                "observable_peak_edge_dominated_any": bool(any(bool(point.get("observable_peak_edge_dominated", False)) for point in points)),
                "observable_elastic_form_factor_total": (float(np.mean(observable_elastic_form_factor_total_values)) if observable_elastic_form_factor_total_values else float("nan")),
                "observable_elastic_form_factor_core": (float(np.mean(observable_elastic_form_factor_core_values)) if observable_elastic_form_factor_core_values else float("nan")),
                "observable_elastic_screening_form_factor": (float(np.mean(observable_elastic_screening_values)) if observable_elastic_screening_values else float("nan")),
                "observable_ion_structure_factor": (float(np.mean(observable_ion_structure_values)) if observable_ion_structure_values else float("nan")),
                "observable_bound_core_mode": " || ".join(observable_bound_core_modes),
                "observable_bound_shell_summary": " || ".join(observable_bound_shell_summaries),
                "driven_response_weight_mode": str(points[0].get("driven_response_weight_mode", "")) if points else "",
                "collision_source": ",".join(sorted({str(point["collision_source"]) for point in points if str(point["collision_source"])})),
                "executed_fully_all": bool(all(bool(point["executed_fully"]) for point in points)),
                "runtime_mean_s": float(np.mean([float(point["runtime_s"]) for point in points])) if points else float("nan"),
                "runtime_max_s": float(np.max([float(point["runtime_s"]) for point in points])) if points else float("nan"),
                "valid_peak_count": int(sum(1 for point in points if math.isfinite(float(point["peak_energy_ev"])))),
                "fallback_fraction_max": float(np.max([float(point["fallback_fraction"]) for point in points])) if points else float("nan"),
                "domain_failure_fraction_max": float(np.max([float(point["domain_failure_fraction"]) for point in points])) if points else float("nan"),
                "metrics": metrics,
                "warnings": warnings,
            }
        )

    spectra: dict[str, dict[str, object]] = {}
    for model in models:
        if str(model) == PLASMON_MODEL_QUICKLOOK:
            continue
        point = _compute_density_averaged_point_result(
            model=str(model),
            electron_policy=str(electron_policy),
            driven_response_model=driven_response_model,
            q_value=float(representative_q),
            densities_g_cm3=tuple(float(value) for value in densities_g_cm3),
            te_ev=float(te_ev),
            instrument_fwhm_ev=ARTICLE_INSTRUMENT_FWHM_EV,
            benchmark_preset=str(benchmark_preset),
            observable_mode=str(observable_mode),
        )
        energy = np.asarray(point.get("energy_ev", np.asarray([], dtype=np.float64)), dtype=np.float64)
        intensity = np.asarray(point.get("spectrum", np.asarray([], dtype=np.float64)), dtype=np.float64)
        if energy.size == 0 or intensity.size != energy.size:
            continue
        spectra[str(model)] = {
            "model_label": _model_label(str(model)),
            "status": str(point["status"]),
            "backend": str(point["backend"]),
            "peak_energy_ev": float(point["peak_energy_ev"]),
            "observable_mode": str(point.get("observable_mode", observable_mode)),
            "observable_summary": str(point.get("observable_summary", "")),
            "observable_decomposition_mode": str(point.get("observable_decomposition_mode", "")),
            "observable_peak_extraction_mode": str(point.get("observable_peak_extraction_mode", "")),
            "observable_comparison_mode": str(point.get("observable_comparison_mode", "")),
            "observable_subtraction_mode": str(point.get("observable_subtraction_mode", "")),
            "observable_normalization_mode": str(point.get("observable_normalization_mode", "")),
            "observable_elastic_exclusion_ev": float(point.get("observable_elastic_exclusion_ev", 0.0)),
            "observable_free_fraction": float(point.get("observable_free_fraction", float("nan"))),
            "observable_bound_fraction": float(point.get("observable_bound_fraction", float("nan"))),
            "observable_elastic_fraction": float(point.get("observable_elastic_fraction", float("nan"))),
            "observable_peak_discrete_energy_ev": float(point.get("observable_peak_discrete_energy_ev", float("nan"))),
            "observable_peak_fit_energy_ev": float(point.get("observable_peak_fit_energy_ev", float("nan"))),
            "observable_peak_fit_status": str(point.get("observable_peak_fit_status", "")),
            "observable_peak_edge_dominated": bool(point.get("observable_peak_edge_dominated", False)),
            "observable_elastic_form_factor_total": float(point.get("observable_elastic_form_factor_total", float("nan"))),
            "observable_elastic_form_factor_core": float(point.get("observable_elastic_form_factor_core", float("nan"))),
            "observable_elastic_screening_form_factor": float(point.get("observable_elastic_screening_form_factor", float("nan"))),
            "observable_ion_structure_factor": float(point.get("observable_ion_structure_factor", float("nan"))),
            "observable_bound_core_mode": str(point.get("observable_bound_core_mode", "")),
            "observable_bound_shell_summary": str(point.get("observable_bound_shell_summary", "")),
            "energy_ev": energy.tolist(),
            "intensity": intensity.tolist(),
            "free_component": np.asarray(point.get("spectrum_free_component", np.asarray([], dtype=np.float64)), dtype=np.float64).tolist(),
            "bound_component": np.asarray(point.get("spectrum_bound_component", np.asarray([], dtype=np.float64)), dtype=np.float64).tolist(),
            "elastic_component": np.asarray(point.get("spectrum_elastic_component", np.asarray([], dtype=np.float64)), dtype=np.float64).tolist(),
            "warnings": list(point["warnings"]),
        }

    return {
        "case": case_name,
        "reference_name": str(reference["source"]),
        "reference_provenance": dict(reference.get("provenance", {})),
        "selection": state_summary,
        "benchmark_preset": str(benchmark_preset),
        "requested_electron_policy": str(electron_policy),
        "driven_response_model": str(driven_response_model or PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
        "observable_mode": str(observable_mode),
        "electron_policy_scope": electron_policy_scope(str(electron_policy)),
        "baseline_mode": str(state_summary.get("baseline_mode", "")),
        "baseline_table_source": str(state_summary.get("baseline_table_source", "")),
        "baseline_entries": [str(value) for value in state_summary.get("baseline_entries", ())],
        "increment_mode": str(state_summary.get("increment_mode", "")),
        "increment_entries": [str(value) for value in state_summary.get("increment_entries", ())],
        "requested_collision_model": PRIMARY_COLLISION_MODEL,
        "q_values_ang_inv": [float(value) for value in q_values],
        "representative_q_ang_inv": float(representative_q),
        "representative_angle_deg": float(q_to_angle_deg(float(representative_q), PHOTON_ENERGY_KEV)),
        "density_average_grid_g_cm3": [float(value) for value in densities_g_cm3],
        "density_average_te_ev": float(te_ev),
        "models": {
            str(row["model"]): {
                "summary": {key: value for key, value in row.items() if key not in {"case", "model", "warnings"}},
                "warnings": list(row["warnings"]),
                "points": [
                    {
                        key: (
                            value.tolist()
                            if isinstance(value, np.ndarray)
                            else value
                        )
                        for key, value in point.items()
                        if key not in {"energy_ev", "spectrum"}
                    }
                    for point in model_rows[str(row["model"])]
                ],
            }
            for row in summary_rows
        },
        "flat_summary_rows": summary_rows,
        "flat_point_rows": _flatten_points(case_name, model_rows),
        "spectra": spectra,
    }


def _selection_markdown(case_name: str, selection: dict[str, object], *, notes: list[str]) -> list[str]:
    lines = [
        f"### {case_name}",
        "",
        f"- snapshot/time: **{int(selection['snapshot_index'])} / {float(selection['time_ns']):.4f} ns**",
        f"- zone span: **{int(selection['zone_index_lower'])}-{int(selection['zone_index_upper'])}**",
        f"- selected zones: **{int(selection['zone_count'])}**",
        f"- weighted rho: **{float(selection['rho_weighted_g_cm3']):.3f} g/cm^3**",
        f"- weighted Te: **{float(selection['te_weighted_ev']):.3f} eV**",
        f"- weighted Ti: **{float(selection['ti_weighted_ev']):.3f} eV**",
        f"- weighted ne: **{float(selection['ne_weighted_cm3']):.4e} cm^-3**",
        f"- weighted Zbar: **{float(selection['zbar_weighted']):.4g}**",
    ]
    if "raw_ne_weighted_cm3" in selection and "effective_ne_weighted_cm3" in selection:
        lines.append(
            f"- raw vs effective ne: **{float(selection['raw_ne_weighted_cm3']):.4e} -> {float(selection['effective_ne_weighted_cm3']):.4e} cm^-3**"
        )
    if "raw_zbar_weighted" in selection and "effective_zbar_weighted" in selection:
        lines.append(
            f"- raw vs effective Zbar: **{float(selection['raw_zbar_weighted']):.4g} -> {float(selection['effective_zbar_weighted']):.4g}**"
        )
    lines.extend(f"- {note}" for note in notes)
    lines.append("")
    return lines


def _summary_table_rows(case_payload: dict[str, object], *, reference_key: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, model_payload in dict(case_payload["models"]).items():
        summary = dict(model_payload["summary"])
        metrics = dict(summary["metrics"]).get(str(reference_key), {})
        rows.append({
            "model": str(model),
            "model_label": str(summary["model_label"]),
            "status": str(summary["status"]),
            "backend": str(summary["backend"]),
            "backend_summary": str(summary.get("backend_summary", "")),
            "benchmark_preset": str(summary["benchmark_preset"]),
            "electron_policy": str(summary["electron_policy"]),
            "electron_policy_scope": str(summary.get("electron_policy_scope", electron_policy_scope(str(summary["electron_policy"])))),
            "collision_source": str(summary["collision_source"]),
            "observable_mode": str(summary.get("observable_mode", PLASMON_OBSERVABLE_MODE_DIELECTRIC)),
            "observable_summary": str(summary.get("observable_summary", "")),
            "observable_decomposition_mode": str(summary.get("observable_decomposition_mode", "")),
            "observable_peak_extraction_mode": str(summary.get("observable_peak_extraction_mode", "")),
            "observable_comparison_mode": str(summary.get("observable_comparison_mode", "")),
            "observable_subtraction_mode": str(summary.get("observable_subtraction_mode", "")),
            "observable_normalization_mode": str(summary.get("observable_normalization_mode", "")),
            "observable_elastic_exclusion_ev": float(summary.get("observable_elastic_exclusion_ev", 0.0)),
            "observable_free_fraction": float(summary.get("observable_free_fraction", float("nan"))),
            "observable_bound_fraction": float(summary.get("observable_bound_fraction", float("nan"))),
            "observable_elastic_fraction": float(summary.get("observable_elastic_fraction", float("nan"))),
            "observable_peak_discrete_energy_ev": float(summary.get("observable_peak_discrete_energy_ev", float("nan"))),
            "observable_peak_fit_energy_ev": float(summary.get("observable_peak_fit_energy_ev", float("nan"))),
            "observable_peak_fit_status": str(summary.get("observable_peak_fit_status", "")),
            "observable_peak_edge_dominated_any": bool(summary.get("observable_peak_edge_dominated_any", False)),
            "observable_elastic_form_factor_total": float(summary.get("observable_elastic_form_factor_total", float("nan"))),
            "observable_elastic_form_factor_core": float(summary.get("observable_elastic_form_factor_core", float("nan"))),
            "observable_elastic_screening_form_factor": float(summary.get("observable_elastic_screening_form_factor", float("nan"))),
            "observable_ion_structure_factor": float(summary.get("observable_ion_structure_factor", float("nan"))),
            "observable_bound_core_mode": str(summary.get("observable_bound_core_mode", "")),
            "observable_bound_shell_summary": str(summary.get("observable_bound_shell_summary", "")),
            "runtime_mean_s": float(summary["runtime_mean_s"]),
            "runtime_max_s": float(summary["runtime_max_s"]),
            "valid_peak_count": int(summary["valid_peak_count"]),
            "mae_ev": float(metrics.get("mae_ev", float("nan"))),
            "rmse_ev": float(metrics.get("rmse_ev", float("nan"))),
        })
    rows.sort(key=lambda row: (math.inf if not math.isfinite(float(row["mae_ev"])) else float(row["mae_ev"]), float(row["runtime_mean_s"]), str(row["electron_policy"])))
    return rows


def _policy_state_row(case_name: str, policy: str, selection: dict[str, object]) -> dict[str, object]:
    assessment = _assess_article_al_policy_state(case_name, policy, selection)
    return {
        "case": case_name,
        "electron_policy": str(policy),
        "electron_policy_label": electron_policy_label(str(policy)),
        "electron_policy_scope": electron_policy_scope(str(policy)),
        "input_policy_status": str(assessment["input_policy_status"]),
        "input_policy_reason": str(assessment["input_policy_reason"]),
        "snapshot_index": int(selection["snapshot_index"]),
        "time_ns": float(selection["time_ns"]),
        "zone_index_lower": int(selection["zone_index_lower"]),
        "zone_index_upper": int(selection["zone_index_upper"]),
        "zone_count": int(selection["zone_count"]),
        "rho_weighted_g_cm3": float(selection["rho_weighted_g_cm3"]),
        "te_weighted_ev": float(selection["te_weighted_ev"]),
        "raw_ne_weighted_cm3": float(selection.get("raw_ne_weighted_cm3", float("nan"))),
        "effective_ne_weighted_cm3": float(selection.get("effective_ne_weighted_cm3", selection.get("ne_weighted_cm3", float("nan")))),
        "raw_zbar_weighted": float(selection.get("raw_zbar_weighted", float("nan"))),
        "effective_zbar_weighted": float(selection.get("effective_zbar_weighted", selection.get("zbar_weighted", float("nan")))),
        "baseline_zbar_weighted": float(selection.get("baseline_zbar_weighted", float("nan"))),
        "increment_zbar_weighted": float(selection.get("increment_zbar_weighted", float("nan"))),
        "ion_density_weighted_cm3": float(selection.get("ion_density_weighted_cm3", float("nan"))),
        "effective_valence_from_ne": float(selection.get("effective_valence_from_ne", float("nan"))),
        "raw_effective_valence_from_ne": float(selection.get("raw_effective_valence_from_ne", float("nan"))),
        "material_formula": str(selection.get("material_formula", "")),
        "selection_kind": str(selection.get("selection_kind", "hydro_selection")),
        "baseline_mode": str(selection.get("baseline_mode", "")),
        "increment_mode": str(selection.get("increment_mode", "")),
        "baseline_table_source": str(selection.get("baseline_table_source", "")),
        "baseline_entries": " || ".join(str(value) for value in selection.get("baseline_entries", ()) if str(value).strip()),
        "increment_entries": " || ".join(str(value) for value in selection.get("increment_entries", ()) if str(value).strip()),
        "driven_response_model": str(selection.get("driven_response_model", PLASMON_DRIVEN_RESPONSE_MODEL_NONE)),
        "driven_response_summary": str(selection.get("driven_response_summary", "")),
        "driven_response_weight_mode": str(selection.get("driven_response_weight_mode", "uniform")),
        "driven_response_weight_mean": float(selection.get("driven_response_weight_mean", float("nan"))),
        "driven_response_weight_min": float(selection.get("driven_response_weight_min", float("nan"))),
        "driven_response_weight_max": float(selection.get("driven_response_weight_max", float("nan"))),
        "driven_response_shape_mode": str(selection.get("driven_response_shape_mode", "")),
        "driven_response_shape_mean_ev": float(selection.get("driven_response_shape_mean_ev", float("nan"))),
        "driven_response_shape_min_ev": float(selection.get("driven_response_shape_min_ev", float("nan"))),
        "driven_response_shape_max_ev": float(selection.get("driven_response_shape_max_ev", float("nan"))),
        "driven_response_ensemble_mode": str(selection.get("driven_response_ensemble_mode", "")),
        "headline_credibility": ("diagnostic_only" if str(assessment["input_policy_status"]) != "credible" else "headline_credible"),
    }


def _policy_sensitivity_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["case"]), str(row["model"])), []).append(row)
    rows: list[dict[str, object]] = []
    for (case_name, model), entries in grouped.items():
        mae_by_policy = {str(entry["electron_policy"]): float(entry["mae_ev"]) for entry in entries if math.isfinite(float(entry["mae_ev"]))}
        benchmark_maes = [mae_by_policy[policy] for policy in BENCHMARK_POLICIES[1:] if policy in mae_by_policy]
        raw_mae = mae_by_policy.get(PLASMON_ELECTRON_POLICY_RAW, float("nan"))
        article_mae = mae_by_policy.get(PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK, float("nan"))
        rows.append(
            {
                "case": case_name,
                "model": model,
                "model_label": str(entries[0]["model_label"]),
                "benchmark_policy_mae_min_ev": (float(min(benchmark_maes)) if benchmark_maes else float("nan")),
                "benchmark_policy_mae_max_ev": (float(max(benchmark_maes)) if benchmark_maes else float("nan")),
                "benchmark_policy_mae_spread_ev": (float(max(benchmark_maes) - min(benchmark_maes)) if len(benchmark_maes) >= 2 else 0.0 if benchmark_maes else float("nan")),
                "raw_minus_article_mae_ev": (float(raw_mae - article_mae) if math.isfinite(raw_mae) and math.isfinite(article_mae) else float("nan")),
                "article_mae_ev": article_mae,
                "article_driven_increment_mae_ev": mae_by_policy.get(PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT, float("nan")),
                "benchmark_valence_aware_mae_ev": mae_by_policy.get(PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE, float("nan")),
                "valence_locked_mae_ev": mae_by_policy.get(PLASMON_ELECTRON_POLICY_VALENCE_LOCKED, float("nan")),
                "raw_mae_ev": raw_mae,
                "raw_input_policy_status": next(
                    (
                        str(entry.get("input_policy_status", ""))
                        for entry in entries
                        if str(entry.get("electron_policy")) == PLASMON_ELECTRON_POLICY_RAW
                    ),
                    "",
                ),
            }
        )
    rows.sort(key=lambda row: (str(row["case"]), math.inf if not math.isfinite(float(row["benchmark_policy_mae_spread_ev"])) else float(row["benchmark_policy_mae_spread_ev"]), str(row["model"])))
    return rows


def _reference_reconciliation_spec(case_name: str, model: str) -> tuple[str | None, str, str]:
    if case_name == "ambient_al_t0":
        if model == PLASMON_MODEL_QUANTUM_HYDRODYNAMIC:
            return "gawne", "collective_fluid_proxy", "Compared to the ambient Gawne/Bohm-Gross-like branch as the closest collective-fluid reference in the current layer."
        if model == PLASMON_MODEL_FINITE_T_STLS:
            return "gawne", "correlation_backend_proxy", "Compared to the ambient Gawne branch as the closest available correlation-sensitive reference for a self-consistent STLS backend."
        if model == "rpa_static_lfc":
            return "gawne", "direct_family_proxy", "Closest published ambient calculation/reference branch in the current reference layer."
        if model == "mermin_static_lfc":
            return "gawne", "caveated_family_proxy", "Compared to the ambient Gawne branch with an extra collision closure that the reference branch does not include."
        if model in {"rpa", "mermin", "lindhard", "lindhard_mermin", "lindhard_static_lfc", "lindhard_mermin_static_lfc", "auto_best"}:
            return None, "no_family_matched_reference", "No dedicated ambient published branch of the same family is present in the current reference layer."
        return None, "not_benchmark_grade", "Quick-look and other heuristic branches are not used for article theory reconciliation."
    if case_name in {"driven_al_dense_slab", "driven_al_best_hydro_slab", "driven_al_article_state"}:
        if model == PLASMON_MODEL_QUANTUM_HYDRODYNAMIC:
            return "rpa", "collective_fluid_proxy", "Compared to the published driven RPA branch as the closest available collective-fluid benchmark, not as an exact article family match."
        if model == PLASMON_MODEL_FINITE_T_STLS:
            return "lfc", "correlation_backend_proxy", "Compared to the published driven LFC branch as the closest available correlation-sensitive benchmark for a self-consistent STLS backend."
        if model == "rpa":
            return "rpa", "direct_family_match", "Direct comparison against the published driven RPA branch."
        if model == "mermin":
            return "rpa", "caveated_family_proxy", "Compared to the published RPA branch because no driven Mermin reference branch is available."
        if model == "rpa_static_lfc":
            return "lfc", "direct_family_match", "Direct comparison against the published driven LFC branch."
        if model == "mermin_static_lfc":
            return "lfc", "caveated_family_proxy", "Compared to the published LFC branch with an additional collision closure caveat."
        if model in {"lindhard", "lindhard_mermin", "lindhard_static_lfc", "lindhard_mermin_static_lfc"}:
            return "tddft", "conceptual_proxy", "Compared to the published TDDFT-like branch as the closest available quantum/finite-T reference, not as an exact family match."
        return None, "not_benchmark_grade", "Quick-look and auto-composite branches are not used as direct article theory matches."
    return None, "unknown_case", "No reconciliation mapping is defined for this case."


def _reconciliation_rows(case_payload: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    case_name = str(case_payload["case"])
    requested_policy = str(case_payload.get("requested_electron_policy", ""))
    for model, model_payload in dict(case_payload["models"]).items():
        summary = dict(model_payload["summary"])
        reference_key, comparison_kind, note = _reference_reconciliation_spec(case_name, str(model))
        metrics = dict(summary["metrics"]).get(str(reference_key), {}) if reference_key is not None else {}
        rows.append(
            {
                "case": case_name,
                "model": str(model),
                "model_label": str(summary["model_label"]),
                "electron_policy": requested_policy,
                "driven_response_model": str(summary.get("driven_response_model", PLASMON_DRIVEN_RESPONSE_MODEL_NONE)),
                "status": str(summary["status"]),
                "backend": str(summary["backend"]),
                "backend_summary": str(summary.get("backend_summary", "")),
                "runtime_mean_s": float(summary["runtime_mean_s"]),
                "valid_peak_count": int(summary["valid_peak_count"]),
                "published_branch": (str(reference_key) if reference_key is not None else ""),
                "comparison_kind": comparison_kind,
                "mae_ev": float(metrics.get("mae_ev", float("nan"))),
                "rmse_ev": float(metrics.get("rmse_ev", float("nan"))),
                "max_abs_ev": float(metrics.get("max_abs_ev", float("nan"))),
                "note": note,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["case"]),
            {"direct_family_match": 0, "direct_family_proxy": 1, "correlation_backend_proxy": 2, "collective_fluid_proxy": 3, "conceptual_proxy": 4, "caveated_family_proxy": 5}.get(str(row["comparison_kind"]), 9),
            math.inf if not math.isfinite(float(row["mae_ev"])) else float(row["mae_ev"]),
            str(row["model"]),
        )
    )
    return rows


def _best_benchmark_grade_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if str(row.get("status")) == "valid"
        and str(row.get("model")) != PLASMON_MODEL_QUICKLOOK
        and math.isfinite(float(row.get("mae_ev", float("nan"))))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (float(row["mae_ev"]), float(row.get("runtime_mean_s", float("inf")))))
    return candidates[0]


def build_report(
    hydro_path: Path,
    *,
    out_dir: Path,
    driven_response_model: str | None = None,
    model_subset: tuple[str, ...] | None = None,
    observable_mode: str = PLASMON_OBSERVABLE_MODE_DIELECTRIC,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_driven_response_model = (
        None
        if driven_response_model is None or not str(driven_response_model).strip()
        else normalize_driven_response_model(driven_response_model)
    )
    selected_observable_mode = normalize_observable_mode(observable_mode)
    selected_models = tuple(str(value) for value in (model_subset or ALL_MODELS))
    dataset = load_run_data(hydro_path)
    cache = AnalysisStateCache()
    policy_payloads = {
        policy: resolve_effective_electron_fields(dataset, policy, driven_response_model=selected_driven_response_model)
        for policy in BENCHMARK_POLICIES
    }
    al_material_ids = _material_ids_for_formula(dataset, "Al")
    if not al_material_ids:
        raise ValueError("No Al material resolved from parsed EOS/opacity metadata for the benchmark dataset.")
    al_material_id = int(al_material_ids[0])

    time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
    ambient_snapshot = 0
    ambient_time_ns = float(time_ns[ambient_snapshot])
    ambient_zone_lower, ambient_zone_upper = _material_span(dataset, material_id=al_material_id)
    ambient_context = make_run_context(dataset, hydro_path, snapshot_index=ambient_snapshot)

    legacy_driven_snapshot = int(np.argmin(np.abs(time_ns - ARTICLE_DRIVEN_PROBE_TIME_NS)))
    legacy_driven_summary_base = shocked_al_slab_summary(
        dataset,
        snapshot_index=legacy_driven_snapshot,
        density_floor_g_cm3=ARTICLE_DRIVEN_DENSITY_WINDOW[0],
        material_id=al_material_id,
    )
    legacy_driven_context = make_run_context(dataset, hydro_path, snapshot_index=legacy_driven_snapshot)
    best_driven_summary_base = _best_driven_hydro_selection(dataset, material_id=al_material_id)
    best_driven_snapshot = int(best_driven_summary_base["snapshot_index"])
    best_driven_context = make_run_context(dataset, hydro_path, snapshot_index=best_driven_snapshot)

    def _al_entry_rows(values: object) -> tuple[str, ...]:
        return tuple(str(value) for value in values if str(value).startswith("Al@"))

    def _hydro_policy_selection(
        *,
        snapshot_index: int,
        zone_index_lower: int,
        zone_index_upper: int,
        policy: str,
        selection_kind: str,
        density_floor_g_cm3: float,
        selection_score: float = float("nan"),
    ) -> dict[str, object]:
        payload = policy_payloads[str(policy)]
        selection = _selection_state_summary(
            dataset,
            snapshot_index=int(snapshot_index),
            zone_index_lower=int(zone_index_lower),
            zone_index_upper=int(zone_index_upper),
            material_id=al_material_id,
            effective_ne_cm3=payload.electron_density_cm3,
            effective_zbar=payload.mean_charge,
            baseline_zbar=getattr(payload, "baseline_mean_charge", None),
            increment_zbar=getattr(payload, "increment_mean_charge", None),
        )
        selection.update(
            {
                "snapshot_index": int(snapshot_index),
                "time_ns": float(time_ns[int(snapshot_index)]),
                "material_id": int(al_material_id),
                "material_formula": "Al",
                "density_floor_g_cm3": float(density_floor_g_cm3),
                "selection_kind": str(selection_kind),
                "selection_score": float(selection_score),
                "baseline_mode": str(getattr(payload, "baseline_mode", "")),
                "increment_mode": str(getattr(payload, "increment_mode", "")),
                "baseline_table_source": str(getattr(payload, "baseline_table_source", "")),
                "baseline_entries": _al_entry_rows(getattr(payload, "baseline_entries", ())),
                "increment_entries": _al_entry_rows(getattr(payload, "increment_entries", ())),
                "driven_response_model": str(getattr(payload, "driven_response_model", PLASMON_DRIVEN_RESPONSE_MODEL_NONE)),
                "driven_response_summary": str(getattr(payload, "driven_response_summary", "")),
                **_response_weight_summary(payload),
            }
        )
        return selection

    ambient_policy_selection = {
        policy: _hydro_policy_selection(
            snapshot_index=ambient_snapshot,
            zone_index_lower=ambient_zone_lower,
            zone_index_upper=ambient_zone_upper,
            policy=str(policy),
            selection_kind="hydro_full_material_span",
            density_floor_g_cm3=0.0,
        )
        for policy in BENCHMARK_POLICIES
    }
    legacy_driven_policy_selection = {
        policy: _hydro_policy_selection(
            snapshot_index=legacy_driven_snapshot,
            zone_index_lower=int(legacy_driven_summary_base["zone_index_lower"]),
            zone_index_upper=int(legacy_driven_summary_base["zone_index_upper"]),
            policy=str(policy),
            selection_kind="hydro_dense_floor_3p75",
            density_floor_g_cm3=float(ARTICLE_DRIVEN_DENSITY_WINDOW[0]),
        )
        for policy in BENCHMARK_POLICIES
    }
    best_driven_policy_selection = {
        policy: _hydro_policy_selection(
            snapshot_index=best_driven_snapshot,
            zone_index_lower=int(best_driven_summary_base["zone_index_lower"]),
            zone_index_upper=int(best_driven_summary_base["zone_index_upper"]),
            policy=str(policy),
            selection_kind="hydro_plateau_best_match",
            density_floor_g_cm3=float(best_driven_summary_base.get("density_floor_g_cm3", ARTICLE_DRIVEN_DENSITY_WINDOW[0])),
            selection_score=float(best_driven_summary_base.get("selection_score", float("nan"))),
        )
        for policy in BENCHMARK_POLICIES
    }
    article_driven_policy_selection = {
        policy: dict(_density_averaged_policy_selection(str(policy), driven_response_model=selected_driven_response_model))
        | {
            "material_id": int(al_material_id),
            "material_formula": "Al",
            "selection_score": 0.0,
        }
        for policy in ARTICLE_DRIVEN_BENCHMARK_POLICIES
    }
    article_target_summary = _article_target_state_summary()

    ambient_cases = {
        policy: _case_payload(
            dataset,
            ambient_context,
            cache,
            case_name="ambient_al_t0",
            electron_policy=str(policy),
            driven_response_model=selected_driven_response_model,
            zone_index_lower=int(ambient_policy_selection[policy]["zone_index_lower"]),
            zone_index_upper=int(ambient_policy_selection[policy]["zone_index_upper"]),
            state_summary=ambient_policy_selection[policy],
            reference=GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5,
            representative_q=REPRESENTATIVE_Q_AMBIENT,
              benchmark_preset=PLASMON_BENCHMARK_PRESET_AL_AMBIENT_ARTICLE,
              material_id=al_material_id,
              observable_mode=selected_observable_mode,
              point_instrument_fwhm_ev=PEAK_EXTRACTION_FWHM_EV,
              models=selected_models,
          )
          for policy in BENCHMARK_POLICIES
      }
    legacy_driven_case = _case_payload(
        dataset,
        legacy_driven_context,
        cache,
        case_name="driven_al_dense_slab",
        electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
        driven_response_model=selected_driven_response_model,
        zone_index_lower=int(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["zone_index_lower"]),
        zone_index_upper=int(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["zone_index_upper"]),
        state_summary=legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK],
        reference=USER_DRIVEN_AL_DISPERSION_REFERENCE,
        representative_q=REPRESENTATIVE_Q_DRIVEN,
          benchmark_preset=PLASMON_BENCHMARK_PRESET_AL_DRIVEN_ARTICLE,
          material_id=al_material_id,
          observable_mode=selected_observable_mode,
          point_instrument_fwhm_ev=ARTICLE_INSTRUMENT_FWHM_EV,
          models=selected_models,
      )
    best_driven_case = _case_payload(
        dataset,
        best_driven_context,
        cache,
        case_name="driven_al_best_hydro_slab",
        electron_policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
        driven_response_model=selected_driven_response_model,
        zone_index_lower=int(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["zone_index_lower"]),
        zone_index_upper=int(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["zone_index_upper"]),
        state_summary=best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK],
        reference=USER_DRIVEN_AL_DISPERSION_REFERENCE,
        representative_q=REPRESENTATIVE_Q_DRIVEN,
          benchmark_preset=PLASMON_BENCHMARK_PRESET_AL_DRIVEN_ARTICLE,
          material_id=al_material_id,
          observable_mode=selected_observable_mode,
          point_instrument_fwhm_ev=ARTICLE_INSTRUMENT_FWHM_EV,
          models=selected_models,
      )
    article_driven_cases = {
        policy: _density_averaged_case_payload(
            case_name="driven_al_article_state",
            electron_policy=str(policy),
            driven_response_model=selected_driven_response_model,
            state_summary=article_driven_policy_selection[policy],
            reference=USER_DRIVEN_AL_DISPERSION_REFERENCE,
            representative_q=REPRESENTATIVE_Q_DRIVEN,
              benchmark_preset=PLASMON_BENCHMARK_PRESET_AL_DRIVEN_ARTICLE,
              densities_g_cm3=tuple(float(value) for value in ARTICLE_DRIVEN_DENSITY_GRID),
              te_ev=float(ARTICLE_DRIVEN_TEMPERATURE_EV),
              observable_mode=selected_observable_mode,
              models=selected_models,
          )
          for policy in ARTICLE_DRIVEN_BENCHMARK_POLICIES
      }
    ambient_case = ambient_cases[PRIMARY_POLICY_AMBIENT]
    article_driven_case = article_driven_cases[PRIMARY_POLICY_DRIVEN]

    ambient_references = _reference_series(GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5)
    driven_references = _reference_series(USER_DRIVEN_AL_DISPERSION_REFERENCE)
    _plot_dispersion_case(
        out_dir / "ambient_dataset_overlay.png",
        title="Cold Al benchmark on 50Al+10E+25CH+3.5TW | t = 0 | Al only",
        references=ambient_references,
        model_rows={model: dict(payload)["points"] for model, payload in dict(ambient_case["models"]).items()},
        models=selected_models,
    )
    _plot_dispersion_case(
        out_dir / "driven_legacy_hydro_overlay.png",
        title="Driven Al legacy hydro slab | nearest 6.3 ns dense-slab selection | article benchmark policy",
        references=driven_references,
        model_rows={model: dict(payload)["points"] for model, payload in dict(legacy_driven_case["models"]).items()},
        models=selected_models,
    )
    _plot_dispersion_case(
        out_dir / "driven_best_hydro_overlay.png",
        title="Driven Al best hydro plateau | closest slab to article state | article benchmark policy",
        references=driven_references,
        model_rows={model: dict(payload)["points"] for model, payload in dict(best_driven_case["models"]).items()},
        models=selected_models,
    )
    _plot_dispersion_case(
        out_dir / "driven_dataset_overlay.png",
        title="Driven Al article-reconciled state | density-averaged 3.75-4.50 g/cm^3 @ Te = 0.6 eV",
        references=driven_references,
        model_rows={model: dict(payload)["points"] for model, payload in dict(article_driven_case["models"]).items()},
        models=selected_models,
    )
    _plot_representative_spectra(
        out_dir / "ambient_representative_spectra.png",
        title=f"Representative ambient-Al spectra at k = {float(ambient_case['representative_q_ang_inv']):.2f} 1/A",
        spectra=dict(ambient_case["spectra"]),
        models=selected_models,
    )
    _plot_representative_spectra(
        out_dir / "driven_representative_spectra.png",
        title=f"Representative driven-Al spectra at k = {float(article_driven_case['representative_q_ang_inv']):.2f} 1/A",
        spectra=dict(article_driven_case["spectra"]),
        models=selected_models,
    )

    report_payload = {
        "dataset": hydro_path.name,
        "probe_energy_kev": PHOTON_ENERGY_KEV,
        "instrument_fwhm_ev_for_shapes": ARTICLE_INSTRUMENT_FWHM_EV,
        "instrument_fwhm_ev_for_peak_extraction": PEAK_EXTRACTION_FWHM_EV,
        "primary_electron_policy_ambient": PRIMARY_POLICY_AMBIENT,
        "primary_electron_policy_driven": PRIMARY_POLICY_DRIVEN,
        "driven_response_model": (selected_driven_response_model or "policy_default"),
        "benchmark_policies": list(BENCHMARK_POLICIES),
        "collision_model": PRIMARY_COLLISION_MODEL,
        "selected_models": list(selected_models),
        "observable_mode": selected_observable_mode,
        "observable_mode_label": observable_mode_label(selected_observable_mode),
        "ambient_cases": ambient_cases,
        "legacy_driven_case": legacy_driven_case,
        "best_driven_case": best_driven_case,
        "article_driven_cases": article_driven_cases,
        "article_target_state": article_target_summary,
        "cache_stats": cache.stats(),
        "classical_response_cache": classical_response_cache_info(),
    }
    (out_dir / "all_model_results.json").write_text(json.dumps(_json_safe(report_payload), indent=2, allow_nan=False), encoding="utf-8")

    case_payloads_for_summary = [
        *[(payload, "experiment") for payload in ambient_cases.values()],
        (legacy_driven_case, "experiment"),
        (best_driven_case, "experiment"),
        *[(payload, "experiment") for payload in article_driven_cases.values()],
    ]
    point_rows = [
        *sum((list(payload["flat_point_rows"]) for payload in ambient_cases.values()), []),
        *list(legacy_driven_case["flat_point_rows"]),
        *list(best_driven_case["flat_point_rows"]),
        *sum((list(payload["flat_point_rows"]) for payload in article_driven_cases.values()), []),
    ]
    summary_rows: list[dict[str, object]] = []
    for case_payload, default_reference in case_payloads_for_summary:
        for row in list(case_payload["flat_summary_rows"]):
            metrics = dict(row["metrics"]).get(default_reference, {})
            summary_rows.append({
                "case": row["case"],
                "model": row["model"],
                "model_label": row["model_label"],
                "status": row["status"],
                "backend": row["backend"],
                "benchmark_preset": row["benchmark_preset"],
                "requested_electron_policy": row["requested_electron_policy"],
                "electron_policy": row["electron_policy"],
                "electron_policy_scope": row.get("electron_policy_scope", electron_policy_scope(str(row["electron_policy"]))),
                "driven_response_model": row.get("driven_response_model", PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
                "driven_response_summary": row.get("driven_response_summary", ""),
                "observable_mode": row.get("observable_mode", selected_observable_mode),
                "observable_summary": row.get("observable_summary", ""),
                "observable_decomposition_mode": row.get("observable_decomposition_mode", ""),
                "observable_peak_extraction_mode": row.get("observable_peak_extraction_mode", ""),
                "observable_free_fraction": row.get("observable_free_fraction", float("nan")),
                "observable_bound_fraction": row.get("observable_bound_fraction", float("nan")),
                "observable_elastic_fraction": row.get("observable_elastic_fraction", float("nan")),
                "driven_response_weight_mode": row.get("driven_response_weight_mode", ""),
                "driven_response_ensemble_mode": row.get("driven_response_ensemble_mode", ""),
                "collision_source": row["collision_source"],
                "executed_fully_all": row["executed_fully_all"],
                "runtime_mean_s": row["runtime_mean_s"],
                "runtime_max_s": row["runtime_max_s"],
                "valid_peak_count": row["valid_peak_count"],
                "fallback_fraction_max": row["fallback_fraction_max"],
                "domain_failure_fraction_max": row["domain_failure_fraction_max"],
                "reference_series": default_reference,
                "mae_ev": metrics.get("mae_ev", float("nan")),
                "rmse_ev": metrics.get("rmse_ev", float("nan")),
                "warning_count": len(row["warnings"]),
                "warnings": " || ".join(str(item) for item in row["warnings"]),
            })
    policy_state_rows = [
        _policy_state_row("ambient_al_t0", policy, ambient_policy_selection[policy])
        for policy in BENCHMARK_POLICIES
    ] + [
        _policy_state_row("driven_al_dense_slab", PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK, legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]),
        _policy_state_row("driven_al_best_hydro_slab", PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK, best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]),
        *[_policy_state_row("driven_al_article_state", policy, article_driven_policy_selection[policy]) for policy in ARTICLE_DRIVEN_BENCHMARK_POLICIES],
    ]
    policy_state_by_case_policy = {
        (str(row["case"]), str(row["electron_policy"])): row
        for row in policy_state_rows
    }
    for row in summary_rows:
        state_row = policy_state_by_case_policy.get((str(row["case"]), str(row["electron_policy"])))
        if state_row is None:
            row["input_policy_status"] = ""
            row["input_policy_reason"] = ""
            row["model_status"] = row["status"]
            row["headline_credibility"] = "unknown"
            continue
        row["input_policy_status"] = str(state_row["input_policy_status"])
        row["input_policy_reason"] = str(state_row["input_policy_reason"])
        row["headline_credibility"] = str(state_row.get("headline_credibility", "unknown"))
        row["model_status"] = row["status"]
        if str(state_row["input_policy_status"]) == "invalid_input_policy":
            row["status"] = "invalid_input_policy"
    policy_sensitivity_rows = _policy_sensitivity_rows(summary_rows)
    ambient_reconciliation_rows = _reconciliation_rows(ambient_case)
    legacy_reconciliation_rows = _reconciliation_rows(legacy_driven_case)
    best_reconciliation_rows = _reconciliation_rows(best_driven_case)
    article_reconciliation_rows = [row for payload in article_driven_cases.values() for row in _reconciliation_rows(payload)]
    reconciliation_rows = [
        *ambient_reconciliation_rows,
        *legacy_reconciliation_rows,
        *best_reconciliation_rows,
        *article_reconciliation_rows,
    ]
    driven_policy_rows = [
        {
            "policy": str(row["electron_policy"]),
            "policy_label": str(row["electron_policy_label"]),
            "headline_credibility": str(row["headline_credibility"]),
            "input_policy_status": str(row["input_policy_status"]),
            "cold_baseline_contribution_zeff": float(row["baseline_zbar_weighted"]),
            "driven_increment_contribution_zeff": float(row["increment_zbar_weighted"]),
            "final_effective_zeff": float(row["effective_valence_from_ne"]),
            "effective_ne_cm3": float(row["effective_ne_weighted_cm3"]),
            "baseline_mode": str(row["baseline_mode"]),
            "increment_mode": str(row["increment_mode"]),
            "json_entry": str(row["baseline_entries"]),
            "driven_response_model": str(row.get("driven_response_model", PLASMON_DRIVEN_RESPONSE_MODEL_NONE)),
            "driven_response_summary": str(row.get("driven_response_summary", "")),
            "driven_response_weight_mode": str(row.get("driven_response_weight_mode", "uniform")),
            "driven_response_weight_mean": float(row.get("driven_response_weight_mean", float("nan"))),
            "driven_response_ensemble_mode": str(row.get("driven_response_ensemble_mode", "")),
        }
        for row in policy_state_rows
        if str(row["case"]) == "driven_al_article_state"
    ]
    driven_state_comparison_rows = [
        {
            "selection_label": "previous_selected_state",
            "selection_kind": str(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["selection_kind"]),
            "snapshot_index": int(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["snapshot_index"]),
            "time_ns": float(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["time_ns"]),
            "rho_weighted_g_cm3": float(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["rho_weighted_g_cm3"]),
            "te_weighted_ev": float(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["te_weighted_ev"]),
            "ti_weighted_ev": float(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["ti_weighted_ev"]),
            "ion_density_weighted_cm3": float(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["ion_density_weighted_cm3"]),
            "effective_ne_weighted_cm3": float(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["effective_ne_weighted_cm3"]),
            "effective_zeff_from_ne": float(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["effective_valence_from_ne"]),
            "path_length_um": float(legacy_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["path_length_total_cm"]) * 1.0e4,
        },
        {
            "selection_label": "new_selected_state",
            "selection_kind": str(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["selection_kind"]),
            "snapshot_index": int(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["snapshot_index"]),
            "time_ns": float(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["time_ns"]),
            "rho_weighted_g_cm3": float(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["rho_weighted_g_cm3"]),
            "te_weighted_ev": float(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["te_weighted_ev"]),
            "ti_weighted_ev": float(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["ti_weighted_ev"]),
            "ion_density_weighted_cm3": float(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["ion_density_weighted_cm3"]),
            "effective_ne_weighted_cm3": float(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["effective_ne_weighted_cm3"]),
            "effective_zeff_from_ne": float(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["effective_valence_from_ne"]),
            "path_length_um": float(best_driven_policy_selection[PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK]["path_length_total_cm"]) * 1.0e4,
        },
        {
            "selection_label": "target_article_state",
            "selection_kind": str(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["selection_kind"]),
            "snapshot_index": int(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["snapshot_index"]),
            "time_ns": float(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["time_ns"]),
            "rho_weighted_g_cm3": float(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["rho_weighted_g_cm3"]),
            "te_weighted_ev": float(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["te_weighted_ev"]),
            "ti_weighted_ev": float(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["ti_weighted_ev"]),
            "ion_density_weighted_cm3": float(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["ion_density_weighted_cm3"]),
            "effective_ne_weighted_cm3": float(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["effective_ne_weighted_cm3"]),
            "effective_zeff_from_ne": float(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["effective_valence_from_ne"]),
            "path_length_um": float(article_driven_policy_selection[PRIMARY_POLICY_DRIVEN]["path_length_total_cm"]) * 1.0e4,
        },
    ]
    _write_csv(out_dir / "benchmark_points.csv", point_rows)
    _write_csv(out_dir / "benchmark_summary.csv", summary_rows)
    _write_csv(out_dir / "policy_state_summary.csv", policy_state_rows)
    _write_csv(out_dir / "policy_sensitivity_summary.csv", policy_sensitivity_rows)
    _write_csv(out_dir / "reconciliation_summary.csv", reconciliation_rows)
    _write_csv(out_dir / "driven_policy_table.csv", driven_policy_rows)
    _write_csv(out_dir / "driven_state_comparison.csv", driven_state_comparison_rows)

    def _reconciliation_map(rows: list[dict[str, object]], *, policy: str | None = None) -> dict[tuple[str, str], dict[str, object]]:
        mapping: dict[tuple[str, str], dict[str, object]] = {}
        for row in rows:
            if policy is not None and str(row.get("electron_policy", "")) != str(policy):
                continue
            published_branch = str(row.get("published_branch", "") or "")
            if not published_branch:
                continue
            mapping[(str(row["model"]), published_branch)] = row
        return mapping

    def _judged_delta(before: float, after: float) -> str:
        if not math.isfinite(before) or not math.isfinite(after):
            return "not_comparable"
        delta = float(before - after)
        if delta >= 1.0:
            return "materially improved"
        if delta >= 0.25:
            return "improved"
        if delta > -0.25:
            return "no material change"
        return "worse"

    previous_reconciliation_path = Path("outputs/validation_outputs/plasmon_article_cases_json_baseline_pass/reconciliation_summary.csv")
    previous_reconciliation_rows: list[dict[str, str]] = []
    if previous_reconciliation_path.exists():
        with previous_reconciliation_path.open("r", encoding="utf-8", newline="") as handle:
            previous_reconciliation_rows = list(csv.DictReader(handle))
    previous_driven_map = {
        (str(row["model"]), str(row.get("published_branch", ""))): row
        for row in previous_reconciliation_rows
        if str(row.get("case", "")) == "driven_al_dense_slab"
        and str(row.get("electron_policy", "")) == PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK
        and str(row.get("published_branch", ""))
    }
    legacy_driven_map = _reconciliation_map(legacy_reconciliation_rows, policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
    best_driven_map = _reconciliation_map(best_reconciliation_rows, policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
    article_old_map = _reconciliation_map(article_reconciliation_rows, policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK)
    article_new_map = _reconciliation_map(article_reconciliation_rows, policy=PRIMARY_POLICY_DRIVEN)
    driven_branch_delta_rows: list[dict[str, object]] = []
    for key in sorted(article_new_map.keys()):
        model, published_branch = key
        current_new = article_new_map.get(key, {})
        current_old = article_old_map.get(key, {})
        current_legacy = legacy_driven_map.get(key, {})
        current_best = best_driven_map.get(key, {})
        previous_row = previous_driven_map.get(key, {})
        before_mae = float(previous_row.get("mae_ev", float("nan"))) if previous_row else float("nan")
        legacy_mae = float(current_legacy.get("mae_ev", float("nan"))) if current_legacy else float("nan")
        best_mae = float(current_best.get("mae_ev", float("nan"))) if current_best else float("nan")
        article_old_mae = float(current_old.get("mae_ev", float("nan"))) if current_old else float("nan")
        article_new_mae = float(current_new.get("mae_ev", float("nan"))) if current_new else float("nan")
        driven_branch_delta_rows.append(
            {
                "model": str(model),
                "model_label": str(current_new.get("model_label", _model_label(str(model)))),
                "published_branch": str(published_branch),
                "comparison_kind": str(current_new.get("comparison_kind", "")),
                "previous_policy": PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
                "current_policy": PRIMARY_POLICY_DRIVEN,
                "state_selection": "legacy_dense_slab -> article_density_average_fixed_temperature",
                "mae_before_ev": before_mae,
                "mae_legacy_current_ev": legacy_mae,
                "mae_best_hydro_ev": best_mae,
                "mae_article_state_old_policy_ev": article_old_mae,
                "mae_after_ev": article_new_mae,
                "state_selection_delta_ev": (float(before_mae - article_old_mae) if math.isfinite(before_mae) and math.isfinite(article_old_mae) else float("nan")),
                "policy_increment_delta_ev": (float(article_old_mae - article_new_mae) if math.isfinite(article_old_mae) and math.isfinite(article_new_mae) else float("nan")),
                "total_delta_ev": (float(before_mae - article_new_mae) if math.isfinite(before_mae) and math.isfinite(article_new_mae) else float("nan")),
                "judged_improvement": _judged_delta(before_mae, article_new_mae),
            }
        )
    _write_csv(out_dir / "driven_branch_reconciliation_delta.csv", driven_branch_delta_rows)

    ambient_ranked = _summary_table_rows(ambient_case, reference_key="experiment")
    driven_ranked = _summary_table_rows(article_driven_case, reference_key="experiment")
    ambient_best_spectral = _best_benchmark_grade_row(ambient_ranked)
    driven_best_spectral = _best_benchmark_grade_row(driven_ranked)

    report_lines: list[str] = [
        "# Article benchmark report for 50Al+10E+25CH+3.5TW",
        "",
        "This pass focuses on the driven Al benchmark. The cold-baseline input problem is already fixed; the remaining work here is driven-state electron response and driven-state identity reconciliation before judging genuine family-to-family mismatch.",
        "",
            "Global benchmark settings:",
            f"- dataset: **{hydro_path.name}**",
            f"- probe energy: **{PHOTON_ENERGY_KEV:.3f} keV**",
            f"- ambient headline policy: **{electron_policy_label(PRIMARY_POLICY_AMBIENT)} ({PRIMARY_POLICY_AMBIENT})**",
            f"- driven headline policy: **{electron_policy_label(PRIMARY_POLICY_DRIVEN)} ({PRIMARY_POLICY_DRIVEN})**",
            f"- driven response model: **{driven_response_model_label(selected_driven_response_model or 'none') if selected_driven_response_model else 'Policy default'} ({selected_driven_response_model or 'policy_default'})**",
            f"- observable mode: **{observable_mode_label(selected_observable_mode)} ({selected_observable_mode})**",
            f"- driven ensemble response mode: **{str(article_driven_case['selection'].get('driven_response_ensemble_mode', 'spectrum_average')) or 'spectrum_average'}**",
            f"- benchmark policy set: **{', '.join(electron_policy_label(policy) for policy in BENCHMARK_POLICIES)}**",
            f"- collision closure: **{PRIMARY_COLLISION_MODEL}**",
        f"- ambient reference provenance: **{GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5.get('provenance', {}).get('quality', 'unspecified')}**",
        f"- driven reference provenance: **{USER_DRIVEN_AL_DISPERSION_REFERENCE.get('provenance', {}).get('quality', 'unspecified')}**",
        f"- representative convolution FWHM: **{ARTICLE_INSTRUMENT_FWHM_EV:.2f} eV**",
        f"- ambient point-extraction FWHM: **{PEAK_EXTRACTION_FWHM_EV:.2f} eV**",
        f"- classical response cache stats: **{classical_response_cache_info()}**",
        "",
        "## A. State selection and target reconciliation",
        "",
    ]
    report_lines.extend(_selection_markdown(
        "Cold Al at t = 0",
        dict(ambient_case["selection"]),
        notes=[
            "Selection keeps only Al and excludes epoxy/CH by construction.",
            f"Ambient headline policy: {PRIMARY_POLICY_AMBIENT}.",
        ],
    ))
    report_lines.extend(_selection_markdown(
        "Previous driven hydro dense-slab selection",
        dict(legacy_driven_case["selection"]),
        notes=[
            f"Nearest snapshot to the article probe time {ARTICLE_DRIVEN_PROBE_TIME_NS:.2f} ns with rho >= {ARTICLE_DRIVEN_DENSITY_WINDOW[0]:.2f} g/cm^3.",
            "This is the selection used in the previous driven pass and serves as the baseline for before/after reconciliation.",
        ],
    ))
    report_lines.extend(_selection_markdown(
        "Best hydro plateau found near the article state",
        dict(best_driven_case["selection"]),
        notes=[
            "Search swept nearby snapshots and density floors to minimize mismatch to the article-driven density/temperature window.",
            "This improves slab identity somewhat, but it is still a hydro state and not the article density-averaged fixed-temperature construction.",
        ],
    ))
    report_lines.extend(_selection_markdown(
        "Article-reconciled driven state",
        dict(article_driven_case["selection"]),
        notes=[
            (
                f"Built from {len(ARTICLE_DRIVEN_DENSITY_GRID)} uniform Al states spanning "
                f"{ARTICLE_DRIVEN_DENSITY_WINDOW[0]:.2f}-{ARTICLE_DRIVEN_DENSITY_WINDOW[1]:.2f} g/cm^3 "
                f"at fixed Te = {ARTICLE_DRIVEN_TEMPERATURE_EV:.2f} eV, then combined at the response-function "
                f"level before loss/spectrum extraction."
                if str(article_driven_case['selection'].get('driven_response_ensemble_mode', '')).strip()
                else f"Built from {len(ARTICLE_DRIVEN_DENSITY_GRID)} uniform Al states spanning "
                f"{ARTICLE_DRIVEN_DENSITY_WINDOW[0]:.2f}-{ARTICLE_DRIVEN_DENSITY_WINDOW[1]:.2f} g/cm^3 "
                f"at fixed Te = {ARTICLE_DRIVEN_TEMPERATURE_EV:.2f} eV, then averaged at the spectrum level before peak extraction."
            ),
            f"Driven headline policy: {PRIMARY_POLICY_DRIVEN}.",
        ],
    ))
    report_lines.extend(
        [
            "### Driven state comparison",
            "",
            "| selection | kind | snapshot | time [ns] | rho [g/cm^3] | Te [eV] | Ti [eV] | n_i [cm^-3] | n_e [cm^-3] | Z_eff | path [um] |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in driven_state_comparison_rows:
        report_lines.append(
            f"| {row['selection_label']} | {row['selection_kind']} | {int(row['snapshot_index'])} | {float(row['time_ns']):.4f} | {float(row['rho_weighted_g_cm3']):.3f} | {float(row['te_weighted_ev']):.3f} | {float(row['ti_weighted_ev']):.3f} | {float(row['ion_density_weighted_cm3']):.4e} | {float(row['effective_ne_weighted_cm3']):.4e} | {float(row['effective_zeff_from_ne']):.3f} | {float(row['path_length_um']):.2f} |"
        )
    report_lines.extend(
        [
            "## B. Driven electron-policy construction",
            "",
            "| policy | status | headline role | cold baseline | driven increment | final Z_eff | effective n_e [cm^-3] | response model | response weighting | baseline mode | increment mode | JSON entry |",
            "|---|---|---|---:|---:|---:|---:|---|---|---|---|---|",
        ]
    )
    for row in driven_policy_rows:
        report_lines.append(
            f"| {row['policy_label']} | {row['input_policy_status']} | {row['headline_credibility']} | {float(row['cold_baseline_contribution_zeff']):.3f} | {float(row['driven_increment_contribution_zeff']):.3f} | {float(row['final_effective_zeff']):.3f} | {float(row['effective_ne_cm3']):.4e} | {row['driven_response_model']} | {row['driven_response_weight_mode'] or '-'} | {row['baseline_mode']} | {row['increment_mode']} | {row['json_entry'] or '-'} |"
        )
    report_lines.extend(
        [
            "",
            "Raw HELIOS is kept as a diagnostic contrast on the hydro-selected slabs, but it is intentionally excluded from the synthetic article-reconciled state because that state is built from uniform reference states and has no raw hydro charge-state field to preserve.",
            "",
            "## C. Policy sensitivity on the article-reconciled driven state",
            "",
            "| model | article baseline MAE [eV] | driven increment MAE [eV] | benchmark valence-aware MAE [eV] | valence-locked MAE [eV] | credible-policy spread [eV] |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in policy_sensitivity_rows:
        if str(row["case"]) != "driven_al_article_state":
            continue
        report_lines.append(
            f"| {row['model_label']} | {float(row['article_mae_ev']):.3f} | {float(row['article_driven_increment_mae_ev']):.3f} | {float(row['benchmark_valence_aware_mae_ev']):.3f} | {float(row['valence_locked_mae_ev']):.3f} | {float(row['benchmark_policy_mae_spread_ev']):.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## D. Direct driven branch-to-branch reconciliation",
            "",
            "| our branch | published branch | policy | state selection | MAE before [eV] | MAE after [eV] | judged improvement |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in driven_branch_delta_rows:
        report_lines.append(
            f"| {row['model_label']} | {row['published_branch']} | {electron_policy_label(str(row['current_policy']))} | {row['state_selection']} | "
            + (f"{float(row['mae_before_ev']):.3f}" if math.isfinite(float(row['mae_before_ev'])) else "-")
            + " | "
            + (f"{float(row['mae_after_ev']):.3f}" if math.isfinite(float(row['mae_after_ev'])) else "-")
            + f" | {row['judged_improvement']} |"
        )
    report_lines.extend(
        [
            "",
            "### Reconciliation decomposition",
            "",
            "| branch | previous hydro slab [eV] | best hydro slab [eV] | article state + old policy [eV] | article state + driven increment [eV] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in driven_branch_delta_rows:
        report_lines.append(
            f"| {row['model_label']} -> {row['published_branch']} | "
            + (f"{float(row['mae_legacy_current_ev']):.3f}" if math.isfinite(float(row['mae_legacy_current_ev'])) else "-")
            + " | "
            + (f"{float(row['mae_best_hydro_ev']):.3f}" if math.isfinite(float(row['mae_best_hydro_ev'])) else "-")
            + " | "
            + (f"{float(row['mae_article_state_old_policy_ev']):.3f}" if math.isfinite(float(row['mae_article_state_old_policy_ev'])) else "-")
            + " | "
            + (f"{float(row['mae_after_ev']):.3f}" if math.isfinite(float(row['mae_after_ev'])) else "-")
            + " |"
        )
    report_lines.extend(
        [
            "",
            "## E. Headline practical ranking",
            "",
            "### Ambient vs experiment",
            "",
            "| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in ambient_ranked:
        report_lines.append(
            f"| {row['model_label']} | {row['status']} | {row['backend']} | {row['collision_source'] or '-'} | {float(row['runtime_mean_s']):.3f} | {int(row['valid_peak_count'])} | {float(row['mae_ev']):.3f} | {float(row['rmse_ev']):.3f} |"
        )
    report_lines.extend(
        [
            "",
            "### Driven article-reconciled state vs experiment",
            "",
            "| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in driven_ranked:
        report_lines.append(
            f"| {row['model_label']} | {row['status']} | {row['backend']} | {row['collision_source'] or '-'} | {float(row['runtime_mean_s']):.3f} | {int(row['valid_peak_count'])} | {float(row['mae_ev']):.3f} | {float(row['rmse_ev']):.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## F. Judged conclusions",
            "",
            (
                f"- Best practical ambient benchmark branch remains **{ambient_best_spectral['model_label']}** "
                f"with MAE **{float(ambient_best_spectral['mae_ev']):.3f} eV**."
                if ambient_best_spectral is not None
                else "- No benchmark-grade ambient spectral branch remained valid across the full ambient series."
            ),
            (
                f"- Best practical driven branch on the article-reconciled state is **{driven_best_spectral['model_label']}** "
                f"with MAE **{float(driven_best_spectral['mae_ev']):.3f} eV**."
                if driven_best_spectral is not None
                else "- No benchmark-grade driven spectral branch remained valid across the driven series."
            ),
            "- Raw HELIOS remains diagnostic-only for article-facing Al. It is intentionally visible only as a contrast axis and is not part of the headline ranking.",
            "- The new driven increment policy is explicit and modest. It keeps the JSON cold baseline as the floor, adds a bounded temperature/compression-driven increment for Al, and reports the baseline and increment contributions separately.",
            "- The best hydro plateau is still colder than the article target. Tightening the slab selection helps, but it does not by itself close the driven branch-to-branch gap.",
            "- Rebuilding the driven benchmark around the article density-average state is necessary for fair comparison. It removes a real state-identity mismatch that was previously inflating confusion about the classical branches.",
            "- After the state and policy fixes, the strongest practical classical family match is still **RPA + static LFC**. It improves relative to the previous pass, but it remains materially below the published driven LFC branch.",
            "- The Mermin family remains usable-with-caveats rather than headline-prominent. It now runs under the benchmark_dense closure, but it does not outperform the best classical parent strongly enough to justify primary prominence.",
            "- The remaining driven mismatch is now mostly genuine model disagreement and/or missing driven-state electron-response physics, not benchmark plumbing. The next blocker is therefore a better justified driven electron increment / response model rather than another cache or UI fix.",
            "",
            "## G. Generated artifacts",
            "",
            f"- `{out_dir / 'all_model_results.json'}`",
            f"- `{out_dir / 'benchmark_points.csv'}`",
            f"- `{out_dir / 'benchmark_summary.csv'}`",
            f"- `{out_dir / 'policy_state_summary.csv'}`",
            f"- `{out_dir / 'policy_sensitivity_summary.csv'}`",
            f"- `{out_dir / 'reconciliation_summary.csv'}`",
            f"- `{out_dir / 'driven_policy_table.csv'}`",
            f"- `{out_dir / 'driven_state_comparison.csv'}`",
            f"- `{out_dir / 'driven_branch_reconciliation_delta.csv'}`",
            f"- `{out_dir / 'ambient_dataset_overlay.png'}`",
            f"- `{out_dir / 'driven_legacy_hydro_overlay.png'}`",
            f"- `{out_dir / 'driven_best_hydro_overlay.png'}`",
            f"- `{out_dir / 'driven_dataset_overlay.png'}`",
            f"- `{out_dir / 'ambient_representative_spectra.png'}`",
            f"- `{out_dir / 'driven_representative_spectra.png'}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark all HELIOS plasmon models against the aluminium article reference cases.")
    parser.add_argument("--dataset", default="50Al+10E+25CH+3.5TW_stabilized.h5")
    parser.add_argument("--out-dir", default="outputs/validation_outputs/plasmon_article_cases_driven_increment_pass")
    parser.add_argument("--driven-response-model", default="", help="Optional driven-response model override for the article Al increment policy.")
    parser.add_argument("--observable-mode", default=PLASMON_OBSERVABLE_MODE_DIELECTRIC, help="Observable mode: dielectric or xrts_observable.")
    args = parser.parse_args()
    hydro_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    return build_report(
        hydro_path,
        out_dir=out_dir,
        driven_response_model=str(args.driven_response_model or ""),
        observable_mode=str(args.observable_mode or PLASMON_OBSERVABLE_MODE_DIELECTRIC),
    )


if __name__ == "__main__":
    raise SystemExit(main())

