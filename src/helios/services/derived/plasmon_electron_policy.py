"""Electron-density / effective-valence policies for plasmon benchmarking.

The default HELIOS workflow keeps the raw hydro electron-density and mean-charge
fields. Literature-facing benchmark policies need an explicit cold baseline for
conduction electrons so article comparisons do not silently depend on raw HELIOS
``ne/zbar`` in states where that mapping is not physically credible.

The authoritative cold-state baseline source for benchmark policies is the
repository-root JSON table ``hed_helios_cold_electronic_baseline_core.json``.
If a material is missing from that table, the benchmark policy keeps the raw
HELIOS fields and reports the unresolved material explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from helios.services.derived.models import DerivedRunData
from helios.services.derived.plasmon_driven_response import (
    PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
    DrivenElectronResponseState,
    apply_driven_response_model,
    default_driven_response_model_for_policy,
    normalize_driven_response_model,
)

NA = 6.02214076e23

PLASMON_ELECTRON_POLICY_RAW = "raw_helios"
PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE = "benchmark_valence_aware"
PLASMON_ELECTRON_POLICY_VALENCE_LOCKED = "valence_locked"
PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK = "article_al_benchmark"
PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT = "article_al_driven_increment"
PLASMON_BENCHMARK_POLICY_COMPARISON = (
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
    PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE,
    PLASMON_ELECTRON_POLICY_VALENCE_LOCKED,
)
PLASMON_POLICY_SENSITIVITY_SEQUENCE = (
    PLASMON_ELECTRON_POLICY_RAW,
    *PLASMON_BENCHMARK_POLICY_COMPARISON,
)

_BASELINE_JSON_NAME = "hed_helios_cold_electronic_baseline_core.json"


@dataclass(frozen=True, slots=True)
class MaterialPolicyEntry:
    formula: str
    benchmark_valence_per_nucleus: float
    atomic_number: int | None = None
    aliases: tuple[str, ...] = ()
    baseline_entry_path: str = ""
    driven_policy_default: str = "cold_plus_increment"
    confidence: str = ""
    notes: str = ""
    cold_electronic_state_class: str = ""


@dataclass(frozen=True, slots=True)
class ElectronPolicyPayload:
    requested_policy: str
    policy: str
    electron_density_cm3: np.ndarray
    mean_charge: np.ndarray
    source_label: str
    summary: str
    resolved_materials: tuple[str, ...] = ()
    unresolved_materials: tuple[str, ...] = ()
    raw_kept_materials: tuple[str, ...] = ()
    material_formula_map: dict[int, str] | None = None
    baseline_mode: str = ""
    baseline_entries: tuple[str, ...] = ()
    baseline_table_source: str = ""
    baseline_mean_charge: np.ndarray | None = None
    increment_mean_charge: np.ndarray | None = None
    increment_mode: str = ""
    increment_entries: tuple[str, ...] = ()
    driven_response_model: str = PLASMON_DRIVEN_RESPONSE_MODEL_NONE
    driven_response_summary: str = ""
    driven_response_weight_mode: str = ""
    driven_response_weight_multiplier: np.ndarray | None = None
    driven_response_shape_mode: str = ""
    driven_response_shape_fwhm_ev: np.ndarray | None = None
    driven_response_ensemble_mode: str = ""


_SUPPORTED_POLICIES = (
    PLASMON_ELECTRON_POLICY_RAW,
    PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE,
    PLASMON_ELECTRON_POLICY_VALENCE_LOCKED,
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
)

_POLICY_LABELS = {
    PLASMON_ELECTRON_POLICY_RAW: "Raw HELIOS ne/zbar",
    PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE: "Benchmark valence-aware",
    PLASMON_ELECTRON_POLICY_VALENCE_LOCKED: "Valence-locked benchmark",
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK: "Article Al benchmark",
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT: "Article Al + driven increment",
}
_POLICY_SCOPE = {
    PLASMON_ELECTRON_POLICY_RAW: "general_purpose",
    PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE: "benchmark_only",
    PLASMON_ELECTRON_POLICY_VALENCE_LOCKED: "benchmark_only",
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK: "benchmark_only",
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT: "benchmark_only",
}

_TOKEN_RE = re.compile(r"[\s\-]+")
_NON_TOKEN_RE = re.compile(r"[^a-z0-9_]+")

_FORMULA_ALIAS_HINTS: dict[str, tuple[str, ...]] = {
    "Al": ("al", "aluminium", "aluminum"),
    "Be": ("be", "beryllium"),
    "B": ("b", "boron"),
    "C": ("c", "carbon"),
    "Si": ("si", "silicon"),
    "Ge": ("ge", "germanium"),
    "Ti": ("ti", "titanium"),
    "Fe": ("fe", "iron"),
    "Ni": ("ni", "nickel"),
    "Cu": ("cu", "copper"),
    "Zr": ("zr", "zirconium"),
    "Mo": ("mo", "molybdenum"),
    "Ag": ("ag", "silver"),
    "Sn": ("sn", "tin"),
    "Ta": ("ta", "tantalum"),
    "Au": ("au", "gold"),
    "CH": ("ch", "plastic_ch", "ch_generic"),
    "CH2": ("ch2", "polyethylene", "ch2_generic"),
    "C2H4O": ("epoxy", "c2h4o", "epoxy_c2h4o", "epoxy_generic"),
    "SiO2": ("sio2", "silica", "glass", "glass_sio2", "sio2_generic"),
    "C22H10N2O5": ("kapton", "polyimide", "c22h10n2o5", "kapton_polyimide"),
    "carbon_graphitic": ("graphite", "carbon_graphitic"),
    "carbon_diamond_like": ("diamond", "diamond_like_carbon", "carbon_diamond_like"),
    "boron_semiconducting": ("boron_semiconducting",),
}

_FORMULA_JSON_PATHS: dict[str, tuple[str, str]] = {
    "CH": ("compound_and_polymer_baselines", "CH_generic"),
    "CH2": ("compound_and_polymer_baselines", "CH2_generic"),
    "C2H4O": ("compound_and_polymer_baselines", "Epoxy_generic"),
    "SiO2": ("compound_and_polymer_baselines", "SiO2_generic"),
    "C22H10N2O5": ("compound_and_polymer_baselines", "Kapton_polyimide"),
    "carbon_graphitic": ("special_material_overrides", "carbon_graphitic"),
    "carbon_diamond_like": ("special_material_overrides", "carbon_diamond_like"),
    "boron_semiconducting": ("special_material_overrides", "boron_semiconducting"),
}

_FALLBACK_ATOMIC_WEIGHTS_G_MOL: dict[str, float] = {
    "Al": 26.9815,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "Si": 28.085,
    "Ge": 72.630,
    "Ti": 47.867,
    "Fe": 55.845,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zr": 91.224,
    "Mo": 95.95,
    "Ag": 107.8682,
    "Sn": 118.710,
    "Ta": 180.94788,
    "Au": 196.96657,
    "CH": 13.0,
    "CH2": 14.0,
    "C2H4O": 44.0,
    "SiO2": 60.083,
    "C22H10N2O5": 382.33,
    "carbon_graphitic": 12.011,
    "carbon_diamond_like": 12.011,
    "boron_semiconducting": 10.81,
}


def supported_policies() -> tuple[str, ...]:
    return tuple(sorted(_SUPPORTED_POLICIES))


def policy_label(policy: str) -> str:
    return _POLICY_LABELS.get(str(policy), str(policy).replace("_", " ").title())


def policy_scope(policy: str) -> str:
    return _POLICY_SCOPE.get(str(policy), "general_purpose")


def policy_choices() -> tuple[tuple[str, str], ...]:
    return tuple((policy_label(value), value) for value in _SUPPORTED_POLICIES)


def benchmark_policy_choices(*, include_raw: bool = False) -> tuple[tuple[str, str], ...]:
    policies = PLASMON_POLICY_SENSITIVITY_SEQUENCE if include_raw else PLASMON_BENCHMARK_POLICY_COMPARISON
    return tuple((policy_label(value), value) for value in policies)


def normalize_policy(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _SUPPORTED_POLICIES else PLASMON_ELECTRON_POLICY_RAW


def _normalized_token(label: str | None) -> str:
    text = str(label or "").strip().lower()
    text = _TOKEN_RE.sub("_", text)
    text = _NON_TOKEN_RE.sub("_", text)
    text = re.sub("_+", "_", text).strip("_")
    return text


def _stem_token(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _normalized_token(Path(text).stem)


def _baseline_table_path() -> Path | None:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        candidate = parent / _BASELINE_JSON_NAME
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def _baseline_table_bundle() -> tuple[dict[str, Any], str]:
    path = _baseline_table_path()
    if path is None:
        return {}, f"{_BASELINE_JSON_NAME} (missing)"
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def _baseline_table_source() -> str:
    return _baseline_table_bundle()[1]


def _entry_from_json(
    *,
    formula: str,
    item: dict[str, Any],
    entry_path: str,
) -> MaterialPolicyEntry:
    aliases = _FORMULA_ALIAS_HINTS.get(str(formula), ())
    return MaterialPolicyEntry(
        formula=str(formula),
        benchmark_valence_per_nucleus=float(item.get("cold_benchmark_zeff_default", 0.0)),
        atomic_number=(int(item["atomic_number"]) if item.get("atomic_number") is not None else None),
        aliases=tuple(str(value) for value in aliases),
        baseline_entry_path=str(entry_path),
        driven_policy_default=str(item.get("driven_policy_default", "cold_plus_increment") or "cold_plus_increment"),
        confidence=str(item.get("confidence", "")),
        notes=str(item.get("notes", "")),
        cold_electronic_state_class=str(item.get("cold_electronic_state_class", "")),
    )


@lru_cache(maxsize=1)
def _json_backed_registry() -> dict[str, MaterialPolicyEntry]:
    data, _source = _baseline_table_bundle()
    registry: dict[str, MaterialPolicyEntry] = {}
    for formula, item in dict(data.get("elements", {})).items():
        if isinstance(item, dict):
            registry[str(formula)] = _entry_from_json(formula=str(formula), item=item, entry_path=f"elements.{formula}")
    for formula, path_tokens in _FORMULA_JSON_PATHS.items():
        section, entry_name = path_tokens
        section_map = dict(data.get(section, {}))
        item = section_map.get(entry_name)
        if isinstance(item, dict):
            registry[str(formula)] = _entry_from_json(formula=str(formula), item=item, entry_path=f"{section}.{entry_name}")
    return registry


def material_policy_registry() -> dict[str, MaterialPolicyEntry]:
    return dict(_json_backed_registry())


def material_atomic_number(formula: str | None) -> int | None:
    entry = _json_backed_registry().get(str(formula or "").strip())
    if entry is None:
        return None
    return None if entry.atomic_number is None else int(entry.atomic_number)


@lru_cache(maxsize=1)
def _alias_to_formula() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for formula, entry in _json_backed_registry().items():
        mapping[_normalized_token(formula)] = str(formula)
        for alias in entry.aliases:
            mapping[_normalized_token(alias)] = str(formula)
    return mapping


def _extract_material_field(materials: dict[str, Any], *candidates: str) -> np.ndarray | None:
    for name in candidates:
        value = materials.get(name)
        if value is None:
            continue
        return np.asarray(value, dtype=object)
    return None


def _resolve_formula_from_token(token: str) -> str | None:
    if not token:
        return None
    return _alias_to_formula().get(token)


def resolved_material_formula_map(dataset: DerivedRunData) -> dict[int, str]:
    materials = dict(dataset.materials)
    indices = np.asarray(materials.get("index", []), dtype=np.int32)
    explicit = _extract_material_field(materials, "chemical_formula", "formula", "composition", "label")
    eos_paths = _extract_material_field(materials, "eos_file_path")
    opacity_paths = _extract_material_field(materials, "opacity_file_path")
    mapping: dict[int, str] = {}
    for idx, material_id in enumerate(indices.tolist(), start=0):
        token = ""
        if explicit is not None and idx < explicit.size:
            token = _normalized_token(explicit[idx])
        if not token and eos_paths is not None and idx < eos_paths.size:
            token = _stem_token(eos_paths[idx])
        if not token and opacity_paths is not None and idx < opacity_paths.size:
            token = _stem_token(opacity_paths[idx])
        formula = _resolve_formula_from_token(token)
        if formula is not None:
            mapping[int(material_id)] = formula
    return mapping


def _mean_atomic_weight_for_formula(dataset: DerivedRunData, formula: str, material_mask: np.ndarray) -> float:
    zone_weights = np.asarray(dataset.zone_atomic_weight, dtype=np.float64)
    if zone_weights.ndim == 1 and zone_weights.shape == material_mask.shape:
        finite = material_mask & np.isfinite(zone_weights) & (zone_weights > 0.0)
        if np.any(finite):
            return float(np.nanmean(zone_weights[finite]))
    return float(_FALLBACK_ATOMIC_WEIGHTS_G_MOL.get(str(formula), float("nan")))


def _article_policy_formula_applies(formula: str) -> bool:
    return formula == "Al"


def _ion_density_cm3(density_g_cm3: np.ndarray, atomic_weight: float) -> np.ndarray:
    if not math.isfinite(float(atomic_weight)) or float(atomic_weight) <= 0.0:
        return np.full(np.asarray(density_g_cm3, dtype=np.float64).shape, np.nan, dtype=np.float64)
    return np.asarray(density_g_cm3, dtype=np.float64) * (NA / float(atomic_weight))


def _policy_mode_for_entry(requested_policy: str, entry: MaterialPolicyEntry) -> str:
    if requested_policy == PLASMON_ELECTRON_POLICY_RAW:
        return "raw_fields"
    if requested_policy == PLASMON_ELECTRON_POLICY_VALENCE_LOCKED:
        return "cold_baseline_only"
    if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT:
        return "cold_baseline_plus_benchmark_increment"
    driven_default = str(entry.driven_policy_default or "cold_plus_increment")
    if driven_default == "cold_plus_increment":
        return "cold_baseline_plus_increment"
    if driven_default == "cold_baseline_only":
        return "cold_baseline_only"
    return driven_default


def _resolved_entry_label(
    requested_policy: str,
    formula: str,
    entry: MaterialPolicyEntry,
    z_eff: np.ndarray,
    *,
    baseline_z_eff: np.ndarray,
    increment_z_eff: np.ndarray,
) -> str:
    mean_value = float(np.nanmean(np.asarray(z_eff, dtype=np.float64)))
    baseline_value = float(np.nanmean(np.asarray(baseline_z_eff, dtype=np.float64)))
    increment_value = float(np.nanmean(np.asarray(increment_z_eff, dtype=np.float64)))
    return (
        f"{formula}@{entry.baseline_entry_path}"
        f"(cold_Zeff={entry.benchmark_valence_per_nucleus:.3f}, mode={_policy_mode_for_entry(requested_policy, entry)})"
        f" -> baseline {baseline_value:.3f} + increment {increment_value:.3f} = {mean_value:.3f}"
    )


def resolve_effective_electron_fields(
    dataset: DerivedRunData,
    policy: str | None,
    *,
    driven_response_model: str | None = None,
) -> ElectronPolicyPayload:
    requested_policy = normalize_policy(policy)
    requested_driven_response_model = (
        default_driven_response_model_for_policy(requested_policy)
        if driven_response_model is None
        else normalize_driven_response_model(driven_response_model)
    )
    raw_ne = np.asarray(dataset.electron_density_cm3, dtype=np.float64)
    raw_zbar = np.asarray(dataset.mean_charge, dtype=np.float64)
    rho = np.asarray(dataset.density_g_cm3, dtype=np.float64)
    if rho.shape != raw_ne.shape or rho.shape != raw_zbar.shape:
        raise ValueError("Density and mean-charge arrays must share snapshot x zone shape for plasmon electron-policy mapping.")
    if rho.ndim != 2:
        raise ValueError("Expected snapshot x zone arrays for plasmon electron-policy mapping.")

    formula_map = resolved_material_formula_map(dataset)
    baseline_table_source = _baseline_table_source()
    if requested_policy == PLASMON_ELECTRON_POLICY_RAW:
        return ElectronPolicyPayload(
            requested_policy=requested_policy,
            policy=PLASMON_ELECTRON_POLICY_RAW,
            electron_density_cm3=raw_ne.copy(),
            mean_charge=raw_zbar.copy(),
            source_label="raw HELIOS ne/zbar",
            summary="Raw HELIOS electron-density and mean-charge fields.",
            material_formula_map=formula_map,
            baseline_mode="raw_fields",
            baseline_table_source=baseline_table_source,
            baseline_mean_charge=np.zeros_like(raw_zbar),
            increment_mean_charge=np.zeros_like(raw_zbar),
            increment_mode="raw_fields",
            driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
            driven_response_summary="",
            driven_response_weight_mode="",
            driven_response_weight_multiplier=None,
            driven_response_shape_mode="",
            driven_response_shape_fwhm_ev=None,
            driven_response_ensemble_mode="",
        )

    registry = material_policy_registry()
    zone_material_index = np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32))
    recomputed_ne = raw_ne.copy()
    recomputed_zbar = raw_zbar.copy()
    baseline_zbar = np.zeros_like(raw_zbar)
    increment_zbar = np.zeros_like(raw_zbar)
    resolved_entries: list[str] = []
    driven_response_entries: list[str] = []
    driven_response_weight_mode = ""
    driven_response_weight_multiplier = np.ones_like(raw_ne, dtype=np.float64)
    driven_response_shape_mode = ""
    driven_response_shape_fwhm_ev = np.zeros_like(raw_ne, dtype=np.float64)
    driven_response_ensemble_mode = ""
    unresolved: set[str] = set()
    raw_kept: set[str] = set()
    baseline_entries: list[str] = []
    increment_entries: list[str] = []
    for material_id in np.unique(zone_material_index):
        material_id_int = int(material_id)
        formula = formula_map.get(material_id_int)
        material_mask = zone_material_index == material_id_int
        if formula is None:
            unresolved.add(f"material {material_id_int}")
            raw_kept.add(f"material {material_id_int}")
            continue
        entry = registry.get(str(formula))
        if entry is None:
            unresolved.add(f"{formula} (no JSON baseline entry)")
            raw_kept.add(str(formula))
            continue
        if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK and not _article_policy_formula_applies(formula):
            raw_kept.add(formula)
            continue
        atomic_weight = _mean_atomic_weight_for_formula(dataset, formula, material_mask)
        if not math.isfinite(atomic_weight) or atomic_weight <= 0.0:
            unresolved.add(f"{formula} (invalid atomic weight)")
            raw_kept.add(formula)
            continue
        local_rho = rho[:, material_mask]
        ion_density = _ion_density_cm3(local_rho, atomic_weight)
        policy_mode = _policy_mode_for_entry(requested_policy, entry)
        baseline_local = np.full(local_rho.shape, float(entry.benchmark_valence_per_nucleus), dtype=np.float64)
        if policy_mode == "cold_baseline_only":
            increment_local = np.zeros(local_rho.shape, dtype=np.float64)
        elif policy_mode == "cold_baseline_plus_benchmark_increment":
            response = apply_driven_response_model(
                DrivenElectronResponseState(
                    material_formula=str(formula),
                    density_g_cm3=np.asarray(local_rho, dtype=np.float64),
                    electron_temperature_ev=np.asarray(dataset.temperature_e_ev[:, material_mask], dtype=np.float64),
                    ion_temperature_ev=np.asarray(dataset.temperature_i_ev[:, material_mask], dtype=np.float64),
                    ion_density_cm3=np.asarray(ion_density, dtype=np.float64),
                    raw_electron_density_cm3=np.asarray(raw_ne[:, material_mask], dtype=np.float64),
                    raw_mean_charge=np.asarray(raw_zbar[:, material_mask], dtype=np.float64),
                    baseline_mean_charge=np.asarray(baseline_local, dtype=np.float64),
                    baseline_entry=str(entry.baseline_entry_path),
                    baseline_table_source=str(baseline_table_source),
                    state_origin="electron_policy.article_al_driven_increment",
                ),
                requested_driven_response_model,
            )
            increment_local = np.asarray(response.increment_mean_charge, dtype=np.float64)
            driven_response_entries.append(str(response.summary))
            response_modifiers = dict(getattr(response, "response_modifiers", {}) or {})
            local_weight_mode = str(response_modifiers.get("ensemble_weight_mode", "")).strip()
            if local_weight_mode and not driven_response_weight_mode:
                driven_response_weight_mode = local_weight_mode
            local_weight_multiplier = response_modifiers.get("ensemble_weight_multiplier")
            if local_weight_multiplier is not None:
                local_weight_array = np.asarray(local_weight_multiplier, dtype=np.float64)
                if local_weight_array.shape == local_rho.shape:
                    driven_response_weight_multiplier[:, material_mask] = local_weight_array
            local_shape_mode = str(response_modifiers.get("shape_modifier_mode", "")).strip()
            if local_shape_mode and not driven_response_shape_mode:
                driven_response_shape_mode = local_shape_mode
            local_shape_fwhm = response_modifiers.get("shape_modifier_fwhm_ev")
            if local_shape_fwhm is not None:
                local_shape_array = np.asarray(local_shape_fwhm, dtype=np.float64)
                if local_shape_array.shape == local_rho.shape:
                    driven_response_shape_fwhm_ev[:, material_mask] = local_shape_array
            local_ensemble_mode = str(response_modifiers.get("ensemble_response_mode", "")).strip()
            if local_ensemble_mode and not driven_response_ensemble_mode:
                driven_response_ensemble_mode = local_ensemble_mode
        else:
            increment_local = np.clip(
                np.asarray(raw_zbar[:, material_mask], dtype=np.float64) - float(entry.benchmark_valence_per_nucleus),
                0.0,
                None,
            )
        z_eff = baseline_local + increment_local
        baseline_zbar[:, material_mask] = baseline_local
        increment_zbar[:, material_mask] = increment_local
        recomputed_zbar[:, material_mask] = z_eff
        recomputed_ne[:, material_mask] = ion_density * z_eff
        resolved_entries.append(
            _resolved_entry_label(
                requested_policy,
                formula,
                entry,
                z_eff,
                baseline_z_eff=baseline_local,
                increment_z_eff=increment_local,
            )
        )
        baseline_entries.append(
            f"{formula}@{entry.baseline_entry_path}: cold_Zeff={entry.benchmark_valence_per_nucleus:.3f}, "
            f"driven_default={entry.driven_policy_default or 'cold_plus_increment'}, "
            f"class={entry.cold_electronic_state_class or 'unspecified'}"
        )
        increment_entries.append(
            f"{formula}@{entry.baseline_entry_path}: baseline={float(entry.benchmark_valence_per_nucleus):.3f}, "
            f"increment_mean={float(np.nanmean(np.asarray(increment_local, dtype=np.float64))):.3f}, "
            f"final_mean={float(np.nanmean(np.asarray(z_eff, dtype=np.float64))):.3f}, "
            f"mode={policy_mode}"
        )

    if requested_policy == PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE:
        source_label = f"benchmark valence-aware ne from {_BASELINE_JSON_NAME}"
        baseline_mode = "cold_baseline_plus_increment"
        increment_mode = "raw_positive_increment_only"
    elif requested_policy == PLASMON_ELECTRON_POLICY_VALENCE_LOCKED:
        source_label = f"valence-locked ne from {_BASELINE_JSON_NAME}"
        baseline_mode = "cold_baseline_only"
        increment_mode = "none"
    elif requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT:
        source_label = f"article Al driven-increment ne from {_BASELINE_JSON_NAME} via {requested_driven_response_model}"
        baseline_mode = "cold_baseline_only"
        increment_mode = "benchmark_driven_increment"
    else:
        source_label = f"article Al benchmark ne from {_BASELINE_JSON_NAME}"
        baseline_mode = "cold_baseline_plus_increment"
        increment_mode = "raw_positive_increment_only"

    if requested_policy == PLASMON_ELECTRON_POLICY_BENCHMARK_VALENCE_AWARE:
        summary_prefix = "Benchmark valence-aware electron policy"
    elif requested_policy == PLASMON_ELECTRON_POLICY_VALENCE_LOCKED:
        summary_prefix = "Valence-locked electron policy"
    elif requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT:
        summary_prefix = "Article-facing Al driven-increment policy"
    else:
        summary_prefix = "Article-facing Al benchmark electron policy"
    summary = (
        f"{summary_prefix}: "
        f"{', '.join(resolved_entries) if resolved_entries else 'no resolved benchmark materials'}; "
        f"unresolved [{', '.join(sorted(unresolved)) if unresolved else 'none'}]; "
        f"raw-kept [{', '.join(sorted(raw_kept)) if raw_kept else 'none'}]."
    )
    if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT and driven_response_entries:
        summary = f"{summary} driven-response [{' || '.join(sorted(driven_response_entries))}]."
    return ElectronPolicyPayload(
        requested_policy=requested_policy,
        policy=requested_policy,
        electron_density_cm3=recomputed_ne,
        mean_charge=recomputed_zbar,
        source_label=source_label,
        summary=summary,
        resolved_materials=tuple(sorted(resolved_entries)),
        unresolved_materials=tuple(sorted(unresolved)),
        raw_kept_materials=tuple(sorted(raw_kept)),
        material_formula_map=formula_map,
        baseline_mode=baseline_mode,
        baseline_entries=tuple(sorted(baseline_entries)),
        baseline_table_source=baseline_table_source,
        baseline_mean_charge=baseline_zbar,
        increment_mean_charge=increment_zbar,
        increment_mode=increment_mode,
        increment_entries=tuple(sorted(increment_entries)),
        driven_response_model=(
            requested_driven_response_model
            if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT
            else PLASMON_DRIVEN_RESPONSE_MODEL_NONE
        ),
        driven_response_summary=(
            " || ".join(sorted(driven_response_entries))
            if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT
            else ""
        ),
        driven_response_weight_mode=(
            str(driven_response_weight_mode)
            if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT
            else ""
        ),
        driven_response_weight_multiplier=(
            np.asarray(driven_response_weight_multiplier, dtype=np.float64)
            if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT
            else None
        ),
        driven_response_shape_mode=(
            str(driven_response_shape_mode)
            if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT
            else ""
        ),
        driven_response_shape_fwhm_ev=(
            np.asarray(driven_response_shape_fwhm_ev, dtype=np.float64)
            if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT and str(driven_response_shape_mode).strip()
            else None
        ),
        driven_response_ensemble_mode=(
            str(driven_response_ensemble_mode)
            if requested_policy == PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT
            else ""
        ),
    )
