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

import numpy as np

from helios.services.derived.plasmon_config import (
    PLASMON_MODEL_FINITE_T_STLS,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_MODEL_RPA,
    PLASMON_MODEL_RPA_STATIC_LFC,
    PLASMON_OBSERVABLE_MODE_XRTS,
    PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE,
)
from helios.services.derived.plasmon_xrts_observable import observable_mode_label

try:
    from benchmark_plasmon_article_cases import (
        PRIMARY_POLICY_AMBIENT,
        PRIMARY_POLICY_DRIVEN,
        _model_label,
        build_report,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.benchmark_plasmon_article_cases import (  # type: ignore
        PRIMARY_POLICY_AMBIENT,
        PRIMARY_POLICY_DRIVEN,
        _model_label,
        build_report,
    )


DEFAULT_OUTPUT_DIR = Path("outputs/validation_outputs/plasmon_article_native_observable_experiment")
AUDIT_PATH = Path("outputs/validation_outputs/plasmon_article_native_observable_audit.md")
NEXT_STEP_PATH = Path("outputs/validation_outputs/plasmon_article_native_next_step.md")
MODEL_SUBSET = (
    PLASMON_MODEL_RPA,
    PLASMON_MODEL_RPA_STATIC_LFC,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_MODEL_FINITE_T_STLS,
)
SUBRUNS = (
    ("minimal_xrts", PLASMON_OBSERVABLE_MODE_XRTS),
    ("article_native_al", PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE),
)
PRIMARY_CASE_POLICIES = {
    "ambient_al_t0": PRIMARY_POLICY_AMBIENT,
    "driven_al_article_state": PRIMARY_POLICY_DRIVEN,
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


def _json_dumps(values: object) -> str:
    return json.dumps(values, ensure_ascii=True, allow_nan=False)


def _judged_delta(before: float, after: float) -> str:
    if not math.isfinite(before) or not math.isfinite(after):
        return "not_comparable"
    delta = float(before - after)
    if delta >= 1.0:
        return "material improvement"
    if delta >= 0.25:
        return "partial improvement"
    if delta > -0.25:
        return "marginal change"
    return "worse"


def _integral_area(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(np.clip(y[finite], a_min=0.0, a_max=None), x[finite]))
    return float(np.trapz(np.clip(y[finite], a_min=0.0, a_max=None), x[finite]))


def _reconciliation_map(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("case", "")), str(row.get("model", "")), str(row.get("electron_policy", "")))
        metric = _float(row, "mae_ev")
        incumbent = best.get(key)
        if incumbent is None or (math.isfinite(metric) and (not math.isfinite(_float(incumbent, "mae_ev")) or metric < _float(incumbent, "mae_ev"))):
            best[key] = row
    return best


def _primary_summary_rows(rows: list[dict[str, str]], *, observable_mode: str) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in rows:
        case_name = str(row.get("case", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != str(row.get("electron_policy", "")):
            continue
        model = str(row.get("model", ""))
        if model not in MODEL_SUBSET:
            continue
        filtered.append(
            {
                "case": case_name,
                "model": model,
                "model_label": str(row.get("model_label", _model_label(model))),
                "electron_policy": str(row.get("electron_policy", "")),
                "status": str(row.get("status", "")),
                "backend": str(row.get("backend", "")),
                "runtime_mean_s": _float(row, "runtime_mean_s"),
                "experiment_mae_ev": _float(row, "mae_ev"),
                "experiment_rmse_ev": _float(row, "rmse_ev"),
                "observable_mode": observable_mode,
                "observable_mode_label": observable_mode_label(observable_mode),
                "observable_summary": str(row.get("observable_summary", "")),
                "observable_decomposition_mode": str(row.get("observable_decomposition_mode", "")),
                "observable_peak_extraction_mode": str(row.get("observable_peak_extraction_mode", "")),
                "observable_comparison_mode": str(row.get("observable_comparison_mode", "")),
                "observable_subtraction_mode": str(row.get("observable_subtraction_mode", "")),
                "observable_normalization_mode": str(row.get("observable_normalization_mode", "")),
                "observable_elastic_exclusion_ev": _float(row, "observable_elastic_exclusion_ev"),
                "observable_free_fraction": _float(row, "observable_free_fraction"),
                "observable_bound_fraction": _float(row, "observable_bound_fraction"),
                "observable_elastic_fraction": _float(row, "observable_elastic_fraction"),
                "observable_peak_discrete_energy_ev": _float(row, "observable_peak_discrete_energy_ev"),
                "observable_peak_fit_energy_ev": _float(row, "observable_peak_fit_energy_ev"),
                "observable_peak_fit_status": str(row.get("observable_peak_fit_status", "")),
                "observable_peak_edge_dominated_any": str(row.get("observable_peak_edge_dominated_any", "")),
                "observable_elastic_form_factor_total": _float(row, "observable_elastic_form_factor_total"),
                "observable_elastic_form_factor_core": _float(row, "observable_elastic_form_factor_core"),
                "observable_elastic_screening_form_factor": _float(row, "observable_elastic_screening_form_factor"),
                "observable_ion_structure_factor": _float(row, "observable_ion_structure_factor"),
                "observable_bound_core_mode": str(row.get("observable_bound_core_mode", "")),
                "observable_bound_shell_summary": str(row.get("observable_bound_shell_summary", "")),
            }
        )
    filtered.sort(key=lambda row: (str(row["case"]), str(row["model"])))
    return filtered


def _comparison_rows(summary_rows: list[dict[str, object]], reconciliation_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    recon_map = _reconciliation_map(reconciliation_rows)
    rows: list[dict[str, object]] = []
    for row in summary_rows:
        recon = recon_map.get((str(row["case"]), str(row["model"]), str(row["electron_policy"])), {})
        rows.append(
            {
                **row,
                "published_branch": str(recon.get("published_branch", "")),
                "comparison_kind": str(recon.get("comparison_kind", "")),
                "matched_branch_mae_ev": _float(recon, "mae_ev"),
                "matched_branch_rmse_ev": _float(recon, "rmse_ev"),
                "comparison_note": str(recon.get("note", recon.get("comparison_note", ""))),
            }
        )
    return rows


def _load_representative_spectra(path: Path, *, observable_mode: str) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    cases = {
        "ambient_al_t0": payload.get("ambient_cases", {}).get(PRIMARY_POLICY_AMBIENT, {}),
        "driven_al_article_state": payload.get("article_driven_cases", {}).get(PRIMARY_POLICY_DRIVEN, {}),
    }
    for case_name, case_payload in cases.items():
        spectra = dict(case_payload.get("spectra", {}))
        for model, spectrum in spectra.items():
            if str(model) not in MODEL_SUBSET:
                continue
            energy = np.asarray(spectrum.get("energy_ev", []), dtype=np.float64)
            intensity = np.asarray(spectrum.get("intensity", []), dtype=np.float64)
            free = np.asarray(spectrum.get("free_component", []), dtype=np.float64)
            bound = np.asarray(spectrum.get("bound_component", []), dtype=np.float64)
            elastic = np.asarray(spectrum.get("elastic_component", []), dtype=np.float64)
            rows.append(
                {
                    "case": case_name,
                    "model": str(model),
                    "model_label": str(spectrum.get("model_label", _model_label(str(model)))),
                    "backend": str(spectrum.get("backend", "")),
                    "status": str(spectrum.get("status", "")),
                    "observable_mode": observable_mode,
                    "observable_mode_label": observable_mode_label(observable_mode),
                    "observable_summary": str(spectrum.get("observable_summary", "")),
                    "observable_decomposition_mode": str(spectrum.get("observable_decomposition_mode", "")),
                    "observable_peak_extraction_mode": str(spectrum.get("observable_peak_extraction_mode", "")),
                    "observable_comparison_mode": str(spectrum.get("observable_comparison_mode", "")),
                    "observable_subtraction_mode": str(spectrum.get("observable_subtraction_mode", "")),
                    "observable_normalization_mode": str(spectrum.get("observable_normalization_mode", "")),
                    "observable_elastic_exclusion_ev": _float(spectrum, "observable_elastic_exclusion_ev"),
                    "observable_free_fraction": _float(spectrum, "observable_free_fraction"),
                    "observable_bound_fraction": _float(spectrum, "observable_bound_fraction"),
                    "observable_elastic_fraction": _float(spectrum, "observable_elastic_fraction"),
                    "observable_peak_discrete_energy_ev": _float(spectrum, "observable_peak_discrete_energy_ev"),
                    "observable_peak_fit_energy_ev": _float(spectrum, "observable_peak_fit_energy_ev"),
                    "observable_peak_fit_status": str(spectrum.get("observable_peak_fit_status", "")),
                    "observable_peak_edge_dominated": str(spectrum.get("observable_peak_edge_dominated", "")),
                    "observable_elastic_form_factor_total": _float(spectrum, "observable_elastic_form_factor_total"),
                    "observable_elastic_form_factor_core": _float(spectrum, "observable_elastic_form_factor_core"),
                    "observable_elastic_screening_form_factor": _float(spectrum, "observable_elastic_screening_form_factor"),
                    "observable_ion_structure_factor": _float(spectrum, "observable_ion_structure_factor"),
                    "observable_bound_core_mode": str(spectrum.get("observable_bound_core_mode", "")),
                    "observable_bound_shell_summary": str(spectrum.get("observable_bound_shell_summary", "")),
                    "peak_energy_ev": _float(spectrum, "peak_energy_ev"),
                    "energy_ev_json": _json_dumps(energy.tolist()),
                    "intensity_json": _json_dumps(intensity.tolist()),
                    "free_component_json": _json_dumps(free.tolist()),
                    "bound_component_json": _json_dumps(bound.tolist()),
                    "elastic_component_json": _json_dumps(elastic.tolist()),
                }
            )
    rows.sort(key=lambda row: (str(row["case"]), str(row["model"]), str(row["observable_mode"])))
    return rows


def _component_breakdown_rows(spectrum_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in spectrum_rows:
        energy = np.asarray(json.loads(str(row["energy_ev_json"])), dtype=np.float64)
        intensity = np.asarray(json.loads(str(row["intensity_json"])), dtype=np.float64)
        free = np.asarray(json.loads(str(row["free_component_json"])), dtype=np.float64)
        bound = np.asarray(json.loads(str(row["bound_component_json"])), dtype=np.float64)
        elastic = np.asarray(json.loads(str(row["elastic_component_json"])), dtype=np.float64)
        if free.size != energy.size:
            free = np.asarray(intensity, dtype=np.float64) if intensity.size == energy.size else np.zeros_like(energy, dtype=np.float64)
        if bound.size != energy.size:
            bound = np.zeros_like(energy, dtype=np.float64)
        if elastic.size != energy.size:
            elastic = np.zeros_like(energy, dtype=np.float64)
        free_area = _integral_area(energy, free)
        bound_area = _integral_area(energy, bound)
        elastic_area = _integral_area(energy, elastic)
        total_area = free_area + bound_area + elastic_area
        rows.append(
            {
                "case": str(row["case"]),
                "model": str(row["model"]),
                "model_label": str(row["model_label"]),
                "backend": str(row["backend"]),
                "observable_mode": str(row["observable_mode"]),
                "free_area": float(free_area),
                "bound_area": float(bound_area),
                "elastic_area": float(elastic_area),
                "total_area": float(total_area),
                "free_fraction": (float(free_area / total_area) if total_area > 0.0 else float("nan")),
                "bound_fraction": (float(bound_area / total_area) if total_area > 0.0 else float("nan")),
                "elastic_fraction": (float(elastic_area / total_area) if total_area > 0.0 else float("nan")),
                "peak_energy_ev": _float(row, "peak_energy_ev"),
            }
        )
    rows.sort(key=lambda row: (str(row["case"]), str(row["model"]), str(row["observable_mode"])))
    return rows


def _merge_spectrum_details(
    comparison_rows: list[dict[str, object]],
    spectrum_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    spectrum_map = {
        (str(row["case"]), str(row["model"]), str(row["observable_mode"])): row
        for row in spectrum_rows
    }
    rows: list[dict[str, object]] = []
    for row in comparison_rows:
        merged = dict(row)
        spectrum = spectrum_map.get((str(row["case"]), str(row["model"]), str(row["observable_mode"])), {})
        for key, value in spectrum.items():
            if key in {"energy_ev_json", "intensity_json", "free_component_json", "bound_component_json", "elastic_component_json"}:
                continue
            if key not in merged or str(key).startswith("observable_"):
                merged[key] = value
        rows.append(merged)
    rows.sort(key=lambda row: (str(row["case"]), str(row["model"]), str(row["observable_mode"])))
    return rows


def _delta_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    row_map = {
        (str(row["case"]), str(row["model"]), str(row["electron_policy"]), str(row["observable_mode"])): row
        for row in comparison_rows
    }
    rows: list[dict[str, object]] = []
    for case_name, electron_policy in PRIMARY_CASE_POLICIES.items():
        for model in MODEL_SUBSET:
            minimal = row_map.get((case_name, model, electron_policy, PLASMON_OBSERVABLE_MODE_XRTS))
            article_native = row_map.get((case_name, model, electron_policy, PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE))
            if minimal is None or article_native is None:
                continue
            rows.append(
                {
                    "case": case_name,
                    "model": model,
                    "model_label": str(article_native["model_label"]),
                    "electron_policy": electron_policy,
                    "backend": str(article_native["backend"]),
                    "minimal_status": str(minimal["status"]),
                    "article_native_status": str(article_native["status"]),
                    "experiment_mae_minimal_ev": _float(minimal, "experiment_mae_ev"),
                    "experiment_mae_article_native_ev": _float(article_native, "experiment_mae_ev"),
                    "experiment_delta_minimal_minus_article_native_ev": (_float(minimal, "experiment_mae_ev") - _float(article_native, "experiment_mae_ev")),
                    "experiment_judgement": _judged_delta(_float(minimal, "experiment_mae_ev"), _float(article_native, "experiment_mae_ev")),
                    "matched_branch_minimal": str(minimal.get("published_branch", "")),
                    "matched_branch_article_native": str(article_native.get("published_branch", "")),
                    "matched_branch_mae_minimal_ev": _float(minimal, "matched_branch_mae_ev"),
                    "matched_branch_mae_article_native_ev": _float(article_native, "matched_branch_mae_ev"),
                    "matched_branch_delta_minimal_minus_article_native_ev": (_float(minimal, "matched_branch_mae_ev") - _float(article_native, "matched_branch_mae_ev")),
                    "matched_branch_judgement": _judged_delta(_float(minimal, "matched_branch_mae_ev"), _float(article_native, "matched_branch_mae_ev")),
                    "observable_summary": str(article_native.get("observable_summary", "")),
                    "observable_decomposition_mode": str(article_native.get("observable_decomposition_mode", "")),
                }
            )
    rows.sort(key=lambda row: (str(row["case"]), str(row["model"])))
    return rows


def _observable_provenance_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        {
            "case": str(row["case"]),
            "model": str(row["model"]),
            "model_label": str(row["model_label"]),
            "electron_policy": str(row["electron_policy"]),
            "backend": str(row["backend"]),
            "observable_mode": str(row["observable_mode"]),
            "observable_mode_label": str(row["observable_mode_label"]),
            "observable_summary": str(row.get("observable_summary", "")),
            "observable_decomposition_mode": str(row.get("observable_decomposition_mode", "")),
            "observable_peak_extraction_mode": str(row.get("observable_peak_extraction_mode", "")),
            "observable_comparison_mode": str(row.get("observable_comparison_mode", "")),
            "observable_subtraction_mode": str(row.get("observable_subtraction_mode", "")),
            "observable_normalization_mode": str(row.get("observable_normalization_mode", "")),
            "observable_elastic_exclusion_ev": _float(row, "observable_elastic_exclusion_ev"),
        }
        for row in comparison_rows
    ]
    rows.sort(key=lambda row: (str(row["case"]), str(row["model"]), str(row["observable_mode"])))
    return rows


def _normalization_subtraction_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        {
            "case": str(row["case"]),
            "model": str(row["model"]),
            "model_label": str(row["model_label"]),
            "backend": str(row["backend"]),
            "observable_mode": str(row["observable_mode"]),
            "observable_peak_extraction_mode": str(row.get("observable_peak_extraction_mode", "")),
            "observable_comparison_mode": str(row.get("observable_comparison_mode", "")),
            "observable_subtraction_mode": str(row.get("observable_subtraction_mode", "")),
            "observable_normalization_mode": str(row.get("observable_normalization_mode", "")),
            "observable_elastic_exclusion_ev": _float(row, "observable_elastic_exclusion_ev"),
            "observable_peak_discrete_energy_ev": _float(row, "observable_peak_discrete_energy_ev"),
            "observable_peak_fit_energy_ev": _float(row, "observable_peak_fit_energy_ev"),
            "observable_peak_fit_status": str(row.get("observable_peak_fit_status", "")),
        }
        for row in comparison_rows
    ]
    rows.sort(key=lambda row: (str(row["case"]), str(row["model"]), str(row["observable_mode"])))
    return rows


def _elastic_feature_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        {
            "case": str(row["case"]),
            "model": str(row["model"]),
            "model_label": str(row["model_label"]),
            "backend": str(row["backend"]),
            "observable_mode": str(row["observable_mode"]),
            "observable_elastic_fraction": _float(row, "observable_elastic_fraction"),
            "observable_elastic_form_factor_total": _float(row, "observable_elastic_form_factor_total"),
            "observable_elastic_form_factor_core": _float(row, "observable_elastic_form_factor_core"),
            "observable_elastic_screening_form_factor": _float(row, "observable_elastic_screening_form_factor"),
            "observable_ion_structure_factor": _float(row, "observable_ion_structure_factor"),
        }
        for row in comparison_rows
    ]
    rows.sort(key=lambda row: (str(row["case"]), str(row["model"]), str(row["observable_mode"])))
    return rows


def _bound_core_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        {
            "case": str(row["case"]),
            "model": str(row["model"]),
            "model_label": str(row["model_label"]),
            "backend": str(row["backend"]),
            "observable_mode": str(row["observable_mode"]),
            "observable_bound_fraction": _float(row, "observable_bound_fraction"),
            "observable_bound_core_mode": str(row.get("observable_bound_core_mode", "")),
            "observable_bound_shell_summary": str(row.get("observable_bound_shell_summary", "")),
        }
        for row in comparison_rows
    ]
    rows.sort(key=lambda row: (str(row["case"]), str(row["model"]), str(row["observable_mode"])))
    return rows


def _write_audit_note() -> None:
    lines = [
        "# Article-native Al XRTS observable audit",
        "",
        "The minimal XRTS observable seam already established that the remaining driven residual is not removed by simply wrapping the backend DSF with a free+elastic proxy.",
        "",
        "Before the article-native layer, the observable path still had four concrete limitations:",
        "- elastic/ion feature was represented only by a compact generic Al proxy, not an explicit Al atomic-form-factor split",
        "- bound/core inelastic contribution existed only as placeholder bookkeeping",
        "- density-averaged representative spectra dropped the component arrays during export, which made component-breakdown CSVs misleading",
        "- density-averaged observable peak extraction could fail catastrophically at high q because the local quadratic fit was accepted even when it detached from the true inelastic maximum",
        "",
        "This pass upgrades the observable construction specifically for Al:",
        "- free-electron inelastic term still comes from the validated backend DSF (QHD, STLS, RPA, RPA+static LFC)",
        "- elastic feature now uses explicit Al Cromer-Mann form factors",
        "- neutral-Al and Al3+ form-factor bookkeeping is separated so the screening/core split is visible in provenance",
        "- bound/core inelastic contribution is shell-resolved and explicitly zero below the Al L-shell onset in the current 45 eV benchmark window",
        "- article-facing comparison is made on the positive inelastic branch after explicit elastic subtraction rather than on the raw total spectrum peak",
        "",
        "Honesty limits remain explicit:",
        "- ion structure factor is still treated with the unity assumption",
        "- no full bound-free atomic cross section is introduced",
        "- no hidden rescaling or normalization fit is applied",
        "- exact article-side background subtraction and detector processing assumptions are still not recoverable from the current repo assets",
    ]
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_next_step_note(*, report_dir: Path, delta_rows: list[dict[str, object]]) -> None:
    experiment_deltas = [float(row["experiment_delta_minimal_minus_article_native_ev"]) for row in delta_rows if math.isfinite(float(row["experiment_delta_minimal_minus_article_native_ev"]))]
    mean_delta = float(np.mean(experiment_deltas)) if experiment_deltas else float("nan")
    best_delta = float(np.max(experiment_deltas)) if experiment_deltas else float("nan")
    worst_delta = float(np.min(experiment_deltas)) if experiment_deltas else float("nan")
    lines = [
        "# Article-native observable next step",
        "",
        f"Current output directory: `{report_dir}`",
        "",
        f"- mean experiment-facing minimal-minus-article-native MAE delta: `{mean_delta:.3f} eV`" if math.isfinite(mean_delta) else "- mean experiment-facing delta: not available",
        f"- best single experiment-facing improvement: `{best_delta:.3f} eV`" if math.isfinite(best_delta) else "- best single improvement: not available",
        f"- worst experiment-facing regression: `{worst_delta:.3f} eV`" if math.isfinite(worst_delta) else "- worst regression: not available",
        "",
        "What was added in this pass:",
        "- explicit Al elastic form-factor bookkeeping (neutral/core/screening split)",
        "- explicit elastic subtraction before article-facing peak extraction",
        "- shell-thresholded core bookkeeping tied to the current benchmark window",
        "",
        "What still remains missing if the residual stays large:",
        "- nontrivial S_ii(q) / ion-feature physics",
        "- real bound-free Al inelastic scattering, not just shell-threshold bookkeeping",
        "- article-native normalization, subtraction, and detector/background assumptions beyond the current recoverable level",
        "",
        "If the residual is still large after this pass, another generic dielectric/backend tweak is no longer justified by the evidence. The next blocker is the missing material-specific atomic/ion observable layer itself.",
    ]
    NEXT_STEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEXT_STEP_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    *,
    out_dir: Path,
    article_native_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
) -> None:
    experiment_deltas = [float(row["experiment_delta_minimal_minus_article_native_ev"]) for row in delta_rows if math.isfinite(float(row["experiment_delta_minimal_minus_article_native_ev"]))]
    best_row = None
    worst_row = None
    if delta_rows:
        best_row = max(
            delta_rows,
            key=lambda row: float(row["experiment_delta_minimal_minus_article_native_ev"]) if math.isfinite(float(row["experiment_delta_minimal_minus_article_native_ev"])) else -1.0e30,
        )
        worst_row = min(
            delta_rows,
            key=lambda row: float(row["experiment_delta_minimal_minus_article_native_ev"]) if math.isfinite(float(row["experiment_delta_minimal_minus_article_native_ev"])) else 1.0e30,
        )
    lines = [
        "# Article-native Al XRTS observable experiment",
        "",
        "This pass compares the current minimal XRTS observable against a more article-native Al observable assembly built on the same validated backend responses.",
        "",
        "## Summary",
        "",
        f"- compared modes: `{observable_mode_label(PLASMON_OBSERVABLE_MODE_XRTS)}` vs `{observable_mode_label(PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE)}`",
        f"- backend subset: {', '.join(_model_label(model) for model in MODEL_SUBSET)}",
        f"- mean experiment-facing minimal-minus-article-native MAE delta: `{float(np.mean(experiment_deltas)):.3f} eV`" if experiment_deltas else "- mean experiment-facing delta: not available",
    ]
    if best_row is not None:
        lines.append(
            f"- best experiment-facing change: `{best_row['case']}` / `{best_row['model_label']}` = `{float(best_row['experiment_delta_minimal_minus_article_native_ev']):.3f} eV` ({best_row['experiment_judgement']})"
        )
    if worst_row is not None:
        lines.append(
            f"- worst experiment-facing change: `{worst_row['case']}` / `{worst_row['model_label']}` = `{float(worst_row['experiment_delta_minimal_minus_article_native_ev']):.3f} eV` ({worst_row['experiment_judgement']})"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The article-native layer changes the observable construction level, not the backend dielectric.",
            "",
            "- free-electron inelastic response still comes from the backend DSF",
            "- the elastic feature is now assembled from Al form-factor bookkeeping rather than the minimal proxy alone",
            "- the comparison peak is taken from the inelastic branch after explicit elastic subtraction",
            "",
            "## Article-native Component Breakdown",
            "",
        ]
    )
    for row in component_rows:
        if str(row["observable_mode"]) != PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE:
            continue
        lines.append(
            f"- `{row['case']}` / `{row['model_label']}`: free `{float(row['free_fraction']):.3f}`, bound `{float(row['bound_fraction']):.3f}`, elastic `{float(row['elastic_fraction']):.3f}`"
        )
    lines.extend(["", "## Article-native Headline Rows", ""])
    for row in article_native_rows:
        lines.append(
            f"- `{row['case']}` / `{row['model_label']}`: experiment MAE `{float(row['experiment_mae_ev']):.3f} eV`, matched-branch MAE `{float(row['matched_branch_mae_ev']):.3f} eV`, status `{row['status']}`"
        )
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the article-native Al XRTS observable layer against the current minimal XRTS observable.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    hydro_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_audit_note()

    for slug, observable_mode in SUBRUNS:
        subdir = out_dir / slug
        build_report(
            hydro_path,
            out_dir=subdir,
            model_subset=MODEL_SUBSET,
            observable_mode=observable_mode,
        )

    comparison_rows: list[dict[str, object]] = []
    spectra_rows: list[dict[str, object]] = []
    for slug, observable_mode in SUBRUNS:
        subdir = out_dir / slug
        summary_rows = _primary_summary_rows(_read_csv_rows(subdir / "benchmark_summary.csv"), observable_mode=observable_mode)
        recon_rows = _read_csv_rows(subdir / "reconciliation_summary.csv")
        comparison_rows.extend(_comparison_rows(summary_rows, recon_rows))
        spectra_rows.extend(_load_representative_spectra(subdir / "all_model_results.json", observable_mode=observable_mode))

    comparison_rows.sort(key=lambda row: (str(row["case"]), str(row["model"]), str(row["observable_mode"])))
    merged_rows = _merge_spectrum_details(comparison_rows, spectra_rows)
    article_native_rows = [row for row in merged_rows if str(row["observable_mode"]) == PLASMON_OBSERVABLE_MODE_XRTS_ARTICLE_NATIVE]
    delta_rows = _delta_rows(comparison_rows)
    component_rows = _component_breakdown_rows(spectra_rows)
    provenance_rows = _observable_provenance_rows(merged_rows)
    normalization_rows = _normalization_subtraction_rows(merged_rows)
    elastic_rows = _elastic_feature_rows(merged_rows)
    bound_rows = _bound_core_rows(merged_rows)

    _write_csv(out_dir / "benchmark_summary.csv", comparison_rows)
    _write_csv(out_dir / "article_native_observable_summary.csv", article_native_rows)
    _write_csv(out_dir / "control_vs_article_native_delta.csv", delta_rows)
    _write_csv(out_dir / "component_breakdown.csv", component_rows)
    _write_csv(out_dir / "observable_provenance.csv", provenance_rows)
    _write_csv(out_dir / "normalization_subtraction_diagnostics.csv", normalization_rows)
    _write_csv(out_dir / "elastic_feature_diagnostics.csv", elastic_rows)
    _write_csv(out_dir / "bound_core_diagnostics.csv", bound_rows)
    _write_csv(out_dir / "xrts_spectra_comparison.csv", spectra_rows)
    _write_report(
        out_dir=out_dir,
        article_native_rows=article_native_rows,
        delta_rows=delta_rows,
        component_rows=component_rows,
    )
    _write_next_step_note(report_dir=out_dir, delta_rows=delta_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
