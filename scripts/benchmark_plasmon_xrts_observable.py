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
    PLASMON_MODEL_CHOICES,
    PLASMON_MODEL_FINITE_T_STLS,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_MODEL_RPA,
    PLASMON_MODEL_RPA_STATIC_LFC,
    PLASMON_OBSERVABLE_MODE_DIELECTRIC,
    PLASMON_OBSERVABLE_MODE_XRTS,
)
from helios.services.derived.plasmon_xrts_observable import observable_mode_label

try:
    from benchmark_plasmon_article_cases import (
        ARTICLE_DRIVEN_DENSITY_GRID,
        ARTICLE_DRIVEN_TEMPERATURE_EV,
        PRIMARY_POLICY_AMBIENT,
        PRIMARY_POLICY_DRIVEN,
        REPRESENTATIVE_Q_DRIVEN,
        _compute_density_averaged_point_result,
        _model_label,
        build_report,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.benchmark_plasmon_article_cases import (  # type: ignore
        ARTICLE_DRIVEN_DENSITY_GRID,
        ARTICLE_DRIVEN_TEMPERATURE_EV,
        PRIMARY_POLICY_AMBIENT,
        PRIMARY_POLICY_DRIVEN,
        REPRESENTATIVE_Q_DRIVEN,
        _compute_density_averaged_point_result,
        _model_label,
        build_report,
    )


DEFAULT_OUTPUT_DIR = Path("outputs/validation_outputs/plasmon_xrts_observable_experiment")
AUDIT_PATH = Path("outputs/validation_outputs/plasmon_article_observable_audit.md")
NEXT_STEP_PATH = Path("outputs/validation_outputs/plasmon_xrts_next_step.md")
MODEL_SUBSET = (
    PLASMON_MODEL_RPA,
    PLASMON_MODEL_RPA_STATIC_LFC,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_MODEL_FINITE_T_STLS,
)
SUBRUNS = (
    ("dielectric", PLASMON_OBSERVABLE_MODE_DIELECTRIC),
    ("xrts_observable", PLASMON_OBSERVABLE_MODE_XRTS),
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


def _json_dumps(values: object) -> str:
    return json.dumps(values, ensure_ascii=True, allow_nan=False)


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
                "observable_elastic_exclusion_ev": _float(row, "observable_elastic_exclusion_ev"),
                "observable_free_fraction": _float(row, "observable_free_fraction"),
                "observable_bound_fraction": _float(row, "observable_bound_fraction"),
                "observable_elastic_fraction": _float(row, "observable_elastic_fraction"),
            }
        )
    filtered.sort(key=lambda row: (str(row["case"]), str(row["model"])))
    return filtered


def _comparison_rows(
    summary_rows: list[dict[str, object]],
    reconciliation_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
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
                "comparison_note": str(recon.get("comparison_note", "")),
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
                    "observable_free_fraction": _float(spectrum, "observable_free_fraction"),
                    "observable_bound_fraction": _float(spectrum, "observable_bound_fraction"),
                    "observable_elastic_fraction": _float(spectrum, "observable_elastic_fraction"),
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
            free = (np.asarray(intensity, dtype=np.float64) if intensity.size == energy.size else np.zeros_like(energy, dtype=np.float64))
        if bound.size != energy.size:
            bound = np.zeros_like(energy, dtype=np.float64)
        if elastic.size != energy.size:
            elastic = np.zeros_like(energy, dtype=np.float64)
        total = free + bound + elastic
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


def _delta_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    row_map = {
        (str(row["case"]), str(row["model"]), str(row["electron_policy"]), str(row["observable_mode"])): row
        for row in comparison_rows
    }
    rows: list[dict[str, object]] = []
    for case_name, electron_policy in PRIMARY_CASE_POLICIES.items():
        for model in MODEL_SUBSET:
            dielectric = row_map.get((case_name, model, electron_policy, PLASMON_OBSERVABLE_MODE_DIELECTRIC))
            observable = row_map.get((case_name, model, electron_policy, PLASMON_OBSERVABLE_MODE_XRTS))
            if dielectric is None or observable is None:
                continue
            rows.append(
                {
                    "case": case_name,
                    "model": model,
                    "model_label": str(observable["model_label"]),
                    "electron_policy": electron_policy,
                    "backend": str(observable["backend"]),
                    "dielectric_status": str(dielectric["status"]),
                    "xrts_status": str(observable["status"]),
                    "experiment_mae_dielectric_ev": _float(dielectric, "experiment_mae_ev"),
                    "experiment_mae_xrts_ev": _float(observable, "experiment_mae_ev"),
                    "experiment_delta_dielectric_minus_xrts_ev": (
                        _float(dielectric, "experiment_mae_ev") - _float(observable, "experiment_mae_ev")
                    ),
                    "experiment_judgement": _judged_delta(_float(dielectric, "experiment_mae_ev"), _float(observable, "experiment_mae_ev")),
                    "matched_branch_dielectric": str(dielectric.get("published_branch", "")),
                    "matched_branch_xrts": str(observable.get("published_branch", "")),
                    "matched_branch_mae_dielectric_ev": _float(dielectric, "matched_branch_mae_ev"),
                    "matched_branch_mae_xrts_ev": _float(observable, "matched_branch_mae_ev"),
                    "matched_branch_delta_dielectric_minus_xrts_ev": (
                        _float(dielectric, "matched_branch_mae_ev") - _float(observable, "matched_branch_mae_ev")
                    ),
                    "matched_branch_judgement": _judged_delta(_float(dielectric, "matched_branch_mae_ev"), _float(observable, "matched_branch_mae_ev")),
                    "observable_summary": str(observable.get("observable_summary", "")),
                    "observable_decomposition_mode": str(observable.get("observable_decomposition_mode", "")),
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
            "status": str(row["status"]),
            "observable_summary": str(row.get("observable_summary", "")),
            "observable_decomposition_mode": str(row.get("observable_decomposition_mode", "")),
            "observable_peak_extraction_mode": str(row.get("observable_peak_extraction_mode", "")),
            "observable_elastic_exclusion_ev": _float(row, "observable_elastic_exclusion_ev"),
            "observable_free_fraction": _float(row, "observable_free_fraction"),
            "observable_bound_fraction": _float(row, "observable_bound_fraction"),
            "observable_elastic_fraction": _float(row, "observable_elastic_fraction"),
        }
        for row in comparison_rows
        if str(row.get("observable_mode", "")) == PLASMON_OBSERVABLE_MODE_XRTS
    ]
    rows.sort(key=lambda row: (str(row["case"]), str(row["model"])))
    return rows


def _convolution_sensitivity_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in (PLASMON_MODEL_QUANTUM_HYDRODYNAMIC, PLASMON_MODEL_FINITE_T_STLS):
        for fwhm_ev in (0.2, 1.0, 3.5, 5.0):
            row = _compute_density_averaged_point_result(
                model=str(model),
                electron_policy=PRIMARY_POLICY_DRIVEN,
                driven_response_model=None,
                q_value=float(REPRESENTATIVE_Q_DRIVEN),
                densities_g_cm3=tuple(float(value) for value in ARTICLE_DRIVEN_DENSITY_GRID),
                te_ev=float(ARTICLE_DRIVEN_TEMPERATURE_EV),
                instrument_fwhm_ev=float(fwhm_ev),
                benchmark_preset="al_driven_article",
                observable_mode=PLASMON_OBSERVABLE_MODE_XRTS,
            )
            rows.append(
                {
                    "case": "driven_al_article_state",
                    "model": str(model),
                    "model_label": _model_label(str(model)),
                    "instrument_fwhm_ev": float(fwhm_ev),
                    "status": str(row.get("status", "")),
                    "peak_energy_ev": _float(row, "peak_energy_ev"),
                    "peak_fwhm_ev": _float(row, "peak_fwhm_ev"),
                    "observable_free_fraction": _float(row, "observable_free_fraction"),
                    "observable_bound_fraction": _float(row, "observable_bound_fraction"),
                    "observable_elastic_fraction": _float(row, "observable_elastic_fraction"),
                }
            )
    return rows


def _write_audit_note() -> None:
    lines = [
        "# Plasmon article-level observable audit",
        "",
        "The current dielectric benchmark path compares a peak extracted from the backend free-electron loss/DSF response.",
        "",
        "The new observable path raises the comparison level to a minimal experiment-facing XRTS reconstruction:",
        "- free-electron inelastic term from the selected backend (RPA, static-LFC, QHD, STLS)",
        "- explicit central elastic/ion-feature proxy",
        "- explicit bound/core term bookkeeping",
        "- instrument convolution before peak extraction",
        "",
        "Implemented decomposition:",
        "- minimal Chihara-like Al observable",
        "- Al free term comes directly from the backend DSF",
        "- elastic term uses a bound-electron form-factor proxy centered at zero energy transfer",
        "- bound/core inelastic term is kept explicit but currently zeroed in the narrow article benchmark window below the first Al L-shell onset",
        "",
        "Important honesty constraints:",
        "- this is not article-native atomic physics",
        "- no hidden normalization fit is applied",
        "- mixed/unsupported materials fall back to the backend free-electron spectrum with explicit provenance",
        "- the observable peak is extracted from the positive branch after excluding the elastic core window",
        "",
        "Expected leverage:",
        "- if residual is mostly observable-level, XRTS mode should improve experiment-facing MAE without materially changing backend-matched dielectric-branch trends",
        "- if residual stays large, the missing physics is likely bound-electron / atomic-form-factor / article-native comparison structure rather than another dielectric tweak",
    ]
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_next_step_note(
    *,
    report_dir: Path,
    delta_rows: list[dict[str, object]],
) -> None:
    experiment_deltas = [float(row["experiment_delta_dielectric_minus_xrts_ev"]) for row in delta_rows if math.isfinite(float(row["experiment_delta_dielectric_minus_xrts_ev"]))]
    mean_delta = float(np.mean(experiment_deltas)) if experiment_deltas else float("nan")
    best_delta = float(np.max(experiment_deltas)) if experiment_deltas else float("nan")
    worst_delta = float(np.min(experiment_deltas)) if experiment_deltas else float("nan")
    lines = [
        "# XRTS observable next step",
        "",
        f"Current experiment output directory: `{report_dir}`",
        "",
        "Interpretation of the first observable-layer pass:",
        f"- mean experiment-facing dielectric-minus-observable MAE delta across the primary cases/models: `{mean_delta:.3f} eV`" if math.isfinite(mean_delta) else "- mean experiment-facing delta: not available",
        f"- best single experiment-facing improvement: `{best_delta:.3f} eV`" if math.isfinite(best_delta) else "- best single improvement: not available",
        f"- worst experiment-facing regression: `{worst_delta:.3f} eV`" if math.isfinite(worst_delta) else "- worst regression: not available",
        "",
        "Judgement:",
        "- this minimal observable layer does not close the residual",
        "- ambient remains effectively unchanged",
        "- driven RPA and RPA + static LFC improve only marginally",
        "- driven finite-T STLS becomes substantially worse at the experiment-facing comparison level",
        "",
        "What still remains missing if the residual stays large:",
        "- article-native atomic/Chihara assumptions for Al",
        "- material-specific bound/core inelastic term rather than the current explicit-zero narrow-window approximation",
        "- better elastic/ion-feature modeling than a compact central proxy",
        "- possibly article-specific normalization or subtraction conventions if the paper compared a processed observable rather than a raw convolved spectrum",
        "",
        "Recommended next step if observable-layer improvement is only partial:",
        "- keep QHD and finite-T STLS as backend controls",
        "- add a material-specific Al XRTS decomposition layer with explicit atomic form factors / ion feature assumptions before attempting another dielectric backend",
    ]
    NEXT_STEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEXT_STEP_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    *,
    out_dir: Path,
    comparison_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
) -> None:
    experiment_deltas = [float(row["experiment_delta_dielectric_minus_xrts_ev"]) for row in delta_rows if math.isfinite(float(row["experiment_delta_dielectric_minus_xrts_ev"]))]
    matched_deltas = [float(row["matched_branch_delta_dielectric_minus_xrts_ev"]) for row in delta_rows if math.isfinite(float(row["matched_branch_delta_dielectric_minus_xrts_ev"]))]
    best_row = None
    worst_row = None
    if delta_rows:
        best_row = max(
            delta_rows,
            key=lambda row: float(row["experiment_delta_dielectric_minus_xrts_ev"]) if math.isfinite(float(row["experiment_delta_dielectric_minus_xrts_ev"])) else -1.0e30,
        )
        worst_row = min(
            delta_rows,
            key=lambda row: float(row["experiment_delta_dielectric_minus_xrts_ev"]) if math.isfinite(float(row["experiment_delta_dielectric_minus_xrts_ev"])) else 1.0e30,
        )
    lines = [
        "# XRTS observable experiment",
        "",
        "This pass compares the existing dielectric-only benchmark path against a material-specific minimal Al XRTS observable reconstruction built on the same backend responses.",
        "",
        "## Summary",
        "",
        f"- compared modes: `{observable_mode_label(PLASMON_OBSERVABLE_MODE_DIELECTRIC)}` vs `{observable_mode_label(PLASMON_OBSERVABLE_MODE_XRTS)}`",
        f"- backend subset: {', '.join(_model_label(model) for model in MODEL_SUBSET)}",
        f"- mean experiment-facing dielectric-minus-observable MAE delta: `{float(np.mean(experiment_deltas)):.3f} eV`" if experiment_deltas else "- mean experiment-facing delta: not available",
        f"- mean matched-branch dielectric-minus-observable MAE delta: `{float(np.mean(matched_deltas)):.3f} eV`" if matched_deltas else "- mean matched-branch delta: not available",
    ]
    if best_row is not None:
        lines.extend(
            [
                f"- best experiment-facing change: `{best_row['case']}` / `{best_row['model_label']}` = `{float(best_row['experiment_delta_dielectric_minus_xrts_ev']):.3f} eV` ({best_row['experiment_judgement']})",
            ]
        )
    if worst_row is not None:
        lines.extend(
            [
                f"- worst experiment-facing change: `{worst_row['case']}` / `{worst_row['model_label']}` = `{float(worst_row['experiment_delta_dielectric_minus_xrts_ev']):.3f} eV` ({worst_row['experiment_judgement']})",
            ]
        )
    lines.extend(
        [
            "",
            "## Judgement",
            "",
            "The first material-specific observable layer does **not** close most of the residual.",
            "",
            "- ambient comparisons stay effectively unchanged",
            "- driven RPA and driven RPA + static LFC improve only marginally",
            "- driven finite-T STLS degrades strongly once the observable is reconstructed and the inelastic branch is extracted after elastic-core exclusion",
            "- therefore the remaining gap is not solved by a minimal free+elastic Chihara-like reconstruction alone",
            "",
            "## Observable Interpretation",
            "",
            "The observable layer does not change the backend dielectric itself. It changes the comparison level by adding explicit free/electric/bound bookkeeping and extracting the plasmon from the convolved inelastic branch instead of the raw backend DSF peak.",
            "",
            "## Component Notes",
            "",
        ]
    )
    primary_components = [row for row in component_rows if str(row["observable_mode"]) == PLASMON_OBSERVABLE_MODE_XRTS]
    for row in primary_components:
        lines.append(
            f"- `{row['case']}` / `{row['model_label']}`: free `{float(row['free_fraction']):.3f}`, bound `{float(row['bound_fraction']):.3f}`, elastic `{float(row['elastic_fraction']):.3f}`"
        )
    lines.extend(
        [
            "",
            "## Convolution Sensitivity",
            "",
        ]
    )
    for row in sensitivity_rows:
        lines.append(
            f"- `{row['model_label']}` at FWHM `{float(row['instrument_fwhm_ev']):.1f} eV`: peak `{float(row['peak_energy_ev']):.3f} eV`, width `{float(row['peak_fwhm_ev']):.3f} eV`"
        )
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark dielectric vs material-specific XRTS observable modes for the article-facing Al plasmon cases.")
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

    delta_rows = _delta_rows(comparison_rows)
    component_rows = _component_breakdown_rows(spectra_rows)
    provenance_rows = _observable_provenance_rows(comparison_rows)
    sensitivity_rows = _convolution_sensitivity_rows()

    _write_csv(out_dir / "benchmark_summary.csv", comparison_rows)
    _write_csv(out_dir / "response_model_comparison.csv", comparison_rows)
    _write_csv(out_dir / "control_vs_observable_delta.csv", delta_rows)
    _write_csv(out_dir / "xrts_spectra_comparison.csv", spectra_rows)
    _write_csv(out_dir / "component_breakdown.csv", component_rows)
    _write_csv(out_dir / "observable_provenance.csv", provenance_rows)
    _write_csv(out_dir / "convolution_sensitivity.csv", sensitivity_rows)
    _write_report(
        out_dir=out_dir,
        comparison_rows=comparison_rows,
        delta_rows=delta_rows,
        component_rows=component_rows,
        sensitivity_rows=sensitivity_rows,
    )
    _write_next_step_note(report_dir=out_dir, delta_rows=delta_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
