from __future__ import annotations

import argparse
import csv
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
)
from helios.services.derived.plasmon_driven_response import (
    PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
    driven_response_model_label,
)
from helios.services.derived.plasmon_electron_policy import resolve_effective_electron_fields
from helios.services.derived.plasmon_stls import solve_static_stls_state
from helios.services.derived.plasmon_units import electron_fermi_wavevector_m_inv
from helios.services.derived.plasmon_validation import uniform_al_dataset

try:
    from benchmark_plasmon_article_cases import (
        AL_ATOMIC_WEIGHT_G_MOL,
        ARTICLE_DRIVEN_DENSITY_GRID,
        ARTICLE_DRIVEN_TEMPERATURE_EV,
        AVOGADRO,
        PRIMARY_POLICY_AMBIENT,
        PRIMARY_POLICY_DRIVEN,
        REPRESENTATIVE_Q_AMBIENT,
        REPRESENTATIVE_Q_DRIVEN,
        build_report,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.benchmark_plasmon_article_cases import (  # type: ignore
        AL_ATOMIC_WEIGHT_G_MOL,
        ARTICLE_DRIVEN_DENSITY_GRID,
        ARTICLE_DRIVEN_TEMPERATURE_EV,
        AVOGADRO,
        PRIMARY_POLICY_AMBIENT,
        PRIMARY_POLICY_DRIVEN,
        REPRESENTATIVE_Q_AMBIENT,
        REPRESENTATIVE_Q_DRIVEN,
        build_report,
    )


DEFAULT_OUTPUT_DIR = Path("outputs/validation_outputs/plasmon_stls_backend_experiment")
MODEL_SUBSET = (
    PLASMON_MODEL_RPA,
    PLASMON_MODEL_RPA_STATIC_LFC,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
    PLASMON_MODEL_FINITE_T_STLS,
)
PRIMARY_CASE_POLICIES = {
    "ambient_al_t0": PRIMARY_POLICY_AMBIENT,
    "driven_al_article_state": PRIMARY_POLICY_DRIVEN,
}
CONTROL_MODELS = (
    PLASMON_MODEL_RPA,
    PLASMON_MODEL_RPA_STATIC_LFC,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
)


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
        return "material gain"
    if delta >= 0.25:
        return "improved"
    if delta > -0.25:
        return "marginal change"
    return "worse"


def _model_label(model: str) -> str:
    labels = {
        PLASMON_MODEL_RPA: "RPA",
        PLASMON_MODEL_RPA_STATIC_LFC: "RPA + static LFC",
        PLASMON_MODEL_QUANTUM_HYDRODYNAMIC: "Quantum hydrodynamic",
        PLASMON_MODEL_FINITE_T_STLS: "Finite-T STLS",
    }
    return labels.get(str(model), str(model))


def _primary_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        case_name = str(row.get("case", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != str(row.get("electron_policy", "")):
            continue
        if str(row.get("model", "")) not in MODEL_SUBSET:
            continue
        filtered.append(row)
    return filtered


def _primary_reconciliation_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        case_name = str(row.get("case", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != str(row.get("electron_policy", "")):
            continue
        if str(row.get("model", "")) not in MODEL_SUBSET:
            continue
        filtered.append(row)
    return filtered


def _comparison_rows(
    benchmark_rows: list[dict[str, str]],
    reconciliation_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    recon_map = {
        (str(row.get("case", "")), str(row.get("model", "")), str(row.get("electron_policy", ""))): row
        for row in reconciliation_rows
    }
    rows: list[dict[str, object]] = []
    for row in benchmark_rows:
        case_name = str(row.get("case", ""))
        model = str(row.get("model", ""))
        electron_policy = str(row.get("electron_policy", ""))
        recon = recon_map.get((case_name, model, electron_policy), {})
        rows.append(
            {
                "case": case_name,
                "model": model,
                "model_label": str(row.get("model_label", _model_label(model))),
                "electron_policy": electron_policy,
                "status": str(row.get("status", "")),
                "backend": str(row.get("backend", "")),
                "backend_summary": str(row.get("backend_summary", "")),
                "runtime_mean_s": _float(row, "runtime_mean_s"),
                "experiment_mae_ev": _float(row, "mae_ev"),
                "experiment_rmse_ev": _float(row, "rmse_ev"),
                "published_branch": str(recon.get("published_branch", "")),
                "comparison_kind": str(recon.get("comparison_kind", "")),
                "matched_branch_mae_ev": _float(recon, "mae_ev"),
                "matched_branch_rmse_ev": _float(recon, "rmse_ev"),
                "comparison_note": str(recon.get("comparison_note", "")),
                "stls_converged_all": str(row.get("stls_converged_all", "")),
                "stls_iteration_mean": _float(row, "stls_iteration_mean"),
                "stls_residual_mean": _float(row, "stls_residual_mean"),
                "stls_relative_residual_mean": _float(row, "stls_relative_residual_mean"),
                "stls_closure_name": str(row.get("stls_closure_name", "")),
            }
        )
    rows.sort(key=lambda item: (str(item["case"]), str(item["model"])))
    return rows


def _delta_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparison_map = {
        (str(row["case"]), str(row["model"]), str(row["electron_policy"])): row
        for row in comparison_rows
    }
    rows: list[dict[str, object]] = []
    for case_name, electron_policy in PRIMARY_CASE_POLICIES.items():
        stls_row = comparison_map.get((case_name, PLASMON_MODEL_FINITE_T_STLS, electron_policy))
        if stls_row is None:
            continue
        for control_model in CONTROL_MODELS:
            control_row = comparison_map.get((case_name, control_model, electron_policy))
            if control_row is None:
                continue
            stls_experiment = _float(stls_row, "experiment_mae_ev")
            control_experiment = _float(control_row, "experiment_mae_ev")
            rows.append(
                {
                    "case": case_name,
                    "comparison_axis": "experiment",
                    "reference_branch": "experiment",
                    "control_model": control_model,
                    "control_model_label": str(control_row["model_label"]),
                    "control_backend": str(control_row["backend"]),
                    "stls_model": PLASMON_MODEL_FINITE_T_STLS,
                    "stls_model_label": str(stls_row["model_label"]),
                    "stls_backend": str(stls_row["backend"]),
                    "same_reference": False,
                    "control_metric_ev": control_experiment,
                    "stls_metric_ev": stls_experiment,
                    "delta_control_minus_stls_ev": (
                        control_experiment - stls_experiment
                        if math.isfinite(control_experiment) and math.isfinite(stls_experiment)
                        else float("nan")
                    ),
                    "judged_effect": _judged_delta(control_experiment, stls_experiment),
                }
            )
            stls_branch = str(stls_row.get("published_branch", ""))
            control_branch = str(control_row.get("published_branch", ""))
            stls_matched = _float(stls_row, "matched_branch_mae_ev")
            control_matched = _float(control_row, "matched_branch_mae_ev")
            if stls_branch and stls_branch == control_branch:
                rows.append(
                    {
                        "case": case_name,
                        "comparison_axis": "matched_branch",
                        "reference_branch": stls_branch,
                        "control_model": control_model,
                        "control_model_label": str(control_row["model_label"]),
                        "control_backend": str(control_row["backend"]),
                        "stls_model": PLASMON_MODEL_FINITE_T_STLS,
                        "stls_model_label": str(stls_row["model_label"]),
                        "stls_backend": str(stls_row["backend"]),
                        "same_reference": True,
                        "control_metric_ev": control_matched,
                        "stls_metric_ev": stls_matched,
                        "delta_control_minus_stls_ev": (
                            control_matched - stls_matched
                            if math.isfinite(control_matched) and math.isfinite(stls_matched)
                            else float("nan")
                        ),
                        "judged_effect": _judged_delta(control_matched, stls_matched),
                    }
                )
    rows.sort(key=lambda item: (str(item["case"]), str(item["comparison_axis"]), str(item["control_model"])))
    return rows


def _stls_convergence_rows(point_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in point_rows:
        case_name = str(row.get("case", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != str(row.get("electron_policy", "")):
            continue
        if str(row.get("model", "")) != PLASMON_MODEL_FINITE_T_STLS:
            continue
        rows.append(
            {
                "case": case_name,
                "electron_policy": str(row.get("electron_policy", "")),
                "q_ang_inv": _float(row, "q_ang_inv"),
                "angle_deg": _float(row, "angle_deg"),
                "status": str(row.get("status", "")),
                "peak_energy_ev": _float(row, "peak_energy_ev"),
                "peak_fwhm_ev": _float(row, "peak_fwhm_ev"),
                "stls_converged": str(row.get("stls_converged", "")),
                "stls_iteration_count": _float(row, "stls_iteration_count"),
                "stls_convergence_residual": _float(row, "stls_convergence_residual"),
                "stls_convergence_relative_residual": _float(row, "stls_convergence_relative_residual"),
                "stls_closure_name": str(row.get("stls_closure_name", "")),
                "stls_local_field_value": _float(row, "stls_local_field_value"),
                "stls_q_over_qf": _float(row, "stls_q_over_qf"),
                "backend_summary": str(row.get("backend_summary", "")),
            }
        )
    rows.sort(key=lambda item: (str(item["case"]), float(item["q_ang_inv"])))
    return rows


def _state_diag_row(
    *,
    case_name: str,
    state_label: str,
    density_g_cm3: float,
    te_ev: float,
    policy: str,
    representative_q_ang_inv: float,
) -> dict[str, object]:
    dataset, _context = uniform_al_dataset(float(density_g_cm3), float(te_ev))
    payload = resolve_effective_electron_fields(
        dataset,
        str(policy),
        driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
    )
    ne_cm3 = float(np.nanmean(np.asarray(payload.electron_density_cm3, dtype=np.float64)))
    ion_density_cm3 = float(float(density_g_cm3) / AL_ATOMIC_WEIGHT_G_MOL * AVOGADRO)
    effective_zeff = float(ne_cm3 / ion_density_cm3) if ion_density_cm3 > 0.0 else float("nan")
    solution = solve_static_stls_state(
        ne_cm3=ne_cm3,
        te_ev=float(te_ev),
        imag_shift_ev=1.0e-9,
        benchmark=True,
    )
    q_grid = np.asarray(solution["q_grid_m_inv"], dtype=np.float64)
    g_grid = np.asarray(solution["local_field_grid"], dtype=np.float64)
    representative_q_m_inv = float(representative_q_ang_inv) * 1.0e10
    kf = float(electron_fermi_wavevector_m_inv(ne_cm3))
    representative_local_field = (
        float(np.interp(representative_q_m_inv, q_grid, g_grid, left=float(g_grid[0]), right=float(g_grid[-1])))
        if q_grid.size and g_grid.size == q_grid.size
        else float("nan")
    )
    return {
        "case": case_name,
        "state_label": state_label,
        "density_g_cm3": float(density_g_cm3),
        "te_ev": float(te_ev),
        "policy": str(policy),
        "driven_response_model": PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        "effective_ne_cm3": ne_cm3,
        "effective_zeff": effective_zeff,
        "converged": bool(solution["converged"]),
        "iterations": int(solution["iterations"]),
        "residual": float(solution["residual"]),
        "relative_residual": float(solution["relative_residual"]),
        "closure_name": str(solution["closure_name"]),
        "q_grid_count": int(solution["q_grid_count"]),
        "energy_grid_count": int(solution["energy_grid_count"]),
        "q_grid_min_ang_inv": (float(np.nanmin(q_grid) / 1.0e10) if q_grid.size else float("nan")),
        "q_grid_max_ang_inv": (float(np.nanmax(q_grid) / 1.0e10) if q_grid.size else float("nan")),
        "local_field_min": (float(np.nanmin(g_grid)) if g_grid.size else float("nan")),
        "local_field_max": (float(np.nanmax(g_grid)) if g_grid.size else float("nan")),
        "structure_factor_min": float(solution["structure_factor_min"]),
        "structure_factor_max": float(solution["structure_factor_max"]),
        "representative_q_ang_inv": float(representative_q_ang_inv),
        "representative_local_field": representative_local_field,
        "representative_q_over_qf": (
            float(representative_q_m_inv / kf) if math.isfinite(kf) and kf > 0.0 else float("nan")
        ),
    }


def _stls_state_diagnostic_rows() -> list[dict[str, object]]:
    rows = [
        _state_diag_row(
            case_name="ambient_al_t0",
            state_label="ambient_baseline",
            density_g_cm3=2.7,
            te_ev=0.025,
            policy=PRIMARY_POLICY_AMBIENT,
            representative_q_ang_inv=REPRESENTATIVE_Q_AMBIENT,
        )
    ]
    for density in ARTICLE_DRIVEN_DENSITY_GRID:
        rows.append(
            _state_diag_row(
                case_name="driven_al_article_state",
                state_label=f"driven_density_{float(density):.3f}",
                density_g_cm3=float(density),
                te_ev=float(ARTICLE_DRIVEN_TEMPERATURE_EV),
                policy=PRIMARY_POLICY_DRIVEN,
                representative_q_ang_inv=REPRESENTATIVE_Q_DRIVEN,
            )
        )
    return rows


def _provenance_rows(
    policy_rows: list[dict[str, str]],
    benchmark_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    benchmark_map = {
        (str(row.get("case", "")), str(row.get("electron_policy", ""))): row
        for row in benchmark_rows
        if str(row.get("model", "")) == PLASMON_MODEL_FINITE_T_STLS
    }
    rows: list[dict[str, object]] = []
    for row in policy_rows:
        case_name = str(row.get("case", ""))
        electron_policy = str(row.get("electron_policy", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != electron_policy:
            continue
        benchmark_row = benchmark_map.get((case_name, electron_policy), {})
        rows.append(
            {
                "case": case_name,
                "electron_policy": electron_policy,
                "model": PLASMON_MODEL_FINITE_T_STLS,
                "model_label": _model_label(PLASMON_MODEL_FINITE_T_STLS),
                "driven_response_model": PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
                "driven_response_model_label": driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL),
                "baseline_mode": str(row.get("baseline_mode", "")),
                "increment_mode": str(row.get("increment_mode", "")),
                "baseline_entries": str(row.get("baseline_entries", "")),
                "driven_response_summary": str(row.get("driven_response_summary", "")),
                "benchmark_preset": str(row.get("benchmark_preset", "")),
                "backend": str(benchmark_row.get("backend", "")),
                "backend_summary": str(benchmark_row.get("backend_summary", "")),
                "stls_converged_all": str(benchmark_row.get("stls_converged_all", "")),
                "stls_iteration_mean": _float(benchmark_row, "stls_iteration_mean"),
                "stls_residual_mean": _float(benchmark_row, "stls_residual_mean"),
                "stls_relative_residual_mean": _float(benchmark_row, "stls_relative_residual_mean"),
                "stls_closure_name": str(benchmark_row.get("stls_closure_name", "")),
            }
        )
    rows.sort(key=lambda item: str(item["case"]))
    return rows


def _best_row(rows: list[dict[str, object]], *, case_name: str, branch: str) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if str(row["case"]) == case_name
        and str(row.get("published_branch", "")) == branch
        and math.isfinite(float(row.get("matched_branch_mae_ev", float("nan"))))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: float(row["matched_branch_mae_ev"]))
    return candidates[0]


def _summary_row_map(rows: list[dict[str, object]], *, case_name: str) -> list[dict[str, object]]:
    return [row for row in rows if str(row["case"]) == case_name]


def _report_lines(
    *,
    dataset: str,
    out_dir: Path,
    comparison_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    convergence_rows: list[dict[str, object]],
    state_rows: list[dict[str, object]],
) -> list[str]:
    ambient_rows = _summary_row_map(comparison_rows, case_name="ambient_al_t0")
    driven_rows = _summary_row_map(comparison_rows, case_name="driven_al_article_state")
    ambient_rows.sort(key=lambda row: str(row["model"]))
    driven_rows.sort(key=lambda row: str(row["model"]))
    driven_rpa = next((row for row in driven_rows if str(row["model"]) == PLASMON_MODEL_RPA), None)
    driven_qhd = next((row for row in driven_rows if str(row["model"]) == PLASMON_MODEL_QUANTUM_HYDRODYNAMIC), None)
    driven_lfc = next((row for row in driven_rows if str(row["model"]) == PLASMON_MODEL_RPA_STATIC_LFC), None)
    driven_stls = next((row for row in driven_rows if str(row["model"]) == PLASMON_MODEL_FINITE_T_STLS), None)
    ambient_best_corr = _best_row(comparison_rows, case_name="ambient_al_t0", branch="gawne")
    driven_best_rpa = _best_row(comparison_rows, case_name="driven_al_article_state", branch="rpa")
    driven_best_lfc = _best_row(comparison_rows, case_name="driven_al_article_state", branch="lfc")
    convergence_ok = all(str(row.get("stls_converged", "")).lower() == "true" for row in convergence_rows) if convergence_rows else False
    convergence_iter_mean = (
        float(np.mean([float(row["stls_iteration_count"]) for row in convergence_rows if math.isfinite(float(row["stls_iteration_count"]))]))
        if convergence_rows
        else float("nan")
    )
    convergence_rel_mean = (
        float(np.mean([float(row["stls_convergence_relative_residual"]) for row in convergence_rows if math.isfinite(float(row["stls_convergence_relative_residual"]))]))
        if convergence_rows
        else float("nan")
    )
    stls_vs_lfc = next(
        (
            row
            for row in delta_rows
            if str(row["case"]) == "driven_al_article_state"
            and str(row["comparison_axis"]) == "matched_branch"
            and str(row["reference_branch"]) == "lfc"
            and str(row["control_model"]) == PLASMON_MODEL_RPA_STATIC_LFC
        ),
        None,
    )
    experiment_deltas = [
        row
        for row in delta_rows
        if str(row["case"]) == "driven_al_article_state" and str(row["comparison_axis"]) == "experiment"
    ]
    experiment_deltas.sort(
        key=lambda row: float(row["delta_control_minus_stls_ev"])
        if math.isfinite(float(row["delta_control_minus_stls_ev"]))
        else -1.0e12,
        reverse=True,
    )
    best_experiment_delta = experiment_deltas[0] if experiment_deltas else None
    if stls_vs_lfc is not None:
        final_judgement = str(stls_vs_lfc["judged_effect"])
    elif best_experiment_delta is not None:
        final_judgement = str(best_experiment_delta["judged_effect"])
    else:
        final_judgement = "not_comparable"

    lines = [
        "# Finite-T STLS backend experiment",
        "",
        f"- dataset: **{dataset}**",
        f"- models: **{', '.join(_model_label(model) for model in MODEL_SUBSET)}**",
        f"- driven response control: **{driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL)}**",
        "- backend integrity: **real self-consistent static STLS**",
        "  - collisionless",
        "  - finite-T Lindhard ideal kernel",
        "  - explicit `G(q) -> chi(q,omega) -> S(q) -> G(q)` iteration",
        "  - no dynamic `G(q,omega)` and no qSTLS/VS compressibility enforcement in this first baseline",
        "",
        "## 1. STLS convergence",
        "",
        f"- benchmark-point convergence: **{'all converged' if convergence_ok else 'not fully converged'}**",
        f"- benchmark-point mean iterations: **{convergence_iter_mean:.2f}**",
        f"- benchmark-point mean relative residual: **{convergence_rel_mean:.3e}**",
        "",
        "| state | rho [g/cm^3] | Te [eV] | Z_eff | converged | iterations | rel residual | G(q_rep) | q_rep/kF |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in state_rows:
        lines.append(
            f"| {row['state_label']} | {float(row['density_g_cm3']):.3f} | {float(row['te_ev']):.3f} | {float(row['effective_zeff']):.3f} | {row['converged']} | {int(row['iterations'])} | {float(row['relative_residual']):.3e} | {float(row['representative_local_field']):.4f} | {float(row['representative_q_over_qf']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 2. Headline benchmark comparison",
            "",
            "### Ambient experiment-facing MAE",
            "",
            "| model | backend | runtime mean [s] | experiment MAE [eV] | matched branch | matched MAE [eV] |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for row in ambient_rows:
        lines.append(
            f"| {row['model_label']} | {row['backend']} | {float(row['runtime_mean_s']):.3f} | {float(row['experiment_mae_ev']):.3f} | {row['published_branch'] or '-'} | "
            + (f"{float(row['matched_branch_mae_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_ev"])) else "-")
            + " |"
        )
    lines.extend(
        [
            "",
            "### Driven experiment-facing MAE",
            "",
            "| model | backend | runtime mean [s] | experiment MAE [eV] | matched branch | matched MAE [eV] |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for row in driven_rows:
        lines.append(
            f"| {row['model_label']} | {row['backend']} | {float(row['runtime_mean_s']):.3f} | {float(row['experiment_mae_ev']):.3f} | {row['published_branch'] or '-'} | "
            + (f"{float(row['matched_branch_mae_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_ev"])) else "-")
            + " |"
        )
    lines.extend(
        [
            "",
            "## 3. Driven branch-to-branch comparison",
            "",
            "### RPA-like branch",
            "",
            "| model | published branch | matched MAE [eV] | note |",
            "|---|---|---:|---|",
        ]
    )
    for row in (driven_rpa, driven_qhd):
        if row is None:
            continue
        lines.append(
            f"| {row['model_label']} | {row['published_branch'] or '-'} | "
            + (f"{float(row['matched_branch_mae_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_ev"])) else "-")
            + f" | {row['comparison_kind'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "### Correlation-sensitive / LFC branch",
            "",
            "| model | published branch | matched MAE [eV] | note |",
            "|---|---|---:|---|",
        ]
    )
    for row in (driven_lfc, driven_stls):
        if row is None:
            continue
        lines.append(
            f"| {row['model_label']} | {row['published_branch'] or '-'} | "
            + (f"{float(row['matched_branch_mae_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_ev"])) else "-")
            + f" | {row['comparison_kind'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## 4. Judged comparison",
            "",
            (
                f"- Same-reference driven LFC comparison: **Finite-T STLS vs RPA + static LFC = {float(stls_vs_lfc['delta_control_minus_stls_ev']):+.3f} eV** "
                f"({stls_vs_lfc['judged_effect']})."
                if stls_vs_lfc is not None and math.isfinite(float(stls_vs_lfc["delta_control_minus_stls_ev"]))
                else "- No same-reference driven LFC delta could be formed."
            ),
            (
                f"- Best driven RPA-like proxy remains **{driven_best_rpa['model_label']}** at **{float(driven_best_rpa['matched_branch_mae_ev']):.3f} eV**."
                if driven_best_rpa is not None
                else "- No valid driven RPA-like proxy remained."
            ),
            (
                f"- Best driven LFC/correlation proxy is **{driven_best_lfc['model_label']}** at **{float(driven_best_lfc['matched_branch_mae_ev']):.3f} eV**."
                if driven_best_lfc is not None
                else "- No valid driven LFC/correlation proxy remained."
            ),
            (
                f"- Best ambient correlation-sensitive proxy is **{ambient_best_corr['model_label']}** at **{float(ambient_best_corr['matched_branch_mae_ev']):.3f} eV**."
                if ambient_best_corr is not None
                else "- No valid ambient correlation-sensitive proxy remained."
            ),
            (
                f"- Strongest experiment-facing STLS delta against an existing control is **{float(best_experiment_delta['delta_control_minus_stls_ev']):+.3f} eV** versus **{best_experiment_delta['control_model_label']}** "
                f"({best_experiment_delta['judged_effect']})."
                if best_experiment_delta is not None and math.isfinite(float(best_experiment_delta["delta_control_minus_stls_ev"]))
                else "- No experiment-facing STLS delta could be judged."
            ),
            "",
            "## 5. Final judgement",
            "",
        ]
    )
    if final_judgement == "material gain":
        lines.append("- **Real STLS implemented and gives material gain.**")
    elif final_judgement == "improved":
        lines.append("- **Real STLS implemented and improves the correlation-sensitive branch, but not at a material level.**")
    elif final_judgement == "marginal change":
        lines.append("- **Real STLS implemented but only gives marginal change.**")
    elif final_judgement == "worse":
        lines.append("- **Real STLS implemented and worsens agreement on the closest same-reference comparison.**")
    else:
        lines.append("- **Real STLS implemented, but the current benchmark comparison is not sufficient to judge a clean gain.**")
    lines.extend(
        [
            "",
            "## 6. Generated artifacts",
            "",
            f"- `{out_dir / 'benchmark_summary.csv'}`",
            f"- `{out_dir / 'control_vs_new_backend_delta.csv'}`",
            f"- `{out_dir / 'stls_convergence_summary.csv'}`",
            f"- `{out_dir / 'stls_state_diagnostics.csv'}`",
            f"- `{out_dir / 'experimental_model_provenance.csv'}`",
            f"- `{out_dir / 'all_model_results.json'}`",
            f"- `{out_dir / 'ambient_dataset_overlay.png'}`",
            f"- `{out_dir / 'driven_dataset_overlay.png'}`",
        ]
    )
    return lines


def build_stls_report(hydro_path: Path, *, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    build_report(
        hydro_path,
        out_dir=out_dir,
        driven_response_model=PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        model_subset=MODEL_SUBSET,
    )

    original_report = out_dir / "report.md"
    if original_report.exists():
        (out_dir / "article_benchmark_report_full.md").write_text(original_report.read_text(encoding="utf-8"), encoding="utf-8")

    benchmark_rows = _primary_summary_rows(_read_csv_rows(out_dir / "benchmark_summary.csv"))
    reconciliation_rows = _primary_reconciliation_rows(_read_csv_rows(out_dir / "reconciliation_summary.csv"))
    point_rows = _read_csv_rows(out_dir / "benchmark_points.csv")
    policy_rows = _read_csv_rows(out_dir / "policy_state_summary.csv")

    comparison_rows = _comparison_rows(benchmark_rows, reconciliation_rows)
    delta_rows = _delta_rows(comparison_rows)
    convergence_rows = _stls_convergence_rows(point_rows)
    state_rows = _stls_state_diagnostic_rows()
    provenance_rows = _provenance_rows(policy_rows, benchmark_rows)

    _write_csv(out_dir / "control_vs_new_backend_delta.csv", delta_rows)
    _write_csv(out_dir / "stls_convergence_summary.csv", convergence_rows)
    _write_csv(out_dir / "stls_state_diagnostics.csv", state_rows)
    _write_csv(out_dir / "experimental_model_provenance.csv", provenance_rows)

    report_lines = _report_lines(
        dataset=hydro_path.name,
        out_dir=out_dir,
        comparison_rows=comparison_rows,
        delta_rows=delta_rows,
        convergence_rows=convergence_rows,
        state_rows=state_rows,
    )
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the finite-T STLS backend against the existing article-facing Al controls.")
    parser.add_argument("--dataset", default="50Al+10E+25CH+3.5TW_stabilized.h5")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    return build_stls_report(Path(args.dataset), out_dir=Path(args.out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
