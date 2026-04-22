from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    import _script_bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from scripts import _script_bootstrap  # type: ignore  # noqa: F401

from helios.services.derived.plasmon_config import PLASMON_MODEL_AUTO_BEST, PLASMON_MODEL_QUANTUM_HYDRODYNAMIC, PLASMON_MODEL_QUICKLOOK
from helios.services.derived.plasmon_driven_response import (
    PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE,
    PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED,
    PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
    PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE,
    PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
    driven_response_model_label,
)

try:
    from benchmark_plasmon_article_cases import build_report
    from benchmark_plasmon_experimental_response_model import (
        FROZEN_BASELINE_DIR,
        PRIMARY_CASE_POLICIES,
        _decorate_rows,
        _equivalence_rows,
        _float,
        _judged_delta,
        _read_csv_rows,
        _summary_comparison_rows,
        _write_csv,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.benchmark_plasmon_article_cases import build_report  # type: ignore
    from scripts.benchmark_plasmon_experimental_response_model import (  # type: ignore
        FROZEN_BASELINE_DIR,
        PRIMARY_CASE_POLICIES,
        _decorate_rows,
        _equivalence_rows,
        _float,
        _judged_delta,
        _read_csv_rows,
        _summary_comparison_rows,
        _write_csv,
    )


DEFAULT_OUTPUT_DIR = Path("outputs/validation_outputs/plasmon_new_backend_experiment")
SELECTION_MEMO_PATH = Path("outputs/validation_outputs/plasmon_new_backend_selection.md")
RESPONSE_MODEL_RUNS = (
    ("noop", PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
    ("scalar_control", PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL),
    ("electron_column_weighted_control", PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED),
    ("collision_shape_broadened_experimental", PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE),
    ("response_function_ensemble_experimental", PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE),
)
_EXCLUDED_SCALAR_BASELINE_MODELS = {
    PLASMON_MODEL_QUICKLOOK,
    PLASMON_MODEL_AUTO_BEST,
    PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
}


def _selection_memo_lines() -> list[str]:
    return [
        "# New plasmon backend selection",
        "",
        "Selected backend: **quantum hydrodynamic (QHD)**.",
        "",
        "Candidates considered:",
        "- **Quantum hydrodynamic dielectric backend** (selected): changes the response object itself, keeps density-sensitive collective pressure and Bohm recoil explicit, and fits the current benchmark architecture without hidden article-specific hacks.",
        "- **Denser many-body / static-structure backend** (not selected now): would likely need a trustworthy STLS/TDDFT-like implementation or external reference data that are not present in the repo.",
        "- **Article-native calculation ingestion** (blocked): the repo still has digitized reference curves, but not article-native response tables or executable calculation assets.",
        "- **Stronger dense-collision dielectric variant** (rejected now): too close to the current closure family and too likely to become another disguised surrogate instead of a real backend jump.",
        "",
        "Why QHD was selected:",
        "- It is genuinely distinct from the existing classical Maxwellian and finite-T Lindhard families.",
        "- It changes the dielectric response itself rather than fields, weights, or final-spectrum post-processing.",
        "- It remains compatible with the current per-state / LOS-integrated / article-benchmark harness.",
        "- It does not require external libraries or article-native tables that the repo does not have.",
        "",
        "Physics change introduced:",
        "- The new backend evaluates a damped quantum-fluid dielectric",
        "- `epsilon = 1 - omega_p^2 / (omega*(omega + i nu) - beta_eff^2 q^2 - omega_B^2)`",
        "- with `beta_eff^2 = 3 v_th^2 + 3/5 v_F^2` and Bohm recoil retained explicitly.",
        "",
        "Important limitation:",
        "- This is still an experimental collective-fluid proxy, not a recovered article-native backend or a many-body TDDFT/STLS solver.",
    ]


def _read_decorated_subrun_rows(subdir: Path, *, slug: str, response_model: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    benchmark_rows = _decorate_rows(_read_csv_rows(subdir / "benchmark_summary.csv"), slug=slug, model=response_model)
    reconciliation_rows = _decorate_rows(_read_csv_rows(subdir / "reconciliation_summary.csv"), slug=slug, model=response_model)
    policy_rows = _decorate_rows(_read_csv_rows(subdir / "policy_state_summary.csv"), slug=slug, model=response_model)
    point_rows = _decorate_rows(_read_csv_rows(subdir / "benchmark_points.csv"), slug=slug, model=response_model)
    return benchmark_rows, reconciliation_rows, policy_rows, point_rows


def _metric_value(row: dict[str, object]) -> float:
    matched = _float(row, "matched_branch_mae_ev")
    if math.isfinite(matched):
        return matched
    return _float(row, "experiment_mae_ev")


def _best_same_reference_scalar_control_rows(comparison_rows: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    best: dict[tuple[str, str], dict[str, object]] = {}
    for row in comparison_rows:
        if str(row.get("response_model", "")) != PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL:
            continue
        case_name = str(row.get("case", ""))
        electron_policy = str(row.get("electron_policy", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != electron_policy:
            continue
        model = str(row.get("model", ""))
        if model in _EXCLUDED_SCALAR_BASELINE_MODELS:
            continue
        published_branch = str(row.get("published_branch", ""))
        if not published_branch:
            continue
        metric = _metric_value(row)
        if not math.isfinite(metric):
            continue
        key = (case_name, electron_policy, published_branch)
        incumbent = best.get(key)
        if incumbent is None or metric < _metric_value(incumbent):
            best[key] = row
    return best


def _new_backend_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        row
        for row in comparison_rows
        if str(row.get("model", "")) == PLASMON_MODEL_QUANTUM_HYDRODYNAMIC
        and PRIMARY_CASE_POLICIES.get(str(row.get("case", ""))) == str(row.get("electron_policy", ""))
    ]
    rows.sort(key=lambda row: (str(row.get("case", "")), str(row.get("response_model", ""))))
    return rows


def _delta_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    backend_rows = _new_backend_rows(comparison_rows)
    scalar_same_backend = {
        (str(row["case"]), str(row["electron_policy"])): row
        for row in backend_rows
        if str(row.get("response_model", "")) == PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL
    }
    best_scalar_control = _best_same_reference_scalar_control_rows(comparison_rows)
    rows: list[dict[str, object]] = []
    for row in backend_rows:
        case_name = str(row["case"])
        electron_policy = str(row["electron_policy"])
        published_branch = str(row.get("published_branch", ""))
        same_backend_control = scalar_same_backend.get((case_name, electron_policy))
        best_existing = best_scalar_control.get((case_name, electron_policy, published_branch))
        current_metric = _metric_value(row)
        control_metric = _metric_value(same_backend_control) if same_backend_control is not None else float("nan")
        best_existing_metric = _metric_value(best_existing) if best_existing is not None else float("nan")
        rows.append(
            {
                "case": case_name,
                "response_model_slug": str(row.get("response_model_slug", "")),
                "response_model": str(row.get("response_model", "")),
                "response_model_label": str(row.get("response_model_label", "")),
                "model": str(row.get("model", "")),
                "model_label": str(row.get("model_label", "")),
                "electron_policy": electron_policy,
                "published_branch": published_branch,
                "comparison_kind": str(row.get("comparison_kind", "")),
                "ambient_experiment_mae_ev": (_float(row, "experiment_mae_ev") if case_name == "ambient_al_t0" else float("nan")),
                "driven_matched_branch_mae_ev": (_float(row, "matched_branch_mae_ev") if case_name != "ambient_al_t0" else float("nan")),
                "current_metric_ev": current_metric,
                "scalar_control_same_backend_metric_ev": control_metric,
                "delta_vs_scalar_control_same_backend_ev": (control_metric - current_metric if math.isfinite(control_metric) and math.isfinite(current_metric) else float("nan")),
                "scalar_control_same_reference_model": (str(best_existing.get("model", "")) if best_existing is not None else ""),
                "scalar_control_same_reference_model_label": (str(best_existing.get("model_label", "")) if best_existing is not None else ""),
                "scalar_control_same_reference_metric_ev": best_existing_metric,
                "delta_vs_same_reference_control_ev": (best_existing_metric - current_metric if math.isfinite(best_existing_metric) and math.isfinite(current_metric) else float("nan")),
                "judged_effect_vs_same_reference": _judged_delta(best_existing_metric, current_metric),
            }
        )
    return rows


def _backend_provenance_rows(
    policy_rows: list[dict[str, object]],
    backend_rows: list[dict[str, object]],
    backend_point_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    backend_map = {
        (
            str(row.get("response_model", "")),
            str(row.get("case", "")),
            str(row.get("electron_policy", "")),
        ): row
        for row in backend_rows
    }
    backend_summary_map: dict[tuple[str, str, str], str] = {}
    for row in backend_point_rows:
        if str(row.get("model", "")) != PLASMON_MODEL_QUANTUM_HYDRODYNAMIC:
            continue
        key = (
            str(row.get("response_model", "")),
            str(row.get("case", "")),
            str(row.get("electron_policy", "")),
        )
        summary = str(row.get("backend_summary", "")).strip()
        if key not in backend_summary_map and summary:
            backend_summary_map[key] = summary
    rows: list[dict[str, object]] = []
    for row in policy_rows:
        case_name = str(row.get("case", ""))
        electron_policy = str(row.get("electron_policy", ""))
        response_model = str(row.get("response_model", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != electron_policy:
            continue
        backend_row = backend_map.get((response_model, case_name, electron_policy))
        rows.append(
            {
                "response_model_slug": str(row.get("response_model_slug", "")),
                "response_model": response_model,
                "response_model_label": str(row.get("response_model_label", "")),
                "case": case_name,
                "electron_policy": electron_policy,
                "baseline_mode": str(row.get("baseline_mode", "")),
                "increment_mode": str(row.get("increment_mode", "")),
                "baseline_entries": str(row.get("baseline_entries", "")),
                "driven_response_summary": str(row.get("driven_response_summary", "")),
                "driven_response_weight_mode": str(row.get("driven_response_weight_mode", "")),
                "driven_response_shape_mode": str(row.get("driven_response_shape_mode", "")),
                "driven_response_ensemble_mode": str(row.get("driven_response_ensemble_mode", "")),
                "new_backend": PLASMON_MODEL_QUANTUM_HYDRODYNAMIC,
                "new_backend_label": "Quantum hydrodynamic (experimental)",
                "new_backend_summary": backend_summary_map.get((response_model, case_name, electron_policy), str(backend_row.get("backend_summary", "")) if backend_row is not None else ""),
            }
        )
    rows.sort(key=lambda row: (str(row["case"]), str(row["response_model"])))
    return rows


def _backend_diagnostics(point_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in point_rows:
        if str(row.get("model", "")) != PLASMON_MODEL_QUANTUM_HYDRODYNAMIC:
            continue
        case_name = str(row.get("case", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != str(row.get("electron_policy", "")):
            continue
        rows.append(
            {
                "case": case_name,
                "response_model_slug": str(row.get("response_model_slug", "")),
                "response_model": str(row.get("response_model", "")),
                "response_model_label": str(row.get("response_model_label", "")),
                "q_ang_inv": _float(row, "q_ang_inv"),
                "angle_deg": _float(row, "angle_deg"),
                "status": str(row.get("status", "")),
                "backend": str(row.get("backend", "")),
                "backend_summary": str(row.get("backend_summary", "")),
                "runtime_s": _float(row, "runtime_s"),
                "peak_energy_ev": _float(row, "peak_energy_ev"),
                "peak_fwhm_ev": _float(row, "peak_fwhm_ev"),
                "warnings": str(row.get("warnings", "")),
            }
        )
    rows.sort(key=lambda row: (str(row["case"]), str(row["response_model"]), float(row["q_ang_inv"])))
    return rows


def _report_lines(
    *,
    dataset: str,
    out_dir: Path,
    comparison_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    equivalence_rows: list[dict[str, object]],
) -> list[str]:
    backend_rows = _new_backend_rows(comparison_rows)
    ambient_rows = [row for row in delta_rows if str(row["case"]) == "ambient_al_t0"]
    driven_rows = [row for row in delta_rows if str(row["case"]) == "driven_al_article_state"]
    equivalence_ok = all(bool(row.get("within_tolerance")) for row in equivalence_rows) if equivalence_rows else False
    best_driven = min(
        (row for row in driven_rows if math.isfinite(float(row["current_metric_ev"]))),
        key=lambda row: float(row["current_metric_ev"]),
        default=None,
    )
    best_same_reference_control = min(
        (
            row
            for row in driven_rows
            if math.isfinite(float(row["scalar_control_same_reference_metric_ev"]))
        ),
        key=lambda row: float(row["scalar_control_same_reference_metric_ev"]),
        default=None,
    )
    ambient_max = max(
        (abs(float(row["delta_vs_scalar_control_same_backend_ev"])) for row in ambient_rows if math.isfinite(float(row["delta_vs_scalar_control_same_backend_ev"]))),
        default=0.0,
    )
    lines = [
        "# New backend experiment",
        "",
        f"- dataset: **{dataset}**",
        "- controls compared:",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_NONE)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE)}",
        "- new backend: **Quantum hydrodynamic (experimental)**",
        "",
        "## 1. Backend selection",
        "",
        "- Selected backend: **quantum hydrodynamic (QHD)**.",
        "- It is structurally different because it replaces the response object with a damped quantum-fluid dielectric rather than altering fields, weights, broadening, or ensemble mixing inside the old backend family.",
        "",
        "## 2. Control preservation",
        "",
        f"- Scalar control equivalence against the frozen driven-increment baseline: **{'passed' if equivalence_ok else 'check required'}**.",
        f"- Equivalence rows written: **{len(equivalence_rows)}**.",
        "",
        "## 3. Ambient headline effects",
        "",
        f"- Maximum ambient delta between the new backend and its scalar-control variant: **{ambient_max:.6f} eV**.",
        "",
        "| response model | ambient experiment MAE [eV] | delta vs scalar-control same backend [eV] |",
        "|---|---:|---:|",
    ]
    for row in ambient_rows:
        lines.append(
            f"| {row['response_model_label']} | "
            + (f"{float(row['ambient_experiment_mae_ev']):.3f}" if math.isfinite(float(row["ambient_experiment_mae_ev"])) else "-")
            + " | "
            + (f"{float(row['delta_vs_scalar_control_same_backend_ev']):+.3f}" if math.isfinite(float(row["delta_vs_scalar_control_same_backend_ev"])) else "-")
            + " |"
        )
    lines.extend(
        [
            "",
            "## 4. Driven headline effects",
            "",
            "| response model | published branch | new backend matched MAE [eV] | delta vs scalar same-backend [eV] | delta vs best same-reference scalar-control branch [eV] | judgement |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in driven_rows:
        lines.append(
            f"| {row['response_model_label']} | {row['published_branch'] or '-'} | "
            + (f"{float(row['driven_matched_branch_mae_ev']):.3f}" if math.isfinite(float(row["driven_matched_branch_mae_ev"])) else "-")
            + " | "
            + (f"{float(row['delta_vs_scalar_control_same_backend_ev']):+.3f}" if math.isfinite(float(row["delta_vs_scalar_control_same_backend_ev"])) else "-")
            + " | "
            + (f"{float(row['delta_vs_same_reference_control_ev']):+.3f}" if math.isfinite(float(row["delta_vs_same_reference_control_ev"])) else "-")
            + f" | {row['judged_effect_vs_same_reference']} |"
        )
    lines.extend(["", "## 5. Judgement", ""])
    if best_driven is not None:
        lines.append(
            f"- Best driven new-backend variant: **{best_driven['response_model_label']}** with matched MAE **{float(best_driven['current_metric_ev']):.3f} eV** against **{best_driven['published_branch'] or '-'}**."
        )
    if best_same_reference_control is not None:
        lines.append(
            f"- Best same-reference scalar-control comparator remains **{best_same_reference_control['scalar_control_same_reference_model_label']}** at **{float(best_same_reference_control['scalar_control_same_reference_metric_ev']):.3f} eV**."
        )
    headline_delta = (
        float(best_same_reference_control["delta_vs_same_reference_control_ev"])
        if best_same_reference_control is not None and math.isfinite(float(best_same_reference_control["delta_vs_same_reference_control_ev"]))
        else float("nan")
    )
    if math.isfinite(headline_delta) and headline_delta >= 1.0:
        lines.append("- The new backend gives a **defensible material gain** over the strongest same-reference scalar-control branch.")
    elif math.isfinite(headline_delta) and headline_delta >= 0.25:
        lines.append("- The new backend improves on the strongest same-reference scalar-control branch, but the gain is still moderate.")
    elif math.isfinite(headline_delta) and headline_delta > -0.25:
        lines.append("- The new backend gives only **marginal change** relative to the strongest same-reference scalar-control branch.")
    else:
        lines.append("- The new backend **does not beat the strongest same-reference scalar-control branch** and likely worsens or redistributes error.")
    lines.extend(
        [
            "",
            "## 6. Generated artifacts",
            "",
            f"- `{out_dir / 'report.md'}`",
            f"- `{out_dir / 'benchmark_summary.csv'}`",
            f"- `{out_dir / 'response_model_comparison.csv'}`",
            f"- `{out_dir / 'control_vs_new_backend_delta.csv'}`",
            f"- `{out_dir / 'experimental_model_provenance.csv'}`",
            f"- `{out_dir / 'backend_diagnostics.csv'}`",
            f"- `{out_dir / 'response_model_equivalence.csv'}`",
            f"- `{out_dir / 'backend_selection_summary.md'}`",
        ]
    )
    del backend_rows
    return lines


def run(hydro_path: Path, *, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_lines = _selection_memo_lines()
    (out_dir / "backend_selection_summary.md").write_text("\n".join(selection_lines) + "\n", encoding="utf-8")
    SELECTION_MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_MEMO_PATH.write_text("\n".join(selection_lines) + "\n", encoding="utf-8")

    benchmark_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for slug, response_model in RESPONSE_MODEL_RUNS:
        run_dir = out_dir / slug
        build_report(hydro_path, out_dir=run_dir, driven_response_model=str(response_model))
        run_benchmark_rows, run_reconciliation_rows, run_policy_rows, run_point_rows = _read_decorated_subrun_rows(
            run_dir,
            slug=slug,
            response_model=str(response_model),
        )
        benchmark_rows.extend(run_benchmark_rows)
        reconciliation_rows.extend(run_reconciliation_rows)
        policy_rows.extend(run_policy_rows)
        point_rows.extend(run_point_rows)

    comparison_rows = _summary_comparison_rows(benchmark_rows, reconciliation_rows)
    delta_rows = _delta_rows(comparison_rows)
    provenance_rows = _backend_provenance_rows(policy_rows, _new_backend_rows(comparison_rows), point_rows)
    diagnostics_rows = _backend_diagnostics(point_rows)
    equivalence_rows = _equivalence_rows(out_dir / "scalar_control")

    _write_csv(out_dir / "benchmark_summary.csv", comparison_rows)
    _write_csv(out_dir / "response_model_comparison.csv", comparison_rows)
    _write_csv(out_dir / "control_vs_new_backend_delta.csv", delta_rows)
    _write_csv(out_dir / "experimental_model_provenance.csv", provenance_rows)
    _write_csv(out_dir / "backend_diagnostics.csv", diagnostics_rows)
    _write_csv(out_dir / "response_model_equivalence.csv", equivalence_rows)

    report_lines = _report_lines(
        dataset=hydro_path.name,
        out_dir=out_dir,
        comparison_rows=comparison_rows,
        delta_rows=delta_rows,
        equivalence_rows=equivalence_rows,
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the first genuinely distinct plasmon backend against the driven-response controls.")
    parser.add_argument(
        "--dataset",
        default="50Al+10E+25CH+3.5TW_stabilized.h5",
        help="Path to the benchmark HDF5 dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the new-backend experiment artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    hydro_path = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    return run(hydro_path, out_dir=out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
