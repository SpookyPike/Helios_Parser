from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    import _script_bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from scripts import _script_bootstrap  # type: ignore  # noqa: F401

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


DEFAULT_OUTPUT_DIR = Path("outputs/validation_outputs/plasmon_higher_level_model_jump")
SELECTION_MEMO_PATH = Path("outputs/validation_outputs/plasmon_higher_level_branch_selection.md")
RESPONSE_MODEL_RUNS = (
    ("noop", PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
    ("scalar_control", PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL),
    ("electron_column_weighted_control", PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED),
    ("collision_shape_broadened_experimental", PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE),
    ("response_function_ensemble_experimental", PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE),
)
_CLASSICAL_MODELS = {"rpa", "mermin", "rpa_static_lfc", "mermin_static_lfc"}
_LINDHARD_MODELS = {"lindhard", "lindhard_mermin", "lindhard_static_lfc", "lindhard_mermin_static_lfc"}


def _selection_memo_lines() -> list[str]:
    return [
        "# Higher-level plasmon branch selection",
        "",
        "Selected branch: **Branch C — true ensemble-response formulation at the response-function level**.",
        "",
        "Why this branch was selected:",
        "- It changes the response object itself, not just scalar fields, weights, or post-hoc broadening.",
        "- It is compatible with the current benchmark harness because state-resolved dielectric arrays already exist in the per-state benchmark results.",
        "- It preserves all current controls exactly and can be run as an explicit experimental comparison path.",
        "- It avoids inventing a fake new backend or another disguised scalar closure tweak.",
        "",
        "Why the other branches were not chosen now:",
        "- Branch A, different response backend: no genuinely distinct article-facing backend is recoverable from the repo today; the likely result would be a thin wrapper around the current families.",
        "- Branch B, more explicit finite-density collision/dielectric treatment: the current residual gap is no longer dominated by closure tuning, and another collision-side change would likely become another local surrogate.",
        "- Branch D, article-native ingestion/reproduction: the repo contains digitized reference series, but not article-native response tables or executable article-side calculation assets.",
        "",
        "Exact structural change introduced by Branch C:",
        "- The synthetic driven article ensemble can now be combined at the dielectric-response level.",
        "- For the new experimental branch, statewise complex dielectric functions are averaged before computing the loss function and spectrum.",
        "- This is different from the existing control paths, which preserve the current benchmark baseline and average final state spectra.",
    ]


def _delta_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in comparison_rows:
        key = (str(row["case"]), str(row["model"]), str(row["electron_policy"]))
        grouped.setdefault(key, {})[str(row["response_model"])] = row
    rows: list[dict[str, object]] = []
    for (case_name, model, electron_policy), variants in sorted(grouped.items()):
        noop = variants.get(PLASMON_DRIVEN_RESPONSE_MODEL_NONE, {})
        control = variants.get(PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL, {})
        weighted = variants.get(PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED, {})
        shape = variants.get(PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE, {})
        higher = variants.get(PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE, {})
        rows.append(
            {
                "case": case_name,
                "model": model,
                "model_label": str(
                    control.get(
                        "model_label",
                        higher.get("model_label", weighted.get("model_label", shape.get("model_label", noop.get("model_label", model)))),
                    )
                ),
                "electron_policy": electron_policy,
                "published_branch": str(
                    control.get(
                        "published_branch",
                        higher.get("published_branch", weighted.get("published_branch", shape.get("published_branch", noop.get("published_branch", "")))),
                    )
                ),
                "experiment_mae_noop_ev": _float(noop, "experiment_mae_ev"),
                "experiment_mae_control_ev": _float(control, "experiment_mae_ev"),
                "experiment_mae_weighted_ev": _float(weighted, "experiment_mae_ev"),
                "experiment_mae_shape_ev": _float(shape, "experiment_mae_ev"),
                "experiment_mae_higher_level_ev": _float(higher, "experiment_mae_ev"),
                "matched_branch_mae_noop_ev": _float(noop, "matched_branch_mae_ev"),
                "matched_branch_mae_control_ev": _float(control, "matched_branch_mae_ev"),
                "matched_branch_mae_weighted_ev": _float(weighted, "matched_branch_mae_ev"),
                "matched_branch_mae_shape_ev": _float(shape, "matched_branch_mae_ev"),
                "matched_branch_mae_higher_level_ev": _float(higher, "matched_branch_mae_ev"),
                "matched_branch_delta_control_to_weighted_ev": (
                    _float(control, "matched_branch_mae_ev") - _float(weighted, "matched_branch_mae_ev")
                    if math.isfinite(_float(control, "matched_branch_mae_ev")) and math.isfinite(_float(weighted, "matched_branch_mae_ev"))
                    else float("nan")
                ),
                "matched_branch_delta_control_to_shape_ev": (
                    _float(control, "matched_branch_mae_ev") - _float(shape, "matched_branch_mae_ev")
                    if math.isfinite(_float(control, "matched_branch_mae_ev")) and math.isfinite(_float(shape, "matched_branch_mae_ev"))
                    else float("nan")
                ),
                "matched_branch_delta_control_to_higher_level_ev": (
                    _float(control, "matched_branch_mae_ev") - _float(higher, "matched_branch_mae_ev")
                    if math.isfinite(_float(control, "matched_branch_mae_ev")) and math.isfinite(_float(higher, "matched_branch_mae_ev"))
                    else float("nan")
                ),
                "judged_effect_higher_level": _judged_delta(
                    _float(control, "matched_branch_mae_ev") if math.isfinite(_float(control, "matched_branch_mae_ev")) else _float(control, "experiment_mae_ev"),
                    _float(higher, "matched_branch_mae_ev") if math.isfinite(_float(higher, "matched_branch_mae_ev")) else _float(higher, "experiment_mae_ev"),
                ),
            }
        )
    return rows


def _family_delta(delta_rows: list[dict[str, object]], *, case_name: str, models: set[str]) -> float:
    values = [
        float(row["matched_branch_delta_control_to_higher_level_ev"])
        for row in delta_rows
        if str(row["case"]) == case_name
        and str(row["model"]) in models
        and math.isfinite(float(row["matched_branch_delta_control_to_higher_level_ev"]))
    ]
    return float(sum(values) / len(values)) if values else float("nan")


def _response_ensemble_diagnostics(point_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in point_rows:
        if str(row.get("response_model", "")) != PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE:
            continue
        rows.append(
            {
                "case": str(row.get("case", "")),
                "model": str(row.get("model", "")),
                "model_label": str(row.get("model_label", "")),
                "electron_policy": str(row.get("electron_policy", "")),
                "q_ang_inv": _float(row, "q_ang_inv"),
                "peak_energy_ev": _float(row, "peak_energy_ev"),
                "peak_fwhm_ev": _float(row, "peak_fwhm_ev"),
                "status": str(row.get("status", "")),
                "backend": str(row.get("backend", "")),
                "driven_response_model": str(row.get("driven_response_model", "")),
                "driven_response_ensemble_mode": str(row.get("driven_response_ensemble_mode", "")),
                "warnings": str(row.get("warnings", "")),
            }
        )
    return rows


def _provenance_rows(policy_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in policy_rows:
        case_name = str(row.get("case", ""))
        policy = str(row.get("electron_policy", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != policy:
            continue
        if str(row.get("response_model", "")) != PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE:
            continue
        rows.append(
            {
                "case": case_name,
                "electron_policy": policy,
                "response_model": str(row.get("response_model", "")),
                "response_model_label": str(row.get("response_model_label", "")),
                "baseline_mode": str(row.get("baseline_mode", "")),
                "increment_mode": str(row.get("increment_mode", "")),
                "driven_response_summary": str(row.get("driven_response_summary", "")),
                "driven_response_weight_mode": str(row.get("driven_response_weight_mode", "")),
                "driven_response_shape_mode": str(row.get("driven_response_shape_mode", "")),
                "driven_response_ensemble_mode": str(row.get("driven_response_ensemble_mode", "")),
                "baseline_entries": str(row.get("baseline_entries", "")),
            }
        )
    return rows


def _report_lines(
    *,
    dataset: str,
    out_dir: Path,
    delta_rows: list[dict[str, object]],
    equivalence_rows: list[dict[str, object]],
) -> list[str]:
    ambient_rows = [row for row in delta_rows if str(row["case"]) == "ambient_al_t0"]
    driven_rows = [row for row in delta_rows if str(row["case"]) == "driven_al_article_state"]
    ambient_max_shift = max(
        (
            abs(float(row["experiment_mae_control_ev"]) - float(row["experiment_mae_higher_level_ev"]))
            for row in ambient_rows
            if math.isfinite(float(row["experiment_mae_control_ev"])) and math.isfinite(float(row["experiment_mae_higher_level_ev"]))
        ),
        default=0.0,
    )
    comparable_driven = [row for row in driven_rows if math.isfinite(float(row["matched_branch_delta_control_to_higher_level_ev"]))]
    comparable_driven.sort(key=lambda row: float(row["matched_branch_delta_control_to_higher_level_ev"]), reverse=True)
    best_higher = comparable_driven[0] if comparable_driven else None
    classical_mean_delta = _family_delta(delta_rows, case_name="driven_al_article_state", models=_CLASSICAL_MODELS)
    lindhard_mean_delta = _family_delta(delta_rows, case_name="driven_al_article_state", models=_LINDHARD_MODELS)
    equivalence_ok = all(bool(row.get("within_tolerance")) for row in equivalence_rows) if equivalence_rows else False
    lines = [
        "# Higher-level plasmon model jump",
        "",
        f"- dataset: **{dataset}**",
        "- models compared:",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_NONE)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_RESPONSE_ENSEMBLE)}",
        "",
        "## 1. Branch selection",
        "",
        "- Selected higher-level branch: **true ensemble-response formulation at the response-function level**.",
        "- The new experimental path averages statewise dielectric response before loss/spectrum extraction.",
        "",
        "## 2. Control preservation",
        "",
        f"- Scalar control equivalence against the frozen driven-increment baseline: **{'passed' if equivalence_ok else 'check required'}**.",
        f"- Equivalence rows written: **{len(equivalence_rows)}**.",
        "",
        "## 3. Ambient behaviour",
        "",
        f"- Maximum ambient experiment-MAE shift between scalar control and the higher-level branch: **{ambient_max_shift:.6f} eV**.",
        "",
        "## 4. Driven article-facing behaviour",
        "",
    ]
    if best_higher is not None:
        lines.append(
            f"- Best matched-branch shift from the higher-level branch: **{best_higher['model_label']}** "
            f"({best_higher['published_branch']}) with delta MAE = "
            f"**{float(best_higher['matched_branch_delta_control_to_higher_level_ev']):+.3f} eV**."
        )
    else:
        lines.append("- No driven branch produced a finite matched-branch delta for the higher-level branch.")
    lines.extend(
        [
            f"- Mean classical-family delta vs scalar control: **{classical_mean_delta:+.3f} eV**.",
            f"- Mean Lindhard-family delta vs scalar control: **{lindhard_mean_delta:+.3f} eV**.",
            "",
            "| case | branch | control matched MAE [eV] | weighted matched MAE [eV] | shape matched MAE [eV] | higher-level matched MAE [eV] | judgement |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in driven_rows:
        lines.append(
            f"| {row['case']} | {row['model_label']} -> {row['published_branch'] or '-'} | "
            + (f"{float(row['matched_branch_mae_control_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_control_ev"])) else "-")
            + " | "
            + (f"{float(row['matched_branch_mae_weighted_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_weighted_ev"])) else "-")
            + " | "
            + (f"{float(row['matched_branch_mae_shape_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_shape_ev"])) else "-")
            + " | "
            + (f"{float(row['matched_branch_mae_higher_level_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_higher_level_ev"])) else "-")
            + f" | {row['judged_effect_higher_level']} |"
        )
    lines.extend(["", "## 5. Judgement", ""])
    if best_higher is not None and float(best_higher["matched_branch_delta_control_to_higher_level_ev"]) >= 0.25:
        lines.append("- The higher-level branch gives a **defensible material gain**.")
        lines.append("- The gain comes from a real response-function ensemble change, not from another scalar or weighting surrogate.")
    elif best_higher is not None and float(best_higher["matched_branch_delta_control_to_higher_level_ev"]) > -0.25:
        lines.append("- The higher-level branch gives only **marginal change**.")
        lines.append("- The driven mismatch still appears to need a deeper physics upgrade than response-function ensemble mixing alone.")
    else:
        lines.append("- The higher-level branch **worsens agreement or does not materially help**.")
        lines.append("- The current framework likely needs a still larger physics jump, not another local modification.")
    lines.extend(
        [
            "",
            "## 6. Generated artifacts",
            "",
            f"- `{out_dir / 'report.md'}`",
            f"- `{out_dir / 'benchmark_summary.csv'}`",
            f"- `{out_dir / 'response_model_comparison.csv'}`",
            f"- `{out_dir / 'control_vs_new_branch_delta.csv'}`",
            f"- `{out_dir / 'experimental_model_provenance.csv'}`",
            f"- `{out_dir / 'response_model_equivalence.csv'}`",
            f"- `{out_dir / 'response_ensemble_diagnostics.csv'}`",
            f"- `{out_dir / 'branch_selection_summary.md'}`",
        ]
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the first higher-level plasmon model jump against current driven-response controls.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    run_dirs: dict[str, Path] = {}

    memo_text = "\n".join(_selection_memo_lines()) + "\n"
    (out_dir / "branch_selection_summary.md").write_text(memo_text, encoding="utf-8")
    SELECTION_MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_MEMO_PATH.write_text(memo_text, encoding="utf-8")

    for slug, model in RESPONSE_MODEL_RUNS:
        run_dir = out_dir / slug
        run_dirs[model] = run_dir
        rc = build_report(dataset_path, out_dir=run_dir, driven_response_model=model)
        if rc != 0:
            raise SystemExit(rc)
        benchmark_rows.extend(_decorate_rows(_read_csv_rows(run_dir / "benchmark_summary.csv"), slug=slug, model=model))
        policy_rows.extend(_decorate_rows(_read_csv_rows(run_dir / "policy_state_summary.csv"), slug=slug, model=model))
        reconciliation_rows.extend(_decorate_rows(_read_csv_rows(run_dir / "reconciliation_summary.csv"), slug=slug, model=model))
        point_rows.extend(_decorate_rows(_read_csv_rows(run_dir / "benchmark_points.csv"), slug=slug, model=model))

    comparison_rows = _summary_comparison_rows(benchmark_rows, reconciliation_rows)
    delta_rows = _delta_rows(comparison_rows)
    equivalence_rows = _equivalence_rows(run_dirs[PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL])
    provenance_rows = _provenance_rows(policy_rows)
    diagnostics_rows = _response_ensemble_diagnostics(point_rows)

    _write_csv(out_dir / "benchmark_summary.csv", benchmark_rows)
    _write_csv(out_dir / "policy_state_summary.csv", policy_rows)
    _write_csv(out_dir / "response_model_comparison.csv", comparison_rows)
    _write_csv(out_dir / "control_vs_new_branch_delta.csv", delta_rows)
    _write_csv(out_dir / "experimental_model_provenance.csv", provenance_rows)
    _write_csv(out_dir / "response_model_equivalence.csv", equivalence_rows)
    _write_csv(out_dir / "response_ensemble_diagnostics.csv", diagnostics_rows)
    (out_dir / "report.md").write_text(
        "\n".join(
            _report_lines(
                dataset=dataset_path.name,
                out_dir=out_dir,
                delta_rows=delta_rows,
                equivalence_rows=equivalence_rows,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
