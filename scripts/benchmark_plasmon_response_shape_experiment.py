from __future__ import annotations

import argparse
import csv
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
    PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
    driven_response_model_label,
)
from helios.services.derived.plasmon_electron_policy import PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT

try:
    from benchmark_plasmon_article_cases import (
        ARTICLE_DRIVEN_DENSITY_GRID,
        ARTICLE_DRIVEN_TEMPERATURE_EV,
        build_report,
        _uniform_policy_state_summary,
    )
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
    from scripts.benchmark_plasmon_article_cases import (  # type: ignore
        ARTICLE_DRIVEN_DENSITY_GRID,
        ARTICLE_DRIVEN_TEMPERATURE_EV,
        build_report,
        _uniform_policy_state_summary,
    )
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


DEFAULT_OUTPUT_DIR = Path("outputs/validation_outputs/plasmon_article_cases_response_shape_experiment")
RESPONSE_MODEL_RUNS = (
    ("noop", PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
    ("scalar_control", PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL),
    ("electron_column_weighted_control", PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED),
    ("new_response_shape_experimental", PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE),
)
_CLASSICAL_MODELS = {"rpa", "mermin", "rpa_static_lfc", "mermin_static_lfc"}
_LINDHARD_MODELS = {"lindhard", "lindhard_mermin", "lindhard_static_lfc", "lindhard_mermin_static_lfc"}


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
        rows.append(
            {
                "case": case_name,
                "model": model,
                "model_label": str(
                    control.get(
                        "model_label",
                        shape.get("model_label", weighted.get("model_label", noop.get("model_label", model))),
                    )
                ),
                "electron_policy": electron_policy,
                "published_branch": str(
                    control.get(
                        "published_branch",
                        shape.get("published_branch", weighted.get("published_branch", noop.get("published_branch", ""))),
                    )
                ),
                "experiment_mae_noop_ev": _float(noop, "experiment_mae_ev"),
                "experiment_mae_control_ev": _float(control, "experiment_mae_ev"),
                "experiment_mae_weighted_ev": _float(weighted, "experiment_mae_ev"),
                "experiment_mae_shape_ev": _float(shape, "experiment_mae_ev"),
                "matched_branch_mae_noop_ev": _float(noop, "matched_branch_mae_ev"),
                "matched_branch_mae_control_ev": _float(control, "matched_branch_mae_ev"),
                "matched_branch_mae_weighted_ev": _float(weighted, "matched_branch_mae_ev"),
                "matched_branch_mae_shape_ev": _float(shape, "matched_branch_mae_ev"),
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
                "judged_effect_shape": _judged_delta(
                    _float(control, "matched_branch_mae_ev") if math.isfinite(_float(control, "matched_branch_mae_ev")) else _float(control, "experiment_mae_ev"),
                    _float(shape, "matched_branch_mae_ev") if math.isfinite(_float(shape, "matched_branch_mae_ev")) else _float(shape, "experiment_mae_ev"),
                ),
            }
        )
    return rows


def _shape_state_diagnostics_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for response_model in (
        PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
        PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
        PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED,
        PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE,
    ):
        for density in ARTICLE_DRIVEN_DENSITY_GRID:
            summary = _uniform_policy_state_summary(
                rho_g_cm3=float(density),
                te_ev=float(ARTICLE_DRIVEN_TEMPERATURE_EV),
                policy=PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
                driven_response_model=response_model,
            )
            rows.append(
                {
                    "response_model": response_model,
                    "response_model_label": driven_response_model_label(response_model),
                    "rho_g_cm3": float(density),
                    "te_ev": float(ARTICLE_DRIVEN_TEMPERATURE_EV),
                    "effective_zeff": float(summary.get("effective_zbar_weighted", float("nan"))),
                    "effective_ne_cm3": float(summary.get("effective_ne_weighted_cm3", float("nan"))),
                    "weight_mode": str(summary.get("driven_response_weight_mode", "")),
                    "weight_mean": float(summary.get("driven_response_weight_mean", float("nan"))),
                    "shape_mode": str(summary.get("driven_response_shape_mode", "")),
                    "shape_mean_ev": float(summary.get("driven_response_shape_mean_ev", float("nan"))),
                    "shape_min_ev": float(summary.get("driven_response_shape_min_ev", float("nan"))),
                    "shape_max_ev": float(summary.get("driven_response_shape_max_ev", float("nan"))),
                    "summary": str(summary.get("driven_response_summary", "")),
                }
            )
    return rows


def _family_delta(delta_rows: list[dict[str, object]], *, case_name: str, models: set[str]) -> float:
    values = [
        float(row["matched_branch_delta_control_to_shape_ev"])
        for row in delta_rows
        if str(row["case"]) == case_name
        and str(row["model"]) in models
        and math.isfinite(float(row["matched_branch_delta_control_to_shape_ev"]))
    ]
    return float(sum(values) / len(values)) if values else float("nan")


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
            abs(float(row["experiment_mae_control_ev"]) - float(row["experiment_mae_shape_ev"]))
            for row in ambient_rows
            if math.isfinite(float(row["experiment_mae_control_ev"])) and math.isfinite(float(row["experiment_mae_shape_ev"]))
        ),
        default=0.0,
    )
    comparable_driven = [row for row in driven_rows if math.isfinite(float(row["matched_branch_delta_control_to_shape_ev"]))]
    comparable_driven.sort(key=lambda row: float(row["matched_branch_delta_control_to_shape_ev"]), reverse=True)
    best_shape = comparable_driven[0] if comparable_driven else None
    classical_mean_delta = _family_delta(delta_rows, case_name="driven_al_article_state", models=_CLASSICAL_MODELS)
    lindhard_mean_delta = _family_delta(delta_rows, case_name="driven_al_article_state", models=_LINDHARD_MODELS)
    equivalence_ok = all(bool(row.get("within_tolerance")) for row in equivalence_rows) if equivalence_rows else False
    lines = [
        "# Response-shape experimental model comparison",
        "",
        f"- dataset: **{dataset}**",
        "- models compared:",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_NONE)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE)}",
        "",
        "## 1. Control preservation",
        "",
        f"- Scalar control equivalence against the frozen driven-increment baseline: **{'passed' if equivalence_ok else 'check required'}**.",
        f"- Equivalence rows written: **{len(equivalence_rows)}**.",
        "",
        "## 2. Ambient behaviour",
        "",
        f"- Maximum ambient experiment-MAE shift between scalar control and the response-shape model: **{ambient_max_shift:.6f} eV**.",
        "",
        "## 3. Driven article-facing behaviour",
        "",
    ]
    if best_shape is not None:
        lines.append(
            f"- Best matched-branch shift from the response-shape model: **{best_shape['model_label']}** "
            f"({best_shape['published_branch']}) with delta MAE = "
            f"**{float(best_shape['matched_branch_delta_control_to_shape_ev']):+.3f} eV**."
        )
    else:
        lines.append("- No driven branch produced a finite matched-branch delta for the response-shape model.")
    lines.extend(
        [
            f"- Mean classical-family delta vs scalar control: **{classical_mean_delta:+.3f} eV**.",
            f"- Mean Lindhard-family delta vs scalar control: **{lindhard_mean_delta:+.3f} eV**.",
            "",
            "| case | branch | control matched MAE [eV] | weighted matched MAE [eV] | shape matched MAE [eV] | judgement |",
            "|---|---|---:|---:|---:|---|",
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
            + f" | {row['judged_effect_shape']} |"
        )
    lines.extend(["", "## 4. Judgement", ""])
    if best_shape is not None and float(best_shape["matched_branch_delta_control_to_shape_ev"]) >= 0.25:
        lines.append("- The first response-shape model gives a **defensible material gain**.")
        lines.append("- The gain comes from per-state shape broadening before ensemble averaging, not from another scalar density or weight tweak.")
    elif best_shape is not None and float(best_shape["matched_branch_delta_control_to_shape_ev"]) > -0.25:
        lines.append("- The first response-shape model gives only **marginal change**.")
        lines.append("- The residual driven mismatch still points to a larger physics upgrade beyond simple experimental response modifiers.")
    else:
        lines.append("- The first response-shape model **worsens agreement or does not materially help**.")
        lines.append("- The current framework likely needs a deeper response-model upgrade rather than another benchmark-side modifier.")
    lines.extend(
        [
            "",
            "## 5. Generated artifacts",
            "",
            f"- `{out_dir / 'report.md'}`",
            f"- `{out_dir / 'response_model_comparison.csv'}`",
            f"- `{out_dir / 'control_vs_experimental_delta.csv'}`",
            f"- `{out_dir / 'experimental_model_provenance.csv'}`",
            f"- `{out_dir / 'benchmark_summary.csv'}`",
            f"- `{out_dir / 'policy_state_summary.csv'}`",
            f"- `{out_dir / 'response_model_equivalence.csv'}`",
            f"- `{out_dir / 'shape_modifier_state_diagnostics.csv'}`",
        ]
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare response-shape driven-response modifiers against the scalar and weighting controls.")
    parser.add_argument("--dataset", default="50Al+10E+25CH+3.5TW_stabilized.h5")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_benchmark_rows: list[dict[str, object]] = []
    aggregate_policy_rows: list[dict[str, object]] = []
    aggregate_reconciliation_rows: list[dict[str, object]] = []

    run_dirs: dict[str, Path] = {}
    for slug, model in RESPONSE_MODEL_RUNS:
        run_dir = out_dir / slug
        rc = build_report(dataset_path, out_dir=run_dir, driven_response_model=model)
        if rc != 0:
            return int(rc)
        run_dirs[model] = run_dir
        aggregate_benchmark_rows.extend(_decorate_rows(_read_csv_rows(run_dir / "benchmark_summary.csv"), slug=slug, model=model))
        aggregate_policy_rows.extend(_decorate_rows(_read_csv_rows(run_dir / "policy_state_summary.csv"), slug=slug, model=model))
        aggregate_reconciliation_rows.extend(_decorate_rows(_read_csv_rows(run_dir / "reconciliation_summary.csv"), slug=slug, model=model))

    comparison_rows = _summary_comparison_rows(aggregate_benchmark_rows, aggregate_reconciliation_rows)
    delta_rows = _delta_rows(comparison_rows)
    equivalence_rows = _equivalence_rows(run_dirs[PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL])
    experimental_provenance_rows = [
        row
        for row in aggregate_policy_rows
        if str(row.get("response_model", "")) == PLASMON_DRIVEN_RESPONSE_MODEL_COLLISION_SHAPE
        and PRIMARY_CASE_POLICIES.get(str(row.get("case", ""))) == str(row.get("electron_policy", ""))
    ]
    shape_diagnostic_rows = _shape_state_diagnostics_rows()

    _write_csv(out_dir / "benchmark_summary.csv", aggregate_benchmark_rows)
    _write_csv(out_dir / "policy_state_summary.csv", aggregate_policy_rows)
    _write_csv(out_dir / "response_model_comparison.csv", comparison_rows)
    _write_csv(out_dir / "control_vs_experimental_delta.csv", delta_rows)
    _write_csv(out_dir / "experimental_model_provenance.csv", experimental_provenance_rows)
    _write_csv(out_dir / "response_model_equivalence.csv", equivalence_rows)
    _write_csv(out_dir / "shape_modifier_state_diagnostics.csv", shape_diagnostic_rows)
    (out_dir / "report.md").write_text(
        "\n".join(
            _report_lines(
                dataset=dataset_path.name,
                out_dir=out_dir,
                delta_rows=delta_rows,
                equivalence_rows=equivalence_rows,
            )
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
