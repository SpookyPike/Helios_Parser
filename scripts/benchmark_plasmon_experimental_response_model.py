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
    PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED,
    PLASMON_DRIVEN_RESPONSE_MODEL_NONE,
    PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL,
    driven_response_model_label,
)

try:
    from benchmark_plasmon_article_cases import build_report
except ModuleNotFoundError:  # pragma: no cover
    from scripts.benchmark_plasmon_article_cases import build_report


DEFAULT_OUTPUT_DIR = Path("outputs/validation_outputs/plasmon_article_cases_experimental_response_model")
FROZEN_BASELINE_DIR = Path("outputs/validation_outputs/plasmon_article_cases_driven_increment_pass")
PRIMARY_CASE_POLICIES = {
    "ambient_al_t0": "article_al_benchmark",
    "driven_al_article_state": "article_al_driven_increment",
}
RESPONSE_MODEL_RUNS = (
    ("noop", PLASMON_DRIVEN_RESPONSE_MODEL_NONE),
    ("scalar_control", PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL),
    ("experimental", PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED),
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


def _decorate_rows(rows: list[dict[str, str]], *, slug: str, model: str) -> list[dict[str, object]]:
    decorated: list[dict[str, object]] = []
    for row in rows:
        decorated.append(
            {
                "response_model_slug": slug,
                "response_model": model,
                "response_model_label": driven_response_model_label(model),
                **row,
            }
        )
    return decorated


def _float(row: dict[str, object], key: str) -> float:
    value = row.get(key, float("nan"))
    try:
        return float(value)
    except Exception:
        return float("nan")


def _summary_comparison_rows(
    benchmark_rows: list[dict[str, object]],
    reconciliation_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    recon_map = {
        (
            str(row.get("response_model", "")),
            str(row.get("case", "")),
            str(row.get("model", "")),
            str(row.get("electron_policy", "")),
        ): row
        for row in reconciliation_rows
    }
    rows: list[dict[str, object]] = []
    for row in benchmark_rows:
        case_name = str(row.get("case", ""))
        electron_policy = str(row.get("electron_policy", ""))
        if PRIMARY_CASE_POLICIES.get(case_name) != electron_policy:
            continue
        recon = recon_map.get(
            (
                str(row.get("response_model", "")),
                case_name,
                str(row.get("model", "")),
                electron_policy,
            ),
            {},
        )
        rows.append(
            {
                "response_model_slug": str(row.get("response_model_slug", "")),
                "response_model": str(row.get("response_model", "")),
                "response_model_label": str(row.get("response_model_label", "")),
                "case": case_name,
                "model": str(row.get("model", "")),
                "model_label": str(row.get("model_label", "")),
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
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["case"]),
            str(row["model"]),
            str(row["response_model"]),
        )
    )
    return rows


def _judged_delta(control_mae: float, experimental_mae: float) -> str:
    if not math.isfinite(control_mae) or not math.isfinite(experimental_mae):
        return "not_comparable"
    delta = float(control_mae - experimental_mae)
    if delta >= 1.0:
        return "material gain"
    if delta >= 0.25:
        return "improved"
    if delta > -0.25:
        return "marginal change"
    return "worse"


def _delta_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in comparison_rows:
        key = (str(row["case"]), str(row["model"]), str(row["electron_policy"]))
        grouped.setdefault(key, {})[str(row["response_model"])] = row
    rows: list[dict[str, object]] = []
    for (case_name, model, electron_policy), variants in sorted(grouped.items()):
        noop = variants.get(PLASMON_DRIVEN_RESPONSE_MODEL_NONE, {})
        control = variants.get(PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL, {})
        experimental = variants.get(PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED, {})
        rows.append(
            {
                "case": case_name,
                "model": model,
                "model_label": str(control.get("model_label", experimental.get("model_label", noop.get("model_label", model)))),
                "electron_policy": electron_policy,
                "published_branch": str(control.get("published_branch", experimental.get("published_branch", noop.get("published_branch", "")))),
                "experiment_mae_noop_ev": _float(noop, "experiment_mae_ev"),
                "experiment_mae_control_ev": _float(control, "experiment_mae_ev"),
                "experiment_mae_experimental_ev": _float(experimental, "experiment_mae_ev"),
                "matched_branch_mae_noop_ev": _float(noop, "matched_branch_mae_ev"),
                "matched_branch_mae_control_ev": _float(control, "matched_branch_mae_ev"),
                "matched_branch_mae_experimental_ev": _float(experimental, "matched_branch_mae_ev"),
                "experiment_delta_control_to_experimental_ev": (
                    _float(control, "experiment_mae_ev") - _float(experimental, "experiment_mae_ev")
                    if math.isfinite(_float(control, "experiment_mae_ev")) and math.isfinite(_float(experimental, "experiment_mae_ev"))
                    else float("nan")
                ),
                "matched_branch_delta_control_to_experimental_ev": (
                    _float(control, "matched_branch_mae_ev") - _float(experimental, "matched_branch_mae_ev")
                    if math.isfinite(_float(control, "matched_branch_mae_ev")) and math.isfinite(_float(experimental, "matched_branch_mae_ev"))
                    else float("nan")
                ),
                "judged_effect": _judged_delta(
                    _float(control, "matched_branch_mae_ev") if math.isfinite(_float(control, "matched_branch_mae_ev")) else _float(control, "experiment_mae_ev"),
                    _float(experimental, "matched_branch_mae_ev") if math.isfinite(_float(experimental, "matched_branch_mae_ev")) else _float(experimental, "experiment_mae_ev"),
                ),
            }
        )
    return rows


def _equivalence_rows(current_scalar_dir: Path) -> list[dict[str, object]]:
    baseline_rows = _read_csv_rows(FROZEN_BASELINE_DIR / "benchmark_summary.csv")
    current_rows = _read_csv_rows(current_scalar_dir / "benchmark_summary.csv")
    baseline_map = {
        (str(row.get("case", "")), str(row.get("electron_policy", "")), str(row.get("model", ""))): row
        for row in baseline_rows
    }
    current_map = {
        (str(row.get("case", "")), str(row.get("electron_policy", "")), str(row.get("model", ""))): row
        for row in current_rows
    }
    rows: list[dict[str, object]] = []
    keys = {
        ("ambient_al_t0", "article_al_benchmark"),
        ("driven_al_article_state", "article_al_driven_increment"),
    }
    for key, baseline in sorted(baseline_map.items()):
        case_name, electron_policy, model = key
        if (case_name, electron_policy) not in keys:
            continue
        current = current_map.get(key)
        if current is None:
            continue
        for metric, tol in (("mae_ev", 1.0e-12), ("rmse_ev", 1.0e-12)):
            baseline_value = _float(baseline, metric)
            current_value = _float(current, metric)
            if math.isfinite(baseline_value) and math.isfinite(current_value):
                delta = current_value - baseline_value
                within_tolerance = bool(abs(delta) <= tol)
            elif not math.isfinite(baseline_value) and not math.isfinite(current_value):
                delta = 0.0
                within_tolerance = True
            else:
                delta = float("nan")
                within_tolerance = False
            rows.append(
                {
                    "case": case_name,
                    "electron_policy": electron_policy,
                    "model": model,
                    "metric_name": metric,
                    "baseline_value": baseline_value,
                    "current_value": current_value,
                    "delta": delta,
                    "tolerance": tol,
                    "within_tolerance": within_tolerance,
                }
            )
    return rows


def _report_lines(
    *,
    dataset: str,
    out_dir: Path,
    comparison_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    equivalence_rows: list[dict[str, object]],
) -> list[str]:
    ambient_rows = [row for row in delta_rows if str(row["case"]) == "ambient_al_t0"]
    driven_rows = [row for row in delta_rows if str(row["case"]) == "driven_al_article_state"]
    ambient_max_shift = max(
        (abs(float(row["experiment_delta_control_to_experimental_ev"])) for row in ambient_rows if math.isfinite(float(row["experiment_delta_control_to_experimental_ev"]))),
        default=0.0,
    )
    driven_best = None
    comparable_driven = [row for row in driven_rows if math.isfinite(float(row["matched_branch_delta_control_to_experimental_ev"]))]
    if comparable_driven:
        comparable_driven.sort(key=lambda row: float(row["matched_branch_delta_control_to_experimental_ev"]), reverse=True)
        driven_best = comparable_driven[0]
    equivalence_ok = all(bool(row.get("within_tolerance")) for row in equivalence_rows) if equivalence_rows else False
    lines = [
        "# Experimental driven-response model comparison",
        "",
        f"- dataset: **{dataset}**",
        "- models compared:",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_NONE)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_SCALAR_CONTROL)}",
        f"  - {driven_response_model_label(PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED)}",
        "",
        "## 1. Control preservation",
        "",
        f"- Scalar control equivalence against the frozen driven-increment baseline: **{'passed' if equivalence_ok else 'check required'}**.",
        f"- Equivalence rows written: **{len(equivalence_rows)}**.",
        "",
        "## 2. Ambient behaviour",
        "",
        f"- Maximum ambient experiment-MAE shift between scalar control and the experimental model: **{ambient_max_shift:.6f} eV**.",
        "- This pass should not materially change the ambient benchmark because the driven-response layer is only active for the article-driven increment policy.",
        "",
        "## 3. Driven article-facing behaviour",
        "",
    ]
    if driven_best is not None:
        lines.append(
            f"- Best matched-branch improvement from the experimental model: **{driven_best['model_label']}** "
            f"({driven_best['published_branch']}) with delta MAE = "
            f"**{float(driven_best['matched_branch_delta_control_to_experimental_ev']):+.3f} eV**."
        )
    else:
        lines.append("- No driven branch produced a finite matched-branch delta between scalar control and the experimental model.")
    lines.extend(
        [
            "",
            "| case | branch | policy | control matched MAE [eV] | experimental matched MAE [eV] | judgement |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in driven_rows:
        lines.append(
            f"| {row['case']} | {row['model_label']} -> {row['published_branch'] or '-'} | {row['electron_policy']} | "
            + (f"{float(row['matched_branch_mae_control_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_control_ev"])) else "-")
            + " | "
            + (f"{float(row['matched_branch_mae_experimental_ev']):.3f}" if math.isfinite(float(row["matched_branch_mae_experimental_ev"])) else "-")
            + f" | {row['judged_effect']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Judgement",
            "",
        ]
    )
    if driven_best is not None and float(driven_best["matched_branch_delta_control_to_experimental_ev"]) >= 0.25:
        lines.append("- The first experimental non-scalar model gives a **defensible but limited gain** on the driven article-facing branch comparison.")
        lines.append("- The gain comes from response-level ensemble weighting, not from another scalar Zeff increment.")
        lines.append("- The remaining mismatch is still large enough that a deeper response-model change is likely required after this first experiment.")
    elif driven_best is not None and float(driven_best["matched_branch_delta_control_to_experimental_ev"]) > -0.25:
        lines.append("- The first experimental non-scalar model gives only **marginal change**.")
        lines.append("- This suggests the current framework still needs a larger response-level upgrade beyond simple ensemble weighting.")
    else:
        lines.append("- The first experimental non-scalar model does **not materially help** and should not replace the scalar control path.")
        lines.append("- The current framework likely needs a deeper response-model change rather than another benchmark-side surrogate.")
    lines.extend(
        [
            "",
            "## 5. Generated artifacts",
            "",
            f"- `{out_dir / 'report.md'}`",
            f"- `{out_dir / 'response_model_comparison.csv'}`",
            f"- `{out_dir / 'benchmark_summary.csv'}`",
            f"- `{out_dir / 'policy_state_summary.csv'}`",
            f"- `{out_dir / 'control_vs_experimental_delta.csv'}`",
            f"- `{out_dir / 'experimental_model_provenance.csv'}`",
            f"- `{out_dir / 'response_model_equivalence.csv'}`",
        ]
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the first experimental driven-response model against the frozen scalar control.")
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
        if str(row.get("response_model", "")) == PLASMON_DRIVEN_RESPONSE_MODEL_ELECTRON_COLUMN_WEIGHTED
        and PRIMARY_CASE_POLICIES.get(str(row.get("case", ""))) == str(row.get("electron_policy", ""))
    ]

    _write_csv(out_dir / "benchmark_summary.csv", aggregate_benchmark_rows)
    _write_csv(out_dir / "policy_state_summary.csv", aggregate_policy_rows)
    _write_csv(out_dir / "response_model_comparison.csv", comparison_rows)
    _write_csv(out_dir / "control_vs_experimental_delta.csv", delta_rows)
    _write_csv(out_dir / "experimental_model_provenance.csv", experimental_provenance_rows)
    _write_csv(out_dir / "response_model_equivalence.csv", equivalence_rows)
    (out_dir / "report.md").write_text(
        "\n".join(
            _report_lines(
                dataset=dataset_path.name,
                out_dir=out_dir,
                comparison_rows=comparison_rows,
                delta_rows=delta_rows,
                equivalence_rows=equivalence_rows,
            )
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
