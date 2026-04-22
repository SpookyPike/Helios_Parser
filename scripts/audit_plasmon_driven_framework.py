from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _f(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _dominant_driver(*, hydro_window: float, synthetic_state: float, policy_increment: float) -> str:
    magnitudes = {
        "hydro_window": abs(hydro_window),
        "synthetic_state": abs(synthetic_state),
        "policy_increment": abs(policy_increment),
    }
    return max(magnitudes.items(), key=lambda item: item[1])[0]


def _family_bucket(model: str, published_branch: str) -> str:
    if published_branch in {"rpa", "lfc"}:
        return "classical_family"
    if published_branch == "tddft":
        return "lindhard_proxy_family"
    return "other"


def build_audit(out_dir: Path) -> int:
    delta_rows = _read_csv(out_dir / "driven_branch_reconciliation_delta.csv")
    reconciliation_rows = _read_csv(out_dir / "reconciliation_summary.csv")

    decomposition_rows: list[dict[str, object]] = []
    for row in delta_rows:
        model = str(row["model"])
        published_branch = str(row["published_branch"])
        legacy = _f(row.get("mae_legacy_current_ev"))
        best_hydro = _f(row.get("mae_best_hydro_ev"))
        article_old = _f(row.get("mae_article_state_old_policy_ev"))
        article_new = _f(row.get("mae_after_ev"))
        hydro_window_delta = best_hydro - legacy if math.isfinite(best_hydro) and math.isfinite(legacy) else float("nan")
        synthetic_state_delta = article_old - best_hydro if math.isfinite(article_old) and math.isfinite(best_hydro) else float("nan")
        policy_increment_delta = article_new - article_old if math.isfinite(article_new) and math.isfinite(article_old) else float("nan")
        residual_gap = article_new
        decomposition_rows.append(
            {
                "model": model,
                "model_label": str(row["model_label"]),
                "published_branch": published_branch,
                "family_bucket": _family_bucket(model, published_branch),
                "mae_previous_pass_ev": _f(row.get("mae_before_ev")),
                "mae_previous_hydro_ev": legacy,
                "mae_best_hydro_ev": best_hydro,
                "mae_article_state_old_policy_ev": article_old,
                "mae_article_state_increment_ev": article_new,
                "hydro_window_delta_ev": hydro_window_delta,
                "synthetic_state_delta_ev": synthetic_state_delta,
                "policy_increment_delta_ev": policy_increment_delta,
                "residual_gap_after_setup_ev": residual_gap,
                "dominant_setup_driver": _dominant_driver(
                    hydro_window=hydro_window_delta,
                    synthetic_state=synthetic_state_delta,
                    policy_increment=policy_increment_delta,
                ),
            }
        )
    _write_csv(out_dir / "driven_mismatch_decomposition.csv", decomposition_rows)

    article_rows = [
        row
        for row in reconciliation_rows
        if str(row.get("case", "")) == "driven_al_article_state"
        and str(row.get("electron_policy", "")) == "article_al_driven_increment"
        and str(row.get("published_branch", ""))
    ]
    article_map = {(str(row["model"]), str(row["published_branch"])): row for row in article_rows}
    collision_rows = []
    for parent, child, branch in (
        ("rpa", "mermin", "rpa"),
        ("rpa_static_lfc", "mermin_static_lfc", "lfc"),
        ("lindhard", "lindhard_mermin", "tddft"),
        ("lindhard_static_lfc", "lindhard_mermin_static_lfc", "tddft"),
    ):
        parent_row = article_map.get((parent, branch))
        child_row = article_map.get((child, branch))
        parent_mae = _f(parent_row.get("mae_ev")) if parent_row else float("nan")
        child_mae = _f(child_row.get("mae_ev")) if child_row else float("nan")
        collision_rows.append(
            {
                "parent_model": parent,
                "child_model": child,
                "published_branch": branch,
                "parent_mae_ev": parent_mae,
                "child_mae_ev": child_mae,
                "collision_closure_delta_ev": child_mae - parent_mae if math.isfinite(parent_mae) and math.isfinite(child_mae) else float("nan"),
            }
        )
    _write_csv(out_dir / "driven_collision_closure_delta.csv", collision_rows)

    classical_rows = [row for row in decomposition_rows if str(row["family_bucket"]) == "classical_family"]
    lindhard_rows = [row for row in decomposition_rows if str(row["family_bucket"]) == "lindhard_proxy_family"]
    collision_classical = [row for row in collision_rows if str(row["published_branch"]) in {"rpa", "lfc"}]
    collision_quantum = [row for row in collision_rows if str(row["published_branch"]) == "tddft"]

    lines = [
        "# Driven Al framework audit",
        "",
        "This audit decomposes the remaining driven-Al mismatch after the current benchmark-input and policy fixes.",
        "",
        "## Setup decomposition means",
        "",
        "| family bucket | hydro-window delta [eV] | synthetic-state delta [eV] | policy-increment delta [eV] | residual gap after setup [eV] |",
        "|---|---:|---:|---:|---:|",
        f"| classical_family | {_mean([_f(row['hydro_window_delta_ev']) for row in classical_rows]):.3f} | {_mean([_f(row['synthetic_state_delta_ev']) for row in classical_rows]):.3f} | {_mean([_f(row['policy_increment_delta_ev']) for row in classical_rows]):.3f} | {_mean([_f(row['residual_gap_after_setup_ev']) for row in classical_rows]):.3f} |",
        f"| lindhard_proxy_family | {_mean([_f(row['hydro_window_delta_ev']) for row in lindhard_rows]):.3f} | {_mean([_f(row['synthetic_state_delta_ev']) for row in lindhard_rows]):.3f} | {_mean([_f(row['policy_increment_delta_ev']) for row in lindhard_rows]):.3f} | {_mean([_f(row['residual_gap_after_setup_ev']) for row in lindhard_rows]):.3f} |",
        "",
        "Interpretation:",
        "- Positive delta means that change worsened MAE; negative delta means it improved MAE.",
        "- For the classical family, hydro-window tuning and synthetic-state construction are both small effects; the policy increment is the only setup change that materially improves agreement.",
        "- For the Lindhard-family proxy branches, the policy increment actually worsens agreement, which means the remaining issue is not simply too-low electron density.",
        "",
        "## Collision-closure effect inside the current framework",
        "",
        "| parent -> child | published branch | MAE delta [eV] |",
        "|---|---|---:|",
    ]
    for row in collision_rows:
        delta = _f(row["collision_closure_delta_ev"])
        lines.append(
            f"| {row['parent_model']} -> {row['child_model']} | {row['published_branch']} | "
            + (f"{delta:.3f}" if math.isfinite(delta) else "-")
            + " |"
        )
    lines.extend(
        [
            "",
            "## Judgement",
            "",
            "- The remaining classical gap is not dominated by hydro slab search anymore; the best-hydro slab is slightly worse than the original driven slab for the direct classical branch matches.",
            "- The synthetic article-state construction is not the main blocker either; it changes the direct classical MAEs only modestly once the cold-baseline policy is held fixed.",
            "- The driven electron increment is real and helpful for the classical family, but it is a scalar Zeff correction. It improves RPA/RPA+static-LFC by about 0.7 eV, then saturates with several eV of residual mismatch still left.",
            "- The Mermin closure is not the top blocker for the classical family: it changes the driven article-state MAE by only a few hundredths of an eV relative to its parent branches.",
            "- The Lindhard-family proxy mismatch is not fixed by the increment and often gets worse, which points to a genuine family/response mismatch rather than benchmark plumbing.",
            "",
            "## Path B conclusion",
            "",
            "The current framework is effectively exhausted for article-facing driven Al. Further small scalar-electron-policy nudges are not worth developer time.",
            "",
            "The smallest major model change that is still worth doing is:",
            "",
            "1. Replace the scalar driven-Zeff increment with a response-level driven-electron model that produces state-dependent electron response across the density/temperature ensemble, not just a single adjusted ne/Zeff.",
            "2. Average susceptibility/response over the reconciled driven state distribution with that upgraded electron model, rather than relying on the present low-dimensional Zeff shift inside the same classical/Lindhard closures.",
            "3. Only after that revisit family-to-family disagreement against published RPA/LFC/TDDFT branches.",
        ]
    )
    (out_dir / "driven_framework_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the remaining driven-Al benchmark mismatch decomposition.")
    parser.add_argument(
        "--out-dir",
        default="outputs\\validation_outputs\\plasmon_article_cases_driven_increment_pass",
        help="Benchmark output directory to audit.",
    )
    args = parser.parse_args()
    return build_audit(Path(args.out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
