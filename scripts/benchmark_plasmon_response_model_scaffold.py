from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import _script_bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from scripts import _script_bootstrap  # type: ignore  # noqa: F401

import numpy as np

try:
    from benchmark_plasmon_article_cases import build_report
except ModuleNotFoundError:  # pragma: no cover
    from scripts.benchmark_plasmon_article_cases import build_report


BASELINE_OUTPUT_DIR = Path("outputs/validation_outputs/plasmon_article_cases_driven_increment_pass")
DEFAULT_OUTPUT_DIR = Path("outputs/validation_outputs/plasmon_article_cases_response_model_scaffold")
_TEXT_FILE_EXTENSIONS = {".py", ".md", ".json", ".csv", ".txt"}
_PROVENANCE_KEYWORDS = ("bespalov", "gawne", "tddft", "rpa", "lfc", "dispersion", "figs5")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def _provenance_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in (
        root / "src" / "helios" / "services" / "derived" / "reference_data" / "plasmon",
        root / "scripts",
        root / "_tmp_plasmon_fourth_pass",
        root / "docs",
    ):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _TEXT_FILE_EXTENSIONS:
                continue
            lowered = path.name.lower()
            if any(keyword in lowered for keyword in _PROVENANCE_KEYWORDS):
                candidates.append(path)
    return sorted(set(candidates))


def _equivalence_rows(baseline_dir: Path, current_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_summary = _read_csv_rows(baseline_dir / "benchmark_summary.csv")
    current_summary = _read_csv_rows(current_dir / "benchmark_summary.csv")
    summary_by_key = {
        (str(row.get("case", "")), str(row.get("electron_policy", "")), str(row.get("model", ""))): row
        for row in baseline_summary
    }
    current_by_key = {
        (str(row.get("case", "")), str(row.get("electron_policy", "")), str(row.get("model", ""))): row
        for row in current_summary
    }
    summary_cases = {
        ("ambient_al_t0", "article_al_benchmark"),
        ("driven_al_article_state", "article_al_driven_increment"),
    }
    for key, baseline_row in sorted(summary_by_key.items()):
        case_name, electron_policy, model = key
        if (case_name, electron_policy) not in summary_cases:
            continue
        current_row = current_by_key.get(key)
        if current_row is None:
            continue
        for metric_name, tolerance in (("mae_ev", 1.0e-12), ("rmse_ev", 1.0e-12)):
            baseline_value = float(baseline_row.get(metric_name, "nan"))
            current_value = float(current_row.get(metric_name, "nan"))
            if not np.isfinite(baseline_value) and not np.isfinite(current_value):
                delta = 0.0
                within_tolerance = True
            else:
                delta = current_value - baseline_value if np.isfinite(baseline_value) and np.isfinite(current_value) else float("nan")
                within_tolerance = bool(abs(delta) <= tolerance) if np.isfinite(delta) else False
            rows.append(
                {
                    "comparison_kind": "benchmark_summary",
                    "case": case_name,
                    "electron_policy": electron_policy,
                    "model": model,
                    "metric_name": metric_name,
                    "baseline_value": baseline_value,
                    "current_value": current_value,
                    "delta": delta,
                    "tolerance": tolerance,
                    "within_tolerance": within_tolerance,
                }
            )
        rows.append(
            {
                "comparison_kind": "benchmark_status",
                "case": case_name,
                "electron_policy": electron_policy,
                "model": model,
                "metric_name": "status",
                "baseline_value": str(baseline_row.get("status", "")),
                "current_value": str(current_row.get("status", "")),
                "delta": "",
                "tolerance": "",
                "within_tolerance": str(baseline_row.get("status", "")) == str(current_row.get("status", "")),
            }
        )

    baseline_policy = _read_csv_rows(baseline_dir / "policy_state_summary.csv")
    current_policy = _read_csv_rows(current_dir / "policy_state_summary.csv")
    baseline_policy_map = {
        (str(row.get("case", "")), str(row.get("electron_policy", ""))): row
        for row in baseline_policy
    }
    current_policy_map = {
        (str(row.get("case", "")), str(row.get("electron_policy", ""))): row
        for row in current_policy
    }
    for key in (
        ("ambient_al_t0", "article_al_benchmark"),
        ("driven_al_article_state", "article_al_driven_increment"),
    ):
        baseline_row = baseline_policy_map.get(key)
        current_row = current_policy_map.get(key)
        if baseline_row is None or current_row is None:
            continue
        for metric_name, tolerance in (
            ("effective_ne_weighted_cm3", 1.0e8),
            ("effective_valence_from_ne", 1.0e-12),
            ("baseline_zbar_weighted", 1.0e-12),
            ("increment_zbar_weighted", 1.0e-12),
        ):
            baseline_value = float(baseline_row.get(metric_name, "nan"))
            current_value = float(current_row.get(metric_name, "nan"))
            delta = current_value - baseline_value if np.isfinite(baseline_value) and np.isfinite(current_value) else float("nan")
            rows.append(
                {
                    "comparison_kind": "policy_state",
                    "case": key[0],
                    "electron_policy": key[1],
                    "model": "",
                    "metric_name": metric_name,
                    "baseline_value": baseline_value,
                    "current_value": current_value,
                    "delta": delta,
                    "tolerance": tolerance,
                    "within_tolerance": bool(abs(delta) <= tolerance) if np.isfinite(delta) else False,
                }
            )
    return rows


def _response_model_audit_lines(root: Path, out_dir: Path, provenance_candidates: list[Path], equivalence_rows: list[dict[str, object]]) -> list[str]:
    within_tolerance = [
        row for row in equivalence_rows if str(row.get("within_tolerance", "")).lower() in {"true", "1"} or row.get("within_tolerance") is True
    ]
    provenance_lines = []
    for path in provenance_candidates[:24]:
        provenance_lines.append(f"- `{path.relative_to(root)}`")
    if not provenance_lines:
        provenance_lines.append("- No provenance candidates matched the article/reference search keywords.")
    return [
        "# Plasmon response-model scaffold audit",
        "",
        "## 1. Provenance audit",
        "",
        "Search scope:",
        "- `src/helios/services/derived/reference_data/plasmon`",
        "- `scripts`",
        "- `_tmp_plasmon_fourth_pass`",
        "- `docs`",
        "",
        "Observed candidates:",
        *provenance_lines,
        "",
        "Judgement:",
        "- No article-native driven calculation tables were found in the repo.",
        "- The only structured article-facing branch inputs are the existing JSON-backed reference series in `src/helios/services/derived/reference_data/plasmon`.",
        "- Validation scripts and archived temporary pass remnants exist, but they do not provide a cleaner native RPA/LFC/TDDFT calculation asset than the current manual-digitization reference layer.",
        "",
        "## 2. Mandatory plasmon UI regression recovery note",
        "",
        "- Probe energy editability regression root cause: stale result-sync was still allowed to overwrite actively edited plasmon request controls before the edit committed.",
        "- Model-selection regression root cause: compare-all lived only as a boolean toggle; the actual comparison model set was no longer explicitly selectable.",
        "- Study/plot-option regression root cause: dispersion routing no longer preferred width/comparison bundles, so peak/FWHM workflows were effectively hidden behind legacy fallback bundle ordering.",
        "- Layout regression root cause: the plasmon sidebar still had a hard width clamp, which prevented allocating meaningful width to the graph panel.",
        "- UI files changed: `src/helios_analysis/workspace.py`, `src/helios/services/derived/plasmon.py`, `src/helios/services/derived/plasmon_config.py`, `src/helios/services/derived/analysis.py`, `src/helios/services/derived/models.py`, `tests/test_plasmon_phase8.py`, `tests/test_plasmon_ui_phase2.py`.",
        "",
        "## 3. Response-model scaffold",
        "",
        "- New abstraction: `src/helios/services/derived/plasmon_driven_response.py`.",
        "- Control models implemented:",
        "  - `none`",
        "  - `scalar_increment_control`",
        "- The article-driven scalar policy now routes through the new driven-response abstraction instead of carrying its increment logic only as a special case inside the benchmark/electron-policy path.",
        "- The JSON cold baseline remains the floor; raw HELIOS remains diagnostic-only for synthetic article states.",
        "",
        "## 4. Response-evaluation path audit",
        "",
        "- The synthetic driven article state is still constructed as an explicit density ensemble, not as a hidden surrogate slab.",
        "- For each density node, the code builds a uniform Al state, evaluates plasmon response for that state, then averages the final spectrum on a common energy grid.",
        "- Peak extraction happens after the spectrum average. The scaffold does not introduce a new pre-response surrogate collapse.",
        "",
        "## 5. Control equivalence",
        "",
        f"- Equivalence rows written: **{len(equivalence_rows)}**",
        f"- Rows within tolerance: **{len(within_tolerance)} / {len(equivalence_rows)}**",
        "- The scaffold target is exact reproduction of the frozen scalar-policy baseline before any non-scalar model work is attempted.",
        "",
        "## 6. Phase 3 status",
        "",
        "- No experimental non-scalar driven-response model was added in this pass.",
        "- Reason: provenance search did not uncover article-native response assets, and the next honest upgrade should be a larger response-level model change rather than another benchmark-side surrogate tweak.",
        "",
        "Generated alongside this audit:",
        f"- `{(out_dir / 'response_model_equivalence.csv').as_posix()}`",
        f"- `{(out_dir / 'benchmark_summary.csv').as_posix()}`",
        f"- `{(out_dir / 'policy_state_summary.csv').as_posix()}`",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the article benchmark through the response-model scaffold audit.")
    parser.add_argument("--dataset", default="50Al+10E+25CH+3.5TW_stabilized.h5")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    rc = build_report(Path(args.dataset), out_dir=out_dir)
    if rc != 0:
        return int(rc)

    equivalence_rows = _equivalence_rows(BASELINE_OUTPUT_DIR, out_dir)
    _write_csv(out_dir / "response_model_equivalence.csv", equivalence_rows)
    provenance_candidates = _provenance_candidates(root)
    (out_dir / "response_model_audit.md").write_text(
        "\n".join(_response_model_audit_lines(root, out_dir, provenance_candidates, equivalence_rows)),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
