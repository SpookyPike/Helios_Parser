from __future__ import annotations

from pathlib import Path

from _validation_common import (
    VALIDATION_ROOT,
    build_registry,
    compute_result_for_record,
    plot_bundle_grid,
    preferred_hdf5_records,
    save_json,
)


OUTPUT_DIR = VALIDATION_ROOT / "plasmon"


def main() -> int:
    registry = build_registry()
    summary: dict[str, object] = {}
    for record in preferred_hdf5_records(registry):
        _dataset, _context, result = compute_result_for_record(record)
        plot_bundle_grid(OUTPUT_DIR / f"{Path(record.filename).stem}_time.png", f"{record.filename} | Plasmon time traces", result.plasmon.time_plots)
        plot_bundle_grid(OUTPUT_DIR / f"{Path(record.filename).stem}_profile.png", f"{record.filename} | Plasmon snapshot profiles", result.plasmon.profile_plots)
        summary[record.filename] = {
            "weighting_mode": result.plasmon.weighting_mode,
            "geometry_summary": result.plasmon.geometry_summary,
            "regime": result.plasmon.regime_label,
            "k_lambda_d": result.plasmon.k_lambda_debye,
            "collectivity_parameter": result.plasmon.collectivity_parameter,
            "time_plots": len(result.plasmon.time_plots),
            "profile_plots": len(result.plasmon.profile_plots),
            "warnings": [{"severity": warning.severity, "message": warning.message} for warning in result.plasmon.warnings],
        }
    save_json(OUTPUT_DIR / "summary.json", summary)
    print(f"Validated plasmon regime quick-look on {len(summary)} datasets -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
