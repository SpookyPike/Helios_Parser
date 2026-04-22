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


OUTPUT_DIR = VALIDATION_ROOT / "transmission"


def main() -> int:
    registry = build_registry()
    summary: dict[str, object] = {}
    for record in preferred_hdf5_records(registry):
        _dataset, _context, result = compute_result_for_record(record)
        plot_bundle_grid(OUTPUT_DIR / f"{Path(record.filename).stem}_time.png", f"{record.filename} | Transmission time traces", result.transmission.time_plots)
        plot_bundle_grid(OUTPUT_DIR / f"{Path(record.filename).stem}_profile.png", f"{record.filename} | Transmission cumulative profiles", result.transmission.profile_plots)
        summary[record.filename] = {
            "geometry_summary": result.transmission.geometry_summary,
            "areal_density_g_cm2": result.transmission.areal_density_g_cm2,
            "electron_column_cm2": result.transmission.electron_column_cm2,
            "tau": result.transmission.thomson_tau,
            "transmission": result.transmission.thomson_transmission,
            "region_budgets": len(result.transmission.region_budgets),
            "warnings": [{"severity": warning.severity, "message": warning.message} for warning in result.transmission.warnings],
        }
    save_json(OUTPUT_DIR / "summary.json", summary)
    print(f"Validated transmission quick-look on {len(summary)} datasets -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
