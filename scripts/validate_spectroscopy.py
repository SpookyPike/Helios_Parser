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


OUTPUT_DIR = VALIDATION_ROOT / "spectroscopy"


def main() -> int:
    registry = build_registry()
    summary: dict[str, object] = {}
    for record in preferred_hdf5_records(registry):
        _dataset, _context, result = compute_result_for_record(record)
        plot_bundle_grid(OUTPUT_DIR / f"{Path(record.filename).stem}_time.png", f"{record.filename} | Spectroscopy time traces", result.spectroscopy.time_plots)
        plot_bundle_grid(OUTPUT_DIR / f"{Path(record.filename).stem}_profile.png", f"{record.filename} | Spectroscopy snapshot profiles", result.spectroscopy.profile_plots)
        summary[record.filename] = {
            "weighting_mode": result.spectroscopy.weighting_mode,
            "geometry_summary": result.spectroscopy.geometry_summary,
            "bulk_velocity_cm_s": result.spectroscopy.bulk_velocity_cm_s,
            "los_velocity_cm_s": result.spectroscopy.los_velocity_cm_s,
            "doppler_shift_nm": result.spectroscopy.doppler_shift_nm,
            "thermal_width_nm": result.spectroscopy.thermal_width_nm,
            "warnings": [{"severity": warning.severity, "message": warning.message} for warning in result.spectroscopy.warnings],
        }
    save_json(OUTPUT_DIR / "summary.json", summary)
    print(f"Validated spectroscopy quick-look on {len(summary)} datasets -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
