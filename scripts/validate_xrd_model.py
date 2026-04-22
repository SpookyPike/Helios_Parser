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


OUTPUT_DIR = VALIDATION_ROOT / "xrd"


def main() -> int:
    registry = build_registry()
    summary: dict[str, object] = {}
    for record in preferred_hdf5_records(registry):
        _dataset, _context, result = compute_result_for_record(record)
        plot_bundle_grid(OUTPUT_DIR / f"{Path(record.filename).stem}_time.png", f"{record.filename} | XRD time traces", result.xrd.time_plots)
        plot_bundle_grid(OUTPUT_DIR / f"{Path(record.filename).stem}_profile.png", f"{record.filename} | XRD snapshot profiles", result.xrd.profile_plots)
        summary[record.filename] = {
            "weighting_mode": result.xrd.weighting_mode,
            "layer_count": len(result.xrd.layers),
            "time_plots": len(result.xrd.time_plots),
            "profile_plots": len(result.xrd.profile_plots),
            "warnings": [{"severity": warning.severity, "message": warning.message} for warning in result.xrd.warnings],
        }
    save_json(OUTPUT_DIR / "summary.json", summary)
    print(f"Validated XRD quick-look on {len(summary)} datasets -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
