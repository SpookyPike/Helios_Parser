from __future__ import annotations

from _validation_common import VALIDATION_ROOT, build_registry, compute_result_for_record, preferred_hdf5_records, save_json
from helios.services.derived import DerivedAnalysisParameters


OUTPUT_DIR = VALIDATION_ROOT / "weighting_modes"
WEIGHTING_MODES = ("simple_mean", "width", "mass", "electron_density", "electron_column")


def main() -> int:
    registry = build_registry()
    summary: dict[str, object] = {}
    for record in preferred_hdf5_records(registry):
        dataset_summary: dict[str, object] = {}
        for weighting_mode in WEIGHTING_MODES:
            _dataset, _context, result = compute_result_for_record(record, parameters=DerivedAnalysisParameters(weighting_mode=weighting_mode))
            dataset_summary[weighting_mode] = {
                "selected_zones": result.selected_zone_count,
                "plasmon_ne_cm3": result.plasmon.electron_density_cm3,
                "transmission_tau": result.transmission.thomson_tau,
                "spectroscopy_bulk_velocity_cm_s": result.spectroscopy.bulk_velocity_cm_s,
                "warnings": [{"severity": warning.severity, "message": warning.message} for warning in result.warnings],
            }
        summary[record.filename] = dataset_summary
    save_json(OUTPUT_DIR / "summary.json", summary)
    print(f"Validated weighting modes on {len(summary)} datasets -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
