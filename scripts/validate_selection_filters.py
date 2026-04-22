from __future__ import annotations

from pathlib import Path

from _validation_common import VALIDATION_ROOT, build_registry, compute_result_for_record, preferred_hdf5_records, save_json
from helios.services.derived import DerivedAnalysisParameters


OUTPUT_DIR = VALIDATION_ROOT / "selection_filters"


def main() -> int:
    registry = build_registry()
    summary: dict[str, object] = {}
    comparison_cases = {
        "baseline": DerivedAnalysisParameters(),
        "exclude_blowoff": DerivedAnalysisParameters(exclude_low_density=True, min_density_g_cm3=0.01),
        "exclude_opposite_flow": DerivedAnalysisParameters(exclude_opposite_velocity=True),
        "back_oblique": DerivedAnalysisParameters(observation_side="back", line_of_sight_angle_deg=35.0),
    }
    for record in preferred_hdf5_records(registry):
        dataset_summary: dict[str, object] = {}
        for label, parameters in comparison_cases.items():
            _dataset, _context, result = compute_result_for_record(record, parameters=parameters)
            dataset_summary[label] = {
                "selected_zones": result.selected_zone_count,
                "geometry": {
                    "side": result.geometry.observation_side,
                    "los_cos": result.geometry.line_of_sight_cosine,
                    "profile_coordinate": result.geometry.profile_coordinate_mode,
                },
                "selection": {
                    "exclude_low_density": result.selection.exclude_low_density,
                    "exclude_opposite_velocity": result.selection.exclude_opposite_velocity,
                    "min_density_g_cm3": result.selection.min_density_g_cm3,
                    "notes": list(result.selection.notes),
                },
            }
        summary[record.filename] = dataset_summary
    save_json(OUTPUT_DIR / "summary.json", summary)
    print(f"Validated geometry/filter propagation on {len(summary)} datasets -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
