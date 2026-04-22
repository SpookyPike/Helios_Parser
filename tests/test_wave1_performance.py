from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import HDF5_ROOT, get_app, process_events, reset_test_settings
from helios.instrumentation import format_metrics_summary, reset_metrics, snapshot_metrics
from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry, build_analysis_mask
from helios_viewer.plots import FieldMapWidget


def _context_from_dataset(path: Path, dataset) -> RunContext:
    return RunContext(
        path=path,
        summary=dict(dataset.summary),
        metadata=dict(dataset.metadata),
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(),
        static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(),
        zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
        has_dynamic_radius=dataset.radius_cm is not None,
        snapshot_index=min(3, max(0, dataset.time_s.size - 1)),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="zone",
        selected_region_ids=tuple(int(value) for value in np.asarray(dataset.regions["region_index"], dtype=np.int32)),
        selected_material_ids=tuple(int(value) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


class Wave1PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def setUp(self) -> None:
        reset_test_settings()
        reset_metrics()

    def test_field_map_auto_levels_handle_nan_and_inf_without_crashing(self) -> None:
        widget = FieldMapWidget()
        try:
            data = np.linspace(-4.0, 9.0, num=512 * 512, dtype=np.float64).reshape(512, 512)
            data[0, 0] = np.nan
            data[10, 10] = np.inf
            data[20, 20] = -np.inf
            coordinate = np.linspace(0.0, 10.0, num=data.shape[1], dtype=np.float64)
            time_values = np.linspace(0.0, 5.0, num=data.shape[0], dtype=np.float64)

            widget.set_field_map(
                data,
                coordinate,
                time_values,
                orientation="coord_x_time_y",
                title="Synthetic density",
                x_label="Coordinate",
                y_label="Time",
                colorbar_label="Density",
                auto_levels=True,
            )
            process_events(20)

            expected_finite = data[np.isfinite(data)]
            self.assertGreater(expected_finite.size, 0)
            expected = (float(np.min(expected_finite)), float(np.max(expected_finite)))
            levels = widget._colorbar.levels()
            self.assertAlmostEqual(levels[0], expected[0], places=12)
            self.assertAlmostEqual(levels[1], expected[1], places=12)

            metrics = snapshot_metrics()
            self.assertIn("viewer.render.field_map", metrics["timers"])
            self.assertEqual(metrics["counters"].get("viewer.render.field_map.calls"), 1)
        finally:
            widget.close()

    def test_cached_analysis_masks_are_read_only_and_require_explicit_copy_for_mutation(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)
        parameters = DerivedAnalysisParameters(
            weighting_mode="electron_column",
            exclude_low_density=True,
            min_density_g_cm3=0.01,
            zone_index_upper=250,
        )
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=parameters.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        analysis_cache = AnalysisStateCache()

        mask_a, _, _ = build_analysis_mask(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            geometry=geometry,
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            weighting_mode=parameters.weighting_mode,
            analysis_cache=analysis_cache,
        )
        mask_b, _, _ = build_analysis_mask(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            geometry=geometry,
            reuse_viewer_subset=parameters.reuse_viewer_subset,
            derived_region_ids=parameters.derived_region_ids,
            derived_material_ids=parameters.derived_material_ids,
            exclude_entry_region=parameters.exclude_entry_region,
            exclude_low_density=parameters.exclude_low_density,
            min_density_g_cm3=parameters.min_density_g_cm3,
            exclude_opposite_velocity=parameters.exclude_opposite_velocity,
            zone_index_lower=parameters.zone_index_lower,
            zone_index_upper=parameters.zone_index_upper,
            weighting_mode=parameters.weighting_mode,
            analysis_cache=analysis_cache,
        )

        self.assertFalse(mask_a.flags.writeable)
        self.assertFalse(mask_b.flags.writeable)
        self.assertTrue(np.shares_memory(mask_a, mask_b))
        with self.assertRaises(ValueError):
            mask_b[0] = not bool(mask_b[0])

        mutable_copy = mask_b.copy()
        mutable_copy[0] = not bool(mutable_copy[0])
        self.assertNotEqual(bool(mutable_copy[0]), bool(mask_b[0]))

        self.assertGreaterEqual(analysis_cache.mask_hits, 1)
        metrics = snapshot_metrics()
        self.assertGreaterEqual(metrics["counters"].get("derived.cache.mask.hit", 0), 1)
        self.assertGreaterEqual(metrics["counters"].get("derived.cache.mask.miss", 0), 1)

    def test_instrumentation_summary_reports_real_metrics(self) -> None:
        widget = FieldMapWidget()
        try:
            data = np.array([[1.0, np.nan], [np.inf, 4.0]], dtype=np.float64)
            widget.set_field_map(
                data,
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([0.0, 1.0], dtype=np.float64),
                orientation="coord_x_time_y",
                title="Instrumentation",
                x_label="x",
                y_label="t",
                colorbar_label="rho",
                auto_levels=True,
            )
            process_events(10)
            summary = format_metrics_summary()
            self.assertIn("viewer.render.field_map", summary)
            self.assertIn("viewer.render.field_map.calls", summary)
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
