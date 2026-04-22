from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import HDF5_ROOT, ROOT, get_app, process_events, reset_test_settings
from helios.runtime import RunContext
from helios_parser import write_hdf5
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.common import load_run_data
from helios_analysis.workspace import HeliosDerivedWorkspace


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
        snapshot_index=min(20, max(0, dataset.time_s.size - 1)),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="moving_radius" if dataset.radius_cm is not None else "zone",
        selected_region_ids=tuple(int(value) for value in np.asarray(dataset.regions["region_index"], dtype=np.int32)),
        selected_material_ids=tuple(int(abs(value)) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


class DerivedPhase41Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def setUp(self) -> None:
        reset_test_settings()

    def test_geometry_filtering_and_plot_generation_propagate(self) -> None:
        path = HDF5_ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)
        parameters = DerivedAnalysisParameters(
            observation_side="back",
            line_of_sight_angle_deg=60.0,
            profile_coordinate_mode="moving_radius",
            derived_region_ids=(1, 3),
            exclude_low_density=True,
            min_density_g_cm3=0.05,
            exclude_opposite_velocity=True,
            weighting_mode="electron_column",
        )
        result = compute_analysis_result(dataset, context, parameters=parameters, context_key=("phase41",))
        self.assertEqual(result.geometry.observation_side, "back")
        self.assertAlmostEqual(result.geometry.line_of_sight_cosine, 0.5, places=6)
        self.assertEqual(result.geometry.profile_coordinate_mode, "moving_radius")
        self.assertEqual(result.selection.derived_region_ids, (1, 3))
        self.assertTrue(result.selection.exclude_low_density)
        self.assertTrue(result.selection.exclude_opposite_velocity)
        self.assertGreater(result.selected_zone_count, 0)
        self.assertGreaterEqual(len(result.xrd.time_plots), 5)
        self.assertGreaterEqual(len(result.plasmon.time_plots), 5)
        self.assertGreaterEqual(len(result.transmission.time_plots), 4)
        self.assertGreaterEqual(len(result.spectroscopy.time_plots), 3)
        self.assertIn("back side", result.transmission.geometry_summary)
        self.assertEqual(result.plasmon.weighting_mode, "electron_column")

    def test_empty_selection_produces_error_warning_and_grouped_ui(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)
        parameters = DerivedAnalysisParameters(derived_region_ids=(999,))
        result = compute_analysis_result(dataset, context, parameters=parameters, context_key=("empty",))
        self.assertEqual(result.selected_zone_count, 0)
        severities = {(warning.source, warning.severity) for warning in result.warnings}
        self.assertIn(("selection", "error"), severities)

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.set_result(result)
            process_events(20)
            top_level_labels = [workspace.warnings_tree.topLevelItem(index).text(0).lower() for index in range(workspace.warnings_tree.topLevelItemCount())]
            self.assertIn("selection", top_level_labels)
            selection_index = top_level_labels.index("selection")
            selection_item = workspace.warnings_tree.topLevelItem(selection_index)
            self.assertGreater(selection_item.childCount(), 0)
            self.assertEqual(selection_item.child(0).text(0), "ERROR")
        finally:
            workspace.close()

    def test_cylindrical_coordinate_labels_and_path_approximation_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = ROOT / "Cu1e17_cyl.log"
            target = Path(tmpdir) / "Cu1e17_cyl.h5"
            write_hdf5(source, target, overwrite=True)
            dataset = load_run_data(target)
            context = _context_from_dataset(target, dataset)
            parameters = DerivedAnalysisParameters(profile_coordinate_mode="static_x")
            result = compute_analysis_result(dataset, context, parameters=parameters, context_key=("cyl",))
            self.assertTrue(result.xrd.profile_coordinate_label.startswith("Radius ["))
            self.assertEqual(result.geometry.path_length_mode, "cylindrical-shell-centerline")
            self.assertIn("cylindrical shell LOS applied", result.plasmon.geometry_summary)
            self.assertIn("cylindrical shell LOS applied", result.transmission.geometry_summary)
            self.assertIn("cylindrical shell LOS applied", result.spectroscopy.geometry_summary)
            approximation_sources = {
                warning.source
                for warning in result.warnings
                if "cylindrical los geometry uses shell intersections" in warning.message.lower()
            }
            self.assertTrue({"plasmon", "transmission", "spectroscopy"}.issubset(approximation_sources))

    def test_picosecond_drive_assumption_warnings_are_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = ROOT / "Cu1e17.log"
            target = Path(tmpdir) / "Cu1e17.h5"
            write_hdf5(source, target, overwrite=True)
            dataset = load_run_data(target)
            context = _context_from_dataset(target, dataset)
            result = compute_analysis_result(dataset, context, parameters=DerivedAnalysisParameters(), context_key=("ps",))
            ps_sources = {
                warning.source
                for warning in result.warnings
                if "ps-scale" in warning.message.lower()
            }
            self.assertTrue({"xrd", "plasmon", "transmission", "spectroscopy"}.issubset(ps_sources))


if __name__ == "__main__":
    unittest.main()
