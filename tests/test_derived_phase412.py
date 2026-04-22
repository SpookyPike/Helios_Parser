from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import get_app, process_events, reset_test_settings, combo_set_data
from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.common import load_run_data
from helios_analysis.workspace import HeliosDerivedWorkspace


ROOT = Path(__file__).resolve().parents[1]


def _context_from_dataset(path: Path, dataset, *, snapshot_index: int | None = None) -> RunContext:
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
        snapshot_index=min(20, max(0, dataset.time_s.size - 1)) if snapshot_index is None else int(snapshot_index),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "zone",
        slice_coordinate="zone",
        selected_region_ids=tuple(int(value) for value in np.asarray(dataset.regions["region_index"], dtype=np.int32)),
        selected_material_ids=tuple(int(value) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


class DerivedPhase412Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def setUp(self) -> None:
        reset_test_settings()

    def test_xrd_display_switch_updates_headers_and_titles(self) -> None:
        path = ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)
        result = compute_analysis_result(dataset, context, parameters=DerivedAnalysisParameters(), context_key=("phase412", "xrd"))

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.set_result(result)
            process_events(20)

            combo_set_data(workspace.xrd_display_combo, "q")
            process_events(20)
            headers_q = [workspace.xrd_table.horizontalHeaderItem(index).text() for index in range(workspace.xrd_table.columnCount())]
            self.assertEqual(headers_q[4:7], ["Q0", "Q", "Delta Q"])
            self.assertEqual(workspace.xrd_plot_panel.time_combo.currentData(), "q_compressed")

            combo_set_data(workspace.xrd_display_combo, "degrees")
            process_events(20)
            headers_deg = [workspace.xrd_table.horizontalHeaderItem(index).text() for index in range(workspace.xrd_table.columnCount())]
            self.assertEqual(headers_deg[4:7], ["theta0 [deg]", "theta [deg]", "Shift [deg]"])
            self.assertIn("snapshot", workspace.xrd_plot_panel.profile_plot.current_title.lower())
        finally:
            workspace.close()

    def test_multicurve_legends_survive_tab_switches(self) -> None:
        path = ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)
        result = compute_analysis_result(dataset, context, parameters=DerivedAnalysisParameters(), context_key=("phase412", "legend"))

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.set_result(result)
            process_events(20)

            combo_set_data(workspace.plasmon_plot_panel.time_combo, "temperatures")
            combo_set_data(workspace.spectroscopy_plot_panel.time_combo, "velocity")
            process_events(20)
            self.assertIsNotNone(workspace.plasmon_plot_panel.time_plot._legend)
            self.assertIsNotNone(workspace.spectroscopy_plot_panel.time_plot._legend)

            for index in range(workspace.result_tabs.count()):
                workspace.result_tabs.setCurrentIndex(index)
                process_events(10)
            for index in range(workspace.result_tabs.count() - 1, -1, -1):
                workspace.result_tabs.setCurrentIndex(index)
                process_events(10)

            self.assertIsNotNone(workspace.plasmon_plot_panel.time_plot._legend)
            self.assertIsNotNone(workspace.spectroscopy_plot_panel.time_plot._legend)
        finally:
            workspace.close()

    def test_real_dataset_filters_and_empty_selection_propagate(self) -> None:
        path = ROOT / "50Al+10E+25CH+3.5TW_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)

        base = compute_analysis_result(dataset, context, parameters=DerivedAnalysisParameters(), context_key=("phase412", "base"))
        filtered = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(exclude_low_density=True, min_density_g_cm3=0.05, exclude_opposite_velocity=True),
            context_key=("phase412", "filtered"),
        )
        empty = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(derived_region_ids=()),
            context_key=("phase412", "empty"),
        )

        self.assertNotEqual(base.selected_zone_count, filtered.selected_zone_count)
        self.assertNotAlmostEqual(base.plasmon.electron_temperature_ev, filtered.plasmon.electron_temperature_ev, places=8)
        self.assertEqual(empty.selected_zone_count, 0)
        self.assertTrue(any(w.source == "selection" and w.severity == "error" for w in empty.warnings))

    def test_default_profile_coordinate_follow_viewer_is_applied_to_dynamic_radius_context(self) -> None:
        path = ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        dataset = load_run_data(ROOT / "outputs" / "hdf5" / path.name)
        context = _context_from_dataset(ROOT / "outputs" / "hdf5" / path.name, dataset)

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            process_events(20)
            self.assertEqual(workspace.profile_coordinate_combo.currentData(), "viewer")
        finally:
            workspace.close()


if __name__ == "__main__":
    unittest.main()
