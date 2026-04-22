from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import HDF5_ROOT, get_app, process_events, reset_test_settings, wait_until
from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.common import load_run_data
from helios.services.units.conversions import photon_energy_ev_from_wavelength_nm
from helios_analysis.controller import DerivedController
from helios_analysis.workspace import HeliosDerivedWorkspace
from helios_viewer.settings import default_viewer_settings


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


class DerivedStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def setUp(self) -> None:
        reset_test_settings()

    def test_filter_and_weighting_changes_propagate_into_results(self) -> None:
        path = HDF5_ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)

        base = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(weighting_mode="width"),
            context_key=("stability", "base"),
        )
        filtered = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(
                weighting_mode="mass",
                exclude_low_density=True,
                min_density_g_cm3=0.05,
                exclude_opposite_velocity=True,
                zone_index_upper=900,
            ),
            context_key=("stability", "filtered"),
        )

        self.assertNotEqual(base.selected_zone_count, filtered.selected_zone_count)
        self.assertNotAlmostEqual(base.transmission.areal_density_g_cm2, filtered.transmission.areal_density_g_cm2, places=8)
        self.assertTrue(base.xrd.layers and filtered.xrd.layers)
        self.assertNotAlmostEqual(
            base.xrd.layers[-1].compressed_density_g_cm3,
            filtered.xrd.layers[-1].compressed_density_g_cm3,
            places=8,
        )

    def test_empty_selection_disables_plot_panels_without_crashing(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(derived_region_ids=(999,)),
            context_key=("stability", "empty"),
        )

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.set_result(result)
            process_events(50)
            for tab_index in range(workspace.result_tabs.count()):
                workspace.result_tabs.setCurrentIndex(tab_index)
                process_events(10)
            self.assertEqual(result.selected_zone_count, 0)
            self.assertFalse(workspace.xrd_plot_panel.empty_state_label.isHidden())
            self.assertFalse(workspace.xrd_plot_panel.time_combo.isEnabled())
            self.assertFalse(workspace.plasmon_plot_panel.time_combo.isEnabled())
            self.assertFalse(workspace.spectroscopy_plot_panel.profile_combo.isEnabled())
        finally:
            workspace.close()

    def test_display_only_switches_refresh_xrd_and_spectroscopy_without_recompute(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("stability", "display"),
        )

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.set_result(result)
            process_events(20)

            workspace.xrd_display_combo.setCurrentIndex(workspace.xrd_display_combo.findData("q"))
            process_events(20)
            self.assertEqual(workspace.xrd_plot_panel.time_combo.currentData(), "q_compressed")

            workspace.spectroscopy_shift_unit_combo.setCurrentIndex(workspace.spectroscopy_shift_unit_combo.findData("ev"))
            process_events(20)
            self.assertIn("eV", workspace.spectroscopy_summary_label.text())
            self.assertIn("eV", workspace.spectroscopy_metrics.toPlainText())
        finally:
            workspace.close()

    def test_controller_does_not_submit_duplicate_inflight_request(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)

        controller = DerivedController()
        workspace = controller.widget()
        original_submit = controller._tasks.submit
        try:
            workspace.show()
            controller._context = context.copy()
            workspace.set_context(context)
            request_key = controller._build_request_key(workspace.parameters())
            controller._active_task_id = "inflight-task"
            controller._active_task_request_key = request_key
            controller._active_request_key = request_key

            submit_called = {"value": False}

            def _submit_wrapper(*args, **kwargs):
                submit_called["value"] = True
                return original_submit(*args, **kwargs)

            controller._tasks.submit = _submit_wrapper  # type: ignore[method-assign]
            controller._start_recompute()
            process_events(20)
            self.assertEqual("inflight-task", controller._active_task_id)
            self.assertEqual(request_key, controller._active_task_request_key)
            self.assertFalse(submit_called["value"])
        finally:
            controller._tasks.submit = original_submit  # type: ignore[method-assign]
            controller.shutdown()
            workspace.close()

    def test_workspace_parameter_changes_are_debounced(self) -> None:
        workspace = HeliosDerivedWorkspace()
        emitted = {"count": 0}
        try:
            workspace.parameters_changed.connect(lambda: emitted.__setitem__("count", emitted["count"] + 1))
            for value in (8.1, 8.2, 8.3, 8.4, 8.5):
                workspace.xrd_energy_spin.setValue(value)
                process_events(5)
            wait_until(lambda: emitted["count"] > 0, timeout_s=5.0)
            process_events(100)
            self.assertEqual(emitted["count"], 1)
        finally:
            workspace.close()

    def test_display_settings_convert_controls_back_to_native_units(self) -> None:
        workspace = HeliosDerivedWorkspace()
        try:
            updated = replace(
                default_viewer_settings(),
                angle_unit="rad",
                photon_unit="eV",
                density_unit="kg/m3",
            )
            workspace.set_display_settings(updated)
            workspace.los_angle_spin.setValue(math.pi / 6.0)
            workspace.xrd_angle_spin.setValue(math.pi / 4.0)
            workspace.plasmon_angle_spin.setValue(math.pi / 2.0)
            workspace.xrd_energy_spin.setValue(8200.0)
            workspace.plasmon_energy_spin.setValue(7900.0)
            workspace.spectroscopy_wavelength_spin.setValue(photon_energy_ev_from_wavelength_nm(500.0))
            workspace.min_density_spin.setValue(2500.0)

            parameters = workspace.parameters()
            self.assertAlmostEqual(parameters.line_of_sight_angle_deg, 30.0, places=4)
            self.assertAlmostEqual(parameters.xrd_initial_bragg_angle_deg, 45.0, places=4)
            self.assertAlmostEqual(parameters.plasmon_scattering_angle_deg, 90.0, places=4)
            self.assertAlmostEqual(parameters.xrd_photon_energy_kev, 8.2, places=6)
            self.assertAlmostEqual(parameters.plasmon_photon_energy_kev, 7.9, places=6)
            self.assertAlmostEqual(parameters.spectroscopy_line_wavelength_nm, 500.0, delta=1.0e-5)
            self.assertAlmostEqual(parameters.min_density_g_cm3, 2.5, places=6)
        finally:
            workspace.close()

    def test_controller_cancels_superseded_inflight_request(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)

        controller = DerivedController()
        workspace = controller.widget()
        original_cancel = controller._tasks.cancel
        try:
            workspace.show()
            controller._active = True
            controller._context = context.copy()
            workspace.set_context(context)
            controller._active_task_id = "running-task"
            cancelled = {"task_id": None}

            def _cancel(task_id):
                cancelled["task_id"] = task_id
                return original_cancel(task_id)

            controller._tasks.cancel = _cancel  # type: ignore[method-assign]
            workspace.zone_upper_spin.setValue(100)
            process_events(10)
            controller._start_recompute()
            self.assertEqual("running-task", cancelled["task_id"])
            self.assertIsNotNone(controller._pending_request)
        finally:
            controller._tasks.cancel = original_cancel  # type: ignore[method-assign]
            controller.shutdown()
            workspace.close()


if __name__ == "__main__":
    unittest.main()
