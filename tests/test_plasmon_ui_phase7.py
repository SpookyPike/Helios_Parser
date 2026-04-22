from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

try:
    from PySide6 import QtWidgets  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    QtWidgets = None  # type: ignore

import _test_bootstrap  # noqa: F401

if QtWidgets is not None:
    from _viewer_test_utils import get_app, process_events, reset_test_settings
    from test_plasmon_ui_phase3 import _synthetic_dataset
    from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
    from helios.services.derived.models import PreheatSummary
    from helios_analysis.controller import DerivedController
    from helios_analysis.workspace import HeliosDerivedWorkspace


@unittest.skipIf(QtWidgets is None, 'PySide6 is not available in this environment')
class PlasmonUiPhase7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = get_app()

    def setUp(self) -> None:
        reset_test_settings()

    def test_workspace_enables_export_for_spectral_plasmon_result(self) -> None:
        dataset, context = _synthetic_dataset()
        parameters = DerivedAnalysisParameters(
            plasmon_model='mermin',
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
            plasmon_collision_model='manual_constant',
            plasmon_manual_collision_rate_s=1.5e15,
        )
        result = compute_analysis_result(dataset, context, parameters=parameters, context_key=('ui-phase7', 1), requested_time_plot_modules=frozenset({'plasmon'}), include_wavefront=False)
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace.show()
        workspace.set_result(result)
        workspace.result_tabs.setCurrentWidget(workspace.plasmon_tab)
        process_events(50)
        self.assertTrue(workspace.plasmon_export_button.isEnabled())
        self.assertTrue(bool(workspace.plasmon_export_button.toolTip()))
        workspace.close()

    def test_controller_cancel_button_cancels_active_request_and_clears_busy_ui(self) -> None:
        controller = DerivedController()
        workspace = controller._workspace
        workspace.show()
        workspace.set_busy(True, "Busy test")
        controller._active_task_id = "task-1"
        process_events(20)
        with mock.patch.object(controller._tasks, "cancel") as cancel_mock:
            workspace.cancel_button.click()
            process_events(20)
        cancel_mock.assert_called_once_with("task-1")
        self.assertFalse(workspace.activity_progress.isVisible())
        self.assertFalse(controller._busy)
        controller.shutdown()

    def test_preheat_manual_navigation_follows_global_snapshot_and_disables_local_snapshot_controls(self) -> None:
        dataset, context = _synthetic_dataset()
        del dataset
        context = context.copy()
        context.set_snapshot_index(2)
        workspace = HeliosDerivedWorkspace()
        workspace.set_context(context)
        workspace._preheat_time_mode = "manual"
        preheat = PreheatSummary(
            available=True,
            supported=True,
            method="synthetic",
            candidate_metric_names=(),
            scalar_summaries={},
            snapshot_indices=np.asarray([0, 1, 2], dtype=np.int32),
        )
        selected_snapshot_index, note = workspace._resolve_preheat_snapshot_index(preheat)
        self.assertEqual(selected_snapshot_index, 2)
        self.assertIn("global snapshot", note.lower())
        workspace._sync_preheat_navigation_controls(preheat, selected_snapshot_index=selected_snapshot_index)
        self.assertEqual(workspace.preheat_snapshot_slider.value(), 2)
        self.assertEqual(workspace.preheat_snapshot_spin.value(), 2)
        self.assertFalse(workspace.preheat_snapshot_slider.isEnabled())
        self.assertFalse(workspace.preheat_snapshot_spin.isEnabled())
        self.assertFalse(workspace.preheat_time_spin.isEnabled())
        workspace.close()


if __name__ == '__main__':
    unittest.main()
