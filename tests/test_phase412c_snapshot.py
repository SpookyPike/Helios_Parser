from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import combo_set_data, get_app, process_events, reset_test_settings, wait_until
from helios.runtime import RunContext
from helios.services.derived import DerivedAnalysisParameters, compute_analysis_result, load_run_data, refresh_analysis_result_for_snapshot
from helios_analysis.workspace import HeliosDerivedWorkspace
from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state


ROOT = Path(__file__).resolve().parents[1]


def _resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No candidate exists: {candidates}")


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


class SnapshotRefreshServiceTests(unittest.TestCase):
    def test_snapshot_refresh_reuses_full_time_traces_and_updates_snapshot_profiles(self) -> None:
        path = _resolve_existing_path(ROOT / "Cu_0166_stabilized.h5", ROOT / "outputs" / "hdf5" / "Cu_0166_stabilized.h5")
        dataset = load_run_data(path)
        base_context = _context_from_dataset(path, dataset, snapshot_index=20)
        params = DerivedAnalysisParameters()

        base = compute_analysis_result(dataset, base_context, parameters=params, context_key=("phase412c", "base"))
        updated_context = base_context.copy()
        updated_context.set_snapshot_index(120)
        refreshed = refresh_analysis_result_for_snapshot(
            dataset,
            updated_context,
            parameters=params,
            context_key=("phase412c", "snapshot"),
            base_result=base,
        )

        self.assertEqual(refreshed.snapshot_index, 120)
        self.assertEqual(len(refreshed.plasmon.time_plots), len(base.plasmon.time_plots))
        self.assertEqual(refreshed.transmission.time_plots[0].key, base.transmission.time_plots[0].key)
        self.assertIs(refreshed.shock, base.shock)

        base_profile = np.asarray(base.plasmon.profile_plots[0].y_series[0], dtype=np.float64)
        refreshed_profile = np.asarray(refreshed.plasmon.profile_plots[0].y_series[0], dtype=np.float64)
        finite = np.isfinite(base_profile) & np.isfinite(refreshed_profile)
        self.assertTrue(np.any(finite))
        self.assertGreater(float(np.nanmax(np.abs(base_profile[finite] - refreshed_profile[finite]))), 0.0)

    def test_workspace_snapshot_context_updates_do_not_reset_filter_controls(self) -> None:
        app = get_app()
        del app
        path = _resolve_existing_path(
            ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5",
            ROOT / "outputs" / "hdf5" / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5",
        )
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=40)
        result = compute_analysis_result(dataset, context, parameters=DerivedAnalysisParameters(), context_key=("phase412c", "workspace"))

        workspace = HeliosDerivedWorkspace()
        try:
            workspace.set_context(context)
            workspace.set_result(result)
            combo_set_data(workspace.weighting_combo, "mass")
            combo_set_data(workspace.profile_coordinate_combo, "moving_radius")
            workspace.exclude_low_density_checkbox.setChecked(True)
            workspace.min_density_spin.setValue(0.05)
            workspace.zone_lower_spin.setValue(15)
            workspace.zone_upper_spin.setValue(320)
            process_events(30)

            next_context = context.copy()
            next_context.set_snapshot_index(41)
            workspace.set_context(next_context)
            process_events(30)

            self.assertEqual(workspace.weighting_combo.currentData(), "mass")
            self.assertEqual(workspace.profile_coordinate_combo.currentData(), "moving_radius")
            self.assertTrue(workspace.exclude_low_density_checkbox.isChecked())
            self.assertAlmostEqual(float(workspace.min_density_spin.value()), 0.05, places=9)
            self.assertEqual(int(workspace.zone_lower_spin.value()), 15)
            self.assertEqual(int(workspace.zone_upper_spin.value()), 320)
            self.assertIn("Updating snapshot 41", workspace.result_status_label.text())
            self.assertTrue(workspace.plasmon_plot_panel.time_plot.current_cursor_visible)
        finally:
            workspace.close()


class SnapshotRefreshAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def open_window(self) -> HeliosParseViewMainWindow:
        reset_test_settings()
        reset_session_state()
        window = HeliosParseViewMainWindow()
        window.show()
        process_events(20)
        return window

    def test_global_slider_drag_latest_request_wins_on_cu(self) -> None:
        window = self.open_window()
        try:
            window._open_path(_resolve_existing_path(ROOT / "Cu_0166_stabilized.h5", ROOT / "outputs" / "hdf5" / "Cu_0166_stabilized.h5"))
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 30.0)
            window._set_mode("derived")
            workspace = window.derived_controller.widget()
            controller = window.derived_controller._controller
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.plasmon_tab))
            process_events(10)
            wait_until(
                lambda: (
                    workspace._current_result is not None
                    and not controller._busy
                    and bool(workspace.plasmon_plot_panel.time_combo.count())
                ),
                60.0,
            )

            final_target = 180
            window._on_global_snapshot_slider_pressed()
            for target in (25, 90, 140, final_target):
                window.global_snapshot_slider.setValue(target)
                window._on_global_snapshot_slider_moved(target)
                process_events(5)
                self.assertIn("updating", window.global_snapshot_label.text().lower())
            window._on_global_snapshot_slider_released()

            wait_until(lambda: window.viewer_controller.current_run_context().snapshot_index == final_target, 30.0)
            wait_until(lambda: workspace._current_result is not None and workspace._current_result.snapshot_index == final_target, 60.0)
            wait_until(lambda: controller._active_task_id is None and controller._pending_request is None and not controller._busy, 60.0)

            self.assertEqual(controller._last_completed_update_kind, "snapshot")
            self.assertLessEqual(controller._tasks.stats().max_thread_count, 1)
            self.assertIn(f"snapshot {final_target}", workspace.plasmon_plot_panel.profile_plot.current_title.lower())
            self.assertTrue(workspace.plasmon_plot_panel.time_plot.current_cursor_visible)
        finally:
            window.close()

    def test_large_run_snapshot_changes_coalesce_and_finish_on_final_target(self) -> None:
        window = self.open_window()
        try:
            window._open_path(
                _resolve_existing_path(
                    ROOT / "50Al+10E+25CH+3.5TW_stabilized.h5",
                    ROOT / "outputs" / "hdf5" / "50Al+10E+25CH+3.5TW_stabilized.h5",
                )
            )
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 90.0)
            window._set_mode("derived")
            workspace = window.derived_controller.widget()
            controller = window.derived_controller._controller
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.transmission_tab))
            process_events(10)
            wait_until(
                lambda: (
                    workspace._current_result is not None
                    and not controller._busy
                    and bool(workspace.transmission_plot_panel.time_combo.count())
                ),
                180.0,
            )

            targets = [5, 25, 75, 125, 175]
            for target in targets:
                window.global_snapshot_spin.setValue(target)
            process_events(20)
            self.assertTrue(
                "updating" in window.global_snapshot_label.text().lower()
                or "updating" in workspace.result_status_label.text().lower()
                or controller._busy
            )

            wait_until(
                lambda: (
                    controller._active_task_id is not None
                    or controller._pending_request is not None
                    or controller._busy
                    or (
                        workspace._current_result is not None
                        and workspace._current_result.snapshot_index == targets[-1]
                    )
                ),
                30.0,
            )
            wait_until(lambda: controller._active_task_id is None and controller._pending_request is None and not controller._busy, 240.0)
            wait_until(lambda: workspace._current_result is not None and workspace._current_result.snapshot_index == targets[-1], 120.0)

            self.assertEqual(window.viewer_controller.current_run_context().snapshot_index, targets[-1])
            self.assertEqual(controller._last_completed_update_kind, "snapshot")
            self.assertLessEqual(controller._tasks.stats().max_thread_count, 1)
            self.assertIn(f"snapshot {targets[-1]}", workspace.transmission_plot_panel.profile_plot.current_title.lower())
            self.assertTrue(workspace.transmission_plot_panel.time_plot.current_cursor_visible)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
