from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

import _test_bootstrap  # noqa: F401
from PySide6 import QtCore, QtTest

from _viewer_test_utils import get_app, process_events, reset_test_settings, wait_until, combo_set_data
from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state
from helios_viewer.settings import load_viewer_settings


ROOT = Path(__file__).resolve().parents[1]


class AppPhase412Tests(unittest.TestCase):
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

    def test_immediate_open_to_derived_uses_global_snapshot_state(self) -> None:
        window = self.open_window()
        try:
            window._open_path(ROOT / "Cu_0166_stabilized.h5")
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 30.0)
            self.assertTrue(window.global_snapshot_widget.isVisible())
            self.assertFalse(window.viewer_controller.window.snapshot_controls_widget.isVisible())

            window._set_mode("derived")
            wait_until(lambda: window.derived_controller.widget()._current_result is not None, 30.0)

            window.global_snapshot_spin.setValue(17)
            process_events(50)
            wait_until(lambda: window.viewer_controller.current_run_context().snapshot_index == 17, 30.0)
            wait_until(lambda: window.derived_controller.widget()._current_result is not None and window.derived_controller.widget()._current_result.snapshot_index == 17, 30.0)

            workspace = window.derived_controller.widget()
            self.assertIn("snapshot 17", workspace.xrd_plot_panel.profile_plot.current_title.lower())
            self.assertIn("17", workspace.snapshot_label.text())
            self.assertIn(window.viewer_controller.current_time_unit(), window.global_snapshot_label.text())
        finally:
            window.close()

    def test_fresh_start_uses_requested_defaults(self) -> None:
        window = self.open_window()
        try:
            settings = window.viewer_controller.current_viewer_settings()
            self.assertEqual(settings.theme_mode, "system")
            self.assertEqual(settings.time_unit, "ns")
            self.assertEqual(settings.length_unit, "um")
            self.assertEqual(settings.pressure_unit, "GPa")
            self.assertEqual(settings.density_unit, "g/cm3")
            self.assertEqual(settings.temperature_unit, "eV")
            self.assertEqual(settings.velocity_unit, "km/s")
            self.assertEqual(settings.angle_unit, "deg")
            self.assertEqual(settings.photon_unit, "eV")
            self.assertEqual(settings.default_profile_coordinate, "viewer_follow")
            self.assertTrue(settings.wheel_guard_enabled)
            self.assertEqual(window.viewer_controller.theme_mode(), "system")
            self.assertEqual(window.viewer_controller.default_profile_coordinate_mode(), "viewer_follow")
        finally:
            window.close()

    def test_wheel_guard_setting_persists_and_updates_viewer_and_derived(self) -> None:
        window = self.open_window()
        try:
            viewer = window.viewer_controller.window
            updated = replace(viewer.current_viewer_settings(), wheel_guard_enabled=False)
            viewer._apply_viewer_settings(updated)
            process_events(50)

            current = window.viewer_controller.current_viewer_settings()
            self.assertFalse(current.wheel_guard_enabled)
            self.assertFalse(viewer._wheel_guard._enabled)
            self.assertFalse(window.derived_controller.widget()._wheel_guard._enabled)
            self.assertFalse(load_viewer_settings().wheel_guard_enabled)

            reset_session_state()
            reopened = HeliosParseViewMainWindow()
            reopened.show()
            process_events(20)
            try:
                reopened_current = reopened.viewer_controller.current_viewer_settings()
                self.assertFalse(reopened_current.wheel_guard_enabled)
                self.assertFalse(reopened.viewer_controller.window._wheel_guard._enabled)
                self.assertFalse(reopened.derived_controller.widget()._wheel_guard._enabled)
            finally:
                reopened.close()
        finally:
            window.close()

    def test_viewer_colorbar_survives_roundtrip_and_constant_field(self) -> None:
        window = self.open_window()
        try:
            window._open_path(ROOT / "Cu_0166_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )
            viewer = window.viewer_controller.window
            field_map = viewer.field_map_widget

            def assert_colorbar_state() -> None:
                levels = field_map._colorbar.levels()
                self.assertTrue(field_map.current_colorbar_label)
                self.assertIsNotNone(levels)
                assert levels is not None
                self.assertGreater(float(levels[1]), float(levels[0]))

            assert_colorbar_state()
            viewer.plot_tabs.setCurrentWidget(viewer.mouse_tab)
            process_events(50)
            assert_colorbar_state()

            window._set_mode("derived")
            process_events(100)
            window._set_mode("viewer")
            process_events(100)
            assert_colorbar_state()

            viewer._select_list_item_by_data(viewer.field_list, "zone_width")
            wait_until(lambda: viewer.current_field_name == "zone_width", 20.0)
            process_events(100)
            assert_colorbar_state()

            viewer.set_theme_mode("dark")
            process_events(100)
            assert_colorbar_state()
            viewer.set_theme_mode("light")
            process_events(100)
            assert_colorbar_state()
        finally:
            window.close()

    def test_large_run_control_changes_stay_single_inflight(self) -> None:
        window = self.open_window()
        try:
            window._open_path(ROOT / "50Al+10E+25CH+3.5TW_stabilized.h5")
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 60.0)
            window._set_mode("derived")
            workspace = window.derived_controller.widget()
            controller = window.derived_controller._controller
            wait_until(lambda: workspace._current_result is not None, 120.0)

            max_threads = 0
            for index in range(12):
                combo_set_data(workspace.weighting_combo, "mass" if index % 2 == 0 else "electron_column")
                combo_set_data(workspace.profile_coordinate_combo, "zone" if index % 2 == 0 else "moving_radius")
                workspace.exclude_low_density_checkbox.setChecked(index % 2 == 0)
                workspace.exclude_opposite_velocity_checkbox.setChecked(index % 3 == 0)
                workspace.min_density_spin.setValue(0.05 if index % 2 == 0 else 0.0)
                process_events(20)
                max_threads = max(max_threads, controller._tasks.stats().max_thread_count)
                self.assertLessEqual(controller._tasks.stats().active, 1)

            wait_until(
                lambda: controller._active_task_id is None and controller._pending_request is None and not controller._busy,
                180.0,
            )
            self.assertIsNotNone(workspace._current_result)
            self.assertLessEqual(max_threads, 1)
        finally:
            window.close()

    def test_global_snapshot_slider_updates_large_run_viewer_and_derived(self) -> None:
        window = self.open_window()
        try:
            window._open_path(ROOT / "50Al+10E+25CH+3.5TW_stabilized.h5")
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 60.0)
            context = window.viewer_controller.current_run_context()
            max_index = int(context.n_snapshots) - 1
            target = max_index // 3

            window._request_global_snapshot_index(target, immediate=True)
            process_events(50)
            wait_until(lambda: window.viewer_controller.current_run_context().snapshot_index == target, 30.0)

            window._set_mode("derived")
            workspace = window.derived_controller.widget()
            wait_until(lambda: workspace._current_result is not None and workspace._current_result.snapshot_index == target, 120.0)
            self.assertIn(f"snapshot {target}", workspace.xrd_plot_panel.profile_plot.current_title.lower())
        finally:
            window.close()

    def test_global_display_units_propagate_into_derived_without_reopen(self) -> None:
        window = self.open_window()
        try:
            window._open_path(ROOT / "Cu_0166_stabilized.h5")
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 30.0)
            window._set_mode("derived")
            workspace = window.derived_controller.widget()
            wait_until(lambda: workspace._current_result is not None, 30.0)

            viewer_window = window.viewer_controller.window
            updated = replace(
                viewer_window.current_viewer_settings(),
                time_unit="ps",
                length_unit="nm",
                density_unit="kg/m3",
                temperature_unit="K",
                velocity_unit="m/s",
                number_density_unit="1/m3",
                angle_unit="rad",
                photon_unit="eV",
            )
            viewer_window._viewer_settings = updated
            window._on_viewer_settings_changed(updated)
            process_events(100)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.xrd_tab))
            process_events(100)

            self.assertIn(" ps", window.global_time_spin.suffix())
            self.assertIn("rad", workspace.los_angle_spin.suffix())
            self.assertIn("eV", workspace.xrd_energy_spin.suffix())
            self.assertIn("eV", workspace.spectroscopy_wavelength_spin.suffix())
            self.assertIn("kg/m3", workspace.min_density_spin.suffix())
            self.assertIn("ps", workspace.snapshot_label.text())
            wait_until(lambda: bool(workspace.xrd_plot_panel.time_plot.current_x_label), 30.0)
            self.assertIn("[ps]", workspace.xrd_plot_panel.time_plot.current_x_label)
            self.assertIn("[m/s]", workspace.shock_velocity_plot.current_y_label)
            self.assertIn("K", workspace.plasmon_metrics.toPlainText())
        finally:
            window.close()

    def test_global_snapshot_slider_clicks_settle_to_latest_clicked_position(self) -> None:
        window = self.open_window()
        try:
            window._open_path(ROOT / "Cu_0166_stabilized.h5")
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 30.0)
            slider = window.global_snapshot_slider
            max_index = max(1, int(window.viewer_controller.current_run_context().n_snapshots) - 1)
            right_point = QtCore.QPoint(max(4, int(slider.width() * 0.85)), slider.rect().center().y())
            left_point = QtCore.QPoint(max(2, int(slider.width() * 0.12)), slider.rect().center().y())

            QtTest.QTest.mouseClick(slider, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, right_point)
            wait_until(lambda: slider.value() >= int(max_index * 0.6), 10.0)
            wait_until(lambda: window.viewer_controller.current_run_context().snapshot_index == slider.value(), 30.0)

            QtTest.QTest.mouseClick(slider, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, left_point)
            wait_until(lambda: slider.value() <= int(max_index * 0.4), 10.0)
            wait_until(lambda: window.viewer_controller.current_run_context().snapshot_index == slider.value(), 30.0)
            self.assertNotIn("updating...", window.global_snapshot_label.text().lower())
        finally:
            window.close()

    def test_theme_switch_keeps_derived_plots_rendered(self) -> None:
        window = self.open_window()
        try:
            window._open_path(ROOT / "Cu_0166_stabilized.h5")
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 30.0)
            window._set_mode("derived")
            workspace = window.derived_controller.widget()
            wait_until(lambda: workspace._current_result is not None, 30.0)
            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.xrd_tab))
            wait_until(lambda: bool(workspace.xrd_plot_panel.time_combo.count()), 30.0)

            window._set_theme_mode("dark")
            process_events(100)
            self.assertGreater(workspace.xrd_plot_panel.time_plot.current_curve_count, 0)
            self.assertTrue(workspace.xrd_plot_panel.time_plot.current_x_label)
            self.assertTrue(workspace.shock_velocity_plot.current_y_label)

            workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.plasmon_tab))
            wait_until(lambda: bool(workspace.plasmon_plot_panel.time_combo.count()), 30.0)
            window._set_theme_mode("system")
            process_events(100)
            self.assertGreater(workspace.plasmon_plot_panel.time_plot.current_curve_count, 0)
            self.assertTrue(workspace.plasmon_plot_panel.time_plot.current_x_label)
            self.assertTrue(workspace.spectroscopy_plot_panel.profile_plot.current_y_label)
        finally:
            window.close()

    def test_profile_coordinate_setting_persists_and_recomputes_follow_2d_map(self) -> None:
        dynamic_path = ROOT / "outputs" / "hdf5" / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"

        window = self.open_window()
        try:
            window._open_path(dynamic_path)
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 60.0)
            window._set_mode("derived")
            workspace = window.derived_controller.widget()
            wait_until(lambda: workspace._current_result is not None, 120.0)

            viewer_window = window.viewer_controller.window
            zone_settings = replace(viewer_window.current_viewer_settings(), default_profile_coordinate="zone")
            viewer_window._apply_viewer_settings(zone_settings)
            wait_until(lambda: workspace.profile_coordinate_combo.currentData() == "zone", 30.0)
            wait_until(
                lambda: workspace._current_result is not None
                and workspace._current_result.geometry.profile_coordinate_mode == "zone",
                120.0,
            )
            wait_until(lambda: "Zone index" in workspace.xrd_plot_panel.profile_plot.current_x_label, 120.0)

            follow_settings = replace(zone_settings, default_profile_coordinate="viewer_follow")
            viewer_window._apply_viewer_settings(follow_settings)
            expected_mode = window.viewer_controller.current_run_context().slice_coordinate

            wait_until(lambda: workspace.profile_coordinate_combo.currentData() == "viewer", 30.0)
            wait_until(
                lambda: workspace._current_result is not None
                and workspace._current_result.geometry.profile_coordinate_mode == expected_mode,
                120.0,
            )
            wait_until(
                lambda: "Zone index" not in workspace.xrd_plot_panel.profile_plot.current_x_label,
                120.0,
            )
            self.assertEqual(window.viewer_controller.default_profile_coordinate_mode(), "viewer_follow")
        finally:
            window.close()

        reopened = self.open_window()
        try:
            self.assertEqual(reopened.viewer_controller.default_profile_coordinate_mode(), "viewer_follow")
            reopened._open_path(dynamic_path)
            wait_until(lambda: reopened.viewer_controller.has_loaded_run(), 60.0)
            reopened._set_mode("derived")
            workspace = reopened.derived_controller.widget()
            wait_until(lambda: workspace._current_result is not None, 120.0)
            self.assertEqual(workspace.profile_coordinate_combo.currentData(), "viewer")
            self.assertEqual(
                workspace._current_result.geometry.profile_coordinate_mode,
                reopened.viewer_controller.current_run_context().slice_coordinate,
            )
        finally:
            reopened.close()

    def test_breakout_navigation_uses_canonical_snapshot_request_and_resets_on_run_change(self) -> None:
        window = self.open_window()
        try:
            first_path = ROOT / "Cu_0166_stabilized.h5"
            second_path = ROOT / "5Fe+4.9TW+light_stabilized.h5"
            window._open_path(first_path)
            wait_until(lambda: window.viewer_controller.has_loaded_run(), 30.0)
            context = window.viewer_controller.current_run_context()
            breakout_index = min(5, max(0, int(context.time_values.size) - 1))
            breakout_time_s = float(context.time_values[breakout_index])
            synthetic_result = SimpleNamespace(
                dataset_path=context.path,
                shock=SimpleNamespace(breakout_time_s=breakout_time_s),
            )

            window._on_derived_analysis_ready(synthetic_result)
            self.assertTrue(window.jump_breakout_button.isEnabled())
            self.assertIn(str(breakout_index), window.jump_breakout_button.toolTip())

            with mock.patch.object(window, "_request_global_snapshot_index") as request_snapshot:
                window._jump_to_breakout_snapshot()
                request_snapshot.assert_called_once_with(breakout_index, immediate=True)

            window._open_path(second_path)
            wait_until(
                lambda: window.viewer_controller.has_loaded_run()
                and Path(window.viewer_controller.current_run_context().path).resolve() == second_path.resolve(),
                60.0,
            )
            self.assertFalse(window.jump_breakout_button.isEnabled())
            self.assertIn("until derived Shock analysis completes", window.jump_breakout_button.toolTip())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
