from __future__ import annotations

import unittest

import numpy as np
from PySide6 import QtGui

from _viewer_test_utils import HDF5_ROOT, combo_set_data, get_app, process_events, reset_test_settings, wait_until

import _test_bootstrap  # noqa: F401
from helios_viewer.main_window import HeliosViewerMainWindow
from helios_viewer.settings import ViewerSettingsDialog, default_viewer_settings


def _to_linear(channel: int) -> float:
    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def contrast_ratio(color_a: QtGui.QColor, color_b: QtGui.QColor) -> float:
    def luminance(color: QtGui.QColor) -> float:
        r = _to_linear(color.red())
        g = _to_linear(color.green())
        b = _to_linear(color.blue())
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    light = max(luminance(color_a), luminance(color_b))
    dark = min(luminance(color_a), luminance(color_b))
    return (light + 0.05) / (dark + 0.05)


class ViewerPhase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def open_window(self, name: str) -> HeliosViewerMainWindow:
        reset_test_settings()
        window = HeliosViewerMainWindow()
        window.load_file(HDF5_ROOT / name)
        wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None, timeout_s=30.0)
        return window

    def test_laser_metadata_summary_and_overlay_are_interpreted_correctly(self) -> None:
        window = self.open_window("10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
        try:
            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
            info = window._laser_entry_info
            self.assertIsNotNone(info)
            self.assertEqual(info["incident_boundary"], "high-index boundary")
            self.assertEqual(info["first_physical_zone"], 1300)
            self.assertEqual(info["incident_region"], 3)
            self.assertEqual(info["incident_region_boundary"], "Region 3 high-index boundary")
            self.assertIn("Laser entry: high-index boundary", window.summary_text.toPlainText())
            self.assertIn("First illuminated zone: 1300", window.summary_text.toPlainText())
            self.assertTrue(window.field_map_widget.current_laser_entry_visible)

            combo_set_data(window.map_coordinate_combo, "static_x")
            wait_until(lambda: window._map_coordinate_mode() == "static_x", timeout_s=5.0)
            wait_until(lambda: window.field_map_widget.current_laser_entry_position is not None, timeout_s=5.0)
            static_edges = window._static_x_edge_values()
            self.assertAlmostEqual(window.field_map_widget.current_laser_entry_position, float(static_edges[-1]))
            self.assertIsNone(window.field_map_widget.current_laser_entry_curve)

            combo_set_data(window.map_coordinate_combo, "zone")
            wait_until(lambda: window._map_coordinate_mode() == "zone", timeout_s=5.0)
            wait_until(lambda: window.field_map_widget.current_laser_entry_position is not None, timeout_s=5.0)
            self.assertAlmostEqual(window.field_map_widget.current_laser_entry_position, 1300.5)

            combo_set_data(window.map_coordinate_combo, "moving_radius")
            wait_until(lambda: window._map_coordinate_mode() == "moving_radius", timeout_s=5.0)
            wait_until(lambda: window.field_map_widget.current_laser_entry_curve is not None, timeout_s=5.0)
            self.assertIsNone(window.field_map_widget.current_laser_entry_position)
            self.assertIsNotNone(window.field_map_widget.current_laser_entry_curve)
            self.assertEqual(window.field_map_widget.current_laser_entry_curve.shape[1], 2)
            expected_curve = np.column_stack(
                [
                    np.asarray(window.run_payload.time, dtype=np.float64),
                    np.asarray(window.field_map_widget.last_mesh_y[:-1, -1], dtype=np.float64),
                ]
            )
            np.testing.assert_allclose(window.field_map_widget.current_laser_entry_curve, expected_curve, rtol=1e-12, atol=1e-12)
        finally:
            window.close()

    def test_moving_mesh_hover_reuses_mesh_and_does_not_rerender_map(self) -> None:
        window = self.open_window("10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
        try:
            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
            combo_set_data(window.map_orientation_combo, "time_x_coord_y")
            combo_set_data(window.map_coordinate_combo, "moving_radius")
            window.plot_tabs.setCurrentWidget(window.mouse_tab)
            process_events()

            render_count = window.field_map_widget.render_call_count
            mesh_count = window.field_map_widget.mesh_render_count
            cache_misses = window._moving_mesh_cache_misses
            curve_updates_before = window.mouse_vertical_plot.curve_item_update_count

            radius = np.asarray(window.radius_payload.data, dtype=np.float64)
            time_values = np.asarray(window.run_payload.time, dtype=np.float64)
            for step in range(40):
                snapshot_index = step % time_values.size
                zone_index = (step * 29) % radius.shape[1]
                window._on_map_probe_moved(float(time_values[snapshot_index]), float(radius[snapshot_index, zone_index]))
                process_events(15)

            self.assertEqual(window.field_map_widget.render_call_count, render_count)
            self.assertEqual(window.field_map_widget.mesh_render_count, mesh_count)
            self.assertEqual(window._moving_mesh_cache_misses, cache_misses)
            self.assertGreater(window.mouse_vertical_plot.curve_item_update_count, curve_updates_before)

            hits_before = window._moving_mesh_cache_hits
            misses_before = window._moving_mesh_cache_misses
            window._refresh_field_map()
            process_events()
            self.assertEqual(window._moving_mesh_cache_misses, misses_before)
            self.assertGreater(window._moving_mesh_cache_hits, hits_before)
        finally:
            window.close()

    def test_theme_switching_and_settings_persistence_work(self) -> None:
        reset_test_settings()
        defaults = default_viewer_settings()
        window = HeliosViewerMainWindow()
        try:
            window._set_theme_mode("dark")
            combo_set_data(window.colormap_combo, "inferno")
            combo_set_data(window.map_scale_combo, "log10")
            combo_set_data(window.line_scale_combo, "signed_log10")
            combo_set_data(window.diagnostic_scale_combo, "log10")
            combo_set_data(window.clip_mode_combo, "percentile")
            window.boundary_overlay_checkbox.setChecked(False)
            window._viewer_settings.hover_interval_ms = 24
            window._save_current_preferences()

            self.assertEqual(window._theme.name, "dark")
            palette = self.app.palette()
            self.assertGreater(contrast_ratio(palette.color(QtGui.QPalette.Window), palette.color(QtGui.QPalette.WindowText)), 7.0)
            self.assertIn("#111827", self.app.styleSheet())
        finally:
            window.close()

        reopened = HeliosViewerMainWindow()
        try:
            self.assertEqual(reopened._theme_mode, "dark")
            self.assertEqual(reopened.colormap_combo.currentData(), "inferno")
            self.assertEqual(reopened.map_scale_combo.currentData(), "log10")
            self.assertEqual(reopened.line_scale_combo.currentData(), "signed_log10")
            self.assertEqual(reopened.diagnostic_scale_combo.currentData(), "log10")
            self.assertEqual(reopened.clip_mode_combo.currentData(), "percentile")
            self.assertFalse(reopened.boundary_overlay_checkbox.isChecked())
            self.assertEqual(reopened._viewer_settings.hover_interval_ms, 24)

            reopened._reset_preferences_to_defaults()
            reset = HeliosViewerMainWindow()
            try:
                self.assertEqual(reset._theme_mode, defaults.theme_mode)
                self.assertEqual(reset.colormap_combo.currentData(), defaults.colormap)
                self.assertEqual(reset.map_scale_combo.currentData(), defaults.map_scale_mode)
                self.assertEqual(reset.line_scale_combo.currentData(), defaults.line_scale_mode)
                self.assertEqual(reset.diagnostic_scale_combo.currentData(), defaults.diagnostic_scale_mode)
                self.assertEqual(reset.clip_mode_combo.currentData(), defaults.clip_mode)
                self.assertEqual(reset._viewer_settings.hover_interval_ms, defaults.hover_interval_ms)
            finally:
                reset.close()
        finally:
            reopened.close()

    def test_settings_dialog_reset_restores_dialog_defaults(self) -> None:
        dialog = ViewerSettingsDialog(default_viewer_settings())
        try:
            dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("dark"))
            dialog.hover_interval_spin.setValue(40)
            dialog._reset_defaults()
            self.assertEqual(dialog.theme_combo.currentData(), "light")
            self.assertEqual(dialog.hover_interval_spin.value(), default_viewer_settings().hover_interval_ms)
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
