from __future__ import annotations

import math
import unittest

import numpy as np
from PySide6 import QtGui, QtWidgets

from _viewer_test_utils import HDF5_ROOT, combo_set_data, find_row_by_data, get_app, process_events, reset_test_settings, set_checked_values, wait_until

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosRun
from helios_viewer.main_window import HeliosViewerMainWindow


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


def allclose_nan(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def select_field(window: HeliosViewerMainWindow, field_name: str) -> None:
    row = find_row_by_data(window.field_list, field_name)
    if row < 0:
        raise AssertionError(f"Field {field_name!r} not found.")
    window.field_list.setCurrentRow(row)
    wait_until(lambda: window.current_field_payload is not None and window.current_field_payload.field_name == field_name, 30.0)


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.size == 1:
        delta = max(abs(float(centers[0])) * 0.5, 0.5)
        return np.asarray([float(centers[0]) - delta, float(centers[0]) + delta], dtype=np.float64)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def centers_to_corner_grid(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    n_time, n_zone = centers.shape
    zone_edges = np.empty((n_time, n_zone + 1), dtype=np.float64)
    if n_zone == 1:
        delta = np.maximum(np.abs(centers[:, 0]) * 0.5, 0.5)
        zone_edges[:, 0] = centers[:, 0] - delta
        zone_edges[:, 1] = centers[:, 0] + delta
    else:
        zone_edges[:, 1:-1] = 0.5 * (centers[:, :-1] + centers[:, 1:])
        zone_edges[:, 0] = centers[:, 0] - (zone_edges[:, 1] - centers[:, 0])
        zone_edges[:, -1] = centers[:, -1] + (centers[:, -1] - zone_edges[:, -2])

    corners = np.empty((n_time + 1, n_zone + 1), dtype=np.float64)
    if n_time == 1:
        corners[0] = zone_edges[0]
        corners[1] = zone_edges[0]
    else:
        corners[1:-1] = 0.5 * (zone_edges[:-1] + zone_edges[1:])
        corners[0] = zone_edges[0] - (corners[1] - zone_edges[0])
        corners[-1] = zone_edges[-1] + (zone_edges[-1] - corners[-2])
    return corners


def edge_rows_to_corner_grid(values: np.ndarray) -> np.ndarray:
    edge_rows = np.asarray(values, dtype=np.float64)
    n_time, n_edge = edge_rows.shape
    corners = np.empty((n_time + 1, n_edge), dtype=np.float64)
    if n_time == 1:
        corners[0] = edge_rows[0]
        corners[1] = edge_rows[0]
    else:
        corners[1:-1] = 0.5 * (edge_rows[:-1] + edge_rows[1:])
        corners[0] = edge_rows[0] - (corners[1] - edge_rows[0])
        corners[-1] = edge_rows[-1] + (edge_rows[-1] - corners[-2])
    return corners


class ViewerInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def open_window(self, name: str) -> HeliosViewerMainWindow:
        reset_test_settings()
        window = HeliosViewerMainWindow()
        window.load_file(HDF5_ROOT / name)
        wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None, timeout_s=30.0)
        wait_until(lambda: window.current_field_payload is not None and window.current_field_payload.field_name == "density", timeout_s=30.0)
        return window

    def test_theme_and_popup_contrast_are_readable(self) -> None:
        window = HeliosViewerMainWindow()
        try:
            menu = QtWidgets.QMenu(window)
            combo_view = window.map_orientation_combo.view()
            app_palette = self.app.palette()
            self.assertGreater(contrast_ratio(app_palette.color(QtGui.QPalette.Window), app_palette.color(QtGui.QPalette.WindowText)), 7.0)
            self.assertGreater(contrast_ratio(combo_view.palette().color(QtGui.QPalette.Base), combo_view.palette().color(QtGui.QPalette.Text)), 7.0)
            self.assertGreater(contrast_ratio(menu.palette().color(QtGui.QPalette.Window), menu.palette().color(QtGui.QPalette.WindowText)), 7.0)
            self.assertIn("QMenu", self.app.styleSheet())
            self.assertIn("QAbstractItemView", self.app.styleSheet())
        finally:
            window.close()

    def test_snapshot_slider_remains_functional_after_multiple_mode_switches(self) -> None:
        name = "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        window = self.open_window(name)
        try:
            with HeliosRun(HDF5_ROOT / name) as run:
                wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
                combo_set_data(window.map_orientation_combo, "coord_x_time_y")
                combo_set_data(window.map_coordinate_combo, "moving_radius")
                combo_set_data(window.slice_mode_combo, "time_trace")
                combo_set_data(window.line_coordinate_combo, "static_x")
                window.trace_coordinate_spin.setValue(window._display_length_value(float(run.get_grid("x")[499])))
                process_events()
                set_checked_values(window.region_list, {1, 3})
                set_checked_values(window.material_list, {1})
                window._update_filter_summary()
                window._refresh_visuals()
                window.plot_tabs.setCurrentWidget(window.mouse_tab)
                process_events()
                window._set_probe_selection(10, 50, frozen=True)
                combo_set_data(window.slice_mode_combo, "snapshot_lineout")
                combo_set_data(window.line_coordinate_combo, "radius")
                window.plot_tabs.setCurrentWidget(window.lineout_plot)
                process_events()
                window.snapshot_slider.setValue(70)
                process_events()

                combined_mask = np.asarray((run.get_region_mask(1) | run.get_region_mask(3)) & run.get_material_mask(1), dtype=bool)
                expected_y = np.where(combined_mask, np.asarray(run.get_field("density", time_slice=70), dtype=np.float64), np.nan)
                expected_x = np.asarray(run.get_radius(snapshot_index=70), dtype=np.float64)
                allclose_nan(window.lineout_plot.last_y_series[0], expected_y)
                allclose_nan(window.lineout_plot.last_x_values, expected_x)
                self.assertIn("snapshot 70", window.lineout_plot.current_title)
                self.assertIn(f"{float(run.get_time(selection=70)):.4e}", window.snapshot_time_label.text())
        finally:
            window.close()

    def test_mouse_mode_hover_freeze_and_resume_match_field_payloads(self) -> None:
        name = "Cu_0166_stabilized.h5"
        window = self.open_window(name)
        try:
            with HeliosRun(HDF5_ROOT / name) as run:
                field = np.asarray(run.get_field("density"), dtype=np.float64)
                static_x = np.asarray(run.get_grid("x"), dtype=np.float64)
                window.plot_tabs.setCurrentWidget(window.mouse_tab)
                process_events()
                window._on_map_probe_moved(float(run.get_time(selection=1)), float(static_x[4]))
                process_events()
                self.assertEqual(window._probe_mode, "live")
                self.assertTrue(window.field_map_widget.probe_visible)
                self.assertFalse(window.field_map_widget.probe_frozen)
                allclose_nan(window.mouse_vertical_plot.last_y_series[0], field[1])
                allclose_nan(window.mouse_horizontal_plot.last_y_series[0], field[:, 4])
                self.assertIn("live hover", window.mouse_mode_state_label.text().lower())

                window._on_map_probe_clicked(float(run.get_time(selection=1)), float(static_x[4]))
                process_events()
                self.assertEqual(window._probe_mode, "frozen")
                self.assertTrue(window.field_map_widget.probe_frozen)
                self.assertTrue(window.resume_hover_button.isEnabled())
                self.assertIn("frozen", window.mouse_mode_state_label.text().lower())

                window._resume_hover_probe()
                process_events()
                self.assertEqual(window._probe_mode, "live")
                self.assertFalse(window.field_map_widget.probe_frozen)
        finally:
            window.close()

    def test_moving_mesh_render_and_mouse_slices_use_radius_time_surface(self) -> None:
        name = "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        window = self.open_window(name)
        try:
            with HeliosRun(HDF5_ROOT / name) as run:
                wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
                combo_set_data(window.map_orientation_combo, "time_x_coord_y")
                combo_set_data(window.map_coordinate_combo, "moving_radius")
                field = np.asarray(run.get_field("density"), dtype=np.float64)
                radius = np.asarray(run.get_field("radius"), dtype=np.float64)
                radius_edge = run.get_dynamic_coordinate(location="edge")
                time_values = np.asarray(run.get_time(), dtype=np.float64)
                time_edges = centers_to_edges(time_values)
                radius_corners = (
                    edge_rows_to_corner_grid(np.asarray(radius_edge, dtype=np.float64))
                    if radius_edge is not None
                    else centers_to_corner_grid(radius)
                )
                expected_time_grid = np.repeat(time_edges[:, None], radius_corners.shape[1], axis=1)

                self.assertEqual(window.field_map_widget.current_render_mode, "mesh")
                allclose_nan(window.field_map_widget.last_mesh_z, field)
                allclose_nan(window.field_map_widget.last_mesh_x, expected_time_grid)
                allclose_nan(window.field_map_widget.last_mesh_y, radius_corners)

                window.plot_tabs.setCurrentWidget(window.mouse_tab)
                process_events()
                window._on_map_probe_moved(float(time_values[1]), float(radius[1, 4]))
                process_events()
                allclose_nan(window.mouse_vertical_plot.last_x_values, radius[1])
                allclose_nan(window.mouse_vertical_plot.last_y_series[0], field[1])
                allclose_nan(window.mouse_horizontal_plot.last_y_series[0], field[:, 4])
                self.assertTrue(
                    "moving-mesh radius" in window.mouse_mode_probe_label.text().lower()
                    or "moving-mesh x" in window.mouse_mode_probe_label.text().lower()
                )
        finally:
            window.close()

    def test_titles_and_state_labels_stay_consistent_across_mode_switches(self) -> None:
        name = "5Fe+4.9TW+light_stabilized.h5"
        window = self.open_window(name)
        try:
            combo_set_data(window.map_orientation_combo, "coord_x_time_y")
            combo_set_data(window.map_coordinate_combo, "zone")
            combo_set_data(window.slice_mode_combo, "time_trace")
            combo_set_data(window.line_coordinate_combo, "static_x")
            wait_until(lambda: "zone index on x, time on y" in window.field_map_widget.current_title.lower(), timeout_s=5.0)
            self.assertIn("zone index on x, time on y", window.field_map_widget.current_title.lower())
            self.assertIn("time trace", window.lineout_plot.current_title.lower())
            self.assertIn("static x", window.lineout_plot.current_title.lower())
            self.assertIn("Active analysis: time trace", window.active_analysis_label.text())

            window.plot_tabs.setCurrentWidget(window.mouse_tab)
            process_events()
            window._set_probe_selection(1, 4, frozen=True)
            process_events()
            self.assertIn("mouse mode (frozen)", window.active_analysis_label.text().lower())
            self.assertIn("frozen", window.mouse_mode_state_label.text().lower())
        finally:
            window.close()

    def test_rapid_percentile_changes_coalesce_field_map_refresh(self) -> None:
        window = self.open_window("Cu_0166_stabilized.h5")
        try:
            render_before = window.field_map_widget.render_call_count
            for value in (2.0, 3.0, 4.0, 5.0, 6.0):
                window.percentile_low_spin.setValue(value)
            wait_until(lambda: window.field_map_widget.render_call_count > render_before, timeout_s=5.0)
            process_events(100)
            self.assertAlmostEqual(window.percentile_low_spin.value(), 6.0, places=6)
            self.assertLessEqual(window.field_map_widget.render_call_count - render_before, 2)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
