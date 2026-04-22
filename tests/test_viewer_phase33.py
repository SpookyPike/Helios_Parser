from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import HDF5_ROOT, combo_set_data, find_row_by_data, get_app, process_events, reset_test_settings, wait_until
from helios_viewer.main_window import HeliosViewerMainWindow


def _view_covers_bounds(widget) -> bool:
    bounds = widget._current_data_bounds()
    if bounds is None:
        return False
    x_view, y_view = widget._plot.getViewBox().viewRange()
    x_tol = max(1.0e-12, abs(bounds[1] - bounds[0]) * 0.05)
    y_tol = max(1.0e-12, abs(bounds[3] - bounds[2]) * 0.05)
    return (
        x_view[0] <= bounds[0] + x_tol
        and x_view[1] >= bounds[1] - x_tol
        and y_view[0] <= bounds[2] + y_tol
        and y_view[1] >= bounds[3] - y_tol
    )


def _select_field(window: HeliosViewerMainWindow, field_name: str) -> None:
    row = find_row_by_data(window.field_list, field_name)
    if row < 0:
        raise AssertionError(f"Field {field_name!r} not found.")
    window.field_list.setCurrentRow(row)
    wait_until(lambda: window.current_field_payload is not None and window.current_field_payload.field_name == field_name, 30.0)


class ViewerPhase33Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def open_window(self, name: str) -> HeliosViewerMainWindow:
        reset_test_settings()
        window = HeliosViewerMainWindow()
        window.load_file(HDF5_ROOT / name)
        wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None, timeout_s=30.0)
        return window

    def test_mouse_mode_layout_is_compact_and_resizable(self) -> None:
        window = self.open_window("5Fe+4.9TW+light_stabilized.h5")
        try:
            window.plot_tabs.setCurrentWidget(window.mouse_tab)
            process_events()
            self.assertEqual(window.mouse_plot_splitter.orientation(), QtCore.Qt.Horizontal)
            self.assertEqual(window.mouse_plot_splitter.count(), 2)
            self.assertGreaterEqual(window.mouse_time_slider.minimumHeight(), 24)
            self.assertGreaterEqual(window.mouse_coordinate_slider.minimumHeight(), 24)
            sizes = window.mouse_plot_splitter.sizes()
            self.assertEqual(len(sizes), 2)
            self.assertTrue(all(size > 0 for size in sizes))
        finally:
            window.close()

    def test_autoscale_covers_field_map_and_line_plot_without_clipping(self) -> None:
        window = self.open_window("10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
        try:
            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
            _select_field(window, "velocity")
            combo_set_data(window.map_coordinate_combo, "moving_radius")
            window._reset_plot_views()
            process_events()
            self.assertTrue(_view_covers_bounds(window.field_map_widget))

            window.plot_tabs.setCurrentWidget(window.lineout_plot)
            process_events()
            finite = np.asarray(window.lineout_plot.last_y_series[0], dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            self.assertGreater(finite.size, 0)
            x_view, y_view = window.lineout_plot._plot.getViewBox().viewRange()
            self.assertLessEqual(float(y_view[0]), float(np.min(finite)))
            self.assertGreaterEqual(float(y_view[1]), float(np.max(finite)))
        finally:
            window.close()

    def test_png_export_respects_requested_size(self) -> None:
        window = self.open_window("5Fe+4.9TW+light_stabilized.h5")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "phase33_export.png"
                transparent_path = Path(temp_dir) / "phase33_export_transparent.png"
                window._save_png_export(window.field_map_widget, path, transparent=False, width=1400, height=900, dpi=180)
                window._save_png_export(window.field_map_widget, transparent_path, transparent=True, width=1200, height=800, dpi=144)
                image = QtGui.QImage(str(path))
                self.assertEqual((image.width(), image.height()), (1400, 900))
                transparent = QtGui.QImage(str(transparent_path)).convertToFormat(QtGui.QImage.Format_ARGB32)
                bits = transparent.bits()
                alpha = np.frombuffer(bits, dtype=np.uint8, count=transparent.sizeInBytes()).reshape((transparent.height(), transparent.width(), 4))[:, :, 3]
                self.assertTrue(bool(np.any(alpha == 0)))
        finally:
            window.close()

    def test_opening_new_run_resets_mouse_probe_state_and_prefers_moving_mesh_when_available(self) -> None:
        window = self.open_window("10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
        try:
            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
            window.plot_tabs.setCurrentWidget(window.mouse_tab)
            process_events()
            window._set_probe_selection(10, 40, frozen=True)
            process_events()
            self.assertEqual(window._probe_mode, "frozen")

            window.load_file(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5")
            wait_until(
                lambda: window.run_payload is not None
                and window.current_field_payload is not None
                and window.run_payload.path.name == "5Fe+4.9TW+light_stabilized.h5"
                and window.radius_payload is not None
                and window.map_coordinate_combo.currentData() == "moving_radius",
                timeout_s=30.0,
            )
            self.assertIsNone(window._probe_snapshot_index)
            self.assertIsNone(window._probe_zone_index)
            self.assertEqual(window.map_coordinate_combo.currentData(), "moving_radius")
            self.assertEqual(window.mouse_mode_state_label.text(), "Probe: live hover")
            self.assertEqual(window.mouse_plot_splitter.count(), 2)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
