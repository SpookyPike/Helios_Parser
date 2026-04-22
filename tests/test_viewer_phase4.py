from __future__ import annotations

import unittest

import numpy as np

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import HDF5_ROOT, combo_set_data, get_app, process_events, reset_test_settings, wait_until
from helios_viewer.main_window import HeliosViewerMainWindow


def _open_window(name: str) -> HeliosViewerMainWindow:
    reset_test_settings()
    window = HeliosViewerMainWindow()
    window.load_file(HDF5_ROOT / name)
    wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None, timeout_s=60.0)
    return window


def _curve_view_covers_bounds(widget) -> bool:
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


class ViewerPhase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_mouse_mode_live_hover_autoscales_like_slice_view_without_rerender(self) -> None:
        window = _open_window("10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
        try:
            wait_until(lambda: window.radius_payload is not None, timeout_s=60.0)
            combo_set_data(window.map_coordinate_combo, "moving_radius")
            combo_set_data(window.map_orientation_combo, "time_x_coord_y")
            process_events(50)
            window.plot_tabs.setCurrentWidget(window.mouse_tab)
            process_events(50)
            window._resume_hover_probe()
            display_time = window._display_time_values(np.asarray(window.run_payload.time, dtype=np.float64))
            radius_display, _ = window._display_field_data("radius", window.radius_payload.unit, window.radius_payload.data)
            render_before = window.field_map_widget.render_call_count
            mesh_before = window.field_map_widget.mesh_render_count

            for snapshot_index, zone_index in ((20, 300), (60, 900)):
                x_value = float(display_time[snapshot_index])
                y_value = float(radius_display[snapshot_index, zone_index])
                window._on_map_probe_moved(x_value, y_value)
                wait_until(lambda snap=snapshot_index: window._probe_snapshot_index == snap and window._probe_mode == "live", 20.0)
                process_events(50)
                self.assertTrue(_curve_view_covers_bounds(window.mouse_vertical_plot))
                self.assertTrue(_curve_view_covers_bounds(window.mouse_horizontal_plot))

            self.assertEqual(window.field_map_widget.render_call_count, render_before)
            self.assertEqual(window.field_map_widget.mesh_render_count, mesh_before)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
