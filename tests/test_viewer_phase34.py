from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import HDF5_ROOT, ROOT, combo_set_data, find_row_by_data, get_app, process_events, reset_test_settings, wait_until
from helios_parser import HeliosRun, write_hdf5
from helios_viewer.main_window import ExportDialog, HeliosViewerMainWindow
from helios_viewer.plots import FieldMapWidget, resolve_colormap


def _open_window(name: str) -> HeliosViewerMainWindow:
    reset_test_settings()
    window = HeliosViewerMainWindow()
    window.load_file(HDF5_ROOT / name)
    wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None, timeout_s=30.0)
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


def _expected_coordinate_boundaries(edges: np.ndarray, max_zones: np.ndarray, active_mask: np.ndarray) -> list[float]:
    positions: list[float] = []
    coordinate_edges = np.asarray(edges, dtype=np.float64)
    for max_zone in np.asarray(max_zones, dtype=np.int32)[:-1]:
        left = int(max_zone) - 1
        right = int(max_zone)
        if active_mask[left] or active_mask[right]:
            positions.append(float(coordinate_edges[int(max_zone)]))
    return positions


class ViewerPhase34Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_required_colormaps_load_and_apply(self) -> None:
        for name in ("turbo", "viridis", "plasma", "inferno", "magma", "jet", "hot", "gray", "grey"):
            self.assertIsNotNone(resolve_colormap(name))

        window = _open_window("5Fe+4.9TW+light_stabilized.h5")
        try:
            row = find_row_by_data(window.field_list, "density")
            window.field_list.setCurrentRow(row)
            window.plot_tabs.setCurrentWidget(window.lineout_plot)
            process_events()

            for ui_name in ("turbo", "viridis", "plasma", "inferno", "magma", "jet", "hot", "gray"):
                combo_set_data(window.colormap_combo, ui_name)
                process_events(20)
                self.assertEqual(window.field_map_widget.current_colormap, ui_name)
                self.assertEqual(window.lineout_plot.current_colormap, ui_name)
                self.assertGreaterEqual(window.lineout_plot.current_curve_count, 1)

            window.field_map_widget.set_colormap("grey")
            window.lineout_plot.set_colormap("grey")
            self.assertEqual(window.lineout_plot.current_colormap, "grey")
        finally:
            window.close()

    def test_cylindrical_runs_use_radius_labels_and_explicit_edge_boundaries(self) -> None:
        handle, target_text = tempfile.mkstemp(suffix="_Cu1e17_cyl.h5")
        os.close(handle)
        target = Path(target_text)
        write_hdf5(ROOT / "Cu1e17_cyl.log", target, overwrite=True)
        reset_test_settings()
        window = HeliosViewerMainWindow()
        try:
            window.load_file(target)
            wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None, timeout_s=30.0)
            combo_set_data(window.map_coordinate_combo, "static_x")
            combo_set_data(window.line_coordinate_combo, "static_x")
            process_events(50)
            with HeliosRun(target) as run:
                edges = np.asarray(run.get_static_coordinate(location="edge"), dtype=np.float64)
                displayed_edges = window._display_length_values(edges)
                active = np.ones(run.n_zones, dtype=bool)
                max_zones = np.asarray(run.get_regions()["max_zone_index"], dtype=np.int32)
                self.assertEqual(window._coordinate_mode_text("static_x", capitalize=True), "Radius")
                self.assertEqual(window.map_coordinate_combo.itemText(window.map_coordinate_combo.findData("static_x")), "Radius (legacy)")
                self.assertEqual(window.line_coordinate_combo.itemText(window.line_coordinate_combo.findData("static_x")), "Radius (legacy)")
                self.assertIn("Radius [", window.field_map_widget.current_y_label)
                self.assertIn("Radius [", window.lineout_plot.current_x_label)
                self.assertIn("radius", window.coordinate_note_label.text().lower())
                self.assertGreaterEqual(float(edges[0]), 0.0)
                self.assertEqual(
                    list(window.field_map_widget.current_boundary_positions),
                    _expected_coordinate_boundaries(displayed_edges, max_zones, active),
                )
        finally:
            window.close()
            process_events(50)
            wait_until(lambda: not window.controller.worker_thread.isRunning(), timeout_s=5.0)
            try:
                target.unlink()
            except PermissionError:
                pass

    def test_field_map_overlay_items_are_reused_across_refreshes(self) -> None:
        widget = FieldMapWidget()
        try:
            widget._set_boundary_lines((1.0, 2.0), angle=90)
            line_ids = [id(item) for item in widget._boundary_lines]
            self.assertEqual(len(line_ids), 2)
            widget._set_boundary_lines((1.5,), angle=90)
            self.assertEqual(id(widget._boundary_lines[0]), line_ids[0])
            self.assertFalse(widget._boundary_lines[1].isVisible())

            curves = (
                np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
                np.asarray([[0.0, 3.0], [1.0, 4.0]], dtype=np.float64),
            )
            widget._set_boundary_curves(curves)
            curve_ids = [id(item) for item in widget._boundary_curves]
            self.assertEqual(len(curve_ids), 2)
            widget._set_boundary_curves(curves[:1])
            self.assertEqual(id(widget._boundary_curves[0]), curve_ids[0])
            self.assertFalse(widget._boundary_curves[1].isVisible())

            widget._set_inactive_ranges(((0.0, 1.0), (2.0, 3.0)), orientation="coord_x_time_y")
            vertical_pool = widget._inactive_region_pools["vertical"]
            vertical_ids = [id(item) for item in vertical_pool[:2]]
            self.assertEqual(len(vertical_ids), 2)
            widget._set_inactive_ranges(((4.0, 5.0),), orientation="coord_x_time_y")
            self.assertEqual(id(vertical_pool[0]), vertical_ids[0])
            self.assertFalse(vertical_pool[1].isVisible())
        finally:
            widget.close()
            widget.deleteLater()

    def test_field_map_colormap_stays_aligned_across_mesh_and_image_modes(self) -> None:
        window = _open_window("Cu_0166_stabilized.h5")
        try:
            row = find_row_by_data(window.field_list, "density")
            window.field_list.setCurrentRow(row)
            process_events(50)

            combo_set_data(window.colormap_combo, "plasma")
            process_events(50)
            expected_lut = resolve_colormap("plasma").getLookupTable(nPts=256, alpha=True)
            initial_levels = window.field_map_widget._colorbar.levels()

            self.assertEqual(window.field_map_widget.current_render_mode, "mesh")
            mesh_color_map = window.field_map_widget._mesh_item.getColorMap()
            self.assertIsNotNone(mesh_color_map)
            np.testing.assert_array_equal(mesh_color_map.getLookupTable(nPts=256, alpha=True), expected_lut)

            combo_set_data(window.map_coordinate_combo, "zone")
            process_events(100)
            self.assertEqual(window.field_map_widget.current_render_mode, "image")
            image_color_map = window.field_map_widget._image_item.getColorMap()
            self.assertIsNotNone(image_color_map)
            np.testing.assert_array_equal(image_color_map.getLookupTable(nPts=256, alpha=True), expected_lut)
            self.assertEqual(window.field_map_widget._colorbar.levels(), initial_levels)

            combo_set_data(window.map_coordinate_combo, "static_x")
            process_events(100)
            image_color_map = window.field_map_widget._image_item.getColorMap()
            self.assertIsNotNone(image_color_map)
            np.testing.assert_array_equal(image_color_map.getLookupTable(nPts=256, alpha=True), expected_lut)
            self.assertEqual(window.field_map_widget._colorbar.levels(), initial_levels)

            clone = window.field_map_widget._build_export_clone()
            try:
                clone_image_color_map = clone._image_item.getColorMap()
                self.assertIsNotNone(clone_image_color_map)
                np.testing.assert_array_equal(clone_image_color_map.getLookupTable(nPts=256, alpha=True), expected_lut)
                self.assertEqual(clone._colorbar.levels(), initial_levels)
            finally:
                clone.close()
                clone.deleteLater()

            combo_set_data(window.map_coordinate_combo, "moving_radius")
            process_events(100)
            self.assertEqual(window.field_map_widget.current_render_mode, "mesh")
            mesh_color_map = window.field_map_widget._mesh_item.getColorMap()
            self.assertIsNotNone(mesh_color_map)
            np.testing.assert_array_equal(mesh_color_map.getLookupTable(nPts=256, alpha=True), expected_lut)
            self.assertEqual(window.field_map_widget._colorbar.levels(), initial_levels)
        finally:
            window.close()

    def test_png_export_respects_current_viewport_and_requested_sizes(self) -> None:
        window = _open_window("Cu_0166_stabilized.h5")
        try:
            combo_set_data(window.map_coordinate_combo, "static_x")
            process_events(20)
            bounds = window.field_map_widget._current_data_bounds()
            self.assertIsNotNone(bounds)
            assert bounds is not None
            x0 = bounds[0] + (bounds[1] - bounds[0]) * 0.22
            x1 = bounds[0] + (bounds[1] - bounds[0]) * 0.61
            y0 = bounds[2] + (bounds[3] - bounds[2]) * 0.18
            y1 = bounds[2] + (bounds[3] - bounds[2]) * 0.52
            view_box = window.field_map_widget._plot.getViewBox()
            view_box.setXRange(x0, x1, padding=0.0)
            view_box.setYRange(y0, y1, padding=0.0)
            process_events(50)
            x_view, y_view = window.field_map_widget._plot.getViewBox().viewRange()
            expected = (
                float(min(x_view)),
                float(min(y_view)),
                float(max(x_view) - min(x_view)),
                float(max(y_view) - min(y_view)),
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                for width, height in ((1600, 900), (1920, 1080), (2000, 2000)):
                    path = Path(temp_dir) / f"export_{width}x{height}.png"
                    window._save_png_export(
                        window.field_map_widget,
                        path,
                        transparent=bool(width == 2000),
                        width=width,
                        height=height,
                        dpi=300,
                    )
                    image = QtGui.QImage(str(path))
                    self.assertEqual((image.width(), image.height()), (width, height))
                actual = window.field_map_widget.last_png_export_source_rect
                self.assertIsNotNone(actual)
                assert actual is not None
                self.assertAlmostEqual(actual[0], expected[0], places=3)
                self.assertAlmostEqual(actual[1], expected[1], places=3)
                self.assertAlmostEqual(actual[2], expected[2], places=3)
                self.assertAlmostEqual(actual[3], expected[3], places=3)
        finally:
            window.close()

    def test_probe_adjustment_triggers_line_autoscale(self) -> None:
        window = _open_window("10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
        try:
            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
            window.plot_tabs.setCurrentWidget(window.mouse_tab)
            process_events()
            window._set_probe_selection(20, 300, frozen=True)
            process_events(50)

            for plot in (window.mouse_vertical_plot, window.mouse_horizontal_plot):
                bounds = plot._current_data_bounds()
                self.assertIsNotNone(bounds)
                assert bounds is not None
                plot._plot.getViewBox().setXRange(bounds[0], bounds[0] + max(1.0e-12, (bounds[1] - bounds[0]) * 0.01), padding=0.0)
                plot._plot.getViewBox().setYRange(bounds[2], bounds[2] + max(1.0e-12, (bounds[3] - bounds[2]) * 0.01), padding=0.0)
            process_events(20)

            new_snapshot = min(window.mouse_time_slider.maximum(), 40)
            new_coordinate = min(window.mouse_coordinate_slider.maximum(), 550)
            window.mouse_time_slider.setValue(new_snapshot)
            window.mouse_coordinate_slider.setValue(new_coordinate)
            wait_until(
                lambda: window._probe_snapshot_index == new_snapshot and window._probe_zone_index is not None,
                timeout_s=20.0,
            )
            process_events(50)

            self.assertTrue(_curve_view_covers_bounds(window.mouse_vertical_plot))
            self.assertTrue(_curve_view_covers_bounds(window.mouse_horizontal_plot))
        finally:
            window.close()

    def test_export_dialog_limits_dpi_and_zone_is_not_legacy(self) -> None:
        dialog = ExportDialog("phase34.png", {"field_map": QtCore.QSize(1280, 720)})
        try:
            self.assertEqual([dialog.dpi_combo.itemData(index) for index in range(dialog.dpi_combo.count())], [100, 300, 600])
        finally:
            dialog.close()

        window = _open_window("10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
        try:
            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
            zone_text = window.map_coordinate_combo.itemText(window.map_coordinate_combo.findData("zone")).lower()
            static_text = window.map_coordinate_combo.itemText(window.map_coordinate_combo.findData("static_x")).lower()
            self.assertNotIn("legacy", zone_text)
            self.assertIn("legacy", static_text)
        finally:
            window.close()

    def test_zoom_is_preserved_across_orientation_and_coordinate_changes_when_possible(self) -> None:
        window = _open_window("5Fe+4.9TW+light_stabilized.h5")
        try:
            combo_set_data(window.map_coordinate_combo, "static_x")
            process_events(20)
            bounds = window.field_map_widget._current_data_bounds()
            self.assertIsNotNone(bounds)
            assert bounds is not None
            x0 = bounds[0] + (bounds[1] - bounds[0]) * 0.30
            x1 = bounds[0] + (bounds[1] - bounds[0]) * 0.60
            y0 = bounds[2] + (bounds[3] - bounds[2]) * 0.25
            y1 = bounds[2] + (bounds[3] - bounds[2]) * 0.50
            view_box = window.field_map_widget._plot.getViewBox()
            view_box.setXRange(x0, x1, padding=0.0)
            view_box.setYRange(y0, y1, padding=0.0)
            process_events(20)

            combo_set_data(window.map_orientation_combo, "time_x_coord_y")
            process_events(50)
            x_view, y_view = view_box.viewRange()
            full_bounds = window.field_map_widget._current_data_bounds()
            self.assertIsNotNone(full_bounds)
            assert full_bounds is not None
            self.assertLess(float(x_view[1] - x_view[0]), float(full_bounds[1] - full_bounds[0]) * 0.95)
            self.assertLess(float(y_view[1] - y_view[0]), float(full_bounds[3] - full_bounds[2]) * 0.95)

            combo_set_data(window.map_coordinate_combo, "zone")
            process_events(50)
            x_view2, y_view2 = view_box.viewRange()
            full_bounds2 = window.field_map_widget._current_data_bounds()
            self.assertIsNotNone(full_bounds2)
            assert full_bounds2 is not None
            self.assertLess(float(x_view2[1] - x_view2[0]), float(full_bounds2[1] - full_bounds2[0]) * 0.98)
            self.assertLess(float(y_view2[1] - y_view2[0]), float(full_bounds2[3] - full_bounds2[2]) * 0.98)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
