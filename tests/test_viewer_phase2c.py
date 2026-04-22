from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6 import QtGui

from _viewer_test_utils import HDF5_ROOT, combo_set_data, find_row_by_data, get_app, process_events, reset_test_settings, wait_until

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosRun
from helios_viewer.main_window import HeliosViewerMainWindow
from helios_viewer.units import DisplayUnitChoices, EV_TO_K, convert_field_values, unit_options_for_field


def allclose_nan(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def assert_field_map_payload(window: HeliosViewerMainWindow, expected: np.ndarray) -> None:
    if window.field_map_widget.current_render_mode == "mesh":
        actual = window.field_map_widget.last_mesh_z
    else:
        actual = window.field_map_widget.last_display_image
        expected = np.asarray(expected, dtype=np.float64).T
    allclose_nan(actual, expected)


def select_field(window: HeliosViewerMainWindow, field_name: str) -> None:
    row = find_row_by_data(window.field_list, field_name)
    if row < 0:
        raise AssertionError(f"Field {field_name!r} not found.")
    window.field_list.setCurrentRow(row)
    wait_until(lambda: window.current_field_payload is not None and window.current_field_payload.field_name == field_name, 30.0)


class ViewerPhase2cTests(unittest.TestCase):
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

    def test_time_trace_reference_controls_follow_coordinate_modes(self) -> None:
        name = "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        window = self.open_window(name)
        try:
            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
            with HeliosRun(HDF5_ROOT / name) as run:
                combo_set_data(window.slice_mode_combo, "time_trace")

                combo_set_data(window.line_coordinate_combo, "zone")
                self.assertEqual(window.trace_reference_stack.currentIndex(), 0)
                self.assertEqual(window.trace_reference_row_label.text(), "Reference zone")
                window.trace_zone_spin.setValue(500)
                expected_zone = int(window._resolved_trace_zone_index()) + 1
                wait_until(
                    lambda: f"zone {expected_zone}" in window.trace_reference_label.text().lower()
                    or "requested zone" in window.trace_reference_label.text().lower(),
                    timeout_s=5.0,
                )
                self.assertIn(f"zone {expected_zone}", window.trace_reference_label.text().lower())

                combo_set_data(window.line_coordinate_combo, "static_x")
                process_events()
                self.assertEqual(window.trace_reference_stack.currentIndex(), 1)
                self.assertIn("static x", window.trace_reference_row_label.text().lower())
                target_x = float(run.get_grid("x")[499])
                window.trace_coordinate_spin.setValue(window._display_length_value(target_x))
                wait_until(lambda: "trace uses zone" in window.trace_reference_label.text().lower(), timeout_s=5.0)
                expected_zone = int(window._resolved_trace_zone_index()) + 1
                self.assertIn("static x =", window.trace_reference_label.text().lower())
                self.assertIn(f"trace uses zone {expected_zone}", window.trace_reference_label.text().lower())
                self.assertIn("static x", window.lineout_plot.current_title.lower())

                window.snapshot_slider.setValue(20)
                process_events()
                combo_set_data(window.line_coordinate_combo, "moving_radius")
                process_events()
                self.assertEqual(window.trace_reference_stack.currentIndex(), 1)
                moving_label = window._coordinate_mode_text("moving_radius").lower()
                self.assertIn(moving_label, window.trace_reference_row_label.text().lower())
                target_radius = float(run.get_radius(snapshot_index=20)[600])
                window.trace_coordinate_spin.setValue(window._display_length_value(target_radius))
                wait_until(lambda: "trace uses zone" in window.trace_reference_label.text().lower(), timeout_s=5.0)
                expected_zone = int(np.argmin(np.abs(np.asarray(run.get_radius(snapshot_index=20), dtype=np.float64) - target_radius))) + 1
                self.assertIn(moving_label, window.trace_reference_label.text().lower())
                self.assertIn("snapshot 20", window.trace_reference_label.text().lower())
                self.assertIn(f"zone {expected_zone}", window.trace_reference_label.text().lower())
                self.assertIn(moving_label, window.lineout_plot.current_title.lower())
        finally:
            window.close()

    def test_mouse_mode_fine_adjustment_controls_resolve_nearest_samples_without_rerendering_map(self) -> None:
        name = "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        window = self.open_window(name)
        try:
            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
            combo_set_data(window.map_coordinate_combo, "moving_radius")
            window.plot_tabs.setCurrentWidget(window.mouse_tab)
            process_events()
            window._set_probe_selection(10, 40, frozen=True)
            process_events()

            render_count = window.field_map_widget.render_call_count
            mesh_count = window.field_map_widget.mesh_render_count

            self.assertTrue(window.mouse_time_slider.isEnabled())
            self.assertTrue(window.mouse_coordinate_slider.isEnabled())
            self.assertIn(window._coordinate_mode_text("moving_radius").lower(), window.mouse_coordinate_row_label.text().lower())

            window.mouse_time_slider.setValue(15)
            process_events()
            self.assertEqual(window._probe_snapshot_index, 15)

            time_values = window._display_time_values(np.asarray(window.run_payload.time, dtype=np.float64))
            window.mouse_time_spin.setValue(float(time_values[18]) * 1.0000001)
            process_events()
            self.assertEqual(window._probe_snapshot_index, 18)

            radius_values = window._display_length_values(window.radius_payload.data[18])
            window.mouse_coordinate_slider.setValue(120)
            process_events()
            self.assertEqual(window._probe_zone_index, 120)

            window.mouse_coordinate_spin.setValue(float(radius_values[333]))
            process_events()
            self.assertEqual(window._probe_zone_index, 333)
            self.assertEqual(window.field_map_widget.render_call_count, render_count)
            self.assertEqual(window.field_map_widget.mesh_render_count, mesh_count)
            self.assertIn("zone 334", window.mouse_adjustment_label.text().lower())
        finally:
            window.close()

    def test_snapshot_controls_are_slice_view_only_and_zoom_persists_across_benign_changes(self) -> None:
        window = self.open_window("5Fe+4.9TW+light_stabilized.h5")
        try:
            self.assertFalse(window.snapshot_controls_widget.isHidden())
            field_view = window.field_map_widget._plot.getViewBox()
            line_view = window.lineout_plot._plot.getViewBox()

            field_full_before = np.asarray(field_view.viewRange(), dtype=np.float64)
            line_full_before = np.asarray(line_view.viewRange(), dtype=np.float64)
            x_start = float(field_full_before[0, 0] + 0.2 * (field_full_before[0, 1] - field_full_before[0, 0]))
            x_stop = float(field_full_before[0, 0] + 0.4 * (field_full_before[0, 1] - field_full_before[0, 0]))
            y_start = float(field_full_before[1, 0] + 0.2 * (field_full_before[1, 1] - field_full_before[1, 0]))
            y_stop = float(field_full_before[1, 0] + 0.4 * (field_full_before[1, 1] - field_full_before[1, 0]))
            line_x_start = float(line_full_before[0, 0] + 0.2 * (line_full_before[0, 1] - line_full_before[0, 0]))
            line_x_stop = float(line_full_before[0, 0] + 0.4 * (line_full_before[0, 1] - line_full_before[0, 0]))

            field_view.setXRange(x_start, x_stop, padding=0.0)
            field_view.setYRange(y_start, y_stop, padding=0.0)
            line_view.setXRange(line_x_start, line_x_stop, padding=0.0)
            process_events()
            field_before = np.asarray(field_view.viewRange(), dtype=np.float64)
            line_before = np.asarray(line_view.viewRange(), dtype=np.float64)

            combo_set_data(window.colormap_combo, "inferno")
            process_events()
            np.testing.assert_allclose(np.asarray(field_view.viewRange(), dtype=np.float64), field_before, rtol=1e-6, atol=1e-12)

            window.plot_tabs.setCurrentWidget(window.mouse_tab)
            process_events()
            self.assertTrue(window.snapshot_controls_widget.isHidden())
            window.plot_tabs.setCurrentWidget(window.lineout_plot)
            process_events()
            self.assertFalse(window.snapshot_controls_widget.isHidden())
            np.testing.assert_allclose(np.asarray(field_view.viewRange(), dtype=np.float64), field_before, rtol=1e-6, atol=1e-12)
            np.testing.assert_allclose(np.asarray(line_view.viewRange(), dtype=np.float64), line_before, rtol=1e-6, atol=1e-12)

            window._reset_plot_views()
            process_events()
            self.assertFalse(np.allclose(np.asarray(field_view.viewRange(), dtype=np.float64), field_before))
        finally:
            window.close()

    def test_display_unit_coverage_and_colorbar_cleanup(self) -> None:
        window = self.open_window("Cu_0166_stabilized.h5")
        try:
            with HeliosRun(HDF5_ROOT / "Cu_0166_stabilized.h5") as run:
                window._viewer_settings.time_unit = "ns"
                window._viewer_settings.length_unit = "um"
                window._viewer_settings.pressure_unit = "Mbar"
                window._viewer_settings.density_unit = "kg/m3"
                window._viewer_settings.temperature_unit = "K"
                window._viewer_settings.velocity_unit = "km/s"
                window._viewer_settings.specific_energy_unit = "MJ/kg"
                window._viewer_settings.rate_unit = "TW/kg"
                window._viewer_settings.heat_capacity_unit = "J/kg/K"
                window._viewer_settings.number_density_unit = "1/m3"
                window._display_field_cache_key = None
                window._display_field_cache_value = None
                window._refresh_field_list_labels()
                window._refresh_visuals(preserve_view=True)

                select_field(window, "velocity")
                expected_velocity = np.asarray(run.get_field("velocity"), dtype=np.float64) * 1.0e-5
                assert_field_map_payload(window, expected_velocity)
                self.assertEqual(window.field_map_widget.current_colorbar_label, "Velocity [km/s]")
                self.assertEqual(window.lineout_plot.current_y_label, "Velocity [km/s]")

                select_field(window, "ion_energy")
                expected_energy = np.asarray(run.get_field("ion_energy"), dtype=np.float64) * 1.0e-3
                assert_field_map_payload(window, expected_energy)
                self.assertEqual(window.field_map_widget.current_colorbar_label, "Ion specific energy [MJ/kg]")

                select_field(window, "laser_deposition")
                expected_rate = np.asarray(run.get_field("laser_deposition"), dtype=np.float64) * 1.0e-9
                assert_field_map_payload(window, expected_rate)
                self.assertEqual(window.field_map_widget.current_colorbar_label, "Laser deposition [TW/kg]")

                select_field(window, "electron_heat_capacity")
                expected_capacity = np.asarray(run.get_field("electron_heat_capacity"), dtype=np.float64) * (1.0e3 / EV_TO_K)
                assert_field_map_payload(window, expected_capacity)
                self.assertEqual(window.field_map_widget.current_colorbar_label, "Electron heat capacity [J/kg/K]")

                select_field(window, "electron_density")
                expected_number_density = np.asarray(run.get_field("electron_density"), dtype=np.float64) * 1.0e6
                assert_field_map_payload(window, expected_number_density)
                self.assertEqual(window.field_map_widget.current_colorbar_label, "Electron density [1/m3]")
                self.assertNotIn("Value", window.field_map_widget.current_colorbar_label)

                self.assertEqual(unit_options_for_field("velocity", "cm/s"), ("cm/s", "m/s", "km/s"))
                self.assertEqual(unit_options_for_field("laser_deposition", "J/g/s"), ("J/g/s", "TW/kg"))
                self.assertEqual(unit_options_for_field("electron_heat_capacity", "J/g/eV"), ("J/g/eV", "J/kg/eV", "J/g/K", "J/kg/K"))

                converted, unit = convert_field_values(
                    "temperature_e",
                    np.asarray([1.0], dtype=np.float64),
                    "eV",
                    DisplayUnitChoices(
                        time_unit="ns",
                        length_unit="um",
                        pressure_unit="Mbar",
                        density_unit="kg/m3",
                        temperature_unit="K",
                        velocity_unit="km/s",
                        specific_energy_unit="MJ/kg",
                        rate_unit="TW/kg",
                        heat_capacity_unit="J/kg/K",
                        number_density_unit="1/m3",
                    ),
                )
                self.assertEqual(unit, "K")
                np.testing.assert_allclose(converted, np.asarray([EV_TO_K], dtype=np.float64))
        finally:
            window.close()

    def test_export_png_pdf_and_transparent_background_work(self) -> None:
        window = self.open_window("5Fe+4.9TW+light_stabilized.h5")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                png_path = root / "field_map.png"
                transparent_png_path = root / "field_map_transparent.png"
                pdf_path = root / "field_map.pdf"

                window._save_png_export(window.field_map_widget, png_path, transparent=False)
                window._save_png_export(window.field_map_widget, transparent_png_path, transparent=True)
                window._save_pdf_export(window.field_map_widget, pdf_path)

                self.assertTrue(png_path.exists())
                self.assertTrue(pdf_path.exists())
                self.assertGreater(png_path.stat().st_size, 0)
                self.assertGreater(pdf_path.stat().st_size, 0)

                transparent = QtGui.QImage(str(transparent_png_path)).convertToFormat(QtGui.QImage.Format_ARGB32)
                bits = transparent.bits()
                alpha = np.frombuffer(bits, dtype=np.uint8, count=transparent.sizeInBytes()).reshape((transparent.height(), transparent.width(), 4))[:, :, 3]
                self.assertTrue(bool(np.any(alpha == 0)))
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
