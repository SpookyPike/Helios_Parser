from __future__ import annotations

import unittest

import numpy as np

from _viewer_test_utils import HDF5_ROOT, combo_set_data, find_row_by_data, get_app, process_events, reset_test_settings, wait_until

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosRun
from helios_viewer.main_window import HeliosViewerMainWindow
from helios_viewer.units import EV_TO_K


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


def apply_display_units(
    window: HeliosViewerMainWindow,
    *,
    time_unit: str = "s",
    length_unit: str = "cm",
    pressure_unit: str = "J/cm3",
    density_unit: str = "g/cm3",
    temperature_unit: str = "eV",
    velocity_unit: str = "cm/s",
    specific_energy_unit: str = "J/g",
    rate_unit: str = "J/g/s",
    heat_capacity_unit: str = "J/g/eV",
    number_density_unit: str = "1/cm3",
) -> None:
    window._viewer_settings.time_unit = time_unit
    window._viewer_settings.length_unit = length_unit
    window._viewer_settings.pressure_unit = pressure_unit
    window._viewer_settings.density_unit = density_unit
    window._viewer_settings.temperature_unit = temperature_unit
    window._viewer_settings.velocity_unit = velocity_unit
    window._viewer_settings.specific_energy_unit = specific_energy_unit
    window._viewer_settings.rate_unit = rate_unit
    window._viewer_settings.heat_capacity_unit = heat_capacity_unit
    window._viewer_settings.number_density_unit = number_density_unit
    window._display_field_cache_key = None
    window._display_field_cache_value = None
    window._refresh_field_list_labels()
    window._refresh_visuals()
    window._refresh_diagnostic_plot()
    process_events()


class ViewerPhase2bTests(unittest.TestCase):
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

    def test_time_trace_semantics_and_overlay_ownership_are_explicit(self) -> None:
        window = self.open_window("10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
        try:
            with HeliosRun(HDF5_ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5") as run:
                density = np.asarray(run.get_field("density"), dtype=np.float64)
                static_x = np.asarray(run.get_grid("x"), dtype=np.float64)
                time = np.asarray(run.get_time(), dtype=np.float64)

                window.plot_tabs.setCurrentWidget(window.lineout_plot)
                process_events()
                combo_set_data(window.map_coordinate_combo, "static_x")
                combo_set_data(window.slice_mode_combo, "time_trace")
                combo_set_data(window.line_coordinate_combo, "static_x")
                window.trace_coordinate_spin.setValue(window._display_length_value(float(static_x[499])))
                wait_until(lambda: window._slice_mode() == "time_trace", timeout_s=5.0)
                wait_until(lambda: np.asarray(window.lineout_plot.last_x_values, dtype=np.float64).size == time.size, timeout_s=5.0)
                wait_until(lambda: window.field_map_widget.current_reference_position is not None, timeout_s=5.0)

                resolved_zone = window._resolved_trace_zone_index()
                self.assertIsNotNone(resolved_zone)
                expected_trace = density[:, int(resolved_zone)]
                allclose_nan(window.lineout_plot.last_x_values, time)
                allclose_nan(window.lineout_plot.last_y_series[0], expected_trace)
                self.assertTrue(window.lineout_plot.current_cursor_visible)
                self.assertAlmostEqual(window.lineout_plot.current_cursor_position, float(time[0]))
                self.assertTrue(window.field_map_widget.current_time_marker_visible)
                self.assertTrue(window.field_map_widget.current_reference_visible)
                self.assertAlmostEqual(window.field_map_widget.current_reference_position, float(static_x[int(resolved_zone)]))
                self.assertIn("full history", window.lineout_plot.current_title.lower())
                self.assertIn("current-time cursor", window.lineout_plot.current_title.lower())
                self.assertIn("slider controls current-time cursor", window.active_analysis_label.text())

                baseline_trace = np.asarray(window.lineout_plot.last_y_series[0], dtype=np.float64).copy()
                window.snapshot_slider.setValue(70)
                process_events()
                allclose_nan(window.lineout_plot.last_y_series[0], baseline_trace)
                self.assertAlmostEqual(window.lineout_plot.current_cursor_position, float(time[70]))
                self.assertIn("current time:", window.snapshot_time_label.text().lower())

                window.plot_tabs.setCurrentWidget(window.mouse_tab)
                process_events()
                self.assertFalse(window.field_map_widget.current_time_marker_visible)
                self.assertFalse(window.field_map_widget.current_reference_visible)
                self.assertFalse(window.field_map_widget.probe_visible)

                window._on_map_probe_moved(float(time[1]), float(static_x[4]))
                process_events()
                self.assertTrue(window.field_map_widget.probe_visible)

                window.plot_tabs.setCurrentWidget(window.diagnostic_plot)
                process_events()
                self.assertFalse(window.field_map_widget.probe_visible)
                self.assertFalse(window.field_map_widget.current_time_marker_visible)
                self.assertFalse(window.field_map_widget.current_reference_visible)

                window.plot_tabs.setCurrentWidget(window.lineout_plot)
                process_events()
                self.assertTrue(window.field_map_widget.current_time_marker_visible)
                self.assertTrue(window.field_map_widget.current_reference_visible)
                self.assertAlmostEqual(window.lineout_plot.current_cursor_position, float(time[70]))
        finally:
            window.close()

    def test_display_unit_conversions_keep_map_slice_and_labels_consistent(self) -> None:
        window = self.open_window("5Fe+4.9TW+light_stabilized.h5")
        try:
            with HeliosRun(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5") as run:
                apply_display_units(
                    window,
                    time_unit="ns",
                    length_unit="um",
                    pressure_unit="GPa",
                    density_unit="kg/m3",
                    temperature_unit="K",
                )

                density = np.asarray(run.get_field("density"), dtype=np.float64) * 1.0e3
                assert_field_map_payload(window, density)
                self.assertEqual(window.field_map_widget.current_colorbar_label, "Density [kg/m3]")
                self.assertEqual(window.lineout_plot.current_y_label, "Density [kg/m3]")
                self.assertEqual(window.field_map_widget.current_x_label, "Time [ns]")
                self.assertIn(window.field_map_widget.current_y_label, {"Static x [um]", "Moving-mesh x [um]", "Moving-mesh radius [um]", "Radius [um]"})

                combo_set_data(window.slice_mode_combo, "time_trace")
                combo_set_data(window.line_coordinate_combo, "static_x")
                window.trace_coordinate_spin.setValue(window._display_length_value(float(run.get_grid("x")[4])))
                wait_until(lambda: window.lineout_plot.current_x_label == "Time [ns]", timeout_s=5.0)
                self.assertIn("static x =", window.trace_reference_label.text().lower())
                self.assertIn("um", window.trace_reference_label.text())
                self.assertEqual(window.lineout_plot.current_x_label, "Time [ns]")
                self.assertEqual(window.lineout_plot.current_y_label, "Density [kg/m3]")

                select_field(window, "pressure")
                expected_pressure = np.asarray(run.get_field("pressure"), dtype=np.float64) * 1.0e-3
                assert_field_map_payload(window, expected_pressure)
                self.assertEqual(window.field_map_widget.current_colorbar_label, "Total pressure [GPa]")
                self.assertEqual(window.lineout_plot.current_y_label, "Total pressure [GPa]")

                select_field(window, "temperature_e")
                expected_temperature = np.asarray(run.get_field("temperature_e"), dtype=np.float64) * EV_TO_K
                assert_field_map_payload(window, expected_temperature)
                self.assertEqual(window.field_map_widget.current_colorbar_label, "Electron temperature [K]")
                self.assertEqual(window.lineout_plot.current_y_label, "Electron temperature [K]")
        finally:
            window.close()

    def test_toolbar_dark_theme_and_disabled_transform_menus_are_consistent(self) -> None:
        window = self.open_window("Cu_0166_stabilized.h5")
        try:
            self.assertFalse(window.field_map_widget._plot.menuEnabled())
            self.assertFalse(window.lineout_plot._plot.menuEnabled())

            window._set_plot_navigation_mode("zoom")
            self.assertEqual(window.field_map_widget.current_navigation_mode, "zoom")
            self.assertEqual(window.lineout_plot.current_navigation_mode, "zoom")

            window.plot_tabs.setCurrentWidget(window.mouse_tab)
            process_events()
            self.assertEqual(window.mouse_vertical_plot.current_navigation_mode, "zoom")
            self.assertEqual(window.mouse_horizontal_plot.current_navigation_mode, "zoom")

            window._reset_plot_views()
            window._set_theme_mode("dark")
            process_events()
            self.assertEqual(window._theme.name, "dark")
            map_background = window.field_map_widget._graphics.backgroundBrush().color().name().lower()
            line_background = window.lineout_plot._plot.backgroundBrush().color().name().lower()
            self.assertEqual(map_background, window._theme.plot_background.lower())
            self.assertEqual(line_background, window._theme.plot_background.lower())
            self.assertTrue(window.plot_toolbar.isEnabled())
            self.assertTrue(window.plot_zoom_action.isEnabled())
            self.assertTrue(window.plot_home_action.isEnabled())
        finally:
            window.close()

    def test_laser_region_text_and_unit_settings_persist(self) -> None:
        reset_test_settings()
        window = self.open_window("10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
        try:
            region_texts = [window.region_list.item(index).text() for index in range(window.region_list.count())]
            self.assertTrue(any("laser-entry region" in text for text in region_texts))

            apply_display_units(
                window,
                time_unit="ps",
                length_unit="mm",
                pressure_unit="Mbar",
                density_unit="kg/m3",
                temperature_unit="K",
            )
            window._set_theme_mode("dark")
            window._save_current_preferences()
        finally:
            window.close()

        reopened = HeliosViewerMainWindow()
        try:
            self.assertEqual(reopened._viewer_settings.time_unit, "ps")
            self.assertEqual(reopened._viewer_settings.length_unit, "mm")
            self.assertEqual(reopened._viewer_settings.pressure_unit, "Mbar")
            self.assertEqual(reopened._viewer_settings.density_unit, "kg/m3")
            self.assertEqual(reopened._viewer_settings.temperature_unit, "K")
            self.assertEqual(reopened._theme_mode, "dark")
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
