from __future__ import annotations

import unittest

import numpy as np

from _viewer_test_utils import EXPECTED, HDF5_ROOT, combo_set_data, find_row_by_data, get_app, process_events, reset_test_settings, set_checked_values, wait_until

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosRun
from helios_viewer.main_window import HeliosViewerMainWindow
from helios_viewer.models import OpenRunPayload
from helios_viewer.plots import signed_log10_transform


def build_open_run_payload(path) -> OpenRunPayload:
    with HeliosRun(path) as run:
        fields = run.list_fields()
        diagnostics = run.list_diagnostics()
        return OpenRunPayload(
            run_generation=0,
            path=path,
            summary=run.summary(),
            metadata=run.get_metadata(),
            fields=fields,
            field_units={name: run.get_field_unit(name) for name in fields},
            diagnostics=diagnostics,
            diagnostic_units={name: run.get_diagnostic_unit(name) for name in diagnostics},
            regions=run.get_regions(),
            materials=run.get_materials(),
            time=np.asarray(run.get_time(), dtype=np.float64),
            time_unit=run.get_time_unit(),
            static_x=np.asarray(run.get_static_coordinate(location="center"), dtype=np.float64),
            static_x_edges=np.asarray(run.get_static_coordinate(location="edge"), dtype=np.float64),
            static_x_unit=run.get_grid_unit("x"),
            zone_region_id=np.asarray(run.get_grid("zone_region_id"), dtype=np.int32),
            zone_material_index=np.asarray(run.get_grid("zone_material_index"), dtype=np.int32),
            has_dynamic_radius="radius" in fields,
            radius_unit=run.get_field_unit("radius") if "radius" in fields else "",
        )


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


def expected_zone_boundaries(max_zones: np.ndarray, active_mask: np.ndarray) -> list[float]:
    positions: list[float] = []
    for max_zone in np.asarray(max_zones, dtype=np.int32)[:-1]:
        left = int(max_zone) - 1
        right = int(max_zone)
        if active_mask[left] or active_mask[right]:
            positions.append(float(max_zone) + 0.5)
    return positions


def expected_coordinate_boundaries(edges: np.ndarray, max_zones: np.ndarray, active_mask: np.ndarray) -> list[float]:
    positions: list[float] = []
    coordinate_edges = np.asarray(edges, dtype=np.float64)
    for max_zone in np.asarray(max_zones, dtype=np.int32)[:-1]:
        left = int(max_zone) - 1
        right = int(max_zone)
        if active_mask[left] or active_mask[right]:
            positions.append(float(coordinate_edges[int(max_zone)]))
    return positions


def select_field(window: HeliosViewerMainWindow, field_name: str) -> None:
    row = find_row_by_data(window.field_list, field_name)
    if row < 0:
        raise AssertionError(f"Field {field_name!r} not found.")
    window.field_list.setCurrentRow(row)
    wait_until(
        lambda: window.current_field_payload is not None and window.current_field_payload.field_name == field_name,
        timeout_s=30.0,
    )


def select_diagnostic(window: HeliosViewerMainWindow, path: str) -> None:
    row = find_row_by_data(window.diagnostic_list, path)
    if row < 0:
        raise AssertionError(f"Diagnostic {path!r} not found.")
    window.diagnostic_list.setCurrentRow(row)
    wait_until(
        lambda: window.current_diagnostic_payload is not None and window.current_diagnostic_payload.path == path,
        timeout_s=30.0,
    )


class ViewerPayloadTests(unittest.TestCase):
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

    def test_dynamic_radius_mode_stays_disabled_until_radius_payload_is_loaded(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        payload = build_open_run_payload(path)
        window = HeliosViewerMainWindow()
        try:
            window._on_run_opened(payload)
            radius_index = window.line_coordinate_combo.findData("radius")
            self.assertGreaterEqual(radius_index, 0)
            self.assertEqual(window.line_coordinate_combo.currentData(), "zone")
            model = window.line_coordinate_combo.model()
            item = model.item(radius_index) if hasattr(model, "item") else None
            self.assertIsNotNone(item)
            self.assertFalse(bool(item.isEnabled()))
        finally:
            window.close()

    def test_2d_orientation_and_coordinate_payloads_match_reader(self) -> None:
        for name in EXPECTED:
            with self.subTest(example=name):
                window = self.open_window(name)
                try:
                    with HeliosRun(HDF5_ROOT / name) as run:
                        if run.has_dynamic_coordinate():
                            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
                            combo_set_data(window.map_coordinate_combo, "static_x")
                            wait_until(
                                lambda: window._map_coordinate_mode() == "static_x"
                                and window.field_map_widget.current_render_mode == "image"
                                and window.field_map_widget.last_display_image is not None,
                                timeout_s=30.0,
                            )
                        field = np.asarray(run.get_field("density"), dtype=np.float64)
                        time = np.asarray(run.get_time(), dtype=np.float64)
                        static_x = np.asarray(run.get_grid("x"), dtype=np.float64)
                        static_x_edges = np.asarray(run.get_static_coordinate(location="edge"), dtype=np.float64)
                        max_zones = np.asarray(run.get_regions()["max_zone_index"], dtype=np.int32)
                        active = np.ones(run.n_zones, dtype=bool)

                        allclose_nan(window.field_map_widget.last_display_image, field.T)
                        allclose_nan(window.field_map_widget.last_coordinate_values, static_x)
                        allclose_nan(window.field_map_widget.last_time_values, time)
                        self.assertEqual(window.field_map_widget.current_orientation, "time_x_coord_y")
                        self.assertEqual(window.field_map_widget.current_boundary_angle, 0.0 if len(max_zones) > 1 else None)
                        self.assertEqual(window.field_map_widget.current_time_marker_angle, 90.0)
                        self.assertEqual(
                            list(window.field_map_widget.current_boundary_positions),
                            expected_coordinate_boundaries(static_x_edges, max_zones, active),
                        )

                        combo_set_data(window.map_orientation_combo, "coord_x_time_y")
                        wait_until(
                            lambda: window.field_map_widget.current_orientation == "coord_x_time_y"
                            and window.field_map_widget.last_display_image is not None
                            and window.field_map_widget.last_display_image.shape == field.shape,
                            timeout_s=30.0,
                        )
                        allclose_nan(window.field_map_widget.last_display_image, field)
                        self.assertEqual(window.field_map_widget.current_orientation, "coord_x_time_y")
                        self.assertEqual(window.field_map_widget.current_boundary_angle, 90.0 if len(max_zones) > 1 else None)
                        self.assertEqual(window.field_map_widget.current_time_marker_angle, 0.0)

                        expected_zone = np.arange(1, run.n_zones + 1, dtype=np.float64)
                        combo_set_data(window.map_coordinate_combo, "zone")
                        wait_until(
                            lambda: window.field_map_widget.last_coordinate_values is not None
                            and window.field_map_widget.last_coordinate_values.shape == expected_zone.shape,
                            timeout_s=30.0,
                        )
                        allclose_nan(window.field_map_widget.last_display_image, field)
                        allclose_nan(window.field_map_widget.last_coordinate_values, expected_zone)
                        self.assertIn("Zone index", window.field_map_widget.current_x_label)
                        self.assertEqual(
                            list(window.field_map_widget.current_boundary_positions),
                            expected_zone_boundaries(max_zones, active),
                        )
                finally:
                    window.close()

    def test_snapshot_lineouts_and_time_traces_match_reader(self) -> None:
        for name in EXPECTED:
            with self.subTest(example=name):
                window = self.open_window(name)
                try:
                    with HeliosRun(HDF5_ROOT / name) as run:
                        field = np.asarray(run.get_field("density"), dtype=np.float64)
                        static_x = np.asarray(run.get_grid("x"), dtype=np.float64)
                        time = np.asarray(run.get_time(), dtype=np.float64)
                        snapshot_index = min(1, run.n_snapshots - 1)
                        window.snapshot_slider.setValue(snapshot_index)
                        process_events()
                        if run.has_dynamic_coordinate():
                            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)

                        combo_set_data(window.slice_mode_combo, "snapshot_lineout")
                        combo_set_data(window.line_coordinate_combo, "static_x")
                        wait_until(
                            lambda: np.asarray(window.lineout_plot.last_x_values, dtype=np.float64).shape == static_x.shape,
                            timeout_s=30.0,
                        )
                        allclose_nan(window.lineout_plot.last_x_values, static_x)
                        allclose_nan(window.lineout_plot.last_y_series[0], field[snapshot_index])

                        combo_set_data(window.line_coordinate_combo, "zone")
                        expected_zone = np.arange(1, run.n_zones + 1, dtype=np.float64)
                        wait_until(
                            lambda: np.asarray(window.lineout_plot.last_x_values, dtype=np.float64).shape == expected_zone.shape
                            and np.isclose(float(np.asarray(window.lineout_plot.last_x_values, dtype=np.float64)[0]), 1.0),
                            timeout_s=30.0,
                        )
                        allclose_nan(window.lineout_plot.last_x_values, expected_zone)
                        allclose_nan(window.lineout_plot.last_y_series[0], field[snapshot_index])

                        if "radius" in run.list_fields():
                            wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
                            combo_set_data(window.line_coordinate_combo, "radius")
                            expected_radius = np.asarray(run.get_radius(snapshot_index=snapshot_index), dtype=np.float64)
                            wait_until(
                                lambda: np.asarray(window.lineout_plot.last_x_values, dtype=np.float64).shape == expected_radius.shape,
                                timeout_s=30.0,
                            )
                            allclose_nan(window.lineout_plot.last_x_values, expected_radius)
                            allclose_nan(window.lineout_plot.last_y_series[0], field[snapshot_index])

                        combo_set_data(window.map_coordinate_combo, "static_x")
                        combo_set_data(window.slice_mode_combo, "time_trace")
                        trace_zone = min(5, run.n_zones)
                        combo_set_data(window.line_coordinate_combo, "static_x")
                        window.trace_coordinate_spin.setValue(window._display_length_value(float(static_x[trace_zone - 1])))
                        wait_until(
                            lambda: np.asarray(window.lineout_plot.last_x_values, dtype=np.float64).shape == time.shape,
                            timeout_s=30.0,
                        )
                        resolved_zone = window._resolved_trace_zone_index()
                        self.assertIsNotNone(resolved_zone)
                        allclose_nan(window.lineout_plot.last_x_values, time)
                        allclose_nan(window.lineout_plot.last_y_series[0], field[:, int(resolved_zone)])
                        self.assertEqual(window.field_map_widget.current_reference_position, float(static_x[int(resolved_zone)]))
                finally:
                    window.close()

    def test_region_and_material_masks_apply_only_along_coordinate_dimension(self) -> None:
        name = "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        window = self.open_window(name)
        try:
            with HeliosRun(HDF5_ROOT / name) as run:
                wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)
                combo_set_data(window.map_coordinate_combo, "static_x")
                wait_until(
                    lambda: window._map_coordinate_mode() == "static_x"
                    and window.field_map_widget.current_render_mode == "image"
                    and window.field_map_widget.last_display_image is not None,
                    timeout_s=30.0,
                )
                field = np.asarray(run.get_field("density"), dtype=np.float64)
                static_x = np.asarray(run.get_grid("x"), dtype=np.float64)
                static_x_edges = np.asarray(run.get_static_coordinate(location="edge"), dtype=np.float64)
                time = np.asarray(run.get_time(), dtype=np.float64)
                max_zones = np.asarray(run.get_regions()["max_zone_index"], dtype=np.int32)

                set_checked_values(window.region_list, {1, 3})
                set_checked_values(window.material_list, {1, 2})
                window._update_filter_summary()
                window._refresh_visuals()
                wait_until(lambda: window.field_map_widget.last_display_image is not None, timeout_s=30.0)
                region_mask = np.asarray(run.get_region_mask(1) | run.get_region_mask(3), dtype=bool)
                expected_masked = np.where(region_mask[None, :], field, np.nan)
                allclose_nan(window._combined_zone_mask(), region_mask)
                allclose_nan(window.field_map_widget.last_display_image, expected_masked.T)
                allclose_nan(window.field_map_widget.last_coordinate_values, static_x)
                allclose_nan(window.field_map_widget.last_time_values, time)
                self.assertEqual(window.field_map_widget.last_display_image.shape, (run.n_zones, run.n_snapshots))
                self.assertEqual(
                    list(window.field_map_widget.current_boundary_positions),
                    expected_coordinate_boundaries(static_x_edges, max_zones, region_mask),
                )
                self.assertEqual(
                    window.field_map_widget.current_inactive_ranges,
                    ((float(static_x_edges[100]), float(static_x_edges[1100])),),
                )

                snapshot_index = min(1, run.n_snapshots - 1)
                window.snapshot_slider.setValue(snapshot_index)
                process_events()
                combo_set_data(window.slice_mode_combo, "snapshot_lineout")
                combo_set_data(window.line_coordinate_combo, "static_x")
                allclose_nan(window.lineout_plot.last_y_series[0], np.where(region_mask, field[snapshot_index], np.nan))

                set_checked_values(window.region_list, {1})
                set_checked_values(window.material_list, {1, 2})
                window._update_filter_summary()
                window._refresh_visuals()
                wait_until(lambda: window.field_map_widget.last_display_image is not None, timeout_s=30.0)
                region1_mask = np.asarray(run.get_region_mask(1), dtype=bool)
                self.assertEqual(
                    list(window.field_map_widget.current_boundary_positions),
                    expected_coordinate_boundaries(static_x_edges, max_zones, region1_mask),
                )

                set_checked_values(window.region_list, {1, 2})
                set_checked_values(window.material_list, {1})
                window._update_filter_summary()
                window._refresh_visuals()
                wait_until(lambda: window.field_map_widget.last_display_image is not None, timeout_s=30.0)
                combined_mask = np.asarray((run.get_region_mask(1) | run.get_region_mask(2)) & run.get_material_mask(1), dtype=bool)
                allclose_nan(window._combined_zone_mask(), combined_mask)
                allclose_nan(window.field_map_widget.last_display_image, np.where(combined_mask[None, :], field, np.nan).T)
                self.assertEqual(
                    list(window.field_map_widget.current_boundary_positions),
                    expected_coordinate_boundaries(static_x_edges, max_zones, combined_mask),
                )

                combo_set_data(window.slice_mode_combo, "time_trace")
                combo_set_data(window.line_coordinate_combo, "zone")
                window.trace_zone_spin.setValue(500)
                wait_until(lambda: window.lineout_plot.last_x_values is not None and window.lineout_plot.last_x_values.shape == time.shape, timeout_s=30.0)
                allclose_nan(window.lineout_plot.last_x_values, time)
                allclose_nan(window.lineout_plot.last_y_series[0], field[:, 99])
                self.assertIn("requested zone 500 -> nearest active zone 100", window.lineout_plot.current_title)
                self.assertIn("requested zone 500 -> nearest active zone 100", window.trace_reference_label.text())
        finally:
            window.close()

    def test_boundary_overlays_and_scale_modes_match_expected_payloads(self) -> None:
        name = "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
        window = self.open_window(name)
        try:
            with HeliosRun(HDF5_ROOT / name) as run:
                max_zones = np.asarray(run.get_regions()["max_zone_index"], dtype=np.int32)
                active = np.ones(run.n_zones, dtype=bool)
                wait_until(lambda: window.radius_payload is not None, timeout_s=30.0)

                combo_set_data(window.map_orientation_combo, "time_x_coord_y")
                combo_set_data(window.map_coordinate_combo, "zone")
                wait_until(lambda: window._map_coordinate_mode() == "zone", timeout_s=30.0)
                self.assertEqual(list(window.field_map_widget.current_boundary_positions), expected_zone_boundaries(max_zones, active))
                self.assertEqual(window.field_map_widget.current_boundary_angle, 0.0)

                combo_set_data(window.map_orientation_combo, "coord_x_time_y")
                wait_until(lambda: window.field_map_widget.current_orientation == "coord_x_time_y", timeout_s=30.0)
                self.assertEqual(list(window.field_map_widget.current_boundary_positions), expected_zone_boundaries(max_zones, active))
                self.assertEqual(window.field_map_widget.current_boundary_angle, 90.0)

                select_field(window, "velocity")
                snapshot_index = 70
                window.snapshot_slider.setValue(snapshot_index)
                process_events()
                combo_set_data(window.slice_mode_combo, "snapshot_lineout")
                combo_set_data(window.line_coordinate_combo, "zone")
                combo_set_data(window.line_scale_combo, "signed_log10")
                expected_velocity = np.asarray(run.get_field("velocity", time_slice=snapshot_index), dtype=np.float64)
                wait_until(lambda: window.lineout_plot.value_scale_mode == "signed_log10", timeout_s=30.0)
                allclose_nan(window.lineout_plot.last_y_series[0], signed_log10_transform(expected_velocity))
                self.assertEqual(window.lineout_plot.value_scale_mode, "signed_log10")

                select_field(window, "density")
                combo_set_data(window.line_scale_combo, "log10")
                expected_density = np.asarray(run.get_field("density", time_slice=snapshot_index), dtype=np.float64)
                wait_until(lambda: window.lineout_plot.value_scale_mode == "log10", timeout_s=30.0)
                allclose_nan(window.lineout_plot.last_y_series[0], expected_density)
                self.assertEqual(window.lineout_plot.value_scale_mode, "log10")

                select_diagnostic(window, "energy_summary/current/ions")
                combo_set_data(window.diagnostic_scale_combo, "log10")
                expected_diag = np.asarray(run.get_diagnostic("energy_summary/current/ions"), dtype=np.float64)
                wait_until(lambda: window.diagnostic_plot.value_scale_mode == "log10", timeout_s=30.0)
                allclose_nan(window.diagnostic_plot.last_y_series[0], expected_diag)
                self.assertEqual(window.diagnostic_plot.value_scale_mode, "log10")
        finally:
            window.close()

    def test_new_helios_format_field_and_diagnostic_loading_remain_consistent(self) -> None:
        name = "Cu_0166_stabilized.h5"
        window = self.open_window(name)
        try:
            with HeliosRun(HDF5_ROOT / name) as run:
                self.assertIn("radiation_sink", run.list_fields())
                select_field(window, "radiation_sink")
                expected = np.asarray(run.get_field("radiation_sink"), dtype=np.float64)
                assert_field_map_payload(window, expected)

                select_diagnostic(window, "radiation_boundary_fluxes/region_net_cooling_rate")
                expected_diag = np.asarray(run.get_diagnostic("radiation_boundary_fluxes/region_net_cooling_rate"), dtype=np.float64)
                self.assertEqual(expected_diag.shape, (run.n_snapshots, run.n_regions))
                allclose_nan(window.diagnostic_plot.last_y_series[0], expected_diag[:, 0])
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
