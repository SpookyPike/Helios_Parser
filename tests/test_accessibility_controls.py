from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import get_app, process_events, reset_test_settings
from PySide6 import QtGui, QtWidgets

from helios_analysis.workspace import HeliosDerivedWorkspace
from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state
from helios_viewer.main_window import HeliosViewerMainWindow


class AccessibilityControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def assert_accessible_name(self, widget: QtWidgets.QWidget, expected: str) -> None:
        self.assertEqual(widget.accessibleName(), expected)
        interface = QtGui.QAccessible.queryAccessibleInterface(widget)
        self.assertIsNotNone(interface)
        self.assertEqual(interface.text(QtGui.QAccessible.Text.Name), expected)

    def test_snapshot_controls_have_accessible_names_and_descriptions(self) -> None:
        reset_test_settings()
        reset_session_state()
        window = HeliosParseViewMainWindow()
        try:
            controls = {
                window.global_snapshot_slider: "Active snapshot",
                window.global_snapshot_spin: "Snapshot index",
                window.global_time_spin: "Snapshot time",
                window.jump_breakout_button: "Jump to breakout snapshot",
            }
            for control, expected_name in controls.items():
                with self.subTest(control=expected_name):
                    self.assertEqual(control.accessibleName(), expected_name)
                    interface = QtGui.QAccessible.queryAccessibleInterface(control)
                    self.assertIsNotNone(interface)
                    self.assertEqual(
                        interface.text(QtGui.QAccessible.Text.Name),
                        expected_name,
                    )
            self.assertTrue(window.global_snapshot_slider.accessibleDescription())
            self.assertTrue(window.global_snapshot_spin.accessibleDescription())
            self.assertTrue(window.global_time_spin.accessibleDescription())
        finally:
            window.close()
            process_events()

    def test_density_and_zone_controls_have_named_buddies(self) -> None:
        workspace = HeliosDerivedWorkspace()
        try:
            self.assertEqual(
                workspace.min_density_spin.accessibleName(), "Minimum density"
            )
            self.assertEqual(workspace.zone_lower_spin.accessibleName(), "First zone")
            self.assertEqual(workspace.zone_upper_spin.accessibleName(), "Last zone")
            labels = workspace.findChildren(QtWidgets.QLabel)
            self.assertTrue(
                any(label.buddy() is workspace.min_density_spin for label in labels)
            )
            self.assertTrue(
                any(label.buddy() is workspace.zone_lower_spin for label in labels)
            )
            self.assertTrue(
                any(label.buddy() is workspace.zone_upper_spin for label in labels)
            )
        finally:
            workspace.close()
            process_events()

    def test_viewer_probe_and_snapshot_controls_are_named(self) -> None:
        viewer = HeliosViewerMainWindow()
        try:
            self.assertEqual(viewer.snapshot_slider.accessibleName(), "Snapshot")
            self.assertEqual(viewer.snapshot_spin.accessibleName(), "Snapshot index")
            self.assertEqual(viewer.mouse_time_slider.accessibleName(), "Probe time")
            self.assertEqual(viewer.mouse_time_spin.accessibleName(), "Probe time")
            self.assertEqual(
                viewer.mouse_coordinate_slider.accessibleName(), "Probe coordinate"
            )
            self.assertEqual(
                viewer.mouse_coordinate_spin.accessibleName(), "Probe coordinate"
            )
        finally:
            viewer.close()
            process_events()

    def test_remaining_editors_and_plot_selectors_have_accessible_names(self) -> None:
        reset_test_settings()
        reset_session_state()
        window = HeliosParseViewMainWindow()
        try:
            viewer = window.viewer_controller.window
            workspace = window.derived_controller.widget()
            named_controls = (
                (window.output_path_edit, "Output HDF5 path"),
                (viewer.trace_zone_spin, "Reference zone index"),
                (viewer.trace_coordinate_spin, "Reference coordinate"),
                (viewer.percentile_low_spin, "Lower percentile clip threshold"),
                (viewer.percentile_high_spin, "Upper percentile clip threshold"),
                (viewer.level_min_edit, "Manual color range minimum"),
                (viewer.level_max_edit, "Manual color range maximum"),
                (workspace.preheat_snapshot_slider, "Preheat snapshot"),
                (workspace.preheat_snapshot_spin, "Preheat snapshot index"),
                (workspace.preheat_time_spin, "Preheat snapshot time"),
                (workspace.preheat_plot_panel.time_combo, "Time traces"),
                (workspace.preheat_plot_panel.profile_combo, "Snapshot profiles"),
            )
            for control, expected in named_controls:
                with self.subTest(control=expected):
                    self.assert_accessible_name(control, expected)
        finally:
            window.close()
            process_events()


if __name__ == "__main__":
    unittest.main()
