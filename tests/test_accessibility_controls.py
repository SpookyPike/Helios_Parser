from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
from PySide6 import QtWidgets

from _viewer_test_utils import get_app, process_events, reset_test_settings
from helios_analysis.workspace import HeliosDerivedWorkspace
from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state


class AccessibilityControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

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
            self.assertTrue(window.global_snapshot_slider.accessibleDescription())
            self.assertTrue(window.global_snapshot_spin.accessibleDescription())
            self.assertTrue(window.global_time_spin.accessibleDescription())
        finally:
            window.close()
            process_events()

    def test_density_and_zone_controls_have_named_buddies(self) -> None:
        workspace = HeliosDerivedWorkspace()
        try:
            self.assertEqual(workspace.min_density_spin.accessibleName(), "Minimum density")
            self.assertEqual(workspace.zone_lower_spin.accessibleName(), "First zone")
            self.assertEqual(workspace.zone_upper_spin.accessibleName(), "Last zone")
            labels = workspace.findChildren(QtWidgets.QLabel)
            self.assertTrue(any(label.buddy() is workspace.min_density_spin for label in labels))
            self.assertTrue(any(label.buddy() is workspace.zone_lower_spin for label in labels))
            self.assertTrue(any(label.buddy() is workspace.zone_upper_spin for label in labels))
        finally:
            workspace.close()
            process_events()


if __name__ == "__main__":
    unittest.main()
