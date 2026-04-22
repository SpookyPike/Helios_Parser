from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import HDF5_ROOT, ROOT, get_app, process_events, reset_test_settings, wait_until

from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state


class AppPhase33Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def open_window(self) -> HeliosParseViewMainWindow:
        reset_test_settings()
        reset_session_state()
        window = HeliosParseViewMainWindow()
        window.show()
        process_events()
        return window

    def test_mode_registry_keeps_shell_ready_for_future_extension(self) -> None:
        window = self.open_window()
        try:
            self.assertEqual(window.available_mode_ids(), ("parser", "viewer", "derived"))
            self.assertEqual(window._current_mode_id(), "parser")
            self.assertFalse(window.export_action.isEnabled())

            window._open_path(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )
            self.assertEqual(window._current_mode_id(), "viewer")
            self.assertTrue(window.export_action.isEnabled())

            window._set_mode("parser")
            process_events()
            self.assertEqual(window._current_mode_id(), "parser")
            self.assertFalse(window.export_action.isEnabled())
        finally:
            window.close()

    def test_open_log_then_open_hdf5_keeps_mode_routing_stable(self) -> None:
        window = self.open_window()
        try:
            window._open_path(ROOT / "5Fe+4.9TW+light.log")
            wait_until(lambda: window.current_preview is not None, 30.0)
            self.assertEqual(window._current_mode_id(), "parser")
            self.assertTrue(window.parse_action.isEnabled())

            window._open_path(HDF5_ROOT / "Cu_0166_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )
            self.assertEqual(window._current_mode_id(), "viewer")
            self.assertEqual(window.viewer_controller.window.plot_tabs.count(), 3)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
