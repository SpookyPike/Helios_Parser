from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import HDF5_ROOT, get_app, process_events, reset_test_settings, set_checked_values, wait_until

from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state


class AppPhase4Tests(unittest.TestCase):
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

    def test_derived_mode_registers_and_receives_current_run_context(self) -> None:
        window = self.open_window()
        try:
            self.assertEqual(window.available_mode_ids(), ("parser", "viewer", "derived"))
            window._open_path(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )
            window._set_mode("derived")
            wait_until(lambda: window.derived_controller.widget()._current_result is not None, 30.0)
            process_events(50)
            result = window.derived_controller.widget()._current_result
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.dataset_path.name, "5Fe+4.9TW+light_stabilized.h5")
            self.assertEqual(window._current_mode_id(), "derived")
            self.assertIn("500 zones", window.derived_controller.widget().run_summary_label.text())
        finally:
            window.close()

    def test_derived_mode_follows_snapshot_and_subset_changes(self) -> None:
        window = self.open_window()
        try:
            window._open_path(HDF5_ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                60.0,
            )
            window._set_mode("derived")
            wait_until(lambda: window.derived_controller.widget()._current_result is not None, 60.0)

            viewer = window.viewer_controller.window
            window.global_snapshot_spin.setValue(5)
            process_events(50)
            wait_until(lambda: window.derived_controller.widget()._current_result.snapshot_index == 5, 30.0)

            set_checked_values(viewer.region_list, {1, 3})
            viewer._update_filter_summary()
            viewer._refresh_visuals()
            process_events(50)
            wait_until(lambda: window.derived_controller.widget()._current_result.selected_zone_count == 300, 60.0)

            result = window.derived_controller.widget()._current_result
            self.assertEqual(result.snapshot_index, 5)
            self.assertEqual(result.selected_zone_count, 300)
            self.assertGreaterEqual(len(result.shock.time_s), 1)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
