from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import HDF5_ROOT, ROOT, get_app, process_events, reset_test_settings, wait_until

from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state


LOG_ROOT = ROOT


class AppPhase3Tests(unittest.TestCase):
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

    def test_log_open_shows_fast_preview(self) -> None:
        window = self.open_window()
        try:
            window._open_path(LOG_ROOT / "5Fe+4.9TW+light.log")
            wait_until(lambda: window.current_preview is not None, timeout_s=30.0)
            preview = window.current_preview
            assert preview is not None
            self.assertEqual(window.mode_stack.currentWidget(), window.parser_page)
            self.assertEqual(preview.n_zones, 500)
            self.assertEqual(preview.n_snapshots, 8)
            self.assertIn("Detected fields", window.preview_summary.toPlainText())
            self.assertGreater(window.preview_field_list.count(), 0)
            self.assertTrue(window.parse_action.isEnabled())
        finally:
            window.close()

    def test_preview_parse_writes_hdf5_and_auto_opens_viewer(self) -> None:
        window = self.open_window()
        temp_dir = Path(tempfile.mkdtemp())
        try:
            window._open_path(LOG_ROOT / "5Fe+4.9TW+light.log")
            wait_until(lambda: window.current_preview is not None, timeout_s=30.0)
            output_path = temp_dir / "5Fe_phase3_test.h5"
            window.output_path_edit.setText(str(output_path))
            window.auto_open_checkbox.setChecked(True)
            window.overwrite_checkbox.setChecked(True)
            window._start_parse_from_controls()
            wait_until(lambda: output_path.exists() and window.last_parse_result is not None, timeout_s=60.0)
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                timeout_s=60.0,
            )
            self.assertEqual(window.mode_stack.currentWidget(), window.viewer_page)
            viewer = window.viewer_controller.window
            assert viewer.run_payload is not None
            self.assertEqual(viewer.run_payload.path, output_path)
            self.assertEqual(int(viewer.run_payload.summary["n_zones"]), 500)
            self.assertTrue(window.open_viewer_action.isEnabled())
        finally:
            window.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_open_existing_hdf5_loads_viewer_and_mode_switching_preserves_viewer_state(self) -> None:
        window = self.open_window()
        try:
            path = HDF5_ROOT / "Cu_0166_stabilized.h5"
            window._open_path(path)
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                timeout_s=30.0,
            )
            viewer = window.viewer_controller.window
            self.assertEqual(window.mode_stack.currentWidget(), window.viewer_page)
            self.assertEqual(viewer.field_list.count(), 27)
            self.assertEqual(viewer.plot_tabs.count(), 3)
            window._set_mode("parser")
            process_events()
            self.assertEqual(window.mode_stack.currentWidget(), window.parser_page)
            window._set_mode("viewer")
            process_events()
            self.assertEqual(window.mode_stack.currentWidget(), window.viewer_page)
            self.assertIsNotNone(viewer.run_payload)
            self.assertEqual(viewer.run_payload.path, path)
        finally:
            window.close()

    def test_recent_files_and_heavy_hdf5_open_work(self) -> None:
        window = self.open_window()
        try:
            small_log = LOG_ROOT / "5Fe+4.9TW+light.log"
            heavy_hdf5 = HDF5_ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"
            window._open_path(small_log)
            wait_until(lambda: window.current_preview is not None, timeout_s=30.0)
            window._open_path(heavy_hdf5)
            wait_until(lambda: window.viewer_controller.window.run_payload is not None, timeout_s=30.0)
            viewer = window.viewer_controller.window
            assert viewer.run_payload is not None
            self.assertEqual(int(viewer.run_payload.summary["n_zones"]), 1300)
            self.assertGreaterEqual(len(window.session_state.recent_files or []), 2)
            recent = set(window.session_state.recent_files or [])
            self.assertIn(str(small_log), recent)
            self.assertIn(str(heavy_hdf5), recent)
            self.assertGreater(window.recent_files_menu.actions().__len__(), 0)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
