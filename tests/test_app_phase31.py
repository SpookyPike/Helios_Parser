from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _test_bootstrap  # noqa: F401
from PySide6 import QtCore, QtWidgets

from _viewer_test_utils import HDF5_ROOT, ROOT, get_app, process_events, reset_test_settings, wait_until

from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import load_session_state, reset_session_state


class _StubExportDialog:
    next_options: dict[str, object] | None = None

    def __init__(
        self,
        suggested_name: str,
        target_sizes: dict[str, object] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        self._options = dict(type(self).next_options or {})

    def exec(self) -> int:
        return QtWidgets.QDialog.Accepted

    def export_options(self) -> dict[str, object]:
        return dict(self._options)


class AppPhase31Tests(unittest.TestCase):
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

    def test_shell_action_state_and_file_routing(self) -> None:
        window = self.open_window()
        try:
            self.assertFalse(window.export_action.isEnabled())
            self.assertFalse(window.open_viewer_action.isEnabled())
            self.assertFalse(window.parse_action.isEnabled())

            window._open_path(ROOT / "5Fe+4.9TW+light.log")
            wait_until(lambda: window.current_preview is not None, 30.0)
            self.assertEqual(window.mode_stack.currentWidget(), window.parser_page)
            self.assertTrue(window.parse_action.isEnabled())
            self.assertFalse(window.export_action.isEnabled())

            path = HDF5_ROOT / "Cu_0166_stabilized.h5"
            window._open_path(path)
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )
            self.assertEqual(window.mode_stack.currentWidget(), window.viewer_page)
            self.assertTrue(window.export_action.isEnabled())
            self.assertEqual(window.viewer_controller.window.plot_tabs.count(), 3)
        finally:
            window.close()

    def test_shell_and_viewer_settings_persist_across_restart(self) -> None:
        window = self.open_window()
        try:
            window.compression_combo.setCurrentIndex(window.compression_combo.findData("lzf"))
            window.overwrite_checkbox.setChecked(False)
            window.auto_open_checkbox.setChecked(False)
            window._set_theme_mode("dark")
            process_events()
            window.close()

            reopened = HeliosParseViewMainWindow()
            reopened.show()
            process_events()
            try:
                self.assertEqual(reopened.compression_combo.currentData(), "lzf")
                self.assertFalse(reopened.overwrite_checkbox.isChecked())
                self.assertFalse(reopened.auto_open_checkbox.isChecked())
                self.assertEqual(reopened.viewer_controller.theme_mode(), "dark")
                self.assertTrue(reopened.theme_actions["dark"].isChecked())
                state = load_session_state()
                self.assertEqual(state.parse_compression, "lzf")
                self.assertFalse(state.parse_overwrite)
                self.assertFalse(state.auto_open_after_parse)
            finally:
                reopened.close()
        finally:
            if window.isVisible():
                window.close()

    def test_integrated_export_writes_png_pdf_and_transparent_png(self) -> None:
        window = self.open_window()
        temp_dir = Path(tempfile.mkdtemp())
        try:
            window._open_path(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )

            png_path = temp_dir / "shell_export.png"
            transparent_path = temp_dir / "shell_export_transparent.png"
            pdf_path = temp_dir / "shell_export.pdf"

            with mock.patch("helios_viewer.main_window.ExportDialog", _StubExportDialog):
                _StubExportDialog.next_options = {
                    "target": "field_map",
                    "format": "png",
                    "transparent": False,
                    "path": png_path,
                }
                window._export_current_view()
                _StubExportDialog.next_options = {
                    "target": "field_map",
                    "format": "png",
                    "transparent": True,
                    "path": transparent_path,
                }
                window._export_current_view()
                _StubExportDialog.next_options = {
                    "target": "active_tab",
                    "format": "pdf",
                    "transparent": False,
                    "path": pdf_path,
                }
                window._export_current_view()

            self.assertTrue(png_path.exists())
            self.assertTrue(transparent_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(transparent_path.stat().st_size, 0)
            self.assertGreater(pdf_path.stat().st_size, 0)
        finally:
            window.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_overwrite_disabled_reports_parse_error_without_crashing(self) -> None:
        window = self.open_window()
        temp_dir = Path(tempfile.mkdtemp())
        errors: list[tuple[str, str]] = []
        try:
            window.parser_controller.error_occurred.connect(lambda message, details: errors.append((message, details)))
            window._open_path(ROOT / "5Fe+4.9TW+light.log")
            wait_until(lambda: window.current_preview is not None, 30.0)
            output_path = temp_dir / "existing_output.h5"
            output_path.write_bytes(b"already here")
            window.output_path_edit.setText(str(output_path))
            window.overwrite_checkbox.setChecked(False)

            with mock.patch.object(QtWidgets.QMessageBox, "exec", return_value=0):
                window._start_parse_from_controls()
                wait_until(lambda: not window._parser_busy, 60.0)

            self.assertIsNone(window.last_parse_result)
            self.assertTrue(errors)
            self.assertIn("Failed to parse", errors[0][0])
            self.assertIn("already exists", errors[0][1])
        finally:
            window.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
