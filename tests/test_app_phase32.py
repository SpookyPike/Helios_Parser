from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _test_bootstrap  # noqa: F401
from PySide6 import QtWidgets

from _viewer_test_utils import HDF5_ROOT, ROOT, get_app, process_events, reset_test_settings, wait_until

from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state
from helios_viewer.workspace import HeliosViewerWorkspace


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


class AppPhase32Tests(unittest.TestCase):
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

    def test_parser_progress_updates_ui_during_parse(self) -> None:
        window = self.open_window()
        progress_events = []
        temp_dir = Path(tempfile.mkdtemp())
        try:
            window.parser_controller.progress_changed.connect(progress_events.append)
            window._open_path(ROOT / "5Fe+4.9TW+light.log")
            wait_until(lambda: window.current_preview is not None, 30.0)
            output_path = temp_dir / "phase32_progress.h5"
            window.output_path_edit.setText(str(output_path))
            window.auto_open_checkbox.setChecked(False)
            window.overwrite_checkbox.setChecked(True)
            window._start_parse_from_controls()
            wait_until(lambda: window.last_parse_result is not None and output_path.exists(), 60.0)
            process_events(50)

            self.assertGreaterEqual(len(progress_events), 3)
            self.assertIn("snapshots", {payload.stage for payload in progress_events})
            self.assertEqual(progress_events[-1].stage, "done")
            self.assertAlmostEqual(progress_events[-1].fraction, 1.0)
            self.assertEqual(window.parse_progress.value(), 1000)
            self.assertIn("100%", window.parse_progress.format())
        finally:
            window.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_shell_embeds_workspace_widget_not_nested_mainwindow(self) -> None:
        window = self.open_window()
        try:
            self.assertIsInstance(window.viewer_page, HeliosViewerWorkspace)
            self.assertNotIsInstance(window.viewer_page, QtWidgets.QMainWindow)
            window._open_path(HDF5_ROOT / "Cu_0166_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )
            self.assertEqual(window.viewer_controller.window.plot_tabs.count(), 3)
        finally:
            window.close()

    def test_vector_svg_and_pdf_export_for_slice_plot(self) -> None:
        window = self.open_window()
        temp_dir = Path(tempfile.mkdtemp())
        try:
            window._open_path(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )
            svg_path = temp_dir / "slice_plot.svg"
            pdf_path = temp_dir / "slice_plot.pdf"

            with mock.patch("helios_viewer.main_window.ExportDialog", _StubExportDialog):
                _StubExportDialog.next_options = {
                    "target": "active_tab",
                    "format": "svg",
                    "transparent": False,
                    "path": svg_path,
                }
                window._export_current_view()
                _StubExportDialog.next_options = {
                    "target": "active_tab",
                    "format": "pdf",
                    "transparent": False,
                    "path": pdf_path,
                }
                window._export_current_view()

            self.assertTrue(svg_path.exists())
            self.assertTrue(pdf_path.exists())

            svg_text = svg_path.read_text(encoding="utf-8", errors="replace").lower()
            self.assertIn("<svg", svg_text)
            self.assertNotIn("<image", svg_text)
            self.assertTrue(any(tag in svg_text for tag in ("<path", "<polyline", "<text")))

            pdf_bytes = pdf_path.read_bytes()
            self.assertGreater(len(pdf_bytes), 0)
            self.assertNotIn(b"/Subtype /Image", pdf_bytes)
            self.assertIn(b"/Font", pdf_bytes)
        finally:
            window.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
