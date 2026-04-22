from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _test_bootstrap  # noqa: F401
import h5py
from PySide6 import QtCore, QtWidgets

from _viewer_test_utils import HDF5_ROOT, ROOT, get_app, process_events, reset_test_settings, wait_until

from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state


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


def _view_covers_bounds(field_map_widget) -> bool:
    bounds = field_map_widget._current_data_bounds()
    if bounds is None:
        return False
    x_view, y_view = field_map_widget._plot.getViewBox().viewRange()
    x_tol = max(1.0e-12, abs(bounds[1] - bounds[0]) * 0.05)
    y_tol = max(1.0e-12, abs(bounds[3] - bounds[2]) * 0.05)
    return (
        x_view[0] <= bounds[0] + x_tol
        and x_view[1] >= bounds[1] - x_tol
        and y_view[0] <= bounds[2] + y_tol
        and y_view[1] >= bounds[3] - y_tol
    )


def _has_2d_payload(field_map_widget) -> bool:
    return field_map_widget.last_display_image is not None or field_map_widget.last_mesh_z is not None


class AppPhase32RTests(unittest.TestCase):
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

    def test_first_render_after_hdf5_open_has_visible_bounds(self) -> None:
        window = self.open_window()
        try:
            window._open_path(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None
                and _has_2d_payload(window.viewer_controller.window.field_map_widget),
                30.0,
            )
            viewer = window.viewer_controller.window
            self.assertTrue(_has_2d_payload(viewer.field_map_widget))
            self.assertTrue(_view_covers_bounds(viewer.field_map_widget))
        finally:
            window.close()

    def test_opening_new_file_clears_stale_field_map_viewport(self) -> None:
        window = self.open_window()
        try:
            window._open_path(HDF5_ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )
            viewer = window.viewer_controller.window
            view_box = viewer.field_map_widget._plot.getViewBox()
            view_box.setXRange(0.0, 1.0e-12, padding=0.0)
            view_box.setYRange(0.0, 1.0e-6, padding=0.0)
            process_events(20)

            window._open_path(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and Path(window.viewer_controller.window.run_payload.path) == HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
                and window.viewer_controller.window.current_field_payload is not None
                and _has_2d_payload(window.viewer_controller.window.field_map_widget),
                30.0,
            )
            self.assertTrue(_view_covers_bounds(window.viewer_controller.window.field_map_widget))
        finally:
            window.close()

    def test_auto_open_after_parse_waits_for_ready_hdf5_and_visible_field(self) -> None:
        window = self.open_window()
        temp_dir = Path(tempfile.mkdtemp())
        try:
            output_path = temp_dir / "phase32r_auto_open.h5"
            window._open_path(ROOT / "5Fe+4.9TW+light.log")
            wait_until(lambda: window.current_preview is not None, 30.0)
            window.output_path_edit.setText(str(output_path))
            window.auto_open_checkbox.setChecked(True)
            window.overwrite_checkbox.setChecked(True)
            window._start_parse_from_controls()
            wait_until(
                lambda: output_path.exists()
                and window.viewer_controller.window.run_payload is not None
                and Path(window.viewer_controller.window.run_payload.path) == output_path
                and window.viewer_controller.window.current_field_payload is not None
                and _has_2d_payload(window.viewer_controller.window.field_map_widget),
                60.0,
            )
            with h5py.File(output_path, "r") as handle:
                self.assertIn("fields", handle)
                self.assertTrue(handle["fields"].keys())
            viewer = window.viewer_controller.window
            self.assertEqual(viewer.current_field_payload.field_name, "density")
            self.assertTrue(_view_covers_bounds(viewer.field_map_widget))
        finally:
            window.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_field_map_pdf_export_is_disabled_in_hotfix(self) -> None:
        window = self.open_window()
        temp_dir = Path(tempfile.mkdtemp())
        try:
            window._open_path(HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5")
            wait_until(
                lambda: window.viewer_controller.window.run_payload is not None
                and window.viewer_controller.window.current_field_payload is not None,
                30.0,
            )
            pdf_path = temp_dir / "field_map.pdf"
            with mock.patch("helios_viewer.main_window.ExportDialog", _StubExportDialog):
                _StubExportDialog.next_options = {
                    "target": "field_map",
                    "format": "pdf",
                    "transparent": False,
                    "path": pdf_path,
                }
                with mock.patch.object(QtWidgets.QMessageBox, "information", return_value=QtWidgets.QMessageBox.Ok) as information_mock:
                    window._export_current_view()
            self.assertFalse(pdf_path.exists())
            self.assertTrue(information_mock.called)
        finally:
            window.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
