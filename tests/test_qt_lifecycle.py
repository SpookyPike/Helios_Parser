from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import HDF5_ROOT, ROOT, get_app
from PySide6 import QtCore

from helios_analysis.workspace import HeliosDerivedWorkspace
from helios_app.main_app import HeliosParseViewMainWindow
from helios_viewer.main_window import HeliosViewerMainWindow


class _ShortTask(QtCore.QRunnable):
    def __init__(self, finished: list[bool]) -> None:
        super().__init__()
        self.finished = finished

    @QtCore.Slot()
    def run(self) -> None:
        time.sleep(0.05)
        self.finished.append(True)


class QtLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_workspace_close_stops_timers_and_drains_active_preset_work(self) -> None:
        workspace = HeliosDerivedWorkspace()
        self.assertTrue(workspace._parameter_change_timer.isActive())
        finished: list[bool] = []
        workspace._preset_pool.start(_ShortTask(finished))

        workspace.close()

        self.assertFalse(workspace._parameter_change_timer.isActive())
        self.assertFalse(workspace._busy_elapsed_timer.isActive())
        self.assertEqual(finished, [True])
        self.assertEqual(workspace._preset_pool.activeThreadCount(), 0)

    def test_viewer_close_drains_queued_open_and_stops_worker_thread(self) -> None:
        viewer = HeliosViewerMainWindow()
        viewer.controller.open_file(HDF5_ROOT / "Cu_0166_stabilized.h5")
        QtCore.QThread.msleep(10)

        viewer.close()

        self.assertFalse(viewer.controller.worker_thread.isRunning())

    def test_parser_close_waits_for_active_conversion(self) -> None:
        source = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"
        with tempfile.TemporaryDirectory(prefix="helios-close-parse-") as temp_dir:
            output = Path(temp_dir) / "close_during_parse.h5"
            window = HeliosParseViewMainWindow()
            window.parser_controller.parse_log(source, output, overwrite=True)
            QtCore.QThread.msleep(20)

            window.close()

            self.assertFalse(window.parser_controller._thread.isRunning())

    def test_repeated_workspace_and_viewer_close_process_exits_cleanly(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["PYTHONPATH"] = os.pathsep.join(
            (
                str(repo_root / "tests"),
                str(repo_root / "src"),
                env.get("PYTHONPATH", ""),
            )
        )
        script = """
import shiboken6
from PySide6 import QtCore, QtWidgets
from _viewer_test_utils import get_app, process_events
from helios_analysis.workspace import HeliosDerivedWorkspace
from helios_viewer.main_window import HeliosViewerMainWindow
from helios_viewer.slider import apply_absolute_click_slider_behavior

app = get_app()
app_style = app.style()
style_name = app_style.name()
slider = QtWidgets.QSlider()
apply_absolute_click_slider_behavior(slider)
proxy_style = slider.style()
assert proxy_style is not app_style
assert proxy_style.parent() == slider
assert proxy_style.baseStyle().name() == style_name
slider.close()
slider.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
process_events()
assert shiboken6.isValid(app_style)
assert app.style().name() == style_name
for _ in range(3):
    workspace = HeliosDerivedWorkspace()
    workspace.close()
    process_events()
    viewer = HeliosViewerMainWindow()
    viewer.close()
    process_events()
print('repeated Qt close cycles complete')
"""
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("repeated Qt close cycles complete", result.stdout)


if __name__ == "__main__":
    unittest.main()
