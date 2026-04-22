from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtWidgets


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from helios_app.main_app import HeliosParseViewMainWindow  # noqa: E402
from helios_app.session_state import reset_session_state  # noqa: E402


REPORT_ROOT = ROOT / "outputs" / "reports"
SCREENSHOT_ROOT = ROOT / "outputs" / "screenshots"
TEST_SETTINGS_ROOT = ROOT / "outputs" / "test_settings"


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No candidate exists: {candidates}")


CU_HDF5 = resolve_existing_path(ROOT / "Cu_0166_stabilized.h5", ROOT / "outputs" / "hdf5" / "Cu_0166_stabilized.h5")
HEAVY_HDF5 = resolve_existing_path(
    ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5",
    ROOT / "outputs" / "hdf5" / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5",
)
STRESS_HDF5 = resolve_existing_path(
    ROOT / "50Al+10E+25CH+3.5TW_stabilized.h5",
    ROOT / "outputs" / "hdf5" / "50Al+10E+25CH+3.5TW_stabilized.h5",
)


def configure_test_settings() -> None:
    TEST_SETTINGS_ROOT.mkdir(parents=True, exist_ok=True)
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(TEST_SETTINGS_ROOT))
    QtCore.QSettings("HeliosViewer", "HELIOS HDF5 Quick Look").clear()
    QtCore.QSettings("HeliosViewer", "HELIOS HDF5 Quick Look").sync()
    reset_session_state()


def process_events(delay_ms: int = 0) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    if delay_ms > 0:
        QtCore.QThread.msleep(delay_ms)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def wait_until(predicate, timeout_s: float = 120.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        process_events(20)
        if predicate():
            return
    raise RuntimeError("Timed out waiting for GUI state.")


def save_screenshot(window: QtWidgets.QWidget, name: str) -> str:
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_ROOT / name
    window.grab().save(str(path))
    return str(path)


def open_run(window: HeliosParseViewMainWindow, path: Path) -> None:
    window._open_path(path)
    wait_until(lambda: window.viewer_controller.has_loaded_run() and str(window.viewer_controller.current_run_context().path) == str(path))


def wait_for_derived_result(window: HeliosParseViewMainWindow, timeout_s: float = 180.0) -> None:
    derived = window.derived_controller.widget()
    controller = window.derived_controller._controller
    wait_until(lambda: derived._current_result is not None, timeout_s)
    wait_until(lambda: controller._active_task_id is None and controller._pending_request is None and not controller._busy, timeout_s)


def snapshot_title_map(workspace) -> dict[str, str]:
    return {
        "xrd": workspace.xrd_plot_panel.profile_plot.current_title,
        "plasmon": workspace.plasmon_plot_panel.profile_plot.current_title,
        "transmission": workspace.transmission_plot_panel.profile_plot.current_title,
        "spectroscopy": workspace.spectroscopy_plot_panel.profile_plot.current_title,
    }


def snapshot_scalar_map(workspace) -> dict[str, str]:
    return {
        "xrd": workspace.xrd_summary_label.text(),
        "plasmon": workspace.plasmon_summary_label.text(),
        "transmission": workspace.transmission_summary_label.text(),
        "spectroscopy": workspace.spectroscopy_summary_label.text(),
    }


def rapid_slider_sequence(window: HeliosParseViewMainWindow, targets: list[int]) -> dict[str, object]:
    controller = window.derived_controller._controller
    workspace = window.derived_controller.widget()
    max_threads = len(controller._tasks._threads)
    preview_messages: list[str] = []
    window._on_global_snapshot_slider_pressed()
    for target in targets:
        window.global_snapshot_slider.setValue(int(target))
        window._on_global_snapshot_slider_moved(int(target))
        process_events(5)
        preview_messages.append(window.global_snapshot_label.text())
        max_threads = max(max_threads, len(controller._tasks._threads))
    window._on_global_snapshot_slider_released()
    process_events(20)
    preview_messages.append(workspace.result_status_label.text())
    return {
        "max_threads_during_drag": max_threads,
        "preview_messages": preview_messages,
    }


def validate_snapshot_sync(window: HeliosParseViewMainWindow, path: Path, *, screenshot_name: str, targets: list[int]) -> dict[str, object]:
    open_run(window, path)
    window._set_mode("derived")
    wait_for_derived_result(window)
    workspace = window.derived_controller.widget()
    controller = window.derived_controller._controller

    before_index = int(workspace._current_result.snapshot_index)
    before_titles = snapshot_title_map(workspace)
    before_scalars = snapshot_scalar_map(workspace)

    started = time.perf_counter()
    slider_info = rapid_slider_sequence(window, targets)
    final_target = int(targets[-1])
    wait_until(lambda: window.viewer_controller.current_run_context().snapshot_index == final_target, 60.0)
    wait_until(lambda: workspace._current_result is not None and workspace._current_result.snapshot_index == final_target, 120.0)
    wait_until(lambda: controller._active_task_id is None and controller._pending_request is None and not controller._busy, 120.0)
    elapsed_s = time.perf_counter() - started

    after_titles = snapshot_title_map(workspace)
    after_scalars = snapshot_scalar_map(workspace)
    screenshot = save_screenshot(window, screenshot_name)
    return {
        "file": path.name,
        "before_snapshot": before_index,
        "after_snapshot": int(workspace._current_result.snapshot_index),
        "viewer_snapshot": int(window.viewer_controller.current_run_context().snapshot_index),
        "elapsed_s": round(elapsed_s, 3),
        "update_kind": controller._last_completed_update_kind,
        "max_inflight_threads": slider_info["max_threads_during_drag"],
        "titles_changed": {
            key: before_titles[key] != after_titles[key]
            for key in before_titles
        },
        "scalars_changed": {
            key: before_scalars[key] != after_scalars[key]
            for key in before_scalars
        },
        "status_label": workspace.result_status_label.text(),
        "global_snapshot_label": window.global_snapshot_label.text(),
        "snapshot_cursor_visible": {
            "xrd": bool(workspace.xrd_plot_panel.time_plot.current_cursor_visible),
            "plasmon": bool(workspace.plasmon_plot_panel.time_plot.current_cursor_visible),
            "transmission": bool(workspace.transmission_plot_panel.time_plot.current_cursor_visible),
            "spectroscopy": bool(workspace.spectroscopy_plot_panel.time_plot.current_cursor_visible),
            "shock_position": bool(workspace.shock_position_plot.current_cursor_visible),
            "shock_velocity": bool(workspace.shock_velocity_plot.current_cursor_visible),
        },
        "preview_messages": slider_info["preview_messages"],
        "screenshot": screenshot,
    }


def validate_viewer_colorbar(window: HeliosParseViewMainWindow, path: Path, *, screenshot_name: str) -> dict[str, object]:
    open_run(window, path)
    viewer = window.viewer_controller.window
    wait_until(lambda: viewer.current_field_payload is not None, 30.0)
    field_map = viewer.field_map_widget

    def colorbar_state() -> dict[str, object]:
        levels = field_map._colorbar.levels()
        return {
            "label": field_map.current_colorbar_label,
            "levels": None if levels is None else [float(levels[0]), float(levels[1])],
            "render_mode": field_map.current_render_mode,
            "has_span": False if levels is None else float(levels[1]) > float(levels[0]),
        }

    initial = colorbar_state()
    viewer.plot_tabs.setCurrentWidget(viewer.mouse_tab)
    process_events(50)
    mouse = colorbar_state()
    window._set_mode("derived")
    process_events(80)
    window._set_mode("viewer")
    process_events(80)
    roundtrip = colorbar_state()
    viewer.set_theme_mode("dark")
    process_events(80)
    dark = colorbar_state()
    viewer.set_theme_mode("light")
    process_events(80)
    light = colorbar_state()
    screenshot = save_screenshot(window, screenshot_name)
    return {
        "file": path.name,
        "initial": initial,
        "mouse": mouse,
        "roundtrip": roundtrip,
        "dark": dark,
        "light": light,
        "screenshot": screenshot,
    }


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    configure_test_settings()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = HeliosParseViewMainWindow()
    window.show()
    process_events(100)

    report: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paths": {
            "cu": str(CU_HDF5),
            "heavy": str(HEAVY_HDF5),
            "stress": str(STRESS_HDF5),
        },
    }
    try:
        report["cu_snapshot_sync"] = validate_snapshot_sync(
            window,
            CU_HDF5,
            screenshot_name="phase412c_cu_snapshot_sync.png",
            targets=[20, 60, 120, 180],
        )
        report["heavy_snapshot_sync"] = validate_snapshot_sync(
            window,
            HEAVY_HDF5,
            screenshot_name="phase412c_heavy_snapshot_sync.png",
            targets=[30, 120, 240, 360],
        )
        report["stress_snapshot_sync"] = validate_snapshot_sync(
            window,
            STRESS_HDF5,
            screenshot_name="phase412c_stress_snapshot_sync.png",
            targets=[10, 50, 100, 175],
        )
        report["cu_viewer_colorbar"] = validate_viewer_colorbar(
            window,
            CU_HDF5,
            screenshot_name="phase412c_cu_viewer_colorbar.png",
        )
    finally:
        window.close()
        process_events(100)

    report_path = REPORT_ROOT / "phase412c_snapshot_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)
    app.processEvents()


if __name__ == "__main__":
    main()
