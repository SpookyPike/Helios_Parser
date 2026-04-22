"""Capture current HELIOS Analyzer screenshots used by the manual.

This script is intentionally small and dataset-specific. The goal is to keep
the documentation screenshots aligned with the current edge/center coordinate
model, viewer geometry semantics, and Derived workspace behavior.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PySide6 import QtCore, QtWidgets

from helios_app.main_app import HeliosParseViewMainWindow
from helios_parser import write_hdf5


ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = ROOT / "docs" / "assets" / "screenshots"
GENERATED_HDF5_DIR = ROOT / "outputs" / "hdf5"


def process_events(delay_ms: int = 0) -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    if delay_ms > 0:
        QtCore.QThread.msleep(delay_ms)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def wait_until(predicate, timeout_s: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        process_events(40)
        if predicate():
            return
    raise RuntimeError("Timed out while capturing documentation screenshot.")


def combo_set_data(combo: QtWidgets.QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        raise RuntimeError(f"Combo value {value!r} was not found.")
    combo.setCurrentIndex(index)
    process_events(80)


def save_window(window: QtWidgets.QWidget, name: str) -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_DIR / name
    window.grab().save(str(path))
    return path


def capture_parser_preview(window: HeliosParseViewMainWindow) -> Path:
    window._set_theme_mode("light")
    window._open_path(ROOT / "5Fe+4.9TW+light.log")
    wait_until(lambda: window.current_preview is not None, 30.0)
    window._set_mode("parser")
    process_events(200)
    return save_window(window, "parser_preview_current.png")


def capture_viewer_zone_index(window: HeliosParseViewMainWindow) -> Path:
    window._set_theme_mode("light")
    window._open_path(ROOT / "outputs" / "hdf5" / "Cu_0166_stabilized.h5")
    viewer = window.viewer_controller.window
    wait_until(lambda: viewer.current_field_payload is not None, 30.0)
    window._set_mode("viewer")
    combo_set_data(viewer.map_orientation_combo, "time_x_coord_y")
    combo_set_data(viewer.map_coordinate_combo, "zone")
    viewer.boundary_overlay_checkbox.setChecked(True)
    viewer.plot_tabs.setCurrentWidget(viewer.lineout_plot)
    process_events(250)
    return save_window(window, "viewer_zone_index_current.png")


def capture_viewer_mouse_probe(window: HeliosParseViewMainWindow) -> Path:
    window._set_theme_mode("dark")
    window._open_path(ROOT / "outputs" / "hdf5" / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5")
    viewer = window.viewer_controller.window
    wait_until(lambda: viewer.current_field_payload is not None, 60.0)
    wait_until(lambda: viewer.radius_payload is not None, 60.0)
    window._set_mode("viewer")
    combo_set_data(viewer.map_coordinate_combo, "moving_radius")
    combo_set_data(viewer.map_orientation_combo, "time_x_coord_y")
    viewer.boundary_overlay_checkbox.setChecked(True)
    viewer.plot_tabs.setCurrentWidget(viewer.mouse_tab)
    process_events(120)
    display_time = viewer._display_time_values(np.asarray(viewer.run_payload.time, dtype=np.float64))
    radius_display, _ = viewer._display_field_data("radius", viewer.radius_payload.unit, viewer.radius_payload.data)
    snapshot_index = min(80, max(0, display_time.size - 1))
    zone_index = min(1000, max(0, radius_display.shape[1] - 1))
    viewer._on_map_probe_clicked(float(display_time[snapshot_index]), float(radius_display[snapshot_index, zone_index]))
    process_events(250)
    return save_window(window, "viewer_mouse_probe_current.png")


def capture_viewer_cylindrical(window: HeliosParseViewMainWindow) -> Path:
    window._set_theme_mode("light")
    GENERATED_HDF5_DIR.mkdir(parents=True, exist_ok=True)
    hdf5_path = GENERATED_HDF5_DIR / "Cu1e17_cyl_docs.h5"
    write_hdf5(ROOT / "Cu1e17_cyl.log", hdf5_path, overwrite=True)
    window._open_path(hdf5_path)
    viewer = window.viewer_controller.window
    wait_until(lambda: viewer.current_field_payload is not None, 30.0)
    window._set_mode("viewer")
    combo_set_data(viewer.map_orientation_combo, "time_x_coord_y")
    combo_set_data(viewer.map_coordinate_combo, "static_x")
    viewer.boundary_overlay_checkbox.setChecked(True)
    viewer.plot_tabs.setCurrentWidget(viewer.lineout_plot)
    process_events(250)
    return save_window(window, "viewer_cylindrical_radius_current.png")


def capture_derived_overview(window: HeliosParseViewMainWindow) -> Path:
    window._set_theme_mode("light")
    window._open_path(ROOT / "outputs" / "hdf5" / "Cu_0166_stabilized.h5")
    viewer = window.viewer_controller.window
    derived = window.derived_controller.widget()
    wait_until(lambda: viewer.current_field_payload is not None, 30.0)
    window._set_mode("derived")
    wait_until(lambda: derived._current_result is not None, 30.0)
    derived.result_tabs.setCurrentWidget(derived.plasmon_tab)
    process_events(250)
    return save_window(window, "derived_plasmon_current.png")


def capture_preheat(window: HeliosParseViewMainWindow) -> Path:
    window._set_theme_mode("light")
    window._open_path(ROOT / "50Al+10E+25CH+3.5TW_stabilized.h5")
    viewer = window.viewer_controller.window
    derived = window.derived_controller.widget()
    wait_until(lambda: viewer.current_field_payload is not None, 60.0)
    window._set_mode("derived")
    wait_until(lambda: derived._current_result is not None, 120.0)
    derived.result_tabs.setCurrentWidget(derived.preheat_tab)
    wait_until(
        lambda: derived._current_result is not None
        and derived._current_result.preheat is not None
        and not window.derived_controller._controller._busy,
        240.0,
    )
    process_events(250)
    return save_window(window, "derived_preheat_current.png")


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = HeliosParseViewMainWindow()
    window.show()
    process_events(200)
    outputs = [
        capture_parser_preview(window),
        capture_viewer_zone_index(window),
        capture_viewer_mouse_probe(window),
        capture_viewer_cylindrical(window),
        capture_derived_overview(window),
        capture_preheat(window),
    ]
    for path in outputs:
        print(path)
    window.close()
    process_events(100)
    app.quit()


if __name__ == "__main__":
    main()
