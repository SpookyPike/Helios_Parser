from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from helios_viewer.main_window import HeliosViewerMainWindow  # noqa: E402


HDF5_ROOT = ROOT / "outputs" / "hdf5"
SCREENSHOT_ROOT = ROOT / "outputs" / "screenshots"
EXPORT_ROOT = ROOT / "outputs" / "exports"
REPORT_ROOT = ROOT / "outputs" / "reports"


def wait_until(predicate, timeout_s: float = 40.0) -> None:
    deadline = time.perf_counter() + timeout_s
    app = QtWidgets.QApplication.instance()
    assert app is not None
    while time.perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        QtCore.QThread.msleep(20)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if predicate():
            return
    raise RuntimeError("Timed out waiting for GUI state.")


def process_events(delay_ms: int = 0) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    if delay_ms > 0:
        QtCore.QThread.msleep(delay_ms)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def combo_set_data(combo: QtWidgets.QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        raise RuntimeError(f"Combo value {value!r} not found.")
    combo.setCurrentIndex(index)
    process_events(20)


def select_field(window: HeliosViewerMainWindow, field_name: str) -> None:
    for row in range(window.field_list.count()):
        item = window.field_list.item(row)
        if str(item.data(QtCore.Qt.UserRole)) == field_name:
            window.field_list.setCurrentRow(row)
            wait_until(lambda: window.current_field_payload is not None and window.current_field_payload.field_name == field_name)
            return
    raise RuntimeError(f"Field {field_name!r} not found.")


def visible_bounds(widget) -> bool:
    bounds = widget._current_data_bounds()
    if bounds is None:
        return False
    x_view, y_view = widget._plot.getViewBox().viewRange()
    x_tol = max(1.0e-12, abs(bounds[1] - bounds[0]) * 0.05)
    y_tol = max(1.0e-12, abs(bounds[3] - bounds[2]) * 0.05)
    return (
        x_view[0] <= bounds[0] + x_tol
        and x_view[1] >= bounds[1] - x_tol
        and y_view[0] <= bounds[2] + y_tol
        and y_view[1] >= bounds[3] - y_tol
    )


def image_size(path: Path) -> tuple[int, int]:
    image = QtGui.QImage(str(path))
    return image.width(), image.height()


def ensure_output_dirs() -> None:
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def save_screenshot(window: HeliosViewerMainWindow, name: str) -> Path:
    path = SCREENSHOT_ROOT / name
    window.grab().save(str(path))
    return path


def validate_file(path: Path) -> dict[str, object]:
    window = HeliosViewerMainWindow()
    window.show()
    started = time.perf_counter()
    window.load_file(path)
    wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None)
    wait_until(lambda: window.field_map_widget.last_display_image is not None or window.field_map_widget.last_mesh_z is not None)
    open_seconds = time.perf_counter() - started
    result: dict[str, object] = {
        "file": path.name,
        "open_seconds": round(open_seconds, 3),
        "first_render_visible": bool(visible_bounds(window.field_map_widget)),
        "field": window.current_field_payload.field_name if window.current_field_payload is not None else None,
        "render_mode": window.field_map_widget.current_render_mode,
    }

    if path.name.startswith("5Fe"):
        window.set_theme_mode("light")
        for cmap in ("turbo", "viridis", "plasma", "inferno", "magma", "jet", "hot", "gray"):
            combo_set_data(window.colormap_combo, cmap)
        window.plot_tabs.setCurrentWidget(window.mouse_tab)
        process_events(50)
        window._set_probe_selection(3, 120, frozen=True)
        process_events(50)
        small_png = EXPORT_ROOT / "phase34_small_field_map_1600x900.png"
        window._save_png_export(window.field_map_widget, small_png, transparent=False, width=1600, height=900, dpi=300)
        result["export_1600x900"] = {"path": str(small_png), "size": image_size(small_png)}
        result["screenshot"] = str(save_screenshot(window, "phase34_small_mouse_light.png"))
        result["mouse_mode_vertical_visible"] = bool(visible_bounds(window.mouse_vertical_plot))
        result["mouse_mode_horizontal_visible"] = bool(visible_bounds(window.mouse_horizontal_plot))
    elif path.name.startswith("Cu_0166"):
        window.set_theme_mode("light")
        select_field(window, "radiation_sink")
        combo_set_data(window.colormap_combo, "jet")
        combo_set_data(window.map_orientation_combo, "time_x_coord_y")
        combo_set_data(window.map_coordinate_combo, "static_x")
        medium_png = EXPORT_ROOT / "phase34_cu_field_map_1920x1080.png"
        window._save_png_export(window.field_map_widget, medium_png, transparent=False, width=1920, height=1080, dpi=300)
        result["export_1920x1080"] = {"path": str(medium_png), "size": image_size(medium_png)}
        result["screenshot"] = str(save_screenshot(window, "phase34_cu_slice_light.png"))
    else:
        wait_until(lambda: window.radius_payload is not None)
        window.set_theme_mode("dark")
        combo_set_data(window.map_coordinate_combo, "moving_radius")
        combo_set_data(window.colormap_combo, "hot")
        window.plot_tabs.setCurrentWidget(window.mouse_tab)
        process_events(50)
        window._set_probe_selection(20, 300, frozen=True)
        process_events(50)
        render_before = int(window.field_map_widget.render_call_count)
        mesh_before = int(window.field_map_widget.mesh_render_count)
        window.mouse_time_slider.setValue(min(window.mouse_time_slider.maximum(), 40))
        process_events(20)
        window.mouse_coordinate_slider.setValue(min(window.mouse_coordinate_slider.maximum(), 550))
        process_events(50)
        large_png = EXPORT_ROOT / "phase34_large_field_map_2000x2000.png"
        window._save_png_export(window.field_map_widget, large_png, transparent=True, width=2000, height=2000, dpi=300)
        result["export_2000x2000"] = {"path": str(large_png), "size": image_size(large_png)}
        result["probe_adjustment_render_delta"] = int(window.field_map_widget.render_call_count) - render_before
        result["probe_adjustment_mesh_delta"] = int(window.field_map_widget.mesh_render_count) - mesh_before
        result["mouse_mode_vertical_visible"] = bool(visible_bounds(window.mouse_vertical_plot))
        result["mouse_mode_horizontal_visible"] = bool(visible_bounds(window.mouse_horizontal_plot))
        result["screenshot"] = str(save_screenshot(window, "phase34_large_mouse_dark.png"))

    window.close()
    process_events(20)
    return result


def main() -> None:
    ensure_output_dirs()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": [],
    }
    for file_name in (
        "5Fe+4.9TW+light_stabilized.h5",
        "Cu_0166_stabilized.h5",
        "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5",
    ):
        report["files"].append(validate_file(HDF5_ROOT / file_name))
    report_path = REPORT_ROOT / "phase34_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)
    app.processEvents()


if __name__ == "__main__":
    main()
