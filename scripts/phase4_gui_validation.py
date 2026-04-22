from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from helios_app.main_app import HeliosParseViewMainWindow  # noqa: E402


HDF5_ROOT = ROOT / "outputs" / "hdf5"
SCREENSHOT_ROOT = ROOT / "outputs" / "screenshots"
REPORT_ROOT = ROOT / "outputs" / "reports"


def wait_until(predicate, timeout_s: float = 60.0) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    deadline = time.perf_counter() + timeout_s
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
    process_events(40)


def ensure_outputs() -> None:
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def save_screenshot(window: HeliosParseViewMainWindow, name: str) -> str:
    path = SCREENSHOT_ROOT / name
    window.grab().save(str(path))
    return str(path)


def _derived_ready(window: HeliosParseViewMainWindow) -> bool:
    result = window.derived_controller.widget()._current_result
    return result is not None


def validate_file(path: Path) -> dict[str, object]:
    window = HeliosParseViewMainWindow()
    window.show()
    started = time.perf_counter()
    window._open_path(path)
    wait_until(
        lambda: window.viewer_controller.window.run_payload is not None
        and window.viewer_controller.window.current_field_payload is not None,
        90.0,
    )
    wait_until(lambda: _derived_ready(window), 120.0)
    open_seconds = time.perf_counter() - started
    viewer = window.viewer_controller.window
    derived = window.derived_controller.widget()
    result = derived._current_result
    assert result is not None

    record: dict[str, object] = {
        "file": path.name,
        "open_seconds": round(open_seconds, 3),
        "viewer_field": viewer.current_field_name,
        "derived_snapshot_index": int(result.snapshot_index),
        "selected_zone_count": int(result.selected_zone_count),
        "warning_count": len(result.warnings),
        "shock_breakout_time_ns": None if result.shock.breakout_time_s is None else float(result.shock.breakout_time_s) * 1.0e9,
        "plasmon_regime": result.plasmon.regime_label,
        "thomson_tau": float(result.transmission.thomson_tau),
        "doppler_shift_nm": float(result.spectroscopy.doppler_shift_nm),
    }

    if path.name.startswith("5Fe"):
        window._set_theme_mode("light")
        window._set_mode("derived")
        process_events(80)
        record["screenshot"] = save_screenshot(window, "phase4_small_derived_light.png")
    elif path.name.startswith("Cu_0166"):
        row = next(
            index
            for index in range(viewer.field_list.count())
            if str(viewer.field_list.item(index).data(QtCore.Qt.UserRole)) == "radiation_sink"
        )
        viewer.field_list.setCurrentRow(row)
        wait_until(lambda: viewer.current_field_payload is not None and viewer.current_field_payload.field_name == "radiation_sink", 60.0)
        window._set_mode("derived")
        process_events(80)
        record["new_format_field"] = "radiation_sink"
        record["screenshot"] = save_screenshot(window, "phase4_cu_derived_light.png")
    else:
        wait_until(lambda: viewer.radius_payload is not None, 120.0)
        window._set_theme_mode("dark")
        combo_set_data(viewer.map_coordinate_combo, "moving_radius")
        combo_set_data(viewer.map_orientation_combo, "time_x_coord_y")
        viewer.plot_tabs.setCurrentWidget(viewer.mouse_tab)
        process_events(80)
        viewer._resume_hover_probe()
        display_time = viewer._display_time_values(np.asarray(viewer.run_payload.time, dtype=np.float64))
        radius_display, _ = viewer._display_field_data("radius", viewer.radius_payload.unit, viewer.radius_payload.data)
        render_before = int(viewer.field_map_widget.render_call_count)
        mesh_before = int(viewer.field_map_widget.mesh_render_count)
        for snapshot_index, zone_index in ((20, 300), (60, 900)):
            viewer._on_map_probe_moved(float(display_time[snapshot_index]), float(radius_display[snapshot_index, zone_index]))
            wait_until(lambda snap=snapshot_index: viewer._probe_snapshot_index == snap, 30.0)
            process_events(40)
        record["hover_render_delta"] = int(viewer.field_map_widget.render_call_count) - render_before
        record["hover_mesh_delta"] = int(viewer.field_map_widget.mesh_render_count) - mesh_before
        record["mouse_vertical_curves"] = int(viewer.mouse_vertical_plot.current_curve_count)
        record["mouse_horizontal_curves"] = int(viewer.mouse_horizontal_plot.current_curve_count)
        record["mouse_screenshot"] = save_screenshot(window, "phase4_large_mouse_dark.png")
        window._set_mode("derived")
        process_events(80)
        record["screenshot"] = save_screenshot(window, "phase4_large_derived_dark.png")

    window.close()
    process_events(20)
    return record


def main() -> None:
    ensure_outputs()
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
    path = REPORT_ROOT / "phase4_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)
    app.processEvents()


if __name__ == "__main__":
    main()
