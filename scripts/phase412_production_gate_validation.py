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


SCREENSHOT_ROOT = ROOT / "outputs" / "screenshots"
REPORT_ROOT = ROOT / "outputs" / "reports"
TEST_SETTINGS_ROOT = ROOT / "outputs" / "test_settings"


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No candidate exists: {candidates}")


SMALL_HDF5 = resolve_existing_path(
    ROOT / "5Fe+4.9TW+light_stabilized.h5",
    ROOT / "outputs" / "hdf5" / "5Fe+4.9TW+light_stabilized.h5",
)
CU_HDF5 = resolve_existing_path(
    ROOT / "Cu_0166_stabilized.h5",
    ROOT / "outputs" / "hdf5" / "Cu_0166_stabilized.h5",
)
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


def wait_until(predicate, timeout_s: float = 90.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        process_events(30)
        if predicate():
            return
    raise RuntimeError("Timed out waiting for GUI state.")


def combo_set_data(combo: QtWidgets.QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        raise RuntimeError(f"Combo value {value!r} not found")
    combo.setCurrentIndex(index)
    process_events(40)


def save_screenshot(window: QtWidgets.QWidget, name: str) -> str:
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_ROOT / name
    window.grab().save(str(path))
    return str(path)


def open_run(window: HeliosParseViewMainWindow, path: Path) -> None:
    window._open_path(path)
    wait_until(lambda: window.viewer_controller.has_loaded_run() and str(window.viewer_controller.current_run_context().path) == str(path), 120.0)


def wait_for_derived_result(window: HeliosParseViewMainWindow, timeout_s: float = 120.0) -> None:
    derived = window.derived_controller.widget()
    wait_until(lambda: derived._current_result is not None, timeout_s)
    wait_until(lambda: not window.derived_controller._controller._busy, timeout_s)


def colorbar_state(field_map) -> dict[str, object]:
    levels = field_map._colorbar.levels()
    return {
        "label": field_map.current_colorbar_label,
        "levels": None if levels is None else [float(levels[0]), float(levels[1])],
        "render_mode": field_map.current_render_mode,
        "has_span": False if levels is None else float(levels[1]) > float(levels[0]),
    }


def validate_small_snapshot_sync(window: HeliosParseViewMainWindow) -> dict[str, object]:
    open_run(window, SMALL_HDF5)
    window.global_snapshot_spin.setValue(3)
    process_events(100)
    window._set_mode("derived")
    wait_for_derived_result(window, 60.0)
    derived = window.derived_controller.widget()
    window.global_time_spin.setValue(float(window.viewer_controller.current_run_context().time_values[5]) * 1.0e9)
    process_events(100)
    wait_until(lambda: window.viewer_controller.current_run_context().snapshot_index == 5, 30.0)
    wait_until(lambda: derived._current_result is not None and derived._current_result.snapshot_index == 5, 60.0)
    screenshot = save_screenshot(window, "phase412_small_global_snapshot.png")
    return {
        "file": SMALL_HDF5.name,
        "global_snapshot_label": window.global_snapshot_label.text(),
        "derived_snapshot_label": derived.snapshot_label.text(),
        "xrd_profile_title": derived.xrd_plot_panel.profile_plot.current_title,
        "screenshot": screenshot,
    }


def validate_cu_interaction(window: HeliosParseViewMainWindow) -> dict[str, object]:
    open_run(window, CU_HDF5)
    viewer = window.viewer_controller.window
    wait_until(lambda: viewer.current_field_payload is not None, 30.0)
    initial_colorbar = colorbar_state(viewer.field_map_widget)
    window._set_mode("derived")
    wait_for_derived_result(window, 60.0)
    derived = window.derived_controller.widget()

    for tab_index in range(derived.result_tabs.count()):
        derived.result_tabs.setCurrentIndex(tab_index)
        process_events(25)
    for _ in range(2):
        combo_set_data(derived.xrd_display_combo, "q")
        combo_set_data(derived.xrd_display_combo, "degrees")
        combo_set_data(derived.weighting_combo, "mass")
        combo_set_data(derived.weighting_combo, "electron_density")
        combo_set_data(derived.profile_coordinate_combo, "zone")
        combo_set_data(derived.profile_coordinate_combo, "static_x")
        derived.exclude_low_density_checkbox.setChecked(not derived.exclude_low_density_checkbox.isChecked())
        derived.exclude_opposite_velocity_checkbox.setChecked(not derived.exclude_opposite_velocity_checkbox.isChecked())
        derived.min_density_spin.setValue(0.02 if derived.min_density_spin.value() == 0.0 else 0.0)
        derived.zone_lower_spin.setValue(10 if derived.zone_lower_spin.value() == 1 else 1)
        derived.zone_upper_spin.setValue(250 if derived.zone_upper_spin.value() == derived._context.n_zones else derived._context.n_zones)
        process_events(30)

    wait_until(
        lambda: window.derived_controller._controller._active_task_id is None
        and window.derived_controller._controller._pending_request is None
        and not window.derived_controller._controller._busy,
        120.0,
    )

    combo_set_data(derived.plasmon_plot_panel.time_combo, "temperatures")
    combo_set_data(derived.spectroscopy_plot_panel.time_combo, "velocity")
    process_events(60)
    derived_ready = derived._current_result is not None
    derived_selected_zones = None if derived._current_result is None else int(derived._current_result.selected_zone_count)
    plasmon_multicurve_legend = derived.plasmon_plot_panel.time_plot._legend is not None
    spectroscopy_multicurve_legend = derived.spectroscopy_plot_panel.time_plot._legend is not None
    derived_screenshot = save_screenshot(window, "phase412_cu_derived.png")

    window._set_mode("viewer")
    process_events(80)
    viewer.plot_tabs.setCurrentWidget(viewer.mouse_tab)
    process_events(50)
    mouse_colorbar = colorbar_state(viewer.field_map_widget)
    viewer._select_list_item_by_data(viewer.field_list, "zone_width")
    wait_until(lambda: viewer.current_field_name == "zone_width", 30.0)
    process_events(80)
    zone_width_colorbar = colorbar_state(viewer.field_map_widget)
    viewer.set_theme_mode("dark")
    process_events(80)
    dark_colorbar = colorbar_state(viewer.field_map_widget)
    viewer.set_theme_mode("light")
    process_events(80)
    light_colorbar = colorbar_state(viewer.field_map_widget)
    viewer_screenshot = save_screenshot(window, "phase412_cu_viewer_colorbar.png")

    return {
        "file": CU_HDF5.name,
        "derived_ready": derived_ready,
        "derived_selected_zones": derived_selected_zones,
        "xrd_headers_degrees": [derived.xrd_table.horizontalHeaderItem(index).text() for index in range(derived.xrd_table.columnCount())],
        "plasmon_multicurve_legend": plasmon_multicurve_legend,
        "spectroscopy_multicurve_legend": spectroscopy_multicurve_legend,
        "colorbar_states": {
            "initial": initial_colorbar,
            "mouse": mouse_colorbar,
            "zone_width": zone_width_colorbar,
            "dark": dark_colorbar,
            "light": light_colorbar,
        },
        "derived_screenshot": derived_screenshot,
        "viewer_screenshot": viewer_screenshot,
    }


def validate_large_run(window: HeliosParseViewMainWindow, path: Path, *, screenshot_name: str) -> dict[str, object]:
    open_run(window, path)
    window._set_mode("derived")
    wait_for_derived_result(window, 180.0)
    derived = window.derived_controller.widget()
    controller = window.derived_controller._controller
    max_threads = 0

    for index in range(10):
        combo_set_data(derived.weighting_combo, "mass" if index % 2 == 0 else "electron_column")
        combo_set_data(derived.profile_coordinate_combo, "zone" if index % 2 == 0 else "moving_radius")
        derived.exclude_low_density_checkbox.setChecked(index % 2 == 0)
        derived.exclude_opposite_velocity_checkbox.setChecked(index % 3 == 0)
        derived.min_density_spin.setValue(0.05 if index % 2 == 0 else 0.0)
        process_events(25)
        max_threads = max(max_threads, len(controller._tasks._threads))

    window.global_snapshot_slider.setValue(0)
    process_events(20)
    window.global_snapshot_slider.setValue(max(1, int(window.viewer_controller.current_run_context().n_snapshots) // 3))
    process_events(20)
    window.global_snapshot_slider.setValue(int(window.viewer_controller.current_run_context().n_snapshots) - 1)
    process_events(20)
    window.global_snapshot_slider.setValue(max(1, int(window.viewer_controller.current_run_context().n_snapshots) // 4))
    process_events(20)

    wait_until(
        lambda: controller._active_task_id is None and controller._pending_request is None and not controller._busy,
        240.0,
    )
    screenshot = save_screenshot(window, screenshot_name)
    result = derived._current_result
    return {
        "file": path.name,
        "selected_zones": None if result is None else int(result.selected_zone_count),
        "snapshot_label": derived.snapshot_label.text(),
        "global_snapshot_label": window.global_snapshot_label.text(),
        "profile_coordinate": derived.profile_coordinate_combo.currentData(),
        "max_inflight_threads": max_threads,
        "final_threads": len(controller._tasks._threads),
        "screenshot": screenshot,
    }


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    configure_test_settings()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = HeliosParseViewMainWindow()
    window.show()
    process_events(120)
    report: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paths": {
            "small": str(SMALL_HDF5),
            "cu": str(CU_HDF5),
            "heavy": str(HEAVY_HDF5),
            "stress": str(STRESS_HDF5),
        },
    }
    try:
        report["small_snapshot_sync"] = validate_small_snapshot_sync(window)
        report["cu_interaction"] = validate_cu_interaction(window)
        report["heavy_multilayer"] = validate_large_run(window, HEAVY_HDF5, screenshot_name="phase412_heavy_multilayer.png")
        report["stress_al_epoxy_ch"] = validate_large_run(window, STRESS_HDF5, screenshot_name="phase412_stress_al_epoxy_ch.png")
    finally:
        window.close()
        process_events(100)
    report_path = REPORT_ROOT / "phase412_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)
    app.processEvents()


if __name__ == "__main__":
    main()
