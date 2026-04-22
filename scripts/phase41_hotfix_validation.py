from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets, QtTest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from helios_app.main_app import HeliosParseViewMainWindow  # noqa: E402


HDF5_ROOT = ROOT / "outputs" / "hdf5"
SCREENSHOT_ROOT = ROOT / "outputs" / "screenshots"
REPORT_ROOT = ROOT / "outputs" / "reports"


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
        process_events(40)
        if predicate():
            return
    raise RuntimeError("Timed out waiting for GUI state.")


def combo_set_data(combo: QtWidgets.QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        raise RuntimeError(f"Combo value {value!r} not found.")
    combo.setCurrentIndex(index)
    process_events(60)


def save_screenshot(window: QtWidgets.QWidget, name: str) -> str:
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_ROOT / name
    window.grab().save(str(path))
    return str(path)


def _derived_controller(window: HeliosParseViewMainWindow):
    return window.derived_controller._controller


def wait_for_derived_ready(window: HeliosParseViewMainWindow, timeout_s: float = 120.0) -> None:
    derived_controller = _derived_controller(window)
    wait_until(
        lambda: window.viewer_controller.window.current_field_payload is not None
        and window.derived_controller.widget()._current_result is not None
        and not derived_controller._busy,
        timeout_s,
    )


def wait_for_result_change(window: HeliosParseViewMainWindow, previous_key: tuple[object, ...], timeout_s: float = 120.0) -> None:
    derived_controller = _derived_controller(window)
    wait_until(
        lambda: window.derived_controller.widget()._current_result is not None
        and window.derived_controller.widget()._current_result.context_key != previous_key
        and not derived_controller._busy,
        timeout_s,
    )


def set_checked_values(widget: QtWidgets.QListWidget, keep: set[int]) -> None:
    widget.blockSignals(True)
    for index in range(widget.count()):
        item = widget.item(index)
        value = int(item.data(QtCore.Qt.UserRole))
        item.setCheckState(QtCore.Qt.Checked if value in keep else QtCore.Qt.Unchecked)
    widget.blockSignals(False)
    process_events(40)


def validate_small(window: HeliosParseViewMainWindow, path: Path) -> dict[str, object]:
    window._set_theme_mode("light")
    window._open_path(path)
    wait_for_derived_ready(window, 60.0)
    window._set_mode("derived")
    process_events(120)
    derived = window.derived_controller.widget()

    base_key = derived._current_result.context_key
    base_selected = int(derived._current_result.selected_zone_count)
    derived.exclude_entry_region_checkbox.setChecked(True)
    wait_for_result_change(window, base_key, 60.0)
    empty_result = derived._current_result
    assert empty_result is not None
    empty_panel_state = {
        "xrd_time_enabled": derived.xrd_plot_panel.time_combo.isEnabled(),
        "plasmon_time_enabled": derived.plasmon_plot_panel.time_combo.isEnabled(),
        "spectroscopy_profile_enabled": derived.spectroscopy_plot_panel.profile_combo.isEnabled(),
    }

    screenshot = save_screenshot(window, "phase411_small_empty_selection_light.png")
    derived.exclude_entry_region_checkbox.setChecked(False)
    wait_for_result_change(window, empty_result.context_key, 60.0)

    return {
        "file": path.name,
        "base_selected_zones": base_selected,
        "empty_selection_warning_count": len(empty_result.warnings),
        "empty_selection_selected_zones": int(empty_result.selected_zone_count),
        "empty_panels_hidden": empty_panel_state,
        "screenshot": screenshot,
    }


def validate_medium(window: HeliosParseViewMainWindow, path: Path) -> dict[str, object]:
    window._set_theme_mode("light")
    window._open_path(path)
    wait_for_derived_ready(window, 90.0)
    window._set_mode("derived")
    process_events(120)
    derived = window.derived_controller.widget()

    for tab_index in range(derived.result_tabs.count()):
        derived.result_tabs.setCurrentIndex(tab_index)
        process_events(30)

    combo_set_data(derived.xrd_display_combo, "q")
    xrd_key = derived.xrd_plot_panel.time_combo.currentData()
    combo_set_data(derived.spectroscopy_shift_unit_combo, "ev")
    spectroscopy_metrics = derived.spectroscopy_metrics.toPlainText()

    screenshot = save_screenshot(window, "phase411_cu_derived_q_light.png")
    return {
        "file": path.name,
        "xrd_display_time_key": xrd_key,
        "spectroscopy_metrics_contains_ev": "eV" in spectroscopy_metrics,
        "warning_summary": derived.warning_summary_label.text(),
        "screenshot": screenshot,
    }


def validate_large(window: HeliosParseViewMainWindow, path: Path) -> dict[str, object]:
    window._set_theme_mode("dark")
    window._open_path(path)
    wait_for_derived_ready(window, 150.0)
    window._set_mode("derived")
    process_events(200)
    derived = window.derived_controller.widget()

    initial = derived._current_result
    assert initial is not None
    initial_key = initial.context_key

    combo_set_data(derived.weighting_combo, "mass")
    wait_for_result_change(window, initial_key, 120.0)
    weighted = derived._current_result
    assert weighted is not None

    weighted_key = weighted.context_key
    derived.exclude_low_density_checkbox.setChecked(True)
    derived.min_density_spin.setValue(0.05)
    combo_set_data(derived.observation_side_combo, "back")
    derived.zone_upper_spin.setValue(900)
    wait_for_result_change(window, weighted_key, 120.0)
    filtered = derived._current_result
    assert filtered is not None

    derived.result_tabs.setCurrentWidget(derived.xrd_tab)
    process_events(80)
    view_box = derived.xrd_plot_panel.time_plot._plot.getViewBox()
    initial_range = tuple(tuple(axis) for axis in view_box.viewRange())
    x_values = np.asarray(derived.xrd_plot_panel.time_plot.last_x_values, dtype=np.float64)
    finite_x = x_values[np.isfinite(x_values)]
    if finite_x.size >= 2:
        span = float(finite_x.max() - finite_x.min())
        left = float(finite_x.min())
        view_box.setXRange(left, left + span * 0.4, padding=0.0)
        process_events(60)
    zoomed_range = tuple(tuple(axis) for axis in view_box.viewRange())
    derived.xrd_plot_panel.reset_time_button.click()
    process_events(60)
    reset_range = tuple(tuple(axis) for axis in view_box.viewRange())
    QtTest.QTest.mouseDClick(
        derived.xrd_plot_panel.time_plot._plot.viewport(),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        derived.xrd_plot_panel.time_plot._plot.viewport().rect().center(),
    )
    process_events(60)
    double_click_range = tuple(tuple(axis) for axis in view_box.viewRange())

    screenshot = save_screenshot(window, "phase411_large_filtered_dark.png")
    return {
        "file": path.name,
        "initial_selected_zones": int(initial.selected_zone_count),
        "weighted_selected_zones": int(weighted.selected_zone_count),
        "filtered_selected_zones": int(filtered.selected_zone_count),
        "xrd_last_layer_density_before": float(initial.xrd.layers[-1].compressed_density_g_cm3) if initial.xrd.layers else None,
        "xrd_last_layer_density_mass_weighted": float(weighted.xrd.layers[-1].compressed_density_g_cm3) if weighted.xrd.layers else None,
        "xrd_last_layer_density_after": float(filtered.xrd.layers[-1].compressed_density_g_cm3) if filtered.xrd.layers else None,
        "plasmon_te_before": float(initial.plasmon.electron_temperature_ev),
        "plasmon_te_mass_weighted": float(weighted.plasmon.electron_temperature_ev),
        "plasmon_te_after": float(filtered.plasmon.electron_temperature_ev),
        "spectroscopy_bulk_velocity_before_cm_s": float(initial.spectroscopy.bulk_velocity_cm_s),
        "spectroscopy_bulk_velocity_mass_weighted_cm_s": float(weighted.spectroscopy.bulk_velocity_cm_s),
        "spectroscopy_bulk_velocity_after_cm_s": float(filtered.spectroscopy.bulk_velocity_cm_s),
        "shock_breakout_time_ns": None if filtered.shock.breakout_time_s is None else float(filtered.shock.breakout_time_s) * 1.0e9,
        "xrd_zoom_reset": {
            "initial": initial_range,
            "zoomed": zoomed_range,
            "reset": reset_range,
            "double_click": double_click_range,
        },
        "screenshot": screenshot,
    }


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": [],
    }
    window = HeliosParseViewMainWindow()
    window.show()
    process_events(200)
    try:
        report["files"].append(validate_small(window, HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"))
        report["files"].append(validate_medium(window, HDF5_ROOT / "Cu_0166_stabilized.h5"))
        report["files"].append(validate_large(window, HDF5_ROOT / "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"))
    finally:
        window.close()
        process_events(100)

    report_path = REPORT_ROOT / "phase41_hotfix_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)
    app.processEvents()


if __name__ == "__main__":
    main()
