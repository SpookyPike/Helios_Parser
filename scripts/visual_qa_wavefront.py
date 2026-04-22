from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtWidgets


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from helios_app.main_app import HeliosParseViewMainWindow  # noqa: E402
from helios.instrumentation import snapshot_metrics  # noqa: E402
from helios_app.session_state import reset_session_state  # noqa: E402


SCREENSHOT_ROOT = ROOT / "outputs" / "screenshots" / "visual_qa"
REPORT_PATH = SCREENSHOT_ROOT / "visual_qa_report.json"
TEST_SETTINGS_ROOT = ROOT / "outputs" / "test_settings"
SIMPLE_HDF5 = ROOT / "outputs" / "hdf5" / "5Fe+4.9TW+light_stabilized.h5"
COMPLEX_HDF5 = ROOT / "50Al+10E+25CH+3.5TW_stabilized.h5"


def configure_test_settings() -> None:
    TEST_SETTINGS_ROOT.mkdir(parents=True, exist_ok=True)
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(TEST_SETTINGS_ROOT))
    store = QtCore.QSettings("HeliosViewer", "HELIOS HDF5 Quick Look")
    store.clear()
    store.sync()
    reset_session_state()


def process_events(delay_ms: int = 0) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    if delay_ms > 0:
        QtCore.QThread.msleep(delay_ms)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def wait_until(predicate, timeout_s: float = 120.0, *, delay_ms: int = 30) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        process_events(delay_ms)
        if predicate():
            return
    raise RuntimeError("Timed out waiting for UI state.")


def save_screenshot(window: QtWidgets.QWidget, name: str) -> str:
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCREENSHOT_ROOT / name
    window.grab().save(str(path))
    return str(path)


def open_run(window: HeliosParseViewMainWindow, path: Path) -> float:
    started = time.perf_counter()
    window._open_path(path)
    wait_until(
        lambda: window.viewer_controller.has_loaded_run()
        and str(window.viewer_controller.current_run_context().path) == str(path),
        180.0,
    )
    return time.perf_counter() - started


def wait_for_legacy_ready(window: HeliosParseViewMainWindow, *, timeout_s: float = 180.0) -> float:
    derived = window.derived_controller.widget()
    started = time.perf_counter()
    wait_until(lambda: derived._current_result is not None and not window.derived_controller._controller._busy, timeout_s)
    return time.perf_counter() - started


def wait_for_wavefront_ready(window: HeliosParseViewMainWindow, *, timeout_s: float = 240.0) -> float:
    derived = window.derived_controller.widget()
    started = time.perf_counter()
    wait_until(
        lambda: derived._current_result is not None
        and derived._current_result.wave_tracking is not None
        and not window.derived_controller._controller._busy,
        timeout_s,
    )
    return time.perf_counter() - started


def main() -> None:
    configure_test_settings()
    SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    report: dict[str, object] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    window = HeliosParseViewMainWindow()
    window.show()
    process_events(100)
    derived = window.derived_controller.widget()

    report["simple_open_s"] = open_run(window, SIMPLE_HDF5)
    report["simple_viewer_screenshot"] = save_screenshot(window, "simple_viewer_open.png")

    window._set_mode("derived")
    report["simple_legacy_derived_s"] = wait_for_legacy_ready(window, timeout_s=60.0)
    report["simple_derived_screenshot"] = save_screenshot(window, "simple_derived_shock.png")

    derived.result_tabs.setCurrentWidget(derived.wavefront_tab)
    process_events(50)
    if window.derived_controller._controller._busy:
        report["simple_wavefront_busy_screenshot"] = save_screenshot(window, "simple_wavefront_busy.png")
    report["simple_wavefront_wall_s"] = wait_for_wavefront_ready(window, timeout_s=60.0)
    report["simple_wavefront_screenshot"] = save_screenshot(window, "simple_wavefront_primary.png")

    report["complex_open_s"] = open_run(window, COMPLEX_HDF5)
    report["complex_viewer_screenshot"] = save_screenshot(window, "complex_viewer_open.png")

    window._set_mode("derived")
    report["complex_legacy_derived_s"] = wait_for_legacy_ready(window, timeout_s=60.0)
    report["complex_derived_screenshot"] = save_screenshot(window, "complex_derived_shock.png")

    derived.result_tabs.setCurrentWidget(derived.wavefront_tab)
    wait_until(lambda: window.derived_controller._controller._busy and derived.activity_progress.isVisible(), timeout_s=30.0)
    report["complex_wavefront_busy_screenshot"] = save_screenshot(window, "complex_wavefront_busy.png")
    report["complex_wavefront_first_wall_s"] = wait_for_wavefront_ready(window, timeout_s=240.0)
    report["complex_wavefront_primary_screenshot"] = save_screenshot(window, "complex_wavefront_primary.png")

    derived.wavefront_display_combo.setCurrentIndex(derived.wavefront_display_combo.findData("events"))
    process_events(120)
    report["complex_wavefront_events_screenshot"] = save_screenshot(window, "complex_wavefront_events.png")

    derived.result_tabs.setCurrentWidget(derived.transmission_tab)
    process_events(120)
    report["complex_transmission_screenshot"] = save_screenshot(window, "complex_transmission.png")

    context = window.viewer_controller.current_run_context()
    browse_samples: list[float] = []
    for snapshot_index in (
        max(0, context.snapshot_index - 3),
        min(context.n_snapshots - 1, context.snapshot_index + 2),
        min(context.n_snapshots - 1, context.snapshot_index + 8),
    ):
        started = time.perf_counter()
        window.global_snapshot_spin.setValue(int(snapshot_index))
        wait_until(
            lambda expected=int(snapshot_index): (
                window.viewer_controller.current_run_context().snapshot_index == expected
                and not window.derived_controller._controller._busy
            ),
            timeout_s=60.0,
        )
        browse_samples.append(time.perf_counter() - started)
    report["complex_snapshot_browse_s"] = browse_samples

    metrics_before = snapshot_metrics()
    wavefront_timer_before = metrics_before.get("timers", {}).get("derived.compute.wavefront")
    count_before = None if wavefront_timer_before is None else int(wavefront_timer_before.count)
    derived.result_tabs.setCurrentWidget(derived.shock_tab)
    process_events(80)
    started = time.perf_counter()
    derived.result_tabs.setCurrentWidget(derived.wavefront_tab)
    process_events(200)
    report["complex_wavefront_reopen_wall_s"] = time.perf_counter() - started
    metrics_after = snapshot_metrics()
    wavefront_timer_after = metrics_after.get("timers", {}).get("derived.compute.wavefront")
    report["complex_wavefront_timer_count_before"] = count_before
    report["complex_wavefront_timer_count_after"] = None if wavefront_timer_after is None else int(wavefront_timer_after.count)
    report["complex_wavefront_reopen_screenshot"] = save_screenshot(window, "complex_wavefront_reopen.png")

    timers = snapshot_metrics().get("timers", {})
    report["timer_summary_s"] = {
        name: float(metric.last_s)
        for name, metric in timers.items()
        if name in {
            "derived.compute.full",
            "derived.compute.shock",
            "derived.compute.wavefront",
            "derived.compute.wave_tracking",
            "derived.compute.interface_events",
            "derived.compute.snapshot_refresh",
        }
    }
    report["wavefront_summary"] = derived.wavefront_summary_label.text()
    report["wavefront_overview"] = derived.wavefront_overview_label.text()
    report["wavefront_metrics"] = derived.wavefront_metrics_label.text()
    report["general_performance"] = derived.performance_summary_label.text()
    report["event_rows"] = int(derived.wavefront_event_table.rowCount())
    report["branch_rows"] = int(derived.wavefront_branch_table.rowCount())
    report["notes_head"] = derived.wavefront_notes.toPlainText().splitlines()[:14]

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(REPORT_PATH)

    window.parser_controller.shutdown()
    window.viewer_controller.shutdown()
    window.derived_controller.shutdown()
    window.close()
    process_events(120)
    app.processEvents()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
