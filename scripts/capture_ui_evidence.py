"""Capture PNG evidence for the 2.0 UI changes plus an HTML contact sheet.

The assertions live in `tests/test_ui_interaction_perf.py`; this script exists
only for the visual half, because the app is native PySide6 with no web surface
and no automated tool can drive it. It renders each scenario offscreen, writes
`outputs/screenshots/perf2/<scenario>.png`, and writes `index.html` beside them
so the frames can be reviewed as one page.

Usage:
    QT_QPA_PLATFORM=offscreen python scripts/capture_ui_evidence.py

Nothing here asserts correctness. A blank or stale frame is the failure this
script exists to make visible to a human or to the vision agent.
"""

from __future__ import annotations

import html
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

from helios_viewer.main_window import HeliosViewerMainWindow  # noqa: E402

HDF5_ROOT = ROOT / "outputs" / "hdf5"
EVIDENCE_ROOT = ROOT / "outputs" / "screenshots" / "perf2"
FIXTURE = "Cu_0166_stabilized.h5"
WINDOW_SIZE = (1600, 1000)


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def process_events(delay_ms: int = 0) -> None:
    app = _app()
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    if delay_ms > 0:
        QtCore.QThread.msleep(delay_ms)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def wait_until(predicate, timeout_s: float = 60.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        process_events(20)
        if predicate():
            return
    raise RuntimeError("Timed out waiting for GUI state.")


def combo_set_data(combo: QtWidgets.QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        raise RuntimeError(f"Combo value {value!r} not found.")
    combo.setCurrentIndex(index)
    process_events(30)


def select_field(window: HeliosViewerMainWindow, field_name: str) -> None:
    for row in range(window.field_list.count()):
        if str(window.field_list.item(row).data(QtCore.Qt.UserRole)) == field_name:
            window.field_list.setCurrentRow(row)
            break
    else:
        raise RuntimeError(f"Field {field_name!r} not found.")
    wait_until(
        lambda: window.current_field_payload is not None and window.current_field_payload.field_name == field_name
    )


def _grab(window: QtWidgets.QWidget, name: str, captures: list[tuple[str, Path, str]]) -> Path:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_ROOT / f"{name}.png"
    window.grab().save(str(path))
    captures.append((name, path, ""))
    return path


def _contact_sheet(captures: list[tuple[str, Path, str]]) -> Path:
    rows = []
    for name, path, note in captures:
        rows.append(
            "<figure>"
            f'<img src="{html.escape(path.name)}" alt="{html.escape(name)}">'
            f"<figcaption><strong>{html.escape(name)}</strong>"
            + (f"<br><span>{html.escape(note)}</span>" if note else "")
            + "</figcaption></figure>"
        )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>HELIOS 2.0 UI evidence</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:24px;background:#14161a;color:#e6e8eb}"
        "h1{font-size:18px;font-weight:600}"
        "figure{margin:0 0 28px}"
        "img{max-width:100%;border:1px solid #2b2f36;border-radius:6px;display:block}"
        "figcaption{font-size:13px;color:#a8b0bb;margin-top:6px}"
        "</style></head><body>"
        f"<h1>HELIOS Parse / View 2.0 UI evidence ({len(captures)} frames)</h1>"
        + "".join(rows)
        + "</body></html>"
    )
    sheet = EVIDENCE_ROOT / "index.html"
    sheet.write_text(document, encoding="utf-8")
    return sheet


def main() -> int:
    _app()
    captures: list[tuple[str, Path, str]] = []
    window = HeliosViewerMainWindow()
    window.resize(*WINDOW_SIZE)
    window.show()
    try:
        window.load_file(HDF5_ROOT / FIXTURE)
        wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None)
        select_field(window, "density")
        process_events(200)
        _grab(window, "01_field_map_initial", captures)

        # F15: the colormap must still reach the map, the mesh and the colorbar
        # after the unchanged-name guard.
        combo_set_data(window.colormap_combo, "plasma")
        process_events(200)
        _grab(window, "02_colormap_plasma", captures)
        combo_set_data(window.colormap_combo, "viridis")
        process_events(200)
        _grab(window, "03_colormap_viridis", captures)

        # F16: both clipping branches must still render a usable image.
        combo_set_data(window.clip_mode_combo, "percentile")
        process_events(200)
        _grab(window, "04_clip_percentile", captures)
        combo_set_data(window.clip_mode_combo, "auto")
        process_events(200)
        _grab(window, "05_clip_auto", captures)

        # F22: after a slider move the plots must show the new snapshot.
        slider = window.snapshot_slider
        target = min(slider.maximum(), slider.value() + 40)
        slider.setValue(target)
        process_events(300)
        _grab(window, "06_snapshot_after_slider", captures)

        # Moving-mesh render path: the other colormap consumer.
        if window.map_coordinate_combo.findData("moving_radius") >= 0:
            combo_set_data(window.map_coordinate_combo, "moving_radius")
            wait_until(lambda: window.field_map_widget.current_render_mode == "mesh")
            process_events(200)
            _grab(window, "07_moving_radius_mesh", captures)
    finally:
        window.close()
        process_events(100)

    sheet = _contact_sheet(captures)
    for name, path, _ in captures:
        print(f"{name}: {path}")
    print(f"contact sheet: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())