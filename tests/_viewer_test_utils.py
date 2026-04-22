from __future__ import annotations

import os
import time
from pathlib import Path

import _test_bootstrap  # noqa: F401
from PySide6 import QtCore, QtWidgets


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
HDF5_ROOT = ROOT / "outputs" / "hdf5"
TEST_SETTINGS_ROOT = ROOT / "outputs" / "test_settings"
EXPECTED = {
    "5Fe+4.9TW+light_stabilized.h5": {"zones": 500, "snapshots": 8, "regions": 1, "materials": 1},
    "Cu_0166_stabilized.h5": {"zones": 300, "snapshots": 461, "regions": 1, "materials": 1},
    "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5": {"zones": 1300, "snapshots": 701, "regions": 3, "materials": 2},
}


def get_app() -> QtWidgets.QApplication:
    TEST_SETTINGS_ROOT.mkdir(parents=True, exist_ok=True)
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(TEST_SETTINGS_ROOT))
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def reset_test_settings() -> None:
    app = get_app()
    del app
    store = QtCore.QSettings("HeliosViewer", "HELIOS HDF5 Quick Look")
    store.clear()
    store.sync()


def process_events(delay_ms: int = 0) -> None:
    app = get_app()
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    if delay_ms > 0:
        QtCore.QThread.msleep(delay_ms)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def wait_until(predicate, timeout_s: float = 20.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        process_events(20)
        if predicate():
            return
    raise AssertionError("Timed out waiting for viewer state.")


def find_row_by_data(widget: QtWidgets.QListWidget, value: str) -> int:
    for index in range(widget.count()):
        if str(widget.item(index).data(QtCore.Qt.UserRole)) == value:
            return index
    return -1


def set_checked_values(widget: QtWidgets.QListWidget, values: set[int]) -> None:
    widget.blockSignals(True)
    for index in range(widget.count()):
        item = widget.item(index)
        item_value = int(item.data(QtCore.Qt.UserRole))
        item.setCheckState(QtCore.Qt.Checked if item_value in values else QtCore.Qt.Unchecked)
    widget.blockSignals(False)


def combo_set_data(combo: QtWidgets.QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        raise AssertionError(f"Combo value {value!r} not found.")
    combo.setCurrentIndex(index)
    process_events()
