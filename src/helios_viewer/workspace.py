"""Embeddable root widget for the HELIOS viewer.

The scientific viewer logic still lives in ``helios_viewer.main_window``. This
widget exists so the unified shell can embed the viewer workspace directly
without nesting a full ``QMainWindow`` inside another ``QMainWindow``.
"""

from __future__ import annotations

from PySide6 import QtWidgets


class HeliosViewerWorkspace(QtWidgets.QWidget):
    """Root viewer workspace used by both standalone and embedded modes."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.root_layout = QtWidgets.QHBoxLayout(self)
        self.root_layout.setContentsMargins(8, 8, 8, 8)
        self.root_layout.setSpacing(8)
