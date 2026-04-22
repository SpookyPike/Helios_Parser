"""Thin shell bridge for the HELIOS Derived / Analysis workspace."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from helios.runtime import RunContext
from helios_analysis.controller import DerivedController as _DerivedController
from helios_analysis.workspace import HeliosDerivedWorkspace


class DerivedController(QtCore.QObject):
    """Embed the derived workspace in the shell without mixing responsibilities."""

    status_changed = QtCore.Signal(str)
    busy_changed = QtCore.Signal(bool)
    analysis_ready = QtCore.Signal(object)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = _DerivedController(self)
        self._controller.status_changed.connect(self.status_changed)
        self._controller.busy_changed.connect(self.busy_changed)
        self._controller.analysis_ready.connect(self.analysis_ready)

    def widget(self) -> HeliosDerivedWorkspace:
        return self._controller.widget()

    def set_theme_mode(self, mode: str) -> None:
        self._controller.set_theme_mode(mode)

    def set_display_settings(self, settings: object) -> None:
        self._controller.set_display_settings(settings)

    def set_default_profile_coordinate_mode(self, mode: str) -> None:
        self._controller.set_default_profile_coordinate_mode(mode)

    def set_active(self, active: bool) -> None:
        self._controller.set_active(active)

    def set_run_context(self, context: RunContext) -> None:
        self._controller.set_run_context(context)

    def current_run_context(self) -> RunContext:
        return self._controller.current_run_context()

    def shutdown(self) -> None:
        self._controller.shutdown()
