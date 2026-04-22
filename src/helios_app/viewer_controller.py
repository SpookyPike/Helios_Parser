"""Thin bridge that embeds the existing HELIOS HDF5 viewer in the shell.

The embedded viewer remains the production scientific viewer. This controller
only hides standalone chrome, forwards a few top-level shell actions, and
re-emits viewer lifecycle/status signals for the unified application.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from helios_viewer.main_window import HeliosViewerMainWindow
from helios_viewer.workspace import HeliosViewerWorkspace


class ViewerController(QtCore.QObject):
    """Embed and proxy the existing viewer without changing its internals."""

    run_loaded = QtCore.Signal(object)
    context_changed = QtCore.Signal(object)
    field_visualized = QtCore.Signal(str)
    status_changed = QtCore.Signal(str)
    busy_changed = QtCore.Signal(bool)
    settings_changed = QtCore.Signal(object)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.window = HeliosViewerMainWindow()
        self.window.set_embedded_mode(True)
        self.window.set_external_snapshot_controls(True)
        self._workspace = self.window.take_workspace_widget()
        self.window.run_loaded.connect(self.run_loaded)
        self.window.context_changed.connect(self.context_changed)
        self.window.field_visualized.connect(self.field_visualized)
        self.window.settings_changed.connect(self.settings_changed)
        self.window.controller.status_changed.connect(self.status_changed)
        self.window.controller.busy_changed.connect(self.busy_changed)

    def widget(self) -> HeliosViewerWorkspace:
        return self._workspace

    def load_file(self, path: str | Path) -> None:
        self.window.load_file(path)

    def open_file_dialog(self) -> None:
        self.window.open_file_dialog()

    def export_current_view(self) -> None:
        self.window.export_current_view()

    def open_settings_dialog(self) -> None:
        self.window.open_settings_dialog()

    def reset_settings_to_defaults(self) -> None:
        self.window.reset_viewer_settings_to_defaults()

    def set_theme_mode(self, mode: str) -> None:
        self.window.set_theme_mode(mode)

    def theme_mode(self) -> str:
        return self.window.current_theme_mode()

    def current_time_unit(self) -> str:
        return self.window.current_time_unit()

    def current_viewer_settings(self):
        return self.window.current_viewer_settings()

    def default_profile_coordinate_mode(self) -> str:
        return self.window.default_profile_coordinate_mode()

    def has_loaded_run(self) -> bool:
        return self.window.has_loaded_run()

    def can_export_current_view(self) -> bool:
        return self.window.current_field_payload is not None

    def current_run_context(self):
        return self.window.run_context

    def active_snapshot_index(self) -> int:
        return self.window.active_snapshot_index()

    def set_active_snapshot_index(self, index: int) -> None:
        self.window.set_active_snapshot_index(index)

    def nearest_snapshot_index_for_display_time(self, display_time: float) -> int:
        return self.window.nearest_snapshot_index_for_display_time(display_time)

    def display_time_for_snapshot(self, snapshot_index: int) -> float:
        return self.window.display_time_for_snapshot(snapshot_index)

    def refresh_embedded_view(self) -> None:
        self.window.refresh_embedded_view()

    def shutdown(self) -> None:
        self.window.close()
