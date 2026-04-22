"""Persistent shell-level session state for the unified Parse/View app.

Viewer visualization preferences stay in the dedicated viewer settings store.
This module only persists application-shell state such as recent files, active
mode, and parser-mode defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore


APP_ORGANIZATION = "HeliosViewer"
APP_NAME = "HELIOS Parse View"
MAX_RECENT_FILES = 10


@dataclass(slots=True)
class AppSessionState:
    """Small persisted state bundle for the unified shell."""

    current_mode: str = "parser"
    current_file: str = ""
    recent_files: list[str] | None = None
    last_log_directory: str = ""
    last_hdf5_directory: str = ""
    last_output_directory: str = ""
    parse_compression: str = "none"
    parse_overwrite: bool = True
    auto_open_after_parse: bool = True

    def __post_init__(self) -> None:
        if self.recent_files is None:
            self.recent_files = []


def _settings() -> QtCore.QSettings:
    return QtCore.QSettings(APP_ORGANIZATION, APP_NAME)


def default_session_state() -> AppSessionState:
    return AppSessionState()


def load_session_state() -> AppSessionState:
    store = _settings()
    defaults = default_session_state()
    recent_files = [str(value) for value in store.value("recent_files", defaults.recent_files or [], type=list)]
    return AppSessionState(
        current_mode=str(store.value("current_mode", defaults.current_mode)),
        current_file=str(store.value("current_file", defaults.current_file)),
        recent_files=recent_files,
        last_log_directory=str(store.value("last_log_directory", defaults.last_log_directory)),
        last_hdf5_directory=str(store.value("last_hdf5_directory", defaults.last_hdf5_directory)),
        last_output_directory=str(store.value("last_output_directory", defaults.last_output_directory)),
        parse_compression=str(store.value("parse_compression", defaults.parse_compression)),
        parse_overwrite=bool(store.value("parse_overwrite", defaults.parse_overwrite, type=bool)),
        auto_open_after_parse=bool(store.value("auto_open_after_parse", defaults.auto_open_after_parse, type=bool)),
    )


def save_session_state(state: AppSessionState) -> None:
    store = _settings()
    store.setValue("current_mode", state.current_mode)
    store.setValue("current_file", state.current_file)
    store.setValue("recent_files", list(state.recent_files or []))
    store.setValue("last_log_directory", state.last_log_directory)
    store.setValue("last_hdf5_directory", state.last_hdf5_directory)
    store.setValue("last_output_directory", state.last_output_directory)
    store.setValue("parse_compression", state.parse_compression)
    store.setValue("parse_overwrite", bool(state.parse_overwrite))
    store.setValue("auto_open_after_parse", bool(state.auto_open_after_parse))
    store.sync()


def reset_session_state() -> AppSessionState:
    store = _settings()
    store.clear()
    state = default_session_state()
    save_session_state(state)
    return state


def add_recent_file(state: AppSessionState, path: str | Path) -> None:
    resolved = str(Path(path))
    recent = [resolved]
    recent.extend(existing for existing in state.recent_files or [] if existing != resolved)
    state.recent_files = recent[:MAX_RECENT_FILES]
