"""Unified HELIOS Parse/View application shell.

This module is the thin orchestration layer introduced in Phase 3. It does not
reimplement parser or viewer science logic; it coordinates existing components
into one desktop workflow:

    .log -> preview -> parse -> HDF5 -> viewer -> export

The shell owns mode switching, recent files, parser-mode controls, and top-level
menus/toolbars. Scientific parsing stays in ``helios_parser`` and scientific
visualization stays in ``helios_viewer``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import sys
from pathlib import Path

import h5py
from PySide6 import QtCore, QtGui, QtWidgets

from helios_viewer.icon import apply_application_icon
from helios_viewer.slider import apply_absolute_click_slider_behavior
from helios_viewer.style import THEME_MODES, build_mono_font, configure_application, configure_combo_box_interaction

from .derived_controller import DerivedController
from .parser_controller import ParsePreviewPayload, ParseProgressPayload, ParseResultPayload, ParserController
from .release import AUTHOR_AFFILIATION, AUTHOR_NAME, RELEASE_DATE, RELEASE_VERSION
from .session_state import AppSessionState, add_recent_file, load_session_state, save_session_state
from .viewer_controller import ViewerController


MODE_PARSER = "parser"
MODE_VIEWER = "viewer"
MODE_DERIVED = "derived"
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ShellModeSpec:
    """Registered top-level shell mode.

    Mode metadata stays centralized so Parser, Viewer, and Derived / Analysis
    workspaces can coexist without scattering shell routing logic.
    """

    mode_id: str
    page: QtWidgets.QWidget
    action: QtGui.QAction
    supports_export: bool = False


def _human_bytes(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    index = 0
    while value >= 1024.0 and index < len(units) - 1:
        value /= 1024.0
        index += 1
    return f"{value:.2f} {units[index]}"


def _configure_logging() -> None:
    level_name = os.environ.get("HELIOS_ANALYZER_LOG_LEVEL")
    if not level_name and os.environ.get("HELIOS_ANALYZER_DEBUG"):
        level_name = "DEBUG"
    if not level_name:
        return
    try:
        level = getattr(logging, str(level_name).upper())
    except AttributeError:
        level = logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root.setLevel(level)


def _pretty_field(name: str) -> str:
    return name.replace("_", " ").capitalize()


class HeliosParseViewMainWindow(QtWidgets.QMainWindow):
    """Top-level Parse/View shell for HELIOS Analyzer.

    The window exposes three user-facing modes:

    - Parser Mode for fast log preview and HDF5 generation
    - Viewer Mode for the existing mature HDF5 viewer
    - Derived / Analysis for experiment-facing quick-look diagnostics

    The class deliberately keeps parser/viewer internals separate and only
    manages workflow routing, session state, and shell-level actions.
    """

    def __init__(self) -> None:
        configure_application(QtWidgets.QApplication.instance())
        super().__init__()
        apply_application_icon(self)
        self.setWindowTitle("HELIOS Parse / View")
        self.resize(1860, 1120)

        self.session_state = load_session_state()
        self.parser_controller = ParserController(self)
        self.viewer_controller = ViewerController(self)
        self.derived_controller = DerivedController(self)
        self._mode_specs: dict[str, ShellModeSpec] = {}
        self.current_preview: ParsePreviewPayload | None = None
        self.last_parse_result: ParseResultPayload | None = None
        self._parser_busy = False
        self._viewer_busy = False
        self._derived_busy = False
        self._syncing_global_snapshot_controls = False
        self._pending_global_snapshot_index: int | None = None
        self._global_snapshot_slider_drag_active = False
        self._breakout_snapshot_index: int | None = None
        self._breakout_run_identity: str | None = None
        self._breakout_reason = "Breakout is unavailable until a derived Shock result is ready."
        self._pending_auto_open_path: Path | None = None
        self._pending_auto_open_attempts = 0
        self._auto_open_timer = QtCore.QTimer(self)
        self._auto_open_timer.setSingleShot(True)
        self._auto_open_timer.timeout.connect(self._attempt_pending_auto_open)
        self._snapshot_apply_timer = QtCore.QTimer(self)
        self._snapshot_apply_timer.setSingleShot(True)
        self._snapshot_apply_timer.timeout.connect(self._flush_pending_global_snapshot)

        self.parser_controller.preview_ready.connect(self._on_preview_ready)
        self.parser_controller.parse_succeeded.connect(self._on_parse_succeeded)
        self.parser_controller.progress_changed.connect(self._on_parser_progress)
        self.parser_controller.status_changed.connect(self._on_parser_status)
        self.parser_controller.error_occurred.connect(self._on_parser_error)
        self.parser_controller.busy_changed.connect(self._on_parser_busy_changed)

        self.viewer_controller.run_loaded.connect(self._on_viewer_run_loaded)
        self.viewer_controller.context_changed.connect(self._on_viewer_context_changed)
        self.viewer_controller.context_changed.connect(self.derived_controller.set_run_context)
        self.viewer_controller.field_visualized.connect(self._on_viewer_field_visualized)
        self.viewer_controller.status_changed.connect(self._on_viewer_status)
        self.viewer_controller.busy_changed.connect(self._on_viewer_busy_changed)
        self.viewer_controller.settings_changed.connect(self._on_viewer_settings_changed)
        self.derived_controller.status_changed.connect(self._on_derived_status)
        self.derived_controller.busy_changed.connect(self._on_derived_busy_changed)
        self.derived_controller.analysis_ready.connect(self._on_derived_analysis_ready)

        self._build_ui()
        for combo in self.findChildren(QtWidgets.QComboBox):
            configure_combo_box_interaction(combo)
        self._register_modes()
        self._restore_session_state()
        self._sync_theme_actions()
        self.derived_controller.set_theme_mode(self.viewer_controller.theme_mode())
        self.derived_controller.set_display_settings(self.viewer_controller.current_viewer_settings())
        self.derived_controller.set_default_profile_coordinate_mode(self.viewer_controller.default_profile_coordinate_mode())
        self._rebuild_recent_files_menu()
        self._set_mode(
            self.session_state.current_mode
            if self.session_state.current_mode in {MODE_PARSER, MODE_VIEWER, MODE_DERIVED}
            else MODE_PARSER
        )
        self._update_action_state()

    def _build_ui(self) -> None:
        self._build_actions()
        self._build_menus()
        self._build_toolbar()

        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self.mode_stack = QtWidgets.QStackedWidget()
        central_layout.addWidget(self.mode_stack, 1)
        central_layout.addWidget(self._build_global_snapshot_bar())
        self.setCentralWidget(central)

        self.parser_page = self._build_parser_page()
        self.viewer_page = self.viewer_controller.widget()
        self.derived_page = self.derived_controller.widget()
        self.mode_stack.addWidget(self.parser_page)
        self.mode_stack.addWidget(self.viewer_page)
        self.mode_stack.addWidget(self.derived_page)

        self.status_label = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)
        self.busy_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.busy_label)

    def _register_modes(self) -> None:
        self._mode_specs = {
            MODE_PARSER: ShellModeSpec(MODE_PARSER, self.parser_page, self.parser_mode_action, supports_export=False),
            MODE_VIEWER: ShellModeSpec(MODE_VIEWER, self.viewer_page, self.viewer_mode_action, supports_export=True),
            MODE_DERIVED: ShellModeSpec(MODE_DERIVED, self.derived_page, self.derived_mode_action, supports_export=False),
        }

    def available_mode_ids(self) -> tuple[str, ...]:
        """Return the registered top-level shell modes.

        This keeps future mode addition explicit instead of spreading Parser /
        Viewer assumptions through the shell.
        """

        return tuple(self._mode_specs)

    def _current_mode_id(self) -> str:
        current_page = self.mode_stack.currentWidget()
        for mode_id, spec in self._mode_specs.items():
            if current_page is spec.page:
                return mode_id
        return MODE_PARSER

    def _build_actions(self) -> None:
        self.open_log_action = QtGui.QAction("Open Log...", self)
        self.open_log_action.triggered.connect(self.open_log_dialog)
        self.open_hdf5_action = QtGui.QAction("Open HDF5...", self)
        self.open_hdf5_action.triggered.connect(self.open_hdf5_dialog)
        self.export_action = QtGui.QAction("Export", self)
        self.export_action.triggered.connect(self._export_current_view)
        self.exit_action = QtGui.QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)

        self.parse_action = QtGui.QAction("Parse", self)
        self.parse_action.triggered.connect(self._start_parse_from_controls)
        self.open_viewer_action = QtGui.QAction("Open Viewer", self)
        self.open_viewer_action.triggered.connect(self._open_last_result_in_viewer)

        self.parser_mode_action = QtGui.QAction("Parser Mode", self, checkable=True)
        self.viewer_mode_action = QtGui.QAction("Viewer Mode", self, checkable=True)
        self.derived_mode_action = QtGui.QAction("Derived / Analysis", self, checkable=True)
        mode_group = QtGui.QActionGroup(self)
        mode_group.setExclusive(True)
        mode_group.addAction(self.parser_mode_action)
        mode_group.addAction(self.viewer_mode_action)
        mode_group.addAction(self.derived_mode_action)
        self.parser_mode_action.triggered.connect(lambda checked=False: self._set_mode(MODE_PARSER))
        self.viewer_mode_action.triggered.connect(lambda checked=False: self._set_mode(MODE_VIEWER))
        self.derived_mode_action.triggered.connect(lambda checked=False: self._set_mode(MODE_DERIVED))

        self.theme_actions: dict[str, QtGui.QAction] = {}
        self.theme_group = QtGui.QActionGroup(self)
        self.theme_group.setExclusive(True)
        for label, mode in (("Light", "light"), ("Dark", "dark"), ("System", "system")):
            action = QtGui.QAction(label, self, checkable=True)
            action.triggered.connect(lambda checked=False, selected_mode=mode: self._set_theme_mode(selected_mode))
            self.theme_group.addAction(action)
            self.theme_actions[mode] = action

        self.units_action = QtGui.QAction("Display Units...", self)
        self.units_action.triggered.connect(self._open_viewer_settings)
        self.settings_action = QtGui.QAction("Settings...", self)
        self.settings_action.triggered.connect(self._open_viewer_settings)
        self.reset_settings_action = QtGui.QAction("Reset Viewer Settings", self)
        self.reset_settings_action.triggered.connect(self._reset_viewer_settings)

        self.about_action = QtGui.QAction("About", self)
        self.about_action.triggered.connect(self._show_about_dialog)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_log_action)
        file_menu.addAction(self.open_hdf5_action)
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        mode_menu = self.menuBar().addMenu("&Mode")
        mode_menu.addAction(self.parser_mode_action)
        mode_menu.addAction(self.viewer_mode_action)
        mode_menu.addAction(self.derived_mode_action)

        view_menu = self.menuBar().addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        for mode in ("light", "dark", "system"):
            theme_menu.addAction(self.theme_actions[mode])
        units_menu = view_menu.addMenu("&Units")
        units_menu.addAction(self.units_action)
        view_menu.addAction(self.settings_action)
        view_menu.addAction(self.reset_settings_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.about_action)

    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Application", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        toolbar.addAction(self.open_log_action)
        toolbar.addAction(self.open_hdf5_action)
        toolbar.addSeparator()
        toolbar.addAction(self.parser_mode_action)
        toolbar.addAction(self.viewer_mode_action)
        toolbar.addAction(self.derived_mode_action)
        toolbar.addSeparator()
        toolbar.addAction(self.parse_action)
        toolbar.addAction(self.open_viewer_action)
        toolbar.addSeparator()
        toolbar.addAction(self.export_action)
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

    def _build_global_snapshot_bar(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(QtWidgets.QLabel("Active snapshot"))

        self.global_snapshot_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.global_snapshot_slider.setMinimumHeight(24)
        apply_absolute_click_slider_behavior(self.global_snapshot_slider)
        self.global_snapshot_slider.setTracking(False)
        self.global_snapshot_slider.valueChanged.connect(self._on_global_snapshot_slider_changed)
        self.global_snapshot_slider.sliderMoved.connect(self._on_global_snapshot_slider_moved)
        self.global_snapshot_slider.sliderPressed.connect(self._on_global_snapshot_slider_pressed)
        self.global_snapshot_slider.sliderReleased.connect(self._on_global_snapshot_slider_released)
        layout.addWidget(self.global_snapshot_slider, 1)

        self.global_snapshot_spin = QtWidgets.QSpinBox()
        self.global_snapshot_spin.setMinimumHeight(24)
        self.global_snapshot_spin.setKeyboardTracking(False)
        self.global_snapshot_spin.valueChanged.connect(self._on_global_snapshot_spin_changed)
        layout.addWidget(self.global_snapshot_spin)

        self.global_time_spin = QtWidgets.QDoubleSpinBox()
        self.global_time_spin.setMinimumHeight(24)
        self.global_time_spin.setDecimals(6)
        self.global_time_spin.setRange(-1.0e18, 1.0e18)
        self.global_time_spin.setSingleStep(0.1)
        self.global_time_spin.setKeyboardTracking(False)
        self.global_time_spin.valueChanged.connect(self._on_global_time_changed)
        layout.addWidget(self.global_time_spin)

        self.jump_breakout_button = QtWidgets.QPushButton("Jump breakout")
        self.jump_breakout_button.clicked.connect(self._jump_to_breakout_snapshot)
        self.jump_breakout_button.setEnabled(False)
        self.jump_breakout_button.setToolTip(self._breakout_reason)
        layout.addWidget(self.jump_breakout_button)

        self.global_snapshot_label = QtWidgets.QLabel("No active run")
        layout.addWidget(self.global_snapshot_label)
        self.global_snapshot_widget = widget
        self._set_global_snapshot_controls_visible(False)
        return widget

    def _build_parser_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.log_file_label = QtWidgets.QLabel("Open a HELIOS .log file to preview its structure.")
        self.log_file_label.setWordWrap(True)
        self.log_file_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        left_layout.addWidget(self.log_file_label)

        summary_group = QtWidgets.QGroupBox("Parse Preview")
        summary_layout = QtWidgets.QVBoxLayout(summary_group)
        self.preview_summary = QtWidgets.QPlainTextEdit()
        self.preview_summary.setReadOnly(True)
        self.preview_summary.setFont(build_mono_font())
        summary_layout.addWidget(self.preview_summary)
        left_layout.addWidget(summary_group, 1)

        first_snapshot_group = QtWidgets.QGroupBox("First Snapshot / Time Grid")
        first_snapshot_layout = QtWidgets.QVBoxLayout(first_snapshot_group)
        self.preview_snapshot = QtWidgets.QPlainTextEdit()
        self.preview_snapshot.setReadOnly(True)
        self.preview_snapshot.setFont(build_mono_font())
        first_snapshot_layout.addWidget(self.preview_snapshot)
        left_layout.addWidget(first_snapshot_group, 1)

        fields_group = QtWidgets.QGroupBox("Detected Fields")
        fields_layout = QtWidgets.QVBoxLayout(fields_group)
        self.preview_field_list = QtWidgets.QListWidget()
        fields_layout.addWidget(self.preview_field_list)
        left_layout.addWidget(fields_group, 1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        options_group = QtWidgets.QGroupBox("Parse Options")
        options_layout = QtWidgets.QFormLayout(options_group)
        self.output_path_edit = QtWidgets.QLineEdit()
        browse_output_button = QtWidgets.QPushButton("Browse...")
        browse_output_button.clicked.connect(self._browse_output_path)
        output_widget = QtWidgets.QWidget()
        output_layout = QtWidgets.QHBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(6)
        output_layout.addWidget(self.output_path_edit, 1)
        output_layout.addWidget(browse_output_button)
        options_layout.addRow("Output HDF5", output_widget)

        self.compression_combo = QtWidgets.QComboBox()
        self.compression_combo.addItem("None", "none")
        self.compression_combo.addItem("gzip", "gzip")
        self.compression_combo.addItem("lzf", "lzf")
        self.compression_combo.currentIndexChanged.connect(self._persist_shell_preferences)
        options_layout.addRow("Compression", self.compression_combo)

        self.overwrite_checkbox = QtWidgets.QCheckBox("Overwrite existing output")
        self.overwrite_checkbox.toggled.connect(self._persist_shell_preferences)
        options_layout.addRow("", self.overwrite_checkbox)

        self.auto_open_checkbox = QtWidgets.QCheckBox("Open parsed HDF5 in Viewer automatically")
        self.auto_open_checkbox.toggled.connect(self._persist_shell_preferences)
        options_layout.addRow("", self.auto_open_checkbox)

        button_row = QtWidgets.QHBoxLayout()
        self.refresh_preview_button = QtWidgets.QPushButton("Refresh Preview")
        self.refresh_preview_button.clicked.connect(self._refresh_current_preview)
        self.parse_button = QtWidgets.QPushButton("Parse to HDF5")
        self.parse_button.clicked.connect(self._start_parse_from_controls)
        self.open_viewer_button = QtWidgets.QPushButton("Open in Viewer")
        self.open_viewer_button.clicked.connect(self._open_last_result_in_viewer)
        button_row.addWidget(self.refresh_preview_button)
        button_row.addWidget(self.parse_button)
        button_row.addWidget(self.open_viewer_button)
        options_layout.addRow("", self._wrap_layout(button_row))

        right_layout.addWidget(options_group)

        status_group = QtWidgets.QGroupBox("Parser Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        self.parse_status_label = QtWidgets.QLabel("Ready")
        self.parse_progress = QtWidgets.QProgressBar()
        self.parse_progress.setRange(0, 1000)
        self.parse_progress.setValue(0)
        self.parse_progress.setFormat("Idle")
        self.parse_log = QtWidgets.QPlainTextEdit()
        self.parse_log.setReadOnly(True)
        self.parse_log.setFont(build_mono_font())
        status_layout.addWidget(self.parse_status_label)
        status_layout.addWidget(self.parse_progress)
        status_layout.addWidget(self.parse_log, 1)
        right_layout.addWidget(status_group, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([980, 560])
        return page

    def _wrap_layout(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def _restore_session_state(self) -> None:
        if self.compression_combo.findData(self.session_state.parse_compression) >= 0:
            self.compression_combo.setCurrentIndex(self.compression_combo.findData(self.session_state.parse_compression))
        self.overwrite_checkbox.setChecked(bool(self.session_state.parse_overwrite))
        self.auto_open_checkbox.setChecked(bool(self.session_state.auto_open_after_parse))

    def _set_mode(self, mode: str) -> None:
        normalized = mode if mode in self._mode_specs else MODE_PARSER
        spec = self._mode_specs.get(normalized, self._mode_specs[MODE_PARSER])
        self.mode_stack.setCurrentWidget(spec.page)
        self.session_state.current_mode = normalized
        save_session_state(self.session_state)
        for mode_id, registered in self._mode_specs.items():
            registered.action.setChecked(mode_id == normalized)
        self.derived_controller.set_active(normalized == MODE_DERIVED)
        if normalized == MODE_VIEWER and self.viewer_controller.has_loaded_run():
            self.viewer_controller.refresh_embedded_view()
        self._update_action_state()

    def _set_global_snapshot_controls_visible(self, visible: bool) -> None:
        self.global_snapshot_widget.setVisible(bool(visible))

    def _active_run_identity(self) -> str | None:
        if not self.viewer_controller.has_loaded_run():
            return None
        context = self.viewer_controller.current_run_context()
        path = getattr(context, "path", None)
        if path is None:
            return None
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(Path(path))

    def _update_breakout_button_state(self) -> None:
        active_run_identity = self._active_run_identity()
        enabled = (
            self._breakout_snapshot_index is not None
            and self._breakout_run_identity is not None
            and self._breakout_run_identity == active_run_identity
            and not self._derived_busy
        )
        self.jump_breakout_button.setEnabled(bool(enabled))
        if self._derived_busy:
            self.jump_breakout_button.setToolTip("Breakout navigation is updating for the current derived analysis.")
        elif enabled:
            self.jump_breakout_button.setToolTip(f"Jump to breakout at snapshot {int(self._breakout_snapshot_index)}.")
        else:
            self.jump_breakout_button.setToolTip(self._breakout_reason)

    def _set_breakout_target(
        self,
        snapshot_index: int | None,
        *,
        run_identity: str | None = None,
        reason: str,
    ) -> None:
        self._breakout_snapshot_index = None if snapshot_index is None else int(snapshot_index)
        self._breakout_run_identity = None if snapshot_index is None else str(run_identity or "")
        self._breakout_reason = str(reason)
        self._update_breakout_button_state()

    def _update_global_snapshot_controls(self, context) -> None:
        has_run = bool(getattr(context, "has_run", False))
        self._set_global_snapshot_controls_visible(has_run)
        if not has_run:
            self._pending_global_snapshot_index = None
            self._snapshot_apply_timer.stop()
            self._global_snapshot_slider_drag_active = False
            self._set_breakout_target(None, reason="Breakout is unavailable because no run is loaded.")
            self.global_snapshot_label.setText("No active run")
            return
        n_snapshots = max(1, int(context.n_snapshots))
        snapshot_index = int(context.snapshot_index)
        display_time = self.viewer_controller.display_time_for_snapshot(snapshot_index)
        time_unit = self.viewer_controller.current_time_unit()
        self._syncing_global_snapshot_controls = True
        try:
            self.global_snapshot_slider.setRange(0, n_snapshots - 1)
            self.global_snapshot_spin.setRange(0, n_snapshots - 1)
            self.global_snapshot_slider.setValue(snapshot_index)
            self.global_snapshot_spin.setValue(snapshot_index)
            self.global_time_spin.setSuffix(f" {time_unit}")
            if context.time_values.size:
                low_time = self.viewer_controller.display_time_for_snapshot(0)
                high_time = self.viewer_controller.display_time_for_snapshot(n_snapshots - 1)
                self.global_time_spin.setRange(min(low_time, high_time), max(low_time, high_time))
                if int(context.time_values.size) > 1:
                    spacing = abs(
                        self.viewer_controller.display_time_for_snapshot(1)
                        - self.viewer_controller.display_time_for_snapshot(0)
                    )
                    self.global_time_spin.setSingleStep(max(spacing, 1.0e-6))
            self.global_time_spin.setValue(display_time)
        finally:
            self._syncing_global_snapshot_controls = False
        if context.time_values.size:
            self.global_snapshot_label.setText(f"Snapshot {snapshot_index} | t = {display_time:.6g} {time_unit}")
        else:
            self.global_snapshot_label.setText(f"Snapshot {snapshot_index}")
        self._update_breakout_button_state()

    def _apply_global_snapshot_index(self, snapshot_index: int) -> None:
        if self._syncing_global_snapshot_controls or not self.viewer_controller.has_loaded_run():
            return
        clamped = max(0, min(int(snapshot_index), int(self.viewer_controller.current_run_context().n_snapshots) - 1))
        if clamped == self.viewer_controller.active_snapshot_index():
            self._update_global_snapshot_controls(self.viewer_controller.current_run_context())
            return
        self.viewer_controller.set_active_snapshot_index(clamped)

    def _preview_global_snapshot_index(self, snapshot_index: int, *, pending: bool, sync_slider: bool = True) -> None:
        if not self.viewer_controller.has_loaded_run():
            return
        context = self.viewer_controller.current_run_context()
        n_snapshots = max(1, int(context.n_snapshots))
        clamped = max(0, min(int(snapshot_index), n_snapshots - 1))
        display_time = self.viewer_controller.display_time_for_snapshot(clamped)
        time_unit = self.viewer_controller.current_time_unit()
        self._syncing_global_snapshot_controls = True
        try:
            if sync_slider:
                self.global_snapshot_slider.setValue(clamped)
            self.global_snapshot_spin.setValue(clamped)
            self.global_time_spin.setValue(display_time)
        finally:
            self._syncing_global_snapshot_controls = False
        if pending:
            self.global_snapshot_label.setText(
                f"Snapshot {clamped} | t = {display_time:.6g} {time_unit} | updating..."
            )
        else:
            self.global_snapshot_label.setText(f"Snapshot {clamped} | t = {display_time:.6g} {time_unit}")

    def _request_global_snapshot_index(self, snapshot_index: int, *, immediate: bool = False) -> None:
        if not self.viewer_controller.has_loaded_run():
            return
        context = self.viewer_controller.current_run_context()
        clamped = max(0, min(int(snapshot_index), int(context.n_snapshots) - 1))
        self._pending_global_snapshot_index = clamped
        self._preview_global_snapshot_index(clamped, pending=True)
        self._snapshot_apply_timer.start(0 if immediate else 75)

    def _flush_pending_global_snapshot(self) -> None:
        pending = self._pending_global_snapshot_index
        self._pending_global_snapshot_index = None
        if pending is None:
            return
        self._apply_global_snapshot_index(int(pending))

    def _on_global_snapshot_slider_changed(self, value: int) -> None:
        if self._global_snapshot_slider_drag_active:
            return
        self._request_global_snapshot_index(int(value))

    def _on_global_snapshot_slider_moved(self, value: int) -> None:
        self._preview_global_snapshot_index(int(value), pending=True, sync_slider=False)

    def _on_global_snapshot_slider_pressed(self) -> None:
        self._global_snapshot_slider_drag_active = True

    def _on_global_snapshot_slider_released(self) -> None:
        self._global_snapshot_slider_drag_active = False
        self._request_global_snapshot_index(int(self.global_snapshot_slider.value()), immediate=True)

    def _on_global_snapshot_spin_changed(self, value: int) -> None:
        self._request_global_snapshot_index(int(value))

    def _on_global_time_changed(self, value: float) -> None:
        if self._syncing_global_snapshot_controls or not self.viewer_controller.has_loaded_run():
            return
        context = self.viewer_controller.current_run_context()
        if not context.time_values.size:
            return
        snapshot_index = self.viewer_controller.nearest_snapshot_index_for_display_time(float(value))
        self._request_global_snapshot_index(int(snapshot_index))

    def _jump_to_breakout_snapshot(self) -> None:
        if self._breakout_snapshot_index is None:
            return
        self._request_global_snapshot_index(self._breakout_snapshot_index, immediate=True)

    def _persist_shell_preferences(self) -> None:
        self.session_state.parse_compression = str(self.compression_combo.currentData())
        self.session_state.parse_overwrite = bool(self.overwrite_checkbox.isChecked())
        self.session_state.auto_open_after_parse = bool(self.auto_open_checkbox.isChecked())
        save_session_state(self.session_state)

    def _rebuild_recent_files_menu(self) -> None:
        self.recent_files_menu.clear()
        recent_files = [path for path in self.session_state.recent_files or [] if Path(path).exists()]
        self.session_state.recent_files = recent_files
        if not recent_files:
            action = self.recent_files_menu.addAction("No recent files")
            action.setEnabled(False)
            save_session_state(self.session_state)
            return
        for path_text in recent_files:
            path = Path(path_text)
            action = self.recent_files_menu.addAction(path.name)
            action.setToolTip(str(path))
            action.triggered.connect(lambda checked=False, selected=path: self._open_path(selected))

    def _sync_theme_actions(self) -> None:
        mode = self.viewer_controller.theme_mode()
        if mode not in THEME_MODES:
            mode = "system"
        action = self.theme_actions.get(mode)
        if action is not None:
            action.setChecked(True)

    def _set_theme_mode(self, mode: str) -> None:
        self.viewer_controller.set_theme_mode(mode)
        self.derived_controller.set_theme_mode(mode)
        self._sync_theme_actions()

    def _open_viewer_settings(self) -> None:
        self.viewer_controller.open_settings_dialog()
        self._sync_theme_actions()

    def _reset_viewer_settings(self) -> None:
        self.viewer_controller.reset_settings_to_defaults()
        self._sync_theme_actions()

    def _on_viewer_settings_changed(self, settings: object) -> None:
        self._sync_theme_actions()
        self.derived_controller.set_theme_mode(self.viewer_controller.theme_mode())
        self.derived_controller.set_display_settings(settings)
        self.derived_controller.set_default_profile_coordinate_mode(self.viewer_controller.default_profile_coordinate_mode())
        self._update_global_snapshot_controls(self.viewer_controller.current_run_context())

    def _show_about_dialog(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About HELIOS Parse / View",
            f"HELIOS Parse / View {RELEASE_VERSION}\n"
            f"Release date: {RELEASE_DATE}\n"
            f"Code developed by {AUTHOR_NAME} at {AUTHOR_AFFILIATION}.\n\n"
            "Parser Mode previews and converts HELIOS .log files to stabilized HDF5.\n"
            "Viewer Mode opens stabilized HDF5 files with the production scientific viewer.\n"
            "Derived / Analysis adds quick-look Shock diagnostics plus lazy advanced WaveFront, interface-event, and Preheat analysis.",
        )

    def open_log_dialog(self) -> None:
        start_dir = self.session_state.last_log_directory or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open HELIOS Log", start_dir, "HELIOS Log (*.log)")
        if path:
            self._open_path(Path(path))

    def open_hdf5_dialog(self) -> None:
        start_dir = self.session_state.last_hdf5_directory or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open HELIOS HDF5", start_dir, "HELIOS HDF5 (*.h5 *.hdf5)")
        if path:
            self._open_path(Path(path))

    def _open_path(self, path: str | Path) -> None:
        """Route a file to Parser Mode or Viewer Mode from its extension."""
        resolved = Path(path)
        suffix = resolved.suffix.lower()
        if suffix == ".log":
            self._open_log_preview(resolved)
            return
        if suffix in {".h5", ".hdf5"}:
            self._open_hdf5(resolved)
            return
        QtWidgets.QMessageBox.warning(self, "Unsupported File", f"Unsupported file type: {resolved.suffix}")

    def _open_log_preview(self, path: Path) -> None:
        """Open a HELIOS log in Parser Mode and request a fast preview."""
        LOGGER.info("Routing log %s to Parser Mode", path)
        self.session_state.current_file = str(path)
        self.session_state.last_log_directory = str(path.parent)
        add_recent_file(self.session_state, path)
        save_session_state(self.session_state)
        self._rebuild_recent_files_menu()
        self._set_mode(MODE_PARSER)
        self.log_file_label.setText(str(path))
        self.preview_summary.setPlainText("Previewing log structure...")
        self.preview_snapshot.clear()
        self.preview_field_list.clear()
        self.current_preview = None
        self.last_parse_result = None
        self.open_viewer_button.setEnabled(False)
        self.parse_button.setEnabled(False)
        self.parser_controller.preview_log(path)

    def _open_hdf5(self, path: Path) -> None:
        """Open an existing stabilized HDF5 file in Viewer Mode."""
        self._auto_open_timer.stop()
        self._pending_auto_open_path = None
        self._pending_auto_open_attempts = 0
        LOGGER.info("Routing HDF5 %s to Viewer Mode", path)
        self.session_state.current_file = str(path)
        self.session_state.last_hdf5_directory = str(path.parent)
        add_recent_file(self.session_state, path)
        save_session_state(self.session_state)
        self._rebuild_recent_files_menu()
        self._set_mode(MODE_VIEWER)
        self.viewer_controller.load_file(path)

    def _default_output_path(self, source: Path) -> Path:
        directory = Path(self.session_state.last_output_directory or str(source.parent))
        return directory / f"{source.stem}_stabilized.h5"

    def _browse_output_path(self) -> None:
        start_path = Path(self.output_path_edit.text().strip() or self.session_state.last_output_directory or str(Path.cwd()))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Stabilized HDF5", str(start_path), "HELIOS HDF5 (*.h5)")
        if path:
            resolved = Path(path)
            if resolved.suffix.lower() not in {".h5", ".hdf5"}:
                resolved = resolved.with_suffix(".h5")
            self.output_path_edit.setText(str(resolved))
            self.session_state.last_output_directory = str(resolved.parent)
            save_session_state(self.session_state)

    def _refresh_current_preview(self) -> None:
        if self.current_preview is None:
            current_file = self.session_state.current_file
            if current_file and Path(current_file).suffix.lower() == ".log":
                self._open_log_preview(Path(current_file))
            return
        self._open_log_preview(self.current_preview.source)

    def _start_parse_from_controls(self) -> None:
        """Launch a parse job from the current Parser Mode controls."""
        if self.current_preview is None:
            QtWidgets.QMessageBox.information(self, "No Preview", "Open a HELIOS .log file and wait for the preview before parsing.")
            return
        output_path = Path(self.output_path_edit.text().strip())
        if not output_path:
            output_path = self._default_output_path(self.current_preview.source)
            self.output_path_edit.setText(str(output_path))
        self.session_state.last_output_directory = str(output_path.parent)
        self._persist_shell_preferences()
        self.parse_button.setEnabled(False)
        self.open_viewer_button.setEnabled(False)
        self.parser_controller.parse_log(
            self.current_preview.source,
            output_path,
            compression=str(self.compression_combo.currentData()),
            overwrite=bool(self.overwrite_checkbox.isChecked()),
        )

    def _open_last_result_in_viewer(self) -> None:
        if self.last_parse_result is not None:
            self._open_hdf5(self.last_parse_result.output)
            return
        if self.viewer_controller.has_loaded_run():
            self._set_mode(MODE_VIEWER)

    def _schedule_auto_open(self, path: Path) -> None:
        self._pending_auto_open_path = Path(path)
        self._pending_auto_open_attempts = 20
        self._auto_open_timer.start(0)

    def _is_hdf5_ready_for_viewer(self, path: Path) -> bool:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        try:
            with h5py.File(path, "r") as handle:
                if not {"grid", "time", "fields"}.issubset(handle.keys()):
                    return False
                fields_group = handle["fields"]
                return len(fields_group.keys()) > 0
        except OSError:
            return False

    def _attempt_pending_auto_open(self) -> None:
        path = self._pending_auto_open_path
        if path is None:
            return
        if self._is_hdf5_ready_for_viewer(path):
            self._pending_auto_open_path = None
            self._pending_auto_open_attempts = 0
            self._open_hdf5(path)
            return
        self._pending_auto_open_attempts -= 1
        if self._pending_auto_open_attempts > 0:
            self._auto_open_timer.start(100)
            return
        self._pending_auto_open_path = None
        self._set_mode(MODE_PARSER)
        QtWidgets.QMessageBox.warning(
            self,
            "Auto-open failed",
            f"{path.name} was written, but the viewer could not reopen it safely. "
            "Open the file manually after checking the output.",
        )

    def _on_parser_status(self, message: str) -> None:
        self.parse_status_label.setText(message)
        self.status_label.setText(message)
        self.parse_log.appendPlainText(message)

    def _on_parser_progress(self, payload: ParseProgressPayload) -> None:
        fraction = max(0.0, min(1.0, float(payload.fraction)))
        percent = int(round(fraction * 100.0))
        self.parse_progress.setRange(0, 1000)
        self.parse_progress.setValue(int(round(fraction * 1000.0)))
        eta_text = f" | ETA {payload.eta_s:.1f} s" if payload.eta_s is not None else ""
        self.parse_progress.setFormat(f"{percent}% | {payload.stage}{eta_text}")
        self.parse_status_label.setText(payload.message)
        self.status_label.setText(payload.message)

    def _on_viewer_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_viewer_context_changed(self, context: object) -> None:
        self._update_global_snapshot_controls(context)

    def _on_derived_analysis_ready(self, result: object) -> None:
        breakout_time_s = getattr(getattr(result, "shock", None), "breakout_time_s", None)
        context = self.viewer_controller.current_run_context()
        active_run_identity = self._active_run_identity()
        result_path = getattr(result, "dataset_path", None)
        result_run_identity = None
        if result_path is not None:
            try:
                result_run_identity = str(Path(result_path).resolve())
            except Exception:
                result_run_identity = str(Path(result_path))
        if not getattr(context, "has_run", False) or active_run_identity is None:
            self._set_breakout_target(None, reason="Breakout is unavailable because no run is loaded.")
            return
        if result_run_identity is not None and result_run_identity != active_run_identity:
            self._set_breakout_target(
                None,
                reason="Breakout is unavailable because the current derived result does not match the active run.",
            )
            return
        if breakout_time_s is None:
            self._set_breakout_target(
                None,
                reason="Breakout is unavailable because the current Shock result does not contain a breakout time.",
            )
            return
        if int(context.time_values.size) <= 0:
            self._set_breakout_target(
                None,
                reason="Breakout is unavailable because the active run has no time axis.",
            )
            return
        try:
            nearest = min(
                range(int(context.time_values.size)),
                key=lambda index: abs(float(context.time_values[index]) - float(breakout_time_s)),
            )
        except Exception:
            self._set_breakout_target(
                None,
                reason="Breakout is unavailable because the breakout time could not be mapped onto the active run snapshots.",
            )
            return
        self._set_breakout_target(
            int(nearest),
            run_identity=active_run_identity,
            reason=f"Breakout mapped to snapshot {int(nearest)}.",
        )

    def _on_parser_error(self, message: str, details: str) -> None:
        self.parse_status_label.setText(message)
        self.parse_progress.setRange(0, 1000)
        self.parse_progress.setValue(0)
        self.parse_progress.setFormat("Failed")
        self.parse_log.appendPlainText(f"{message}\n{details}")
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle("Parser Error")
        box.setText(message)
        box.setDetailedText(details)
        box.exec()

    def _on_parser_busy_changed(self, busy: bool) -> None:
        self._parser_busy = busy
        if busy:
            self.parse_progress.setRange(0, 1000)
            self.parse_progress.setValue(0)
            self.parse_progress.setFormat("Starting...")
        else:
            self.parse_progress.setRange(0, 1000)
            self.parse_progress.setValue(1000)
            self.parse_progress.setFormat("Done")
        self._update_busy_indicator()
        self._update_action_state()

    def _on_viewer_busy_changed(self, busy: bool) -> None:
        self._viewer_busy = busy
        self._update_busy_indicator()
        self._update_action_state()

    def _on_derived_busy_changed(self, busy: bool) -> None:
        self._derived_busy = busy
        self._update_breakout_button_state()
        self._update_busy_indicator()
        self._update_action_state()

    def _update_busy_indicator(self) -> None:
        busy = self._parser_busy or self._viewer_busy or self._derived_busy
        self.busy_label.setText("Working..." if busy else "")

    def _on_preview_ready(self, payload: ParsePreviewPayload) -> None:
        """Populate Parser Mode once a background preview completes."""
        self.current_preview = payload
        self.log_file_label.setText(str(payload.source))
        self.preview_summary.setPlainText(self._format_preview_summary(payload))
        self.preview_snapshot.setPlainText(self._format_preview_snapshot(payload))
        self.preview_field_list.clear()
        for name in payload.fields:
            unit = payload.field_units.get(name, "")
            text = f"{_pretty_field(name)} [{unit}]" if unit else _pretty_field(name)
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, name)
            self.preview_field_list.addItem(item)
        self.output_path_edit.setText(str(self._default_output_path(payload.source)))
        self.parse_button.setEnabled(True)
        self.open_viewer_button.setEnabled(self.last_parse_result is not None and self.last_parse_result.output.exists())
        self._update_action_state()

    def _on_parse_succeeded(self, payload: ParseResultPayload) -> None:
        """Handle a finished parse job and optionally auto-open the result."""
        self.last_parse_result = payload
        self.session_state.last_output_directory = str(payload.output.parent)
        add_recent_file(self.session_state, payload.output)
        save_session_state(self.session_state)
        self._rebuild_recent_files_menu()
        self.parse_status_label.setText(f"Parsed {payload.source.name} -> {payload.output.name} in {payload.elapsed_s:.2f} s")
        self.parse_progress.setRange(0, 1000)
        self.parse_progress.setValue(1000)
        self.parse_progress.setFormat(f"100% | complete in {payload.elapsed_s:.2f} s")
        self.parse_log.appendPlainText(f"Parse finished in {payload.elapsed_s:.2f} s\nOutput: {payload.output}")
        self.parse_button.setEnabled(True)
        self.open_viewer_button.setEnabled(True)
        self._update_action_state()
        if self.auto_open_checkbox.isChecked():
            self._schedule_auto_open(payload.output)

    def _on_viewer_run_loaded(self, payload: object) -> None:
        path = getattr(payload, "path", None)
        if path is not None:
            resolved = Path(path)
            self.session_state.current_file = str(resolved)
            self.session_state.last_hdf5_directory = str(resolved.parent)
            add_recent_file(self.session_state, resolved)
            save_session_state(self.session_state)
            self._rebuild_recent_files_menu()
        self._set_breakout_target(
            None,
            reason="Breakout is unavailable until derived Shock analysis completes for the active run.",
        )
        self._update_global_snapshot_controls(self.viewer_controller.current_run_context())
        self._update_action_state()

    def _on_viewer_field_visualized(self, field_name: str) -> None:
        del field_name
        self._update_action_state()

    def _on_derived_status(self, message: str) -> None:
        if self._current_mode_id() == MODE_DERIVED:
            self.status_label.setText(message)

    def _format_preview_summary(self, payload: ParsePreviewPayload) -> str:
        lines = [
            f"Log file: {payload.source.name}",
            f"Simulation: {payload.simulation_name}",
            f"Geometry: {payload.geometry or '-'}",
            f"HELIOS version: {payload.code_version or '-'}",
            f"Calculated: {payload.calculation_datetime or '-'}",
            "",
            f"Snapshots: {payload.n_snapshots}",
            f"Zones: {payload.n_zones}",
            f"Regions: {payload.n_regions}",
            f"Materials: {payload.n_materials}",
            f"Detected fields: {len(payload.fields)}",
            f"Detected field families: {', '.join(payload.field_families) or '-'}",
            f"Approx core numeric size: {_human_bytes(payload.approx_numeric_bytes)}",
            "",
            "Header sections:",
        ]
        lines.extend(f"  - {section}" for section in payload.header_sections)
        return "\n".join(lines)

    def _format_preview_snapshot(self, payload: ParsePreviewPayload) -> str:
        lines = [
            "Fast preview:",
            f"  first cycle: {payload.first_cycle if payload.first_cycle is not None else '-'}",
            f"  first time: {payload.first_time:.4e} s" if payload.first_time is not None else "  first time: -",
            f"  first dt: {payload.first_time_step:.4e} s" if payload.first_time_step is not None else "  first dt: -",
            f"  last cycle: {payload.last_cycle if payload.last_cycle is not None else '-'}",
            f"  last time: {payload.last_time:.4e} s" if payload.last_time is not None else "  last time: -",
        ]
        if payload.first_time is not None and payload.last_time is not None:
            lines.append(f"  simulated span: {payload.last_time - payload.first_time:.4e} s")
        lines.extend(
            [
                "",
                "Example fields:",
            ]
        )
        lines.extend(f"  - {name}" for name in payload.fields[:12])
        return "\n".join(lines)

    def _update_action_state(self) -> None:
        """Keep shell actions enabled only in the modes where they are valid."""
        current_mode = self._current_mode_id()
        supports_export = self._mode_specs.get(current_mode, self._mode_specs[MODE_PARSER]).supports_export
        viewer_ready = self.viewer_controller.has_loaded_run()
        viewer_export_ready = self.viewer_controller.can_export_current_view()
        parse_ready = self.current_preview is not None and not self._parser_busy
        self.parse_action.setEnabled(parse_ready)
        self.open_viewer_action.setEnabled(viewer_ready or (self.last_parse_result is not None and self.last_parse_result.output.exists()))
        self.export_action.setEnabled(supports_export and viewer_ready and viewer_export_ready)
        self.refresh_preview_button.setEnabled(not self._parser_busy and bool(self.session_state.current_file))
        self.open_viewer_button.setEnabled(self.last_parse_result is not None and self.last_parse_result.output.exists())

    def _export_current_view(self) -> None:
        """Forward export to the embedded viewer when Viewer Mode is active."""
        if not self.viewer_controller.has_loaded_run():
            QtWidgets.QMessageBox.information(self, "No Viewer Dataset", "Open a stabilized HELIOS HDF5 file before exporting.")
            return
        self._set_mode(MODE_VIEWER)
        self.viewer_controller.export_current_view()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._auto_open_timer.stop()
        save_session_state(self.session_state)
        self.parser_controller.shutdown()
        self.viewer_controller.shutdown()
        self.derived_controller.shutdown()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    """Launch the unified HELIOS Parse/View desktop application."""
    _configure_logging()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or sys.argv)
    configure_application(app)
    apply_application_icon(app)
    window = HeliosParseViewMainWindow()
    window.show()

    args = list(argv or sys.argv)[1:]
    if args:
        candidate = Path(args[0])
        if candidate.exists():
            QtCore.QTimer.singleShot(0, lambda path=candidate: window._open_path(path))

    return app.exec()
