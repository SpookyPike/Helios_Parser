"""Primary HELIOS HDF5 viewer window.

This remains the production scientific viewer. The unified Parse/View shell
embeds this window rather than reimplementing viewer logic. Two workflows are
retained intentionally:

- Slice View: simpler manual/legacy-style inspection path
- Mouse Mode: main advanced interactive analysis workflow

The module owns scientific plotting, viewer settings, export, and interaction
state. Shell-level workflow routing lives elsewhere.
"""

from __future__ import annotations

from pathlib import Path
import logging
import re
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .controller import RunController
from .models import DiagnosticPayload, FieldPayload, OpenRunPayload
from .plots import CurvePlotWidget, FieldMapWidget
from .slider import apply_absolute_click_slider_behavior
from .settings import ViewerSettingsDialog, default_viewer_settings, load_viewer_settings, reset_viewer_settings, save_viewer_settings
from .style import LIGHT_THEME, THEME_MODES, ViewerTheme, apply_theme, build_mono_font, configure_application, configure_combo_box_interaction
from .units import DisplayUnitChoices, convert_field_values, convert_length_values, convert_time_values, display_unit_for_field, unit_options_for_field
from .workspace import HeliosViewerWorkspace
from helios.cache import AnalyzerCacheSet
from helios.runtime import RunContext
from helios.services.derived.xcom_hook import material_display_labels_by_id


FIELD_LABELS = {
    "artificial_viscosity": "Artificial viscosity",
    "compression": "Compression",
    "density": "Density",
    "electron_density": "Electron density",
    "electron_energy": "Electron specific energy",
    "electron_heat_capacity": "Electron heat capacity",
    "ion_energy": "Ion specific energy",
    "ion_heat_capacity": "Ion heat capacity",
    "kinetic_energy": "Kinetic specific energy",
    "laser_deposition": "Laser deposition",
    "laser_source": "Laser source",
    "mean_charge": "Mean charge",
    "pressure": "Total pressure",
    "pressure_e": "Electron pressure",
    "pressure_i": "Ion pressure",
    "pressure_radiation": "Radiation pressure",
    "radiation_cooling": "Radiation cooling",
    "radiation_energy": "Radiation specific energy",
    "radiation_heating": "Radiation heating",
    "radiation_net_heating": "Net radiation heating",
    "radiation_sink": "Radiation sink",
    "radius": "Radius",
    "temperature_e": "Electron temperature",
    "temperature_i": "Ion temperature",
    "temperature_radiation": "Radiation temperature",
    "velocity": "Velocity",
    "zone_width": "Zone width",
}

COLORMAP_OPTIONS = [
    ("Cividis", "cividis"),
    ("Turbo", "turbo"),
    ("Viridis", "viridis"),
    ("Plasma", "plasma"),
    ("Inferno", "inferno"),
    ("Magma", "magma"),
    ("Jet", "jet"),
    ("Hot", "hot"),
    ("Gray / Grey", "gray"),
]

EXPORT_SIZE_PRESETS = [
    ("Current widget size", None),
    ("Screen", (1280, 720)),
    ("Email / slide", (1600, 900)),
    ("Report", (1800, 1200)),
    ("Presentation", (1920, 1080)),
]

LOGGER = logging.getLogger(__name__)

_REFRESH_FIELD_MAP = 1 << 0
_REFRESH_LINE = 1 << 1
_REFRESH_MOUSE = 1 << 2
_REFRESH_DIAGNOSTIC = 1 << 3
_REFRESH_TRACE_LABEL = 1 << 4
_REFRESH_MOUSE_STATE = 1 << 5
_REFRESH_ACTIVE_LABEL = 1 << 6


class _IgnoreWheelUnlessFocused(QtCore.QObject):
    """Prevent accidental wheel-driven refresh storms on unfocused controls."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._enabled = True
        self._last_wheel_at: dict[int, float] = {}

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if not self._enabled:
            return super().eventFilter(watched, event)
        if event.type() == QtCore.QEvent.Wheel and isinstance(watched, QtWidgets.QWidget):
            if not isinstance(watched, QtWidgets.QAbstractSpinBox):
                return super().eventFilter(watched, event)
            if not watched.hasFocus():
                event.ignore()
                return True
            now = time.perf_counter()
            key = id(watched)
            previous = self._last_wheel_at.get(key)
            if previous is not None and (now - previous) < 0.05:
                event.ignore()
                return True
            self._last_wheel_at[key] = now
        return super().eventFilter(watched, event)


def _pretty_name(name: str) -> str:
    return FIELD_LABELS.get(name, name.replace("_", " ").capitalize())


def _pretty_diagnostic(path: str) -> str:
    return " / ".join(_pretty_name(part) for part in path.split("/"))


def _field_item_text(name: str, unit: str) -> str:
    label = _pretty_name(name)
    return f"{label} [{unit}]" if unit else label


def _diagnostic_item_text(path: str, unit: str) -> str:
    label = _pretty_diagnostic(path)
    return f"{label} [{unit}]" if unit else label


def _label_with_unit(label: str, unit: str) -> str:
    return f"{label} [{unit}]" if unit else label


class ExportDialog(QtWidgets.QDialog):
    def __init__(
        self,
        suggested_name: str,
        target_sizes: dict[str, QtCore.QSize],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Plot")
        self.setModal(True)
        self.resize(460, 320)
        self._target_sizes = {
            key: QtCore.QSize(max(1, size.width()), max(1, size.height()))
            for key, size in target_sizes.items()
        }
        self._suppress_size_sync = False
        self._aspect_ratio = 9.0 / 16.0

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("Full viewer", "viewer")
        self.target_combo.addItem("2D field map", "field_map")
        self.target_combo.addItem("Active lower plot/tab", "active_tab")
        form.addRow("Export target", self.target_combo)

        self.preset_combo = QtWidgets.QComboBox()
        for label, size in EXPORT_SIZE_PRESETS:
            self.preset_combo.addItem(label, size)
        form.addRow("Preset", self.preset_combo)

        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(320, 10000)
        self.width_spin.setSingleStep(40)
        self.width_spin.setSuffix(" px")
        form.addRow("Width", self.width_spin)

        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(240, 10000)
        self.height_spin.setSingleStep(40)
        self.height_spin.setSuffix(" px")
        form.addRow("Height", self.height_spin)

        self.keep_aspect_checkbox = QtWidgets.QCheckBox("Keep aspect ratio")
        self.keep_aspect_checkbox.setChecked(True)
        form.addRow("", self.keep_aspect_checkbox)

        self.dpi_combo = QtWidgets.QComboBox()
        for dpi in (100, 300, 600):
            self.dpi_combo.addItem(f"{dpi} dpi", dpi)
        self.dpi_combo.setCurrentIndex(max(0, self.dpi_combo.findData(300)))
        form.addRow("DPI", self.dpi_combo)

        self.transparent_checkbox = QtWidgets.QCheckBox("Transparent background where supported")
        form.addRow("", self.transparent_checkbox)

        self.path_edit = QtWidgets.QLineEdit(suggested_name)
        browse_button = QtWidgets.QPushButton("Browse...")
        browse_button.clicked.connect(self._browse)
        path_widget = QtWidgets.QWidget()
        path_layout = QtWidgets.QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(6)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse_button)
        form.addRow("Output path", path_widget)

        self.note_label = QtWidgets.QLabel("")
        self.note_label.setWordWrap(True)
        form.addRow("", self.note_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.target_combo.currentIndexChanged.connect(self._sync_capabilities)
        self.target_combo.currentIndexChanged.connect(self._sync_target_size)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.width_spin.valueChanged.connect(self._on_width_changed)
        self.height_spin.valueChanged.connect(self._on_height_changed)
        self.target_combo.setCurrentIndex(max(0, self.target_combo.findData("field_map")))
        self._sync_capabilities()
        self._sync_target_size()

    def _sync_capabilities(self) -> None:
        target = str(self.target_combo.currentData())
        if target == "field_map":
            self.note_label.setText(
                "PNG export is the practical sharing path in this phase. "
                "Use width/height and DPI controls to produce slide/report-ready images."
            )
        elif target == "active_tab":
            self.note_label.setText(
                "Export the active lower plot/tab as PNG using the current labels, units, and visible view state."
            )
        else:
            self.note_label.setText(
                "Export the full viewer as a PNG. Transparent background is best suited for plot targets."
            )

    def _apply_size(self, width: int, height: int) -> None:
        self._suppress_size_sync = True
        try:
            self.width_spin.setValue(max(self.width_spin.minimum(), min(self.width_spin.maximum(), int(width))))
            self.height_spin.setValue(max(self.height_spin.minimum(), min(self.height_spin.maximum(), int(height))))
        finally:
            self._suppress_size_sync = False
        self._aspect_ratio = float(max(1, self.height_spin.value())) / float(max(1, self.width_spin.value()))
        self._sync_extension()

    def _sync_target_size(self) -> None:
        if self.preset_combo.currentData() is not None:
            return
        target = str(self.target_combo.currentData())
        size = self._target_sizes.get(target, QtCore.QSize(1280, 720))
        self._apply_size(size.width(), size.height())

    def _apply_preset(self) -> None:
        preset_size = self.preset_combo.currentData()
        if preset_size is None:
            self._sync_target_size()
            return
        width, height = preset_size
        self._apply_size(int(width), int(height))

    def _on_width_changed(self, value: int) -> None:
        if self._suppress_size_sync:
            return
        if not self.keep_aspect_checkbox.isChecked():
            self._aspect_ratio = float(max(1, self.height_spin.value())) / float(max(1, value))
            self._sync_extension()
            return
        new_height = max(self.height_spin.minimum(), min(self.height_spin.maximum(), int(round(float(value) * self._aspect_ratio))))
        self._suppress_size_sync = True
        try:
            self.height_spin.setValue(new_height)
        finally:
            self._suppress_size_sync = False
        self._sync_extension()

    def _on_height_changed(self, value: int) -> None:
        if self._suppress_size_sync:
            return
        if not self.keep_aspect_checkbox.isChecked():
            self._aspect_ratio = float(max(1, value)) / float(max(1, self.width_spin.value()))
            self._sync_extension()
            return
        ratio = max(self._aspect_ratio, 1.0e-9)
        new_width = max(self.width_spin.minimum(), min(self.width_spin.maximum(), int(round(float(value) / ratio))))
        self._suppress_size_sync = True
        try:
            self.width_spin.setValue(new_width)
        finally:
            self._suppress_size_sync = False
        self._sync_extension()

    def _sync_extension(self) -> None:
        path = Path(self.path_edit.text().strip() or "helios_view.png")
        if path.suffix.lower() != ".png":
            self.path_edit.setText(str(path.with_suffix(".png")))

    def _browse(self) -> None:
        current = Path(self.path_edit.text().strip() or "helios_view.png")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Plot", str(current), "PNG Image (*.png)")
        if path:
            self.path_edit.setText(path)
            self._sync_extension()

    def export_options(self) -> dict[str, object]:
        path = Path(self.path_edit.text().strip() or "helios_view.png")
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        return {
            "target": str(self.target_combo.currentData()),
            "format": "png",
            "transparent": bool(self.transparent_checkbox.isChecked()),
            "path": path,
            "width": int(self.width_spin.value()),
            "height": int(self.height_spin.value()),
            "dpi": int(self.dpi_combo.currentData()),
        }


class HeliosViewerMainWindow(QtWidgets.QMainWindow):
    """Standalone HELIOS HDF5 viewer that can also be embedded in the shell."""

    run_loaded = QtCore.Signal(object)
    context_changed = QtCore.Signal(object)
    field_visualized = QtCore.Signal(str)
    diagnostic_visualized = QtCore.Signal(str)
    settings_changed = QtCore.Signal(object)

    def __init__(self) -> None:
        configure_application(QtWidgets.QApplication.instance())
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("HELIOS HDF5 Quick Look")
        self.resize(1760, 1060)

        self.controller = RunController(self)
        self.controller.run_opened.connect(self._on_run_opened)
        self.controller.field_loaded.connect(self._on_field_loaded)
        self.controller.diagnostic_loaded.connect(self._on_diagnostic_loaded)
        self.controller.status_changed.connect(self._set_status_message)
        self.controller.error_occurred.connect(self._on_error)
        self.controller.busy_changed.connect(self._on_busy_changed)

        self.run_payload: OpenRunPayload | None = None
        self.current_field_name: str | None = None
        self.current_field_payload: FieldPayload | None = None
        self.radius_payload: FieldPayload | None = None
        self.current_diagnostic_path: str | None = None
        self.current_diagnostic_payload: DiagnosticPayload | None = None
        self._updating_snapshot_controls = False
        self._probe_snapshot_index: int | None = None
        self._probe_zone_index: int | None = None
        self._probe_mode = "live"
        self._probe_hover_position: tuple[float, float] | None = None
        self._pending_probe_hover_position: tuple[float, float] | None = None
        self._hover_timer = QtCore.QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._consume_pending_probe_hover)
        self._hover_profile_count = 0
        self._hover_profile_total_ms = 0.0
        self._hover_profile_max_ms = 0.0
        self._last_hover_update_at = 0.0
        self.run_context = RunContext.empty()
        self.cache_layers = AnalyzerCacheSet()
        self._moving_mesh_cache = self.cache_layers.view_cache.bucket("moving_mesh", max_items=6)
        self._moving_mesh_cache_hits = 0
        self._moving_mesh_cache_misses = 0
        self._laser_entry_info: dict[str, object] | None = None
        self._mouse_plot_auto_range_pending = True
        self._display_field_cache_key: tuple[object, ...] | None = None
        self._display_field_cache_value: tuple[np.ndarray, str] | None = None
        self._prefer_primary_coordinates_on_open = False
        self._viewer_settings = load_viewer_settings()
        self._theme_mode = self._viewer_settings.theme_mode if self._viewer_settings.theme_mode in THEME_MODES else "system"
        self._theme: ViewerTheme = apply_theme(QtWidgets.QApplication.instance(), self._theme_mode)
        self._workspace_widget: HeliosViewerWorkspace | None = None
        self._use_external_snapshot_controls = False
        self._suppress_preference_persist = False
        self._suppress_trace_reference_updates = False
        self._suppress_mouse_adjust_updates = False
        self._wheel_guard = _IgnoreWheelUnlessFocused(self)
        self._trace_reference_zone_index = 0
        self._trace_reference_static_x_cm = 0.0
        self._trace_reference_radius_cm = 0.0
        self._trace_reference_anchor_snapshot = 0
        self._combined_zone_mask_key: tuple[object, ...] | None = None
        self._combined_zone_mask_value: np.ndarray | None = None
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._flush_scheduled_refresh)
        self._refresh_debounce_ms = 20
        self._pending_refresh_flags = 0
        self._pending_refresh_preserve_view = True
        self._refresh_batch_in_progress = False
        self._closing = False

        self._build_ui()
        for combo in self.findChildren(QtWidgets.QComboBox):
            configure_combo_box_interaction(combo)
        self._apply_theme_to_widgets()
        self._install_wheel_guard()
        self._wire_persistent_controls()
        self._restore_persistent_preferences()
        self._set_plot_navigation_mode("pan")
        self._set_controls_enabled(False)
        self._update_map_control_state()
        self._update_slice_control_state()

    def _build_ui(self) -> None:
        open_action = QtGui.QAction("&Open HDF5...", self)
        open_action.triggered.connect(self.open_file_dialog)
        export_action = QtGui.QAction("&Export Plot...", self)
        export_action.triggered.connect(self.export_screenshot)
        settings_action = QtGui.QAction("&Settings...", self)
        settings_action.triggered.connect(self._open_settings_dialog)
        reset_settings_action = QtGui.QAction("&Reset Viewer Settings", self)
        reset_settings_action.triggered.connect(self._reset_preferences_to_defaults)
        quit_action = QtGui.QAction("&Quit", self)
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        self.theme_action_group = QtGui.QActionGroup(self)
        self.theme_action_group.setExclusive(True)
        self.theme_actions: dict[str, QtGui.QAction] = {}
        for label, mode in (("Light", "light"), ("Dark", "dark"), ("System", "system")):
            action = QtGui.QAction(label, self, checkable=True)
            action.triggered.connect(lambda checked=False, selected_mode=mode: self._set_theme_mode(selected_mode))
            self.theme_action_group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[mode] = action
        view_menu.addAction(settings_action)
        view_menu.addAction(reset_settings_action)

        self._workspace_widget = HeliosViewerWorkspace()
        self.setCentralWidget(self._workspace_widget)
        root_layout = self._workspace_widget.root_layout

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        self.left_panel = self._build_left_panel()
        splitter.addWidget(self.left_panel)

        self.field_map_widget = FieldMapWidget()
        self.field_map_widget.probe_moved.connect(self._on_map_probe_moved)
        self.field_map_widget.probe_clicked.connect(self._on_map_probe_clicked)
        self.lineout_plot = CurvePlotWidget()
        self.diagnostic_plot = CurvePlotWidget()
        self.mouse_tab = self._build_mouse_tab()
        self.plot_tabs = QtWidgets.QTabWidget()
        # Slice View is retained intentionally as the simpler manual/legacy-style
        # workflow. Mouse Mode is the main advanced interactive path.
        self.plot_tabs.addTab(self.lineout_plot, "Slice View")
        self.plot_tabs.addTab(self.diagnostic_plot, "Diagnostics")
        self.plot_tabs.addTab(self.mouse_tab, "Mouse Mode")
        self.plot_tabs.currentChanged.connect(self._on_plot_tab_changed)
        self.plot_toolbar = self._build_plot_toolbar()

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self.plot_toolbar)
        vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical_splitter.setChildrenCollapsible(False)
        vertical_splitter.addWidget(self.field_map_widget)
        vertical_splitter.addWidget(self.plot_tabs)
        vertical_splitter.setStretchFactor(0, 3)
        vertical_splitter.setStretchFactor(1, 2)
        right_layout.addWidget(vertical_splitter)
        right_layout.addWidget(self._build_snapshot_controls())
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 1280])

        self.status_message = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status_message, 1)
        self.busy_indicator = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.busy_indicator)

    def _install_wheel_guard(self) -> None:
        guarded_widgets = (
            self.map_orientation_combo,
            self.map_coordinate_combo,
            self.slice_mode_combo,
            self.line_coordinate_combo,
            self.trace_zone_spin,
            self.trace_coordinate_spin,
            self.line_scale_combo,
            self.diagnostic_scale_combo,
            self.colormap_combo,
            self.map_scale_combo,
            self.clip_mode_combo,
            self.percentile_low_spin,
            self.percentile_high_spin,
            self.snapshot_slider,
            self.snapshot_spin,
            self.mouse_time_slider,
            self.mouse_time_spin,
            self.mouse_coordinate_slider,
            self.mouse_coordinate_spin,
        )
        for widget in guarded_widgets:
            widget.installEventFilter(self._wheel_guard)

    def _build_left_panel(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(420)
        panel = QtWidgets.QWidget()
        scroll.setWidget(panel)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        open_button = QtWidgets.QPushButton("Open HDF5 File...")
        open_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(open_button)

        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.file_label)

        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumBlockCount(200)
        self.summary_text.setFont(build_mono_font())
        self.summary_text.setPlaceholderText("Open a stabilized HELIOS HDF5 file to inspect the run.")
        summary_group = QtWidgets.QGroupBox("Simulation Summary")
        summary_layout = QtWidgets.QVBoxLayout(summary_group)
        summary_layout.addWidget(self.summary_text)
        layout.addWidget(summary_group)

        self.field_list = QtWidgets.QListWidget()
        self.field_list.currentItemChanged.connect(self._on_field_selection_changed)
        self.field_label = QtWidgets.QLabel("Field: -")
        self.field_label.setObjectName("SubtleLabel")
        self.field_label.setWordWrap(True)
        field_group = QtWidgets.QGroupBox("Fields")
        field_layout = QtWidgets.QVBoxLayout(field_group)
        field_layout.addWidget(self.field_list)
        field_layout.addWidget(self.field_label)
        layout.addWidget(field_group)

        controls_group = QtWidgets.QGroupBox("Plot Controls")
        controls_layout = QtWidgets.QFormLayout(controls_group)

        self.map_orientation_combo = QtWidgets.QComboBox()
        self.map_orientation_combo.addItem("Time on x, coordinate on y", "time_x_coord_y")
        self.map_orientation_combo.addItem("Coordinate on x, time on y", "coord_x_time_y")
        self.map_orientation_combo.currentIndexChanged.connect(self._on_map_controls_changed)

        self.map_coordinate_combo = QtWidgets.QComboBox()
        self.map_coordinate_combo.addItem("Static x", "static_x")
        self.map_coordinate_combo.addItem("Zone index", "zone")
        self.map_coordinate_combo.currentIndexChanged.connect(self._on_map_controls_changed)
        self.map_coordinate_combo.activated.connect(self._on_coordinate_mode_user_override)

        self.slice_mode_combo = QtWidgets.QComboBox()
        self.slice_mode_combo.addItem("Snapshot lineout", "snapshot_lineout")
        self.slice_mode_combo.addItem("Time trace", "time_trace")
        self.slice_mode_combo.currentIndexChanged.connect(self._on_slice_controls_changed)

        self.line_coordinate_combo = QtWidgets.QComboBox()
        self.line_coordinate_combo.addItem("Zone index", "zone")
        self.line_coordinate_combo.addItem("Static x", "static_x")
        self.line_coordinate_combo.currentIndexChanged.connect(self._on_slice_controls_changed)
        self.line_coordinate_combo.activated.connect(self._on_coordinate_mode_user_override)

        self.trace_reference_row_label = QtWidgets.QLabel("Reference zone")
        self.trace_zone_spin = QtWidgets.QSpinBox()
        self.trace_zone_spin.setMinimum(1)
        self.trace_zone_spin.setKeyboardTracking(False)
        self.trace_zone_spin.valueChanged.connect(self._on_trace_reference_changed)
        self.trace_coordinate_spin = QtWidgets.QDoubleSpinBox()
        self.trace_coordinate_spin.setDecimals(12)
        self.trace_coordinate_spin.setKeyboardTracking(False)
        self.trace_coordinate_spin.valueChanged.connect(self._on_trace_reference_changed)
        self.trace_reference_stack = QtWidgets.QStackedWidget()
        self.trace_reference_stack.addWidget(self.trace_zone_spin)
        self.trace_reference_stack.addWidget(self.trace_coordinate_spin)
        self.trace_reference_label = QtWidgets.QLabel("Reference zone: -")
        self.trace_reference_label.setObjectName("SubtleLabel")
        self.trace_reference_label.setWordWrap(True)

        self.line_scale_combo = QtWidgets.QComboBox()
        self.line_scale_combo.addItem("Linear", "linear")
        self.line_scale_combo.addItem("Log10 (positive only)", "log10")
        self.line_scale_combo.addItem("Signed log10", "signed_log10")
        self.line_scale_combo.currentIndexChanged.connect(self._schedule_line_plot_refresh)

        self.diagnostic_scale_combo = QtWidgets.QComboBox()
        self.diagnostic_scale_combo.addItem("Linear", "linear")
        self.diagnostic_scale_combo.addItem("Log10 (positive only)", "log10")
        self.diagnostic_scale_combo.addItem("Signed log10", "signed_log10")
        self.diagnostic_scale_combo.currentIndexChanged.connect(self._schedule_diagnostic_refresh)

        self.colormap_combo = QtWidgets.QComboBox()
        for label, value in COLORMAP_OPTIONS:
            self.colormap_combo.addItem(label, value)
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)

        self.map_scale_combo = QtWidgets.QComboBox()
        self.map_scale_combo.addItem("Linear", "linear")
        self.map_scale_combo.addItem("Log10 (positive values)", "log10")
        self.map_scale_combo.currentIndexChanged.connect(self._refresh_field_map_preserving_view)

        self.clip_mode_combo = QtWidgets.QComboBox()
        self.clip_mode_combo.addItem("Auto levels", "auto")
        self.clip_mode_combo.addItem("Percentile clip", "percentile")
        self.clip_mode_combo.addItem("Manual range", "manual")
        self.clip_mode_combo.currentIndexChanged.connect(self._on_map_controls_changed)

        percentile_widget = QtWidgets.QWidget()
        percentile_layout = QtWidgets.QHBoxLayout(percentile_widget)
        percentile_layout.setContentsMargins(0, 0, 0, 0)
        percentile_layout.setSpacing(4)
        self.percentile_low_spin = QtWidgets.QDoubleSpinBox()
        self.percentile_low_spin.setRange(0.0, 49.0)
        self.percentile_low_spin.setDecimals(1)
        self.percentile_low_spin.setKeyboardTracking(False)
        self.percentile_low_spin.setValue(1.0)
        self.percentile_low_spin.valueChanged.connect(self._refresh_field_map_preserving_view)
        self.percentile_high_spin = QtWidgets.QDoubleSpinBox()
        self.percentile_high_spin.setRange(51.0, 100.0)
        self.percentile_high_spin.setDecimals(1)
        self.percentile_high_spin.setKeyboardTracking(False)
        self.percentile_high_spin.setValue(99.0)
        self.percentile_high_spin.valueChanged.connect(self._refresh_field_map_preserving_view)
        percentile_layout.addWidget(self.percentile_low_spin)
        percentile_layout.addWidget(QtWidgets.QLabel("to"))
        percentile_layout.addWidget(self.percentile_high_spin)

        manual_widget = QtWidgets.QWidget()
        manual_layout = QtWidgets.QHBoxLayout(manual_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(4)
        self.level_min_edit = QtWidgets.QLineEdit()
        self.level_min_edit.setPlaceholderText("min")
        self.level_max_edit = QtWidgets.QLineEdit()
        self.level_max_edit.setPlaceholderText("max")
        self.apply_levels_button = QtWidgets.QPushButton("Apply")
        self.apply_levels_button.clicked.connect(self._refresh_field_map_preserving_view)
        manual_layout.addWidget(self.level_min_edit)
        manual_layout.addWidget(self.level_max_edit)
        manual_layout.addWidget(self.apply_levels_button)

        button_widget = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(4)
        self.auto_levels_button = QtWidgets.QPushButton("Auto color range")
        self.auto_levels_button.clicked.connect(self._reset_color_levels)
        self.reset_view_button = QtWidgets.QPushButton("Reset plot views")
        self.reset_view_button.clicked.connect(self._reset_plot_views)
        button_layout.addWidget(self.auto_levels_button)
        button_layout.addWidget(self.reset_view_button)

        self.coordinate_note_label = QtWidgets.QLabel(
            "2D map and slice view use explicit coordinate modes. Dynamic coordinates are only used for snapshot lineouts."
        )
        self.coordinate_note_label.setObjectName("SubtleLabel")
        self.coordinate_note_label.setWordWrap(True)
        self.coordinate_priority_label = QtWidgets.QLabel(
            "Moving-mesh coordinates are the primary workflow when available. Static coordinates and zone index remain compatibility coordinates."
        )
        self.coordinate_priority_label.setObjectName("SubtleLabel")
        self.coordinate_priority_label.setWordWrap(True)
        self.active_analysis_label = QtWidgets.QLabel("Active analysis: snapshot lineout")
        self.active_analysis_label.setObjectName("SubtleLabel")
        self.active_analysis_label.setWordWrap(True)

        controls_layout.addRow("2D orientation", self.map_orientation_combo)
        controls_layout.addRow("2D coordinate", self.map_coordinate_combo)
        controls_layout.addRow("Slice mode", self.slice_mode_combo)
        controls_layout.addRow("Slice coordinate", self.line_coordinate_combo)
        controls_layout.addRow(self.trace_reference_row_label, self.trace_reference_stack)
        controls_layout.addRow("", self.trace_reference_label)
        controls_layout.addRow("Slice value scale", self.line_scale_combo)
        controls_layout.addRow("Diagnostic scale", self.diagnostic_scale_combo)
        controls_layout.addRow("Colormap", self.colormap_combo)
        controls_layout.addRow("2D color scale", self.map_scale_combo)
        controls_layout.addRow("Color range", self.clip_mode_combo)
        controls_layout.addRow("Percentile", percentile_widget)
        controls_layout.addRow("Manual range", manual_widget)
        controls_layout.addRow("", button_widget)
        controls_layout.addRow("", self.coordinate_note_label)
        controls_layout.addRow("", self.coordinate_priority_label)
        controls_layout.addRow("", self.active_analysis_label)
        layout.addWidget(controls_group)

        selection_group = QtWidgets.QGroupBox("Target Subset")
        selection_layout = QtWidgets.QVBoxLayout(selection_group)
        selection_layout.setContentsMargins(8, 8, 8, 8)
        selection_layout.setSpacing(6)

        selection_layout.addWidget(QtWidgets.QLabel("Regions"))
        self.region_list = QtWidgets.QListWidget()
        self.region_list.itemChanged.connect(self._on_region_selection_changed)
        selection_layout.addWidget(self.region_list)

        region_button_widget = QtWidgets.QWidget()
        region_button_layout = QtWidgets.QHBoxLayout(region_button_widget)
        region_button_layout.setContentsMargins(0, 0, 0, 0)
        region_button_layout.setSpacing(4)
        self.select_all_regions_button = QtWidgets.QPushButton("All regions")
        self.select_all_regions_button.clicked.connect(self._select_all_regions)
        self.clear_regions_button = QtWidgets.QPushButton("Clear regions")
        self.clear_regions_button.clicked.connect(self._clear_all_regions)
        region_button_layout.addWidget(self.select_all_regions_button)
        region_button_layout.addWidget(self.clear_regions_button)
        selection_layout.addWidget(region_button_widget)

        selection_layout.addWidget(QtWidgets.QLabel("Materials"))
        self.material_list = QtWidgets.QListWidget()
        self.material_list.itemChanged.connect(self._on_material_selection_changed)
        selection_layout.addWidget(self.material_list)

        material_button_widget = QtWidgets.QWidget()
        material_button_layout = QtWidgets.QHBoxLayout(material_button_widget)
        material_button_layout.setContentsMargins(0, 0, 0, 0)
        material_button_layout.setSpacing(4)
        self.select_all_materials_button = QtWidgets.QPushButton("All materials")
        self.select_all_materials_button.clicked.connect(self._select_all_materials)
        self.clear_materials_button = QtWidgets.QPushButton("Clear materials")
        self.clear_materials_button.clicked.connect(self._clear_all_materials)
        material_button_layout.addWidget(self.select_all_materials_button)
        material_button_layout.addWidget(self.clear_materials_button)
        selection_layout.addWidget(material_button_widget)

        self.boundary_overlay_checkbox = QtWidgets.QCheckBox("Show region boundaries")
        self.boundary_overlay_checkbox.setChecked(True)
        self.boundary_overlay_checkbox.stateChanged.connect(self._refresh_visuals)
        selection_layout.addWidget(self.boundary_overlay_checkbox)

        self.filter_summary_label = QtWidgets.QLabel("Showing all zones.")
        self.filter_summary_label.setObjectName("SubtleLabel")
        self.filter_summary_label.setWordWrap(True)
        selection_layout.addWidget(self.filter_summary_label)
        layout.addWidget(selection_group)

        self.diagnostic_list = QtWidgets.QListWidget()
        self.diagnostic_list.currentItemChanged.connect(self._on_diagnostic_selection_changed)
        self.diagnostic_label = QtWidgets.QLabel("Diagnostic: -")
        self.diagnostic_label.setObjectName("SubtleLabel")
        self.diagnostic_label.setWordWrap(True)
        diagnostics_group = QtWidgets.QGroupBox("Diagnostics")
        diagnostics_layout = QtWidgets.QVBoxLayout(diagnostics_group)
        diagnostics_layout.addWidget(self.diagnostic_list)
        diagnostics_layout.addWidget(self.diagnostic_label)
        layout.addWidget(diagnostics_group)
        layout.addStretch(1)
        return scroll

    def _build_snapshot_controls(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.snapshot_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.snapshot_slider.setMinimumHeight(24)
        apply_absolute_click_slider_behavior(self.snapshot_slider)
        self.snapshot_slider.valueChanged.connect(self._on_snapshot_changed)
        self.snapshot_spin = QtWidgets.QSpinBox()
        self.snapshot_spin.setMinimumHeight(24)
        self.snapshot_spin.valueChanged.connect(self._on_snapshot_changed)
        self.snapshot_time_label = QtWidgets.QLabel("time: -")

        layout.addWidget(QtWidgets.QLabel("Snapshot"))
        layout.addWidget(self.snapshot_slider, 1)
        layout.addWidget(self.snapshot_spin)
        layout.addWidget(self.snapshot_time_label)
        self.snapshot_controls_widget = widget
        return widget

    def _build_plot_toolbar(self) -> QtWidgets.QToolBar:
        toolbar = QtWidgets.QToolBar("Plot Tools", self)
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)

        action_group = QtGui.QActionGroup(toolbar)
        action_group.setExclusive(True)
        self.plot_pan_action = QtGui.QAction("Pan", toolbar, checkable=True)
        self.plot_pan_action.setToolTip("Drag to pan the visible plots.")
        self.plot_pan_action.triggered.connect(lambda checked=False: self._set_plot_navigation_mode("pan"))
        action_group.addAction(self.plot_pan_action)
        toolbar.addAction(self.plot_pan_action)

        self.plot_zoom_action = QtGui.QAction("Zoom", toolbar, checkable=True)
        self.plot_zoom_action.setToolTip("Drag a rectangle to zoom the visible plots.")
        self.plot_zoom_action.triggered.connect(lambda checked=False: self._set_plot_navigation_mode("zoom"))
        action_group.addAction(self.plot_zoom_action)
        toolbar.addAction(self.plot_zoom_action)

        toolbar.addSeparator()

        self.plot_home_action = QtGui.QAction("Home", toolbar)
        self.plot_home_action.setToolTip("Reset the visible plot views.")
        self.plot_home_action.triggered.connect(self._reset_plot_views)
        toolbar.addAction(self.plot_home_action)

        self.plot_save_action = QtGui.QAction("Save", toolbar)
        self.plot_save_action.setToolTip("Export the current viewer or plot as a practical PNG.")
        self.plot_save_action.triggered.connect(self.export_screenshot)
        toolbar.addAction(self.plot_save_action)

        self.plot_pan_action.setChecked(True)
        return toolbar

    def _build_mouse_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        summary_strip = QtWidgets.QWidget()
        summary_layout = QtWidgets.QHBoxLayout(summary_strip)
        summary_layout.setContentsMargins(6, 4, 6, 4)
        summary_layout.setSpacing(8)
        self.mouse_mode_state_label = QtWidgets.QLabel("Hover ready")
        self.mouse_mode_state_label.setObjectName("SubtleLabel")
        self.mouse_mode_state_label.setMinimumWidth(96)
        self.mouse_mode_probe_label = QtWidgets.QLabel("Move over the 2D map to inspect slices.")
        self.mouse_mode_probe_label.setWordWrap(True)
        self.mouse_mode_probe_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.resume_hover_button = QtWidgets.QPushButton("Resume Hover")
        self.resume_hover_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.resume_hover_button.clicked.connect(self._resume_hover_probe)
        summary_layout.addWidget(self.mouse_mode_state_label, 0)
        summary_layout.addWidget(self.mouse_mode_probe_label, 1)
        summary_layout.addWidget(self.resume_hover_button, 0, QtCore.Qt.AlignRight)
        layout.addWidget(summary_strip)

        adjust_group = QtWidgets.QGroupBox("Probe Adjustment")
        adjust_layout = QtWidgets.QGridLayout(adjust_group)
        adjust_layout.setContentsMargins(8, 6, 8, 6)
        adjust_layout.setHorizontalSpacing(8)
        adjust_layout.setVerticalSpacing(4)

        self.mouse_time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mouse_time_slider.setMinimumHeight(24)
        self.mouse_time_slider.valueChanged.connect(self._on_mouse_time_slider_changed)
        self.mouse_time_spin = QtWidgets.QDoubleSpinBox()
        self.mouse_time_spin.setDecimals(12)
        self.mouse_time_spin.setKeyboardTracking(False)
        self.mouse_time_spin.setMinimumHeight(24)
        self.mouse_time_spin.setMaximumWidth(170)
        self.mouse_time_spin.valueChanged.connect(self._on_mouse_time_spin_changed)
        self.mouse_time_row_label = QtWidgets.QLabel("Time")
        adjust_layout.addWidget(self.mouse_time_row_label, 0, 0)
        adjust_layout.addWidget(self.mouse_time_slider, 0, 1)
        adjust_layout.addWidget(self.mouse_time_spin, 0, 2)

        self.mouse_coordinate_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mouse_coordinate_slider.setMinimumHeight(24)
        self.mouse_coordinate_slider.valueChanged.connect(self._on_mouse_coordinate_slider_changed)
        self.mouse_coordinate_spin = QtWidgets.QDoubleSpinBox()
        self.mouse_coordinate_spin.setDecimals(12)
        self.mouse_coordinate_spin.setKeyboardTracking(False)
        self.mouse_coordinate_spin.setMinimumHeight(24)
        self.mouse_coordinate_spin.setMaximumWidth(190)
        self.mouse_coordinate_spin.valueChanged.connect(self._on_mouse_coordinate_spin_changed)
        self.mouse_coordinate_row_label = QtWidgets.QLabel("Coordinate")
        adjust_layout.addWidget(self.mouse_coordinate_row_label, 1, 0)
        adjust_layout.addWidget(self.mouse_coordinate_slider, 1, 1)
        adjust_layout.addWidget(self.mouse_coordinate_spin, 1, 2)

        self.mouse_adjustment_label = QtWidgets.QLabel("Nearest-cell mapping is used for typed values.")
        self.mouse_adjustment_label.setWordWrap(True)
        self.mouse_adjustment_label.setObjectName("SubtleLabel")
        adjust_layout.addWidget(self.mouse_adjustment_label, 2, 1, 1, 2)
        layout.addWidget(adjust_group, 0)

        self.mouse_vertical_plot = CurvePlotWidget()
        self.mouse_horizontal_plot = CurvePlotWidget()

        vertical_group = QtWidgets.QGroupBox("Vertical Slice")
        vertical_layout = QtWidgets.QVBoxLayout(vertical_group)
        vertical_layout.setContentsMargins(6, 6, 6, 6)
        vertical_layout.addWidget(self.mouse_vertical_plot)

        horizontal_group = QtWidgets.QGroupBox("Horizontal Slice")
        horizontal_layout = QtWidgets.QVBoxLayout(horizontal_group)
        horizontal_layout.setContentsMargins(6, 6, 6, 6)
        horizontal_layout.addWidget(self.mouse_horizontal_plot)
        self.mouse_plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.mouse_plot_splitter.setChildrenCollapsible(False)
        self.mouse_plot_splitter.setHandleWidth(6)
        self.mouse_plot_splitter.addWidget(vertical_group)
        self.mouse_plot_splitter.addWidget(horizontal_group)
        self.mouse_plot_splitter.setStretchFactor(0, 1)
        self.mouse_plot_splitter.setStretchFactor(1, 1)
        self.mouse_plot_splitter.setSizes([640, 640])
        layout.addWidget(self.mouse_plot_splitter, 1)
        return widget

    def _iter_plot_targets(self) -> list[object]:
        targets: list[object] = [self.field_map_widget]
        if self.plot_tabs.currentWidget() is self.lineout_plot:
            targets.append(self.lineout_plot)
        elif self.plot_tabs.currentWidget() is self.diagnostic_plot:
            targets.append(self.diagnostic_plot)
        else:
            targets.extend([self.mouse_vertical_plot, self.mouse_horizontal_plot])
        return targets

    def _set_plot_navigation_mode(self, mode: str) -> None:
        normalized = "zoom" if str(mode) == "zoom" else "pan"
        if normalized == "zoom":
            self.plot_zoom_action.setChecked(True)
        else:
            self.plot_pan_action.setChecked(True)
        for target in self._iter_plot_targets():
            if hasattr(target, "set_navigation_mode"):
                target.set_navigation_mode(normalized)

    def _current_preferences(self) -> dict[str, object]:
        return {
            "theme_mode": self._theme_mode,
            "colormap": str(self.colormap_combo.currentData()),
            "map_scale_mode": str(self.map_scale_combo.currentData()),
            "line_scale_mode": str(self.line_scale_combo.currentData()),
            "diagnostic_scale_mode": str(self.diagnostic_scale_combo.currentData()),
            "clip_mode": str(self.clip_mode_combo.currentData()),
            "show_boundaries": bool(self.boundary_overlay_checkbox.isChecked()),
            "hover_interval_ms": int(self._viewer_settings.hover_interval_ms),
            "time_unit": self._viewer_settings.time_unit,
            "length_unit": self._viewer_settings.length_unit,
            "pressure_unit": self._viewer_settings.pressure_unit,
            "density_unit": self._viewer_settings.density_unit,
            "temperature_unit": self._viewer_settings.temperature_unit,
            "velocity_unit": self._viewer_settings.velocity_unit,
            "specific_energy_unit": self._viewer_settings.specific_energy_unit,
            "rate_unit": self._viewer_settings.rate_unit,
            "heat_capacity_unit": self._viewer_settings.heat_capacity_unit,
            "number_density_unit": self._viewer_settings.number_density_unit,
            "angle_unit": self._viewer_settings.angle_unit,
            "photon_unit": self._viewer_settings.photon_unit,
            "default_profile_coordinate": self._viewer_settings.default_profile_coordinate,
            "wheel_guard_enabled": bool(self._viewer_settings.wheel_guard_enabled),
            "last_open_directory": self._viewer_settings.last_open_directory,
        }

    def _save_current_preferences(self) -> None:
        preferences = self._current_preferences()
        self._viewer_settings = type(self._viewer_settings)(**preferences)
        save_viewer_settings(self._viewer_settings)

    def _restore_persistent_preferences(self) -> None:
        settings = self._viewer_settings
        self._suppress_preference_persist = True
        try:
            if self.colormap_combo.findData(settings.colormap) >= 0:
                self.colormap_combo.setCurrentIndex(self.colormap_combo.findData(settings.colormap))
            if self.map_scale_combo.findData(settings.map_scale_mode) >= 0:
                self.map_scale_combo.setCurrentIndex(self.map_scale_combo.findData(settings.map_scale_mode))
            if self.line_scale_combo.findData(settings.line_scale_mode) >= 0:
                self.line_scale_combo.setCurrentIndex(self.line_scale_combo.findData(settings.line_scale_mode))
            if self.diagnostic_scale_combo.findData(settings.diagnostic_scale_mode) >= 0:
                self.diagnostic_scale_combo.setCurrentIndex(self.diagnostic_scale_combo.findData(settings.diagnostic_scale_mode))
            if self.clip_mode_combo.findData(settings.clip_mode) >= 0:
                self.clip_mode_combo.setCurrentIndex(self.clip_mode_combo.findData(settings.clip_mode))
            self.boundary_overlay_checkbox.setChecked(bool(settings.show_boundaries))
            self._hover_timer.setInterval(max(0, int(settings.hover_interval_ms)))
            self._wheel_guard.set_enabled(bool(settings.wheel_guard_enabled))
            if self._theme_mode in self.theme_actions:
                self.theme_actions[self._theme_mode].setChecked(True)
            else:
                self.theme_actions["system"].setChecked(True)
        finally:
            self._suppress_preference_persist = False

    def _apply_theme_to_widgets(self) -> None:
        self.field_map_widget.apply_theme(self._theme)
        self.lineout_plot.apply_theme(self._theme)
        self.diagnostic_plot.apply_theme(self._theme)
        self.mouse_vertical_plot.apply_theme(self._theme)
        self.mouse_horizontal_plot.apply_theme(self._theme)

    def _display_units(self) -> DisplayUnitChoices:
        return DisplayUnitChoices(
            time_unit=self._viewer_settings.time_unit,
            length_unit=self._viewer_settings.length_unit,
            pressure_unit=self._viewer_settings.pressure_unit,
            density_unit=self._viewer_settings.density_unit,
            temperature_unit=self._viewer_settings.temperature_unit,
            velocity_unit=self._viewer_settings.velocity_unit,
            specific_energy_unit=self._viewer_settings.specific_energy_unit,
            rate_unit=self._viewer_settings.rate_unit,
            heat_capacity_unit=self._viewer_settings.heat_capacity_unit,
            number_density_unit=self._viewer_settings.number_density_unit,
            angle_unit=self._viewer_settings.angle_unit,
            photon_unit=self._viewer_settings.photon_unit,
        )

    def _display_time_values(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.size <= 1:
            return convert_time_values(array, self._viewer_settings.time_unit)
        cache = self.cache_layers.view_cache.bucket("display_axes", max_items=12)
        key = ("time", id(array), array.shape, self._viewer_settings.time_unit, float(array.flat[0]), float(array.flat[-1]))
        cached = cache.get(key)
        if cached is not None:
            return cached
        converted = np.asarray(convert_time_values(array, self._viewer_settings.time_unit), dtype=np.float64)
        cache[key] = converted
        return converted

    def _display_time_value(self, value: float) -> float:
        return float(self._display_time_values(np.asarray([value], dtype=np.float64))[0])

    def _display_length_values(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.size <= 1:
            return convert_length_values(array, self._viewer_settings.length_unit)
        cache = self.cache_layers.view_cache.bucket("display_axes", max_items=12)
        key = ("length", id(array), array.shape, self._viewer_settings.length_unit, float(array.flat[0]), float(array.flat[-1]))
        cached = cache.get(key)
        if cached is not None:
            return cached
        converted = np.asarray(convert_length_values(array, self._viewer_settings.length_unit), dtype=np.float64)
        cache[key] = converted
        return converted

    def _display_length_value(self, value: float) -> float:
        return float(self._display_length_values(np.asarray([value], dtype=np.float64))[0])

    def _native_length_value(self, value: float) -> float:
        scale = self._display_length_value(1.0)
        if scale == 0.0:
            return float(value)
        return float(value) / scale

    def _display_field_data(self, field_name: str, unit: str, data: np.ndarray) -> tuple[np.ndarray, str]:
        cache = self.cache_layers.view_cache.bucket("display_field", max_items=8)
        key = (
            field_name,
            unit,
            id(data),
            self._viewer_settings.time_unit,
            self._viewer_settings.length_unit,
            self._viewer_settings.pressure_unit,
            self._viewer_settings.density_unit,
            self._viewer_settings.temperature_unit,
            self._viewer_settings.velocity_unit,
            self._viewer_settings.specific_energy_unit,
            self._viewer_settings.rate_unit,
            self._viewer_settings.heat_capacity_unit,
            self._viewer_settings.number_density_unit,
        )
        cached = cache.get(key)
        if cached is not None:
            self._display_field_cache_key = key
            self._display_field_cache_value = cached
            return cached
        converted, display_unit = convert_field_values(field_name, data, unit, self._display_units())
        self._display_field_cache_key = key
        self._display_field_cache_value = (np.asarray(converted, dtype=np.float64), display_unit)
        cache[key] = self._display_field_cache_value
        return self._display_field_cache_value

    def _display_field_unit(self, field_name: str, native_unit: str) -> str:
        return display_unit_for_field(field_name, native_unit, self._display_units())

    def _display_radius_bundle(self) -> tuple[np.ndarray, str] | None:
        if self.radius_payload is None:
            return None
        return self._display_field_data("radius", self.radius_payload.unit, self.radius_payload.data)

    def _current_display_field_bundle(self) -> tuple[np.ndarray, str, str] | None:
        if self.current_field_payload is None:
            return None
        display_data, display_unit = self._display_field_data(
            self.current_field_payload.field_name,
            self.current_field_payload.unit,
            self.current_field_payload.data,
        )
        return display_data, display_unit, _pretty_name(self.current_field_payload.field_name)

    def _refresh_field_list_labels(self) -> None:
        if self.run_payload is None:
            return
        for index in range(self.field_list.count()):
            item = self.field_list.item(index)
            field_name = str(item.data(QtCore.Qt.UserRole))
            unit = self._display_field_unit(field_name, self.run_payload.field_units.get(field_name, ""))
            item.setText(_field_item_text(field_name, unit))
        if self.current_field_name is not None and self.run_payload is not None:
            unit = self._display_field_unit(self.current_field_name, self.run_payload.field_units.get(self.current_field_name, ""))
            self.field_label.setText(f"Field: {_field_item_text(self.current_field_name, unit)}")

    def _wire_persistent_controls(self) -> None:
        for combo in (
            self.colormap_combo,
            self.map_scale_combo,
            self.line_scale_combo,
            self.diagnostic_scale_combo,
            self.clip_mode_combo,
        ):
            combo.currentIndexChanged.connect(self._persist_visual_preferences)
        self.boundary_overlay_checkbox.stateChanged.connect(self._persist_visual_preferences)

    def _persist_visual_preferences(self, *args) -> None:
        del args
        if self._suppress_preference_persist:
            return
        self._save_current_preferences()

    def _set_theme_mode(self, mode: str) -> None:
        normalized = str(mode or "system").lower()
        if normalized not in THEME_MODES:
            normalized = "system"
        self._theme_mode = normalized
        self._theme = apply_theme(QtWidgets.QApplication.instance(), normalized)
        self._apply_theme_to_widgets()
        if normalized in self.theme_actions and not self.theme_actions[normalized].isChecked():
            self.theme_actions[normalized].setChecked(True)
        if not self._suppress_preference_persist:
            self._save_current_preferences()
        self._refresh_visuals()
        self._refresh_diagnostic_plot(preserve_view=True)

    def _open_settings_dialog(self) -> None:
        dialog = ViewerSettingsDialog(self._viewer_settings, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        defaults = self._viewer_settings
        updated = dialog.current_settings(defaults)
        self._apply_viewer_settings(updated)

    def _apply_viewer_settings(self, updated) -> None:
        self._viewer_settings = updated
        self._display_field_cache_key = None
        self._display_field_cache_value = None
        self.cache_layers.view_cache.clear()
        self._hover_timer.setInterval(max(0, int(updated.hover_interval_ms)))
        self._wheel_guard.set_enabled(bool(updated.wheel_guard_enabled))
        self._set_theme_mode(updated.theme_mode)
        self._refresh_field_list_labels()
        self._sync_trace_reference_editor_from_state()
        self._update_trace_reference_label()
        self._update_mouse_mode_state()
        self._apply_default_profile_coordinate_to_current_run()
        self._save_current_preferences()
        self._refresh_visuals(preserve_view=True)
        self._refresh_diagnostic_plot(preserve_view=True)
        self.settings_changed.emit(self._viewer_settings)

    def _reset_preferences_to_defaults(self) -> None:
        self._viewer_settings = reset_viewer_settings()
        self._theme_mode = self._viewer_settings.theme_mode
        self._display_field_cache_key = None
        self._display_field_cache_value = None
        self.cache_layers.view_cache.clear()
        self._suppress_preference_persist = True
        try:
            self._set_theme_mode(self._viewer_settings.theme_mode)
            self.colormap_combo.setCurrentIndex(max(0, self.colormap_combo.findData(self._viewer_settings.colormap)))
            self.map_scale_combo.setCurrentIndex(max(0, self.map_scale_combo.findData(self._viewer_settings.map_scale_mode)))
            self.line_scale_combo.setCurrentIndex(max(0, self.line_scale_combo.findData(self._viewer_settings.line_scale_mode)))
            self.diagnostic_scale_combo.setCurrentIndex(max(0, self.diagnostic_scale_combo.findData(self._viewer_settings.diagnostic_scale_mode)))
            self.clip_mode_combo.setCurrentIndex(max(0, self.clip_mode_combo.findData(self._viewer_settings.clip_mode)))
            self.boundary_overlay_checkbox.setChecked(bool(self._viewer_settings.show_boundaries))
            self._hover_timer.setInterval(max(0, int(self._viewer_settings.hover_interval_ms)))
            self._wheel_guard.set_enabled(bool(self._viewer_settings.wheel_guard_enabled))
        finally:
            self._suppress_preference_persist = False
        self._save_current_preferences()
        self._refresh_field_list_labels()
        self._sync_trace_reference_editor_from_state()
        self._update_trace_reference_label()
        self._update_mouse_mode_state()
        self._apply_default_profile_coordinate_to_current_run()
        self._refresh_visuals(preserve_view=True)
        self._refresh_diagnostic_plot(preserve_view=True)
        self.settings_changed.emit(self._viewer_settings)

    def open_file_dialog(self) -> None:
        start_dir = self._viewer_settings.last_open_directory or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open HELIOS HDF5",
            start_dir,
            "HELIOS HDF5 (*.h5 *.hdf5)",
        )
        if path:
            self.load_file(path)

    def set_embedded_mode(self, embedded: bool) -> None:
        """Hide standalone chrome when the viewer is hosted inside the shell."""
        self.menuBar().setVisible(not embedded)
        self.statusBar().setVisible(not embedded)

    def workspace_widget(self) -> HeliosViewerWorkspace:
        """Return the embeddable viewer workspace widget."""
        assert self._workspace_widget is not None
        return self._workspace_widget

    def take_workspace_widget(self) -> HeliosViewerWorkspace:
        """Detach the workspace widget for embedding in the unified shell."""
        assert self._workspace_widget is not None
        if self.centralWidget() is self._workspace_widget:
            widget = self.takeCentralWidget()
            assert isinstance(widget, HeliosViewerWorkspace)
            self._workspace_widget = widget
        return self._workspace_widget

    def set_theme_mode(self, mode: str) -> None:
        self._set_theme_mode(mode)

    def current_theme_mode(self) -> str:
        return self._theme_mode

    def current_time_unit(self) -> str:
        return str(self._viewer_settings.time_unit)

    def current_viewer_settings(self):
        return type(self._viewer_settings)(**self._current_preferences())

    def default_profile_coordinate_mode(self) -> str:
        mode = str(self._viewer_settings.default_profile_coordinate or "zone").strip().lower()
        if mode not in {"zone", "moving_radius", "static_x", "viewer_follow"}:
            return "zone"
        return mode

    def active_snapshot_index(self) -> int:
        return self._current_snapshot_index()

    def set_active_snapshot_index(self, index: int) -> None:
        self._set_snapshot_index(int(index))

    def nearest_snapshot_index_for_display_time(self, display_time: float) -> int:
        if self.run_payload is None or self.run_payload.time.size == 0:
            return 0
        display_values = self._display_time_values(np.asarray(self.run_payload.time, dtype=np.float64))
        finite_mask = np.isfinite(display_values)
        if not np.any(finite_mask):
            return self._current_snapshot_index()
        valid_indices = np.flatnonzero(finite_mask)
        nearest_index = int(valid_indices[np.argmin(np.abs(display_values[finite_mask] - float(display_time)))])
        return nearest_index

    def display_time_for_snapshot(self, snapshot_index: int) -> float:
        if self.run_payload is None or self.run_payload.time.size == 0:
            return float("nan")
        clamped = int(np.clip(snapshot_index, 0, self.run_payload.time.size - 1))
        return self._display_time_value(float(self.run_payload.time[clamped]))

    def set_external_snapshot_controls(self, enabled: bool) -> None:
        self._use_external_snapshot_controls = bool(enabled)
        if hasattr(self, "snapshot_controls_widget"):
            self.snapshot_controls_widget.setVisible(False if enabled else self.plot_tabs.currentWidget() is self.lineout_plot)

    def refresh_embedded_view(self) -> None:
        if self.run_payload is None:
            return
        self.field_map_widget.set_probe_enabled(self.plot_tabs.currentWidget() is self.mouse_tab)
        self._refresh_field_map(preserve_view=True)
        current_tab = self.plot_tabs.currentWidget()
        if current_tab is self.lineout_plot:
            self._refresh_line_plot(preserve_view=True)
        elif current_tab is self.diagnostic_plot:
            self._refresh_diagnostic_plot(preserve_view=True)
        elif current_tab is self.mouse_tab:
            self._refresh_mouse_mode_plots()

    def open_settings_dialog(self) -> None:
        self._open_settings_dialog()

    def reset_viewer_settings_to_defaults(self) -> None:
        self._reset_preferences_to_defaults()

    def export_current_view(self) -> None:
        """Open the export workflow for the current viewer state."""
        self.export_screenshot()

    def has_loaded_run(self) -> bool:
        return self.run_payload is not None

    def load_file(self, path: str | Path) -> None:
        resolved = Path(path)
        self._viewer_settings.last_open_directory = str(resolved.parent)
        self._save_current_preferences()
        self._set_status_message(f"Queueing open: {resolved.name}")
        self.controller.open_file(resolved)

    def _slugify_text(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
        cleaned = cleaned.strip("-")
        return cleaned or "view"

    def _suggest_export_filename(self, extension: str = "png") -> str:
        if self.run_payload is None:
            return f"helios_viewer.{extension}"
        tab_name = self.plot_tabs.tabText(self.plot_tabs.currentIndex()).lower().replace(" ", "_")
        parts = [
            self.run_payload.path.stem,
            self.current_field_name or "field",
            self._map_coordinate_mode(),
            str(self.map_orientation_combo.currentData()),
            tab_name,
            f"snapshot-{self._current_snapshot_index()}",
            str(self.map_scale_combo.currentData()),
            self._filter_brief_label(),
        ]
        return f"{self._slugify_text('__'.join(parts))}.{extension}"

    def _export_target_widget(self, target: str) -> QtWidgets.QWidget:
        if target == "field_map":
            return self.field_map_widget
        if target == "active_tab":
            current = self.plot_tabs.currentWidget()
            return current if current is not None else self.workspace_widget()
        return self.workspace_widget()

    def _export_target_sizes(self) -> dict[str, QtCore.QSize]:
        return {
            "viewer": self.workspace_widget().size(),
            "field_map": self.field_map_widget.size(),
            "active_tab": self._export_target_widget("active_tab").size(),
        }

    @staticmethod
    def _supports_vector_svg_export(widget: QtWidgets.QWidget) -> bool:
        return isinstance(widget, (FieldMapWidget, CurvePlotWidget))

    @staticmethod
    def _supports_vector_pdf_export(widget: QtWidgets.QWidget) -> bool:
        return isinstance(widget, CurvePlotWidget)

    def _grab_widget_image(self, widget: QtWidgets.QWidget) -> QtGui.QImage:
        return widget.grab().toImage().convertToFormat(QtGui.QImage.Format_ARGB32)

    def _background_hex_for_export(self, widget: QtWidgets.QWidget) -> str:
        return self._theme.plot_background if widget is not self.workspace_widget() else self._theme.panel_background

    def _render_widget_to_image(
        self,
        widget: QtWidgets.QWidget,
        *,
        width: int,
        height: int,
        dpi: int,
    ) -> QtGui.QImage:
        target_width = max(1, int(width))
        target_height = max(1, int(height))
        image = QtGui.QImage(target_width, target_height, QtGui.QImage.Format_ARGB32)
        dots_per_meter = int(round(float(dpi) / 25.4 * 1000.0))
        image.setDotsPerMeterX(dots_per_meter)
        image.setDotsPerMeterY(dots_per_meter)
        background_hex = self._background_hex_for_export(widget)
        image.fill(QtGui.QColor(background_hex))
        painter = QtGui.QPainter(image)
        try:
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            source_width = max(1, widget.width())
            source_height = max(1, widget.height())
            scale = min(float(target_width) / float(source_width), float(target_height) / float(source_height))
            x_offset = max(0.0, (float(target_width) - float(source_width) * scale) / 2.0)
            y_offset = max(0.0, (float(target_height) - float(source_height) * scale) / 2.0)
            painter.translate(x_offset, y_offset)
            painter.scale(scale, scale)
            widget.render(painter, QtCore.QPoint(0, 0))
        finally:
            painter.end()
        return image

    def _make_background_transparent(self, image: QtGui.QImage, color_hex: str) -> QtGui.QImage:
        converted = image.convertToFormat(QtGui.QImage.Format_ARGB32)
        bits = converted.bits()
        array = np.frombuffer(bits, dtype=np.uint8, count=converted.sizeInBytes()).reshape((converted.height(), converted.width(), 4))
        background = QtGui.QColor(color_hex)
        mask = (
            (array[:, :, 0] == background.blue())
            & (array[:, :, 1] == background.green())
            & (array[:, :, 2] == background.red())
        )
        array[mask, 3] = 0
        return converted

    def _save_png_export(
        self,
        widget: QtWidgets.QWidget,
        path: Path,
        *,
        transparent: bool,
        width: int | None = None,
        height: int | None = None,
        dpi: int = 144,
    ) -> None:
        """Save the selected widget as a PNG, optionally with transparency."""
        export_width = max(1, int(width or widget.width()))
        export_height = max(1, int(height or widget.height()))
        if hasattr(widget, "save_png"):
            widget.save_png(path, width=export_width, height=export_height, dpi=dpi, transparent=transparent)
            return
        if width is None or height is None:
            image = self._grab_widget_image(widget)
        else:
            image = self._render_widget_to_image(widget, width=width, height=height, dpi=dpi)
        if transparent:
            image = self._make_background_transparent(image, self._background_hex_for_export(widget))
        image.save(str(path))

    def _save_pdf_export(self, widget: QtWidgets.QWidget, path: Path) -> None:
        """Save the selected widget into a clean PDF page export."""
        image = self._grab_widget_image(widget)
        writer = QtGui.QPdfWriter(str(path))
        writer.setResolution(300)
        writer.setPageMargins(QtCore.QMarginsF(8.0, 8.0, 8.0, 8.0), QtGui.QPageLayout.Millimeter)
        painter = QtGui.QPainter(writer)
        try:
            page_rect = writer.pageLayout().paintRectPixels(writer.resolution())
            scaled = image.size()
            scaled.scale(page_rect.size(), QtCore.Qt.KeepAspectRatio)
            target_rect = QtCore.QRect(
                page_rect.x() + max(0, (page_rect.width() - scaled.width()) // 2),
                page_rect.y() + max(0, (page_rect.height() - scaled.height()) // 2),
                scaled.width(),
                scaled.height(),
            )
            painter.drawImage(target_rect, image)
        finally:
            painter.end()

    def _save_vector_pdf_export(self, widget: QtWidgets.QWidget, path: Path) -> None:
        if isinstance(widget, FieldMapWidget):
            widget.save_vector_pdf(path)
            return
        if isinstance(widget, CurvePlotWidget):
            widget.save_vector_pdf(path)
            return
        raise TypeError("Vector PDF export is only available for plot widgets.")

    def _save_svg_export(self, widget: QtWidgets.QWidget, path: Path, *, transparent: bool) -> None:
        if isinstance(widget, FieldMapWidget):
            widget.save_vector_svg(path, transparent=transparent)
            return
        if isinstance(widget, CurvePlotWidget):
            widget.save_vector_svg(path, transparent=transparent)
            return
        raise TypeError("SVG export is only available for plot widgets.")

    def export_screenshot(self) -> None:
        start_path = Path(self._viewer_settings.last_open_directory or str(Path.cwd())) / self._suggest_export_filename("png")
        dialog = ExportDialog(str(start_path), self._export_target_sizes(), self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        options = dialog.export_options()
        target_widget = self._export_target_widget(str(options["target"]))
        path = Path(options["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        export_format = str(options["format"])
        target_name = str(options["target"])
        export_width = int(options.get("width", max(1, target_widget.width())))
        export_height = int(options.get("height", max(1, target_widget.height())))
        export_dpi = int(options.get("dpi", 144))
        started = time.perf_counter()
        if export_format == "pdf":
            if target_name == "field_map":
                QtWidgets.QMessageBox.information(
                    self,
                    "2D PDF export disabled",
                    "2D field-map PDF export is temporarily disabled in this hotfix because the current PDF layout path is not reliable. "
                    "Use PNG or SVG for the field map.",
                )
                return
            if self._supports_vector_pdf_export(target_widget):
                self._save_vector_pdf_export(target_widget, path)
                self._set_status_message(f"Saved vector PDF to {path}")
            else:
                self._save_pdf_export(target_widget, path)
                self._set_status_message(f"Saved raster PDF fallback to {path}")
        elif export_format == "svg":
            if not self._supports_vector_svg_export(target_widget):
                QtWidgets.QMessageBox.information(
                    self,
                    "SVG Export Unavailable",
                    "SVG export is available for the 2D field map and the lower plot widgets. "
                    "Use PNG or PDF for full-viewer exports.",
                )
                return
            self._save_svg_export(target_widget, path, transparent=bool(options["transparent"]))
            self._set_status_message(f"Saved SVG to {path}")
        else:
            self._save_png_export(
                target_widget,
                path,
                transparent=bool(options["transparent"]),
                width=export_width,
                height=export_height,
                dpi=export_dpi,
            )
            self._set_status_message(f"Saved PNG to {path}")
        self._viewer_settings.last_open_directory = str(path.parent)
        self._save_current_preferences()
        LOGGER.info(
            "Exported target=%s format=%s path=%s vector=%s in %.3f s",
            target_name,
            export_format,
            path,
            (
                self._supports_vector_pdf_export(target_widget)
                if export_format == "pdf"
                else self._supports_vector_svg_export(target_widget)
            ) and export_format in {"pdf", "svg"},
            time.perf_counter() - started,
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._closing = True
        self._hover_timer.stop()
        self._refresh_timer.stop()
        self._pending_probe_hover_position = None
        self._pending_refresh_flags = 0
        self._save_current_preferences()
        self.controller.shutdown()
        super().closeEvent(event)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.field_list,
            self.map_orientation_combo,
            self.map_coordinate_combo,
            self.slice_mode_combo,
            self.line_coordinate_combo,
            self.trace_reference_stack,
            self.line_scale_combo,
            self.diagnostic_scale_combo,
            self.colormap_combo,
            self.map_scale_combo,
            self.clip_mode_combo,
            self.percentile_low_spin,
            self.percentile_high_spin,
            self.level_min_edit,
            self.level_max_edit,
            self.apply_levels_button,
            self.auto_levels_button,
            self.reset_view_button,
            self.region_list,
            self.select_all_regions_button,
            self.clear_regions_button,
            self.material_list,
            self.select_all_materials_button,
            self.clear_materials_button,
            self.boundary_overlay_checkbox,
            self.diagnostic_list,
            self.snapshot_slider,
            self.snapshot_spin,
            self.resume_hover_button,
            self.mouse_time_slider,
            self.mouse_time_spin,
            self.mouse_coordinate_slider,
            self.mouse_coordinate_spin,
            self.plot_toolbar,
        ):
            widget.setEnabled(enabled)

    def _set_status_message(self, message: str) -> None:
        self.status_message.setText(message)

    def _on_busy_changed(self, busy: bool) -> None:
        self.busy_indicator.setText("Loading..." if busy else "")

    def _invalidate_combined_zone_mask_cache(self) -> None:
        self._combined_zone_mask_key = None
        self._combined_zone_mask_value = None

    def _schedule_refresh(self, flags: int, *, preserve_view: bool = True) -> None:
        if flags == 0:
            return
        self._pending_refresh_flags |= int(flags)
        self._pending_refresh_preserve_view = bool(self._pending_refresh_preserve_view and preserve_view)
        if self._refresh_batch_in_progress:
            return
        self._refresh_timer.start(self._refresh_debounce_ms)

    def _schedule_field_map_refresh(self, *args) -> None:
        del args
        self._schedule_refresh(_REFRESH_FIELD_MAP, preserve_view=True)

    def _schedule_line_plot_refresh(self, *args) -> None:
        del args
        self._schedule_refresh(_REFRESH_LINE, preserve_view=True)

    def _schedule_diagnostic_refresh(self, *args) -> None:
        del args
        self._schedule_refresh(_REFRESH_DIAGNOSTIC, preserve_view=True)

    def _schedule_visual_refresh(self, *args, preserve_view: bool = True) -> None:
        del args
        self._mouse_plot_auto_range_pending = True
        self._schedule_refresh(
            _REFRESH_FIELD_MAP
            | _REFRESH_LINE
            | _REFRESH_MOUSE
            | _REFRESH_TRACE_LABEL
            | _REFRESH_MOUSE_STATE
            | _REFRESH_ACTIVE_LABEL,
            preserve_view=preserve_view,
        )

    def _flush_scheduled_refresh(self) -> None:
        if self._refresh_batch_in_progress:
            return
        flags = int(self._pending_refresh_flags)
        preserve_view = bool(self._pending_refresh_preserve_view)
        self._pending_refresh_flags = 0
        self._pending_refresh_preserve_view = True
        if flags == 0:
            return
        self._refresh_batch_in_progress = True
        try:
            if flags & _REFRESH_FIELD_MAP:
                self._refresh_field_map(preserve_view=preserve_view)
            if flags & _REFRESH_LINE:
                self._refresh_line_plot(preserve_view=preserve_view)
            if flags & _REFRESH_MOUSE:
                self._refresh_mouse_mode_plots()
            if flags & _REFRESH_DIAGNOSTIC:
                self._refresh_diagnostic_plot(preserve_view=preserve_view)
            if flags & _REFRESH_TRACE_LABEL:
                self._update_trace_reference_label()
            if flags & _REFRESH_MOUSE_STATE:
                self._update_mouse_mode_state()
            if flags & _REFRESH_ACTIVE_LABEL:
                self._update_active_analysis_label()
        finally:
            self._refresh_batch_in_progress = False
        if self._pending_refresh_flags:
            self._refresh_timer.start(self._refresh_debounce_ms)

    def viewer_cache_stats(self) -> dict[str, object]:
        return {
            "controller": self.controller.cache_stats(),
            "view_cache": self.cache_layers.view_cache.stats(),
        }

    def _on_run_opened(self, payload: OpenRunPayload) -> None:
        if int(payload.run_generation) != self.controller.run_generation:
            LOGGER.debug(
                "Ignoring stale run-open payload for generation %s; current generation is %s.",
                payload.run_generation,
                self.controller.run_generation,
            )
            return
        self.run_payload = payload
        self.current_field_name = None
        self.current_field_payload = None
        self.radius_payload = None
        self.current_diagnostic_payload = None
        self.current_diagnostic_path = None
        self._display_field_cache_key = None
        self._display_field_cache_value = None
        self.cache_layers.view_cache.clear()
        self._invalidate_combined_zone_mask_cache()
        self._invalidate_moving_mesh_cache()
        self._probe_snapshot_index = None
        self._probe_zone_index = None
        self._probe_mode = "live"
        self._probe_hover_position = None
        self._pending_probe_hover_position = None
        self._last_hover_update_at = 0.0
        self._laser_entry_info = self._compute_laser_entry_info(payload)
        self._mouse_plot_auto_range_pending = True
        self._prefer_primary_coordinates_on_open = bool(payload.has_dynamic_radius)
        self.field_map_widget.reset_dataset_state()
        self.lineout_plot.reset_dataset_state()
        self.diagnostic_plot.reset_dataset_state()
        self.mouse_vertical_plot.reset_dataset_state()
        self.mouse_horizontal_plot.reset_dataset_state()
        self.field_map_widget.clear_probe()
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Viewer cache stats after run open: %s", self.viewer_cache_stats())

        self.file_label.setText(str(payload.path))
        self.summary_text.setPlainText(self._format_summary(payload))
        self.field_label.setText("Field: -")
        self.diagnostic_label.setText("Diagnostic: -")

        self.field_list.clear()
        for field_name in payload.fields:
            unit = self._display_field_unit(field_name, payload.field_units.get(field_name, ""))
            item = QtWidgets.QListWidgetItem(_field_item_text(field_name, unit))
            item.setData(QtCore.Qt.UserRole, field_name)
            item.setToolTip(field_name)
            self.field_list.addItem(item)

        self.diagnostic_list.clear()
        for path in payload.diagnostics:
            unit = payload.diagnostic_units.get(path, "")
            item = QtWidgets.QListWidgetItem(_diagnostic_item_text(path, unit))
            item.setData(QtCore.Qt.UserRole, path)
            item.setToolTip(path)
            self.diagnostic_list.addItem(item)

        self.map_coordinate_combo.blockSignals(True)
        self.map_coordinate_combo.clear()
        if payload.has_dynamic_radius:
            self.map_coordinate_combo.addItem(self._coordinate_mode_text("moving_radius", capitalize=True), "moving_radius")
            self.map_coordinate_combo.addItem(f"{self._coordinate_mode_text('static_x', capitalize=True)} (legacy)", "static_x")
            self.map_coordinate_combo.addItem("Zone index", "zone")
            default_map_coordinate = "moving_radius"
        else:
            self.map_coordinate_combo.addItem(self._coordinate_mode_text("static_x", capitalize=True), "static_x")
            self.map_coordinate_combo.addItem("Zone index", "zone")
            default_map_coordinate = "static_x"
        self.map_coordinate_combo.setCurrentIndex(self.map_coordinate_combo.findData(default_map_coordinate))
        self.map_coordinate_combo.blockSignals(False)

        self.line_coordinate_combo.blockSignals(True)
        self.line_coordinate_combo.clear()
        if payload.has_dynamic_radius:
            self.line_coordinate_combo.addItem(f"{self._coordinate_mode_text('radius', capitalize=True)} (snapshot)", "radius")
            self.line_coordinate_combo.addItem(self._coordinate_mode_text("moving_radius", capitalize=True), "moving_radius")
            self.line_coordinate_combo.addItem(f"{self._coordinate_mode_text('static_x', capitalize=True)} (legacy)", "static_x")
            self.line_coordinate_combo.addItem("Zone index", "zone")
        else:
            self.line_coordinate_combo.addItem("Zone index", "zone")
            self.line_coordinate_combo.addItem(self._coordinate_mode_text("static_x", capitalize=True), "static_x")
        default_coordinate = self._initial_line_coordinate_mode(payload.has_dynamic_radius)
        self.line_coordinate_combo.setCurrentIndex(self.line_coordinate_combo.findData(default_coordinate))
        self.line_coordinate_combo.blockSignals(False)

        if payload.has_dynamic_radius:
            self.coordinate_priority_label.setText(
                f"{self._coordinate_mode_text('moving_radius', capitalize=True)} is the primary workflow when available. "
                f"{self._coordinate_mode_text('static_x', capitalize=True)} and zone index remain compatibility coordinates."
            )
        else:
            self.coordinate_priority_label.setText(
                f"{self._coordinate_mode_text('static_x', capitalize=True)} and zone index are the primary inspection coordinates for this run."
            )

        self.material_list.blockSignals(True)
        self.material_list.clear()
        material_labels = material_display_labels_by_id(payload.materials if isinstance(payload.materials, dict) else {})
        for material_id in np.asarray(payload.materials["index"], dtype=np.int32):
            zone_count = int((np.abs(payload.zone_material_index) == abs(int(material_id))).sum())
            label = material_labels.get(int(abs(material_id)))
            text = f"Material {int(material_id)}"
            if label:
                text += f" | {label}"
            text += f" | {zone_count} zones"
            item = QtWidgets.QListWidgetItem(text)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setData(QtCore.Qt.UserRole, int(material_id))
            item.setCheckState(QtCore.Qt.Checked)
            self.material_list.addItem(item)
        self.material_list.blockSignals(False)

        self.region_list.blockSignals(True)
        self.region_list.clear()
        region_ids = np.asarray(payload.regions["region_index"], dtype=np.int32)
        min_zones = np.asarray(payload.regions["min_zone_index"], dtype=np.int32)
        max_zones = np.asarray(payload.regions["max_zone_index"], dtype=np.int32)
        for index, region_id in enumerate(region_ids):
            zone_count = int(max_zones[index] - min_zones[index] + 1)
            text = f"Region {int(region_id)} | zones {int(min_zones[index])}-{int(max_zones[index])} | {zone_count} zones"
            if self._laser_entry_info is not None and int(region_id) == int(self._laser_entry_info["incident_region"]):
                text += " | laser-entry region"
            item = QtWidgets.QListWidgetItem(text)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setData(QtCore.Qt.UserRole, int(region_id))
            item.setCheckState(QtCore.Qt.Checked)
            self.region_list.addItem(item)
        self.region_list.blockSignals(False)

        self._trace_reference_zone_index = 0
        self._trace_reference_static_x_cm = float(payload.static_x[0]) if payload.static_x.size else 0.0
        self._trace_reference_radius_cm = self._trace_reference_static_x_cm
        self._trace_reference_anchor_snapshot = 0
        self.trace_zone_spin.blockSignals(True)
        self.trace_zone_spin.setRange(1, payload.summary["n_zones"])
        self.trace_zone_spin.setValue(1)
        self.trace_zone_spin.blockSignals(False)
        self.trace_coordinate_spin.blockSignals(True)
        self.trace_coordinate_spin.setRange(-1.0e12, 1.0e12)
        self.trace_coordinate_spin.setValue(self._display_length_value(self._trace_reference_static_x_cm))
        self.trace_coordinate_spin.blockSignals(False)

        self.snapshot_slider.blockSignals(True)
        self.snapshot_spin.blockSignals(True)
        self.snapshot_slider.setRange(0, payload.summary["n_snapshots"] - 1)
        self.snapshot_spin.setRange(0, payload.summary["n_snapshots"] - 1)
        self.snapshot_slider.setValue(0)
        self.snapshot_spin.setValue(0)
        self.snapshot_slider.blockSignals(False)
        self.snapshot_spin.blockSignals(False)

        self.mouse_time_slider.blockSignals(True)
        self.mouse_time_slider.setRange(0, payload.summary["n_snapshots"] - 1)
        self.mouse_time_slider.setValue(0)
        self.mouse_time_slider.blockSignals(False)
        self.mouse_coordinate_slider.blockSignals(True)
        self.mouse_coordinate_slider.setRange(0, payload.summary["n_zones"] - 1)
        self.mouse_coordinate_slider.setValue(0)
        self.mouse_coordinate_slider.blockSignals(False)

        self._update_time_label(0)
        self._set_controls_enabled(True)
        self.run_context = RunContext.from_payload(payload)
        self.run_context.set_coordinate_modes(
            map_coordinate=str(self.map_coordinate_combo.currentData()),
            slice_coordinate=str(self.line_coordinate_combo.currentData()),
        )
        self._update_map_control_state()
        self._update_slice_control_state()
        self._update_filter_summary()
        self._update_coordinate_note()
        self._update_active_analysis_label()
        self._update_mouse_mode_state()
        self.field_map_widget.set_probe_enabled(self.plot_tabs.currentWidget() is self.mouse_tab)
        self._emit_run_context_changed()
        self.run_loaded.emit(payload)

        default_field = "density" if "density" in payload.fields else payload.fields[0]
        self._select_list_item_by_data(self.field_list, default_field)
        if payload.has_dynamic_radius and default_field != "radius":
            self.controller.load_field("radius")
        if payload.diagnostics:
            preferred = "energy_summary/current/ions" if "energy_summary/current/ions" in payload.diagnostics else payload.diagnostics[0]
            self._select_list_item_by_data(self.diagnostic_list, preferred)

    def _format_summary(self, payload: OpenRunPayload) -> str:
        summary = payload.summary
        metadata = summary["metadata"]
        lines = [
            f"File: {payload.path.name}",
            f"Simulation: {summary['simulation_name']}",
            f"Geometry: {summary.get('geometry', '-')}",
            f"HELIOS version: {summary.get('helios_version', '-')}",
            f"Calculated: {summary.get('calculation_datetime', '-')}",
            "",
            f"Zones: {summary['n_zones']}",
            f"Snapshots: {summary['n_snapshots']}",
            f"Regions: {summary['n_regions']}",
            f"Materials: {summary['n_materials']}",
            f"Fields: {len(summary['available_fields'])}",
            "",
            f"EOS model: {metadata.get('eos_model', '-')}",
            f"Source: {metadata.get('source_file', '-')}",
        ]
        if self._laser_entry_info is not None:
            lines.extend(
                [
                    "",
                    f"Laser entry: {self._laser_entry_info['incident_boundary']}",
                    f"First illuminated zone: {self._laser_entry_info['first_physical_zone']}",
                    f"Illuminated region: {self._laser_entry_info['incident_region']}",
                    f"Incident region boundary: {self._laser_entry_info['incident_region_boundary']}",
                    f"Propagation direction: {self._laser_entry_info['propagation_direction_text']}",
                ]
            )
        return "\n".join(lines)

    def _emit_run_context_changed(self) -> None:
        """Broadcast the mutable shared RunContext after a meaningful update."""

        self.context_changed.emit(self.run_context)

    def _compute_laser_entry_info(self, payload: OpenRunPayload) -> dict[str, object] | None:
        laser = payload.metadata.get("input_parameters", {}).get("laser_source", {})
        if not isinstance(laser, dict):
            return None
        origin = laser.get("origin_zone_index")
        direction = laser.get("propagation_direction")
        if origin is None or direction is None:
            return None
        try:
            origin_zone = int(origin)
        except (TypeError, ValueError):
            return None
        direction_text = str(direction).strip()
        n_zones = int(payload.summary["n_zones"])
        if direction_text == "Rmin":
            propagation_text = "toward Rmin"
            if origin_zone > n_zones:
                incident_boundary = "high-index boundary"
                first_zone = n_zones
                boundary_kind = "high"
            elif 1 <= origin_zone <= n_zones:
                incident_boundary = "internal launch"
                first_zone = origin_zone
                boundary_kind = "internal"
            else:
                return None
        elif direction_text == "Rmax":
            propagation_text = "toward Rmax"
            if origin_zone < 1:
                incident_boundary = "low-index boundary"
                first_zone = 1
                boundary_kind = "low"
            elif 1 <= origin_zone <= n_zones:
                incident_boundary = "internal launch"
                first_zone = origin_zone
                boundary_kind = "internal"
            else:
                return None
        else:
            return None

        if first_zone < 1 or first_zone > payload.zone_region_id.size:
            return None
        incident_region = int(payload.zone_region_id[first_zone - 1])
        region_ids = np.asarray(payload.regions["region_index"], dtype=np.int32)
        min_zones = np.asarray(payload.regions["min_zone_index"], dtype=np.int32)
        max_zones = np.asarray(payload.regions["max_zone_index"], dtype=np.int32)
        region_match = np.flatnonzero(region_ids == incident_region)
        if region_match.size == 0:
            return None
        region_index = int(region_match[0])
        if boundary_kind == "low":
            boundary_label = f"Region {incident_region} low-index boundary"
        elif boundary_kind == "high":
            boundary_label = f"Region {incident_region} high-index boundary"
        else:
            zone_min = int(min_zones[region_index])
            zone_max = int(max_zones[region_index])
            boundary_label = f"Region {incident_region}, launched inside zones {zone_min}-{zone_max}"
        return {
            "origin_zone_index": origin_zone,
            "propagation_direction": direction_text,
            "propagation_direction_text": propagation_text,
            "incident_boundary": incident_boundary,
            "first_physical_zone": int(first_zone),
            "incident_region": incident_region,
            "incident_region_boundary": boundary_label,
            "boundary_kind": boundary_kind,
        }

    def _on_field_selection_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        del previous
        if current is None:
            return
        field_name = str(current.data(QtCore.Qt.UserRole))
        unit = self.run_payload.field_units.get(field_name, "") if self.run_payload is not None else ""
        unit = self._display_field_unit(field_name, unit)
        self.field_label.setText(f"Field: {_field_item_text(field_name, unit)}")
        self.current_field_name = field_name
        self.controller.load_field(field_name)

    def _on_diagnostic_selection_changed(
        self,
        current: QtWidgets.QListWidgetItem | None,
        previous: QtWidgets.QListWidgetItem | None,
    ) -> None:
        del previous
        if current is None:
            return
        path = str(current.data(QtCore.Qt.UserRole))
        unit = self.run_payload.diagnostic_units.get(path, "") if self.run_payload is not None else ""
        self.diagnostic_label.setText(f"Diagnostic: {_diagnostic_item_text(path, unit)}")
        self.current_diagnostic_path = path
        self.controller.load_diagnostic(path)

    def _on_map_controls_changed(self) -> None:
        previous_zone = self._resolved_trace_zone_index()
        # Once the user explicitly leaves the preferred moving-mesh mode before
        # the radius payload arrives, keep that choice instead of snapping back
        # when the dynamic coordinate field finishes loading.
        if self.run_payload is not None and self.run_payload.has_dynamic_radius and str(self.map_coordinate_combo.currentData()) != "moving_radius":
            self._prefer_primary_coordinates_on_open = False
        if self.run_context.has_run:
            self.run_context.set_coordinate_modes(
                map_coordinate=str(self.map_coordinate_combo.currentData()),
                slice_coordinate=str(self.line_coordinate_combo.currentData()),
            )
        self._update_map_control_state()
        self._sync_line_coordinate_to_map_coordinate()
        self._update_slice_control_state()
        self._rebase_trace_reference_for_current_mode(previous_zone)
        self._sync_trace_reference_editor_from_state()
        self._update_trace_reference_label()
        self._update_coordinate_note()
        self._emit_run_context_changed()
        self._schedule_refresh(_REFRESH_FIELD_MAP | _REFRESH_MOUSE | _REFRESH_ACTIVE_LABEL, preserve_view=True)

    def _on_coordinate_mode_user_override(self, index: int) -> None:
        del index
        self._prefer_primary_coordinates_on_open = False

    def _on_slice_controls_changed(self) -> None:
        previous_zone = self._resolved_trace_zone_index()
        if self.run_context.has_run:
            self.run_context.set_coordinate_modes(
                map_coordinate=str(self.map_coordinate_combo.currentData()),
                slice_coordinate=str(self.line_coordinate_combo.currentData()),
            )
        self._update_slice_control_state()
        self._rebase_trace_reference_for_current_mode(previous_zone)
        self._sync_trace_reference_editor_from_state()
        self._update_trace_reference_label()
        self._update_coordinate_note()
        self._emit_run_context_changed()
        self._schedule_refresh(_REFRESH_FIELD_MAP | _REFRESH_LINE | _REFRESH_MOUSE | _REFRESH_ACTIVE_LABEL, preserve_view=True)

    def _on_trace_reference_changed(self) -> None:
        if self._suppress_trace_reference_updates or self.run_payload is None:
            return
        mode = self._time_trace_reference_mode()
        if mode == "zone":
            self._trace_reference_zone_index = max(0, int(self.trace_zone_spin.value()) - 1)
        else:
            requested_cm = self._native_length_value(self.trace_coordinate_spin.value())
            if mode == "static_x":
                self._trace_reference_static_x_cm = requested_cm
            else:
                self._trace_reference_radius_cm = requested_cm
                self._trace_reference_anchor_snapshot = self._current_snapshot_index()
        self._schedule_refresh(_REFRESH_FIELD_MAP | _REFRESH_LINE | _REFRESH_MOUSE | _REFRESH_TRACE_LABEL, preserve_view=True)

    def _on_region_selection_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        del item
        self._invalidate_combined_zone_mask_cache()
        self._update_filter_summary()
        self._schedule_visual_refresh()

    def _on_material_selection_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        del item
        self._invalidate_combined_zone_mask_cache()
        self._update_filter_summary()
        self._schedule_visual_refresh()

    def _select_all_regions(self) -> None:
        self.region_list.blockSignals(True)
        for index in range(self.region_list.count()):
            self.region_list.item(index).setCheckState(QtCore.Qt.Checked)
        self.region_list.blockSignals(False)
        self._invalidate_combined_zone_mask_cache()
        self._update_filter_summary()
        self._schedule_visual_refresh()

    def _clear_all_regions(self) -> None:
        self.region_list.blockSignals(True)
        for index in range(self.region_list.count()):
            self.region_list.item(index).setCheckState(QtCore.Qt.Unchecked)
        self.region_list.blockSignals(False)
        self._invalidate_combined_zone_mask_cache()
        self._update_filter_summary()
        self._schedule_visual_refresh()

    def _select_all_materials(self) -> None:
        self.material_list.blockSignals(True)
        for index in range(self.material_list.count()):
            self.material_list.item(index).setCheckState(QtCore.Qt.Checked)
        self.material_list.blockSignals(False)
        self._invalidate_combined_zone_mask_cache()
        self._update_filter_summary()
        self._schedule_visual_refresh()

    def _clear_all_materials(self) -> None:
        self.material_list.blockSignals(True)
        for index in range(self.material_list.count()):
            self.material_list.item(index).setCheckState(QtCore.Qt.Unchecked)
        self.material_list.blockSignals(False)
        self._invalidate_combined_zone_mask_cache()
        self._update_filter_summary()
        self._schedule_visual_refresh()

    def _set_combo_item_enabled(self, combo: QtWidgets.QComboBox, data: str, enabled: bool) -> None:
        index = combo.findData(data)
        if index < 0:
            return
        model = combo.model()
        item = model.item(index) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(enabled)

    def _selected_region_ids(self) -> list[int]:
        region_ids: list[int] = []
        for index in range(self.region_list.count()):
            item = self.region_list.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                region_ids.append(int(item.data(QtCore.Qt.UserRole)))
        return region_ids

    def _selected_material_ids(self) -> list[int]:
        material_ids: list[int] = []
        for index in range(self.material_list.count()):
            item = self.material_list.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                material_ids.append(int(item.data(QtCore.Qt.UserRole)))
        return material_ids

    def _set_line_coordinate_mode(self, mode: str) -> None:
        index = self.line_coordinate_combo.findData(mode)
        if index < 0:
            return
        self.line_coordinate_combo.blockSignals(True)
        self.line_coordinate_combo.setCurrentIndex(index)
        self.line_coordinate_combo.blockSignals(False)
        if self.run_context.has_run:
            self.run_context.set_coordinate_modes(
                map_coordinate=str(self.map_coordinate_combo.currentData()),
                slice_coordinate=str(self.line_coordinate_combo.currentData()),
            )

    def _configured_profile_coordinate_mode(self) -> str:
        return self.default_profile_coordinate_mode()

    def _resolve_profile_coordinate_mode_for_viewer(self, configured_mode: str, *, has_dynamic_radius: bool) -> str:
        normalized = str(configured_mode or "zone").strip().lower()
        available = {str(self.line_coordinate_combo.itemData(index)) for index in range(self.line_coordinate_combo.count())}
        if normalized == "viewer_follow":
            map_mode = self._map_coordinate_mode()
            if map_mode == "moving_radius":
                return "moving_radius" if "moving_radius" in available else ("radius" if "radius" in available else "zone")
            if map_mode in available:
                return map_mode
            return "zone" if "zone" in available else "static_x"
        if normalized == "moving_radius":
            if "moving_radius" in available and has_dynamic_radius:
                return "moving_radius"
            if "radius" in available and has_dynamic_radius:
                return "radius"
            return "zone" if "zone" in available else "static_x"
        if normalized == "static_x" and "static_x" in available:
            return "static_x"
        if "zone" in available:
            return "zone"
        return "static_x" if "static_x" in available else "radius"

    def _initial_line_coordinate_mode(self, has_dynamic_radius: bool) -> str:
        return self._resolve_profile_coordinate_mode_for_viewer(
            self._configured_profile_coordinate_mode(),
            has_dynamic_radius=bool(has_dynamic_radius),
        )

    def _preferred_line_coordinate_mode(self) -> str:
        return self._resolve_profile_coordinate_mode_for_viewer(
            self._configured_profile_coordinate_mode(),
            has_dynamic_radius=self.radius_payload is not None or bool(self.run_context.has_dynamic_radius),
        )

    def _apply_default_profile_coordinate_to_current_run(self) -> None:
        if self.run_payload is None:
            return
        preferred = self._preferred_line_coordinate_mode()
        if preferred != self._line_coordinate_mode():
            self._set_line_coordinate_mode(preferred)
            self._update_slice_control_state()
            self._update_coordinate_note()
            self._emit_run_context_changed()
            self._refresh_line_plot(preserve_view=True)
            self._refresh_mouse_mode_plots()

    def _time_trace_reference_mode(self) -> str:
        mode = self._line_coordinate_mode()
        if mode == "radius":
            return "moving_radius"
        return mode

    def _coordinate_name(self) -> str:
        if self.run_payload is None:
            return "x"
        model = self.run_payload.metadata.get("coordinate_model", {})
        if isinstance(model, dict):
            value = str(model.get("coordinate_name", "")).strip().lower()
            if value in {"x", "radius"}:
                return value
        geometry = str(self.run_payload.metadata.get("geometry", "")).strip().upper()
        return "radius" if geometry in {"CYLINDRICAL", "SPHERICAL"} else "x"

    def _static_coordinate_text(self, *, capitalize: bool = False) -> str:
        if self._coordinate_name() == "radius":
            return "Radius" if capitalize else "radius"
        return "Static x" if capitalize else "static x"

    def _moving_mesh_coordinate_text(self, *, capitalize: bool = False) -> str:
        if self._coordinate_name() == "radius":
            return "Moving-mesh radius" if capitalize else "moving-mesh radius"
        return "Moving-mesh x" if capitalize else "moving-mesh x"

    def _dynamic_coordinate_text(self, *, capitalize: bool = False) -> str:
        if self._coordinate_name() == "radius":
            return "Dynamic radius" if capitalize else "dynamic radius"
        return "Dynamic x" if capitalize else "dynamic x"

    def _coordinate_mode_text(self, mode: str, *, capitalize: bool = False) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized == "zone":
            return "Zone index" if capitalize else "zone index"
        if normalized == "static_x":
            return self._static_coordinate_text(capitalize=capitalize)
        if normalized == "moving_radius":
            return self._moving_mesh_coordinate_text(capitalize=capitalize)
        if normalized == "radius":
            return self._dynamic_coordinate_text(capitalize=capitalize)
        text = normalized.replace("_", " ")
        return text.capitalize() if capitalize else text

    def _sync_trace_reference_editor_from_state(self) -> None:
        if self.run_payload is None:
            return
        mode = self._time_trace_reference_mode()
        self._suppress_trace_reference_updates = True
        try:
            if mode == "zone":
                self.trace_reference_row_label.setText("Reference zone")
                self.trace_reference_stack.setCurrentIndex(0)
                self.trace_zone_spin.setRange(1, int(self.run_payload.summary["n_zones"]))
                self.trace_zone_spin.setValue(int(self._trace_reference_zone_index) + 1)
            else:
                label = f"Reference {self._coordinate_mode_text(mode)}"
                self.trace_reference_row_label.setText(f"{label} [{self._viewer_settings.length_unit}]")
                self.trace_reference_stack.setCurrentIndex(1)
                self.trace_coordinate_spin.setSuffix(f" {self._viewer_settings.length_unit}")
                self.trace_coordinate_spin.setRange(-1.0e12, 1.0e12)
                requested_cm = self._trace_reference_static_x_cm if mode == "static_x" else self._trace_reference_radius_cm
                self.trace_coordinate_spin.setValue(self._display_length_value(requested_cm))
        finally:
            self._suppress_trace_reference_updates = False

    def _trace_reference_coordinates_cm(self, mode: str, snapshot_index: int | None = None) -> np.ndarray | None:
        if self.run_payload is None:
            return None
        if mode == "static_x":
            return np.asarray(self.run_payload.static_x, dtype=np.float64)
        if mode == "moving_radius" and self.radius_payload is not None:
            if snapshot_index is None:
                snapshot_index = int(np.clip(self._trace_reference_anchor_snapshot, 0, self.run_payload.summary["n_snapshots"] - 1))
            return np.asarray(self.radius_payload.data[int(snapshot_index)], dtype=np.float64)
        return None

    def _resolve_trace_reference(self) -> dict[str, object] | None:
        if self.run_payload is None:
            return None
        mask = self._combined_zone_mask()
        active = np.flatnonzero(mask)
        if active.size == 0:
            return None
        mode = self._time_trace_reference_mode()
        if mode == "zone":
            requested_index = int(np.clip(self._trace_reference_zone_index, 0, self.run_payload.summary["n_zones"] - 1))
            actual_index = requested_index if bool(mask[requested_index]) else int(active[np.argmin(np.abs(active - requested_index))])
            return {
                "mode": mode,
                "zone_index": actual_index,
                "adjusted": actual_index != requested_index,
                "requested_zone": requested_index + 1,
                "actual_zone": actual_index + 1,
                "requested_display": float(requested_index + 1),
                "actual_display": float(actual_index + 1),
                "anchor_snapshot": int(self._trace_reference_anchor_snapshot),
            }
        anchor_snapshot = int(np.clip(self._trace_reference_anchor_snapshot, 0, self.run_payload.summary["n_snapshots"] - 1))
        coordinates = self._trace_reference_coordinates_cm(mode, anchor_snapshot)
        if coordinates is None or coordinates.size == 0:
            return None
        requested_cm = self._trace_reference_static_x_cm if mode == "static_x" else self._trace_reference_radius_cm
        active_coordinates = np.asarray(coordinates[active], dtype=np.float64)
        nearest_active = int(active[int(np.argmin(np.abs(active_coordinates - requested_cm)))])
        actual_cm = float(coordinates[nearest_active])
        return {
            "mode": mode,
            "zone_index": nearest_active,
            "adjusted": not np.isclose(actual_cm, float(requested_cm)),
            "requested_zone": int(self._trace_reference_zone_index) + 1,
            "actual_zone": nearest_active + 1,
            "requested_display": self._display_length_value(float(requested_cm)),
            "actual_display": self._display_length_value(actual_cm),
            "anchor_snapshot": anchor_snapshot,
        }

    def _combined_zone_mask(self) -> np.ndarray:
        if self.run_payload is None:
            return np.array([], dtype=bool)
        selected_regions = tuple(self._selected_region_ids())
        selected_materials = tuple(self._selected_material_ids())
        key = (
            int(self.run_payload.summary["n_zones"]),
            tuple(selected_regions),
            tuple(selected_materials),
        )
        if self._combined_zone_mask_key == key and self._combined_zone_mask_value is not None:
            return self._combined_zone_mask_value
        mask = np.ones(self.run_payload.summary["n_zones"], dtype=bool)
        if selected_regions and len(selected_regions) < self.region_list.count():
            mask &= np.isin(self.run_payload.zone_region_id, np.asarray(selected_regions, dtype=np.int32))
        elif not selected_regions:
            mask &= False
        if selected_materials and len(selected_materials) < self.material_list.count():
            mask &= np.isin(np.abs(self.run_payload.zone_material_index), np.abs(np.asarray(selected_materials, dtype=np.int32)))
        elif not selected_materials:
            mask &= False
        self._combined_zone_mask_key = key
        self._combined_zone_mask_value = mask
        return mask

    def _filter_context_text(self) -> str:
        if self.run_payload is None:
            return "Showing all zones."
        total = int(self.run_payload.summary["n_zones"])
        mask = self._combined_zone_mask()
        shown = int(mask.sum())
        selected_regions = self._selected_region_ids()
        selected_materials = self._selected_material_ids()
        if not selected_regions:
            return f"No regions selected (0/{total} zones)."
        if not selected_materials:
            return f"No materials selected (0/{total} zones)."
        if len(selected_regions) == self.region_list.count() and len(selected_materials) == self.material_list.count():
            return f"Showing all zones ({shown}/{total})."
        region_text = ", ".join(str(region_id) for region_id in selected_regions)
        material_text = f"; materials {', '.join(str(material_id) for material_id in selected_materials)}"
        return f"Showing regions {region_text}{material_text} ({shown}/{total} zones)."

    def _filter_brief_label(self) -> str:
        return self._filter_context_text().replace("Showing ", "").rstrip(".")

    def _update_filter_summary(self) -> None:
        self.filter_summary_label.setText(self._filter_context_text())
        if self.run_context.has_run:
            self.run_context.set_subset(
                region_ids=tuple(self._selected_region_ids()),
                material_ids=tuple(self._selected_material_ids()),
            )
            self._emit_run_context_changed()
        self._update_trace_reference_label()
        self._update_mouse_mode_state()
        self._mouse_plot_auto_range_pending = True

    def _sync_line_coordinate_to_map_coordinate(self) -> None:
        if self._configured_profile_coordinate_mode() != "viewer_follow":
            return
        preferred = self._preferred_line_coordinate_mode()
        if preferred != self._line_coordinate_mode():
            self._set_line_coordinate_mode(preferred)

    def _rebase_trace_reference_for_current_mode(self, previous_zone_index: int | None = None) -> None:
        if self.run_payload is None or self._slice_mode() != "time_trace":
            return
        resolved = self._resolve_trace_reference()
        if previous_zone_index is not None:
            zone_index = int(np.clip(previous_zone_index, 0, self.run_payload.summary["n_zones"] - 1))
        else:
            zone_index = int(resolved["zone_index"]) if resolved is not None else int(np.clip(self._trace_reference_zone_index, 0, self.run_payload.summary["n_zones"] - 1))
        mode = self._time_trace_reference_mode()
        if mode == "zone":
            self._trace_reference_zone_index = zone_index
        elif mode == "static_x":
            self._trace_reference_static_x_cm = float(self.run_payload.static_x[zone_index])
        elif mode == "moving_radius" and self.radius_payload is not None:
            snapshot_index = self._current_snapshot_index()
            self._trace_reference_anchor_snapshot = snapshot_index
            self._trace_reference_radius_cm = float(self.radius_payload.data[snapshot_index, zone_index])

    def _update_map_control_state(self) -> None:
        mode = str(self.clip_mode_combo.currentData())
        percentile_enabled = mode == "percentile"
        manual_enabled = mode == "manual"
        self.percentile_low_spin.setEnabled(percentile_enabled)
        self.percentile_high_spin.setEnabled(percentile_enabled)
        self.level_min_edit.setEnabled(manual_enabled)
        self.level_max_edit.setEnabled(manual_enabled)
        self.apply_levels_button.setEnabled(manual_enabled)
        allow_moving_mesh = self.run_payload is not None and self.run_payload.has_dynamic_radius and self.radius_payload is not None
        self._set_combo_item_enabled(self.map_coordinate_combo, "moving_radius", allow_moving_mesh)
        if not allow_moving_mesh and self._map_coordinate_mode() == "moving_radius":
            fallback_index = self.map_coordinate_combo.findData("static_x")
            if fallback_index < 0:
                fallback_index = self.map_coordinate_combo.findData("zone")
            self.map_coordinate_combo.setCurrentIndex(fallback_index)

    def _slice_mode(self) -> str:
        return str(self.slice_mode_combo.currentData())

    def _line_coordinate_mode(self) -> str:
        if self.run_context.has_run:
            return str(self.run_context.slice_coordinate)
        return str(self.line_coordinate_combo.currentData())

    def _map_coordinate_mode(self) -> str:
        if self.run_context.has_run:
            return str(self.run_context.map_coordinate)
        return str(self.map_coordinate_combo.currentData())

    def _field_map_view_context_key(self, *, coordinate_mode: str, orientation: str, render_mode: str) -> tuple[object, ...] | None:
        if self.run_payload is None:
            return None
        return (
            str(self.run_payload.path),
            "field_map",
            coordinate_mode,
            orientation,
            render_mode,
            int(self.run_payload.summary["n_zones"]),
            int(self.run_payload.summary["n_snapshots"]),
        )

    def _field_map_view_preservation_key(self) -> tuple[object, ...] | None:
        if self.run_payload is None:
            return None
        return (
            str(self.run_payload.path),
            "field_map",
            int(self.run_payload.summary["n_zones"]),
            int(self.run_payload.summary["n_snapshots"]),
        )

    def _line_plot_view_context_key(self) -> tuple[object, ...] | None:
        if self.run_payload is None:
            return None
        slice_mode = self._slice_mode()
        coordinate_mode = self._time_trace_reference_mode() if slice_mode == "time_trace" else self._line_coordinate_mode()
        return (
            str(self.run_payload.path),
            "lineout",
            slice_mode,
            coordinate_mode,
            self.current_field_name or "",
        )

    def _diagnostic_plot_view_context_key(self) -> tuple[object, ...] | None:
        if self.run_payload is None:
            return None
        return (
            str(self.run_payload.path),
            "diagnostic",
            self.current_diagnostic_path or "",
        )

    def _mouse_plot_view_context_key(self, name: str) -> tuple[object, ...] | None:
        if self.run_payload is None:
            return None
        probe_state = "active" if self._probe_snapshot_index is not None and self._probe_zone_index is not None else "empty"
        return (
            str(self.run_payload.path),
            "mouse",
            name,
            self._map_coordinate_mode(),
            self.current_field_name or "",
            probe_state,
        )

    def _update_slice_control_state(self) -> None:
        payload = self.run_payload
        slice_mode = self._slice_mode()
        allow_radius = (
            payload is not None
            and payload.has_dynamic_radius
            and self.radius_payload is not None
            and slice_mode == "snapshot_lineout"
        )
        allow_moving_radius = (
            payload is not None
            and payload.has_dynamic_radius
            and self.radius_payload is not None
            and slice_mode == "time_trace"
        )
        self._set_combo_item_enabled(self.line_coordinate_combo, "radius", allow_radius)
        self._set_combo_item_enabled(self.line_coordinate_combo, "moving_radius", allow_moving_radius)
        current_mode = self._line_coordinate_mode()
        if slice_mode == "time_trace" and current_mode == "radius":
            fallback_mode = self._preferred_line_coordinate_mode()
            if fallback_mode == "radius":
                fallback_mode = "moving_radius" if allow_moving_radius else "static_x"
            self._set_line_coordinate_mode(fallback_mode)
            current_mode = self._line_coordinate_mode()
        if slice_mode == "snapshot_lineout" and current_mode == "moving_radius":
            fallback_mode = self._preferred_line_coordinate_mode()
            if fallback_mode == "moving_radius":
                fallback_mode = "radius" if allow_radius else "static_x"
            self._set_line_coordinate_mode(fallback_mode)
        trace_enabled = slice_mode == "time_trace"
        self.trace_reference_row_label.setEnabled(trace_enabled and self.run_payload is not None)
        self.trace_reference_stack.setEnabled(trace_enabled and self.run_payload is not None)
        self.trace_reference_label.setEnabled(True)
        self._sync_trace_reference_editor_from_state()
        self._update_trace_reference_label()

    def _update_coordinate_note(self) -> None:
        map_orientation = str(self.map_orientation_combo.currentData())
        map_coordinate = self._map_coordinate_mode()
        slice_mode = self._slice_mode()
        line_coordinate = self._line_coordinate_mode()
        map_coordinate_text = self._coordinate_mode_text(map_coordinate)
        line_coordinate_text = self._coordinate_mode_text(line_coordinate)

        if map_orientation == "time_x_coord_y":
            map_note = f"time on x, {map_coordinate_text} on y"
        else:
            map_note = f"{map_coordinate_text} on x, time on y"

        if slice_mode == "time_trace":
            slice_note = f"time trace at fixed {line_coordinate_text} reference; slider controls the current-time cursor"
        else:
            slice_note = f"snapshot lineout versus {line_coordinate_text}"
        mesh_note = ""
        if map_coordinate == "moving_radius":
            mesh_note = (
                f" {self._coordinate_mode_text('moving_radius', capitalize=True)} uses the explicit HELIOS edge grid for 2D cells; "
                "lineouts and probe readouts stay zone-centered."
            )
        else:
            mesh_note = f" {self._coordinate_mode_text('radius', capitalize=True)} remains limited to snapshot lineouts outside moving-mesh 2D mode."
        legacy_note = ""
        if self.run_payload is not None and self.run_payload.has_dynamic_radius and map_coordinate == "static_x":
            legacy_note = f" {self._coordinate_mode_text('static_x', capitalize=True)} legacy/compatibility coordinate mode is active."
        self.coordinate_note_label.setText(f"2D map uses {map_note}. Slice view uses {slice_note}.{mesh_note}{legacy_note}")

    def _update_active_analysis_label(self) -> None:
        if self.plot_tabs.currentWidget() is self.mouse_tab:
            state = "frozen" if self._probe_mode == "frozen" and self._probe_zone_index is not None else "live hover"
            self.active_analysis_label.setText(
                f"Active analysis: mouse mode ({state}) | map={self._coordinate_mode_text(self._map_coordinate_mode())}"
            )
            return
        if self.plot_tabs.currentWidget() is self.diagnostic_plot:
            if self.current_diagnostic_payload is not None:
                self.active_analysis_label.setText(f"Active analysis: diagnostics ({_pretty_diagnostic(self.current_diagnostic_payload.path)})")
            elif self.current_diagnostic_path is not None:
                self.active_analysis_label.setText(f"Active analysis: diagnostics ({_pretty_diagnostic(self.current_diagnostic_path)})")
            else:
                self.active_analysis_label.setText("Active analysis: diagnostics")
            return
        if self._slice_mode() == "time_trace":
            self.active_analysis_label.setText(
                "Active analysis: time trace "
                f"({self._coordinate_mode_text(self._line_coordinate_mode())}) | "
                "slider controls current-time cursor"
            )
        else:
            self.active_analysis_label.setText(
                f"Active analysis: snapshot lineout ({self._coordinate_mode_text(self._line_coordinate_mode())})"
            )

    def _update_time_label(self, snapshot_index: int) -> None:
        if self.run_payload is None:
            self.snapshot_time_label.setText("time: -")
            return
        time_value = float(self.run_payload.time[snapshot_index])
        display_time = self._display_time_value(time_value)
        prefix = "current time" if self._slice_mode() == "time_trace" and self.plot_tabs.currentWidget() is self.lineout_plot else "time"
        self.snapshot_time_label.setText(f"{prefix}: {display_time:.4e} {self._viewer_settings.time_unit}")
        self.field_map_widget.set_time_marker(display_time)

    def _current_snapshot_index(self) -> int:
        if self.run_context.has_run:
            return int(self.run_context.snapshot_index)
        return int(self.snapshot_spin.value())

    def _set_snapshot_index(self, value: int) -> None:
        if self._updating_snapshot_controls:
            return
        self._updating_snapshot_controls = True
        try:
            if self.snapshot_slider.value() != value:
                self.snapshot_slider.setValue(value)
            if self.snapshot_spin.value() != value:
                self.snapshot_spin.setValue(value)
        finally:
            self._updating_snapshot_controls = False
        if self.run_context.has_run:
            self.run_context.set_snapshot_index(int(value))
            self._emit_run_context_changed()
        self._update_time_label(value)
        current_tab = self.plot_tabs.currentWidget()
        if current_tab is self.lineout_plot:
            self._refresh_line_plot(preserve_view=True)
        elif current_tab is self.mouse_tab:
            self._refresh_mouse_mode_plots()
        self._update_active_analysis_label()

    def _on_snapshot_changed(self, value: int) -> None:
        del value
        self._set_snapshot_index(int(self.sender().value()) if self.sender() is not None else self._current_snapshot_index())

    def _on_plot_tab_changed(self, index: int) -> None:
        del index
        current_tab = self.plot_tabs.currentWidget()
        mouse_active = current_tab is self.mouse_tab
        self.snapshot_controls_widget.setVisible((not self._use_external_snapshot_controls) and current_tab is self.lineout_plot)
        self.field_map_widget.set_probe_enabled(mouse_active)
        self._set_plot_navigation_mode("zoom" if self.plot_zoom_action.isChecked() else "pan")
        if mouse_active:
            self._mouse_plot_auto_range_pending = True
            self._sync_probe_overlay()
            self._refresh_mouse_mode_plots()
        else:
            self._hover_timer.stop()
            self._pending_probe_hover_position = None
            self.field_map_widget.clear_probe()
        self._refresh_field_map(preserve_view=True)
        if current_tab is self.lineout_plot:
            self._refresh_line_plot(preserve_view=True)
        elif current_tab is self.diagnostic_plot:
            self._refresh_diagnostic_plot(preserve_view=True)
        self._update_active_analysis_label()
        self._update_mouse_mode_state()

    def _resolved_trace_zone_index(self) -> int | None:
        resolved = self._resolve_trace_reference()
        if resolved is None:
            return None
        return int(resolved["zone_index"])

    def _update_trace_reference_label(self) -> None:
        if self.run_payload is None:
            self.trace_reference_label.setText("Reference: -")
            return
        resolved = self._resolve_trace_reference()
        if resolved is None:
            self.trace_reference_label.setText("Reference: no active zone in current subset")
            return
        mode = str(resolved["mode"])
        actual_zone = int(resolved["actual_zone"])
        adjusted = bool(resolved["adjusted"])
        if mode == "static_x":
            coordinate_text = self._coordinate_mode_text("static_x")
            if adjusted:
                self.trace_reference_label.setText(
                    f"Reference: selected {coordinate_text} = {float(resolved['requested_display']):.4e} {self._viewer_settings.length_unit} "
                    f"-> nearest active zone {actual_zone}, {coordinate_text} = {float(resolved['actual_display']):.4e} "
                    f"{self._viewer_settings.length_unit}; trace uses zone {actual_zone}"
                )
            else:
                self.trace_reference_label.setText(
                    f"Reference: {coordinate_text} = {float(resolved['actual_display']):.4e} {self._viewer_settings.length_unit} "
                    f"(trace uses zone {actual_zone})"
                )
        elif mode == "moving_radius":
            anchor_snapshot = int(resolved["anchor_snapshot"])
            coordinate_text = self._coordinate_mode_text("moving_radius")
            axis_text = self._coordinate_name()
            if adjusted:
                self.trace_reference_label.setText(
                    f"Reference: selected {coordinate_text} = {float(resolved['requested_display']):.4e} {self._viewer_settings.length_unit} "
                    f"at snapshot {anchor_snapshot} -> nearest active zone {actual_zone}, "
                    f"{axis_text} = {float(resolved['actual_display']):.4e} {self._viewer_settings.length_unit}; "
                    f"trace uses zone {actual_zone}"
                )
            else:
                self.trace_reference_label.setText(
                    f"Reference: {coordinate_text} = {float(resolved['actual_display']):.4e} {self._viewer_settings.length_unit} "
                    f"at snapshot {anchor_snapshot} (trace uses zone {actual_zone})"
                )
        else:
            if adjusted:
                self.trace_reference_label.setText(
                    f"Reference: requested zone {int(resolved['requested_zone'])} -> nearest active zone {actual_zone}"
                )
            else:
                self.trace_reference_label.setText(f"Reference: zone {actual_zone}")

    def _masked_matrix(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0 or np.all(mask):
            return np.asarray(data, dtype=np.float64)
        return np.where(mask[None, :], np.asarray(data, dtype=np.float64), np.nan)

    def _masked_vector(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0 or np.all(mask):
            return np.asarray(data, dtype=np.float64)
        return np.where(mask, np.asarray(data, dtype=np.float64), np.nan)

    def _region_boundary_positions(
        self,
        coordinate_mode: str,
        *,
        snapshot_index: int | None = None,
        active_mask: np.ndarray | None = None,
    ) -> list[float]:
        if self.run_payload is None:
            return []
        max_zones = np.asarray(self.run_payload.regions["max_zone_index"], dtype=np.int32)
        if max_zones.size <= 1:
            return []
        if active_mask is None:
            active = np.ones(self.run_payload.summary["n_zones"], dtype=bool)
        else:
            active = np.asarray(active_mask, dtype=bool)
        if coordinate_mode == "zone":
            positions: list[float] = []
            for max_zone in max_zones[:-1]:
                left_index = int(max_zone) - 1
                right_index = int(max_zone)
                if right_index >= active.size:
                    continue
                if not (active[left_index] or active[right_index]):
                    continue
                positions.append(float(max_zone) + 0.5)
            return positions

        if coordinate_mode == "static_x":
            edges = self._static_x_edge_values()
        elif coordinate_mode == "radius":
            edges = self._dynamic_radius_edge_values(snapshot_index)
            if edges is None:
                return []
        else:
            return []

        positions: list[float] = []
        for max_zone in max_zones[:-1]:
            left_index = int(max_zone) - 1
            right_index = int(max_zone)
            if right_index >= active.size or int(max_zone) >= edges.size:
                continue
            if not (active[left_index] or active[right_index]):
                continue
            positions.append(float(edges[int(max_zone)]))
        return positions

    def _invalidate_moving_mesh_cache(self) -> None:
        self._moving_mesh_cache.clear()
        self._moving_mesh_cache_hits = 0
        self._moving_mesh_cache_misses = 0

    def _cache_lookup(self, key: tuple[object, ...], builder):
        cached = self._moving_mesh_cache.get(key)
        if cached is not None:
            self._moving_mesh_cache_hits += 1
            return cached
        value = builder()
        self._moving_mesh_cache[key] = value
        self._moving_mesh_cache_misses += 1
        return value

    @staticmethod
    def _centers_to_edges(values: np.ndarray) -> np.ndarray:
        centers = np.asarray(values, dtype=np.float64)
        if centers.size == 0:
            return np.array([], dtype=np.float64)
        if centers.size == 1:
            delta = abs(float(centers[0])) * 0.5
            if delta == 0.0:
                delta = 0.5
            return np.asarray([float(centers[0]) - delta, float(centers[0]) + delta], dtype=np.float64)
        edges = np.empty(centers.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - (edges[1] - centers[0])
        edges[-1] = centers[-1] + (centers[-1] - edges[-2])
        return edges

    @classmethod
    def _edge_rows_to_corner_grid(cls, values: np.ndarray) -> np.ndarray:
        edge_rows = np.asarray(values, dtype=np.float64)
        if edge_rows.ndim != 2:
            raise ValueError("Moving-mesh edges must be a 2D array.")
        n_time, n_edge = edge_rows.shape
        corners = np.empty((n_time + 1, n_edge), dtype=np.float64)
        if n_time == 1:
            corners[0] = edge_rows[0]
            corners[1] = edge_rows[0]
        else:
            corners[1:-1] = 0.5 * (edge_rows[:-1] + edge_rows[1:])
            corners[0] = edge_rows[0] - (corners[1] - edge_rows[0])
            corners[-1] = edge_rows[-1] + (edge_rows[-1] - corners[-2])
        return corners

    def _zone_coordinate_edges(self) -> np.ndarray:
        if self.run_payload is None:
            return np.array([], dtype=np.float64)
        zone_count = int(self.run_payload.summary["n_zones"])
        return np.linspace(0.5, zone_count + 0.5, zone_count + 1, dtype=np.float64)

    def _static_x_edge_values(self) -> np.ndarray:
        if self.run_payload is None:
            return np.array([], dtype=np.float64)
        return self._display_length_values(np.asarray(self.run_payload.static_x_edges, dtype=np.float64))

    def _dynamic_radius_edge_values(self, snapshot_index: int | None) -> np.ndarray | None:
        if self.radius_payload is None or snapshot_index is None:
            return None
        if self.radius_payload.edge_data is not None:
            return self._display_length_values(np.asarray(self.radius_payload.edge_data[int(snapshot_index)], dtype=np.float64))
        radius_bundle = self._display_radius_bundle()
        if radius_bundle is None:
            return None
        radius_display, _ = radius_bundle
        return self._centers_to_edges(np.asarray(radius_display[int(snapshot_index)], dtype=np.float64))

    @classmethod
    def _centers_to_corner_grid(cls, values: np.ndarray) -> np.ndarray:
        centers = np.asarray(values, dtype=np.float64)
        if centers.ndim != 2:
            raise ValueError("Moving-mesh centers must be a 2D array.")
        n_time, n_zone = centers.shape
        zone_edges = np.empty((n_time, n_zone + 1), dtype=np.float64)
        if n_zone == 1:
            delta = np.maximum(np.abs(centers[:, 0]) * 0.5, 0.5)
            zone_edges[:, 0] = centers[:, 0] - delta
            zone_edges[:, 1] = centers[:, 0] + delta
        else:
            zone_edges[:, 1:-1] = 0.5 * (centers[:, :-1] + centers[:, 1:])
            zone_edges[:, 0] = centers[:, 0] - (zone_edges[:, 1] - centers[:, 0])
            zone_edges[:, -1] = centers[:, -1] + (centers[:, -1] - zone_edges[:, -2])

        corners = np.empty((n_time + 1, n_zone + 1), dtype=np.float64)
        if n_time == 1:
            corners[0] = zone_edges[0]
            corners[1] = zone_edges[0]
        else:
            corners[1:-1] = 0.5 * (zone_edges[:-1] + zone_edges[1:])
            corners[0] = zone_edges[0] - (corners[1] - zone_edges[0])
            corners[-1] = zone_edges[-1] + (zone_edges[-1] - corners[-2])
        return corners

    def _inactive_coordinate_ranges(
        self,
        coordinate_mode: str,
        mask: np.ndarray,
        *,
        snapshot_index: int | None = None,
    ) -> list[tuple[float, float]]:
        if self.run_payload is None or mask.size == 0 or np.all(mask) or coordinate_mode == "moving_radius":
            return []
        if coordinate_mode == "zone":
            edges = self._zone_coordinate_edges()
        elif coordinate_mode == "static_x":
            edges = self._static_x_edge_values()
        elif coordinate_mode == "radius" and self.radius_payload is not None and snapshot_index is not None:
            edges = self._dynamic_radius_edge_values(snapshot_index)
            if edges is None:
                return []
        else:
            return []
        inactive = np.logical_not(mask)
        ranges: list[tuple[float, float]] = []
        start_index: int | None = None
        for index, is_inactive in enumerate(inactive):
            if is_inactive and start_index is None:
                start_index = index
            elif not is_inactive and start_index is not None:
                ranges.append((float(edges[start_index]), float(edges[index])))
                start_index = None
        if start_index is not None:
            ranges.append((float(edges[start_index]), float(edges[inactive.size])))
        return ranges

    def _moving_mesh_surface(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self.run_payload is None or self.radius_payload is None:
            return None
        orientation = str(self.map_orientation_combo.currentData())
        radius_bundle = self._display_radius_bundle()
        if radius_bundle is None:
            return None
        radius_data, _ = radius_bundle
        time_values = self._display_time_values(self.run_payload.time)

        time_edges = self._cache_lookup(
            ("time_edges", self._viewer_settings.time_unit, time_values.shape, float(time_values[0]), float(time_values[-1])),
            lambda: self._centers_to_edges(time_values),
        )
        radius_edge_data = self.radius_payload.edge_data
        if radius_edge_data is not None:
            radius_corners = self._cache_lookup(
                (
                    "radius_edge_corners",
                    self._viewer_settings.length_unit,
                    radius_edge_data.shape,
                    float(radius_edge_data[0, 0]),
                    float(radius_edge_data[-1, -1]),
                ),
                lambda: self._edge_rows_to_corner_grid(self._display_length_values(np.asarray(radius_edge_data, dtype=np.float64))),
            )
        else:
            radius_corners = self._cache_lookup(
                ("radius_corners", self._viewer_settings.length_unit, radius_data.shape, float(radius_data[0, 0]), float(radius_data[-1, -1])),
                lambda: self._centers_to_corner_grid(radius_data),
            )
        time_grid = self._cache_lookup(
            ("time_grid", self._viewer_settings.time_unit, self._viewer_settings.length_unit, time_edges.shape, radius_corners.shape),
            lambda: np.broadcast_to(np.asarray(time_edges, dtype=np.float64)[:, None], np.asarray(radius_corners, dtype=np.float64).shape),
        )
        z_values = np.asarray(values, dtype=np.float64)
        if orientation == "time_x_coord_y":
            return np.asarray(time_grid, dtype=np.float64), np.asarray(radius_corners, dtype=np.float64), z_values
        return np.asarray(radius_corners, dtype=np.float64), np.asarray(time_grid, dtype=np.float64), z_values

    def _moving_mesh_boundary_curves(self, mask: np.ndarray) -> list[np.ndarray]:
        if self.run_payload is None or self.radius_payload is None:
            return []
        orientation = str(self.map_orientation_combo.currentData())
        mask_key = bytes(np.packbits(np.asarray(mask, dtype=np.uint8))) if mask.size else b""
        radius_bundle = self._display_radius_bundle()
        if radius_bundle is None:
            return []
        radius_data, _ = radius_bundle

        def build() -> list[np.ndarray]:
            max_zones = np.asarray(self.run_payload.regions["max_zone_index"], dtype=np.int32)
            time_values = self._display_time_values(self.run_payload.time)
            curves: list[np.ndarray] = []
            radius_edges = self.radius_payload.edge_data
            for max_zone in max_zones[:-1]:
                left_index = int(max_zone) - 1
                right_index = int(max_zone)
                if right_index >= mask.size or not (mask[left_index] or mask[right_index]):
                    continue
                if radius_edges is not None and int(max_zone) < radius_edges.shape[1]:
                    boundary = self._display_length_values(np.asarray(radius_edges[:, int(max_zone)], dtype=np.float64))
                else:
                    boundary = 0.5 * (radius_data[:, left_index] + radius_data[:, right_index])
                if orientation == "time_x_coord_y":
                    curves.append(np.column_stack([time_values, boundary]))
                else:
                    curves.append(np.column_stack([boundary, time_values]))
            return curves

        cached = self._cache_lookup(("boundary_curves", orientation, self._viewer_settings.time_unit, self._viewer_settings.length_unit, mask_key), build)
        return [np.asarray(curve, dtype=np.float64) for curve in cached]

    def _moving_mesh_reference_curve(self) -> np.ndarray | None:
        if self.run_payload is None or self.radius_payload is None or self._slice_mode() != "time_trace":
            return None
        zone_index = self._resolved_trace_zone_index()
        if zone_index is None:
            return None
        orientation = str(self.map_orientation_combo.currentData())
        radius_bundle = self._display_radius_bundle()
        if radius_bundle is None:
            return None
        radius_data, _ = radius_bundle

        def build() -> np.ndarray:
            radius_trace = radius_data[:, zone_index]
            time_values = self._display_time_values(self.run_payload.time)
            if orientation == "time_x_coord_y":
                return np.column_stack([time_values, radius_trace])
            return np.column_stack([radius_trace, time_values])

        cached = self._cache_lookup(
            ("reference_curve", orientation, self._viewer_settings.time_unit, self._viewer_settings.length_unit, int(zone_index)),
            build,
        )
        return np.asarray(cached, dtype=np.float64)

    def _laser_entry_overlay(self) -> tuple[float | None, np.ndarray | None]:
        if self.run_payload is None or self._laser_entry_info is None:
            return None, None
        boundary_kind = str(self._laser_entry_info.get("boundary_kind", ""))
        if boundary_kind not in {"low", "high"}:
            return None, None
        first_zone = int(self._laser_entry_info.get("first_physical_zone", 0))
        mask = self._combined_zone_mask()
        if mask.size and 1 <= first_zone <= mask.size and not bool(mask[first_zone - 1]):
            return None, None
        coordinate_mode = self._map_coordinate_mode()
        orientation = str(self.map_orientation_combo.currentData())
        if coordinate_mode == "zone":
            boundary_position = 0.5 if boundary_kind == "low" else float(self.run_payload.summary["n_zones"]) + 0.5
            return boundary_position, None
        if coordinate_mode == "static_x":
            edges = self._static_x_edge_values()
            boundary_position = float(edges[0] if boundary_kind == "low" else edges[-1])
            return boundary_position, None
        if coordinate_mode == "moving_radius" and self.radius_payload is not None:
            surface = self._moving_mesh_surface(np.asarray(self.current_field_payload.data, dtype=np.float64)) if self.current_field_payload is not None else None
            if surface is None:
                return None, None
            mesh_x, mesh_y, _ = surface
            if orientation == "time_x_coord_y":
                curve_y = np.asarray(mesh_y[:-1, 0] if boundary_kind == "low" else mesh_y[:-1, -1], dtype=np.float64)
                return None, np.column_stack([self._display_time_values(self.run_payload.time), curve_y])
            curve_x = np.asarray(mesh_x[:-1, 0] if boundary_kind == "low" else mesh_x[:-1, -1], dtype=np.float64)
            return None, np.column_stack([curve_x, self._display_time_values(self.run_payload.time)])
        return None, None

    @staticmethod
    def _nearest_monotonic_index(values: np.ndarray, target: float) -> int:
        array = np.asarray(values, dtype=np.float64)
        if array.size <= 1:
            return 0
        if np.all(np.diff(array) >= 0.0):
            index = int(np.searchsorted(array, target))
            if index <= 0:
                return 0
            if index >= array.size:
                return int(array.size - 1)
            return int(index - 1) if abs(target - array[index - 1]) <= abs(array[index] - target) else int(index)
        return int(np.argmin(np.abs(array - target)))

    def _resolve_probe_indices(self, x_value: float, y_value: float) -> tuple[int, int] | None:
        if self.run_payload is None or self.current_field_payload is None:
            return None
        if self.map_orientation_combo.currentData() == "time_x_coord_y":
            time_coordinate = x_value
            spatial_coordinate = y_value
        else:
            time_coordinate = y_value
            spatial_coordinate = x_value

        time_values = self._display_time_values(self.run_payload.time)
        snapshot_index = self._nearest_monotonic_index(time_values, float(time_coordinate))
        coordinate_mode = self._map_coordinate_mode()
        if coordinate_mode == "zone":
            coordinate_values = np.arange(1, self.run_payload.summary["n_zones"] + 1, dtype=np.float64)
        elif coordinate_mode == "moving_radius" and self.radius_payload is not None:
            coordinate_values = self._display_length_values(self.radius_payload.data[snapshot_index])
        else:
            coordinate_values = self._display_length_values(self.run_payload.static_x)
        zone_index = self._nearest_monotonic_index(coordinate_values, float(spatial_coordinate))
        return snapshot_index, zone_index

    def _current_probe_coordinate(self) -> float | None:
        if self.run_payload is None or self._probe_zone_index is None:
            return None
        coordinate_mode = self._map_coordinate_mode()
        if coordinate_mode == "zone":
            return float(self._probe_zone_index + 1)
        if coordinate_mode == "moving_radius" and self.radius_payload is not None and self._probe_snapshot_index is not None:
            return self._display_length_value(float(self.radius_payload.data[self._probe_snapshot_index, self._probe_zone_index]))
        return self._display_length_value(float(self.run_payload.static_x[self._probe_zone_index]))

    def _probe_coordinate_values(self, snapshot_index: int) -> np.ndarray:
        if self.run_payload is None:
            return np.array([], dtype=np.float64)
        coordinate_mode = self._map_coordinate_mode()
        if coordinate_mode == "zone":
            return np.arange(1, self.run_payload.summary["n_zones"] + 1, dtype=np.float64)
        if coordinate_mode == "moving_radius" and self.radius_payload is not None:
            return self._display_length_values(self.radius_payload.data[snapshot_index])
        return self._display_length_values(self.run_payload.static_x)

    def _sync_mouse_adjustment_controls(self) -> None:
        mouse_active = self.plot_tabs.currentWidget() is self.mouse_tab
        if self.run_payload is None or not mouse_active:
            for widget in (self.mouse_time_slider, self.mouse_time_spin, self.mouse_coordinate_slider, self.mouse_coordinate_spin):
                widget.setEnabled(False)
            self.mouse_time_row_label.setText("Time")
            self.mouse_coordinate_row_label.setText("Coordinate")
            self.mouse_adjustment_label.setText("Hover or freeze a point to enable fine adjustment.")
            return
        self.mouse_time_slider.setEnabled(True)
        self.mouse_time_spin.setEnabled(True)
        self.mouse_coordinate_slider.setEnabled(True)
        self.mouse_coordinate_spin.setEnabled(True)
        snapshot_index = int(self._probe_snapshot_index if self._probe_snapshot_index is not None else self._current_snapshot_index())
        zone_index = int(self._probe_zone_index if self._probe_zone_index is not None else 0)
        time_values = self._display_time_values(np.asarray(self.run_payload.time, dtype=np.float64))
        coordinate_values = self._probe_coordinate_values(snapshot_index)
        coordinate_mode = self._map_coordinate_mode()
        self._suppress_mouse_adjust_updates = True
        try:
            self.mouse_time_row_label.setText(f"Time [{self._viewer_settings.time_unit}]")
            self.mouse_time_slider.setRange(0, int(self.run_payload.summary["n_snapshots"]) - 1)
            self.mouse_time_spin.setDecimals(12)
            self.mouse_time_spin.setSuffix(f" {self._viewer_settings.time_unit}")
            self.mouse_time_spin.setRange(float(np.min(time_values)), float(np.max(time_values)))
            self.mouse_time_slider.setValue(snapshot_index)
            self.mouse_time_spin.setValue(float(time_values[snapshot_index]))

            self.mouse_coordinate_slider.setRange(0, int(self.run_payload.summary["n_zones"]) - 1)
            self.mouse_coordinate_slider.setValue(zone_index)
            if coordinate_mode == "zone":
                self.mouse_coordinate_row_label.setText("Zone index")
                self.mouse_coordinate_spin.setDecimals(0)
                self.mouse_coordinate_spin.setSuffix("")
                self.mouse_coordinate_spin.setRange(1.0, float(self.run_payload.summary["n_zones"]))
                self.mouse_coordinate_spin.setValue(float(zone_index + 1))
                self.mouse_adjustment_label.setText(
                    f"Resolved: snapshot {snapshot_index}, zone {zone_index + 1}. Controls use discrete samples."
                )
            else:
                coordinate_label = self._coordinate_mode_text(coordinate_mode, capitalize=True)
                coordinate_value = float(coordinate_values[zone_index]) if coordinate_values.size else 0.0
                self.mouse_coordinate_row_label.setText(f"{coordinate_label} [{self._viewer_settings.length_unit}]")
                self.mouse_coordinate_spin.setDecimals(12)
                self.mouse_coordinate_spin.setSuffix(f" {self._viewer_settings.length_unit}")
                if coordinate_values.size:
                    self.mouse_coordinate_spin.setRange(float(np.min(coordinate_values)), float(np.max(coordinate_values)))
                self.mouse_coordinate_spin.setValue(coordinate_value)
                self.mouse_adjustment_label.setText(
                    f"Resolved: snapshot {snapshot_index}, {coordinate_label.lower()} = {coordinate_value:.4e} "
                    f"{self._viewer_settings.length_unit}, zone {zone_index + 1}. Typed values snap to the nearest cell."
                )
        finally:
            self._suppress_mouse_adjust_updates = False

    def _on_mouse_time_slider_changed(self, value: int) -> None:
        if self._suppress_mouse_adjust_updates or self.run_payload is None:
            return
        zone_index = int(self._probe_zone_index if self._probe_zone_index is not None else 0)
        self._mouse_plot_auto_range_pending = True
        self._set_probe_selection(int(value), zone_index, frozen=True)

    def _on_mouse_time_spin_changed(self, value: float) -> None:
        if self._suppress_mouse_adjust_updates or self.run_payload is None:
            return
        time_values = self._display_time_values(np.asarray(self.run_payload.time, dtype=np.float64))
        snapshot_index = self._nearest_monotonic_index(time_values, float(value))
        zone_index = int(self._probe_zone_index if self._probe_zone_index is not None else 0)
        self._mouse_plot_auto_range_pending = True
        self._set_probe_selection(snapshot_index, zone_index, frozen=True)

    def _on_mouse_coordinate_slider_changed(self, value: int) -> None:
        if self._suppress_mouse_adjust_updates or self.run_payload is None:
            return
        snapshot_index = int(self._probe_snapshot_index if self._probe_snapshot_index is not None else self._current_snapshot_index())
        self._mouse_plot_auto_range_pending = True
        self._set_probe_selection(snapshot_index, int(value), frozen=True)

    def _on_mouse_coordinate_spin_changed(self, value: float) -> None:
        if self._suppress_mouse_adjust_updates or self.run_payload is None:
            return
        snapshot_index = int(self._probe_snapshot_index if self._probe_snapshot_index is not None else self._current_snapshot_index())
        coordinate_values = self._probe_coordinate_values(snapshot_index)
        if coordinate_values.size == 0:
            return
        zone_index = self._nearest_monotonic_index(coordinate_values, float(value))
        self._mouse_plot_auto_range_pending = True
        self._set_probe_selection(snapshot_index, zone_index, frozen=True)

    def _sync_probe_overlay(self) -> None:
        if self.run_payload is None or self._probe_snapshot_index is None or self._probe_zone_index is None:
            self.field_map_widget.clear_probe()
            return
        if self.plot_tabs.currentWidget() is not self.mouse_tab:
            self.field_map_widget.clear_probe()
            return
        time_value = self._display_time_value(float(self.run_payload.time[self._probe_snapshot_index]))
        coordinate_value = self._current_probe_coordinate()
        if coordinate_value is None:
            self.field_map_widget.clear_probe()
            return
        if self.map_orientation_combo.currentData() == "time_x_coord_y":
            x_value = time_value
            y_value = coordinate_value
        else:
            x_value = coordinate_value
            y_value = time_value
        self.field_map_widget.set_probe_point(x_value, y_value, frozen=self._probe_mode == "frozen")

    def _update_mouse_mode_state(self) -> None:
        if self.run_payload is None or self.current_field_payload is None:
            self.mouse_mode_state_label.setText("Probe: inactive")
            self.mouse_mode_probe_label.setText("Open a HELIOS run to enable interactive probing.")
            self.resume_hover_button.setEnabled(False)
            self._sync_mouse_adjustment_controls()
            return
        if self._probe_snapshot_index is None or self._probe_zone_index is None:
            self.mouse_mode_state_label.setText("Probe: live hover")
            self.mouse_mode_probe_label.setText("Move the mouse over the 2D map to inspect slices.")
            self.resume_hover_button.setEnabled(False)
            self._sync_mouse_adjustment_controls()
            return

        display_bundle = self._current_display_field_bundle()
        if display_bundle is None:
            return
        display_field, display_unit, pretty_field = display_bundle
        time_value = self._display_time_value(float(self.run_payload.time[self._probe_snapshot_index]))
        coordinate_mode = self._map_coordinate_mode()
        if coordinate_mode == "zone":
            coordinate_text = f"zone {self._probe_zone_index + 1}"
        elif coordinate_mode == "moving_radius" and self.radius_payload is not None:
            radius_bundle = self._display_radius_bundle()
            if radius_bundle is None:
                return
            radius_display, radius_unit = radius_bundle
            coordinate_text = (
                f"{self._coordinate_mode_text('moving_radius')} = {float(radius_display[self._probe_snapshot_index, self._probe_zone_index]):.4e} "
                f"{radius_unit}"
            )
        else:
            coordinate_text = (
                f"{self._coordinate_mode_text('static_x')} = {self._display_length_value(float(self.run_payload.static_x[self._probe_zone_index])):.4e} "
                f"{self._viewer_settings.length_unit}"
            )

        mask = self._combined_zone_mask()
        value = float(display_field[self._probe_snapshot_index, self._probe_zone_index])
        if mask.size and not bool(mask[self._probe_zone_index]):
            value_text = "masked / inactive in current subset"
        else:
            value_text = f"{pretty_field} = {value:.6e} {display_unit}".rstrip()

        if self._probe_mode == "frozen":
            self.mouse_mode_state_label.setText("Probe: frozen")
            self.resume_hover_button.setEnabled(True)
        else:
            self.mouse_mode_state_label.setText("Probe: live hover")
            self.resume_hover_button.setEnabled(False)
        self.mouse_mode_probe_label.setText(
            f"t = {time_value:.4e} {self._viewer_settings.time_unit} | {coordinate_text} | "
            f"zone {self._probe_zone_index + 1} | {value_text}"
        )
        self._sync_mouse_adjustment_controls()

    def _set_probe_selection(self, snapshot_index: int, zone_index: int, *, frozen: bool) -> None:
        self._probe_snapshot_index = int(snapshot_index)
        self._probe_zone_index = int(zone_index)
        self._probe_mode = "frozen" if frozen else "live"
        self._sync_probe_overlay()
        self._refresh_mouse_mode_plots()
        self._update_mouse_mode_state()
        self._update_active_analysis_label()

    def _on_map_probe_moved(self, x_value: float, y_value: float) -> None:
        self._probe_hover_position = (float(x_value), float(y_value))
        if self.plot_tabs.currentWidget() is not self.mouse_tab or self._probe_mode == "frozen":
            return
        self._pending_probe_hover_position = self._probe_hover_position
        interval = max(0, int(self._viewer_settings.hover_interval_ms))
        if interval <= 0:
            self._consume_pending_probe_hover()
            return
        now = time.perf_counter()
        if not self._hover_timer.isActive() and (now - self._last_hover_update_at) * 1000.0 >= interval:
            self._consume_pending_probe_hover()
            return
        if not self._hover_timer.isActive():
            self._hover_timer.start(interval)

    def _on_map_probe_clicked(self, x_value: float, y_value: float) -> None:
        self._probe_hover_position = (float(x_value), float(y_value))
        self._pending_probe_hover_position = None
        self._hover_timer.stop()
        if self.plot_tabs.currentWidget() is not self.mouse_tab:
            return
        resolved = self._resolve_probe_indices(x_value, y_value)
        if resolved is None:
            return
        self._mouse_plot_auto_range_pending = True
        self._set_probe_selection(resolved[0], resolved[1], frozen=True)

    def _consume_pending_probe_hover(self) -> None:
        if self._pending_probe_hover_position is None:
            return
        if self.plot_tabs.currentWidget() is not self.mouse_tab or self._probe_mode == "frozen":
            self._pending_probe_hover_position = None
            return
        pending = self._pending_probe_hover_position
        self._pending_probe_hover_position = None
        started = time.perf_counter()
        resolved = self._resolve_probe_indices(*pending)
        if resolved is not None:
            self._mouse_plot_auto_range_pending = True
            self._set_probe_selection(resolved[0], resolved[1], frozen=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._last_hover_update_at = time.perf_counter()
        self._hover_profile_count += 1
        self._hover_profile_total_ms += elapsed_ms
        self._hover_profile_max_ms = max(self._hover_profile_max_ms, elapsed_ms)
        if self._pending_probe_hover_position is not None and self._probe_mode != "frozen":
            self._hover_timer.start(max(0, int(self._viewer_settings.hover_interval_ms)))

    def _resume_hover_probe(self) -> None:
        self._probe_mode = "live"
        if self._probe_hover_position is None:
            self._update_mouse_mode_state()
            self._update_active_analysis_label()
            return
        resolved = self._resolve_probe_indices(*self._probe_hover_position)
        if resolved is None:
            self._probe_snapshot_index = None
            self._probe_zone_index = None
            self.field_map_widget.clear_probe()
            self._update_mouse_mode_state()
            self._update_active_analysis_label()
            return
        self._set_probe_selection(resolved[0], resolved[1], frozen=False)

    def _refresh_mouse_mode_plots(self) -> None:
        selected_colormap = str(self.colormap_combo.currentData())
        self.mouse_vertical_plot.set_colormap(selected_colormap)
        self.mouse_horizontal_plot.set_colormap(selected_colormap)
        if self.run_payload is None or self.current_field_payload is None:
            return
        display_bundle = self._current_display_field_bundle()
        if display_bundle is None:
            return
        display_field, display_unit, pretty_field = display_bundle
        display_time = self._display_time_values(np.asarray(self.run_payload.time, dtype=np.float64))
        if self._probe_snapshot_index is None or self._probe_zone_index is None:
            self.mouse_vertical_plot.set_curves(
                np.array([0.0], dtype=np.float64),
                [np.array([np.nan], dtype=np.float64)],
                title="Move over the map to create a vertical slice",
                x_label="Coordinate",
                y_label="Field value",
                value_scale_mode="linear",
                auto_range=True,
                view_context_key=self._mouse_plot_view_context_key("vertical"),
            )
            self.mouse_horizontal_plot.set_curves(
                np.array([0.0], dtype=np.float64),
                [np.array([np.nan], dtype=np.float64)],
                title="Move over the map to create a horizontal slice",
                x_label=f"Time [{self._viewer_settings.time_unit}]",
                y_label="Field value",
                value_scale_mode="linear",
                auto_range=True,
                view_context_key=self._mouse_plot_view_context_key("horizontal"),
            )
            self._mouse_plot_auto_range_pending = True
            return

        mask = self._combined_zone_mask()
        snapshot_index = int(self._probe_snapshot_index)
        zone_index = int(self._probe_zone_index)
        time_value = float(display_time[snapshot_index])
        requested_mode = str(self.line_scale_combo.currentData())
        state_note = "frozen" if self._probe_mode == "frozen" else "hover"
        coordinate_mode = self._map_coordinate_mode()

        if coordinate_mode == "moving_radius" and self.radius_payload is not None:
            radius_bundle = self._display_radius_bundle()
            if radius_bundle is None:
                return
            radius_display, radius_unit = radius_bundle
            vertical_x = np.asarray(radius_display[snapshot_index], dtype=np.float64)
            vertical_label = f"{self._coordinate_mode_text('moving_radius', capitalize=True)} [{radius_unit}]"
            vertical_note = self._coordinate_mode_text("moving_radius")
            vertical_boundaries = self._region_boundary_positions("radius", snapshot_index=snapshot_index, active_mask=mask)
        elif coordinate_mode == "zone":
            vertical_x = np.arange(1, self.run_payload.summary["n_zones"] + 1, dtype=np.float64)
            vertical_label = "Zone index"
            vertical_note = "zone index"
            vertical_boundaries = self._region_boundary_positions("zone", active_mask=mask)
        else:
            vertical_x = self._display_length_values(np.asarray(self.run_payload.static_x, dtype=np.float64))
            vertical_label = f"{self._coordinate_mode_text('static_x', capitalize=True)} [{self._viewer_settings.length_unit}]"
            vertical_note = self._coordinate_mode_text("static_x")
            vertical_boundaries = self._region_boundary_positions("static_x", active_mask=mask)

        vertical_y = self._masked_vector(display_field[snapshot_index], mask)
        vertical_scale_mode, vertical_scale_note = self._effective_value_scale_mode(vertical_y, requested_mode)
        self.mouse_vertical_plot.set_curves(
            vertical_x,
            [vertical_y],
            title=(
                f"{pretty_field} vertical slice | {state_note} | snapshot {snapshot_index} | "
                f"t={time_value:.4e} {self._viewer_settings.time_unit} | {vertical_note} | {vertical_scale_note}"
            ),
            x_label=vertical_label,
            y_label=_label_with_unit(pretty_field, display_unit),
            curve_names=[pretty_field],
            value_scale_mode=vertical_scale_mode,
            boundary_positions=vertical_boundaries,
            show_boundaries=bool(self.boundary_overlay_checkbox.isChecked()),
            auto_range=self._mouse_plot_auto_range_pending,
            view_context_key=self._mouse_plot_view_context_key("vertical"),
        )

        if mask.size and not bool(mask[zone_index]):
            horizontal_y = np.full(self.run_payload.time.shape, np.nan, dtype=np.float64)
        else:
            horizontal_y = np.asarray(display_field[:, zone_index], dtype=np.float64)
        horizontal_scale_mode, horizontal_scale_note = self._effective_value_scale_mode(horizontal_y, requested_mode)
        if coordinate_mode == "moving_radius" and self.radius_payload is not None:
            radius_bundle = self._display_radius_bundle()
            if radius_bundle is None:
                return
            radius_display, radius_unit = radius_bundle
            horizontal_note = (
                f"zone {zone_index + 1} | hovered {self._coordinate_name()} = "
                f"{float(radius_display[snapshot_index, zone_index]):.4e} {radius_unit}"
            )
        elif coordinate_mode == "static_x":
            horizontal_note = (
                f"zone {zone_index + 1} | {self._coordinate_mode_text('static_x')} = "
                f"{self._display_length_value(float(self.run_payload.static_x[zone_index])):.4e} {self._viewer_settings.length_unit}"
            )
        else:
            horizontal_note = f"zone {zone_index + 1}"
        self.mouse_horizontal_plot.set_curves(
            display_time,
            [horizontal_y],
            title=f"{pretty_field} horizontal slice | {state_note} | {horizontal_note} | {horizontal_scale_note}",
            x_label=f"Time [{self._viewer_settings.time_unit}]",
            y_label=_label_with_unit(pretty_field, display_unit),
            curve_names=[pretty_field],
            value_scale_mode=horizontal_scale_mode,
            boundary_positions=None,
            show_boundaries=False,
            auto_range=self._mouse_plot_auto_range_pending,
            cursor_position=time_value,
            show_cursor=True,
            view_context_key=self._mouse_plot_view_context_key("horizontal"),
        )
        self._mouse_plot_auto_range_pending = False

    def _prepare_display_image(self, data: np.ndarray) -> tuple[np.ndarray, str]:
        display = np.asarray(data, dtype=np.float64)
        mode = str(self.map_scale_combo.currentData())
        if mode == "log10":
            positive = display > 0.0
            transformed = np.full(display.shape, np.nan, dtype=np.float64)
            transformed[positive] = np.log10(display[positive])
            return transformed, "log10 positive values"
        return display, "linear scale"

    def _current_map_levels(self, image: np.ndarray) -> tuple[tuple[float, float] | None, bool]:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return None, True
        mode = str(self.clip_mode_combo.currentData())
        if mode == "auto":
            return None, True
        if mode == "percentile":
            low = float(self.percentile_low_spin.value())
            high = float(self.percentile_high_spin.value())
            if high <= low:
                high = min(100.0, low + 1.0)
            values = np.nanpercentile(finite, [low, high])
            return (float(values[0]), float(values[1])), False
        min_text = self.level_min_edit.text().strip()
        max_text = self.level_max_edit.text().strip()
        if not min_text or not max_text:
            return None, True
        try:
            min_value = float(min_text)
            max_value = float(max_text)
        except ValueError:
            return None, True
        if max_value < min_value:
            min_value, max_value = max_value, min_value
        if max_value == min_value:
            max_value = min_value + 1.0
        return (min_value, max_value), False

    def _refresh_visuals(self, *, preserve_view: bool = True) -> None:
        self._mouse_plot_auto_range_pending = True
        self._refresh_field_map(preserve_view=preserve_view)
        self._refresh_line_plot(preserve_view=preserve_view)
        self._refresh_mouse_mode_plots()
        self._update_trace_reference_label()
        self._update_mouse_mode_state()
        self._update_active_analysis_label()

    def _refresh_field_map_preserving_view(self, *args) -> None:
        del args
        self._schedule_field_map_refresh()

    def _on_colormap_changed(self, *args) -> None:
        del args
        self._schedule_refresh(_REFRESH_FIELD_MAP, preserve_view=True)
        current_tab = self.plot_tabs.currentWidget()
        if current_tab is self.lineout_plot:
            self._schedule_refresh(_REFRESH_LINE, preserve_view=True)
        elif current_tab is self.diagnostic_plot:
            self._schedule_refresh(_REFRESH_DIAGNOSTIC, preserve_view=True)
        elif current_tab is self.mouse_tab:
            self._schedule_refresh(_REFRESH_MOUSE, preserve_view=True)

    def _refresh_field_map(self, *, preserve_view: bool = False) -> None:
        if self.run_payload is None or self.current_field_payload is None:
            return
        started = time.perf_counter()
        display_bundle = self._current_display_field_bundle()
        if display_bundle is None:
            return
        display_field, display_unit, pretty_field = display_bundle
        mask = self._combined_zone_mask()
        raw_data = self._masked_matrix(display_field, mask)
        image, scale_note = self._prepare_display_image(raw_data)
        levels, auto_levels = self._current_map_levels(image)
        snapshot_index = self._current_snapshot_index()
        display_time = self._display_time_values(np.asarray(self.run_payload.time, dtype=np.float64))
        coordinate_mode = self._map_coordinate_mode()
        mesh_x: np.ndarray | None = None
        mesh_y: np.ndarray | None = None
        coordinate_edges: np.ndarray | None = None
        boundary_curves: list[np.ndarray] | None = None
        reference_curve: np.ndarray | None = None
        inactive_ranges: list[tuple[float, float]] = []
        laser_entry_position: float | None = None
        laser_entry_curve: np.ndarray | None = None
        if coordinate_mode == "zone":
            coordinate_values = np.arange(1, self.run_payload.summary["n_zones"] + 1, dtype=np.float64)
            coordinate_edges = self._zone_coordinate_edges()
            coordinate_label = "Zone index"
            coordinate_title = "zone index"
            inactive_ranges = self._inactive_coordinate_ranges("zone", mask)
        elif coordinate_mode == "moving_radius" and self.radius_payload is not None:
            radius_bundle = self._display_radius_bundle()
            if radius_bundle is None:
                return
            radius_display, radius_unit = radius_bundle
            coordinate_values = np.asarray(radius_display[snapshot_index], dtype=np.float64)
            coordinate_label = f"{self._coordinate_mode_text('moving_radius', capitalize=True)} [{radius_unit}]"
            coordinate_title = self._coordinate_mode_text("moving_radius")
            mesh = self._moving_mesh_surface(image)
            if mesh is not None:
                mesh_x, mesh_y, image = mesh
            boundary_curves = self._moving_mesh_boundary_curves(mask)
        else:
            coordinate_values = self._display_length_values(np.asarray(self.run_payload.static_x, dtype=np.float64))
            coordinate_edges = self._static_x_edge_values()
            coordinate_label = f"{self._coordinate_mode_text('static_x', capitalize=True)} [{self._viewer_settings.length_unit}]"
            coordinate_title = self._coordinate_mode_text("static_x")
            inactive_ranges = self._inactive_coordinate_ranges("static_x", mask)

        orientation = str(self.map_orientation_combo.currentData())
        if orientation == "time_x_coord_y":
            x_label = f"Time [{self._viewer_settings.time_unit}]"
            y_label = coordinate_label
            title = f"{pretty_field} map | time on x, {coordinate_title} on y"
        else:
            x_label = coordinate_label
            y_label = f"Time [{self._viewer_settings.time_unit}]"
            title = f"{pretty_field} map | {coordinate_title} on x, time on y"
        title += f" | {self._filter_brief_label()} | {scale_note}"

        show_slice_overlays = self.plot_tabs.currentWidget() is self.lineout_plot
        show_reference_marker = show_slice_overlays and self._slice_mode() == "time_trace"
        reference_position: float | None = None
        if show_reference_marker:
            trace_zone = self._resolved_trace_zone_index()
            if trace_zone is not None:
                if coordinate_mode == "zone":
                    reference_position = float(trace_zone + 1)
                elif coordinate_mode == "static_x":
                    reference_position = self._display_length_value(float(self.run_payload.static_x[trace_zone]))
                elif coordinate_mode == "moving_radius":
                    reference_curve = self._moving_mesh_reference_curve()

        colorbar_prefix = "log10 " if str(self.map_scale_combo.currentData()) == "log10" else ""
        colorbar_label = _label_with_unit(f"{colorbar_prefix}{pretty_field}".rstrip(), display_unit)
        boundary_positions = None if coordinate_mode == "moving_radius" else self._region_boundary_positions(coordinate_mode, active_mask=mask)
        laser_entry_position, laser_entry_curve = self._laser_entry_overlay()
        render_mode = "mesh" if mesh_x is not None and mesh_y is not None else "image"
        view_context_key = self._field_map_view_context_key(
            coordinate_mode=coordinate_mode,
            orientation=orientation,
            render_mode=render_mode,
        )
        self.field_map_widget.set_colormap(str(self.colormap_combo.currentData()))
        self.field_map_widget.set_field_map(
            image,
            coordinate_values,
            display_time,
            # Profiles/probes stay zone-centered, but the 2D image path needs the
            # explicit cell edges to avoid reintroducing the half-cell shift that
            # originates in HELIOS' edge-based Radius column.
            orientation=orientation,
            title=title,
            x_label=x_label,
            y_label=y_label,
            colorbar_label=colorbar_label,
            levels=levels,
            auto_levels=auto_levels,
            boundary_positions=boundary_positions,
            show_boundaries=bool(self.boundary_overlay_checkbox.isChecked()),
            reference_position=reference_position,
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            coordinate_edges=coordinate_edges,
            boundary_curves=boundary_curves,
            reference_curve=reference_curve,
            inactive_ranges=inactive_ranges,
            laser_entry_position=laser_entry_position,
            laser_entry_curve=laser_entry_curve,
            show_time_marker=show_slice_overlays,
            show_reference_marker=show_reference_marker,
            preserve_view=preserve_view,
            view_context_key=view_context_key,
            view_preservation_key=self._field_map_view_preservation_key(),
        )
        self._update_time_label(snapshot_index)
        self._sync_probe_overlay()
        LOGGER.debug(
            "Refreshed field map field=%s mode=%s render=%s preserve_view=%s in %.3f s (mesh cache hits=%s misses=%s)",
            self.current_field_name,
            coordinate_mode,
            self.field_map_widget.current_render_mode,
            preserve_view,
            time.perf_counter() - started,
            self._moving_mesh_cache_hits,
            self._moving_mesh_cache_misses,
        )

    def _effective_value_scale_mode(self, values: np.ndarray, requested: str) -> tuple[str, str]:
        finite = values[np.isfinite(values)]
        if requested == "log10":
            positive = finite[finite > 0.0]
            if positive.size == 0:
                self._set_status_message("Log10 scale needs positive values. Falling back to linear for the current slice.")
                return "linear", "linear fallback"
            if positive.size < finite.size:
                return "log10", "log10 positive values only"
            return "log10", "log10"
        if requested == "signed_log10":
            return "signed_log10", "signed log10"
        return "linear", "linear"

    def _refresh_line_plot(self, *, preserve_view: bool = False) -> None:
        self.lineout_plot.set_colormap(str(self.colormap_combo.currentData()))
        if self.run_payload is None or self.current_field_payload is None:
            return
        display_bundle = self._current_display_field_bundle()
        if display_bundle is None:
            return
        display_field, display_unit, pretty_field = display_bundle
        mask = self._combined_zone_mask()
        slice_mode = self._slice_mode()
        display_time = self._display_time_values(np.asarray(self.run_payload.time, dtype=np.float64))

        if slice_mode == "time_trace":
            resolved = self._resolve_trace_reference()
            if resolved is None:
                self.lineout_plot.set_curves(
                    np.array([0.0], dtype=np.float64),
                    [np.array([np.nan], dtype=np.float64)],
                    title="No active zone selected for time trace",
                    x_label=f"Time [{self._viewer_settings.time_unit}]",
                    y_label=_label_with_unit(pretty_field, display_unit),
                    value_scale_mode="linear",
                    preserve_view=preserve_view,
                )
                return
            trace_zone = int(resolved["zone_index"])
            y = np.asarray(display_field[:, trace_zone], dtype=np.float64)
            x = display_time
            line_scale_mode, scale_note = self._effective_value_scale_mode(y, str(self.line_scale_combo.currentData()))
            coord_mode = str(resolved["mode"])
            actual_zone = int(resolved["actual_zone"])
            if coord_mode == "static_x":
                coordinate_text = self._coordinate_mode_text("static_x")
                if bool(resolved["adjusted"]):
                    coord_note = (
                        f"selected {coordinate_text} = {float(resolved['requested_display']):.4e} {self._viewer_settings.length_unit} "
                        f"-> nearest active zone {actual_zone}, {coordinate_text} = {float(resolved['actual_display']):.4e} "
                        f"{self._viewer_settings.length_unit}"
                    )
                else:
                    coord_note = (
                        f"{coordinate_text} = {float(resolved['actual_display']):.4e} {self._viewer_settings.length_unit} "
                        f"(zone {actual_zone})"
                    )
            elif coord_mode == "moving_radius":
                anchor_snapshot = int(resolved["anchor_snapshot"])
                coordinate_text = self._coordinate_mode_text("moving_radius")
                axis_text = self._coordinate_name()
                if bool(resolved["adjusted"]):
                    coord_note = (
                        f"selected {coordinate_text} = {float(resolved['requested_display']):.4e} {self._viewer_settings.length_unit} "
                        f"at snapshot {anchor_snapshot} -> nearest active zone {actual_zone}, "
                        f"{axis_text} = {float(resolved['actual_display']):.4e} {self._viewer_settings.length_unit}"
                    )
                else:
                    coord_note = (
                        f"{coordinate_text} = {float(resolved['actual_display']):.4e} {self._viewer_settings.length_unit} "
                        f"at snapshot {anchor_snapshot} (zone {actual_zone})"
                    )
            else:
                if bool(resolved["adjusted"]):
                    coord_note = f"requested zone {int(resolved['requested_zone'])} -> nearest active zone {actual_zone}"
                else:
                    coord_note = f"zone {actual_zone}"
            current_time = self._display_time_value(float(self.run_payload.time[self._current_snapshot_index()]))
            title = (
                f"{pretty_field} time trace | full history at {coord_note} | "
                f"current-time cursor = {current_time:.4e} {self._viewer_settings.time_unit} | "
                f"{self._filter_brief_label()} | {scale_note}"
            )
            self.lineout_plot.set_curves(
                x,
                [y],
                title=title,
                x_label=f"Time [{self._viewer_settings.time_unit}]",
                y_label=_label_with_unit(pretty_field, display_unit),
                curve_names=[pretty_field],
                value_scale_mode=line_scale_mode,
                boundary_positions=None,
                show_boundaries=False,
                cursor_position=current_time,
                show_cursor=True,
                preserve_view=preserve_view,
                view_context_key=self._line_plot_view_context_key(),
            )
            return

        snapshot_index = self.snapshot_slider.value()
        y = self._masked_vector(display_field[snapshot_index], mask)
        coord_mode = self._line_coordinate_mode()
        if coord_mode == "radius" and self.radius_payload is not None:
            radius_display, radius_unit = self._display_field_data("radius", self.radius_payload.unit, self.radius_payload.data)
            x = np.asarray(radius_display[snapshot_index], dtype=np.float64)
            x_label = f"{self._coordinate_mode_text('radius', capitalize=True)} [{radius_unit}]"
            coord_note = self._coordinate_mode_text("radius")
            boundary_positions = self._region_boundary_positions("radius", snapshot_index=snapshot_index, active_mask=mask)
        elif coord_mode == "zone":
            x = np.arange(1, self.run_payload.summary["n_zones"] + 1, dtype=np.float64)
            x_label = "Zone index"
            coord_note = "zone index"
            boundary_positions = self._region_boundary_positions("zone", active_mask=mask)
        else:
            x = self._display_length_values(np.asarray(self.run_payload.static_x, dtype=np.float64))
            x_label = f"{self._coordinate_mode_text('static_x', capitalize=True)} [{self._viewer_settings.length_unit}]"
            coord_note = self._coordinate_mode_text("static_x")
            boundary_positions = self._region_boundary_positions("static_x", active_mask=mask)

        line_scale_mode, scale_note = self._effective_value_scale_mode(y, str(self.line_scale_combo.currentData()))
        time_value = self._display_time_value(float(self.run_payload.time[snapshot_index]))
        title = (
            f"{pretty_field} lineout | snapshot {snapshot_index} | "
            f"t={time_value:.4e} {self._viewer_settings.time_unit} | {coord_note} | "
            f"{self._filter_brief_label()} | {scale_note}"
        )
        self.lineout_plot.set_curves(
            x,
            [y],
            title=title,
            x_label=x_label,
            y_label=_label_with_unit(pretty_field, display_unit),
            curve_names=[pretty_field],
            value_scale_mode=line_scale_mode,
            boundary_positions=boundary_positions,
            show_boundaries=bool(self.boundary_overlay_checkbox.isChecked()),
            show_cursor=False,
            preserve_view=preserve_view,
            view_context_key=self._line_plot_view_context_key(),
        )

    def _on_field_loaded(self, payload: FieldPayload) -> None:
        if int(payload.run_generation) != self.controller.run_generation:
            LOGGER.debug(
                "Ignoring stale field payload %s for generation %s; current generation is %s.",
                payload.field_name,
                payload.run_generation,
                self.controller.run_generation,
            )
            return
        if payload.field_name == "radius":
            self.radius_payload = payload
            if payload.data.size:
                snapshot_index = int(np.clip(self._trace_reference_anchor_snapshot, 0, payload.data.shape[0] - 1))
                zone_index = int(np.clip(self._trace_reference_zone_index, 0, payload.data.shape[1] - 1))
                self._trace_reference_radius_cm = float(payload.data[snapshot_index, zone_index])
            self._invalidate_moving_mesh_cache()
            self._mouse_plot_auto_range_pending = True
            self._update_map_control_state()
            self._update_slice_control_state()
            if (
                self._prefer_primary_coordinates_on_open
                and self.map_coordinate_combo.findData("moving_radius") >= 0
                and self._map_coordinate_mode() != "moving_radius"
            ):
                self.map_coordinate_combo.setCurrentIndex(self.map_coordinate_combo.findData("moving_radius"))
                self._sync_line_coordinate_to_map_coordinate()
                self._prefer_primary_coordinates_on_open = False
            self._update_coordinate_note()
            self._refresh_field_map(preserve_view=True)
            self._refresh_line_plot(preserve_view=True)
            self._refresh_mouse_mode_plots()
            self._update_mouse_mode_state()
            return
        if payload.field_name != self.current_field_name:
            return
        initial_field_render = self.current_field_payload is None
        self.current_field_payload = payload
        self._display_field_cache_key = None
        self._display_field_cache_value = None
        self._mouse_plot_auto_range_pending = True
        self._refresh_field_map(preserve_view=not initial_field_render)
        self._refresh_line_plot(preserve_view=not initial_field_render)
        self._refresh_mouse_mode_plots()
        self._update_mouse_mode_state()
        self.field_visualized.emit(payload.field_name)

    def _on_diagnostic_loaded(self, payload: DiagnosticPayload) -> None:
        if int(payload.run_generation) != self.controller.run_generation:
            LOGGER.debug(
                "Ignoring stale diagnostic payload %s for generation %s; current generation is %s.",
                payload.path,
                payload.run_generation,
                self.controller.run_generation,
            )
            return
        if payload.path != self.current_diagnostic_path:
            return
        self.current_diagnostic_payload = payload
        self._refresh_diagnostic_plot(preserve_view=True)
        self.diagnostic_visualized.emit(payload.path)

    def _refresh_diagnostic_plot(self, *, preserve_view: bool = False) -> None:
        self.diagnostic_plot.set_colormap(str(self.colormap_combo.currentData()))
        if self.run_payload is None or self.current_diagnostic_payload is None:
            return
        values = np.asarray(self.current_diagnostic_payload.data)
        pretty_name = _pretty_diagnostic(self.current_diagnostic_payload.path)
        if values.ndim == 0:
            x = np.array([0.0], dtype=np.float64)
            ys = [np.array([float(values)], dtype=np.float64)]
            names = [pretty_name]
            x_label = "Index"
        elif values.ndim == 1:
            x = (
                self._display_time_values(np.asarray(self.run_payload.time, dtype=np.float64))
                if values.shape[0] == self.run_payload.time.shape[0]
                else np.arange(values.shape[0], dtype=np.float64)
            )
            ys = [values]
            names = [pretty_name]
            x_label = f"Time [{self._viewer_settings.time_unit}]" if values.shape[0] == self.run_payload.time.shape[0] else "Index"
        else:
            x = (
                self._display_time_values(np.asarray(self.run_payload.time, dtype=np.float64))
                if values.shape[0] == self.run_payload.time.shape[0]
                else np.arange(values.shape[0], dtype=np.float64)
            )
            ys = [values[:, index] for index in range(values.shape[1])]
            if values.shape[1] == len(self.run_payload.regions["region_index"]):
                names = [f"Region {int(region_id)}" for region_id in self.run_payload.regions["region_index"]]
            else:
                names = [f"Series {index + 1}" for index in range(values.shape[1])]
            x_label = f"Time [{self._viewer_settings.time_unit}]" if values.shape[0] == self.run_payload.time.shape[0] else "Index"
        title = f"{pretty_name}"
        if self.current_diagnostic_payload.unit:
            title += f" [{self.current_diagnostic_payload.unit}]"
        diag_scale_mode, scale_note = self._effective_value_scale_mode(
            np.concatenate([np.ravel(np.asarray(series, dtype=np.float64)) for series in ys]) if ys else np.array([], dtype=np.float64),
            str(self.diagnostic_scale_combo.currentData()),
        )
        if scale_note != "linear":
            title += f" | {scale_note}"
        self.diagnostic_plot.set_curves(
            np.asarray(x, dtype=np.float64),
            [np.asarray(series, dtype=np.float64) for series in ys],
            title=title,
            x_label=x_label,
            y_label=self.current_diagnostic_payload.unit or pretty_name,
            curve_names=names,
            value_scale_mode=diag_scale_mode,
            boundary_positions=None,
            show_boundaries=False,
            preserve_view=preserve_view,
            view_context_key=self._diagnostic_plot_view_context_key(),
        )

    def _reset_color_levels(self) -> None:
        self.clip_mode_combo.setCurrentIndex(self.clip_mode_combo.findData("auto"))
        self.level_min_edit.clear()
        self.level_max_edit.clear()
        self._refresh_field_map(preserve_view=True)

    def _reset_plot_views(self) -> None:
        self._mouse_plot_auto_range_pending = True
        self.field_map_widget.reset_view()
        self.lineout_plot.reset_view()
        self.diagnostic_plot.reset_view()
        self.mouse_vertical_plot.reset_view()
        self.mouse_horizontal_plot.reset_view()

    def _select_list_item_by_data(self, widget: QtWidgets.QListWidget, value: str) -> None:
        for index in range(widget.count()):
            item = widget.item(index)
            if str(item.data(QtCore.Qt.UserRole)) == value:
                widget.setCurrentRow(index)
                return

    def _on_error(self, message: str, details: str) -> None:
        LOGGER.error("HELIOS Viewer Error: %s\n%s", message, details)
        self._set_status_message(f"Viewer error: {message}")
        platform_name = str(QtGui.QGuiApplication.platformName()).lower()
        if self._closing or platform_name == "offscreen":
            return
        QtWidgets.QMessageBox.critical(self, "HELIOS Viewer Error", f"{message}\n\n{details}")
