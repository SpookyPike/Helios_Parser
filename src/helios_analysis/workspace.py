"""UI workspace for HELIOS Derived / Analysis mode.

The workspace is a presentation layer only. It owns controls, tables, and plots
for the derived-analysis workflow, while the scientific calculations live in
``helios.services.derived`` and the background orchestration lives in the
controller.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon_export import plasmon_export_is_ready, write_plasmon_spectrum_csv
from helios.services.derived.plasmon_config import (
    PLASMON_AXIS_ANGLE_DEG,
    PLASMON_AXIS_CHOICES,
    PLASMON_AXIS_K_ANGSTROM_INV,
    PLASMON_MODEL_AUTO_BEST,
    PLASMON_BENCHMARK_PRESET_AL_AMBIENT_ARTICLE,
    PLASMON_BENCHMARK_PRESET_AL_DRIVEN_ARTICLE,
    PLASMON_BENCHMARK_PRESET_CHOICES,
    PLASMON_BENCHMARK_PRESET_NONE,
    PLASMON_COLLISION_MODEL_BENCHMARK_DENSE,
    PLASMON_COLLISION_MODEL_CHOICES,
    PLASMON_COLLISION_MODEL_NRL_CONSTANT,
    PLASMON_COMPARISON_MODEL_CHOICES,
    PLASMON_EXECUTION_MODE_BENCHMARK,
    PLASMON_EXECUTION_MODE_CHOICES,
    PLASMON_EXECUTION_MODE_QUICKLOOK,
    PLASMON_INTEGRATION_MODE_CHOICES,
    PLASMON_INTEGRATION_MODE_EFFECTIVE_STATE,
    PLASMON_LFC_MODEL_CHOICES,
    PLASMON_LFC_MODEL_NONE,
    PLASMON_MODEL_CHOICES,
    PLASMON_MODEL_QUICKLOOK,
    PLASMON_OBSERVABLE_MODE_CHOICES,
    PLASMON_OBSERVABLE_MODE_DIELECTRIC,
    PLASMON_NORMALIZATION_CHOICES,
    PLASMON_NORMALIZATION_PEAK,
    PLASMON_STUDY_MODE_CHOICES,
    PLASMON_STUDY_MODE_DISPERSION,
    PLASMON_STUDY_MODE_SPECTRUM,
    PlasmonUiCapabilities,
    plasmon_ui_capabilities,
)
from helios.services.derived.plasmon_electron_policy import (
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK,
    PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT,
    PLASMON_ELECTRON_POLICY_RAW,
    PLASMON_BENCHMARK_POLICY_COMPARISON,
    policy_choices as plasmon_electron_policy_choices,
    policy_label as plasmon_electron_policy_label,
    policy_scope as plasmon_electron_policy_scope,
    resolved_material_formula_map,
)
from helios.services.derived.plasmon_units import plasmon_probe_wavelength_angstrom
from helios.services.derived.plasmon_validation import shocked_al_slab_summary
from helios.services.derived.models import (
    DerivedAnalysisResult,
    DerivedPlotBundle,
    InterfaceEventsResult,
    PlasmonResult,
    PreheatSummary,
    ShockTrackingResult,
    SpectroscopyResult,
    TransmissionResult,
    WaveBranchSummary,
    WaveTrackingResult,
    XrdResult,
)
from helios.services.units.conversions import (
    photon_energy_ev_from_wavelength_nm,
    photon_energy_kev_to_wavelength_angstrom,
    wavelength_shift_nm_to_energy_ev,
)
from helios_viewer.plots import CurvePlotWidget
from helios_viewer.settings import ViewerSettings, default_viewer_settings
from helios_viewer.slider import apply_absolute_click_slider_behavior
from helios_viewer.style import LIGHT_THEME, ViewerTheme, build_mono_font, configure_combo_box_interaction
from helios_viewer.units import (
    DENSITY_FACTORS,
    LENGTH_FACTORS,
    NUMBER_DENSITY_FACTORS,
    PRESSURE_FACTORS,
    RATE_FACTORS,
    SPECIFIC_ENERGY_FACTORS,
    TEMPERATURE_FACTORS,
    TIME_FACTORS,
    VELOCITY_FACTORS,
)


def _format_optional(value: float | None, fmt: str = "{:.3g}", *, suffix: str = "-") -> str:
    if value is None:
        return suffix
    if not math.isfinite(float(value)):
        return suffix
    return fmt.format(float(value))


def _normalize_photon_unit(unit: str) -> str:
    normalized = str(unit or "keV").strip().lower()
    if normalized in {"ev"}:
        return "eV"
    if normalized in {"kev"}:
        return "keV"
    if normalized in {"nm"}:
        return "nm"
    if normalized in {"angstrom", "ang", "a"}:
        return "Angstrom"
    return "keV"


def _photon_unit_label(unit: str) -> str:
    return _normalize_photon_unit(unit)


def _angle_unit_label(unit: str) -> str:
    return "rad" if str(unit).lower() == "rad" else "deg"


def _wavefront_support_label(value: str) -> str:
    labels = {
        "provisional": "Provisional candidate",
        "short_weak": "Short / weak branch",
        "tracked": "Tracked branch",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def _transmission_regime_label(value: str) -> str:
    labels = {
        "xcom": "XCOM",
        "free_free_thomson": "Free-free + Thomson",
        "free_free": "Free-free",
        "thomson": "Thomson",
        "thomson_fallback": "Thomson fallback",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def _transmission_region_mix_label(
    dominant_regime: str,
    *,
    xcom_path_fraction: float | None,
    free_free_thomson_path_fraction: float | None,
    thomson_fallback_path_fraction: float | None,
    xcom_tau_fraction: float | None,
    free_free_thomson_tau_fraction: float | None,
    thomson_fallback_tau_fraction: float | None,
) -> str:
    dominant_label = _transmission_regime_label(dominant_regime)
    fractions = {
        "xcom": (
            0.0 if xcom_path_fraction is None else float(xcom_path_fraction),
            0.0 if xcom_tau_fraction is None else float(xcom_tau_fraction),
        ),
        "free_free_thomson": (
            0.0 if free_free_thomson_path_fraction is None else float(free_free_thomson_path_fraction),
            0.0 if free_free_thomson_tau_fraction is None else float(free_free_thomson_tau_fraction),
        ),
        "thomson_fallback": (
            0.0 if thomson_fallback_path_fraction is None else float(thomson_fallback_path_fraction),
            0.0 if thomson_fallback_tau_fraction is None else float(thomson_fallback_tau_fraction),
        ),
    }
    significant = {
        key
        for key, (path_fraction, tau_fraction) in fractions.items()
        if path_fraction >= 0.05 or tau_fraction >= 0.05
    }
    if len(significant) < 2:
        return dominant_label
    return f"Mixed ({dominant_label}-dominant)"


def _transmission_mode_label(value: str) -> str:
    labels = {
        "auto_hybrid": "Auto hybrid",
        "thomson": "Thomson",
        "free_free": "Free-free",
        "free_free_thomson": "Free-free + Thomson",
        "xcom": "XCOM",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def _bundle_has_finite_data(bundle: DerivedPlotBundle) -> bool:
    x_values = np.asarray(bundle.x_values, dtype=np.float64)
    if x_values.size == 0 or not np.any(np.isfinite(x_values)):
        return False
    for series in bundle.y_series:
        values = np.asarray(series, dtype=np.float64)
        if values.size and np.any(np.isfinite(values)):
            return True
    return False


def _clone_bundle(
    bundle: DerivedPlotBundle,
    *,
    title: str | None = None,
    y_label: str | None = None,
    y_series: tuple[np.ndarray, ...] | None = None,
) -> DerivedPlotBundle:
    return DerivedPlotBundle(
        key=bundle.key,
        title=bundle.title if title is None else title,
        x_label=bundle.x_label,
        y_label=bundle.y_label if y_label is None else y_label,
        x_values=np.asarray(bundle.x_values, dtype=np.float64),
        y_series=tuple(np.asarray(series, dtype=np.float64) for series in (bundle.y_series if y_series is None else y_series)),
        curve_names=tuple(bundle.curve_names),
        boundary_positions=tuple(bundle.boundary_positions),
        value_scale_mode=bundle.value_scale_mode,
    )


def _spectroscopy_shift_unit_label(unit: str) -> str:
    labels = {
        "nm": "nm",
        "ev": "eV",
        "mev": "meV",
        "uev": "ueV",
    }
    return labels.get(str(unit).lower(), "nm")


def _convert_shift_nm(values_nm: float | np.ndarray, line_wavelength_nm: float, unit: str) -> float | np.ndarray:
    normalized = str(unit).lower()
    values_array = np.asarray(values_nm, dtype=np.float64)
    if normalized == "nm":
        return values_array
    energy_values = np.asarray(wavelength_shift_nm_to_energy_ev(values_array, float(line_wavelength_nm)), dtype=np.float64)
    if normalized == "mev":
        return energy_values * 1.0e3
    if normalized == "uev":
        return energy_values * 1.0e6
    return energy_values


def _selected_summary(context: RunContext) -> str:
    region_text = "all viewer regions" if not context.selected_region_ids else "viewer regions " + ", ".join(str(value) for value in context.selected_region_ids)
    material_text = (
        "all viewer materials"
        if not context.selected_material_ids
        else "viewer materials " + ", ".join(str(value) for value in context.selected_material_ids)
    )
    return f"{region_text}; {material_text}"


class _IgnoreWheelUnlessFocused(QtCore.QObject):
    """Prevent accidental wheel-driven recompute storms on unfocused controls."""

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


class _PopupAwareComboBox(QtWidgets.QComboBox):
    """Combo box with explicit popup lifecycle tracking.

    Some Qt platform/style combinations can report popup-view visibility
    ambiguously while the combo is embedded inside nested splitters and heavy
    plots. Tracking showPopup()/hidePopup() directly keeps DerivedPlotPanel
    selector commits tied to the actual popup lifecycle instead of inferring
    state from view/window visibility.
    """

    popup_visibility_changed = QtCore.Signal(bool)
    popup_closed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._popup_visible = False

    def is_popup_visible(self) -> bool:
        return bool(self._popup_visible)

    def showPopup(self) -> None:
        self._popup_visible = True
        self.popup_visibility_changed.emit(True)
        super().showPopup()

    def hidePopup(self) -> None:
        was_visible = bool(self._popup_visible)
        super().hidePopup()
        self._popup_visible = False
        if was_visible:
            self.popup_visibility_changed.emit(False)
            self.popup_closed.emit()


class DerivedPlotPanel(QtWidgets.QWidget):
    """Compact time-trace + snapshot-profile viewer for a derived module."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = LIGHT_THEME
        self._time_bundles: list[DerivedPlotBundle] = []
        self._profile_bundles: list[DerivedPlotBundle] = []
        self._expected_time_options: tuple[tuple[str, str], ...] = ()
        self._expected_profile_options: tuple[tuple[str, str], ...] = ()
        self._missing_time_reason = ""
        self._missing_profile_reason = ""
        self._view_scope = "derived"
        self._snapshot_time_value: float | None = None
        self._pending_time_render_index: int | None = None
        self._pending_profile_render_index: int | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(4)

        self.time_combo = _PopupAwareComboBox()
        self.profile_combo = _PopupAwareComboBox()
        self.time_label = QtWidgets.QLabel("Time traces")
        self.profile_label = QtWidgets.QLabel("Snapshot profiles")
        controls_layout.addWidget(self.time_label, 0, 0)
        controls_layout.addWidget(self.time_combo, 0, 1)
        controls_layout.addWidget(self.profile_label, 1, 0)
        controls_layout.addWidget(self.profile_combo, 1, 1)
        layout.addWidget(controls)

        navigation = QtWidgets.QWidget()
        navigation_layout = QtWidgets.QHBoxLayout(navigation)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(6)
        navigation_layout.addWidget(QtWidgets.QLabel("Interaction"))
        self.pan_button = QtWidgets.QToolButton()
        self.pan_button.setText("Pan")
        self.pan_button.setCheckable(True)
        self.pan_button.setChecked(True)
        self.zoom_button = QtWidgets.QToolButton()
        self.zoom_button.setText("Zoom")
        self.zoom_button.setCheckable(True)
        self.nav_group = QtGui.QActionGroup(self)
        self.nav_group.setExclusive(True)
        pan_action = QtGui.QAction("Pan", self, checkable=True)
        zoom_action = QtGui.QAction("Zoom", self, checkable=True)
        self.nav_group.addAction(pan_action)
        self.nav_group.addAction(zoom_action)
        self.pan_button.setDefaultAction(pan_action)
        self.zoom_button.setDefaultAction(zoom_action)
        pan_action.setChecked(True)
        pan_action.triggered.connect(lambda checked=False: self._set_navigation_mode("pan"))
        zoom_action.triggered.connect(lambda checked=False: self._set_navigation_mode("zoom"))
        self.reset_time_button = QtWidgets.QPushButton("Reset Time View")
        self.reset_profile_button = QtWidgets.QPushButton("Reset Profile View")
        navigation_layout.addWidget(self.pan_button)
        navigation_layout.addWidget(self.zoom_button)
        navigation_layout.addSpacing(8)
        navigation_layout.addWidget(self.reset_time_button)
        navigation_layout.addWidget(self.reset_profile_button)
        navigation_layout.addStretch(1)
        layout.addWidget(navigation)

        self.empty_state_label = QtWidgets.QLabel("No finite derived curves are available for the current selection.")
        self.empty_state_label.setWordWrap(True)
        self.empty_state_label.hide()
        layout.addWidget(self.empty_state_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        self.time_plot = CurvePlotWidget()
        self.profile_plot = CurvePlotWidget()
        splitter.addWidget(self.time_plot)
        splitter.addWidget(self.profile_plot)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 360])
        layout.addWidget(splitter, 1)
        self.reset_time_button.clicked.connect(self.time_plot.reset_view)
        self.reset_profile_button.clicked.connect(self.profile_plot.reset_view)

        # Audit note: these selectors used to redraw plots directly from
        # currentIndexChanged. During popup navigation that allowed highlight /
        # in-progress index churn to trigger heavy CurvePlotWidget redraws,
        # which destabilized popup hover/click handling. Keep popup-open index
        # changes deferred and commit rendering only after activation / close.
        self.time_combo.currentIndexChanged.connect(self._on_time_combo_index_changed)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_combo_index_changed)
        self.time_combo.activated.connect(self._on_time_combo_activated)
        self.profile_combo.activated.connect(self._on_profile_combo_activated)
        self.time_combo.popup_closed.connect(self._flush_pending_time_render)
        self.profile_combo.popup_closed.connect(self._flush_pending_profile_render)
        self.time_combo.view().installEventFilter(self)
        self.profile_combo.view().installEventFilter(self)
        self.time_combo.view().window().installEventFilter(self)
        self.profile_combo.view().window().installEventFilter(self)
        self._set_navigation_mode("pan")

    def apply_theme(self, theme: ViewerTheme) -> None:
        self._theme = theme
        self.time_plot.apply_theme(theme)
        self.profile_plot.apply_theme(theme)
        self.empty_state_label.setStyleSheet(f"color: {theme.subtle_text};")

    def clear(self) -> None:
        self._time_bundles = []
        self._profile_bundles = []
        self._view_scope = "derived"
        self._snapshot_time_value = None
        self._pending_time_render_index = None
        self._pending_profile_render_index = None
        self._sync_combo_items(self.time_combo, self._time_bundles, expected_options=self._expected_time_options, missing_reason=self._missing_time_reason)
        self._sync_combo_items(self.profile_combo, self._profile_bundles, expected_options=self._expected_profile_options, missing_reason=self._missing_profile_reason)
        for plot in (self.time_plot, self.profile_plot):
            plot.clear_plot()
            plot.clear_cursor_marker()
            plot.setEnabled(False)
        self.time_combo.setEnabled(bool(self.time_combo.count()))
        self.profile_combo.setEnabled(bool(self.profile_combo.count()))
        self.reset_time_button.setEnabled(False)
        self.reset_profile_button.setEnabled(False)
        self.empty_state_label.setText("No finite derived curves are available for the current selection.")
        self.empty_state_label.setVisible(True)

    def set_section_labels(self, primary_label: str, secondary_label: str) -> None:
        self.time_label.setText(str(primary_label))
        self.profile_label.setText(str(secondary_label))

    def set_expected_options(
        self,
        time_options: tuple[tuple[str, str], ...],
        profile_options: tuple[tuple[str, str], ...],
        *,
        missing_time_reason: str = "",
        missing_profile_reason: str = "",
        preferred_time_key: str | None = None,
        preferred_profile_key: str | None = None,
    ) -> None:
        normalized_time_options = tuple((str(key), str(label)) for key, label in time_options)
        normalized_profile_options = tuple((str(key), str(label)) for key, label in profile_options)
        preserve_time_key = self.time_combo.currentData() if normalized_time_options == self._expected_time_options else None
        preserve_profile_key = self.profile_combo.currentData() if normalized_profile_options == self._expected_profile_options else None
        self._expected_time_options = normalized_time_options
        self._expected_profile_options = normalized_profile_options
        self._missing_time_reason = str(missing_time_reason or "")
        self._missing_profile_reason = str(missing_profile_reason or "")
        self._sync_combo_items(
            self.time_combo,
            self._time_bundles,
            expected_options=self._expected_time_options,
            missing_reason=self._missing_time_reason,
        )
        self._sync_combo_items(
            self.profile_combo,
            self._profile_bundles,
            expected_options=self._expected_profile_options,
            missing_reason=self._missing_profile_reason,
        )
        self._set_combo_key(
            self.time_combo,
            self._resolved_expected_key(
                self._expected_time_options,
                preserve_time_key,
                preferred_time_key,
            ),
        )
        self._set_combo_key(
            self.profile_combo,
            self._resolved_expected_key(
                self._expected_profile_options,
                preserve_profile_key,
                preferred_profile_key,
            ),
        )
        self.time_combo.setEnabled(bool(self.time_combo.count()))
        self.profile_combo.setEnabled(bool(self.profile_combo.count()))
        self._render_time_bundle_for_index(self.time_combo.currentIndex())
        self._render_profile_bundle_for_index(self.profile_combo.currentIndex())

    def set_bundles(
        self,
        time_bundles: tuple[DerivedPlotBundle, ...],
        profile_bundles: tuple[DerivedPlotBundle, ...],
        *,
        view_scope: str,
        preferred_time_key: str | None = None,
        preferred_profile_key: str | None = None,
    ) -> None:
        current_time_key = self.time_combo.currentData()
        current_profile_key = self.profile_combo.currentData()
        self._time_bundles = [bundle for bundle in time_bundles if _bundle_has_finite_data(bundle)]
        self._profile_bundles = [bundle for bundle in profile_bundles if _bundle_has_finite_data(bundle)]
        if self._expected_time_options:
            selected_time_key = self._resolved_expected_key(self._expected_time_options, current_time_key, preferred_time_key)
        else:
            selected_time_key = self._resolved_bundle_key(self._time_bundles, current_time_key, preferred_time_key)
        if self._expected_profile_options:
            selected_profile_key = self._resolved_expected_key(self._expected_profile_options, current_profile_key, preferred_profile_key)
        else:
            selected_profile_key = self._resolved_bundle_key(self._profile_bundles, current_profile_key, preferred_profile_key)
        self._view_scope = str(view_scope)
        self._pending_time_render_index = None
        self._pending_profile_render_index = None
        self._sync_combo_items(
            self.time_combo,
            self._time_bundles,
            expected_options=self._expected_time_options,
            missing_reason=self._missing_time_reason,
        )
        self._sync_combo_items(
            self.profile_combo,
            self._profile_bundles,
            expected_options=self._expected_profile_options,
            missing_reason=self._missing_profile_reason,
        )
        if self.time_combo.count():
            self.time_combo.setEnabled(True)
            self.time_plot.setEnabled(True)
            self.reset_time_button.setEnabled(True)
            time_index = self._combo_index_for_key(self.time_combo, selected_time_key)
            if self.time_combo.currentIndex() != time_index:
                self.time_combo.blockSignals(True)
                self.time_combo.setCurrentIndex(time_index)
                self.time_combo.blockSignals(False)
            self._render_time_bundle_for_index(self.time_combo.currentIndex())
        else:
            self.time_combo.setEnabled(False)
            self.time_plot.clear_plot()
            self.time_plot.clear_cursor_marker()
            self.time_plot.setEnabled(False)
            self.reset_time_button.setEnabled(False)
        if self.profile_combo.count():
            self.profile_combo.setEnabled(True)
            self.profile_plot.setEnabled(True)
            self.reset_profile_button.setEnabled(True)
            profile_index = self._combo_index_for_key(self.profile_combo, selected_profile_key)
            if self.profile_combo.currentIndex() != profile_index:
                self.profile_combo.blockSignals(True)
                self.profile_combo.setCurrentIndex(profile_index)
                self.profile_combo.blockSignals(False)
            self._render_profile_bundle_for_index(self.profile_combo.currentIndex())
        else:
            self.profile_combo.setEnabled(False)
            self.profile_plot.clear_plot()
            self.profile_plot.clear_cursor_marker()
            self.profile_plot.setEnabled(False)
            self.reset_profile_button.setEnabled(False)
        self.empty_state_label.setVisible(not self._time_bundles and not self._profile_bundles)
        self._apply_snapshot_marker()

    def _render_time_bundle(self) -> None:
        self._render_time_bundle_for_index(self.time_combo.currentIndex())

    def _render_profile_bundle(self) -> None:
        self._render_profile_bundle_for_index(self.profile_combo.currentIndex())

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Hide:
            if watched in {self.time_combo.view(), self.time_combo.view().window()}:
                QtCore.QTimer.singleShot(0, self._flush_pending_time_render)
            elif watched in {self.profile_combo.view(), self.profile_combo.view().window()}:
                QtCore.QTimer.singleShot(0, self._flush_pending_profile_render)
        return super().eventFilter(watched, event)

    @staticmethod
    def _combo_items_signature(
        bundles: list[DerivedPlotBundle],
        *,
        expected_options: tuple[tuple[str, str], ...] = (),
    ) -> tuple[tuple[object, str, bool], ...]:
        bundle_map = {bundle.key: bundle for bundle in bundles}
        if expected_options:
            return tuple(
                (
                    key,
                    bundle_map[key].title if key in bundle_map else label,
                    key in bundle_map,
                )
                for key, label in expected_options
            )
        return tuple((bundle.key, bundle.title, True) for bundle in bundles)

    @staticmethod
    def _current_combo_signature(combo: QtWidgets.QComboBox) -> tuple[tuple[object, str, bool], ...]:
        entries: list[tuple[object, str, bool]] = []
        model = combo.model()
        for index in range(combo.count()):
            model_index = model.index(index, 0)
            enabled = bool(model.flags(model_index) & QtCore.Qt.ItemFlag.ItemIsEnabled)
            entries.append((combo.itemData(index), combo.itemText(index), enabled))
        return tuple(entries)

    def _sync_combo_items(
        self,
        combo: QtWidgets.QComboBox,
        bundles: list[DerivedPlotBundle],
        *,
        expected_options: tuple[tuple[str, str], ...] = (),
        missing_reason: str = "",
    ) -> None:
        signature = self._combo_items_signature(bundles, expected_options=expected_options)
        if self._current_combo_signature(combo) == signature:
            return
        bundle_map = {bundle.key: bundle for bundle in bundles}
        combo.blockSignals(True)
        try:
            combo.clear()
            entries = signature
            for key, title, enabled in entries:
                combo.addItem(title, key)
                item_index = combo.count() - 1
                model = combo.model()
                model_index = model.index(item_index, 0)
                reason = ""
                if expected_options and not enabled:
                    reason = str(missing_reason or "Recompute the current study to populate this view.")
                elif key in bundle_map:
                    reason = bundle_map[key].title
                model.setData(model_index, reason or title, QtCore.Qt.ItemDataRole.ToolTipRole)
                if hasattr(model, "item"):
                    item = model.item(item_index)
                    if item is not None:
                        item.setEnabled(bool(enabled))
        finally:
            combo.blockSignals(False)

    @staticmethod
    def _combo_popup_open(combo: QtWidgets.QComboBox) -> bool:
        if isinstance(combo, _PopupAwareComboBox):
            return combo.is_popup_visible()
        view = combo.view()
        return bool(view.isVisible() or view.window().isVisible())

    def _on_time_combo_index_changed(self, index: int) -> None:
        if self._combo_popup_open(self.time_combo):
            self._pending_time_render_index = int(index)
            return
        self._pending_time_render_index = None
        self._render_time_bundle_for_index(index)

    def _on_profile_combo_index_changed(self, index: int) -> None:
        if self._combo_popup_open(self.profile_combo):
            self._pending_profile_render_index = int(index)
            return
        self._pending_profile_render_index = None
        self._render_profile_bundle_for_index(index)

    def _on_time_combo_activated(self, index: int) -> None:
        self._pending_time_render_index = int(index)
        if not self._combo_popup_open(self.time_combo):
            QtCore.QTimer.singleShot(0, self._flush_pending_time_render)

    def _on_profile_combo_activated(self, index: int) -> None:
        self._pending_profile_render_index = int(index)
        if not self._combo_popup_open(self.profile_combo):
            QtCore.QTimer.singleShot(0, self._flush_pending_profile_render)

    def _flush_pending_time_render(self) -> None:
        if self._combo_popup_open(self.time_combo):
            return
        index = self._pending_time_render_index
        self._pending_time_render_index = None
        if index is None:
            return
        self._render_time_bundle_for_index(index)

    def _flush_pending_profile_render(self) -> None:
        if self._combo_popup_open(self.profile_combo):
            return
        index = self._pending_profile_render_index
        self._pending_profile_render_index = None
        if index is None:
            return
        self._render_profile_bundle_for_index(index)

    def _render_time_bundle_for_index(self, index: int) -> None:
        key = self._combo_key_at(self.time_combo, index)
        bundle = self._bundle_for_key(self._time_bundles, key)
        if bundle is None:
            self.time_plot.clear_plot()
            self.time_plot.clear_cursor_marker()
            self.time_plot.setEnabled(False)
            if self._missing_time_reason:
                self.empty_state_label.setText(self._missing_time_reason)
                self.empty_state_label.setVisible(True)
            return
        self.time_plot.setEnabled(True)
        self.empty_state_label.setVisible(False)
        self._render_bundle(
            self.time_plot,
            bundle,
            ("derived", self._view_scope, "time", bundle.key),
            cursor_position=self._snapshot_time_value,
            show_cursor=self._snapshot_time_value is not None,
        )

    def _render_profile_bundle_for_index(self, index: int) -> None:
        key = self._combo_key_at(self.profile_combo, index)
        bundle = self._bundle_for_key(self._profile_bundles, key)
        if bundle is None:
            self.profile_plot.clear_plot()
            self.profile_plot.clear_cursor_marker()
            self.profile_plot.setEnabled(False)
            if self._missing_profile_reason and not self._time_bundles:
                self.empty_state_label.setText(self._missing_profile_reason)
                self.empty_state_label.setVisible(True)
            return
        self.profile_plot.setEnabled(True)
        if self._time_bundles or self._profile_bundles:
            self.empty_state_label.setVisible(False)
        self._render_bundle(self.profile_plot, bundle, ("derived", self._view_scope, "profile", bundle.key))

    def _set_navigation_mode(self, mode: str) -> None:
        self.time_plot.set_navigation_mode(mode)
        self.profile_plot.set_navigation_mode(mode)

    @staticmethod
    def _combo_index_for_key(combo: QtWidgets.QComboBox, preferred_key: object | None) -> int:
        for index in range(combo.count()):
            if combo.itemData(index) == preferred_key:
                return index
        return 0

    @staticmethod
    def _combo_key_at(combo: QtWidgets.QComboBox, index: int) -> object | None:
        if not (0 <= int(index) < combo.count()):
            return None
        return combo.itemData(int(index))

    @staticmethod
    def _bundle_for_key(bundles: list[DerivedPlotBundle], key: object | None) -> DerivedPlotBundle | None:
        for bundle in bundles:
            if bundle.key == key:
                return bundle
        return None

    @staticmethod
    def _resolved_bundle_key(
        bundles: list[DerivedPlotBundle],
        current_key: object | None,
        preferred_key: object | None,
    ) -> object | None:
        if any(bundle.key == current_key for bundle in bundles):
            return current_key
        if any(bundle.key == preferred_key for bundle in bundles):
            return preferred_key
        return bundles[0].key if bundles else None

    @staticmethod
    def _resolved_expected_key(
        options: tuple[tuple[str, str], ...],
        current_key: object | None,
        preferred_key: object | None,
    ) -> object | None:
        keys = tuple(key for key, _label in options)
        if current_key in keys:
            return current_key
        if preferred_key in keys:
            return preferred_key
        return keys[0] if keys else None

    @staticmethod
    def _set_combo_key(combo: QtWidgets.QComboBox, key: object | None) -> None:
        index = DerivedPlotPanel._combo_index_for_key(combo, key)
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(index if combo.count() else -1)
        finally:
            combo.blockSignals(False)

    @staticmethod
    def _render_bundle(
        plot: CurvePlotWidget,
        bundle: DerivedPlotBundle,
        view_key: object,
        *,
        cursor_position: float | None = None,
        show_cursor: bool = False,
    ) -> None:
        plot.set_curves(
            np.asarray(bundle.x_values, dtype=np.float64),
            [np.asarray(series, dtype=np.float64) for series in bundle.y_series],
            title=bundle.title,
            x_label=bundle.x_label,
            y_label=bundle.y_label,
            curve_names=bundle.curve_names,
            value_scale_mode=bundle.value_scale_mode,
            boundary_positions=bundle.boundary_positions,
            show_boundaries=bool(bundle.boundary_positions),
            auto_range=True,
            cursor_position=cursor_position,
            show_cursor=show_cursor,
            preserve_view=False,
            view_context_key=view_key,
        )

    def set_snapshot_marker(self, snapshot_time_value: float | None) -> None:
        self._snapshot_time_value = None if snapshot_time_value is None or not math.isfinite(float(snapshot_time_value)) else float(snapshot_time_value)
        self._apply_snapshot_marker()

    def _apply_snapshot_marker(self) -> None:
        if self._snapshot_time_value is None:
            self.time_plot.clear_cursor_marker()
            return
        if self._time_bundles and 0 <= self.time_combo.currentIndex() < len(self._time_bundles):
            self.time_plot.set_cursor_marker(float(self._snapshot_time_value), visible=True)
        else:
            self.time_plot.clear_cursor_marker()


class HeliosDerivedWorkspace(QtWidgets.QWidget):
    """Derived / Analysis results workspace for the unified shell."""

    parameters_changed = QtCore.Signal()
    refresh_requested = QtCore.Signal()
    time_plot_modules_changed = QtCore.Signal()
    transmission_refine_requested = QtCore.Signal()
    cancel_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = LIGHT_THEME
        self._context = RunContext.empty()
        self._current_result: DerivedAnalysisResult | None = None
        self._display_settings = default_viewer_settings()
        self._default_profile_coordinate_mode = "viewer_follow"
        self._available_region_ids: tuple[int, ...] = ()
        self._available_material_ids: tuple[int, ...] = ()
        self._module_tab_names: dict[int, str] = {}
        self._busy_message = "Derived mode ready."
        self._busy_started_at = 0.0
        self._last_performance_summary = "Performance: waiting for analysis update."
        self._last_wavefront_performance_summary = "Performance: waiting for WaveFront analysis."
        self._last_preheat_performance_summary = "Performance: waiting for Preheat analysis."
        self._preheat_time_mode = "shock_relative"
        self._preheat_offset_steps = 0
        self._preheat_manual_snapshot_index: int | None = None
        self._preheat_display_snapshot_index: int | None = None
        self._preheat_syncing_controls = False
        self._wheel_guard = _IgnoreWheelUnlessFocused(self)
        self._parameter_change_timer = QtCore.QTimer(self)
        self._parameter_change_timer.setSingleShot(True)
        self._parameter_change_timer.setInterval(60)
        self._parameter_change_timer.timeout.connect(self.parameters_changed.emit)
        self._busy_elapsed_timer = QtCore.QTimer(self)
        self._busy_elapsed_timer.setInterval(200)
        self._busy_elapsed_timer.timeout.connect(self._refresh_busy_status)
        self._plasmon_recompute_pending = False
        self._plasmon_controls_dirty = False
        self._build_ui()
        for combo in self.findChildren(QtWidgets.QComboBox):
            configure_combo_box_interaction(combo)
        self.set_context(self._context)
        self.clear_results("Load a HELIOS run in Viewer Mode, then switch to Derived / Analysis.")

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_scroll.setWidget(controls)

        self.status_label = QtWidgets.QLabel("Derived mode ready.")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)

        context_group = QtWidgets.QGroupBox("Active Run Context")
        context_layout = QtWidgets.QFormLayout(context_group)
        self.run_path_label = QtWidgets.QLabel("-")
        self.run_path_label.setWordWrap(True)
        self.run_path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.run_summary_label = QtWidgets.QLabel("-")
        self.snapshot_label = QtWidgets.QLabel("-")
        self.subset_label = QtWidgets.QLabel("-")
        self.warning_summary_label = QtWidgets.QLabel("Warnings: -")
        context_layout.addRow("Run", self.run_path_label)
        context_layout.addRow("Grid", self.run_summary_label)
        context_layout.addRow("Snapshot", self.snapshot_label)
        context_layout.addRow("Viewer subset", self.subset_label)
        context_layout.addRow("", self.warning_summary_label)
        controls_layout.addWidget(context_group)

        geometry_group = QtWidgets.QGroupBox("Analysis Geometry / Selection")
        geometry_layout = QtWidgets.QFormLayout(geometry_group)
        self.observation_side_combo = QtWidgets.QComboBox()
        self.observation_side_combo.addItem("Front", "front")
        self.observation_side_combo.addItem("Back", "back")
        self.los_angle_spin = QtWidgets.QDoubleSpinBox()
        self.los_angle_spin.setRange(0.0, 89.0)
        self.los_angle_spin.setDecimals(2)
        self.los_angle_spin.setSingleStep(2.5)
        self.los_angle_spin.setSuffix(" deg")
        self.profile_coordinate_combo = QtWidgets.QComboBox()
        self.profile_coordinate_combo.addItem("Follow Viewer / Run", "viewer")
        self.profile_coordinate_combo.addItem("Moving radius", "moving_radius")
        self.profile_coordinate_combo.addItem("Zone index", "zone")
        self.profile_coordinate_combo.addItem("Static x (legacy)", "static_x")
        self.weighting_combo = QtWidgets.QComboBox()
        self.weighting_combo.addItem("Auto", "auto")
        self.weighting_combo.addItem("Electron column", "electron_column")
        self.weighting_combo.addItem("Electron density", "electron_density")
        self.weighting_combo.addItem("Mass", "mass")
        self.weighting_combo.addItem("Width", "width")
        self.weighting_combo.addItem("Simple mean (debug)", "simple_mean")
        self.reuse_viewer_subset_checkbox = QtWidgets.QCheckBox("Reuse active Viewer subset")
        self.reuse_viewer_subset_checkbox.setChecked(True)
        geometry_layout.addRow("Observation side", self.observation_side_combo)
        geometry_layout.addRow("LOS angle", self.los_angle_spin)
        geometry_layout.addRow("Profile coordinate", self.profile_coordinate_combo)
        geometry_layout.addRow("Weighting", self.weighting_combo)
        geometry_layout.addRow("", self.reuse_viewer_subset_checkbox)
        controls_layout.addWidget(geometry_group)

        filter_group = QtWidgets.QGroupBox("Derived Filters")
        filter_layout = QtWidgets.QVBoxLayout(filter_group)
        filter_layout.setContentsMargins(8, 8, 8, 8)
        filter_layout.setSpacing(6)
        filter_layout.addWidget(QtWidgets.QLabel("Regions"))
        self.region_list = QtWidgets.QListWidget()
        self.region_list.setMinimumHeight(110)
        filter_layout.addWidget(self.region_list)
        filter_layout.addWidget(QtWidgets.QLabel("Materials"))
        self.material_list = QtWidgets.QListWidget()
        self.material_list.setMinimumHeight(100)
        filter_layout.addWidget(self.material_list)

        self.exclude_entry_region_checkbox = QtWidgets.QCheckBox("Exclude laser-entry region")
        self.exclude_low_density_checkbox = QtWidgets.QCheckBox("Exclude low-density / blowoff-like zones")
        self.exclude_opposite_velocity_checkbox = QtWidgets.QCheckBox("Exclude opposite-flow zones")
        self.min_density_spin = QtWidgets.QDoubleSpinBox()
        self.min_density_spin.setRange(0.0, 1.0e6)
        self.min_density_spin.setDecimals(4)
        self.min_density_spin.setSingleStep(0.1)
        self.min_density_spin.setSuffix(" g/cm3")
        self.min_density_spin.setValue(0.0)
        zone_clip = QtWidgets.QWidget()
        zone_clip_layout = QtWidgets.QHBoxLayout(zone_clip)
        zone_clip_layout.setContentsMargins(0, 0, 0, 0)
        zone_clip_layout.setSpacing(6)
        self.zone_lower_spin = QtWidgets.QSpinBox()
        self.zone_upper_spin = QtWidgets.QSpinBox()
        zone_clip_layout.addWidget(QtWidgets.QLabel("Zone"))
        zone_clip_layout.addWidget(self.zone_lower_spin)
        zone_clip_layout.addWidget(QtWidgets.QLabel("to"))
        zone_clip_layout.addWidget(self.zone_upper_spin)
        filter_layout.addWidget(self.exclude_entry_region_checkbox)
        filter_layout.addWidget(self.exclude_low_density_checkbox)
        filter_layout.addWidget(self.exclude_opposite_velocity_checkbox)
        density_row = QtWidgets.QWidget()
        density_row_layout = QtWidgets.QHBoxLayout(density_row)
        density_row_layout.setContentsMargins(0, 0, 0, 0)
        density_row_layout.setSpacing(6)
        density_row_layout.addWidget(QtWidgets.QLabel("Min density"))
        density_row_layout.addWidget(self.min_density_spin, 1)
        filter_layout.addWidget(density_row)
        filter_layout.addWidget(zone_clip)
        controls_layout.addWidget(filter_group)

        self.xrd_controls_group = QtWidgets.QGroupBox("XRD Settings")
        xrd_layout = QtWidgets.QFormLayout(self.xrd_controls_group)
        self.xrd_energy_spin = QtWidgets.QDoubleSpinBox()
        self.xrd_energy_spin.setRange(0.1, 30.0)
        self.xrd_energy_spin.setDecimals(3)
        self.xrd_energy_spin.setSingleStep(0.1)
        self.xrd_energy_spin.setSuffix(" keV")
        self.xrd_energy_spin.setValue(8.0)
        self.xrd_angle_spin = QtWidgets.QDoubleSpinBox()
        self.xrd_angle_spin.setRange(0.1, 89.9)
        self.xrd_angle_spin.setDecimals(2)
        self.xrd_angle_spin.setSingleStep(0.5)
        self.xrd_angle_spin.setSuffix(" deg")
        self.xrd_angle_spin.setValue(20.0)
        self.xrd_display_combo = QtWidgets.QComboBox()
        self.xrd_display_combo.addItem("Bragg shift [deg]", "degrees")
        self.xrd_display_combo.addItem("Q [1/A]", "q")
        xrd_layout.addRow("Photon energy", self.xrd_energy_spin)
        xrd_layout.addRow("Initial Bragg", self.xrd_angle_spin)
        xrd_layout.addRow("Display", self.xrd_display_combo)

        self.plasmon_controls_group = QtWidgets.QGroupBox("XRTS / Plasmon Settings")
        self.plasmon_controls_group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        plasmon_layout = QtWidgets.QVBoxLayout(self.plasmon_controls_group)
        plasmon_layout.setContentsMargins(8, 8, 8, 8)
        plasmon_layout.setSpacing(8)
        plasmon_study_group = QtWidgets.QGroupBox("Study")
        plasmon_study_layout = QtWidgets.QFormLayout(plasmon_study_group)
        plasmon_study_layout.setContentsMargins(8, 8, 8, 8)
        plasmon_study_layout.setHorizontalSpacing(8)
        plasmon_study_layout.setVerticalSpacing(4)
        plasmon_solver_group = QtWidgets.QGroupBox("Physics / solver")
        plasmon_solver_layout = QtWidgets.QFormLayout(plasmon_solver_group)
        plasmon_solver_layout.setContentsMargins(8, 8, 8, 8)
        plasmon_solver_layout.setHorizontalSpacing(8)
        plasmon_solver_layout.setVerticalSpacing(4)
        plasmon_spectrum_group = QtWidgets.QGroupBox("Spectrum shaping")
        plasmon_spectrum_layout = QtWidgets.QFormLayout(plasmon_spectrum_group)
        plasmon_spectrum_layout.setContentsMargins(8, 8, 8, 8)
        plasmon_spectrum_layout.setHorizontalSpacing(8)
        plasmon_spectrum_layout.setVerticalSpacing(4)
        self.plasmon_energy_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_energy_spin.setRange(0.1, 30.0)
        self.plasmon_energy_spin.setDecimals(3)
        self.plasmon_energy_spin.setSingleStep(0.1)
        self.plasmon_energy_spin.setKeyboardTracking(False)
        self.plasmon_energy_spin.setSuffix(" keV")
        self.plasmon_energy_spin.setValue(8.0)
        self.plasmon_benchmark_preset_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_BENCHMARK_PRESET_CHOICES:
            self.plasmon_benchmark_preset_combo.addItem(label, value)
        self.plasmon_apply_preset_button = QtWidgets.QPushButton("Load preset")
        self.plasmon_study_mode_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_STUDY_MODE_CHOICES:
            self.plasmon_study_mode_combo.addItem(label, value)
        self.plasmon_coordinate_axis_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_AXIS_CHOICES:
            axis_suffix = " [1/A]" if value == PLASMON_AXIS_K_ANGSTROM_INV else f" [{_angle_unit_label(self._angle_unit())}]"
            self.plasmon_coordinate_axis_combo.addItem(f"{label}{axis_suffix}", value)
        self.plasmon_angle_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_angle_spin.setKeyboardTracking(False)
        self.plasmon_scan_axis_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_AXIS_CHOICES:
            axis_suffix = " [1/A]" if value == PLASMON_AXIS_K_ANGSTROM_INV else f" [{_angle_unit_label(self._angle_unit())}]"
            self.plasmon_scan_axis_combo.addItem(f"{label}{axis_suffix}", value)
        self.plasmon_scan_start_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_scan_start_spin.setKeyboardTracking(False)
        self.plasmon_scan_stop_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_scan_stop_spin.setKeyboardTracking(False)
        self.plasmon_scan_points_spin = QtWidgets.QSpinBox()
        self.plasmon_scan_points_spin.setKeyboardTracking(False)
        self.plasmon_scan_points_spin.setRange(3, 801)
        self.plasmon_scan_points_spin.setValue(61)
        self.plasmon_compare_models_checkbox = QtWidgets.QCheckBox("Compare all spectral models")
        self.plasmon_compare_policies_checkbox = QtWidgets.QCheckBox("Compare benchmark e- policies")
        self.plasmon_compare_model_list = QtWidgets.QListWidget()
        self.plasmon_compare_model_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.plasmon_compare_model_list.setAlternatingRowColors(True)
        self.plasmon_compare_model_list.setUniformItemSizes(True)
        self.plasmon_compare_model_list.setMinimumHeight(84)
        self.plasmon_compare_model_list.setMaximumHeight(132)
        self._populate_plasmon_compare_model_list()
        self.plasmon_gamma_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_gamma_spin.setRange(0.1, 5.0)
        self.plasmon_gamma_spin.setDecimals(3)
        self.plasmon_gamma_spin.setSingleStep(0.05)
        self.plasmon_gamma_spin.setKeyboardTracking(False)
        self.plasmon_gamma_spin.setValue(1.0)
        self.plasmon_model_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_MODEL_CHOICES:
            self.plasmon_model_combo.addItem(label, value)
        self.plasmon_execution_mode_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_EXECUTION_MODE_CHOICES:
            self.plasmon_execution_mode_combo.addItem(label, value)
        self.plasmon_observable_mode_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_OBSERVABLE_MODE_CHOICES:
            self.plasmon_observable_mode_combo.addItem(label, value)
        self.plasmon_electron_policy_combo = QtWidgets.QComboBox()
        for label, value in plasmon_electron_policy_choices():
            self.plasmon_electron_policy_combo.addItem(label, value)
        self.plasmon_integration_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_INTEGRATION_MODE_CHOICES:
            self.plasmon_integration_combo.addItem(label, value)
        self.plasmon_collision_model_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_COLLISION_MODEL_CHOICES:
            self.plasmon_collision_model_combo.addItem(label, value)
        self.plasmon_collision_scale_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_collision_scale_spin.setRange(0.0, 100.0)
        self.plasmon_collision_scale_spin.setDecimals(3)
        self.plasmon_collision_scale_spin.setSingleStep(0.1)
        self.plasmon_collision_scale_spin.setKeyboardTracking(False)
        self.plasmon_collision_scale_spin.setValue(1.0)
        self.plasmon_manual_collision_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_manual_collision_spin.setRange(0.0, 1.0e20)
        self.plasmon_manual_collision_spin.setDecimals(3)
        self.plasmon_manual_collision_spin.setSingleStep(1.0e12)
        self.plasmon_manual_collision_spin.setKeyboardTracking(False)
        self.plasmon_manual_collision_spin.setSuffix(" 1/s")
        self.plasmon_manual_collision_spin.setValue(0.0)
        self.plasmon_lfc_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_LFC_MODEL_CHOICES:
            self.plasmon_lfc_combo.addItem(label, value)
        self.plasmon_window_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_window_spin.setRange(1.0, 5000.0)
        self.plasmon_window_spin.setDecimals(3)
        self.plasmon_window_spin.setSingleStep(1.0)
        self.plasmon_window_spin.setKeyboardTracking(False)
        self.plasmon_window_spin.setSuffix(" eV")
        self.plasmon_window_spin.setValue(80.0)
        self.plasmon_points_spin = QtWidgets.QSpinBox()
        self.plasmon_points_spin.setKeyboardTracking(False)
        self.plasmon_points_spin.setRange(101, 16385)
        self.plasmon_points_spin.setSingleStep(100)
        self.plasmon_points_spin.setValue(1201)
        self.plasmon_fwhm_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_fwhm_spin.setRange(0.0, 5000.0)
        self.plasmon_fwhm_spin.setDecimals(4)
        self.plasmon_fwhm_spin.setSingleStep(0.1)
        self.plasmon_fwhm_spin.setKeyboardTracking(False)
        self.plasmon_fwhm_spin.setSuffix(" eV")
        self.plasmon_fwhm_spin.setValue(0.0)
        self.plasmon_normalization_combo = QtWidgets.QComboBox()
        for label, value in PLASMON_NORMALIZATION_CHOICES:
            self.plasmon_normalization_combo.addItem(label, value)
        self.plasmon_cluster_log_ne_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_cluster_log_ne_spin.setRange(0.0, 1.0)
        self.plasmon_cluster_log_ne_spin.setDecimals(4)
        self.plasmon_cluster_log_ne_spin.setSingleStep(0.01)
        self.plasmon_cluster_log_ne_spin.setKeyboardTracking(False)
        self.plasmon_cluster_log_ne_spin.setValue(0.02)
        self.plasmon_cluster_log_te_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_cluster_log_te_spin.setRange(0.0, 1.0)
        self.plasmon_cluster_log_te_spin.setDecimals(4)
        self.plasmon_cluster_log_te_spin.setSingleStep(0.01)
        self.plasmon_cluster_log_te_spin.setKeyboardTracking(False)
        self.plasmon_cluster_log_te_spin.setValue(0.02)
        self.plasmon_cluster_z_spin = QtWidgets.QDoubleSpinBox()
        self.plasmon_cluster_z_spin.setRange(0.0, 100.0)
        self.plasmon_cluster_z_spin.setDecimals(4)
        self.plasmon_cluster_z_spin.setSingleStep(0.1)
        self.plasmon_cluster_z_spin.setKeyboardTracking(False)
        self.plasmon_cluster_z_spin.setValue(0.1)
        self.plasmon_auto_recompute_checkbox = QtWidgets.QCheckBox("Auto recompute")
        self.plasmon_auto_recompute_checkbox.setChecked(True)
        self.plasmon_recompute_button = QtWidgets.QPushButton("Recompute spectrum")
        self.plasmon_recompute_button.setEnabled(False)
        self.plasmon_export_button = QtWidgets.QPushButton("Export spectrum CSV")
        self.plasmon_export_button.setEnabled(False)
        plasmon_recompute_row = QtWidgets.QWidget()
        plasmon_recompute_layout = QtWidgets.QHBoxLayout(plasmon_recompute_row)
        plasmon_recompute_layout.setContentsMargins(0, 0, 0, 0)
        plasmon_recompute_layout.setSpacing(6)
        plasmon_recompute_layout.addWidget(self.plasmon_auto_recompute_checkbox)
        plasmon_recompute_layout.addWidget(self.plasmon_recompute_button)
        plasmon_recompute_layout.addWidget(self.plasmon_export_button)
        plasmon_recompute_layout.addStretch(1)
        preset_row = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)
        preset_layout.addWidget(self.plasmon_benchmark_preset_combo, 1)
        preset_layout.addWidget(self.plasmon_apply_preset_button, 0)
        plasmon_study_layout.addRow("Model", self.plasmon_model_combo)
        plasmon_study_layout.addRow("Benchmark preset", preset_row)
        plasmon_study_layout.addRow("Study mode", self.plasmon_study_mode_combo)
        plasmon_study_layout.addRow("Probe energy", self.plasmon_energy_spin)
        plasmon_study_layout.addRow("Fixed input", self.plasmon_coordinate_axis_combo)
        plasmon_study_layout.addRow("Angle / k", self.plasmon_angle_spin)
        plasmon_study_layout.addRow("Scan axis", self.plasmon_scan_axis_combo)
        plasmon_study_layout.addRow("Scan start", self.plasmon_scan_start_spin)
        plasmon_study_layout.addRow("Scan stop", self.plasmon_scan_stop_spin)
        plasmon_study_layout.addRow("Scan points", self.plasmon_scan_points_spin)
        plasmon_study_layout.addRow("Compare", self.plasmon_compare_models_checkbox)
        plasmon_study_layout.addRow("Model set", self.plasmon_compare_model_list)
        plasmon_study_layout.addRow("Policy cmp", self.plasmon_compare_policies_checkbox)
        plasmon_solver_layout.addRow("Spectral mode", self.plasmon_execution_mode_combo)
        plasmon_solver_layout.addRow("Observable", self.plasmon_observable_mode_combo)
        plasmon_solver_layout.addRow("Electron policy", self.plasmon_electron_policy_combo)
        plasmon_solver_layout.addRow("Gamma", self.plasmon_gamma_spin)
        plasmon_solver_layout.addRow("Integration", self.plasmon_integration_combo)
        plasmon_solver_layout.addRow("Collision model", self.plasmon_collision_model_combo)
        plasmon_solver_layout.addRow("Collision scale", self.plasmon_collision_scale_spin)
        plasmon_solver_layout.addRow("Manual ν", self.plasmon_manual_collision_spin)
        plasmon_solver_layout.addRow("LFC model", self.plasmon_lfc_combo)
        plasmon_spectrum_layout.addRow("Spectrum window", self.plasmon_window_spin)
        plasmon_spectrum_layout.addRow("Spectrum points", self.plasmon_points_spin)
        plasmon_spectrum_layout.addRow("Instrument FWHM", self.plasmon_fwhm_spin)
        plasmon_spectrum_layout.addRow("Normalization", self.plasmon_normalization_combo)
        plasmon_spectrum_layout.addRow("Cluster log ne tol", self.plasmon_cluster_log_ne_spin)
        plasmon_spectrum_layout.addRow("Cluster log Te tol", self.plasmon_cluster_log_te_spin)
        plasmon_spectrum_layout.addRow("Cluster Z tol", self.plasmon_cluster_z_spin)
        plasmon_layout.addWidget(plasmon_study_group)
        plasmon_layout.addWidget(plasmon_solver_group)
        plasmon_layout.addWidget(plasmon_spectrum_group)
        plasmon_layout.addWidget(plasmon_recompute_row)
        plasmon_layout.addStretch(1)
        self.plasmon_benchmark_preset_combo.setToolTip("Article-facing benchmark preset metadata. Use Load preset to apply the recommended Al benchmark controls and selection filters.")
        self.plasmon_apply_preset_button.setToolTip("Load the selected aluminium article benchmark preset into the current plasmon controls and filters.")
        self.plasmon_model_combo.setToolTip("Select the plasmon/XRTS model. Quick look stays backward-compatible; advanced branches build synthetic spectra. Auto Best keeps to the strongest validated local classical branch, while the finite-T Lindhard options remain manual/experimental.")
        self.plasmon_study_mode_combo.setToolTip("Spectrum mode inspects the line shape at one fixed angle/k. Dispersion mode scans angle or k and tracks the plasmon peak position across that range.")
        self.plasmon_coordinate_axis_combo.setToolTip("Choose whether the fixed spectrum geometry is entered as a scattering angle or directly as k [1/A].")
        self.plasmon_execution_mode_combo.setToolTip("Quicklook mode preserves the lightweight exploratory grid. Benchmark mode auto-refines the spectrum grid, uses local quadratic peak extraction, and hard-rejects selected zones/clusters that leave the active model domain instead of letting LOS averaging hide them.")
        self.plasmon_observable_mode_combo.setToolTip("Choose whether the spectrum represents the backend dielectric/free-electron response directly or a material-specific XRTS observable reconstruction with explicit free, elastic, and bound/core bookkeeping.")
        self.plasmon_electron_policy_combo.setToolTip("Electron-density / effective-valence mapping. Raw HELIOS keeps the hydro ne/zbar fields; benchmark policies make the valence assumption explicit.")
        self.plasmon_integration_combo.setToolTip("Choose a single effective-state spectrum or a LOS-integrated clustered spectrum assembled from local zone states.")
        self.plasmon_collision_model_combo.setToolTip("Collision-rate prescription for Mermin branches. Manual ν uses the entered constant rate directly.")
        self.plasmon_collision_scale_spin.setToolTip("Scale factor applied to the NRL-style constant collision estimate before the Mermin closure is evaluated.")
        self.plasmon_manual_collision_spin.setToolTip("Manual constant collision rate ν [1/s] for Mermin branches.")
        self.plasmon_lfc_combo.setToolTip("Static local-field-correction backend. ESA static is a compact fast surrogate, not the full reference fit.")
        self.plasmon_scan_axis_combo.setToolTip("Choose whether the dispersion scan is specified in scattering angle or directly in k [1/A].")
        self.plasmon_scan_start_spin.setToolTip("Start of the dispersion scan range for the chosen axis.")
        self.plasmon_scan_stop_spin.setToolTip("End of the dispersion scan range for the chosen axis.")
        self.plasmon_scan_points_spin.setToolTip("Number of points used to sample the dispersion curve.")
        self.plasmon_compare_models_checkbox.setToolTip("Overlay all spectral models on the current dispersion or spectrum plot for direct comparison.")
        self.plasmon_compare_model_list.setToolTip("Choose which benchmark-capable spectral models participate in the direct model-comparison overlays.")
        self.plasmon_compare_policies_checkbox.setToolTip("Benchmark-only overlay of the current spectral model under the article Al benchmark, article Al driven increment, benchmark valence-aware, and valence-locked electron policies.")
        self.plasmon_window_spin.setToolTip("Half-width of the symmetric energy-transfer axis used for the synthetic spectrum.")
        self.plasmon_points_spin.setToolTip("Number of energy-transfer grid points. Larger grids resolve peaks better but increase compute cost.")
        self.plasmon_fwhm_spin.setToolTip("Gaussian instrument/probe-response FWHM used to convolve the observed spectrum.")
        self.plasmon_normalization_combo.setToolTip("Display normalization for the observed spectrum. Use None when comparing weighted sums directly.")
        self.plasmon_cluster_log_ne_spin.setToolTip("Clustering tolerance in log10(ne) for LOS-integrated spectra. Lower values preserve more local detail and cost more.")
        self.plasmon_cluster_log_te_spin.setToolTip("Clustering tolerance in log10(Te) for LOS-integrated spectra.")
        self.plasmon_cluster_z_spin.setToolTip("Clustering tolerance in mean charge for LOS-integrated spectra.")
        self.plasmon_auto_recompute_checkbox.setToolTip("When disabled, heavy plasmon controls mark the study as stale until Recompute is pressed.")
        self.plasmon_recompute_button.setToolTip("Recompute the currently selected plasmon study for the active snapshot.")
        self.plasmon_export_button.setToolTip("Export the currently displayed plasmon spectrum arrays to CSV for external inspection.")
        self._configure_plasmon_coordinate_spin(axis=PLASMON_AXIS_ANGLE_DEG, canonical_value=45.0)
        self._configure_plasmon_scan_spins(axis=PLASMON_AXIS_ANGLE_DEG, start=10.0, stop=140.0)

        self.spectroscopy_controls_group = QtWidgets.QGroupBox("Spectroscopy Settings")
        spectroscopy_layout = QtWidgets.QFormLayout(self.spectroscopy_controls_group)
        self.spectroscopy_wavelength_spin = QtWidgets.QDoubleSpinBox()
        self.spectroscopy_wavelength_spin.setRange(0.1, 10000.0)
        self.spectroscopy_wavelength_spin.setDecimals(3)
        self.spectroscopy_wavelength_spin.setSingleStep(1.0)
        self.spectroscopy_wavelength_spin.setSuffix(" nm")
        self.spectroscopy_wavelength_spin.setValue(500.0)
        self.spectroscopy_shift_unit_combo = QtWidgets.QComboBox()
        self.spectroscopy_shift_unit_combo.addItem("nm", "nm")
        self.spectroscopy_shift_unit_combo.addItem("eV", "ev")
        self.spectroscopy_shift_unit_combo.addItem("meV", "mev")
        self.spectroscopy_shift_unit_combo.addItem("ueV", "uev")
        self.spectroscopy_line_label = QtWidgets.QLabel("Line wavelength")
        spectroscopy_layout.addRow(self.spectroscopy_line_label, self.spectroscopy_wavelength_spin)
        spectroscopy_layout.addRow("Shift display", self.spectroscopy_shift_unit_combo)

        button_row = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Recompute")
        self.refresh_button.clicked.connect(self.refresh_requested)
        button_row.addWidget(self.refresh_button)
        button_row.addStretch(1)
        controls_layout.addLayout(button_row)
        controls_layout.addStretch(1)

        self._controls_by_group = [
            self.observation_side_combo,
            self.los_angle_spin,
            self.profile_coordinate_combo,
            self.weighting_combo,
            self.reuse_viewer_subset_checkbox,
            self.region_list,
            self.material_list,
            self.exclude_entry_region_checkbox,
            self.exclude_low_density_checkbox,
            self.exclude_opposite_velocity_checkbox,
            self.min_density_spin,
            self.zone_lower_spin,
            self.zone_upper_spin,
            self.xrd_energy_spin,
            self.xrd_angle_spin,
            self.plasmon_model_combo,
            self.plasmon_benchmark_preset_combo,
            self.plasmon_study_mode_combo,
            self.plasmon_execution_mode_combo,
            self.plasmon_observable_mode_combo,
            self.plasmon_energy_spin,
            self.plasmon_coordinate_axis_combo,
            self.plasmon_angle_spin,
            self.plasmon_scan_axis_combo,
            self.plasmon_scan_start_spin,
            self.plasmon_scan_stop_spin,
            self.plasmon_scan_points_spin,
            self.plasmon_compare_models_checkbox,
            self.plasmon_compare_model_list,
            self.plasmon_compare_policies_checkbox,
            self.plasmon_gamma_spin,
            self.plasmon_electron_policy_combo,
            self.plasmon_integration_combo,
            self.plasmon_collision_model_combo,
            self.plasmon_collision_scale_spin,
            self.plasmon_manual_collision_spin,
            self.plasmon_lfc_combo,
            self.plasmon_window_spin,
            self.plasmon_points_spin,
            self.plasmon_fwhm_spin,
            self.plasmon_normalization_combo,
            self.plasmon_cluster_log_ne_spin,
            self.plasmon_cluster_log_te_spin,
            self.plasmon_cluster_z_spin,
            self.spectroscopy_wavelength_spin,
        ]
        for widget in self._controls_by_group:
            if isinstance(widget, QtWidgets.QAbstractSpinBox):
                widget.valueChanged.connect(self._schedule_parameters_changed)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self._schedule_parameters_changed)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(self._schedule_parameters_changed)
        self.region_list.itemChanged.connect(lambda _item: self._schedule_parameters_changed())
        self.material_list.itemChanged.connect(lambda _item: self._schedule_parameters_changed())
        self.xrd_display_combo.currentIndexChanged.connect(self._refresh_display_only)
        self.plasmon_model_combo.currentIndexChanged.connect(self._sync_plasmon_placeholder_controls_enabled)
        self.plasmon_study_mode_combo.currentIndexChanged.connect(self._on_plasmon_study_controls_changed)
        self.plasmon_coordinate_axis_combo.currentIndexChanged.connect(self._on_plasmon_coordinate_axis_changed)
        self.plasmon_scan_axis_combo.currentIndexChanged.connect(self._on_plasmon_scan_axis_changed)
        self.plasmon_execution_mode_combo.currentIndexChanged.connect(self._sync_plasmon_placeholder_controls_enabled)
        self.plasmon_collision_model_combo.currentIndexChanged.connect(self._sync_plasmon_placeholder_controls_enabled)
        self.plasmon_compare_models_checkbox.toggled.connect(self._sync_plasmon_placeholder_controls_enabled)
        self.plasmon_compare_policies_checkbox.toggled.connect(self._sync_plasmon_placeholder_controls_enabled)
        self.plasmon_apply_preset_button.clicked.connect(self._apply_plasmon_benchmark_preset)
        self.plasmon_auto_recompute_checkbox.toggled.connect(self._on_plasmon_auto_recompute_toggled)
        self.plasmon_recompute_button.clicked.connect(self._trigger_plasmon_recompute)
        self.plasmon_export_button.clicked.connect(self._export_plasmon_spectrum_csv)
        self.plasmon_compare_model_list.itemChanged.connect(self._on_plasmon_compare_model_list_changed)
        self._plasmon_advanced_widgets = (
            self.plasmon_integration_combo,
            self.plasmon_collision_model_combo,
            self.plasmon_collision_scale_spin,
            self.plasmon_manual_collision_spin,
            self.plasmon_lfc_combo,
            self.plasmon_window_spin,
            self.plasmon_points_spin,
            self.plasmon_fwhm_spin,
            self.plasmon_normalization_combo,
            self.plasmon_cluster_log_ne_spin,
            self.plasmon_cluster_log_te_spin,
            self.plasmon_cluster_z_spin,
            self.plasmon_scan_start_spin,
            self.plasmon_scan_stop_spin,
            self.plasmon_scan_points_spin,
        )
        self._plasmon_heavy_controls = (
            self.plasmon_model_combo,
            self.plasmon_benchmark_preset_combo,
            self.plasmon_study_mode_combo,
            self.plasmon_execution_mode_combo,
            self.plasmon_observable_mode_combo,
            self.plasmon_energy_spin,
            self.plasmon_coordinate_axis_combo,
            self.plasmon_angle_spin,
            self.plasmon_scan_axis_combo,
            self.plasmon_scan_start_spin,
            self.plasmon_scan_stop_spin,
            self.plasmon_scan_points_spin,
            self.plasmon_compare_models_checkbox,
            self.plasmon_compare_model_list,
            self.plasmon_compare_policies_checkbox,
            self.plasmon_gamma_spin,
            self.plasmon_electron_policy_combo,
            self.plasmon_integration_combo,
            self.plasmon_collision_model_combo,
            self.plasmon_collision_scale_spin,
            self.plasmon_manual_collision_spin,
            self.plasmon_lfc_combo,
            self.plasmon_window_spin,
            self.plasmon_points_spin,
            self.plasmon_fwhm_spin,
            self.plasmon_normalization_combo,
            self.plasmon_cluster_log_ne_spin,
            self.plasmon_cluster_log_te_spin,
            self.plasmon_cluster_z_spin,
        )
        self._plasmon_request_controls = (
            *self._plasmon_heavy_controls,
            self.region_list,
            self.material_list,
            self.reuse_viewer_subset_checkbox,
            self.exclude_entry_region_checkbox,
            self.exclude_low_density_checkbox,
            self.exclude_opposite_velocity_checkbox,
            self.min_density_spin,
            self.zone_lower_spin,
            self.zone_upper_spin,
        )
        self._sync_plasmon_placeholder_controls_enabled()
        self._update_plasmon_recompute_ui()
        self.spectroscopy_shift_unit_combo.currentIndexChanged.connect(self._refresh_display_only)
        results = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)

        self.analysis_banner = QtWidgets.QLabel("")
        self.analysis_banner.setWordWrap(True)
        results_layout.addWidget(self.analysis_banner)

        self.result_status_label = QtWidgets.QLabel("Waiting for analysis update.")
        self.result_status_label.setWordWrap(True)
        results_layout.addWidget(self.result_status_label)

        self.activity_progress = QtWidgets.QProgressBar()
        self.activity_progress.setTextVisible(False)
        self.activity_progress.setRange(0, 1)
        self.activity_progress.setValue(0)
        self.activity_progress.hide()
        results_layout.addWidget(self.activity_progress)
        self.cancel_button = QtWidgets.QPushButton("Cancel current task")
        self.cancel_button.setEnabled(False)
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        results_layout.addWidget(self.cancel_button)

        self.performance_summary_label = QtWidgets.QLabel("Performance: waiting for analysis update.")
        self.performance_summary_label.setWordWrap(True)
        results_layout.addWidget(self.performance_summary_label)

        self.result_tabs = QtWidgets.QTabWidget()
        self.result_tabs.currentChanged.connect(self._handle_result_tab_changed)
        results_layout.addWidget(self.result_tabs, 1)

        self.shock_tab = QtWidgets.QWidget()
        shock_layout = QtWidgets.QVBoxLayout(self.shock_tab)
        shock_layout.setContentsMargins(6, 6, 6, 6)
        shock_layout.setSpacing(6)
        self.shock_summary_label = QtWidgets.QLabel("-")
        self.shock_summary_label.setWordWrap(True)
        shock_layout.addWidget(self.shock_summary_label)
        shock_navigation = QtWidgets.QWidget()
        shock_navigation_layout = QtWidgets.QHBoxLayout(shock_navigation)
        shock_navigation_layout.setContentsMargins(0, 0, 0, 0)
        shock_navigation_layout.setSpacing(6)
        shock_navigation_layout.addWidget(QtWidgets.QLabel("Interaction"))
        self.shock_pan_button = QtWidgets.QToolButton()
        self.shock_zoom_button = QtWidgets.QToolButton()
        self._shock_nav_group = QtGui.QActionGroup(self)
        self._shock_nav_group.setExclusive(True)
        self._shock_pan_action = QtGui.QAction("Pan", self, checkable=True)
        self._shock_zoom_action = QtGui.QAction("Zoom", self, checkable=True)
        self._shock_nav_group.addAction(self._shock_pan_action)
        self._shock_nav_group.addAction(self._shock_zoom_action)
        self.shock_pan_button.setDefaultAction(self._shock_pan_action)
        self.shock_zoom_button.setDefaultAction(self._shock_zoom_action)
        self._shock_pan_action.setChecked(True)
        self._shock_pan_action.triggered.connect(lambda checked=False: self._set_shock_navigation_mode("pan"))
        self._shock_zoom_action.triggered.connect(lambda checked=False: self._set_shock_navigation_mode("zoom"))
        self.shock_reset_position_button = QtWidgets.QPushButton("Reset Position View")
        self.shock_reset_velocity_button = QtWidgets.QPushButton("Reset Velocity View")
        shock_navigation_layout.addWidget(self.shock_pan_button)
        shock_navigation_layout.addWidget(self.shock_zoom_button)
        shock_navigation_layout.addSpacing(8)
        shock_navigation_layout.addWidget(self.shock_reset_position_button)
        shock_navigation_layout.addWidget(self.shock_reset_velocity_button)
        shock_navigation_layout.addStretch(1)
        shock_layout.addWidget(shock_navigation)
        shock_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        shock_splitter.setChildrenCollapsible(False)
        self.shock_position_plot = CurvePlotWidget()
        self.shock_velocity_plot = CurvePlotWidget()
        shock_splitter.addWidget(self.shock_position_plot)
        shock_splitter.addWidget(self.shock_velocity_plot)
        shock_splitter.setStretchFactor(0, 1)
        shock_splitter.setStretchFactor(1, 1)
        shock_layout.addWidget(shock_splitter, 1)
        self.shock_reset_position_button.clicked.connect(self.shock_position_plot.reset_view)
        self.shock_reset_velocity_button.clicked.connect(self.shock_velocity_plot.reset_view)
        self.shock_interface_table = QtWidgets.QTableWidget(0, 4)
        self.shock_interface_table.setHorizontalHeaderLabels(["Interface", "Boundary zone", "Crossing time", "Position"])
        self.shock_interface_table.horizontalHeader().setStretchLastSection(True)
        self.shock_interface_table.verticalHeader().setVisible(False)
        shock_layout.addWidget(self.shock_interface_table)
        self.result_tabs.addTab(self.shock_tab, "Shock")

        self.wavefront_tab = QtWidgets.QWidget()
        wavefront_layout = QtWidgets.QVBoxLayout(self.wavefront_tab)
        wavefront_layout.setContentsMargins(6, 6, 6, 6)
        wavefront_layout.setSpacing(6)
        self.wavefront_summary_label = QtWidgets.QLabel("Open WaveFront to load multi-branch wave tracking for the active run.")
        self.wavefront_summary_label.setWordWrap(True)
        wavefront_layout.addWidget(self.wavefront_summary_label)
        self.wavefront_metrics_label = QtWidgets.QLabel("Performance: waiting for WaveFront analysis.")
        self.wavefront_metrics_label.setWordWrap(True)
        wavefront_layout.addWidget(self.wavefront_metrics_label)
        self.wavefront_overview_label = QtWidgets.QLabel(
            "Default WaveFront view shows tracked branches only. Provisional detections stay out of the main plots unless explicitly requested."
        )
        self.wavefront_overview_label.setWordWrap(True)
        wavefront_layout.addWidget(self.wavefront_overview_label)
        wavefront_controls = QtWidgets.QWidget()
        wavefront_controls_layout = QtWidgets.QHBoxLayout(wavefront_controls)
        wavefront_controls_layout.setContentsMargins(0, 0, 0, 0)
        wavefront_controls_layout.setSpacing(6)
        wavefront_controls_layout.addWidget(QtWidgets.QLabel("Display"))
        self.wavefront_display_combo = QtWidgets.QComboBox()
        self.wavefront_display_combo.addItem("Primary branch position vs time", "primary_position")
        self.wavefront_display_combo.addItem("Primary branch speed vs time", "primary_speed")
        self.wavefront_display_combo.addItem("Branch position vs time", "position")
        self.wavefront_display_combo.addItem("Branch evidence vs time", "evidence")
        self.wavefront_display_combo.addItem("Branch speed vs time", "speed")
        self.wavefront_display_combo.addItem("Branch width / thickness", "width")
        self.wavefront_display_combo.addItem("Significance / support ranking", "significance")
        self.wavefront_display_combo.addItem("Interface-event summary", "events")
        self.wavefront_display_combo.addItem("Warnings / suppressed detections", "warnings")
        wavefront_controls_layout.addWidget(self.wavefront_display_combo)
        wavefront_controls_layout.addSpacing(12)
        wavefront_controls_layout.addWidget(QtWidgets.QLabel("Branch set"))
        self.wavefront_scope_combo = QtWidgets.QComboBox()
        self.wavefront_scope_combo.addItem("Tracked branches", "tracked")
        self.wavefront_scope_combo.addItem("Top significant branches", "top_significant")
        self.wavefront_scope_combo.addItem("Primary branch only", "primary")
        self.wavefront_scope_combo.addItem("Compressive branches", "compressive")
        self.wavefront_scope_combo.addItem("Release / rarefaction branches", "release")
        self.wavefront_scope_combo.addItem("Reflected branches", "reflected")
        self.wavefront_scope_combo.addItem("Transmitted branches", "transmitted")
        self.wavefront_scope_combo.addItem("Tracked + short / weak", "tracked_weak")
        self.wavefront_scope_combo.addItem("All summaries incl. provisional", "all")
        wavefront_controls_layout.addWidget(self.wavefront_scope_combo)
        wavefront_controls_layout.addSpacing(12)
        wavefront_controls_layout.addWidget(QtWidgets.QLabel("Direction"))
        self.wavefront_direction_combo = QtWidgets.QComboBox()
        self.wavefront_direction_combo.addItem("All directions", "all")
        self.wavefront_direction_combo.addItem("Low to high", "low_to_high")
        self.wavefront_direction_combo.addItem("High to low", "high_to_low")
        wavefront_controls_layout.addWidget(self.wavefront_direction_combo)
        wavefront_controls_layout.addStretch(1)
        wavefront_layout.addWidget(wavefront_controls)
        self.wavefront_plot_empty_label = QtWidgets.QLabel(
            "This WaveFront view is summary-oriented. Use the ranked branch table, interface-event table, and notes below."
        )
        self.wavefront_plot_empty_label.setWordWrap(True)
        self.wavefront_plot_empty_label.hide()
        wavefront_layout.addWidget(self.wavefront_plot_empty_label)
        self.wavefront_plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.wavefront_plot_splitter.setChildrenCollapsible(False)
        self.wavefront_position_plot = CurvePlotWidget()
        self.wavefront_score_plot = CurvePlotWidget()
        self.wavefront_plot_splitter.addWidget(self.wavefront_position_plot)
        self.wavefront_plot_splitter.addWidget(self.wavefront_score_plot)
        self.wavefront_plot_splitter.setStretchFactor(0, 1)
        self.wavefront_plot_splitter.setStretchFactor(1, 1)
        wavefront_layout.addWidget(self.wavefront_plot_splitter, 1)

        self.wavefront_lower_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.wavefront_lower_splitter.setChildrenCollapsible(False)

        branch_panel = QtWidgets.QWidget()
        branch_layout = QtWidgets.QVBoxLayout(branch_panel)
        branch_layout.setContentsMargins(0, 0, 0, 0)
        branch_layout.setSpacing(6)
        branch_layout.addWidget(QtWidgets.QLabel("Branch ranking"))
        self.wavefront_branch_table = QtWidgets.QTableWidget(0, 11)
        self.wavefront_branch_table.setHorizontalHeaderLabels(
            [
                "Branch",
                "Type",
                "Support",
                "Significance",
                "Samples",
                "Duration",
                "Evidence",
                "Confidence",
                "Ambiguous",
                "Direction",
                "Breakout",
            ]
        )
        self.wavefront_branch_table.horizontalHeader().setStretchLastSection(True)
        self.wavefront_branch_table.verticalHeader().setVisible(False)
        branch_layout.addWidget(self.wavefront_branch_table, 1)
        self.wavefront_lower_splitter.addWidget(branch_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(QtWidgets.QLabel("Interface events"))
        self.wavefront_event_table = QtWidgets.QTableWidget(0, 14)
        self.wavefront_event_table.setHorizontalHeaderLabels(
            [
                "Event",
                "Interface",
                "Time",
                "Outcome",
                "Support",
                "Signif",
                "Conf",
                "Incident",
                "Tx / Rx",
                "Impulse",
                "T_E",
                "R_E",
                "Channel",
                "Ambig",
            ]
        )
        self.wavefront_event_table.horizontalHeader().setStretchLastSection(True)
        self.wavefront_event_table.verticalHeader().setVisible(False)
        right_layout.addWidget(self.wavefront_event_table, 1)
        right_layout.addWidget(QtWidgets.QLabel("Warnings and notes"))
        self.wavefront_notes = QtWidgets.QTextBrowser()
        self.wavefront_notes.setFont(build_mono_font())
        right_layout.addWidget(self.wavefront_notes, 1)
        self.wavefront_lower_splitter.addWidget(right_panel)
        self.wavefront_lower_splitter.setStretchFactor(0, 1)
        self.wavefront_lower_splitter.setStretchFactor(1, 1)
        wavefront_layout.addWidget(self.wavefront_lower_splitter, 1)
        self.wavefront_display_combo.currentIndexChanged.connect(self._refresh_wavefront_view)
        self.wavefront_scope_combo.currentIndexChanged.connect(self._refresh_wavefront_view)
        self.wavefront_direction_combo.currentIndexChanged.connect(self._refresh_wavefront_view)

        self.result_tabs.addTab(self.wavefront_tab, "WaveFront")

        self.preheat_tab = QtWidgets.QWidget()
        preheat_layout = QtWidgets.QVBoxLayout(self.preheat_tab)
        preheat_layout.setContentsMargins(6, 6, 6, 6)
        preheat_layout.setSpacing(6)
        self.preheat_summary_label = QtWidgets.QLabel("Open Preheat to load target pre-modification diagnostics for the active run.")
        self.preheat_summary_label.setWordWrap(True)
        preheat_layout.addWidget(self.preheat_summary_label)
        self.preheat_metrics_label = QtWidgets.QLabel("Performance: waiting for Preheat analysis.")
        self.preheat_metrics_label.setWordWrap(True)
        preheat_layout.addWidget(self.preheat_metrics_label)
        self.preheat_overview_label = QtWidgets.QLabel(
            "Preheat diagnoses how the selected target region changes before the tracked primary compressive branch arrives."
        )
        self.preheat_overview_label.setWordWrap(True)
        preheat_layout.addWidget(self.preheat_overview_label)

        preheat_controls = QtWidgets.QWidget()
        preheat_controls_layout = QtWidgets.QGridLayout(preheat_controls)
        preheat_controls_layout.setContentsMargins(0, 0, 0, 0)
        preheat_controls_layout.setHorizontalSpacing(8)
        preheat_controls_layout.setVerticalSpacing(4)
        preheat_controls_layout.addWidget(QtWidgets.QLabel("Region of interest"), 0, 0)
        self.preheat_target_combo = QtWidgets.QComboBox()
        self.preheat_target_combo.addItem("Auto guess", None)
        preheat_controls_layout.addWidget(self.preheat_target_combo, 0, 1)
        preheat_controls_layout.addWidget(QtWidgets.QLabel("Time mode"), 0, 2)
        self.preheat_time_mode_combo = QtWidgets.QComboBox()
        self.preheat_time_mode_combo.addItem("Before shock entry", "shock_relative")
        self.preheat_time_mode_combo.addItem("Manual snapshot / time", "manual")
        preheat_controls_layout.addWidget(self.preheat_time_mode_combo, 0, 3)
        preheat_controls_layout.addWidget(QtWidgets.QLabel("Pre-entry offset"), 0, 4)
        self.preheat_offset_combo = QtWidgets.QComboBox()
        self.preheat_offset_combo.addItem("Latest pre-entry snapshot", 0)
        self.preheat_offset_combo.addItem("1 snapshot earlier", 1)
        self.preheat_offset_combo.addItem("2 snapshots earlier", 2)
        self.preheat_offset_combo.addItem("3 snapshots earlier", 3)
        preheat_controls_layout.addWidget(self.preheat_offset_combo, 0, 5)
        preheat_controls_layout.addWidget(QtWidgets.QLabel("Snapshot"), 1, 0)
        self.preheat_snapshot_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        apply_absolute_click_slider_behavior(self.preheat_snapshot_slider)
        self.preheat_snapshot_slider.setTracking(True)
        preheat_controls_layout.addWidget(self.preheat_snapshot_slider, 1, 1, 1, 3)
        self.preheat_snapshot_spin = QtWidgets.QSpinBox()
        self.preheat_snapshot_spin.setKeyboardTracking(False)
        preheat_controls_layout.addWidget(self.preheat_snapshot_spin, 1, 4)
        self.preheat_time_spin = QtWidgets.QDoubleSpinBox()
        self.preheat_time_spin.setKeyboardTracking(False)
        self.preheat_time_spin.setDecimals(6)
        self.preheat_time_spin.setRange(-1.0e18, 1.0e18)
        self.preheat_time_spin.setSingleStep(0.1)
        self.preheat_time_spin.setSuffix(f" {self._time_unit()}")
        preheat_controls_layout.addWidget(self.preheat_time_spin, 1, 5)
        self.preheat_time_status_label = QtWidgets.QLabel("Select a region of interest and open this tab to inspect preheat before shock entry or at a manual snapshot.")
        self.preheat_time_status_label.setWordWrap(True)
        preheat_controls_layout.addWidget(self.preheat_time_status_label, 2, 0, 1, 6)
        preheat_layout.addWidget(preheat_controls)

        self.preheat_main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.preheat_main_splitter.setChildrenCollapsible(False)

        self.preheat_upper_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.preheat_upper_splitter.setChildrenCollapsible(False)
        self.preheat_plot_panel = DerivedPlotPanel()
        self.preheat_plot_panel.time_combo.setToolTip("Choose a preheat time trace for the selected region of interest.")
        self.preheat_plot_panel.profile_combo.setToolTip("Choose a selected-snapshot target profile field.")
        self.preheat_upper_splitter.addWidget(self.preheat_plot_panel)

        preheat_snapshot_panel = QtWidgets.QWidget()
        preheat_snapshot_layout = QtWidgets.QVBoxLayout(preheat_snapshot_panel)
        preheat_snapshot_layout.setContentsMargins(0, 0, 0, 0)
        preheat_snapshot_layout.setSpacing(6)
        preheat_snapshot_layout.addWidget(QtWidgets.QLabel("Selected snapshot"))
        self.preheat_snapshot_label = QtWidgets.QLabel("No selected snapshot")
        self.preheat_snapshot_label.setWordWrap(True)
        preheat_snapshot_layout.addWidget(self.preheat_snapshot_label)
        self.preheat_snapshot_table = QtWidgets.QTableWidget(0, 3)
        self.preheat_snapshot_table.setHorizontalHeaderLabels(["Metric", "Value", "Notes"])
        self.preheat_snapshot_table.horizontalHeader().setStretchLastSection(True)
        self.preheat_snapshot_table.verticalHeader().setVisible(False)
        preheat_snapshot_layout.addWidget(self.preheat_snapshot_table, 1)
        self.preheat_profile_status_label = QtWidgets.QLabel("Snapshot profiles will appear here once a valid preheat region and time are available.")
        self.preheat_profile_status_label.setWordWrap(True)
        preheat_snapshot_layout.addWidget(self.preheat_profile_status_label)
        self.preheat_upper_splitter.addWidget(preheat_snapshot_panel)
        self.preheat_upper_splitter.setStretchFactor(0, 3)
        self.preheat_upper_splitter.setStretchFactor(1, 2)
        self.preheat_upper_splitter.setSizes([980, 520])

        self.preheat_lower_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.preheat_lower_splitter.setChildrenCollapsible(False)

        preheat_summary_panel = QtWidgets.QWidget()
        preheat_summary_layout = QtWidgets.QVBoxLayout(preheat_summary_panel)
        preheat_summary_layout.setContentsMargins(0, 0, 0, 0)
        preheat_summary_layout.setSpacing(6)
        preheat_summary_layout.addWidget(QtWidgets.QLabel("Summary"))
        self.preheat_summary_table = QtWidgets.QTableWidget(0, 4)
        self.preheat_summary_table.setHorizontalHeaderLabels(["Category", "Metric", "Value", "Notes"])
        self.preheat_summary_table.horizontalHeader().setStretchLastSection(True)
        self.preheat_summary_table.verticalHeader().setVisible(False)
        preheat_summary_layout.addWidget(self.preheat_summary_table, 1)
        preheat_summary_layout.addWidget(QtWidgets.QLabel("Onset markers"))
        self.preheat_onset_table = QtWidgets.QTableWidget(0, 4)
        self.preheat_onset_table.setHorizontalHeaderLabels(["Marker", "First time", "Observed value", "Threshold / Notes"])
        self.preheat_onset_table.horizontalHeader().setStretchLastSection(True)
        self.preheat_onset_table.verticalHeader().setVisible(False)
        preheat_summary_layout.addWidget(self.preheat_onset_table, 1)
        self.preheat_lower_splitter.addWidget(preheat_summary_panel)

        preheat_budget_panel = QtWidgets.QWidget()
        preheat_budget_layout = QtWidgets.QVBoxLayout(preheat_budget_panel)
        preheat_budget_layout.setContentsMargins(0, 0, 0, 0)
        preheat_budget_layout.setSpacing(6)
        preheat_budget_layout.addWidget(QtWidgets.QLabel("Integrated budgets"))
        self.preheat_budget_table = QtWidgets.QTableWidget(0, 4)
        self.preheat_budget_table.setHorizontalHeaderLabels(["Budget", "Integrated value", "Fraction", "Notes"])
        self.preheat_budget_table.horizontalHeader().setStretchLastSection(True)
        self.preheat_budget_table.verticalHeader().setVisible(False)
        preheat_budget_layout.addWidget(self.preheat_budget_table, 1)
        preheat_budget_layout.addWidget(QtWidgets.QLabel("Notes and limitations"))
        self.preheat_notes = QtWidgets.QTextBrowser()
        self.preheat_notes.setFont(build_mono_font())
        preheat_budget_layout.addWidget(self.preheat_notes, 1)
        self.preheat_lower_splitter.addWidget(preheat_budget_panel)
        self.preheat_lower_splitter.setStretchFactor(0, 1)
        self.preheat_lower_splitter.setStretchFactor(1, 1)
        preheat_lower_sizes = [760, 760]
        self.preheat_lower_splitter.setSizes(preheat_lower_sizes)
        self.preheat_main_splitter.addWidget(self.preheat_upper_splitter)
        self.preheat_main_splitter.addWidget(self.preheat_lower_splitter)
        self.preheat_main_splitter.setStretchFactor(0, 2)
        self.preheat_main_splitter.setStretchFactor(1, 2)
        self.preheat_main_splitter.setSizes([760, 620])
        preheat_layout.addWidget(self.preheat_main_splitter, 1)

        self.preheat_target_combo.currentIndexChanged.connect(self._schedule_parameters_changed)
        self.preheat_time_mode_combo.currentIndexChanged.connect(self._on_preheat_time_mode_changed)
        self.preheat_offset_combo.currentIndexChanged.connect(self._on_preheat_time_controls_changed)
        self.preheat_snapshot_slider.valueChanged.connect(self._on_preheat_snapshot_slider_changed)
        self.preheat_snapshot_spin.valueChanged.connect(self._on_preheat_snapshot_spin_changed)
        self.preheat_time_spin.valueChanged.connect(self._on_preheat_time_spin_changed)
        self.result_tabs.addTab(self.preheat_tab, "Preheat")

        self.xrd_tab = QtWidgets.QWidget()
        xrd_layout = QtWidgets.QVBoxLayout(self.xrd_tab)
        xrd_layout.setContentsMargins(6, 6, 6, 6)
        xrd_layout.setSpacing(6)
        self.xrd_summary_label = QtWidgets.QLabel("-")
        self.xrd_summary_label.setWordWrap(True)
        xrd_layout.addWidget(self.xrd_summary_label)
        xrd_layout.addWidget(self.xrd_controls_group)
        self.xrd_plot_panel = DerivedPlotPanel()
        xrd_layout.addWidget(self.xrd_plot_panel, 1)
        self.xrd_table = QtWidgets.QTableWidget(0, 8)
        self.xrd_table.setHorizontalHeaderLabels(["Region", "rho", "rho/rho0", "d/d0", "Q0", "Q", "Bragg shift", "Thickness"])
        self.xrd_table.horizontalHeader().setStretchLastSection(True)
        self.xrd_table.verticalHeader().setVisible(False)
        xrd_layout.addWidget(self.xrd_table)
        self._module_tab_names[self.result_tabs.addTab(self.xrd_tab, "XRD")] = "xrd"

        self.plasmon_tab = QtWidgets.QWidget()
        plasmon_layout = QtWidgets.QVBoxLayout(self.plasmon_tab)
        plasmon_layout.setContentsMargins(6, 6, 6, 6)
        plasmon_layout.setSpacing(6)
        self.plasmon_summary_label = QtWidgets.QLabel("-")
        self.plasmon_summary_label.setWordWrap(True)
        plasmon_layout.addWidget(self.plasmon_summary_label)
        self.plasmon_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.plasmon_splitter.setChildrenCollapsible(False)
        self.plasmon_splitter.setHandleWidth(8)
        self.plasmon_sidebar = QtWidgets.QWidget()
        self.plasmon_sidebar.setMinimumWidth(220)
        self.plasmon_sidebar.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        plasmon_sidebar_layout = QtWidgets.QVBoxLayout(self.plasmon_sidebar)
        plasmon_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        plasmon_sidebar_layout.setSpacing(6)
        self.plasmon_sidebar_scroll = QtWidgets.QScrollArea()
        self.plasmon_sidebar_scroll.setWidgetResizable(True)
        self.plasmon_sidebar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.plasmon_sidebar_scroll.setWidget(self.plasmon_controls_group)
        plasmon_sidebar_layout.addWidget(self.plasmon_sidebar_scroll, 1)
        self.plasmon_splitter.addWidget(self.plasmon_sidebar)
        self.plasmon_results_panel = QtWidgets.QWidget()
        plasmon_results_layout = QtWidgets.QVBoxLayout(self.plasmon_results_panel)
        plasmon_results_layout.setContentsMargins(0, 0, 0, 0)
        plasmon_results_layout.setSpacing(6)
        self.plasmon_metrics = QtWidgets.QTextBrowser()
        self.plasmon_metrics.setFont(build_mono_font())
        self.plasmon_metrics.setMinimumHeight(120)
        self.plasmon_metrics.setMaximumHeight(220)
        plasmon_results_layout.addWidget(self.plasmon_metrics)
        self.plasmon_plot_panel = DerivedPlotPanel()
        self.plasmon_plot_panel.set_section_labels("Primary plots", "Snapshot / scan plots")
        plasmon_results_layout.addWidget(self.plasmon_plot_panel, 1)
        self.plasmon_splitter.addWidget(self.plasmon_results_panel)
        self.plasmon_splitter.setStretchFactor(0, 0)
        self.plasmon_splitter.setStretchFactor(1, 1)
        self.plasmon_splitter.setSizes([280, 1240])
        plasmon_layout.addWidget(self.plasmon_splitter, 1)
        self._sync_plasmon_plot_capabilities()
        self._module_tab_names[self.result_tabs.addTab(self.plasmon_tab, "Plasmon")] = "plasmon"

        self.transmission_tab = QtWidgets.QWidget()
        transmission_layout = QtWidgets.QVBoxLayout(self.transmission_tab)
        transmission_layout.setContentsMargins(6, 6, 6, 6)
        transmission_layout.setSpacing(6)
        self.transmission_summary_label = QtWidgets.QLabel("-")
        self.transmission_summary_label.setWordWrap(True)
        transmission_layout.addWidget(self.transmission_summary_label)
        self.transmission_refinement_group = QtWidgets.QGroupBox("Transmission model")
        transmission_refinement_layout = QtWidgets.QGridLayout(self.transmission_refinement_group)
        transmission_refinement_layout.setContentsMargins(8, 8, 8, 8)
        transmission_refinement_layout.setHorizontalSpacing(8)
        transmission_refinement_layout.setVerticalSpacing(4)
        self.transmission_mode_combo = QtWidgets.QComboBox()
        self.transmission_mode_combo.addItem("Auto hybrid", "auto_hybrid")
        self.transmission_mode_combo.addItem("Thomson", "thomson")
        self.transmission_mode_combo.addItem("Free-free", "free_free")
        self.transmission_mode_combo.addItem("Free-free + Thomson", "free_free_thomson")
        self.transmission_mode_combo.addItem("XCOM", "xcom")
        self.transmission_mode_combo.setCurrentIndex(max(0, self.transmission_mode_combo.findData("thomson")))
        self.transmission_model_label = QtWidgets.QLabel("Model used: Thomson quick-look")
        self.transmission_model_label.setWordWrap(True)
        self.transmission_backend_label = QtWidgets.QLabel("Backend: not yet probed")
        self.transmission_backend_label.setWordWrap(True)
        self.transmission_applicability_label = QtWidgets.QLabel("Applicability: estimated on apply")
        self.transmission_applicability_label.setWordWrap(True)
        self.transmission_refinement_label = QtWidgets.QLabel("Status: Thomson quick-look estimate.")
        self.transmission_refinement_label.setWordWrap(True)
        self.transmission_energy_unit_combo = QtWidgets.QComboBox()
        self.transmission_energy_unit_combo.addItem("eV", "eV")
        self.transmission_energy_unit_combo.addItem("keV", "keV")
        self.transmission_energy_unit_combo.addItem("Angstrom", "Angstrom")
        self.transmission_energy_unit_combo.addItem("nm", "nm")
        self.transmission_energy_spin = QtWidgets.QDoubleSpinBox()
        self.transmission_energy_unit_combo.setCurrentIndex(max(0, self.transmission_energy_unit_combo.findData("eV")))
        self.transmission_energy_spin.setKeyboardTracking(False)
        self.transmission_refine_button = QtWidgets.QPushButton("Apply Transmission Model")
        self.transmission_refine_button.clicked.connect(self.transmission_refine_requested)
        self.transmission_mode_combo.currentIndexChanged.connect(self._on_transmission_controls_changed)
        self.transmission_energy_unit_combo.currentIndexChanged.connect(self._on_transmission_energy_unit_changed)
        self.transmission_energy_spin.valueChanged.connect(self._on_transmission_controls_changed)
        transmission_refinement_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        transmission_refinement_layout.addWidget(self.transmission_mode_combo, 0, 1)
        transmission_refinement_layout.addWidget(QtWidgets.QLabel("Photon energy"), 0, 2)
        transmission_refinement_layout.addWidget(self.transmission_energy_spin, 0, 3)
        transmission_refinement_layout.addWidget(self.transmission_energy_unit_combo, 0, 4)
        transmission_refinement_layout.addWidget(self.transmission_refine_button, 0, 5)
        transmission_refinement_layout.addWidget(self.transmission_model_label, 1, 0, 1, 6)
        transmission_refinement_layout.addWidget(self.transmission_backend_label, 2, 0, 1, 6)
        transmission_refinement_layout.addWidget(self.transmission_applicability_label, 3, 0, 1, 6)
        transmission_refinement_layout.addWidget(self.transmission_refinement_label, 4, 0, 1, 6)
        transmission_layout.addWidget(self.transmission_refinement_group)
        self.transmission_status_pane = QtWidgets.QTextBrowser()
        self.transmission_status_pane.setOpenExternalLinks(False)
        self.transmission_status_pane.setMinimumHeight(120)
        transmission_layout.addWidget(self.transmission_status_pane)
        self.transmission_plot_panel = DerivedPlotPanel()
        transmission_layout.addWidget(self.transmission_plot_panel, 1)
        self.transmission_table = QtWidgets.QTableWidget(0, 15)
        self.transmission_table.setHorizontalHeaderLabels(
            [
                "Region",
                "Areal density",
                "Target share",
                "Electron column",
                "Thomson tau",
                "Free-free tau",
                "XCOM tau",
                "Total tau",
                "XCOM path",
                "FF+Th path",
                "Th fallback path",
                "XCOM tau frac",
                "FF+Th tau frac",
                "Th fallback tau frac",
                "Region mixture",
            ]
        )
        self.transmission_table.horizontalHeader().setStretchLastSection(True)
        self.transmission_table.verticalHeader().setVisible(False)
        transmission_layout.addWidget(self.transmission_table)
        self._module_tab_names[self.result_tabs.addTab(self.transmission_tab, "Transmission")] = "transmission"
        self._configure_transmission_energy_spin(energy_kev=8.0)
        self._on_transmission_controls_changed()

        self.spectroscopy_tab = QtWidgets.QWidget()
        spectroscopy_layout = QtWidgets.QVBoxLayout(self.spectroscopy_tab)
        spectroscopy_layout.setContentsMargins(6, 6, 6, 6)
        spectroscopy_layout.setSpacing(6)
        self.spectroscopy_summary_label = QtWidgets.QLabel("-")
        self.spectroscopy_summary_label.setWordWrap(True)
        spectroscopy_layout.addWidget(self.spectroscopy_summary_label)
        spectroscopy_layout.addWidget(self.spectroscopy_controls_group)
        self.spectroscopy_metrics = QtWidgets.QTextBrowser()
        self.spectroscopy_metrics.setFont(build_mono_font())
        spectroscopy_layout.addWidget(self.spectroscopy_metrics)
        self.spectroscopy_plot_panel = DerivedPlotPanel()
        spectroscopy_layout.addWidget(self.spectroscopy_plot_panel, 1)
        self._module_tab_names[self.result_tabs.addTab(self.spectroscopy_tab, "Spectroscopy")] = "spectroscopy"

        self.warnings_tab = QtWidgets.QWidget()
        warnings_layout = QtWidgets.QVBoxLayout(self.warnings_tab)
        warnings_layout.setContentsMargins(6, 6, 6, 6)
        warnings_layout.setSpacing(6)
        self.warnings_tree = QtWidgets.QTreeWidget()
        self.warnings_tree.setHeaderLabels(["Severity / Module", "Message"])
        self.warnings_tree.header().setStretchLastSection(True)
        warnings_layout.addWidget(self.warnings_tree, 1)
        self.result_tabs.addTab(self.warnings_tab, "Warnings")

        splitter.addWidget(controls_scroll)
        splitter.addWidget(results)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1320])
        for table in (
            self.shock_interface_table,
            self.wavefront_branch_table,
            self.wavefront_event_table,
            self.preheat_summary_table,
            self.preheat_snapshot_table,
            self.preheat_onset_table,
            self.preheat_budget_table,
            self.xrd_table,
            self.transmission_table,
        ):
            self._configure_readonly_table(table)
        for text_panel in (
            self.wavefront_notes,
            self.preheat_notes,
            self.plasmon_metrics,
            self.spectroscopy_metrics,
        ):
            self._configure_readonly_text_panel(text_panel)
        self._install_wheel_guard()

    def _checked_values(self, widget: QtWidgets.QListWidget) -> tuple[int, ...]:
        values: list[int] = []
        for index in range(widget.count()):
            item = widget.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                values.append(int(item.data(QtCore.Qt.UserRole)))
        return tuple(values)

    def _populate_checkable_list(
        self,
        widget: QtWidgets.QListWidget,
        values: tuple[int, ...],
        *,
        label_prefix: str,
        selected: tuple[int, ...],
    ) -> None:
        widget.blockSignals(True)
        widget.clear()
        selected_set = {int(value) for value in selected}
        for value in values:
            item = QtWidgets.QListWidgetItem(f"{label_prefix} {int(value)}")
            item.setData(QtCore.Qt.UserRole, int(value))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if int(value) in selected_set else QtCore.Qt.Unchecked)
            widget.addItem(item)
        widget.blockSignals(False)

    def _populate_preheat_target_combo(
        self,
        *,
        selected_region_id: int | None = None,
        preheat: PreheatSummary | None = None,
    ) -> None:
        auto_text = "Auto guess"
        if preheat is not None and preheat.auto_target_label:
            auto_text = f"Auto guess ({preheat.auto_target_label})"
        selected_id = None if selected_region_id is None else int(selected_region_id)
        self.preheat_target_combo.blockSignals(True)
        try:
            self.preheat_target_combo.clear()
            self.preheat_target_combo.addItem(auto_text, None)
            for region_id in self._available_region_ids:
                label = f"Region {int(region_id)}"
                if preheat is not None:
                    if preheat.auto_target_region_id is not None and int(region_id) == int(preheat.auto_target_region_id) and preheat.auto_target_label:
                        label = f"{preheat.auto_target_label} [auto]"
                    if preheat.target_region_id is not None and int(region_id) == int(preheat.target_region_id) and preheat.target_label:
                        suffix = " [selected]" if str(preheat.target_selection_mode or "auto") == "user_selected" else ""
                        label = f"{preheat.target_label}{suffix}"
                self.preheat_target_combo.addItem(label, int(region_id))
            index = self.preheat_target_combo.findData(selected_id)
            if index < 0:
                index = 0
            self.preheat_target_combo.setCurrentIndex(index)
        finally:
            self.preheat_target_combo.blockSignals(False)

    @staticmethod
    def _configure_readonly_table(table: QtWidgets.QTableWidget) -> None:
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(True)
        table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.verticalHeader().setDefaultSectionSize(24)

    @staticmethod
    def _configure_readonly_text_panel(panel: QtWidgets.QTextEdit) -> None:
        panel.setReadOnly(True)
        panel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        panel.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)

    def _install_wheel_guard(self) -> None:
        guarded_widgets = (
            self.observation_side_combo,
            self.los_angle_spin,
            self.profile_coordinate_combo,
            self.weighting_combo,
            self.min_density_spin,
            self.zone_lower_spin,
            self.zone_upper_spin,
            self.xrd_energy_spin,
            self.xrd_angle_spin,
            self.xrd_display_combo,
            self.plasmon_model_combo,
            self.plasmon_execution_mode_combo,
            self.plasmon_observable_mode_combo,
            self.plasmon_energy_spin,
            self.plasmon_angle_spin,
            self.plasmon_gamma_spin,
            self.plasmon_integration_combo,
            self.plasmon_collision_model_combo,
            self.plasmon_collision_scale_spin,
            self.plasmon_manual_collision_spin,
            self.plasmon_lfc_combo,
            self.plasmon_window_spin,
            self.plasmon_points_spin,
            self.plasmon_fwhm_spin,
            self.plasmon_normalization_combo,
            self.plasmon_cluster_log_ne_spin,
            self.plasmon_cluster_log_te_spin,
            self.plasmon_cluster_z_spin,
            self.spectroscopy_wavelength_spin,
            self.spectroscopy_shift_unit_combo,
            self.preheat_target_combo,
            self.preheat_time_mode_combo,
            self.preheat_offset_combo,
            self.preheat_snapshot_spin,
            self.preheat_time_spin,
        )
        for widget in guarded_widgets:
            widget.installEventFilter(self._wheel_guard)

    def _schedule_parameters_changed(self, *args) -> None:
        del args
        sender = self.sender()
        if sender in getattr(self, "_plasmon_request_controls", ()):
            self._plasmon_controls_dirty = True
        if self._should_defer_plasmon_recompute(sender):
            self._plasmon_recompute_pending = True
            self._update_plasmon_recompute_ui()
            return
        self._parameter_change_timer.start()

    def _active_time_plot_module(self) -> str | None:
        return self._module_tab_names.get(int(self.result_tabs.currentIndex()))

    def requested_time_plot_modules(self) -> frozenset[str]:
        module_name = self._active_time_plot_module()
        if module_name is None:
            return frozenset()
        return frozenset({module_name})

    def transmission_requested(self) -> bool:
        return int(self.result_tabs.currentIndex()) == int(self.result_tabs.indexOf(self.transmission_tab))

    def selected_transmission_mode(self) -> str:
        return str(self.transmission_mode_combo.currentData() or "thomson")

    def _transmission_energy_unit(self) -> str:
        return _normalize_photon_unit(str(self.transmission_energy_unit_combo.currentData() or "eV"))

    def _display_transmission_energy_from_kev(self, value_kev: float) -> float:
        return float(self._display_photon_energy_from_kev(float(value_kev), unit=self._transmission_energy_unit()))

    def _transmission_display_to_kev(self, displayed_value: float) -> float:
        return float(self._photon_display_to_kev(float(displayed_value), unit=self._transmission_energy_unit()))

    def _configure_transmission_energy_spin(self, *, energy_kev: float | None = None) -> None:
        unit = self._transmission_energy_unit()
        native_kev = (
            float(energy_kev)
            if energy_kev is not None and math.isfinite(float(energy_kev))
            else self._transmission_display_to_kev(float(self.transmission_energy_spin.value() or 0.0))
        )
        self.transmission_energy_spin.blockSignals(True)
        try:
            self.transmission_energy_spin.setSuffix(f" {_photon_unit_label(unit)}")
            if unit == "nm":
                self.transmission_energy_spin.setDecimals(5)
                self.transmission_energy_spin.setRange(1.0e-5, 100.0)
                self.transmission_energy_spin.setSingleStep(0.001)
            elif unit == "Angstrom":
                self.transmission_energy_spin.setDecimals(5)
                self.transmission_energy_spin.setRange(1.0e-4, 1000.0)
                self.transmission_energy_spin.setSingleStep(0.01)
            elif unit == "eV":
                self.transmission_energy_spin.setDecimals(3)
                self.transmission_energy_spin.setRange(1.0, 1.0e6)
                self.transmission_energy_spin.setSingleStep(100.0)
            else:
                self.transmission_energy_spin.setDecimals(5)
                self.transmission_energy_spin.setRange(1.0e-3, 1.0e3)
                self.transmission_energy_spin.setSingleStep(0.1)
            self.transmission_energy_spin.setValue(self._display_photon_energy_from_kev(native_kev, unit=unit))
            self.transmission_energy_spin.setProperty("display_unit", unit)
        finally:
            self.transmission_energy_spin.blockSignals(False)

    def _on_transmission_energy_unit_changed(self, *args) -> None:
        del args
        previous_unit = _normalize_photon_unit(str(self.transmission_energy_spin.property("display_unit") or "eV"))
        current_kev = float(self._photon_display_to_kev(float(self.transmission_energy_spin.value()), unit=previous_unit))
        self._configure_transmission_energy_spin(energy_kev=current_kev)
        self._on_transmission_controls_changed()

    def _selected_transmission_energy_kev(self) -> float:
        return float(self._transmission_display_to_kev(float(self.transmission_energy_spin.value())))

    @staticmethod
    def _transmission_mode_supports_energy(mode: str) -> bool:
        return str(mode) in {"auto_hybrid", "free_free", "free_free_thomson", "xcom"}

    def _transmission_request_matches_result(self, transmission: TransmissionResult | None = None) -> bool:
        if transmission is None:
            if self._current_result is None:
                return False
            transmission = self._current_result.transmission
        if self._current_result is None:
            return False
        if int(self._current_result.snapshot_index) != int(self._context.snapshot_index):
            return False
        if int(transmission.snapshot_index) != int(self._context.snapshot_index):
            return False
        requested_mode = self.selected_transmission_mode()
        if str(transmission.selected_mode or "thomson") != requested_mode:
            return False
        if self._transmission_mode_supports_energy(requested_mode):
            result_energy = transmission.photon_energy_kev
            if result_energy is None or not math.isfinite(float(result_energy)):
                return False
            tolerance = max(1.0e-6, abs(float(result_energy)) * 1.0e-6)
            if abs(float(result_energy) - self._selected_transmission_energy_kev()) > tolerance:
                return False
        return True

    def _selected_plasmon_model(self) -> str:
        return str(self.plasmon_model_combo.currentData() or PLASMON_MODEL_QUICKLOOK)

    def _selected_plasmon_execution_mode(self) -> str:
        return str(self.plasmon_execution_mode_combo.currentData() or PLASMON_EXECUTION_MODE_QUICKLOOK)

    def _selected_plasmon_observable_mode(self) -> str:
        return str(self.plasmon_observable_mode_combo.currentData() or PLASMON_OBSERVABLE_MODE_DIELECTRIC)

    def _selected_plasmon_benchmark_preset(self) -> str:
        return str(self.plasmon_benchmark_preset_combo.currentData() or PLASMON_BENCHMARK_PRESET_NONE)

    def _selected_plasmon_study_mode(self) -> str:
        return str(self.plasmon_study_mode_combo.currentData() or PLASMON_STUDY_MODE_SPECTRUM)

    def _selected_plasmon_electron_policy(self) -> str:
        return str(self.plasmon_electron_policy_combo.currentData() or PLASMON_ELECTRON_POLICY_RAW)

    def _selected_plasmon_coordinate_axis(self) -> str:
        return str(self.plasmon_coordinate_axis_combo.currentData() or PLASMON_AXIS_ANGLE_DEG)

    def _selected_plasmon_scan_axis(self) -> str:
        return str(self.plasmon_scan_axis_combo.currentData() or PLASMON_AXIS_ANGLE_DEG)

    def _populate_plasmon_compare_model_list(self) -> None:
        self.plasmon_compare_model_list.blockSignals(True)
        try:
            self.plasmon_compare_model_list.clear()
            for label, value in PLASMON_COMPARISON_MODEL_CHOICES:
                item = QtWidgets.QListWidgetItem(str(label))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(value))
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                self.plasmon_compare_model_list.addItem(item)
        finally:
            self.plasmon_compare_model_list.blockSignals(False)

    def _checked_plasmon_compare_models(self) -> tuple[str, ...]:
        selected: list[str] = []
        for row in range(self.plasmon_compare_model_list.count()):
            item = self.plasmon_compare_model_list.item(row)
            if item is None or item.checkState() != QtCore.Qt.CheckState.Checked:
                continue
            value = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
            if value and value not in selected:
                selected.append(value)
        return tuple(selected)

    def _set_checked_plasmon_compare_models(self, values: tuple[str, ...] | list[str] | None) -> None:
        selected = (
            {str(value).strip() for value in values if str(value).strip()}
            if values is not None
            else {str(value) for _label, value in PLASMON_COMPARISON_MODEL_CHOICES}
        )
        self.plasmon_compare_model_list.blockSignals(True)
        try:
            for row in range(self.plasmon_compare_model_list.count()):
                item = self.plasmon_compare_model_list.item(row)
                if item is None:
                    continue
                value = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
                state = QtCore.Qt.CheckState.Checked if value in selected else QtCore.Qt.CheckState.Unchecked
                item.setCheckState(state)
        finally:
            self.plasmon_compare_model_list.blockSignals(False)

    def _selected_plasmon_probe_energy_kev(self) -> float:
        return float(self._photon_display_to_kev(float(self.plasmon_energy_spin.value()), unit=self._photon_spin_unit(self.plasmon_energy_spin)))

    def _plasmon_k_angstrom_inv_from_angle_deg(self, angle_deg: float, *, energy_kev: float | None = None) -> float:
        wavelength = float(plasmon_probe_wavelength_angstrom(self._selected_plasmon_probe_energy_kev() if energy_kev is None else energy_kev))
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            return float("nan")
        return float((4.0 * math.pi * math.sin(math.radians(float(angle_deg)) / 2.0)) / wavelength)

    def _plasmon_angle_deg_from_k_angstrom_inv(self, k_angstrom_inv: float, *, energy_kev: float | None = None) -> float:
        wavelength = float(plasmon_probe_wavelength_angstrom(self._selected_plasmon_probe_energy_kev() if energy_kev is None else energy_kev))
        if not math.isfinite(wavelength) or wavelength <= 0.0:
            return float("nan")
        argument = float(k_angstrom_inv) * wavelength / (4.0 * math.pi)
        if not math.isfinite(argument) or argument < 0.0 or argument > 1.0:
            return float("nan")
        return float(math.degrees(2.0 * math.asin(argument)))

    def _configure_plasmon_coordinate_spin(self, *, axis: str, canonical_value: float | None = None) -> None:
        normalized_axis = str(axis or PLASMON_AXIS_ANGLE_DEG)
        if canonical_value is None:
            if normalized_axis == PLASMON_AXIS_K_ANGSTROM_INV:
                canonical_value = float(self.plasmon_angle_spin.value())
            else:
                canonical_value = float(self._angle_display_to_deg(float(self.plasmon_angle_spin.value())))
        self.plasmon_angle_spin.blockSignals(True)
        try:
            if normalized_axis == PLASMON_AXIS_K_ANGSTROM_INV:
                self.plasmon_angle_spin.setDecimals(4)
                self.plasmon_angle_spin.setRange(0.001, 25.0)
                self.plasmon_angle_spin.setSingleStep(0.05)
                self.plasmon_angle_spin.setSuffix(" 1/A")
                self.plasmon_angle_spin.setValue(float(canonical_value))
            else:
                self.plasmon_angle_spin.setDecimals(3 if self._angle_unit() == "rad" else 2)
                self.plasmon_angle_spin.setRange(0.1 if self._angle_unit() == "deg" else 0.001, 180.0 if self._angle_unit() == "deg" else math.pi)
                self.plasmon_angle_spin.setSingleStep(1.0 if self._angle_unit() == "deg" else 0.05)
                self.plasmon_angle_spin.setSuffix(f" {_angle_unit_label(self._angle_unit())}")
                self.plasmon_angle_spin.setValue(float(self._display_angle_from_deg(float(canonical_value))))
            self.plasmon_angle_spin.setProperty("plasmon_axis", normalized_axis)
        finally:
            self.plasmon_angle_spin.blockSignals(False)

    def _configure_plasmon_scan_spins(self, *, axis: str, start: float | None = None, stop: float | None = None) -> None:
        normalized_axis = str(axis or PLASMON_AXIS_ANGLE_DEG)
        current_start = float(self.plasmon_scan_start_spin.value())
        current_stop = float(self.plasmon_scan_stop_spin.value())
        if start is None:
            start = current_start if normalized_axis == PLASMON_AXIS_K_ANGSTROM_INV else float(self._angle_display_to_deg(current_start))
        if stop is None:
            stop = current_stop if normalized_axis == PLASMON_AXIS_K_ANGSTROM_INV else float(self._angle_display_to_deg(current_stop))
        for spin, value in ((self.plasmon_scan_start_spin, start), (self.plasmon_scan_stop_spin, stop)):
            spin.blockSignals(True)
            try:
                if normalized_axis == PLASMON_AXIS_K_ANGSTROM_INV:
                    spin.setDecimals(4)
                    spin.setRange(0.001, 25.0)
                    spin.setSingleStep(0.05)
                    spin.setSuffix(" 1/A")
                    spin.setValue(float(value))
                else:
                    spin.setDecimals(3 if self._angle_unit() == "rad" else 2)
                    spin.setRange(0.1 if self._angle_unit() == "deg" else 0.001, 180.0 if self._angle_unit() == "deg" else math.pi)
                    spin.setSingleStep(1.0 if self._angle_unit() == "deg" else 0.05)
                    spin.setSuffix(f" {_angle_unit_label(self._angle_unit())}")
                    spin.setValue(float(self._display_angle_from_deg(float(value))))
                spin.setProperty("plasmon_axis", normalized_axis)
            finally:
                spin.blockSignals(False)

    def _selected_plasmon_coordinate_value(self) -> float:
        axis = self._selected_plasmon_coordinate_axis()
        value = float(self.plasmon_angle_spin.value())
        if axis == PLASMON_AXIS_K_ANGSTROM_INV:
            return value
        return float(self._angle_display_to_deg(value))

    def _selected_plasmon_fixed_angle_deg(self) -> float:
        axis = self._selected_plasmon_coordinate_axis()
        value = self._selected_plasmon_coordinate_value()
        if axis == PLASMON_AXIS_K_ANGSTROM_INV:
            return float(self._plasmon_angle_deg_from_k_angstrom_inv(value))
        return float(value)

    def _selected_plasmon_scan_start(self) -> float:
        axis = self._selected_plasmon_scan_axis()
        value = float(self.plasmon_scan_start_spin.value())
        if axis == PLASMON_AXIS_K_ANGSTROM_INV:
            return value
        return float(self._angle_display_to_deg(value))

    def _selected_plasmon_scan_stop(self) -> float:
        axis = self._selected_plasmon_scan_axis()
        value = float(self.plasmon_scan_stop_spin.value())
        if axis == PLASMON_AXIS_K_ANGSTROM_INV:
            return value
        return float(self._angle_display_to_deg(value))

    def _set_checked_values_for_filter(self, widget: QtWidgets.QListWidget, values: tuple[int, ...] | list[int] | None) -> None:
        selected = None if values is None else {int(value) for value in values}
        widget.blockSignals(True)
        try:
            for row in range(widget.count()):
                item = widget.item(row)
                if item is None:
                    continue
                value = int(item.data(QtCore.Qt.ItemDataRole.UserRole))
                state = QtCore.Qt.CheckState.Checked if selected is None or value in selected else QtCore.Qt.CheckState.Unchecked
                item.setCheckState(state)
        finally:
            widget.blockSignals(False)

    def _al_material_ids_for_dataset(self, dataset) -> tuple[int, ...]:
        mapping = resolved_material_formula_map(dataset)
        return tuple(sorted(int(material_id) for material_id, formula in mapping.items() if str(formula) == "Al"))

    def _material_zone_span(self, dataset, *, material_id: int) -> tuple[int, int]:
        zone_material = np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32))
        indices = np.flatnonzero(zone_material == int(abs(material_id)))
        if indices.size == 0:
            raise ValueError(f"No zones found for material {material_id}.")
        return int(indices[0] + 1), int(indices[-1] + 1)

    def _apply_plasmon_benchmark_preset(self) -> None:
        preset = self._selected_plasmon_benchmark_preset()
        if preset == PLASMON_BENCHMARK_PRESET_NONE:
            self.status_label.setText("Plasmon preset: no preset selected.")
            self.result_status_label.setText("Plasmon preset: no preset selected.")
            return
        if not self._context.has_run or self._context.path is None:
            self.status_label.setText("Plasmon preset: load a run before applying benchmark defaults.")
            self.result_status_label.setText("Plasmon preset: load a run before applying benchmark defaults.")
            return
        try:
            dataset = load_run_data(Path(self._context.path))
            al_ids = self._al_material_ids_for_dataset(dataset)
            if not al_ids:
                raise ValueError("No Al material resolved from EOS/opacity metadata.")
            al_material_id = int(al_ids[0])
            if preset == PLASMON_BENCHMARK_PRESET_AL_DRIVEN_ARTICLE:
                span = shocked_al_slab_summary(
                    dataset,
                    snapshot_index=int(self._context.snapshot_index),
                    density_floor_g_cm3=3.75,
                    material_id=al_material_id,
                )
                zone_lower = int(span["zone_index_lower"])
                zone_upper = int(span["zone_index_upper"])
                note = (
                    f"Applied driven Al article preset at current snapshot {int(self._context.snapshot_index)}. "
                    "This preset expects the global snapshot near 6.3 ns for article-facing comparison."
                )
            else:
                zone_lower, zone_upper = self._material_zone_span(dataset, material_id=al_material_id)
                note = "Applied ambient Al article preset. Use global snapshot 0 for the cold benchmark."
        except Exception as exc:
            self.status_label.setText(f"Plasmon preset failed: {exc}")
            self.result_status_label.setText(f"Plasmon preset failed: {exc}")
            return

        self.plasmon_benchmark_preset_combo.blockSignals(True)
        self.plasmon_model_combo.blockSignals(True)
        self.plasmon_study_mode_combo.blockSignals(True)
        self.plasmon_execution_mode_combo.blockSignals(True)
        self.plasmon_coordinate_axis_combo.blockSignals(True)
        self.plasmon_scan_axis_combo.blockSignals(True)
        self.plasmon_electron_policy_combo.blockSignals(True)
        self.plasmon_integration_combo.blockSignals(True)
        self.plasmon_collision_model_combo.blockSignals(True)
        self.plasmon_lfc_combo.blockSignals(True)
        self.plasmon_compare_models_checkbox.blockSignals(True)
        self.plasmon_compare_model_list.blockSignals(True)
        self.plasmon_compare_policies_checkbox.blockSignals(True)
        self.plasmon_energy_spin.blockSignals(True)
        self.plasmon_window_spin.blockSignals(True)
        self.plasmon_points_spin.blockSignals(True)
        self.plasmon_fwhm_spin.blockSignals(True)
        self.plasmon_scan_points_spin.blockSignals(True)
        self.zone_lower_spin.blockSignals(True)
        self.zone_upper_spin.blockSignals(True)
        try:
            self.plasmon_execution_mode_combo.setCurrentIndex(self.plasmon_execution_mode_combo.findData(PLASMON_EXECUTION_MODE_BENCHMARK))
            self.plasmon_study_mode_combo.setCurrentIndex(self.plasmon_study_mode_combo.findData(PLASMON_STUDY_MODE_DISPERSION))
            self.plasmon_coordinate_axis_combo.setCurrentIndex(self.plasmon_coordinate_axis_combo.findData(PLASMON_AXIS_K_ANGSTROM_INV))
            self.plasmon_scan_axis_combo.setCurrentIndex(self.plasmon_scan_axis_combo.findData(PLASMON_AXIS_K_ANGSTROM_INV))
            preset_policy = (
                PLASMON_ELECTRON_POLICY_ARTICLE_AL_DRIVEN_INCREMENT
                if preset == PLASMON_BENCHMARK_PRESET_AL_DRIVEN_ARTICLE
                else PLASMON_ELECTRON_POLICY_ARTICLE_AL_BENCHMARK
            )
            self.plasmon_electron_policy_combo.setCurrentIndex(self.plasmon_electron_policy_combo.findData(preset_policy))
            self.plasmon_integration_combo.setCurrentIndex(self.plasmon_integration_combo.findData("los_integrated"))
            self.plasmon_collision_model_combo.setCurrentIndex(self.plasmon_collision_model_combo.findData(PLASMON_COLLISION_MODEL_BENCHMARK_DENSE))
            self.plasmon_lfc_combo.setCurrentIndex(self.plasmon_lfc_combo.findData("esa_static"))
            self.plasmon_compare_models_checkbox.setChecked(True)
            self._set_checked_plasmon_compare_models(None)
            self.plasmon_compare_policies_checkbox.setChecked(False)
            self.plasmon_energy_spin.setValue(float(self._display_photon_energy_from_kev(8.307, unit=self._photon_spin_unit(self.plasmon_energy_spin))))
            self.plasmon_window_spin.setValue(45.0)
            self.plasmon_points_spin.setValue(1201)
            self.plasmon_fwhm_spin.setValue(3.5)
            self._configure_plasmon_coordinate_spin(axis=PLASMON_AXIS_K_ANGSTROM_INV, canonical_value=1.28)
            self._configure_plasmon_scan_spins(axis=PLASMON_AXIS_K_ANGSTROM_INV, start=0.92, stop=2.57)
            self.plasmon_scan_points_spin.setValue(61)
            self.zone_lower_spin.setValue(int(zone_lower))
            self.zone_upper_spin.setValue(int(zone_upper))
            self._set_checked_values_for_filter(self.material_list, (al_material_id,))
        finally:
            self.plasmon_benchmark_preset_combo.blockSignals(False)
            self.plasmon_model_combo.blockSignals(False)
            self.plasmon_study_mode_combo.blockSignals(False)
            self.plasmon_execution_mode_combo.blockSignals(False)
            self.plasmon_coordinate_axis_combo.blockSignals(False)
            self.plasmon_scan_axis_combo.blockSignals(False)
            self.plasmon_electron_policy_combo.blockSignals(False)
            self.plasmon_integration_combo.blockSignals(False)
            self.plasmon_collision_model_combo.blockSignals(False)
            self.plasmon_lfc_combo.blockSignals(False)
            self.plasmon_compare_models_checkbox.blockSignals(False)
            self.plasmon_compare_model_list.blockSignals(False)
            self.plasmon_compare_policies_checkbox.blockSignals(False)
            self.plasmon_energy_spin.blockSignals(False)
            self.plasmon_window_spin.blockSignals(False)
            self.plasmon_points_spin.blockSignals(False)
            self.plasmon_fwhm_spin.blockSignals(False)
            self.plasmon_scan_points_spin.blockSignals(False)
            self.zone_lower_spin.blockSignals(False)
            self.zone_upper_spin.blockSignals(False)
        self._sync_plasmon_placeholder_controls_enabled()
        self.status_label.setText(note)
        self.result_status_label.setText(note)
        self._plasmon_controls_dirty = True
        if bool(self.plasmon_auto_recompute_checkbox.isChecked()):
            self.refresh_requested.emit()
        else:
            self._plasmon_recompute_pending = True
            self._update_plasmon_recompute_ui()

    def _on_plasmon_coordinate_axis_changed(self, *args) -> None:
        del args
        previous_axis = str(self.plasmon_angle_spin.property("plasmon_axis") or PLASMON_AXIS_ANGLE_DEG)
        raw_value = float(self.plasmon_angle_spin.value())
        if previous_axis == PLASMON_AXIS_K_ANGSTROM_INV:
            canonical_value = raw_value
        else:
            canonical_value = float(self._angle_display_to_deg(raw_value))
        current_axis = self._selected_plasmon_coordinate_axis()
        self._configure_plasmon_coordinate_spin(axis=current_axis, canonical_value=canonical_value if current_axis == previous_axis else (
            self._plasmon_k_angstrom_inv_from_angle_deg(canonical_value)
            if current_axis == PLASMON_AXIS_K_ANGSTROM_INV
            else self._plasmon_angle_deg_from_k_angstrom_inv(canonical_value)
        ))
        self._on_plasmon_study_controls_changed()

    def _on_plasmon_scan_axis_changed(self, *args) -> None:
        del args
        previous_axis = str(self.plasmon_scan_start_spin.property("plasmon_axis") or PLASMON_AXIS_ANGLE_DEG)
        start_value = float(self.plasmon_scan_start_spin.value())
        stop_value = float(self.plasmon_scan_stop_spin.value())
        if previous_axis == PLASMON_AXIS_K_ANGSTROM_INV:
            canonical_start = start_value
            canonical_stop = stop_value
        else:
            canonical_start = float(self._angle_display_to_deg(start_value))
            canonical_stop = float(self._angle_display_to_deg(stop_value))
        current_axis = self._selected_plasmon_scan_axis()
        if current_axis != previous_axis:
            if current_axis == PLASMON_AXIS_K_ANGSTROM_INV:
                canonical_start = self._plasmon_k_angstrom_inv_from_angle_deg(canonical_start)
                canonical_stop = self._plasmon_k_angstrom_inv_from_angle_deg(canonical_stop)
            else:
                canonical_start = self._plasmon_angle_deg_from_k_angstrom_inv(canonical_start)
                canonical_stop = self._plasmon_angle_deg_from_k_angstrom_inv(canonical_stop)
        self._configure_plasmon_scan_spins(axis=current_axis, start=canonical_start, stop=canonical_stop)
        self._on_plasmon_study_controls_changed()

    def _on_plasmon_study_controls_changed(self, *args) -> None:
        del args
        self._plasmon_controls_dirty = True
        self._sync_plasmon_placeholder_controls_enabled()
        self._schedule_parameters_changed()

    def _on_plasmon_compare_model_list_changed(self, *_args) -> None:
        if not self._checked_plasmon_compare_models():
            fallback_model = self._selected_plasmon_model()
            if fallback_model == PLASMON_MODEL_QUICKLOOK:
                fallback_model = next((str(value) for _label, value in PLASMON_COMPARISON_MODEL_CHOICES), "")
            if fallback_model:
                self._set_checked_plasmon_compare_models((fallback_model,))
        self._plasmon_controls_dirty = True
        self._sync_plasmon_placeholder_controls_enabled()
        self._schedule_parameters_changed()

    def _current_plasmon_ui_capabilities(self) -> PlasmonUiCapabilities:
        return plasmon_ui_capabilities(
            model=self._selected_plasmon_model(),
            execution_mode=self._selected_plasmon_execution_mode(),
            study_mode=self._selected_plasmon_study_mode(),
            compare_models=bool(self.plasmon_compare_models_checkbox.isChecked()),
            compare_policies=bool(self.plasmon_compare_policies_checkbox.isChecked()),
        )

    def _sync_plasmon_plot_capabilities(self) -> None:
        if not hasattr(self, "plasmon_plot_panel"):
            return
        capabilities = self._current_plasmon_ui_capabilities()
        request_matches = (
            self._current_result is not None
            and self._plasmon_request_matches_result(self._current_result.plasmon)
        )
        if self._current_result is None:
            missing_reason = "Run the current plasmon study to populate this view."
        elif request_matches:
            missing_reason = "The selected plasmon view produced no finite data for the current configuration."
        else:
            missing_reason = "Recompute the current plasmon study to populate this view."
        self.plasmon_plot_panel.set_section_labels(capabilities.primary_label, capabilities.secondary_label)
        self.plasmon_plot_panel.set_expected_options(
            tuple((option.key, option.label) for option in capabilities.time_options),
            tuple((option.key, option.label) for option in capabilities.profile_options),
            missing_time_reason=missing_reason,
            missing_profile_reason=missing_reason,
            preferred_time_key=capabilities.preferred_time_key,
            preferred_profile_key=capabilities.preferred_profile_key,
        )

    def _sync_plasmon_controls_from_result(self, plasmon: PlasmonResult) -> None:
        controls = (
            self.plasmon_model_combo,
            self.plasmon_benchmark_preset_combo,
            self.plasmon_study_mode_combo,
            self.plasmon_execution_mode_combo,
            self.plasmon_observable_mode_combo,
            self.plasmon_energy_spin,
            self.plasmon_coordinate_axis_combo,
            self.plasmon_angle_spin,
            self.plasmon_scan_axis_combo,
            self.plasmon_scan_start_spin,
            self.plasmon_scan_stop_spin,
            self.plasmon_scan_points_spin,
            self.plasmon_compare_models_checkbox,
            self.plasmon_compare_policies_checkbox,
            self.plasmon_gamma_spin,
            self.plasmon_electron_policy_combo,
            self.plasmon_integration_combo,
            self.plasmon_collision_model_combo,
            self.plasmon_collision_scale_spin,
            self.plasmon_manual_collision_spin,
            self.plasmon_lfc_combo,
            self.plasmon_window_spin,
            self.plasmon_points_spin,
            self.plasmon_fwhm_spin,
            self.plasmon_normalization_combo,
            self.plasmon_cluster_log_ne_spin,
            self.plasmon_cluster_log_te_spin,
            self.plasmon_cluster_z_spin,
        )
        blocked = [(widget, widget.blockSignals(True)) for widget in controls]
        try:
            for combo, value in (
                (self.plasmon_model_combo, str(plasmon.requested_model_name or plasmon.model_name or PLASMON_MODEL_QUICKLOOK)),
                (self.plasmon_benchmark_preset_combo, str(getattr(plasmon, "benchmark_preset", PLASMON_BENCHMARK_PRESET_NONE))),
                (self.plasmon_study_mode_combo, str(getattr(plasmon, "study_mode", PLASMON_STUDY_MODE_SPECTRUM))),
                (self.plasmon_execution_mode_combo, str(getattr(plasmon, "execution_mode", PLASMON_EXECUTION_MODE_QUICKLOOK))),
                (self.plasmon_observable_mode_combo, str(getattr(plasmon, "observable_mode", PLASMON_OBSERVABLE_MODE_DIELECTRIC))),
                (self.plasmon_electron_policy_combo, str(getattr(plasmon, "requested_electron_policy", getattr(plasmon, "electron_policy", PLASMON_ELECTRON_POLICY_RAW)))),
                (self.plasmon_coordinate_axis_combo, str(getattr(plasmon, "coordinate_axis", PLASMON_AXIS_ANGLE_DEG))),
                (self.plasmon_scan_axis_combo, str(getattr(plasmon, "scan_axis", PLASMON_AXIS_ANGLE_DEG))),
                (self.plasmon_integration_combo, str(plasmon.integration_mode or PLASMON_INTEGRATION_MODE_EFFECTIVE_STATE)),
                (self.plasmon_collision_model_combo, str(plasmon.collision_model or PLASMON_COLLISION_MODEL_NRL_CONSTANT)),
                (self.plasmon_lfc_combo, str(plasmon.lfc_model or PLASMON_LFC_MODEL_NONE)),
                (self.plasmon_normalization_combo, str(plasmon.normalization or PLASMON_NORMALIZATION_PEAK)),
            ):
                index = combo.findData(value)
                if index >= 0:
                    combo.setCurrentIndex(index)
            self.plasmon_energy_spin.setValue(float(self._display_photon_energy_from_kev(float(plasmon.photon_energy_kev), unit=self._photon_spin_unit(self.plasmon_energy_spin))))
            self._configure_plasmon_coordinate_spin(
                axis=str(getattr(plasmon, "coordinate_axis", PLASMON_AXIS_ANGLE_DEG)),
                canonical_value=float(getattr(plasmon, "coordinate_value", plasmon.scattering_angle_deg)),
            )
            self._configure_plasmon_scan_spins(
                axis=str(getattr(plasmon, "scan_axis", PLASMON_AXIS_ANGLE_DEG)),
                start=float(getattr(plasmon, "scan_start", 10.0)),
                stop=float(getattr(plasmon, "scan_stop", 140.0)),
            )
            self.plasmon_scan_points_spin.setValue(int(getattr(plasmon, "scan_points", self.plasmon_scan_points_spin.value())))
            self.plasmon_compare_models_checkbox.setChecked(bool(getattr(plasmon, "compare_models", False)))
            compare_model_values = tuple(str(value) for value in getattr(plasmon, "comparison_models", ()) or ())
            self._set_checked_plasmon_compare_models(None if not compare_model_values else compare_model_values)
            self.plasmon_compare_policies_checkbox.setChecked(bool(getattr(plasmon, "compare_policies", False)))
            self.plasmon_gamma_spin.setValue(float(plasmon.adiabatic_index))
            self.plasmon_collision_scale_spin.setValue(float(plasmon.collision_scale))
            self.plasmon_manual_collision_spin.setValue(float(plasmon.manual_collision_rate_s))
            self.plasmon_window_spin.setValue(float(plasmon.spectrum_window_ev))
            self.plasmon_points_spin.setValue(int(plasmon.spectrum_points))
            self.plasmon_fwhm_spin.setValue(float(plasmon.instrument_fwhm_ev))
            self.plasmon_cluster_log_ne_spin.setValue(float(getattr(plasmon, "cluster_log_ne_tol", self.plasmon_cluster_log_ne_spin.value())))
            self.plasmon_cluster_log_te_spin.setValue(float(getattr(plasmon, "cluster_log_te_tol", self.plasmon_cluster_log_te_spin.value())))
            self.plasmon_cluster_z_spin.setValue(float(getattr(plasmon, "cluster_z_tol", self.plasmon_cluster_z_spin.value())))
        finally:
            for widget, previous in reversed(blocked):
                widget.blockSignals(previous)
        self._sync_plasmon_placeholder_controls_enabled()

    def _plasmon_request_control_has_active_input(self) -> bool:
        for widget in getattr(self, "_plasmon_request_controls", ()):
            if isinstance(widget, QtWidgets.QAbstractSpinBox):
                if widget.hasFocus():
                    return True
                line_edit = widget.lineEdit()
                if line_edit is not None and line_edit.hasFocus():
                    return True
            elif isinstance(widget, _PopupAwareComboBox):
                if widget.hasFocus() or widget.is_popup_visible():
                    return True
            elif isinstance(widget, QtWidgets.QComboBox):
                if widget.hasFocus():
                    return True
                view = widget.view()
                if view is not None and (view.isVisible() or view.window().isVisible()):
                    return True
            elif isinstance(widget, QtWidgets.QAbstractItemView):
                if widget.hasFocus():
                    return True
                if widget.viewport() is not None and widget.viewport().hasFocus():
                    return True
            elif isinstance(widget, QtWidgets.QWidget) and widget.hasFocus():
                return True
        return False

    def _should_defer_plasmon_recompute(self, sender: object | None) -> bool:
        return (
            sender in getattr(self, "_plasmon_heavy_controls", ())
            and self._selected_plasmon_model() != PLASMON_MODEL_QUICKLOOK
            and not bool(self.plasmon_auto_recompute_checkbox.isChecked())
        )

    def _update_plasmon_recompute_ui(self) -> None:
        advanced_selected = self._selected_plasmon_model() != PLASMON_MODEL_QUICKLOOK
        pending = bool(self._plasmon_recompute_pending)
        activity_progress = getattr(self, "activity_progress", None)
        busy = bool(activity_progress is not None and activity_progress.isVisible())
        self.plasmon_recompute_button.setEnabled(advanced_selected and not busy)
        export_ready = plasmon_export_is_ready(None if self._current_result is None else self._current_result.plasmon)
        self.plasmon_export_button.setEnabled(export_ready and not busy)
        if not advanced_selected:
            self.plasmon_recompute_button.setText("Recompute plasmon")
            return
        if pending:
            self.plasmon_recompute_button.setText("Recompute plasmon (pending)")
        elif self.plasmon_auto_recompute_checkbox.isChecked():
            self.plasmon_recompute_button.setText("Recompute plasmon")
        else:
            self.plasmon_recompute_button.setText("Recompute plasmon")

    def _trigger_plasmon_recompute(self) -> None:
        self._plasmon_recompute_pending = False
        self._update_plasmon_recompute_ui()
        self.refresh_requested.emit()

    def _on_plasmon_auto_recompute_toggled(self, checked: bool) -> None:
        if bool(checked) and self._plasmon_recompute_pending and self._selected_plasmon_model() != PLASMON_MODEL_QUICKLOOK:
            self._trigger_plasmon_recompute()
            return
        self._update_plasmon_recompute_ui()

    def _suggest_plasmon_export_path(self) -> Path:
        if self._current_result is None:
            return Path.cwd() / "plasmon_spectrum.csv"
        plasmon = self._current_result.plasmon
        stem = Path(self._current_result.dataset_path).stem
        return Path.cwd() / f"{stem}_snapshot{int(plasmon.snapshot_index):04d}_{str(plasmon.model_name or 'plasmon')}_spectrum.csv"

    def _export_plasmon_spectrum_csv(self) -> None:
        if self._current_result is None or not plasmon_export_is_ready(self._current_result.plasmon):
            self.status_label.setText("Plasmon export: no aligned spectrum arrays are available yet.")
            self.result_status_label.setText("Plasmon export: no aligned spectrum arrays are available yet.")
            return
        start_path = self._suggest_plasmon_export_path()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export plasmon spectrum CSV",
            str(start_path),
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            destination = write_plasmon_spectrum_csv(path, self._current_result.plasmon)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Plasmon export failed", str(exc))
            return
        message = f"Plasmon spectrum exported to {destination}"
        self.status_label.setText(message)
        self.result_status_label.setText(message)

    def _plasmon_request_matches_result(self, plasmon: PlasmonResult | None = None) -> bool:
        if plasmon is None:
            if self._current_result is None:
                return False
            plasmon = self._current_result.plasmon
        if self._current_result is None:
            return False
        if int(self._current_result.snapshot_index) != int(self._context.snapshot_index):
            return False
        if int(plasmon.snapshot_index) != int(self._context.snapshot_index):
            return False
        if str(plasmon.requested_model_name or plasmon.model_name) != self._selected_plasmon_model():
            return False
        if str(getattr(plasmon, "benchmark_preset", PLASMON_BENCHMARK_PRESET_NONE)) != self._selected_plasmon_benchmark_preset():
            return False
        if str(getattr(plasmon, "study_mode", PLASMON_STUDY_MODE_SPECTRUM)) != self._selected_plasmon_study_mode():
            return False
        if str(getattr(plasmon, "execution_mode", PLASMON_EXECUTION_MODE_QUICKLOOK)) != self._selected_plasmon_execution_mode():
            return False
        if str(getattr(plasmon, "observable_mode", PLASMON_OBSERVABLE_MODE_DIELECTRIC)) != self._selected_plasmon_observable_mode():
            return False
        if str(getattr(plasmon, "requested_electron_policy", getattr(plasmon, "electron_policy", PLASMON_ELECTRON_POLICY_RAW))) != self._selected_plasmon_electron_policy():
            return False
        if str(getattr(plasmon, "coordinate_axis", PLASMON_AXIS_ANGLE_DEG)) != self._selected_plasmon_coordinate_axis():
            return False
        if str(getattr(plasmon, "scan_axis", PLASMON_AXIS_ANGLE_DEG)) != self._selected_plasmon_scan_axis():
            return False
        checks = (
            (float(plasmon.photon_energy_kev), float(self._photon_display_to_kev(float(self.plasmon_energy_spin.value()), unit=self._photon_spin_unit(self.plasmon_energy_spin)))),
            (float(getattr(plasmon, "coordinate_value", plasmon.scattering_angle_deg)), float(self._selected_plasmon_coordinate_value())),
            (float(getattr(plasmon, "scan_start", 10.0)), float(self._selected_plasmon_scan_start())),
            (float(getattr(plasmon, "scan_stop", 140.0)), float(self._selected_plasmon_scan_stop())),
            (float(plasmon.adiabatic_index), float(self.plasmon_gamma_spin.value())),
            (float(plasmon.spectrum_window_ev), float(self.plasmon_window_spin.value())),
            (float(plasmon.instrument_fwhm_ev), float(self.plasmon_fwhm_spin.value())),
            (float(plasmon.collision_scale), float(self.plasmon_collision_scale_spin.value())),
            (float(plasmon.manual_collision_rate_s), float(self.plasmon_manual_collision_spin.value())),
        )
        for current_value, requested_value in checks:
            tolerance = max(1.0e-9, abs(current_value) * 1.0e-6)
            if abs(current_value - requested_value) > tolerance:
                return False
        requested_points = int(self.plasmon_points_spin.value())
        if self._selected_plasmon_execution_mode() == PLASMON_EXECUTION_MODE_BENCHMARK:
            if int(plasmon.spectrum_points) < requested_points:
                return False
        elif int(plasmon.spectrum_points) != requested_points:
            return False
        if int(getattr(plasmon, "scan_points", self.plasmon_scan_points_spin.value())) != int(self.plasmon_scan_points_spin.value()):
            return False
        if bool(getattr(plasmon, "compare_models", False)) != bool(self.plasmon_compare_models_checkbox.isChecked()):
            return False
        if bool(self.plasmon_compare_models_checkbox.isChecked()):
            result_models = tuple(str(value) for value in getattr(plasmon, "comparison_models", ()) or ())
            requested_models = self._checked_plasmon_compare_models()
            if result_models != requested_models:
                return False
        if bool(getattr(plasmon, "compare_policies", False)) != bool(self.plasmon_compare_policies_checkbox.isChecked()):
            return False
        if str(plasmon.integration_mode) != str(self.plasmon_integration_combo.currentData() or PLASMON_INTEGRATION_MODE_EFFECTIVE_STATE):
            return False
        if str(plasmon.collision_model) != str(self.plasmon_collision_model_combo.currentData() or PLASMON_COLLISION_MODEL_NRL_CONSTANT):
            return False
        if str(plasmon.lfc_model) != str(self.plasmon_lfc_combo.currentData() or PLASMON_LFC_MODEL_NONE):
            return False
        if str(plasmon.normalization) != str(self.plasmon_normalization_combo.currentData() or PLASMON_NORMALIZATION_PEAK):
            return False
        return True

    def _sync_plasmon_placeholder_controls_enabled(self, *args) -> None:
        del args
        capabilities = self._current_plasmon_ui_capabilities()
        advanced_enabled = bool(capabilities.advanced_model_requested)
        dispersion_mode = self._selected_plasmon_study_mode() == PLASMON_STUDY_MODE_DISPERSION
        manual_collision = str(self.plasmon_collision_model_combo.currentData() or PLASMON_COLLISION_MODEL_NRL_CONSTANT) == "manual_constant"
        for widget in getattr(self, "_plasmon_advanced_widgets", ()):
            widget.setEnabled(advanced_enabled)
        self.plasmon_scan_axis_combo.setEnabled(True)
        self.plasmon_scan_start_spin.setEnabled(dispersion_mode)
        self.plasmon_scan_stop_spin.setEnabled(dispersion_mode)
        self.plasmon_scan_points_spin.setEnabled(dispersion_mode)
        self.plasmon_coordinate_axis_combo.setEnabled(True)
        self.plasmon_angle_spin.setEnabled(True)
        self.plasmon_compare_models_checkbox.setEnabled(True)
        self.plasmon_compare_model_list.setEnabled(
            bool(capabilities.compare_models_available) and bool(self.plasmon_compare_models_checkbox.isChecked())
        )
        self.plasmon_compare_models_checkbox.setToolTip(
            "Overlay all spectral models on the current dispersion or spectrum plot for direct comparison."
            if capabilities.compare_models_available
            else capabilities.compare_models_reason
        )
        self.plasmon_compare_model_list.setToolTip(
            "Choose which benchmark-capable spectral models participate in the direct model-comparison overlays."
            if capabilities.compare_models_available
            else capabilities.compare_models_reason
        )
        self.plasmon_compare_policies_checkbox.setEnabled(bool(capabilities.compare_policies_available))
        self.plasmon_compare_policies_checkbox.setToolTip(
            "Benchmark-only overlay of the current spectral model under the article Al benchmark, article Al driven increment, benchmark valence-aware, and valence-locked electron policies."
            if capabilities.compare_policies_available
            else capabilities.compare_policies_reason
        )
        if advanced_enabled:
            self.plasmon_manual_collision_spin.setEnabled(manual_collision)
        self._sync_plasmon_plot_capabilities()
        self._update_plasmon_recompute_ui()

    def _on_transmission_controls_changed(self, *args) -> None:
        del args
        mode = self.selected_transmission_mode()
        energy_enabled = self._transmission_mode_supports_energy(mode)
        self.transmission_energy_spin.setEnabled(energy_enabled)
        self.transmission_energy_unit_combo.setEnabled(energy_enabled)
        if self._current_result is None:
            return
        if self._transmission_request_matches_result():
            self.transmission_refine_button.setText("Apply Transmission Model")
            return
        self.transmission_refine_button.setText("Apply Selected Transmission")
        if self.transmission_requested():
            requested = _transmission_mode_label(mode)
            self.transmission_refinement_label.setText(f"Status: Pending apply for {requested}.")

    def wavefront_requested(self) -> bool:
        return int(self.result_tabs.currentIndex()) == int(self.result_tabs.indexOf(self.wavefront_tab))

    def preheat_requested(self) -> bool:
        return int(self.result_tabs.currentIndex()) == int(self.result_tabs.indexOf(self.preheat_tab))

    def advanced_requested(self) -> bool:
        return self.wavefront_requested() or self.preheat_requested()

    def active_advanced_request_kind(self) -> str | None:
        if self.preheat_requested():
            return "preheat"
        if self.wavefront_requested():
            return "wavefront"
        return None

    def _effective_profile_coordinate_mode(self) -> str:
        mode = str(self.profile_coordinate_combo.currentData() or "zone")
        if mode == "viewer":
            viewer_mode = str(self._context.slice_coordinate or self._context.map_coordinate or "zone")
            return viewer_mode if viewer_mode in {"moving_radius", "static_x", "zone"} else "zone"
        return mode if mode in {"moving_radius", "static_x", "zone"} else "zone"

    def _preheat_time_axis_s(self, preheat: PreheatSummary) -> np.ndarray:
        if preheat.time_plots:
            return np.asarray(preheat.time_plots[0].x_values, dtype=np.float64)
        if self._context.time_values.size:
            return np.asarray(self._context.time_values, dtype=np.float64)
        return np.asarray([], dtype=np.float64)

    def _reset_preheat_navigation_state(self) -> None:
        self._preheat_time_mode = "shock_relative"
        self._preheat_offset_steps = 0
        self._preheat_manual_snapshot_index = None
        self._preheat_display_snapshot_index = None
        self._preheat_syncing_controls = True
        try:
            mode_index = self.preheat_time_mode_combo.findData("shock_relative")
            if mode_index >= 0:
                self.preheat_time_mode_combo.setCurrentIndex(mode_index)
            offset_index = self.preheat_offset_combo.findData(0)
            if offset_index >= 0:
                self.preheat_offset_combo.setCurrentIndex(offset_index)
        finally:
            self._preheat_syncing_controls = False

    def _resolve_preheat_snapshot_index(self, preheat: PreheatSummary) -> tuple[int | None, str]:
        time_axis_s = self._preheat_time_axis_s(preheat)
        if time_axis_s.size == 0:
            return None, "No preheat time axis is available for the current result."
        if self._preheat_time_mode == "manual":
            clamped = int(np.clip(int(self._context.snapshot_index), 0, time_axis_s.size - 1))
            note = "Manual preheat view follows the global snapshot/time selector."
            if preheat.target_entry_time_s is not None and math.isfinite(float(preheat.target_entry_time_s)):
                if float(time_axis_s[clamped]) >= float(preheat.target_entry_time_s):
                    note += " The selected snapshot is at or after target shock entry, so the profile shows the actual target state rather than pre-entry-only conditions."
            return clamped, note
        anchor = preheat.latest_pre_entry_snapshot_index
        if anchor is None:
            anchor = int(np.clip(self._context.snapshot_index, 0, time_axis_s.size - 1))
            return anchor, "No stable pre-entry anchor snapshot was available, so the tab fell back to the current snapshot."
        requested = int(anchor) - int(self._preheat_offset_steps)
        clamped = int(np.clip(requested, 0, time_axis_s.size - 1))
        if clamped != requested:
            return (
                clamped,
                f"Shock-relative mode requested {int(self._preheat_offset_steps)} snapshots before target entry, but only {int(anchor)} earlier snapshots exist. Showing the earliest available snapshot instead.",
            )
        return (
            clamped,
            "Shock-relative mode is anchored to the latest snapshot before the primary compressive branch enters the selected region of interest.",
        )

    def _sync_preheat_navigation_controls(self, preheat: PreheatSummary | None, *, selected_snapshot_index: int | None = None) -> None:
        self._preheat_syncing_controls = True
        try:
            time_unit_suffix = f" {self._time_unit()}"
            self.preheat_time_spin.setSuffix(time_unit_suffix)
            mode_index = self.preheat_time_mode_combo.findData(self._preheat_time_mode)
            if mode_index >= 0:
                self.preheat_time_mode_combo.setCurrentIndex(mode_index)
            offset_index = self.preheat_offset_combo.findData(int(self._preheat_offset_steps))
            if offset_index >= 0:
                self.preheat_offset_combo.setCurrentIndex(offset_index)
            if preheat is None:
                self.preheat_offset_combo.setEnabled(False)
                self.preheat_snapshot_slider.setEnabled(False)
                self.preheat_snapshot_spin.setEnabled(False)
                self.preheat_time_spin.setEnabled(False)
                self.preheat_snapshot_slider.setRange(0, 0)
                self.preheat_snapshot_spin.setRange(0, 0)
                self.preheat_snapshot_slider.setValue(0)
                self.preheat_snapshot_spin.setValue(0)
                self.preheat_time_spin.setValue(0.0)
                return
            time_axis_s = self._preheat_time_axis_s(preheat)
            n_snapshots = max(1, int(time_axis_s.size))
            selected = 0 if selected_snapshot_index is None else int(np.clip(selected_snapshot_index, 0, n_snapshots - 1))
            self.preheat_snapshot_slider.setRange(0, n_snapshots - 1)
            self.preheat_snapshot_spin.setRange(0, n_snapshots - 1)
            self.preheat_snapshot_slider.setValue(selected)
            self.preheat_snapshot_spin.setValue(selected)
            if time_axis_s.size:
                display_time = float(self._display_time_from_seconds(float(time_axis_s[selected])))
                low_time = float(self._display_time_from_seconds(float(np.nanmin(time_axis_s))))
                high_time = float(self._display_time_from_seconds(float(np.nanmax(time_axis_s))))
                self.preheat_time_spin.setRange(min(low_time, high_time), max(low_time, high_time))
                self.preheat_time_spin.setValue(display_time)
            else:
                self.preheat_time_spin.setValue(0.0)
            manual_mode = self._preheat_time_mode == "manual"
            self.preheat_offset_combo.setEnabled(not manual_mode and preheat.latest_pre_entry_snapshot_index is not None)
            self.preheat_snapshot_slider.setEnabled(False)
            self.preheat_snapshot_spin.setEnabled(False)
            self.preheat_time_spin.setEnabled(False)
            self.preheat_offset_combo.setToolTip(
                "Choose how many snapshots earlier than target shock entry to inspect."
                if self.preheat_offset_combo.isEnabled()
                else "Shock-relative offset is only editable while 'Before shock entry' mode is active."
            )
            manual_tooltip = (
                "Preheat manual mode follows the global snapshot/time selector."
                if manual_mode
                else "Manual snapshot/time display activates when Manual mode is selected; use the global snapshot selector."
            )
            self.preheat_snapshot_slider.setToolTip(manual_tooltip)
            self.preheat_snapshot_spin.setToolTip(manual_tooltip)
            self.preheat_time_spin.setToolTip(manual_tooltip)
        finally:
            self._preheat_syncing_controls = False

    def _preheat_coordinate_axis(
        self,
        preheat: PreheatSummary,
        *,
        snapshot_index: int,
    ) -> tuple[np.ndarray, str, str | None]:
        mode = self._effective_profile_coordinate_mode()
        if mode == "moving_radius" and preheat.target_dynamic_coordinate_cm is not None:
            dynamic = np.asarray(preheat.target_dynamic_coordinate_cm, dtype=np.float64)
            if dynamic.ndim == 2 and 0 <= snapshot_index < dynamic.shape[0]:
                return (
                    np.asarray(self._display_length_from_cm(dynamic[snapshot_index]), dtype=np.float64),
                    f"Moving radius [{self._length_unit()}]",
                    None,
                )
        if mode == "static_x" and preheat.target_static_x_cm is not None:
            return (
                np.asarray(self._display_length_from_cm(np.asarray(preheat.target_static_x_cm, dtype=np.float64)), dtype=np.float64),
                f"Static x [{self._length_unit()}]",
                None,
            )
        if mode == "moving_radius":
            return (
                np.asarray(preheat.target_zone_indices, dtype=np.float64),
                "Zone index",
                "Dynamic target coordinates are unavailable for this run, so the profile falls back to zone index.",
            )
        if preheat.target_static_x_cm is None and mode == "static_x":
            return (
                np.asarray(preheat.target_zone_indices, dtype=np.float64),
                "Zone index",
                "Static-x target coordinates are unavailable for this run, so the profile falls back to zone index.",
            )
        return (np.asarray(preheat.target_zone_indices, dtype=np.float64), "Zone index", None)

    def _convert_preheat_profile_values(self, values: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
        normalized = str(unit or "").strip()
        series = np.asarray(values, dtype=np.float64)
        if normalized == "eV":
            return np.asarray(self._display_temperature_from_ev(series), dtype=np.float64), self._temperature_unit()
        if normalized == "J/cm^3":
            return np.asarray(self._display_pressure_from_j_cm3(series), dtype=np.float64), self._pressure_unit()
        if normalized == "J/g":
            return np.asarray(self._display_specific_energy_from_j_g(series), dtype=np.float64), self._specific_energy_unit()
        if normalized == "J/g/s":
            return np.asarray(self._display_rate_from_j_g_s(series), dtype=np.float64), self._rate_unit()
        if normalized == "cm":
            return np.asarray(self._display_length_from_cm(series), dtype=np.float64), self._length_unit()
        if normalized == "g/cm3":
            return np.asarray(self._display_density_from_g_cm3(series), dtype=np.float64), self._density_unit()
        if normalized == "fraction":
            return series, "0/1"
        return series, normalized

    def _preheat_profile_bundles(self, preheat: PreheatSummary, *, snapshot_index: int) -> tuple[DerivedPlotBundle, ...]:
        if not preheat.profile_fields:
            return ()
        x_values, x_label, fallback_note = self._preheat_coordinate_axis(preheat, snapshot_index=snapshot_index)
        bundles: list[DerivedPlotBundle] = []
        for field in preheat.profile_fields:
            field_values = np.asarray(field.values, dtype=np.float64)
            if field_values.ndim != 2 or not (0 <= snapshot_index < field_values.shape[0]):
                continue
            y_values, y_unit = self._convert_preheat_profile_values(field_values[snapshot_index], field.unit)
            y_label = str(field.label) if not y_unit else f"{field.label} [{y_unit}]"
            title = f"{field.label} @ snapshot {snapshot_index}"
            notes = tuple(field.notes)
            if fallback_note is not None:
                notes = (*notes, fallback_note)
            bundles.append(
                DerivedPlotBundle(
                    key=str(field.key),
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                    x_values=np.asarray(x_values, dtype=np.float64),
                    y_series=(np.asarray(y_values, dtype=np.float64),),
                    curve_names=(str(field.label),),
                    value_scale_mode="linear",
                )
            )
        return tuple(bundles)

    def _populate_preheat_snapshot_table(
        self,
        preheat: PreheatSummary,
        *,
        snapshot_index: int,
        status_note: str,
    ) -> None:
        time_axis_s = self._preheat_time_axis_s(preheat)
        snapshot_time_s = None if not (0 <= snapshot_index < time_axis_s.size) else float(time_axis_s[snapshot_index])
        rows: list[tuple[str, str, str]] = [
            ("Displayed snapshot", str(snapshot_index), status_note),
            ("Displayed time", self._format_time(snapshot_time_s), f"entry={self._format_time(preheat.target_entry_time_s)}"),
        ]
        scalar_series = preheat.snapshot_scalar_series

        def _series_value(key: str) -> float | None:
            series = scalar_series.get(key)
            if series is None:
                return None
            array = np.asarray(series, dtype=np.float64)
            if array.ndim != 1 or not (0 <= snapshot_index < array.size):
                return None
            value = float(array[snapshot_index])
            return value if math.isfinite(value) else None

        rows.extend(
            [
                ("Affected depth", self._format_length(_series_value("affected_depth_cm")), "selected snapshot"),
                ("Thickness fraction", self._format_fraction(_series_value("affected_thickness_fraction")), "selected snapshot"),
                ("Areal-mass fraction", self._format_fraction(_series_value("affected_areal_mass_fraction")), "selected snapshot"),
                ("Delta Te mean", self._format_temperature(_series_value("delta_temperature_e_mean")), "selected snapshot"),
                ("Delta Te peak", self._format_temperature(_series_value("delta_temperature_e_peak")), "selected snapshot"),
                ("Delta Zbar peak", _format_optional(_series_value("delta_mean_charge_peak"), "{:.4g}"), "selected snapshot"),
                ("Pressure peak", self._format_pressure(_series_value("pressure_total_peak_j_cm3")), "selected snapshot if available"),
                ("Radiation peak", self._format_rate(_series_value("radiation_peak_j_g_s")), "selected snapshot if available"),
                ("Laser peak", self._format_rate(_series_value("laser_peak_j_g_s")), "selected snapshot if available"),
            ]
        )
        self.preheat_snapshot_table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for column, text in enumerate(row_values):
                self.preheat_snapshot_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(text))
        self.preheat_snapshot_table.resizeColumnsToContents()

    def _refresh_preheat_snapshot_view(self, preheat: PreheatSummary | None) -> None:
        if preheat is None:
            self.preheat_plot_panel.clear()
            self.preheat_snapshot_label.setText("No selected snapshot")
            self.preheat_snapshot_table.setRowCount(0)
            self.preheat_profile_status_label.setText(
                "Snapshot profiles will appear here once a valid preheat region and time are available."
            )
            self.preheat_time_status_label.setText(
                "Select a region of interest and open this tab to inspect preheat before shock entry or at a manual snapshot."
            )
            self._sync_preheat_navigation_controls(None)
            return
        selected_snapshot_index, status_note = self._resolve_preheat_snapshot_index(preheat)
        self._preheat_display_snapshot_index = selected_snapshot_index
        self._sync_preheat_navigation_controls(preheat, selected_snapshot_index=selected_snapshot_index)
        time_bundles: list[DerivedPlotBundle] = []
        for bundle in preheat.time_plots:
            converted = self._bundle_time_x(bundle)
            if bundle.key == "preheat_temperature":
                converted = self._bundle_temperature_y(converted)
            time_bundles.append(converted)
        profile_bundles = () if selected_snapshot_index is None else self._preheat_profile_bundles(preheat, snapshot_index=selected_snapshot_index)
        selected_time_s = None
        time_axis_s = self._preheat_time_axis_s(preheat)
        if selected_snapshot_index is not None and 0 <= selected_snapshot_index < time_axis_s.size:
            selected_time_s = float(time_axis_s[selected_snapshot_index])
        self.preheat_plot_panel.set_bundles(
            tuple(time_bundles),
            profile_bundles,
            view_scope="preheat",
            preferred_time_key=("preheat_temperature" if any(bundle.key == "preheat_temperature" for bundle in time_bundles) else None),
        )
        self.preheat_plot_panel.set_snapshot_marker(
            None if selected_time_s is None else float(self._display_time_from_seconds(selected_time_s))
        )
        if selected_snapshot_index is None:
            self.preheat_snapshot_label.setText("No valid selected snapshot is available for the current preheat state.")
            self.preheat_snapshot_table.setRowCount(0)
            self.preheat_profile_status_label.setText(
                "No selected snapshot profile could be generated for the current target/time configuration."
            )
        else:
            relation = "pre-entry"
            if preheat.target_entry_time_s is not None and selected_time_s is not None and selected_time_s >= float(preheat.target_entry_time_s):
                relation = "at/after shock entry"
            self.preheat_snapshot_label.setText(
                f"Snapshot {selected_snapshot_index} @ {self._format_time(selected_time_s)} | {relation} | mode={'manual' if self._preheat_time_mode == 'manual' else 'shock-relative'}"
            )
            self._populate_preheat_snapshot_table(preheat, snapshot_index=selected_snapshot_index, status_note=status_note)
            if profile_bundles:
                self.preheat_profile_status_label.setText(
                    "Use the 'Snapshot profiles' selector to inspect target-region fields at the displayed Preheat snapshot."
                )
                self.preheat_plot_panel.profile_combo.setToolTip("Choose a snapshot profile field for the displayed Preheat snapshot.")
            else:
                self.preheat_profile_status_label.setText(
                    "No snapshot profile fields are available for the current target/time configuration."
                )
                self.preheat_plot_panel.profile_combo.setToolTip(
                    "No snapshot profile fields are available for the current target/time configuration."
                )
        self.preheat_time_status_label.setText(status_note)

    @QtCore.Slot()
    def _on_preheat_time_mode_changed(self) -> None:
        if self._preheat_syncing_controls:
            return
        self._preheat_time_mode = str(self.preheat_time_mode_combo.currentData() or "shock_relative")
        if self._current_result is not None:
            self._refresh_preheat_snapshot_view(self._current_result.preheat)

    @QtCore.Slot()
    def _on_preheat_time_controls_changed(self) -> None:
        if self._preheat_syncing_controls:
            return
        self._preheat_offset_steps = int(self.preheat_offset_combo.currentData() or 0)
        if self._current_result is not None:
            self._refresh_preheat_snapshot_view(self._current_result.preheat)

    @QtCore.Slot(int)
    def _on_preheat_snapshot_slider_changed(self, value: int) -> None:
        if self._preheat_syncing_controls or self._preheat_time_mode != "manual":
            return
        self._preheat_manual_snapshot_index = int(value)
        if self._current_result is not None:
            self._refresh_preheat_snapshot_view(self._current_result.preheat)

    @QtCore.Slot(int)
    def _on_preheat_snapshot_spin_changed(self, value: int) -> None:
        if self._preheat_syncing_controls or self._preheat_time_mode != "manual":
            return
        self._preheat_manual_snapshot_index = int(value)
        if self._current_result is not None:
            self._refresh_preheat_snapshot_view(self._current_result.preheat)

    @QtCore.Slot(float)
    def _on_preheat_time_spin_changed(self, value: float) -> None:
        if self._preheat_syncing_controls or self._preheat_time_mode != "manual":
            return
        preheat = None if self._current_result is None else self._current_result.preheat
        if preheat is None:
            return
        time_axis_s = self._preheat_time_axis_s(preheat)
        if time_axis_s.size == 0:
            return
        display_times = np.asarray(self._display_time_from_seconds(time_axis_s), dtype=np.float64)
        finite_mask = np.isfinite(display_times)
        if not np.any(finite_mask):
            return
        index_pool = np.flatnonzero(finite_mask)
        nearest = int(index_pool[int(np.argmin(np.abs(display_times[finite_mask] - float(value))))])
        self._preheat_manual_snapshot_index = nearest
        self._refresh_preheat_snapshot_view(preheat)

    def _wavefront_display_mode(self) -> str:
        return str(self.wavefront_display_combo.currentData() or "primary_position")

    def _wavefront_scope_mode(self) -> str:
        return str(self.wavefront_scope_combo.currentData() or "tracked")

    def _wavefront_direction_mode(self) -> str:
        return str(self.wavefront_direction_combo.currentData() or "all")

    def _set_wavefront_lower_layout_mode(self, display_mode: str) -> None:
        if not hasattr(self, "wavefront_lower_splitter"):
            return
        if display_mode == "events":
            self.wavefront_lower_splitter.setSizes([520, 920])
        elif display_mode in {"warnings", "significance"}:
            self.wavefront_lower_splitter.setSizes([860, 580])
        else:
            self.wavefront_lower_splitter.setSizes([700, 700])

    @staticmethod
    def _wavefront_branch_type_priority(branch: WaveBranchSummary) -> tuple[int, float]:
        significance = 0.0 if branch.significance is None or not math.isfinite(float(branch.significance)) else float(branch.significance)
        branch_type = str(branch.branch_type)
        if branch_type == "compressive_shock":
            return (0, -significance)
        if branch_type == "transmitted_shock":
            return (1, -significance)
        if branch_type == "reflected_shock":
            return (2, -significance)
        if branch_type in {"release_rarefaction", "rear_rarefaction"}:
            return (3, -significance)
        if branch_type == "contact_transition":
            return (4, -significance)
        return (5, -significance)

    def _primary_compressive_branch(
        self,
        branches: tuple[WaveBranchSummary, ...],
    ) -> WaveBranchSummary | None:
        compressive = [
            branch
            for branch in branches
            if str(branch.branch_type) in {"compressive_shock", "transmitted_shock", "reflected_shock"}
            and str(branch.support_class) != "provisional"
        ]
        if not compressive:
            return None
        if self._current_result is not None and self._current_result.wave_tracking is not None:
            primary_branch_id = str(self._current_result.wave_tracking.primary_branch_id or "")
            if primary_branch_id:
                primary_branch = next((branch for branch in compressive if str(branch.branch_id) == primary_branch_id), None)
                if primary_branch is not None:
                    return primary_branch
            primary_branch = next((branch for branch in compressive if bool(branch.primary)), None)
            if primary_branch is not None:
                return primary_branch
        preferred_direction = None
        if self._current_result is not None and self._current_result.shock is not None:
            preferred_direction = str(self._current_result.shock.propagation_direction or "")
        return min(
            compressive,
            key=lambda branch: (
                0 if str(branch.support_class) == "tracked" else 1,
                *self._wavefront_branch_type_priority(branch),
                0 if preferred_direction and str(branch.propagation_direction or "") == preferred_direction else 1,
                0 if not bool(branch.ambiguous) else 1,
                str(branch.branch_id),
            ),
        )

    def _wavefront_visible_branches(
        self,
        branches: tuple[WaveBranchSummary, ...],
    ) -> tuple[list[WaveBranchSummary], list[WaveBranchSummary], str | None]:
        tracked = [branch for branch in branches if str(branch.support_class) == "tracked"]
        short_weak = [branch for branch in branches if str(branch.support_class) == "short_weak"]
        provisional = [branch for branch in branches if str(branch.support_class) == "provisional"]
        primary_compressive = self._primary_compressive_branch(branches)
        scope = self._wavefront_scope_mode()
        note: str | None = None
        if scope == "primary":
            if primary_compressive is None:
                return [], [], "No reliable non-provisional compressive branch was available for primary-branch inspection."
            plotted = [primary_compressive]
            table = [primary_compressive]
            if str(primary_compressive.support_class) != "tracked":
                note = "The best compressive branch is present, but it is only short / weak rather than fully tracked."
        elif scope == "top_significant":
            base = tracked + short_weak
            if not base:
                base = provisional
            table = list(base[:6])
            plotted = [branch for branch in table if str(branch.support_class) != "provisional"]
            if provisional and not (tracked or short_weak):
                note = "Only provisional detections were available, so the top-significant summary is informational rather than fully tracked."
        elif scope == "compressive":
            table = [branch for branch in tracked + short_weak + provisional if str(branch.branch_type) in {"compressive_shock", "transmitted_shock", "reflected_shock"}]
            plotted = [branch for branch in table if str(branch.support_class) != "provisional"]
        elif scope == "release":
            table = [branch for branch in tracked + short_weak + provisional if str(branch.branch_type) in {"release_rarefaction", "rear_rarefaction"}]
            plotted = [branch for branch in table if str(branch.support_class) != "provisional"]
        elif scope == "reflected":
            table = [branch for branch in tracked + short_weak + provisional if str(branch.branch_type) in {"reflected_shock", "rear_rarefaction"}]
            plotted = [branch for branch in table if str(branch.support_class) != "provisional"]
        elif scope == "transmitted":
            table = [branch for branch in tracked + short_weak + provisional if str(branch.branch_type) == "transmitted_shock"]
            plotted = [branch for branch in table if str(branch.support_class) != "provisional"]
        elif scope == "all":
            plotted = tracked + short_weak
            table = tracked + short_weak + provisional
            note = (
                "Provisional detections are listed below for inspection, but they are not drawn as normal branch trajectories because they have fewer than 3 samples."
                if provisional
                else None
            )
        elif scope == "tracked_weak":
            plotted = tracked + short_weak
            table = tracked + short_weak
        elif tracked:
            plotted = tracked
            table = tracked
        elif short_weak:
            plotted = short_weak
            table = short_weak
            note = "No fully tracked branches met the default support threshold; showing short / weak branches instead."
        else:
            plotted = []
            table = provisional
            note = (
                "No tracked branches were available. Only provisional detections were found, so the default WaveFront plots stay hidden."
                if provisional
                else None
            )

        direction_mode = self._wavefront_direction_mode()
        if direction_mode != "all":
            plotted = [branch for branch in plotted if str(branch.propagation_direction or "") == direction_mode]
            table = [branch for branch in table if str(branch.propagation_direction or "") == direction_mode]
            if not table:
                direction_text = "low to high" if direction_mode == "low_to_high" else "high to low"
                detail = f"No visible branches matched the {direction_text} direction filter."
                note = detail if note is None else f"{note} {detail}"
        return plotted, table, note

    def _wavefront_visible_events(
        self,
        events: tuple[object, ...],
        *,
        branch_lookup: dict[str, WaveBranchSummary] | None = None,
        visible_branch_ids: set[str] | None = None,
    ) -> tuple[list[object], str | None]:
        tracked = [event for event in events if str(getattr(event, "support_class", "")) == "tracked"]
        short_weak = [event for event in events if str(getattr(event, "support_class", "")) == "short_weak"]
        provisional = [event for event in events if str(getattr(event, "support_class", "")) == "provisional"]
        scope = self._wavefront_scope_mode()
        note: str | None = None
        if scope == "all":
            visible = tracked + short_weak + provisional
            note = (
                "Provisional interface detections are only shown in summary scope because they do not meet the tracked-branch support threshold."
                if provisional
                else None
            )
        elif scope == "tracked_weak":
            visible = tracked + short_weak
        elif tracked:
            visible = tracked
        elif short_weak:
            visible = short_weak
            note = "No tracked interface events met the default support threshold; showing short / weak events instead."
        else:
            visible = provisional
            note = (
                "Only provisional interface detections were available, so the default event view stays summary-oriented."
                if provisional
                else None
            )
        if visible_branch_ids is not None:
            scoped = [event for event in visible if str(getattr(event, "branch_id", "")) in visible_branch_ids]
            if scoped or not visible:
                visible = scoped
            else:
                detail = "The current branch-set filter excludes the incident branches for the available interface events."
                note = detail if note is None else f"{note} {detail}"
        direction_mode = self._wavefront_direction_mode()
        if direction_mode != "all" and branch_lookup is not None:
            def _event_direction(event: object) -> str:
                branch = branch_lookup.get(str(getattr(event, "branch_id", "")))
                return str("" if branch is None else branch.propagation_direction or "")

            visible = [
                event
                for event in visible
                if _event_direction(event) == direction_mode
            ]
            if not visible:
                direction_text = "low to high" if direction_mode == "low_to_high" else "high to low"
                detail = f"No interface events matched the {direction_text} direction filter."
                note = detail if note is None else f"{note} {detail}"
        return visible, note

    def _refresh_wavefront_view(self) -> None:
        if self._current_result is None:
            return
        self._populate_wavefront(self._current_result.wave_tracking, self._current_result.interface_events, self._current_result.preheat)

    def _module_time_plots_loaded(self, module_name: str) -> bool:
        result = self._current_result
        if result is None:
            return False
        module_result = getattr(result, module_name, None)
        return bool(getattr(module_result, "time_plots", ()))

    def _wavefront_loaded(self) -> bool:
        return bool(
            self._current_result is not None
            and self._current_result.wave_tracking is not None
            and self._current_result.interface_events is not None
        )

    def _preheat_loaded(self) -> bool:
        return bool(
            self._current_result is not None
            and self._current_result.wave_tracking is not None
            and self._current_result.interface_events is not None
            and self._current_result.preheat is not None
        )

    def _handle_result_tab_changed(self, _index: int) -> None:
        self.time_plot_modules_changed.emit()
        if self._current_result is None:
            return
        if self.wavefront_requested():
            if not self._wavefront_loaded():
                self.refresh_requested.emit()
            return
        if self.preheat_requested():
            if not self._preheat_loaded():
                self.refresh_requested.emit()
            return
        if self.transmission_requested():
            if not self._transmission_request_matches_result():
                self.refresh_requested.emit()
                return
        module_name = self._active_time_plot_module()
        if module_name is None:
            return
        if not self._module_time_plots_loaded(module_name):
            self.refresh_requested.emit()

    def set_default_profile_coordinate_mode(self, mode: str) -> None:
        normalized = str(mode or "zone").strip().lower()
        if normalized not in {"zone", "moving_radius", "static_x", "viewer_follow"}:
            normalized = "zone"
        previous_requested = self._default_profile_coordinate_mode
        previous_effective = str(self.profile_coordinate_combo.currentData())
        self._default_profile_coordinate_mode = normalized
        if self._context.has_run:
            self._apply_default_profile_coordinate()
            effective_changed = str(self.profile_coordinate_combo.currentData()) != previous_effective
            if normalized != previous_requested and effective_changed:
                self.refresh_requested.emit()

    def _apply_default_profile_coordinate(self) -> None:
        requested = self._default_profile_coordinate_mode
        target = "viewer" if requested == "viewer_follow" else requested
        index = self.profile_coordinate_combo.findData(target)
        if index < 0:
            index = self.profile_coordinate_combo.findData("zone")
        self.profile_coordinate_combo.blockSignals(True)
        try:
            if index >= 0:
                self.profile_coordinate_combo.setCurrentIndex(index)
        finally:
            self.profile_coordinate_combo.blockSignals(False)

    def _checked_values_for_filter(self, widget: QtWidgets.QListWidget) -> tuple[int, ...] | None:
        checked = self._checked_values(widget)
        if widget.count() == 0:
            return None
        if len(checked) == widget.count():
            return None
        return checked

    @staticmethod
    def _coerce_display_settings(settings: object) -> ViewerSettings:
        defaults = default_viewer_settings()
        return ViewerSettings(
            theme_mode=str(getattr(settings, "theme_mode", defaults.theme_mode)),
            colormap=str(getattr(settings, "colormap", defaults.colormap)),
            map_scale_mode=str(getattr(settings, "map_scale_mode", defaults.map_scale_mode)),
            line_scale_mode=str(getattr(settings, "line_scale_mode", defaults.line_scale_mode)),
            diagnostic_scale_mode=str(getattr(settings, "diagnostic_scale_mode", defaults.diagnostic_scale_mode)),
            clip_mode=str(getattr(settings, "clip_mode", defaults.clip_mode)),
            show_boundaries=bool(getattr(settings, "show_boundaries", defaults.show_boundaries)),
            hover_interval_ms=int(getattr(settings, "hover_interval_ms", defaults.hover_interval_ms)),
            time_unit=str(getattr(settings, "time_unit", defaults.time_unit)),
            length_unit=str(getattr(settings, "length_unit", defaults.length_unit)),
            pressure_unit=str(getattr(settings, "pressure_unit", defaults.pressure_unit)),
            density_unit=str(getattr(settings, "density_unit", defaults.density_unit)),
            temperature_unit=str(getattr(settings, "temperature_unit", defaults.temperature_unit)),
            velocity_unit=str(getattr(settings, "velocity_unit", defaults.velocity_unit)),
            specific_energy_unit=str(getattr(settings, "specific_energy_unit", defaults.specific_energy_unit)),
            rate_unit=str(getattr(settings, "rate_unit", defaults.rate_unit)),
            heat_capacity_unit=str(getattr(settings, "heat_capacity_unit", defaults.heat_capacity_unit)),
            number_density_unit=str(getattr(settings, "number_density_unit", defaults.number_density_unit)),
            angle_unit=str(getattr(settings, "angle_unit", defaults.angle_unit)),
            photon_unit=str(getattr(settings, "photon_unit", defaults.photon_unit)),
            default_profile_coordinate=str(getattr(settings, "default_profile_coordinate", defaults.default_profile_coordinate)),
            wheel_guard_enabled=bool(getattr(settings, "wheel_guard_enabled", defaults.wheel_guard_enabled)),
            last_open_directory=str(getattr(settings, "last_open_directory", defaults.last_open_directory)),
        )

    @staticmethod
    def _convert_between_units(values: float | np.ndarray, source_unit: str, target_unit: str, factors: dict[str, float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        source_factor = float(factors.get(str(source_unit), 1.0))
        target_factor = float(factors.get(str(target_unit), 1.0))
        if source_factor == 0.0:
            return array
        return array / source_factor * target_factor

    @staticmethod
    def _replace_bracket_unit(label: str, unit: str) -> str:
        text = str(label)
        if "[" in text and "]" in text:
            prefix = text.rsplit("[", 1)[0].rstrip()
            return f"{prefix} [{unit}]"
        return f"{text} [{unit}]"

    def _time_unit(self) -> str:
        return str(self._display_settings.time_unit or "s")

    def _length_unit(self) -> str:
        return str(self._display_settings.length_unit or "cm")

    def _density_unit(self) -> str:
        return str(self._display_settings.density_unit or "g/cm3")

    def _pressure_unit(self) -> str:
        return str(self._display_settings.pressure_unit or "J/cm3")

    def _temperature_unit(self) -> str:
        return str(self._display_settings.temperature_unit or "eV")

    def _velocity_unit(self) -> str:
        return str(self._display_settings.velocity_unit or "cm/s")

    def _specific_energy_unit(self) -> str:
        return str(self._display_settings.specific_energy_unit or "J/g")

    def _rate_unit(self) -> str:
        return str(self._display_settings.rate_unit or "J/g/s")

    def _number_density_unit(self) -> str:
        return str(self._display_settings.number_density_unit or "1/cm3")

    def _angle_unit(self) -> str:
        normalized = str(self._display_settings.angle_unit or "deg").lower()
        return "rad" if normalized == "rad" else "deg"

    def _photon_unit(self) -> str:
        normalized = str(self._display_settings.photon_unit or "keV")
        if normalized not in {"keV", "eV", "nm"}:
            return "keV"
        return normalized

    def _display_time_from_seconds(self, values_s: float | np.ndarray) -> np.ndarray:
        return np.asarray(values_s, dtype=np.float64) * TIME_FACTORS.get(self._time_unit(), 1.0)

    def _display_time_from_ns(self, values_ns: float | np.ndarray) -> np.ndarray:
        return self._convert_between_units(values_ns, "ns", self._time_unit(), TIME_FACTORS)

    def _display_length_from_cm(self, values_cm: float | np.ndarray) -> np.ndarray:
        return np.asarray(values_cm, dtype=np.float64) * LENGTH_FACTORS.get(self._length_unit(), 1.0)

    def _display_length_from_um(self, values_um: float | np.ndarray) -> np.ndarray:
        return self._convert_between_units(values_um, "um", self._length_unit(), LENGTH_FACTORS)

    def _display_density_from_g_cm3(self, values_g_cm3: float | np.ndarray) -> np.ndarray:
        return np.asarray(values_g_cm3, dtype=np.float64) * DENSITY_FACTORS.get(self._density_unit(), 1.0)

    def _display_pressure_from_j_cm3(self, values_j_cm3: float | np.ndarray) -> np.ndarray:
        return np.asarray(values_j_cm3, dtype=np.float64) * PRESSURE_FACTORS.get(self._pressure_unit(), 1.0)

    def _density_display_to_g_cm3(self, displayed_value: float, *, unit: str | None = None) -> float:
        density_unit = self._density_unit() if unit is None else str(unit)
        factor = float(DENSITY_FACTORS.get(density_unit, 1.0))
        if factor == 0.0:
            return float(displayed_value)
        return float(displayed_value) / factor

    def _display_temperature_from_ev(self, values_ev: float | np.ndarray) -> np.ndarray:
        return np.asarray(values_ev, dtype=np.float64) * TEMPERATURE_FACTORS.get(self._temperature_unit(), 1.0)

    def _display_velocity_from_cm_s(self, values_cm_s: float | np.ndarray) -> np.ndarray:
        return np.asarray(values_cm_s, dtype=np.float64) * VELOCITY_FACTORS.get(self._velocity_unit(), 1.0)

    def _display_specific_energy_from_j_g(self, values_j_g: float | np.ndarray) -> np.ndarray:
        return np.asarray(values_j_g, dtype=np.float64) * SPECIFIC_ENERGY_FACTORS.get(self._specific_energy_unit(), 1.0)

    def _display_rate_from_j_g_s(self, values_j_g_s: float | np.ndarray) -> np.ndarray:
        return np.asarray(values_j_g_s, dtype=np.float64) * RATE_FACTORS.get(self._rate_unit(), 1.0)

    def _display_velocity_from_km_s(self, values_km_s: float | np.ndarray) -> np.ndarray:
        return self._convert_between_units(values_km_s, "km/s", self._velocity_unit(), VELOCITY_FACTORS)

    def _display_number_density(self, values: float | np.ndarray, *, source_unit: str) -> np.ndarray:
        return self._convert_between_units(values, source_unit, self._number_density_unit(), NUMBER_DENSITY_FACTORS)

    def _display_angle_from_deg(self, values_deg: float | np.ndarray, *, unit: str | None = None) -> np.ndarray:
        array = np.asarray(values_deg, dtype=np.float64)
        if (self._angle_unit() if unit is None else str(unit).lower()) == "rad":
            return np.deg2rad(array)
        return array

    def _angle_display_to_deg(self, displayed_value: float, *, unit: str | None = None) -> float:
        value = float(displayed_value)
        if (self._angle_unit() if unit is None else str(unit).lower()) == "rad":
            return math.degrees(value)
        return value

    def _display_photon_energy_from_kev(self, value_kev: float, *, unit: str | None = None) -> float:
        photon_unit = _normalize_photon_unit(self._photon_unit() if unit is None else str(unit))
        if photon_unit == "eV":
            return float(value_kev) * 1.0e3
        if photon_unit == "Angstrom":
            return photon_energy_kev_to_wavelength_angstrom(float(value_kev))
        if photon_unit == "nm":
            return photon_energy_kev_to_wavelength_angstrom(float(value_kev)) * 0.1
        return float(value_kev)

    def _photon_display_to_kev(self, displayed_value: float, *, unit: str | None = None) -> float:
        unit = _normalize_photon_unit(self._photon_unit() if unit is None else str(unit))
        if unit == "eV":
            return float(displayed_value) * 1.0e-3
        if unit == "Angstrom":
            return photon_energy_ev_from_wavelength_nm(float(displayed_value) * 0.1) * 1.0e-3
        if unit == "nm":
            return photon_energy_ev_from_wavelength_nm(float(displayed_value)) * 1.0e-3
        return float(displayed_value)

    def _photon_spin_unit(self, spin: QtWidgets.QDoubleSpinBox) -> str:
        property_value = spin.property("photon_unit")
        if property_value is not None:
            normalized = _normalize_photon_unit(str(property_value))
            if normalized:
                return normalized
        suffix = str(spin.suffix() or "").strip().lower()
        if suffix == "ev":
            return "eV"
        if suffix == "kev":
            return "keV"
        if suffix == "nm":
            return "nm"
        if suffix in {"angstrom", "a"}:
            return "Angstrom"
        return _normalize_photon_unit(self._photon_unit())

    def _display_line_value_from_nm(self, wavelength_nm: float, *, unit: str | None = None) -> float:
        unit = self._photon_unit() if unit is None else str(unit)
        if unit == "nm":
            return float(wavelength_nm)
        energy_ev = photon_energy_ev_from_wavelength_nm(float(wavelength_nm))
        if unit == "keV":
            return energy_ev * 1.0e-3
        return energy_ev

    def _line_display_value_to_nm(self, displayed_value: float, *, unit: str | None = None) -> float:
        unit = self._photon_unit() if unit is None else str(unit)
        if unit == "nm":
            return float(displayed_value)
        energy_ev = float(displayed_value) * (1.0e3 if unit == "keV" else 1.0)
        if energy_ev <= 0.0:
            raise ValueError("Photon line input must stay positive.")
        return photon_energy_kev_to_wavelength_angstrom(energy_ev * 1.0e-3) * 0.1

    def _format_time(self, value_s: float | None) -> str:
        if value_s is None or not math.isfinite(float(value_s)):
            return "-"
        value = float(np.asarray(self._display_time_from_seconds(float(value_s)), dtype=np.float64).reshape(-1)[0])
        if abs(value) >= 0.1 and abs(value) < 1.0e4:
            return f"{value:.3f} {self._time_unit()}"
        magnitude_s = abs(float(value_s))
        for unit, factor in (("s", 1.0), ("ms", 1.0e3), ("us", 1.0e6), ("ns", 1.0e9), ("ps", 1.0e12), ("fs", 1.0e15)):
            scaled = float(value_s) * factor
            if magnitude_s == 0.0 or (abs(scaled) >= 0.1 and abs(scaled) < 1000.0):
                return f"{scaled:.3f} {unit}"
        return f"{float(value_s):.3e} s"

    def _format_length(self, value_cm: float | None) -> str:
        if value_cm is None or not math.isfinite(float(value_cm)):
            return "-"
        value = float(np.asarray(self._display_length_from_cm(float(value_cm)), dtype=np.float64).reshape(-1)[0])
        return f"{value:.3f} {self._length_unit()}"

    def _format_density(self, value_g_cm3: float | None, fmt: str = "{:.4g}") -> str:
        if value_g_cm3 is None or not math.isfinite(float(value_g_cm3)):
            return "-"
        return f"{fmt.format(float(self._display_density_from_g_cm3(float(value_g_cm3))))} {self._density_unit()}"

    def _format_pressure(self, value_j_cm3: float | None, fmt: str = "{:.4g}") -> str:
        if value_j_cm3 is None or not math.isfinite(float(value_j_cm3)):
            return "-"
        return f"{fmt.format(float(self._display_pressure_from_j_cm3(float(value_j_cm3))))} {self._pressure_unit()}"

    def _format_temperature(self, value_ev: float | None, fmt: str = "{:.4g}") -> str:
        if value_ev is None or not math.isfinite(float(value_ev)):
            return "-"
        return f"{fmt.format(float(self._display_temperature_from_ev(float(value_ev))))} {self._temperature_unit()}"

    def _format_velocity(self, value_cm_s: float | None, fmt: str = "{:.4g}") -> str:
        if value_cm_s is None or not math.isfinite(float(value_cm_s)):
            return "-"
        return f"{fmt.format(float(self._display_velocity_from_cm_s(float(value_cm_s))))} {self._velocity_unit()}"

    def _format_fraction(self, value: float | None, fmt: str = "{:.3f}") -> str:
        if value is None or not math.isfinite(float(value)):
            return "-"
        return fmt.format(float(value))

    def _format_impulse(self, value: float | None, fmt: str = "{:.3g}") -> str:
        if value is None or not math.isfinite(float(value)):
            return "-"
        return f"{fmt.format(float(value))} J s/cm^3"

    def _format_specific_energy(self, value_j_g: float | None, fmt: str = "{:.4g}") -> str:
        if value_j_g is None or not math.isfinite(float(value_j_g)):
            return "-"
        return f"{fmt.format(float(self._display_specific_energy_from_j_g(float(value_j_g))))} {self._specific_energy_unit()}"

    def _format_rate(self, value_j_g_s: float | None, fmt: str = "{:.4g}") -> str:
        if value_j_g_s is None or not math.isfinite(float(value_j_g_s)):
            return "-"
        return f"{fmt.format(float(self._display_rate_from_j_g_s(float(value_j_g_s))))} {self._rate_unit()}"

    def _format_number_density(self, value: float | None, *, source_unit: str, fmt: str = "{:.4g}") -> str:
        if value is None or not math.isfinite(float(value)):
            return "-"
        converted = float(self._display_number_density(float(value), source_unit=source_unit))
        return f"{fmt.format(converted)} {self._number_density_unit()}"

    def _format_angle(self, value_deg: float | None, fmt: str = "{:.4g}") -> str:
        if value_deg is None or not math.isfinite(float(value_deg)):
            return "-"
        converted = float(self._display_angle_from_deg(float(value_deg)))
        return f"{fmt.format(converted)} {_angle_unit_label(self._angle_unit())}"

    def _format_photon_value_from_kev(self, value_kev: float | None, fmt: str = "{:.4g}", *, unit: str | None = None) -> str:
        if value_kev is None or not math.isfinite(float(value_kev)):
            return "-"
        display_unit = self._photon_unit() if unit is None else str(unit)
        converted = self._display_photon_energy_from_kev(float(value_kev), unit=display_unit)
        return f"{fmt.format(float(converted))} {_photon_unit_label(display_unit)}"

    def _set_text_browser_text_preserving_scroll(self, widget: QtWidgets.QTextBrowser, text: str) -> None:
        normalized = str(text)
        if widget.toPlainText() == normalized:
            return
        vertical = widget.verticalScrollBar()
        horizontal = widget.horizontalScrollBar()
        follow_vertical = vertical.value() >= max(0, vertical.maximum() - 4)
        follow_horizontal = horizontal.value() >= max(0, horizontal.maximum() - 4)
        previous_vertical = int(vertical.value())
        previous_horizontal = int(horizontal.value())
        widget.setPlainText(normalized)
        if follow_vertical:
            vertical.setValue(vertical.maximum())
        else:
            vertical.setValue(min(previous_vertical, vertical.maximum()))
        if follow_horizontal:
            horizontal.setValue(horizontal.maximum())
        else:
            horizontal.setValue(min(previous_horizontal, horizontal.maximum()))

    def _apply_display_settings_to_controls(self, previous: ViewerSettings, current: ViewerSettings) -> None:
        angle_unit = self._angle_unit()
        angle_suffix = f" {_angle_unit_label(angle_unit)}"
        for spin, minimum_deg, maximum_deg, step_deg in (
            (self.los_angle_spin, 0.0, 89.0, 2.5),
            (self.xrd_angle_spin, 0.1, 89.9, 0.5),
            (self.plasmon_angle_spin, 0.1, 180.0, 1.0),
        ):
            native_value_deg = self._angle_display_to_deg(float(spin.value()), unit=str(previous.angle_unit))
            spin.blockSignals(True)
            try:
                if angle_unit == "rad":
                    spin.setRange(math.radians(minimum_deg), math.radians(maximum_deg))
                    spin.setSingleStep(math.radians(step_deg))
                    spin.setDecimals(6)
                    spin.setSuffix(angle_suffix)
                    spin.setValue(float(np.deg2rad(native_value_deg)))
                else:
                    spin.setRange(minimum_deg, maximum_deg)
                    spin.setSingleStep(step_deg)
                    spin.setDecimals(2)
                    spin.setSuffix(angle_suffix)
                    spin.setValue(native_value_deg)
            finally:
                spin.blockSignals(False)

        photon_unit = self._photon_unit()
        photon_suffix = f" {_photon_unit_label(photon_unit)}"
        for spin in (self.xrd_energy_spin, self.plasmon_energy_spin):
            previous_unit = _normalize_photon_unit(str(spin.property("photon_unit") or previous.photon_unit))
            native_kev = self._photon_display_to_kev(float(spin.value()), unit=previous_unit)
            spin.blockSignals(True)
            try:
                spin.setDecimals(3 if photon_unit == "keV" else 4)
                spin.setSuffix(photon_suffix)
                if photon_unit == "nm":
                    spin.setRange(0.01, 10000.0)
                    spin.setSingleStep(0.1)
                elif photon_unit == "eV":
                    spin.setRange(100.0, 30000.0)
                    spin.setSingleStep(100.0)
                else:
                    spin.setRange(0.1, 30.0)
                    spin.setSingleStep(0.1)
                spin.setValue(self._display_photon_energy_from_kev(native_kev, unit=photon_unit))
                spin.setProperty("photon_unit", photon_unit)
            finally:
                spin.blockSignals(False)

        native_density_g_cm3 = self._density_display_to_g_cm3(float(self.min_density_spin.value()), unit=str(previous.density_unit))
        self.min_density_spin.blockSignals(True)
        try:
            self.min_density_spin.setDecimals(4 if self._density_unit() == "g/cm3" else 1)
            self.min_density_spin.setSingleStep(0.1 if self._density_unit() == "g/cm3" else 100.0)
            if self._density_unit() == "kg/m3":
                self.min_density_spin.setRange(0.0, 1.0e9)
            else:
                self.min_density_spin.setRange(0.0, 1.0e6)
            self.min_density_spin.setSuffix(f" {self._density_unit()}")
            self.min_density_spin.setValue(float(self._display_density_from_g_cm3(native_density_g_cm3)))
        finally:
            self.min_density_spin.blockSignals(False)

        native_line_nm = self._line_display_value_to_nm(float(self.spectroscopy_wavelength_spin.value()), unit=str(previous.photon_unit))
        self.spectroscopy_wavelength_spin.blockSignals(True)
        try:
            if photon_unit == "nm":
                self.spectroscopy_line_label.setText("Line wavelength")
                self.spectroscopy_wavelength_spin.setRange(0.1, 10000.0)
                self.spectroscopy_wavelength_spin.setSingleStep(1.0)
                self.spectroscopy_wavelength_spin.setDecimals(3)
                self.spectroscopy_wavelength_spin.setSuffix(" nm")
            elif photon_unit == "eV":
                self.spectroscopy_line_label.setText("Line energy")
                self.spectroscopy_wavelength_spin.setRange(0.1, 1.0e6)
                self.spectroscopy_wavelength_spin.setSingleStep(1.0)
                self.spectroscopy_wavelength_spin.setDecimals(6)
                self.spectroscopy_wavelength_spin.setSuffix(" eV")
            else:
                self.spectroscopy_line_label.setText("Line energy")
                self.spectroscopy_wavelength_spin.setRange(1.0e-4, 1.0e3)
                self.spectroscopy_wavelength_spin.setSingleStep(0.1)
                self.spectroscopy_wavelength_spin.setDecimals(7)
                self.spectroscopy_wavelength_spin.setSuffix(" keV")
            self.spectroscopy_wavelength_spin.setToolTip(
                "Photon line input follows the global photon display unit and is converted back to the native wavelength internally."
            )
            self.spectroscopy_wavelength_spin.setValue(self._display_line_value_from_nm(native_line_nm, unit=photon_unit))
        finally:
            self.spectroscopy_wavelength_spin.blockSignals(False)

        degrees_index = self.xrd_display_combo.findData("degrees")
        if degrees_index >= 0:
            self.xrd_display_combo.setItemText(
                degrees_index,
                f"Bragg shift [{_angle_unit_label(self._angle_unit())}]",
            )
        for combo in (self.plasmon_coordinate_axis_combo, self.plasmon_scan_axis_combo):
            angle_index = combo.findData(PLASMON_AXIS_ANGLE_DEG)
            if angle_index >= 0:
                combo.setItemText(angle_index, f"Angle [{_angle_unit_label(self._angle_unit())}]")
            k_index = combo.findData(PLASMON_AXIS_K_ANGSTROM_INV)
            if k_index >= 0:
                combo.setItemText(k_index, "k [1/A]")
        self._configure_plasmon_coordinate_spin(
            axis=self._selected_plasmon_coordinate_axis(),
            canonical_value=self._selected_plasmon_coordinate_value(),
        )
        self._configure_plasmon_scan_spins(
            axis=self._selected_plasmon_scan_axis(),
            start=self._selected_plasmon_scan_start(),
            stop=self._selected_plasmon_scan_stop(),
        )

    def set_display_settings(self, settings: object) -> None:
        previous = self._display_settings
        updated = self._coerce_display_settings(settings)
        self._display_settings = updated
        self._wheel_guard.set_enabled(bool(updated.wheel_guard_enabled))
        self._apply_display_settings_to_controls(previous, updated)
        self._sync_context_summary()
        if self._current_result is not None:
            self._refresh_current_result_display()

    def parameters(self) -> DerivedAnalysisParameters:
        zone_lower = int(self.zone_lower_spin.value())
        zone_upper = int(self.zone_upper_spin.value())
        preheat_target_region_id = self.preheat_target_combo.currentData()
        return DerivedAnalysisParameters(
            xrd_photon_energy_kev=self._photon_display_to_kev(
                float(self.xrd_energy_spin.value()),
                unit=self._photon_spin_unit(self.xrd_energy_spin),
            ),
            xrd_initial_bragg_angle_deg=self._angle_display_to_deg(float(self.xrd_angle_spin.value())),
            plasmon_photon_energy_kev=self._photon_display_to_kev(
                float(self.plasmon_energy_spin.value()),
                unit=self._photon_spin_unit(self.plasmon_energy_spin),
            ),
            plasmon_scattering_angle_deg=self._selected_plasmon_fixed_angle_deg(),
            plasmon_adiabatic_index=float(self.plasmon_gamma_spin.value()),
            plasmon_model=self._selected_plasmon_model(),
            plasmon_execution_mode=self._selected_plasmon_execution_mode(),
            plasmon_observable_mode=self._selected_plasmon_observable_mode(),
            plasmon_energy_window_ev=float(self.plasmon_window_spin.value()),
            plasmon_energy_points=int(self.plasmon_points_spin.value()),
            plasmon_instrument_fwhm_ev=float(self.plasmon_fwhm_spin.value()),
            plasmon_collision_model=str(self.plasmon_collision_model_combo.currentData() or PLASMON_COLLISION_MODEL_NRL_CONSTANT),
            plasmon_collision_scale=float(self.plasmon_collision_scale_spin.value()),
            plasmon_manual_collision_rate_s=float(self.plasmon_manual_collision_spin.value()),
            plasmon_lfc_model=str(self.plasmon_lfc_combo.currentData() or PLASMON_LFC_MODEL_NONE),
            plasmon_integration_mode=str(self.plasmon_integration_combo.currentData() or PLASMON_INTEGRATION_MODE_EFFECTIVE_STATE),
            plasmon_normalization=str(self.plasmon_normalization_combo.currentData() or PLASMON_NORMALIZATION_PEAK),
            plasmon_cluster_log_ne_tol=float(self.plasmon_cluster_log_ne_spin.value()),
            plasmon_cluster_log_te_tol=float(self.plasmon_cluster_log_te_spin.value()),
            plasmon_cluster_z_tol=float(self.plasmon_cluster_z_spin.value()),
            plasmon_electron_policy=self._selected_plasmon_electron_policy(),
            plasmon_benchmark_preset=self._selected_plasmon_benchmark_preset(),
            plasmon_study_mode=self._selected_plasmon_study_mode(),
            plasmon_coordinate_axis=self._selected_plasmon_coordinate_axis(),
            plasmon_coordinate_value=self._selected_plasmon_coordinate_value(),
            plasmon_scan_axis=self._selected_plasmon_scan_axis(),
            plasmon_scan_start=self._selected_plasmon_scan_start(),
            plasmon_scan_stop=self._selected_plasmon_scan_stop(),
            plasmon_scan_points=int(self.plasmon_scan_points_spin.value()),
            plasmon_compare_models=bool(self.plasmon_compare_models_checkbox.isChecked()),
            plasmon_compare_model_names=self._checked_plasmon_compare_models(),
            plasmon_compare_policies=bool(self.plasmon_compare_policies_checkbox.isChecked()),
            spectroscopy_line_wavelength_nm=self._line_display_value_to_nm(float(self.spectroscopy_wavelength_spin.value())),
            transmission_mode=str(self.transmission_mode_combo.currentData()),
            transmission_photon_energy_kev=self._selected_transmission_energy_kev(),
            observation_side=str(self.observation_side_combo.currentData()),
            line_of_sight_angle_deg=self._angle_display_to_deg(float(self.los_angle_spin.value())),
            profile_coordinate_mode=str(self.profile_coordinate_combo.currentData()),
            reuse_viewer_subset=bool(self.reuse_viewer_subset_checkbox.isChecked()),
            derived_region_ids=self._checked_values_for_filter(self.region_list),
            derived_material_ids=self._checked_values_for_filter(self.material_list),
            exclude_entry_region=bool(self.exclude_entry_region_checkbox.isChecked()),
            exclude_low_density=bool(self.exclude_low_density_checkbox.isChecked() or self.min_density_spin.value() > 0.0),
            min_density_g_cm3=self._density_display_to_g_cm3(float(self.min_density_spin.value())),
            exclude_opposite_velocity=bool(self.exclude_opposite_velocity_checkbox.isChecked()),
            zone_index_lower=(zone_lower if zone_lower > 1 else None),
            zone_index_upper=(zone_upper if zone_upper > 0 and zone_upper < max(1, self._context.n_zones) else None),
            weighting_mode=str(self.weighting_combo.currentData()),
            preheat_target_region_id=(None if preheat_target_region_id is None else int(preheat_target_region_id)),
        )

    def _refresh_display_only(self) -> None:
        if self._current_result is None:
            return
        self._populate_xrd(self._current_result.xrd)
        self._populate_spectroscopy(self._current_result.spectroscopy)

    def _set_shock_navigation_mode(self, mode: str) -> None:
        self.shock_position_plot.set_navigation_mode(mode)
        self.shock_velocity_plot.set_navigation_mode(mode)

    def _xrd_display_mode(self) -> str:
        return str(self.xrd_display_combo.currentData() or "degrees")

    def _spectroscopy_shift_unit(self) -> str:
        return str(self.spectroscopy_shift_unit_combo.currentData() or "nm")

    def _snapshot_suffix(self, snapshot_index: int, snapshot_time_s: float | None = None) -> str:
        if snapshot_time_s is None and self._current_result is not None:
            snapshot_time_s = float(self._current_result.snapshot_time_s)
        return f" | snapshot {int(snapshot_index)} @ {self._format_time(snapshot_time_s)}"

    def _with_snapshot_titles(
        self,
        bundles: tuple[DerivedPlotBundle, ...],
        *,
        snapshot_index: int,
        snapshot_time_s: float | None = None,
    ) -> tuple[DerivedPlotBundle, ...]:
        suffix = self._snapshot_suffix(snapshot_index, snapshot_time_s)
        return tuple(_clone_bundle(bundle, title=f"{bundle.title}{suffix}") for bundle in bundles)

    def apply_theme(self, theme: ViewerTheme) -> None:
        self._theme = theme
        panels = (
            self.preheat_plot_panel,
            self.xrd_plot_panel,
            self.plasmon_plot_panel,
            self.transmission_plot_panel,
            self.spectroscopy_plot_panel,
        )
        for panel in panels:
            panel.apply_theme(theme)
        self.shock_position_plot.apply_theme(theme)
        self.shock_velocity_plot.apply_theme(theme)
        self.wavefront_position_plot.apply_theme(theme)
        self.wavefront_score_plot.apply_theme(theme)
        self.analysis_banner.setStyleSheet(
            f"background: {theme.panel_background};"
            f"border: 1px solid {theme.border_color};"
            "border-radius: 6px;"
            "padding: 6px 8px;"
            f"color: {theme.text_color};"
        )
        self.status_label.setStyleSheet(f"color: {theme.subtle_text};")
        self.result_status_label.setStyleSheet(f"color: {theme.subtle_text};")
        self.performance_summary_label.setStyleSheet(f"color: {theme.subtle_text};")
        self.wavefront_metrics_label.setStyleSheet(f"color: {theme.subtle_text};")
        self.warning_summary_label.setStyleSheet(f"color: {theme.subtle_text};")
        self._set_shock_navigation_mode("pan" if self._shock_pan_action.isChecked() else "zoom")
        if self._current_result is not None:
            self._refresh_current_result_display()
        else:
            self._sync_context_summary()

    def _sync_context_summary(self) -> None:
        if not self._context.has_run or self._context.path is None:
            self.run_path_label.setText("-")
            self.run_summary_label.setText("-")
            self.snapshot_label.setText("-")
            self.subset_label.setText("-")
            self.warning_summary_label.setText("Warnings: -")
            return
        snapshot_time = (
            float(self._context.time_values[self._context.snapshot_index])
            if self._context.time_values.size and 0 <= self._context.snapshot_index < self._context.time_values.size
            else float("nan")
        )
        self.run_path_label.setText(str(self._context.path))
        self.run_summary_label.setText(
            f"{self._context.n_zones} zones | {self._context.n_snapshots} snapshots | "
            f"map={self._context.map_coordinate} | slice={self._context.slice_coordinate}"
        )
        self.snapshot_label.setText(f"{self._context.snapshot_index} @ {self._format_time(snapshot_time)}")
        self.subset_label.setText(_selected_summary(self._context))
        if self._current_result is None:
            self.warning_summary_label.setText("Warnings: waiting for analysis update")
            self.result_status_label.setText(
                f"Updating snapshot {self._context.snapshot_index} @ {self._format_time(snapshot_time)}"
            )
        elif self._current_result.snapshot_index != self._context.snapshot_index:
            self.result_status_label.setText(
                f"Updating snapshot {self._context.snapshot_index} @ {self._format_time(snapshot_time)}"
            )

    def _refresh_current_result_display(self) -> None:
        result = self._current_result
        if result is None:
            self._sync_context_summary()
            return
        self.set_result(result)

    def set_context(self, context: RunContext) -> None:
        previous_context = self._context
        previous_context_key = previous_context.context_key if previous_context.has_run else None
        self._context = context.copy()
        current_context_key = self._context.context_key if self._context.has_run else None
        run_changed = previous_context_key != current_context_key
        if not self._context.has_run or self._context.path is None:
            self._reset_preheat_navigation_state()
            self._sync_context_summary()
            self.result_status_label.setText("Load a run, then switch to Derived / Analysis.")
            self._available_region_ids = ()
            self._available_material_ids = ()
            self.zone_lower_spin.setRange(1, 1)
            self.zone_upper_spin.setRange(1, 1)
            self.zone_lower_spin.setValue(1)
            self.zone_upper_spin.setValue(1)
            self._populate_checkable_list(self.region_list, (), label_prefix="Region", selected=())
            self._populate_checkable_list(self.material_list, (), label_prefix="Material", selected=())
            self._populate_preheat_target_combo()
            self._update_snapshot_markers(None)
            return

        snapshot_time = (
            float(self._context.time_values[self._context.snapshot_index])
            if 0 <= self._context.snapshot_index < self._context.time_values.size
            else float("nan")
        )
        if run_changed:
            self._reset_preheat_navigation_state()
            self.warning_summary_label.setText("Warnings: waiting for analysis update")
            self.result_status_label.setText("Waiting for analysis update.")
            self.result_tabs.blockSignals(True)
            self.result_tabs.setCurrentWidget(self.shock_tab)
            self.result_tabs.blockSignals(False)
            self._available_region_ids = tuple(int(value) for value in sorted(set(np.asarray(self._context.zone_region_id, dtype=np.int32).tolist())))
            materials = np.asarray(self._context.zone_material_index, dtype=np.int32)
            self._available_material_ids = tuple(int(value) for value in sorted(set(np.abs(materials).tolist())))
            self._populate_checkable_list(
                self.region_list,
                self._available_region_ids,
                label_prefix="Region",
                selected=self._available_region_ids,
            )
            self._populate_checkable_list(
                self.material_list,
                self._available_material_ids,
                label_prefix="Material",
                selected=self._available_material_ids,
            )

            max_zone = max(1, self._context.n_zones)
            self.zone_lower_spin.blockSignals(True)
            self.zone_upper_spin.blockSignals(True)
            self.zone_lower_spin.setRange(1, max_zone)
            self.zone_upper_spin.setRange(1, max_zone)
            self.zone_lower_spin.setValue(1)
            self.zone_upper_spin.setValue(max_zone)
            self.zone_lower_spin.blockSignals(False)
            self.zone_upper_spin.blockSignals(False)

            self._populate_preheat_target_combo()
            self._apply_default_profile_coordinate()
        self._sync_context_summary()
        if self._current_result is not None and self._current_result.snapshot_index == self._context.snapshot_index:
            self.result_status_label.setText(
                f"Snapshot {self._context.snapshot_index} @ {self._format_time(snapshot_time)} ready."
            )
        self._update_snapshot_markers(snapshot_time)

    def set_busy(self, busy: bool, message: str) -> None:
        self.refresh_button.setEnabled(not busy)
        self.transmission_refine_button.setEnabled(not busy and self._current_result is not None)
        self._busy_message = str(message)
        if busy:
            if not self._busy_elapsed_timer.isActive():
                self._busy_started_at = time.perf_counter()
            self.activity_progress.setRange(0, 0)
            self.activity_progress.show()
            self.cancel_button.setEnabled(True)
            self.cancel_button.show()
            self._busy_elapsed_timer.start()
            self._update_plasmon_recompute_ui()
            self._refresh_busy_status()
            return
        self._busy_elapsed_timer.stop()
        self.activity_progress.hide()
        self.cancel_button.hide()
        self.cancel_button.setEnabled(False)
        self.activity_progress.setRange(0, 1)
        self.activity_progress.setValue(0)
        self._update_plasmon_recompute_ui()
        self.status_label.setText(message)
        self.result_status_label.setText(message)

    def set_performance_summary(self, message: str, *, wavefront: bool = False, preheat: bool = False) -> None:
        text = str(message).strip() or "Performance: waiting for analysis update."
        self._last_performance_summary = text
        self.performance_summary_label.setText(text)
        if wavefront:
            self._last_wavefront_performance_summary = text
            self.wavefront_metrics_label.setText(text)
        if preheat:
            self._last_preheat_performance_summary = text
            self.preheat_metrics_label.setText(text)

    def _refresh_busy_status(self) -> None:
        elapsed_s = max(0.0, time.perf_counter() - float(self._busy_started_at))
        text = f"{self._busy_message} | elapsed {elapsed_s:.1f} s"
        self.status_label.setText(text)
        self.result_status_label.setText(text)

    def clear_results(self, message: str) -> None:
        self._current_result = None
        self._reset_preheat_navigation_state()
        self.cancel_button.hide()
        self.cancel_button.setEnabled(False)
        self.analysis_banner.setText(message)
        self.result_status_label.setText(message)
        self._last_performance_summary = "Performance: waiting for analysis update."
        self._last_wavefront_performance_summary = "Performance: waiting for WaveFront analysis."
        self._last_preheat_performance_summary = "Performance: waiting for Preheat analysis."
        self.performance_summary_label.setText(self._last_performance_summary)
        self.wavefront_metrics_label.setText(self._last_wavefront_performance_summary)
        self.preheat_metrics_label.setText(self._last_preheat_performance_summary)
        self.shock_summary_label.setText("-")
        self.wavefront_summary_label.setText("Open WaveFront to load multi-branch wave tracking for the active run.")
        self.wavefront_overview_label.setText(
            "Default WaveFront view shows tracked branches only. Provisional detections stay out of the main plots unless explicitly requested."
        )
        self.wavefront_plot_empty_label.hide()
        self.wavefront_plot_splitter.show()
        self.preheat_summary_label.setText("Open Preheat to load target pre-modification diagnostics for the active run.")
        self.preheat_overview_label.setText(
            "Preheat diagnoses how the selected target region changes before the tracked primary compressive branch arrives."
        )
        self.xrd_summary_label.setText("-")
        self.plasmon_summary_label.setText("-")
        self.transmission_summary_label.setText("-")
        self.transmission_model_label.setText("Requested: Thomson | Applied: Thomson (baseline)")
        self.transmission_backend_label.setText("Backend: not yet probed")
        self.transmission_applicability_label.setText("Applicability: estimated on apply")
        self.transmission_refinement_label.setText("Status: Thomson quick-look estimate.")
        self.transmission_status_pane.clear()
        self.transmission_refine_button.setText("Apply Transmission Model")
        self.transmission_refine_button.setEnabled(False)
        self.spectroscopy_summary_label.setText("-")
        self.plasmon_metrics.clear()
        self.spectroscopy_metrics.clear()
        self.shock_position_plot.clear_plot()
        self.shock_velocity_plot.clear_plot()
        self.wavefront_position_plot.clear_plot()
        self.wavefront_score_plot.clear_plot()
        self.preheat_plot_panel.clear()
        self.xrd_plot_panel.clear()
        self.plasmon_plot_panel.clear()
        self._plasmon_recompute_pending = False
        self._plasmon_controls_dirty = False
        self._update_plasmon_recompute_ui()
        self.transmission_plot_panel.clear()
        self.spectroscopy_plot_panel.clear()
        self.shock_interface_table.setRowCount(0)
        self.wavefront_branch_table.setRowCount(0)
        self.wavefront_event_table.setRowCount(0)
        self.wavefront_notes.clear()
        self.preheat_summary_table.setRowCount(0)
        self.preheat_onset_table.setRowCount(0)
        self.preheat_budget_table.setRowCount(0)
        self.preheat_notes.clear()
        self.preheat_snapshot_label.setText("No selected snapshot")
        self.preheat_snapshot_table.setRowCount(0)
        self.preheat_profile_status_label.setText(
            "Snapshot profiles will appear here once a valid preheat region and time are available."
        )
        self.preheat_time_status_label.setText(
            "Select a region of interest and open this tab to inspect preheat before shock entry or at a manual snapshot."
        )
        self._populate_preheat_target_combo()
        self._set_wavefront_lower_layout_mode("position")
        self.xrd_table.setRowCount(0)
        self.transmission_table.setRowCount(0)
        self.warnings_tree.clear()
        self.warning_summary_label.setText("Warnings: -")
        self._update_snapshot_markers(None)
        self._on_transmission_controls_changed()

    def set_result(self, result: DerivedAnalysisResult) -> None:
        self._current_result = result
        selected_preheat_region = None if result.preheat is None or str(result.preheat.target_selection_mode or "auto") != "user_selected" else result.preheat.target_region_id
        self._populate_preheat_target_combo(selected_region_id=selected_preheat_region, preheat=result.preheat)
        selection_notes = f" | filters: {'; '.join(result.selection.notes)}" if result.selection.notes else ""
        self.analysis_banner.setText(
            f"{result.dataset_path.name} | snapshot {result.snapshot_index} @ {self._format_time(result.snapshot_time_s)} | "
            f"{result.selected_zone_count} selected zones | weighting={result.selection.weighting_mode} | "
            f"{result.geometry.observation_side} side | LOS cos={result.geometry.line_of_sight_cosine:.3f}"
            f"{selection_notes}"
        )
        self.result_status_label.setText(
            f"Snapshot {result.snapshot_index} @ {self._format_time(result.snapshot_time_s)} ready."
        )
        self.warning_summary_label.setText(self._warning_summary_text(result.warnings))
        self._populate_shock(result.shock)
        self._populate_wavefront(result.wave_tracking, result.interface_events, result.preheat)
        self._populate_preheat(result.preheat)
        self._populate_xrd(result.xrd)
        self._populate_plasmon(result.plasmon)
        self._populate_transmission(result.transmission)
        self._populate_spectroscopy(result.spectroscopy)
        self._populate_warnings(result)
        self._update_snapshot_markers(result.snapshot_time_s)
        self._handle_result_tab_changed(self.result_tabs.currentIndex())

    def _update_snapshot_markers(self, snapshot_time_s: float | None) -> None:
        display_time = None if snapshot_time_s is None or not math.isfinite(float(snapshot_time_s)) else float(self._display_time_from_seconds(float(snapshot_time_s)))
        for panel in (
            self.xrd_plot_panel,
            self.plasmon_plot_panel,
            self.transmission_plot_panel,
            self.spectroscopy_plot_panel,
        ):
            panel.set_snapshot_marker(display_time)
        preheat_display_time = None
        if self._current_result is not None and self._current_result.preheat is not None:
            preheat = self._current_result.preheat
            preheat_time_axis = self._preheat_time_axis_s(preheat)
            selected_snapshot_index = self._preheat_display_snapshot_index
            if (
                selected_snapshot_index is not None
                and 0 <= int(selected_snapshot_index) < int(preheat_time_axis.size)
            ):
                selected_time_s = float(preheat_time_axis[int(selected_snapshot_index)])
                if math.isfinite(selected_time_s):
                    preheat_display_time = float(self._display_time_from_seconds(selected_time_s))
        self.preheat_plot_panel.set_snapshot_marker(preheat_display_time)
        if display_time is None:
            self.shock_position_plot.clear_cursor_marker()
            self.shock_velocity_plot.clear_cursor_marker()
            return
        self.shock_position_plot.set_cursor_marker(display_time, visible=True)
        self.shock_velocity_plot.set_cursor_marker(display_time, visible=True)

    def _bundle_time_x(self, bundle: DerivedPlotBundle) -> DerivedPlotBundle:
        return DerivedPlotBundle(
            key=bundle.key,
            title=bundle.title,
            x_label=self._replace_bracket_unit(bundle.x_label, self._time_unit()),
            y_label=bundle.y_label,
            x_values=np.asarray(self._display_time_from_ns(np.asarray(bundle.x_values, dtype=np.float64)), dtype=np.float64),
            y_series=tuple(np.asarray(series, dtype=np.float64) for series in bundle.y_series),
            curve_names=tuple(bundle.curve_names),
            boundary_positions=tuple(bundle.boundary_positions),
            value_scale_mode=bundle.value_scale_mode,
        )

    def _bundle_length_x(self, bundle: DerivedPlotBundle, *, source_unit: str = "um") -> DerivedPlotBundle:
        return DerivedPlotBundle(
            key=bundle.key,
            title=bundle.title,
            x_label=self._replace_bracket_unit(bundle.x_label, self._length_unit()),
            y_label=bundle.y_label,
            x_values=np.asarray(self._convert_between_units(bundle.x_values, source_unit, self._length_unit(), LENGTH_FACTORS), dtype=np.float64),
            y_series=tuple(np.asarray(series, dtype=np.float64) for series in bundle.y_series),
            curve_names=tuple(bundle.curve_names),
            boundary_positions=tuple(
                float(self._convert_between_units(np.asarray([value], dtype=np.float64), source_unit, self._length_unit(), LENGTH_FACTORS)[0])
                for value in bundle.boundary_positions
            ),
            value_scale_mode=bundle.value_scale_mode,
        )

    def _bundle_angle_x(self, bundle: DerivedPlotBundle) -> DerivedPlotBundle:
        return DerivedPlotBundle(
            key=bundle.key,
            title=bundle.title,
            x_label=self._replace_bracket_unit(bundle.x_label, _angle_unit_label(self._angle_unit())),
            y_label=bundle.y_label,
            x_values=np.asarray(self._display_angle_from_deg(np.asarray(bundle.x_values, dtype=np.float64)), dtype=np.float64),
            y_series=tuple(np.asarray(series, dtype=np.float64) for series in bundle.y_series),
            curve_names=tuple(bundle.curve_names),
            boundary_positions=tuple(bundle.boundary_positions),
            value_scale_mode=bundle.value_scale_mode,
        )

    def _bundle_density_y(self, bundle: DerivedPlotBundle, *, source_unit: str = "g/cm3") -> DerivedPlotBundle:
        converted = (
            tuple(np.asarray(self._display_density_from_g_cm3(np.asarray(series, dtype=np.float64)), dtype=np.float64) for series in bundle.y_series)
            if source_unit == "g/cm3"
            else tuple(np.asarray(self._convert_between_units(np.asarray(series, dtype=np.float64), source_unit, self._density_unit(), DENSITY_FACTORS), dtype=np.float64) for series in bundle.y_series)
        )
        return _clone_bundle(bundle, y_label=self._replace_bracket_unit(bundle.y_label, self._density_unit()), y_series=converted)

    def _bundle_temperature_y(self, bundle: DerivedPlotBundle) -> DerivedPlotBundle:
        converted = tuple(
            np.asarray(self._display_temperature_from_ev(np.asarray(series, dtype=np.float64)), dtype=np.float64)
            for series in bundle.y_series
        )
        return _clone_bundle(bundle, y_label=self._replace_bracket_unit(bundle.y_label, self._temperature_unit()), y_series=converted)

    def _bundle_velocity_y(self, bundle: DerivedPlotBundle, *, source_unit: str = "km/s") -> DerivedPlotBundle:
        converted = tuple(
            np.asarray(self._convert_between_units(np.asarray(series, dtype=np.float64), source_unit, self._velocity_unit(), VELOCITY_FACTORS), dtype=np.float64)
            for series in bundle.y_series
        )
        return _clone_bundle(bundle, y_label=self._replace_bracket_unit(bundle.y_label, self._velocity_unit()), y_series=converted)

    def _bundle_length_y(self, bundle: DerivedPlotBundle, *, source_unit: str = "um") -> DerivedPlotBundle:
        converted = tuple(
            np.asarray(self._convert_between_units(np.asarray(series, dtype=np.float64), source_unit, self._length_unit(), LENGTH_FACTORS), dtype=np.float64)
            for series in bundle.y_series
        )
        return _clone_bundle(bundle, y_label=self._replace_bracket_unit(bundle.y_label, self._length_unit()), y_series=converted)

    def _bundle_number_density_y(self, bundle: DerivedPlotBundle, *, source_unit: str) -> DerivedPlotBundle:
        converted = tuple(
            np.asarray(self._display_number_density(np.asarray(series, dtype=np.float64), source_unit=source_unit), dtype=np.float64)
            for series in bundle.y_series
        )
        return _clone_bundle(bundle, y_label=self._replace_bracket_unit(bundle.y_label, self._number_density_unit()), y_series=converted)

    def _populate_shock(self, shock: ShockTrackingResult) -> None:
        propagation = "high index to low index" if shock.propagation_direction == "high_to_low" else "low index to high index"
        self.shock_summary_label.setText(
            f"Method: {shock.method} | direction: {propagation} | "
            f"activation: {shock.activation_snapshot_index if shock.activation_snapshot_index is not None else '-'} | "
            f"breakout: {self._format_time(shock.breakout_time_s)} | "
            "speed curves report both |v| and signed velocity from the smoothed primary-shock trajectory."
        )
        time_values = np.asarray(self._display_time_from_seconds(np.asarray(shock.time_s, dtype=np.float64)), dtype=np.float64)
        position_values = np.asarray(self._display_length_from_cm(np.asarray(shock.smoothed_position_cm, dtype=np.float64)), dtype=np.float64)
        speed_values = np.asarray(self._display_velocity_from_cm_s(np.asarray(shock.speed_magnitude_cm_s, dtype=np.float64)), dtype=np.float64)
        signed_values = np.asarray(self._display_velocity_from_cm_s(np.asarray(shock.velocity_cm_s, dtype=np.float64)), dtype=np.float64)
        self.shock_position_plot.set_curves(
            time_values,
            [position_values],
            title="Shock position vs time",
            x_label=f"Time [{self._time_unit()}]",
            y_label=f"Shock position [{self._length_unit()}]",
            curve_names=["Primary shock"],
            auto_range=True,
            preserve_view=False,
            view_context_key=("derived", "shock", "position"),
        )
        self.shock_velocity_plot.set_curves(
            time_values,
            [
                speed_values,
                signed_values,
            ],
            title="Shock speed vs time",
            x_label=f"Time [{self._time_unit()}]",
            y_label=f"Velocity [{self._velocity_unit()}]",
            curve_names=["Speed magnitude |v|", "Signed shock velocity"],
            auto_range=True,
            preserve_view=False,
            view_context_key=("derived", "shock", "velocity"),
        )
        self.shock_interface_table.setRowCount(len(shock.interface_crossings))
        for row, crossing in enumerate(shock.interface_crossings):
            self.shock_interface_table.setItem(row, 0, QtWidgets.QTableWidgetItem(crossing.interface_label))
            self.shock_interface_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(crossing.boundary_zone)))
            self.shock_interface_table.setItem(row, 2, QtWidgets.QTableWidgetItem(self._format_time(crossing.crossing_time_s)))
            self.shock_interface_table.setItem(row, 3, QtWidgets.QTableWidgetItem(self._format_length(crossing.crossing_position_cm)))
        self.shock_interface_table.resizeColumnsToContents()

    def _populate_preheat(self, preheat: PreheatSummary | None) -> None:
        if preheat is None:
            self.preheat_summary_label.setText(
                "Preheat is lazy-loaded. Open this tab to evaluate target pre-modification relative to the tracked primary compressive branch."
            )
            self.preheat_overview_label.setText(
                "Choose a region of interest here when sandwich targets need a different sample region than the automatic guess."
            )
            self.preheat_plot_panel.clear()
            self.preheat_summary_table.setRowCount(0)
            self.preheat_onset_table.setRowCount(0)
            self.preheat_budget_table.setRowCount(0)
            self.preheat_notes.setPlainText(
                "Preheat stays out of the fast legacy Shock path until this tab is opened. The region selector here overrides only the preheat region of interest."
            )
            self._refresh_preheat_snapshot_view(None)
            return

        self.preheat_summary_label.setText(
            f"Method: {preheat.method} | ROI={preheat.target_label or '-'} | "
            f"mode={'manual override' if str(preheat.target_selection_mode or 'auto') == 'user_selected' else 'auto guess'} | "
            f"entry={self._format_time(preheat.target_entry_time_s)} | severity={preheat.severity_label or '-'} | "
            f"dominant source={preheat.dominant_source or '-'}"
        )
        self.preheat_overview_label.setText(
            " | ".join(
                [
                    f"Region of interest: {preheat.target_label or '-'}",
                    f"Auto guess: {preheat.auto_target_label or '-'}",
                    f"Incident side: {preheat.incident_region_label or '-'}",
                    f"Deepest reached: {preheat.deepest_reached_label or '-'}",
                    f"Primary branch: {preheat.primary_branch_id or '-'} ({_wavefront_support_label(str(preheat.primary_branch_support_class or 'tracked'))})",
                    f"Depth: {self._format_length(preheat.affected_depth_cm)}",
                    f"Thickness fraction: {self._format_fraction(preheat.affected_thickness_fraction)}",
                    f"Areal-mass fraction: {self._format_fraction(preheat.affected_areal_mass_fraction)}",
                ]
            )
        )

        summary_rows: list[tuple[str, str, str, str]] = [
            (
                "Target",
                "Region of interest",
                str(preheat.target_label or "-"),
                f"mode={'manual override' if str(preheat.target_selection_mode or 'auto') == 'user_selected' else 'auto guess'} | primary={preheat.primary_branch_id or '-'}",
            ),
            ("Target", "Auto guess", str(preheat.auto_target_label or "-"), "heuristic main-target guess"),
            ("Target", "Incident side", str(preheat.incident_region_label or "-"), "laser-entry / upstream-side region"),
            ("Target", "Deepest reached", str(preheat.deepest_reached_label or "-"), "deepest region reached by the primary compressive branch"),
            ("Target", "Shock entry", self._format_time(preheat.target_entry_time_s), str(preheat.target_entry_interface_label or "-")),
            ("Summary", "Severity", str(preheat.severity_label or "-"), f"dominant source={preheat.dominant_source or '-'}"),
            ("Summary", "Penalty ratio", self._format_fraction(preheat.preheat_penalty_ratio), "relative to transmitted loading when reliable"),
            ("Extent", "Depth", self._format_length(preheat.affected_depth_cm), "maximum contiguous affected depth before target entry"),
            ("Extent", "Thickness fraction", self._format_fraction(preheat.affected_thickness_fraction), "affected target thickness before shock arrival"),
            ("Extent", "Areal-mass fraction", self._format_fraction(preheat.affected_areal_mass_fraction), "affected target areal mass before shock arrival"),
        ]
        for metric in preheat.state_metrics:
            if metric.unit == "eV":
                value_text = f"rep={self._format_temperature(metric.representative_value)} | max={self._format_temperature(metric.max_value)}"
            elif metric.unit == "J/cm^3":
                value_text = f"rep={self._format_pressure(metric.representative_value)} | max={self._format_pressure(metric.max_value)}"
            elif metric.unit == "J/g":
                value_text = f"rep={self._format_specific_energy(metric.representative_value)} | max={self._format_specific_energy(metric.max_value)}"
            else:
                value_text = f"rep={_format_optional(metric.representative_value, '{:.4g}')} | max={_format_optional(metric.max_value, '{:.4g}')}"
            summary_rows.append(("State", str(metric.label), value_text, "representative and peak pre-shock values"))
        self.preheat_summary_table.setRowCount(len(summary_rows))
        for row_index, row_values in enumerate(summary_rows):
            for column, text in enumerate(row_values):
                self.preheat_summary_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(text))
        self.preheat_summary_table.resizeColumnsToContents()

        onset_rows: list[tuple[str, str, str, str]] = []
        for marker in preheat.onset_markers:
            if marker.unit == "eV":
                observed_text = self._format_temperature(marker.observed_value)
            else:
                observed_text = _format_optional(marker.observed_value, "{:.4g}")
            threshold_text = "-" if marker.threshold_value is None else f"threshold={_format_optional(marker.threshold_value, '{:.3g}')}"
            notes_text = threshold_text if not marker.notes else f"{threshold_text}; {' '.join(marker.notes)}"
            onset_rows.append((str(marker.label), self._format_time(marker.first_time_s), observed_text, notes_text))
        self.preheat_onset_table.setRowCount(len(onset_rows))
        for row_index, row_values in enumerate(onset_rows):
            for column, text in enumerate(row_values):
                self.preheat_onset_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(text))
        self.preheat_onset_table.resizeColumnsToContents()

        budget_rows: list[tuple[str, str, str, str]] = []
        for row in preheat.budget_rows:
            value_text = "-" if row.integrated_value is None else f"{_format_optional(row.integrated_value, '{:.4g}')} {row.unit}"
            budget_rows.append((str(row.label), value_text, self._format_fraction(row.fraction_of_observed), " ".join(row.notes)))
        self.preheat_budget_table.setRowCount(len(budget_rows))
        for row_index, row_values in enumerate(budget_rows):
            for column, text in enumerate(row_values):
                self.preheat_budget_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(text))
        self.preheat_budget_table.resizeColumnsToContents()

        notes: list[str] = [
            f"Available fields: {', '.join(preheat.available_fields) if preheat.available_fields else '-'}",
            f"Missing fields: {', '.join(preheat.missing_fields) if preheat.missing_fields else 'none'}",
            f"Selection: ROI={preheat.target_label or '-'} | auto guess={preheat.auto_target_label or '-'} | incident side={preheat.incident_region_label or '-'} | deepest reached={preheat.deepest_reached_label or '-'}",
        ]
        if preheat.thresholds is not None:
            notes.append(
                "Thresholds: "
                f"rho/rho0 < {preheat.thresholds.max_density_ratio:.3g}, "
                f"(P-P0)/(P0+eps) < {preheat.thresholds.max_relative_pressure:.3g}, "
                f"DeltaTe > {preheat.thresholds.min_delta_temperature_e_ev:.3g} eV, "
                f"DeltaZ > {preheat.thresholds.min_delta_mean_charge:.3g}, "
                f"DeltaEe > {preheat.thresholds.min_delta_electron_energy_j_g:.3g} J/g, "
                f"q_rad > {preheat.thresholds.min_radiation_net_heating_j_g_s:.3g} J/g/s, "
                f"q_laser > {preheat.thresholds.min_laser_deposition_j_g_s:.3g} J/g/s."
            )
        if preheat.notes:
            notes.extend(str(note) for note in preheat.notes)
        if preheat.warnings:
            notes.append("Warnings:")
            notes.extend(f"[{warning.severity}] {warning.source}: {warning.message}" for warning in preheat.warnings)
        self.preheat_notes.setPlainText("\n".join(notes))
        self._refresh_preheat_snapshot_view(preheat)

    def _populate_wavefront(
        self,
        wave_tracking: WaveTrackingResult | None,
        interface_events: InterfaceEventsResult | None,
        preheat: PreheatSummary | None,
    ) -> None:
        if wave_tracking is None:
            self.wavefront_summary_label.setText(
                "WaveFront is lazy-loaded. Open this tab to compute advanced multi-branch wave tracking for the active run, geometry, and selection."
            )
            self.wavefront_overview_label.setText(
                "Advanced WaveFront outputs stay out of the fast legacy Shock path until this tab is opened."
            )
            self.wavefront_plot_empty_label.setText(
                "Open the WaveFront tab to compute advanced multi-branch tracking. Legacy Shock remains available immediately."
            )
            self.wavefront_plot_empty_label.show()
            self.wavefront_plot_splitter.hide()
            self.wavefront_position_plot.clear_plot()
            self.wavefront_score_plot.clear_plot()
            self.wavefront_branch_table.setRowCount(0)
            self.wavefront_event_table.setRowCount(0)
            self.wavefront_event_table.show()
            self.wavefront_notes.setPlainText(
                "Advanced outputs stay out of the fast legacy Shock path until this tab is opened."
            )
            return

        current_result = self._current_result
        time_axis_s = (
            np.asarray(wave_tracking.evidence_maps[0].time_s, dtype=np.float64)
            if wave_tracking.evidence_maps
            else (np.asarray(current_result.shock.time_s, dtype=np.float64) if current_result is not None else np.asarray([], dtype=np.float64))
        )
        display_time = np.asarray(self._display_time_from_seconds(time_axis_s), dtype=np.float64)
        n_times = int(display_time.size)
        all_branches = list(wave_tracking.branches)
        branch_lookup = {str(branch.branch_id): branch for branch in all_branches}
        plotted_branches, table_branches, branch_scope_note = self._wavefront_visible_branches(tuple(all_branches))
        plot_limit = 6
        plotted_branches = plotted_branches[:plot_limit]
        branch_types = sorted({str(branch.branch_type) for branch in all_branches})
        ambiguous_count = sum(1 for branch in all_branches if bool(branch.ambiguous))
        primary_branch = next((branch for branch in all_branches if bool(branch.primary)), None)
        primary_compressive_branch = self._primary_compressive_branch(tuple(all_branches))
        all_events = () if interface_events is None else tuple(interface_events.events)
        visible_branch_ids = {str(branch.branch_id) for branch in table_branches}
        visible_events, event_scope_note = self._wavefront_visible_events(
            all_events,
            branch_lookup=branch_lookup,
            visible_branch_ids=visible_branch_ids,
        )
        event_count = len(visible_events)
        display_mode = self._wavefront_display_mode()
        self._set_wavefront_lower_layout_mode(display_mode)
        self.wavefront_event_table.setVisible(display_mode == "events")

        def _branch_curve(branch: WaveBranchSummary, values: np.ndarray, *, transform=None) -> np.ndarray:
            series = np.full(n_times, np.nan, dtype=np.float64)
            if n_times <= 0:
                return series
            snapshot_indices = np.asarray(branch.snapshot_indices, dtype=np.int32)
            if snapshot_indices.size == 0:
                return series
            valid = (snapshot_indices >= 0) & (snapshot_indices < n_times)
            if not np.any(valid):
                return series
            branch_values = np.asarray(values, dtype=np.float64)[valid]
            if transform is not None:
                branch_values = np.asarray(transform(branch_values), dtype=np.float64)
            series[snapshot_indices[valid]] = branch_values
            return series

        def _legacy_series(values: np.ndarray, *, transform=None) -> np.ndarray:
            series = np.asarray(values, dtype=np.float64)
            if transform is not None:
                series = np.asarray(transform(series), dtype=np.float64)
            return series

        def _primary_legacy_comparison(primary: WaveBranchSummary | None) -> str | None:
            if primary is None or current_result is None:
                return None
            legacy = current_result.shock
            primary_position = _branch_curve(primary, np.asarray(primary.position_cm, dtype=np.float64))
            legacy_position = np.asarray(legacy.smoothed_position_cm, dtype=np.float64)
            primary_speed = _branch_curve(primary, np.asarray(primary.velocity_cm_s, dtype=np.float64))
            legacy_speed = np.asarray(legacy.velocity_cm_s, dtype=np.float64)

            def _median_delta(left: np.ndarray, right: np.ndarray, *, transform=None) -> str:
                if transform is not None:
                    left = np.asarray(transform(left), dtype=np.float64)
                    right = np.asarray(transform(right), dtype=np.float64)
                finite = np.isfinite(left) & np.isfinite(right)
                if not np.any(finite):
                    return "-"
                return f"{float(np.nanmedian(np.abs(left[finite] - right[finite]))):.3g}"

            position_delta = _median_delta(primary_position, legacy_position, transform=self._display_length_from_cm)
            speed_delta = _median_delta(primary_speed, legacy_speed, transform=self._display_velocity_from_cm_s)
            finite_speed = np.isfinite(primary_speed) & np.isfinite(legacy_speed) & (np.abs(primary_speed) > 0.0) & (np.abs(legacy_speed) > 0.0)
            sign_match = (
                "-"
                if not np.any(finite_speed)
                else f"{100.0 * float(np.mean(np.sign(primary_speed[finite_speed]) == np.sign(legacy_speed[finite_speed]))):.0f}%"
            )
            return (
                f"Legacy comparison: median |Δx|={position_delta} {self._length_unit()} | "
                f"median |Δv|={speed_delta} {self._velocity_unit()} | sign match={sign_match}. "
                "WaveFront uses the multi-branch primary compressive track; legacy Shock uses the single-front quick look."
            )

        def _set_graphs(
            primary_series: list[np.ndarray],
            secondary_series: list[np.ndarray],
            curve_names: list[str],
            *,
            primary_title: str,
            primary_y_label: str,
            secondary_title: str,
            secondary_y_label: str,
            primary_view_key: str,
            secondary_view_key: str,
            empty_message: str,
        ) -> None:
            if primary_series:
                self.wavefront_plot_empty_label.hide()
                self.wavefront_plot_splitter.show()
                self.wavefront_position_plot.set_curves(
                    display_time,
                    primary_series,
                    title=primary_title,
                    x_label=f"Time [{self._time_unit()}]",
                    y_label=primary_y_label,
                    curve_names=curve_names,
                    auto_range=True,
                    preserve_view=False,
                    view_context_key=("derived", "wavefront", primary_view_key),
                )
                self.wavefront_score_plot.set_curves(
                    display_time,
                    secondary_series,
                    title=secondary_title,
                    x_label=f"Time [{self._time_unit()}]",
                    y_label=secondary_y_label,
                    curve_names=curve_names,
                    auto_range=True,
                    preserve_view=False,
                    view_context_key=("derived", "wavefront", secondary_view_key),
                )
                return
            self.wavefront_position_plot.clear_plot()
            self.wavefront_score_plot.clear_plot()
            self.wavefront_plot_splitter.hide()
            self.wavefront_plot_empty_label.setText(empty_message)
            self.wavefront_plot_empty_label.show()

        curve_names = [
            f"{branch.branch_id} | {branch.branch_type} | {_wavefront_support_label(branch.support_class)}"
            for branch in plotted_branches
        ]
        if display_mode == "primary_position":
            if primary_compressive_branch is None:
                self.wavefront_position_plot.clear_plot()
                self.wavefront_score_plot.clear_plot()
                self.wavefront_plot_splitter.hide()
                self.wavefront_plot_empty_label.setText(
                    "No reliable non-provisional compressive branch is available, so the advanced primary-branch position view is unavailable."
                )
                self.wavefront_plot_empty_label.show()
            else:
                primary_position = _branch_curve(
                    primary_compressive_branch,
                    np.asarray(primary_compressive_branch.position_cm, dtype=np.float64),
                    transform=self._display_length_from_cm,
                )
                primary_speed = _branch_curve(
                    primary_compressive_branch,
                    np.asarray(primary_compressive_branch.velocity_cm_s, dtype=np.float64),
                    transform=self._display_velocity_from_cm_s,
                )
                curve_names = [f"WaveFront primary: {primary_compressive_branch.branch_id}"]
                primary_series = [primary_position]
                secondary_series = [primary_speed]
                secondary_names = [f"{curve_names[0]} signed speed"]
                if current_result is not None:
                    primary_series.append(_legacy_series(np.asarray(current_result.shock.smoothed_position_cm, dtype=np.float64), transform=self._display_length_from_cm))
                    curve_names.append("Legacy Shock")
                    secondary_series.append(_legacy_series(np.asarray(current_result.shock.velocity_cm_s, dtype=np.float64), transform=self._display_velocity_from_cm_s))
                    secondary_names = [f"{curve_names[0]} signed speed", "Legacy Shock signed speed"]
                self.wavefront_plot_empty_label.hide()
                self.wavefront_plot_splitter.show()
                self.wavefront_position_plot.set_curves(
                    display_time,
                    primary_series,
                    title="WaveFront primary compressive branch position vs time",
                    x_label=f"Time [{self._time_unit()}]",
                    y_label=f"Position [{self._length_unit()}]",
                    curve_names=curve_names,
                    auto_range=True,
                    preserve_view=False,
                    view_context_key=("derived", "wavefront", "primary_position"),
                )
                self.wavefront_score_plot.set_curves(
                    display_time,
                    secondary_series,
                    title="Primary compressive branch signed speed comparison",
                    x_label=f"Time [{self._time_unit()}]",
                    y_label=f"Velocity [{self._velocity_unit()}]",
                    curve_names=secondary_names,
                    auto_range=True,
                    preserve_view=False,
                    view_context_key=("derived", "wavefront", "primary_position_support"),
                )
        elif display_mode == "primary_speed":
            if primary_compressive_branch is None:
                self.wavefront_position_plot.clear_plot()
                self.wavefront_score_plot.clear_plot()
                self.wavefront_plot_splitter.hide()
                self.wavefront_plot_empty_label.setText(
                    "No reliable non-provisional compressive branch is available, so the advanced primary-branch speed view is unavailable."
                )
                self.wavefront_plot_empty_label.show()
            else:
                primary_signed_speed = _branch_curve(
                    primary_compressive_branch,
                    np.asarray(primary_compressive_branch.velocity_cm_s, dtype=np.float64),
                    transform=self._display_velocity_from_cm_s,
                )
                primary_abs_speed = _branch_curve(
                    primary_compressive_branch,
                    np.abs(np.asarray(primary_compressive_branch.velocity_cm_s, dtype=np.float64)),
                    transform=self._display_velocity_from_cm_s,
                )
                curve_names = [f"WaveFront primary: {primary_compressive_branch.branch_id}"]
                primary_series = [primary_signed_speed]
                secondary_series = [primary_abs_speed]
                secondary_names = [f"{curve_names[0]} |speed|"]
                if current_result is not None:
                    primary_series.append(_legacy_series(np.asarray(current_result.shock.velocity_cm_s, dtype=np.float64), transform=self._display_velocity_from_cm_s))
                    curve_names.append("Legacy Shock signed speed")
                    secondary_series.append(_legacy_series(np.asarray(current_result.shock.speed_magnitude_cm_s, dtype=np.float64), transform=self._display_velocity_from_cm_s))
                    secondary_names.append("Legacy Shock |speed|")
                self.wavefront_plot_empty_label.hide()
                self.wavefront_plot_splitter.show()
                self.wavefront_position_plot.set_curves(
                    display_time,
                    primary_series,
                    title="WaveFront primary compressive branch signed speed vs time",
                    x_label=f"Time [{self._time_unit()}]",
                    y_label=f"Velocity [{self._velocity_unit()}]",
                    curve_names=curve_names,
                    auto_range=True,
                    preserve_view=False,
                    view_context_key=("derived", "wavefront", "primary_speed"),
                )
                self.wavefront_score_plot.set_curves(
                    display_time,
                    secondary_series,
                    title="Primary compressive branch speed magnitude",
                    x_label=f"Time [{self._time_unit()}]",
                    y_label=f"Velocity [{self._velocity_unit()}]",
                    curve_names=secondary_names,
                    auto_range=True,
                    preserve_view=False,
                    view_context_key=("derived", "wavefront", "primary_speed_magnitude"),
                )
        elif display_mode == "position":
            _set_graphs(
                [
                    _branch_curve(branch, np.asarray(branch.position_cm, dtype=np.float64), transform=self._display_length_from_cm)
                    for branch in plotted_branches
                ],
                [_branch_curve(branch, np.asarray(branch.score, dtype=np.float64)) for branch in plotted_branches],
                curve_names,
                primary_title="WaveFront branch positions vs time",
                primary_y_label=f"Position [{self._length_unit()}]",
                secondary_title="WaveFront branch evidence vs time",
                secondary_y_label="Evidence score",
                primary_view_key="position",
                secondary_view_key="evidence",
                empty_message="No tracked WaveFront branches are available to plot in the current scope.",
            )
        elif display_mode == "evidence":
            _set_graphs(
                [_branch_curve(branch, np.asarray(branch.score, dtype=np.float64)) for branch in plotted_branches],
                [
                    _branch_curve(branch, np.asarray(branch.position_cm, dtype=np.float64), transform=self._display_length_from_cm)
                    for branch in plotted_branches
                ],
                curve_names,
                primary_title="WaveFront branch evidence vs time",
                primary_y_label="Evidence score",
                secondary_title="WaveFront branch positions vs time",
                secondary_y_label=f"Position [{self._length_unit()}]",
                primary_view_key="evidence",
                secondary_view_key="position",
                empty_message="No tracked WaveFront branches are available to plot evidence traces in the current scope.",
            )
        elif display_mode == "speed":
            _set_graphs(
                [
                    _branch_curve(branch, np.asarray(branch.velocity_cm_s, dtype=np.float64), transform=self._display_velocity_from_cm_s)
                    for branch in plotted_branches
                ],
                [
                    _branch_curve(branch, np.asarray(branch.width_cm, dtype=np.float64), transform=self._display_length_from_cm)
                    if branch.width_cm is not None
                    else np.full(n_times, np.nan, dtype=np.float64)
                    for branch in plotted_branches
                ],
                curve_names,
                primary_title="WaveFront branch speed vs time",
                primary_y_label=f"Velocity [{self._velocity_unit()}]",
                secondary_title="WaveFront branch width / thickness vs time",
                secondary_y_label=f"Width [{self._length_unit()}]",
                primary_view_key="speed",
                secondary_view_key="width",
                empty_message="No tracked WaveFront branches are available to plot speed traces in the current scope.",
            )
        elif display_mode == "width":
            _set_graphs(
                [
                    _branch_curve(branch, np.asarray(branch.width_cm, dtype=np.float64), transform=self._display_length_from_cm)
                    if branch.width_cm is not None
                    else np.full(n_times, np.nan, dtype=np.float64)
                    for branch in plotted_branches
                ],
                [
                    _branch_curve(branch, np.asarray(branch.velocity_cm_s, dtype=np.float64), transform=self._display_velocity_from_cm_s)
                    for branch in plotted_branches
                ],
                curve_names,
                primary_title="WaveFront branch width / thickness vs time",
                primary_y_label=f"Width [{self._length_unit()}]",
                secondary_title="WaveFront branch speed vs time",
                secondary_y_label=f"Velocity [{self._velocity_unit()}]",
                primary_view_key="width",
                secondary_view_key="speed",
                empty_message="No tracked WaveFront branches are available to plot width traces in the current scope.",
            )
        elif display_mode == "events":
            self.wavefront_position_plot.clear_plot()
            self.wavefront_score_plot.clear_plot()
            self.wavefront_plot_splitter.hide()
            if visible_events:
                self.wavefront_plot_empty_label.setText(
                    "Interface-event summary view is active. Use the table below for classification, support, impulse, transfer fractions, and ambiguity flags."
                )
            elif interface_events is not None and interface_events.suppressed_event_count > 0:
                self.wavefront_plot_empty_label.setText(
                    "No default interface events are visible because only provisional crossings were found. Use notes below for the suppression summary."
                )
            else:
                self.wavefront_plot_empty_label.setText(
                    "No meaningful interface events were available for the current branch-support scope."
                )
            self.wavefront_plot_empty_label.show()
        elif display_mode == "warnings":
            self.wavefront_position_plot.clear_plot()
            self.wavefront_score_plot.clear_plot()
            self.wavefront_plot_splitter.hide()
            self.wavefront_plot_empty_label.setText(
                "Warnings / suppressed detections view is active. Use the notes pane below for ambiguity, suppression, and compatibility details."
            )
            self.wavefront_plot_empty_label.show()
        else:
            self.wavefront_position_plot.clear_plot()
            self.wavefront_score_plot.clear_plot()
            self.wavefront_plot_splitter.hide()
            self.wavefront_plot_empty_label.setText(
                "Significance / support ranking is a summary-oriented WaveFront view. Use the ranked branch table below."
            )
            self.wavefront_plot_empty_label.show()

        self.wavefront_summary_label.setText(
            f"Method: {wave_tracking.method} | kept={len(all_branches)} | tracked={wave_tracking.tracked_branch_count} | "
            f"short/weak={wave_tracking.short_branch_count} | provisional={wave_tracking.provisional_branch_count} | "
            f"suppressed={wave_tracking.suppressed_branch_count} | types={', '.join(branch_types) or '-'} | "
            f"primary={('-' if primary_branch is None else primary_branch.branch_type)} | "
            f"primary compressive={('-' if primary_compressive_branch is None else primary_compressive_branch.branch_id + '/' + primary_compressive_branch.branch_type)} | "
            f"interface events={event_count}"
            f"{'' if interface_events is None else f' (tracked={interface_events.tracked_event_count}, weak={interface_events.weak_event_count}, suppressed={interface_events.suppressed_event_count})'}"
            f" | ambiguous={ambiguous_count}"
        )
        top_branch = table_branches[0] if table_branches else primary_branch
        overview_parts = [
            f"View: {self.wavefront_display_combo.currentText()}",
            f"Branch set: {self.wavefront_scope_combo.currentText()}",
            f"Direction: {self.wavefront_direction_combo.currentText()}",
        ]
        if primary_compressive_branch is not None:
            overview_parts.append(
                f"Primary compressive: {primary_compressive_branch.branch_id} ({primary_compressive_branch.branch_type}, {_wavefront_support_label(primary_compressive_branch.support_class)})"
            )
        else:
            overview_parts.append("Primary compressive: unavailable")
        if top_branch is not None:
            overview_parts.append(
                f"Top branch: {top_branch.branch_id} ({top_branch.branch_type}, {_wavefront_support_label(top_branch.support_class)}, significance={_format_optional(top_branch.significance, '{:.3f}')})"
            )
        if plotted_branches:
            plotted_summary = f"Plotting top {len(plotted_branches)}"
            if len(table_branches) > len(plotted_branches):
                plotted_summary += f" of {len(table_branches)} visible branches"
            plotted_summary += " by significance."
            overview_parts.append(plotted_summary)
        if branch_scope_note:
            overview_parts.append(branch_scope_note)
        if event_scope_note:
            overview_parts.append(event_scope_note)
        if wave_tracking.supported_formula_hooks:
            overview_parts.append(
                "Evidence families: " + ", ".join(str(hook.family).replace("_", "-") for hook in wave_tracking.supported_formula_hooks)
            )
        if interface_events is not None and interface_events.classification_counts:
            overview_parts.append(
                "Event classes: "
                + ", ".join(f"{name.replace('_', ' ')}={count}" for name, count in interface_events.classification_counts)
            )
        if preheat is not None:
            overview_parts.append(
                "Preheat: "
                + f"ROI={preheat.target_label or '-'} ({'manual' if str(preheat.target_selection_mode or 'auto') == 'user_selected' else 'auto'}), "
                + f"auto={preheat.auto_target_label or '-'}, "
                + f"entry={self._format_time(preheat.target_entry_time_s)}, "
                + f"severity={preheat.severity_label or '-'}"
            )
        self.wavefront_overview_label.setText(" | ".join(overview_parts))

        self.wavefront_branch_table.setRowCount(len(table_branches))
        for row, branch in enumerate(table_branches):
            values = (
                str(branch.branch_id),
                str(branch.branch_type),
                _wavefront_support_label(branch.support_class),
                _format_optional(branch.significance, "{:.3f}"),
                str(int(branch.sample_count or np.asarray(branch.snapshot_indices, dtype=np.int32).size)),
                self._format_time(branch.duration_s),
                _format_optional(branch.integrated_score, "{:.3f}"),
                _format_optional(branch.confidence, "{:.3f}"),
                ("yes" if bool(branch.ambiguous) else "no"),
                str(branch.propagation_direction or "-"),
                self._format_time(branch.breakout_time_s),
            )
            for column, text in enumerate(values):
                self.wavefront_branch_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        self.wavefront_branch_table.resizeColumnsToContents()

        self.wavefront_event_table.setRowCount(len(visible_events))
        for row, event in enumerate(visible_events):
            tx_rx = "-".join(
                part
                for part in (
                    None if event.transmitted_branch_id is None else f"T:{event.transmitted_branch_id}",
                    None if event.reflected_branch_id is None else f"R:{event.reflected_branch_id}",
                )
                if part
            ) or "-"
            values = (
                str(event.event_kind),
                str(event.interface_label),
                self._format_time(event.time_s),
                str(event.event_classification or "-").replace("_", " "),
                _wavefront_support_label(str(event.support_class or "tracked")),
                _format_optional(event.significance, "{:.3f}"),
                _format_optional(event.confidence, "{:.3f}"),
                str(event.branch_id or "-"),
                tx_rx,
                self._format_impulse(event.pressure_impulse_upstream_j_s_cm3),
                self._format_fraction(event.transfer_fraction),
                self._format_fraction(event.reflection_fraction),
                str(event.dominant_transfer_channel or "-"),
                ("yes" if bool(event.ambiguous) else "no"),
            )
            for column, text in enumerate(values):
                self.wavefront_event_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        self.wavefront_event_table.resizeColumnsToContents()

        notes: list[str] = []
        if wave_tracking.compatibility_source:
            notes.append(f"Compatibility source: {wave_tracking.compatibility_source}")
        notes.append(
            f"Branch support totals: tracked={wave_tracking.tracked_branch_count}, short/weak={wave_tracking.short_branch_count}, provisional={wave_tracking.provisional_branch_count}, suppressed={wave_tracking.suppressed_branch_count}."
        )
        if top_branch is not None:
            notes.append(
                f"Primary branch summary: {top_branch.branch_id} | type={top_branch.branch_type} | support={_wavefront_support_label(top_branch.support_class)} | significance={_format_optional(top_branch.significance, '{:.3f}')} | samples={top_branch.sample_count} | duration={self._format_time(top_branch.duration_s)} | span={self._format_length(top_branch.position_span_cm)}."
            )
            notes.append(
                f"Primary branch local states: upstream rho={self._format_density(None if top_branch.upstream_state is None else top_branch.upstream_state.density_g_cm3)} / downstream rho={self._format_density(None if top_branch.downstream_state is None else top_branch.downstream_state.density_g_cm3)} | upstream u={self._format_velocity(None if top_branch.upstream_state is None else top_branch.upstream_state.velocity_cm_s)} / downstream u={self._format_velocity(None if top_branch.downstream_state is None else top_branch.downstream_state.velocity_cm_s)}."
            )
        if primary_compressive_branch is not None:
            notes.append(
                f"Primary compressive branch: {primary_compressive_branch.branch_id} | type={primary_compressive_branch.branch_type} | support={_wavefront_support_label(primary_compressive_branch.support_class)} | direction={primary_compressive_branch.propagation_direction or '-'} | significance={_format_optional(primary_compressive_branch.significance, '{:.3f}')}"
            )
            comparison_note = _primary_legacy_comparison(primary_compressive_branch)
            if comparison_note is not None:
                notes.append(comparison_note)
        else:
            notes.append("Primary compressive branch: unavailable because no non-provisional compressive branch met the current result quality threshold.")
        if interface_events is not None and interface_events.notes:
            notes.extend(str(note) for note in interface_events.notes)
        if preheat is not None:
            notes.append(
                f"Preheat is available in the separate Preheat tab: ROI={preheat.target_label or '-'} | entry={self._format_time(preheat.target_entry_time_s)} | severity={preheat.severity_label or '-'}."
            )
        if visible_events:
            notes.append("Top interface events:")
            notes.extend(
                (
                    f"{index + 1}. {event.interface_label} | {str(event.event_classification or '-').replace('_', ' ')}"
                    f" | incident={event.branch_id or '-'} | support={_wavefront_support_label(str(event.support_class or 'tracked'))}"
                    f" | significance={_format_optional(event.significance, '{:.3f}')}"
                    f" | impulse={self._format_impulse(event.pressure_impulse_upstream_j_s_cm3)}"
                    f" | T_E={self._format_fraction(event.transfer_fraction)} | R_E={self._format_fraction(event.reflection_fraction)}"
                    f" | channel={event.dominant_transfer_channel or '-'}"
                )
                for index, event in enumerate(visible_events[:5])
            )
            detailed_event = visible_events[0]
            notes.append(
                f"Lead event state summary: t={self._format_time(detailed_event.time_s)} | x={self._format_length(detailed_event.position_cm)}"
                f" | incident Ppeak={_format_optional(detailed_event.incident_peak_pressure_j_cm3, '{:.3g}')} J/cm^3"
                f" | tx Ppeak={_format_optional(detailed_event.transmitted_peak_pressure_j_cm3, '{:.3g}')} J/cm^3"
                f" | refl Ppeak={_format_optional(detailed_event.reflected_peak_pressure_j_cm3, '{:.3g}')} J/cm^3"
                f" | rho ratio in/out={_format_optional(detailed_event.incident_compression_ratio, '{:.3g}')}/{_format_optional(detailed_event.transmitted_compression_ratio, '{:.3g}')}"
            )
            if detailed_event.impedance_preview_supported:
                notes.append(
                    f"Lead event impedance preview: Z1={_format_optional(detailed_event.impedance_upstream, '{:.3g}')} | "
                    f"Z2={_format_optional(detailed_event.impedance_downstream, '{:.3g}')} | "
                    f"R_I={_format_optional(detailed_event.impedance_reflection_preview, '{:.3f}')} | "
                    f"T_I={_format_optional(detailed_event.impedance_transmission_preview, '{:.3f}')}"
                )
        if table_branches:
            notes.append("Top branch ranking:")
            notes.extend(
                f"{index + 1}. {branch.branch_id} | {branch.branch_type} | {_wavefront_support_label(branch.support_class)} | significance={_format_optional(branch.significance, '{:.3f}')} | samples={branch.sample_count}"
                for index, branch in enumerate(table_branches[:5])
            )
        branch_notes = [f"{branch.branch_id}: {' '.join(branch.notes)}" for branch in all_branches if branch.notes]
        notes.extend(branch_notes[:8])
        warnings = [f"[{warning.severity}] {warning.source}: {warning.message}" for warning in wave_tracking.warnings]
        if interface_events is not None:
            warnings.extend(f"[{warning.severity}] {warning.source}: {warning.message}" for warning in interface_events.warnings)
        if preheat is not None:
            warnings.extend(f"[{warning.severity}] {warning.source}: {warning.message}" for warning in preheat.warnings)
        if warnings:
            notes.append("Warnings:")
            notes.extend(warnings)
        self.wavefront_notes.setPlainText("\n".join(notes) if notes else "No additional WaveFront warnings or notes.")

    def _populate_xrd(self, xrd: XrdResult) -> None:
        display_mode = self._xrd_display_mode()
        display_label = {
            "degrees": f"Bragg shift [{_angle_unit_label(self._angle_unit())}]",
            "q": "Q [1/A]",
        }.get(display_mode, f"Bragg shift [{_angle_unit_label(self._angle_unit())}]")
        time_bundles: list[DerivedPlotBundle] = []
        profile_bundles: list[DerivedPlotBundle] = []
        for bundle in xrd.time_plots:
            converted = self._bundle_time_x(bundle)
            if bundle.key == "density":
                converted = self._bundle_density_y(converted)
            elif bundle.key == "thickness":
                converted = self._bundle_length_y(converted)
            elif bundle.key == "bragg_shift":
                converted = _clone_bundle(
                    converted,
                    y_label=self._replace_bracket_unit(bundle.y_label, _angle_unit_label(self._angle_unit())),
                    y_series=tuple(
                        np.asarray(self._display_angle_from_deg(np.asarray(series, dtype=np.float64)), dtype=np.float64)
                        for series in bundle.y_series
                    ),
                )
            time_bundles.append(converted)
        for bundle in xrd.profile_plots:
            converted = bundle
            if "[um]" in bundle.x_label:
                converted = self._bundle_length_x(converted)
            if bundle.key == "density_profile":
                converted = self._bundle_density_y(converted)
            elif bundle.key == "thickness_profile":
                converted = self._bundle_length_y(converted)
            elif bundle.key == "bragg_shift_profile":
                converted = _clone_bundle(
                    converted,
                    y_label=self._replace_bracket_unit(bundle.y_label, _angle_unit_label(self._angle_unit())),
                    y_series=tuple(
                        np.asarray(self._display_angle_from_deg(np.asarray(series, dtype=np.float64)), dtype=np.float64)
                        for series in bundle.y_series
                    ),
                )
            profile_bundles.append(converted)
        self.xrd_summary_label.setText(
            f"Effective isotropic-compression quick look | snapshot {xrd.snapshot_index} @ {self._format_time(self._current_result.snapshot_time_s if self._current_result is not None else None)} | "
            f"{len(xrd.layers)} active regions | probe={self._format_photon_value_from_kev(xrd.photon_energy_kev, '{:.4g}')} | "
            f"lambda={xrd.wavelength_angstrom * 0.1:.4f} nm | theta0={self._format_angle(xrd.initial_bragg_angle_deg, '{:.4g}')} | "
            f"weighting={xrd.weighting_mode} | {xrd.geometry_summary} | display={display_label}"
        )
        self.xrd_plot_panel.set_bundles(
            tuple(time_bundles),
            self._with_snapshot_titles(tuple(profile_bundles), snapshot_index=xrd.snapshot_index),
            view_scope="xrd",
            preferred_time_key=("bragg_shift" if display_mode == "degrees" else "q_compressed"),
        )
        self.xrd_table.setRowCount(len(xrd.layers))
        if display_mode == "q":
            self.xrd_table.setHorizontalHeaderLabels(["Region", f"rho [{self._density_unit()}]", "rho/rho0", "d/d0", "Q0", "Q", "Delta Q", f"Thickness [{self._length_unit()}]"])
        else:
            angle_label = _angle_unit_label(self._angle_unit())
            self.xrd_table.setHorizontalHeaderLabels(["Region", f"rho [{self._density_unit()}]", "rho/rho0", "d/d0", f"theta0 [{angle_label}]", f"theta [{angle_label}]", f"Shift [{angle_label}]", f"Thickness [{self._length_unit()}]"])
        for row, layer in enumerate(xrd.layers):
            if display_mode == "q":
                values = (
                    str(layer.region_id),
                    _format_optional(float(self._display_density_from_g_cm3(layer.compressed_density_g_cm3)), "{:.4g}"),
                    _format_optional(layer.compression_ratio, "{:.4g}"),
                    _format_optional(layer.d_over_d0, "{:.4g}"),
                    _format_optional(layer.q0_inv_angstrom, "{:.4g}"),
                    _format_optional(layer.q_compressed_inv_angstrom, "{:.4g}"),
                    _format_optional(layer.q_compressed_inv_angstrom - layer.q0_inv_angstrom, "{:.4g}"),
                    _format_optional(float(self._display_length_from_cm(layer.compressed_thickness_cm)), "{:.4g}"),
                )
            else:
                values = (
                    str(layer.region_id),
                    _format_optional(float(self._display_density_from_g_cm3(layer.compressed_density_g_cm3)), "{:.4g}"),
                    _format_optional(layer.compression_ratio, "{:.4g}"),
                    _format_optional(layer.d_over_d0, "{:.4g}"),
                    _format_optional(float(self._display_angle_from_deg(layer.initial_bragg_angle_deg)), "{:.4g}"),
                    _format_optional(None if layer.shifted_bragg_angle_deg is None else float(self._display_angle_from_deg(layer.shifted_bragg_angle_deg)), "{:.4g}"),
                    _format_optional(None if layer.bragg_shift_deg is None else float(self._display_angle_from_deg(layer.bragg_shift_deg)), "{:.4g}"),
                    _format_optional(float(self._display_length_from_cm(layer.compressed_thickness_cm)), "{:.4g}"),
                )
            for column, text in enumerate(values):
                self.xrd_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        self.xrd_table.resizeColumnsToContents()

    def _populate_plasmon(self, plasmon: PlasmonResult) -> None:
        if not self._plasmon_recompute_pending and not self._plasmon_controls_dirty and not self._plasmon_request_control_has_active_input():
            # When no manual plasmon override is pending, the freshly delivered
            # result is the authoritative UI state, including benchmark preset
            # and electron-policy metadata. Pending manual edits still suppress
            # this sync so stale results do not overwrite current user intent.
            self._sync_plasmon_controls_from_result(plasmon)
        request_matches = self._plasmon_request_matches_result(plasmon)
        if request_matches:
            self._plasmon_recompute_pending = False
            self._plasmon_controls_dirty = False
        self._update_plasmon_recompute_ui()
        time_bundles: list[DerivedPlotBundle] = []
        for bundle in plasmon.time_plots:
            converted = self._bundle_time_x(bundle)
            if "Temperature [" in bundle.y_label:
                converted = self._bundle_temperature_y(converted)
            elif "Electron density [" in bundle.y_label:
                converted = self._bundle_number_density_y(converted, source_unit="1/cm3")
            elif "Debye length [" in bundle.y_label:
                converted = self._bundle_length_y(converted)
            time_bundles.append(converted)
        profile_bundles: list[DerivedPlotBundle] = []
        for bundle in plasmon.profile_plots:
            converted = bundle
            if "[um]" in bundle.x_label:
                converted = self._bundle_length_x(converted)
            elif "[deg]" in bundle.x_label:
                converted = self._bundle_angle_x(converted)
            if "Temperature [" in bundle.y_label:
                converted = self._bundle_temperature_y(converted)
            elif "Electron density [" in bundle.y_label:
                converted = self._bundle_number_density_y(converted, source_unit="1/m3")
            profile_bundles.append(converted)
        study_text = "Dispersion scan" if str(getattr(plasmon, "study_mode", PLASMON_STUDY_MODE_SPECTRUM)) == PLASMON_STUDY_MODE_DISPERSION else "Fixed-angle / k spectrum"
        fixed_axis = str(getattr(plasmon, "coordinate_axis", PLASMON_AXIS_ANGLE_DEG))
        scan_axis = str(getattr(plasmon, "scan_axis", PLASMON_AXIS_ANGLE_DEG))
        fixed_value = float(getattr(plasmon, "coordinate_value", plasmon.scattering_angle_deg))
        fixed_value_text = (
            f"{_format_optional(fixed_value, '{:.4g}')} 1/A"
            if fixed_axis == PLASMON_AXIS_K_ANGSTROM_INV
            else self._format_angle(fixed_value, "{:.4g}")
        )
        scan_range_text = (
            f"{_format_optional(float(getattr(plasmon, 'scan_start', 10.0)), '{:.4g}')} to {_format_optional(float(getattr(plasmon, 'scan_stop', 140.0)), '{:.4g}')} 1/A"
            if scan_axis == PLASMON_AXIS_K_ANGSTROM_INV
            else f"{self._format_angle(float(getattr(plasmon, 'scan_start', 10.0)), '{:.4g}')} to {self._format_angle(float(getattr(plasmon, 'scan_stop', 140.0)), '{:.4g}')}"
        )
        model_note = f" | model={plasmon.model_name}" if str(plasmon.model_name) != PLASMON_MODEL_QUICKLOOK else ""
        requested_note = ""
        if str(plasmon.requested_model_name) != str(plasmon.model_name):
            requested_note = f" | requested={plasmon.requested_model_name}"
        mode_note = f" | exec={getattr(plasmon, 'execution_mode', 'quicklook')}"
        benchmark_note = ""
        if str(plasmon.benchmark_status or "not_applicable") != "not_applicable":
            benchmark_note = f" | benchmark={plasmon.benchmark_status}"
            if not bool(plasmon.model_executed_fully):
                benchmark_note += " | incomplete execution"
            if float(plasmon.fallback_fraction or 0.0) > 0.0:
                benchmark_note += f" | fallback={100.0 * float(plasmon.fallback_fraction):.1f}%"
        preset_note = ""
        if str(getattr(plasmon, "benchmark_preset", PLASMON_BENCHMARK_PRESET_NONE)) != PLASMON_BENCHMARK_PRESET_NONE:
            preset_note = f" | preset={getattr(plasmon, 'benchmark_preset', PLASMON_BENCHMARK_PRESET_NONE)}"
        pending_note = " | pending recompute" if not request_matches and self._selected_plasmon_model() != PLASMON_MODEL_QUICKLOOK else ""
        self.plasmon_summary_label.setText(
            f"Snapshot {plasmon.snapshot_index} @ {self._format_time(self._current_result.snapshot_time_s if self._current_result is not None else None)} | "
            f"{study_text} | Regime: {plasmon.regime_label} | weighting={plasmon.weighting_mode}{model_note}{requested_note}{mode_note}{benchmark_note}{preset_note}{pending_note} | "
            f"runtime={float(getattr(plasmon, 'total_runtime_s', 0.0)):.2f}s | {plasmon.geometry_summary}"
        )
        self._set_text_browser_text_preserving_scroll(
            self.plasmon_metrics,
            "\n".join(
                (
                    f"Study         {study_text}",
                    f"Fixed input   {fixed_value_text} ({'k' if fixed_axis == PLASMON_AXIS_K_ANGSTROM_INV else 'angle'})",
                    f"Scan axis     {'k [1/A]' if scan_axis == PLASMON_AXIS_K_ANGSTROM_INV else 'angle'}",
                    f"Scan range    {scan_range_text}",
                    f"Scan points   {int(getattr(plasmon, 'scan_points', 61))}",
                    f"Compare all   {'yes' if bool(getattr(plasmon, 'compare_models', False)) else 'no'}",
                    f"Policy cmp    {'yes' if bool(getattr(plasmon, 'compare_policies', False)) else 'no'}",
                    f"Te            {self._format_temperature(plasmon.electron_temperature_ev, '{:.4g}')}",
                    f"Ti            {self._format_temperature(plasmon.ion_temperature_ev, '{:.4g}')}",
                    f"ne            {self._format_number_density(plasmon.electron_density_cm3, source_unit='1/cm3', fmt='{:.4g}')}",
                    f"Zbar          {_format_optional(plasmon.mean_charge, '{:.4g}')}",
                    f"lambda_D      {self._format_length(plasmon.debye_length_cm)}",
                    f"hbar*omega_pe {_format_optional(plasmon.plasma_frequency_ev, '{:.4g}')} eV",
                    f"nu_e          {_format_optional(plasmon.electron_collision_rate_s, '{:.4g}')} 1/s",
                    f"k lambda_D    {_format_optional(plasmon.k_lambda_debye, '{:.4g}')}",
                    f"Collectivity  {_format_optional(plasmon.collectivity_parameter, '{:.4g}')}",
                    f"Ef            {_format_optional(plasmon.fermi_energy_ev, '{:.4g}')} eV",
                    f"Theta=Te/Ef   {_format_optional(plasmon.theta_degeneracy, '{:.4g}')}",
                    f"r_s           {_format_optional(plasmon.wigner_seitz_rs, '{:.4g}')}",
                    f"q/qF          {_format_optional(plasmon.q_over_qf, '{:.4g}')}",
                    f"Static LFC    {_format_optional(plasmon.static_lfc_value, '{:.4g}')}",
                    f"LFC model     {plasmon.lfc_model}",
                    f"Model         {plasmon.model_name}",
                    f"Requested     {plasmon.requested_model_name}",
                    f"Preset        {getattr(plasmon, 'benchmark_preset', PLASMON_BENCHMARK_PRESET_NONE)}",
                    f"Backend       {getattr(plasmon, 'response_backend', 'classical_maxwellian')}",
                    f"Backend sum   {getattr(plasmon, 'backend_summary', '') or '-'}",
                    f"Auto summary  {getattr(plasmon, 'auto_model_summary', '') or '-'}",
                    f"Model cmp set {', '.join(str(value) for value in getattr(plasmon, 'comparison_models', ())) or '-'}",
                    f"Exec mode     {getattr(plasmon, 'execution_mode', 'quicklook')}",
                    f"e- policy req {getattr(plasmon, 'requested_electron_policy', getattr(plasmon, 'electron_policy', PLASMON_ELECTRON_POLICY_RAW))}",
                    f"e- policy use {plasmon.electron_policy}",
                    f"e- scope      {plasmon_electron_policy_scope(str(getattr(plasmon, 'electron_policy', PLASMON_ELECTRON_POLICY_RAW)))}",
                    f"Policy cmp set {', '.join(plasmon_electron_policy_label(value) for value in getattr(plasmon, 'policy_comparison_policies', ())) or '-'}",
                    f"e- source     {plasmon.electron_density_source}",
                    f"Materials     {getattr(plasmon, 'material_policy_summary', '') or '-'}",
                    f"Resolved mats {', '.join(getattr(plasmon, 'resolved_materials', ())) or '-'}",
                    f"Raw-kept mats {', '.join(getattr(plasmon, 'raw_kept_materials', ())) or '-'}",
                    f"Collision src {getattr(plasmon, 'collision_source', '-')}",
                    f"Collision sum {getattr(plasmon, 'collision_summary', '-') or '-'}",
                    f"Benchmark     {plasmon.benchmark_status}",
                    f"Full exec     {'yes' if plasmon.model_executed_fully else 'no'}",
                    f"Fallback frac {100.0 * float(plasmon.fallback_fraction):.1f}%",
                    f"Domain fail   {100.0 * float(getattr(plasmon, 'domain_failure_fraction', 0.0)):.1f}%",
                    f"Peak fit      {getattr(plasmon, 'peak_fit_method', 'none')}",
                    f"Integration   {plasmon.integration_mode}",
                    f"Runtime total {float(getattr(plasmon, 'total_runtime_s', 0.0)):.3f} s",
                    f"Runtime spec  {float(getattr(plasmon, 'spectrum_runtime_s', 0.0)):.3f} s",
                    f"Runtime cmp   {float(getattr(plasmon, 'comparison_runtime_s', 0.0)):.3f} s",
                    f"Runtime disp  {float(getattr(plasmon, 'dispersion_runtime_s', 0.0)):.3f} s",
                    f"Runtime ts    {float(getattr(plasmon, 'time_series_runtime_s', 0.0)):.3f} s",
                    f"Zones used    {int(plasmon.zone_count_used)}",
                    f"Clusters used {int(plasmon.cluster_count_used)}",
                    f"Degenerate z/c {int(getattr(plasmon, 'degenerate_zone_count', 0))}/{int(getattr(plasmon, 'degenerate_cluster_count', 0))}",
                    f"Noncollect z/c {int(getattr(plasmon, 'noncollective_zone_count', 0))}/{int(getattr(plasmon, 'noncollective_cluster_count', 0))}",
                    f"Weak-cpl z/c  {int(getattr(plasmon, 'weak_coupling_zone_count', 0))}/{int(getattr(plasmon, 'weak_coupling_cluster_count', 0))}",
                    f"LFC OOD z/c   {int(getattr(plasmon, 'lfc_out_of_domain_zone_count', 0))}/{int(getattr(plasmon, 'lfc_out_of_domain_cluster_count', 0))}",
                    f"Invalid nu z/c {int(getattr(plasmon, 'invalid_collision_zone_count', 0))}/{int(getattr(plasmon, 'invalid_collision_cluster_count', 0))}",
                    f"Spectrum dE   ±{_format_optional(plasmon.spectrum_window_ev, '{:.4g}')} eV",
                    f"Points        {int(plasmon.spectrum_points)}",
                    f"Peak dE       {_format_optional(plasmon.peak_energy_ev, '{:.4g}')} eV",
                    f"Peak FWHM     {_format_optional(plasmon.peak_fwhm_ev, '{:.4g}')} eV",
                    f"Imag shift    {_format_optional(plasmon.spectral_imag_shift_ev, '{:.4g}')} eV",
                    f"IF FWHM       {_format_optional(plasmon.instrument_fwhm_ev, '{:.4g}')} eV",
                )
            )
        )
        if plasmon.spectrum_energy_ev.size and profile_bundles:
            pretty_profile_bundles: list[DerivedPlotBundle] = []
            for bundle in profile_bundles:
                if bundle.key == "spectrum_observed":
                    title = f"Representative spectrum | {str(plasmon.model_name).replace('_', ' ').title()} | {str(plasmon.integration_mode).replace('_', ' ')}"
                    pretty_profile_bundles.append(_clone_bundle(bundle, title=title))
                elif bundle.key == "spectrum_compare_models":
                    pretty_profile_bundles.append(_clone_bundle(bundle, title=f"Model-comparison spectra | {str(plasmon.integration_mode).replace('_', ' ')}"))
                elif bundle.key == "spectrum_compare_policies":
                    pretty_profile_bundles.append(_clone_bundle(bundle, title=f"Electron-policy comparison | {str(plasmon.model_name).replace('_', ' ').title()}"))
                else:
                    pretty_profile_bundles.append(bundle)
            profile_bundles = pretty_profile_bundles
        capabilities = self._current_plasmon_ui_capabilities()
        self.plasmon_plot_panel.set_section_labels(capabilities.primary_label, capabilities.secondary_label)
        preferred_time_key = capabilities.preferred_time_key
        preferred_profile_key = capabilities.preferred_profile_key
        if not any(bundle.key == preferred_time_key for bundle in time_bundles):
            preferred_time_key = next(
                (candidate.key for candidate in capabilities.time_options if any(bundle.key == candidate.key for bundle in time_bundles)),
                (time_bundles[0].key if time_bundles else None),
            )
        if not any(bundle.key == preferred_profile_key for bundle in profile_bundles):
            preferred_profile_key = next(
                (candidate.key for candidate in capabilities.profile_options if any(bundle.key == candidate.key for bundle in profile_bundles)),
                None,
            )
        self.plasmon_plot_panel.set_bundles(
            tuple(time_bundles),
            self._with_snapshot_titles(tuple(profile_bundles), snapshot_index=plasmon.snapshot_index),
            view_scope="plasmon",
            preferred_time_key=preferred_time_key,
            preferred_profile_key=preferred_profile_key,
        )

    def _populate_transmission(self, transmission: TransmissionResult) -> None:
        time_bundles = tuple(self._bundle_time_x(bundle) for bundle in transmission.time_plots)
        profile_bundles: list[DerivedPlotBundle] = []
        for bundle in transmission.profile_plots:
            converted = bundle
            if "[um]" in bundle.x_label:
                converted = self._bundle_length_x(converted)
            profile_bundles.append(converted)
        cold = transmission.cold_refinement
        mode_value = str(transmission.selected_mode or transmission.model_type or "thomson")
        applied_mode_value = str(transmission.model_type or mode_value)
        requested_mode = self.selected_transmission_mode()
        requested_text = _transmission_mode_label(requested_mode)
        applied_text = _transmission_mode_label(applied_mode_value)
        source_text = str(transmission.source or "baseline")
        backend_text = "not used for this mode"
        applicability_text = "mode-local quick look"
        refinement_text = str(transmission.status_message or "Transmission estimate ready.")
        if mode_value in {"xcom", "auto_hybrid"} and cold is not None:
            backend_text = (
                f"{cold.backend_name or 'XCOM'}: {cold.backend_status}"
                + (f" ({cold.source})" if cold.source and cold.source not in {"baseline", ""} else "")
            )
            applicability_text = str(cold.applicability or "indeterminate").replace("_", " ")
            refinement_text = str(cold.message or refinement_text)
        elif mode_value == "auto_hybrid":
            backend_text = "XCOM not used in the current snapshot mixture"
            applicability_text = "snapshot-local hybrid partition"
        elif mode_value == "thomson":
            backend_text = "not used in Thomson mode"
            applicability_text = "scattering-loss quick look"
        else:
            backend_text = "not used in Free-free quick-look modes"
            applicability_text = "weak-coupling plasma quick look"
        request_matches = self._transmission_request_matches_result(transmission)
        self.transmission_model_label.setText(f"Requested: {requested_text} | Applied: {applied_text} ({source_text})")
        self.transmission_backend_label.setText(f"Backend: {backend_text}")
        self.transmission_applicability_label.setText(f"Applicability: {applicability_text}")
        if request_matches:
            self.transmission_refinement_label.setText(f"Status: {refinement_text}")
        else:
            self.transmission_refinement_label.setText(
                f"Status: Showing last computed {applied_text} result. Click Apply to recompute {requested_text}."
            )
        self.transmission_refine_button.setEnabled(not self.activity_progress.isVisible())
        selected_tau = transmission.selected_tau if transmission.selected_tau is not None else transmission.thomson_tau
        selected_transmission = (
            transmission.selected_transmission if transmission.selected_transmission is not None else transmission.thomson_transmission
        )
        self.transmission_summary_label.setText(
            f"Snapshot {transmission.snapshot_index} @ {self._format_time(self._current_result.snapshot_time_s if self._current_result is not None else None)} | "
            f"Tau={_format_optional(selected_tau, '{:.4g}')} | "
            f"T={_format_optional(selected_transmission, '{:.4g}')} | "
            + (
                ""
                if transmission.photon_energy_kev is None
                else f"E={self._format_photon_value_from_kev(transmission.photon_energy_kev, '{:.4g}', unit=self._transmission_energy_unit())} | "
            )
            + f"Requested={requested_text} | Applied={applied_text} | "
            f"{transmission.geometry_summary}"
        )
        status_lines = [
            f"Requested mode: {requested_text}",
            f"Applied mode: {applied_text} ({source_text})",
            f"Status: {refinement_text}",
            f"Geometry: {transmission.geometry_summary}",
        ]
        if not request_matches:
            status_lines.append(
                f"Pending apply: the controls currently request {requested_text}, while the panel is still showing the last computed {applied_text} result."
            )
        elif transmission.photon_energy_kev is not None:
            status_lines.append(
                f"Photon energy: {self._format_photon_value_from_kev(transmission.photon_energy_kev, '{:.6g}', unit=self._transmission_energy_unit())}."
            )
        if transmission.partition is not None:
            for regime in transmission.partition.regime_summaries:
                path_fraction = "-" if regime.path_fraction is None else f"{float(regime.path_fraction) * 100.0:.1f}%"
                areal_fraction = "-" if regime.areal_density_fraction is None else f"{float(regime.areal_density_fraction) * 100.0:.1f}%"
                tau_fraction = "-" if regime.tau_fraction is None else f"{float(regime.tau_fraction) * 100.0:.1f}%"
                status_lines.append(
                    f"{_transmission_regime_label(regime.regime)}: {int(regime.zone_count)} zones | path {path_fraction} | areal {areal_fraction} | tau {tau_fraction}"
                )
            for note in transmission.partition.notes:
                status_lines.append(str(note))
            if transmission.partition.unresolved_materials:
                status_lines.append("Unresolved materials: " + ", ".join(str(value) for value in transmission.partition.unresolved_materials))
        if cold is not None and cold.resolved_materials:
            status_lines.append("Resolved XCOM materials: " + ", ".join(str(value) for value in cold.resolved_materials))
        if cold is not None and cold.unresolved_materials:
            status_lines.append("Unresolved XCOM materials: " + ", ".join(str(value) for value in cold.unresolved_materials))
        self._set_text_browser_text_preserving_scroll(self.transmission_status_pane, "\n".join(status_lines))
        preferred_time_key = "selected_transmission"
        if not any(bundle.key == preferred_time_key for bundle in time_bundles):
            preferred_time_key = (time_bundles[0].key if time_bundles else None)
        preferred_profile_key = "cumulative_selected_transmission"
        if not any(bundle.key == preferred_profile_key for bundle in profile_bundles):
            preferred_profile_key = "cumulative_transmission"
        self.transmission_plot_panel.set_bundles(
            time_bundles,
            self._with_snapshot_titles(tuple(profile_bundles), snapshot_index=transmission.snapshot_index),
            view_scope="transmission",
            preferred_time_key=preferred_time_key,
            preferred_profile_key=preferred_profile_key,
        )
        self.transmission_table.setRowCount(len(transmission.region_budgets))
        total_areal_density = float(sum(float(budget.areal_density_g_cm2) for budget in transmission.region_budgets))
        for row, budget in enumerate(transmission.region_budgets):
            target_fraction = None if total_areal_density <= 0.0 else float(budget.areal_density_g_cm2) / total_areal_density
            notes = "\n".join(str(note) for note in budget.notes if str(note))
            values = (
                str(budget.region_id),
                _format_optional(budget.areal_density_g_cm2, "{:.4g}"),
                _format_optional(None if target_fraction is None else target_fraction * 100.0, "{:.1f}") + ("%" if target_fraction is not None else ""),
                _format_optional(budget.electron_column_cm2, "{:.4g}"),
                _format_optional(budget.thomson_tau, "{:.4g}"),
                _format_optional(budget.free_free_tau, "{:.4g}"),
                _format_optional(budget.xcom_tau, "{:.4g}"),
                _format_optional(budget.total_tau, "{:.4g}"),
                _format_optional(None if budget.xcom_path_fraction is None else budget.xcom_path_fraction * 100.0, "{:.1f}") + ("%" if budget.xcom_path_fraction is not None else ""),
                _format_optional(None if budget.free_free_thomson_path_fraction is None else budget.free_free_thomson_path_fraction * 100.0, "{:.1f}") + ("%" if budget.free_free_thomson_path_fraction is not None else ""),
                _format_optional(None if budget.thomson_fallback_path_fraction is None else budget.thomson_fallback_path_fraction * 100.0, "{:.1f}") + ("%" if budget.thomson_fallback_path_fraction is not None else ""),
                _format_optional(None if budget.xcom_tau_fraction is None else budget.xcom_tau_fraction * 100.0, "{:.1f}") + ("%" if budget.xcom_tau_fraction is not None else ""),
                _format_optional(None if budget.free_free_thomson_tau_fraction is None else budget.free_free_thomson_tau_fraction * 100.0, "{:.1f}") + ("%" if budget.free_free_thomson_tau_fraction is not None else ""),
                _format_optional(None if budget.thomson_fallback_tau_fraction is None else budget.thomson_fallback_tau_fraction * 100.0, "{:.1f}") + ("%" if budget.thomson_fallback_tau_fraction is not None else ""),
                _transmission_region_mix_label(
                    budget.dominant_regime,
                    xcom_path_fraction=budget.xcom_path_fraction,
                    free_free_thomson_path_fraction=budget.free_free_thomson_path_fraction,
                    thomson_fallback_path_fraction=budget.thomson_fallback_path_fraction,
                    xcom_tau_fraction=budget.xcom_tau_fraction,
                    free_free_thomson_tau_fraction=budget.free_free_thomson_tau_fraction,
                    thomson_fallback_tau_fraction=budget.thomson_fallback_tau_fraction,
                ),
            )
            for column, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if notes:
                    item.setToolTip(notes)
                self.transmission_table.setItem(row, column, item)
        self.transmission_table.resizeColumnsToContents()
        self._on_transmission_controls_changed()

    def _populate_spectroscopy(self, spectroscopy: SpectroscopyResult) -> None:
        shift_unit = self._spectroscopy_shift_unit()
        shift_label = _spectroscopy_shift_unit_label(shift_unit)
        shift_value = _convert_shift_nm(float(spectroscopy.doppler_shift_nm), spectroscopy.line_wavelength_nm, shift_unit)
        width_value = np.abs(_convert_shift_nm(float(spectroscopy.thermal_width_nm), spectroscopy.line_wavelength_nm, shift_unit))
        transformed_time_bundles: list[DerivedPlotBundle] = []
        transformed_profile_bundles: list[DerivedPlotBundle] = []
        for bundle in spectroscopy.time_plots:
            converted_bundle = self._bundle_time_x(bundle)
            if bundle.key in {"doppler_shift", "thermal_width"}:
                converted_series = tuple(
                    np.asarray(_convert_shift_nm(np.asarray(series, dtype=np.float64), spectroscopy.line_wavelength_nm, shift_unit), dtype=np.float64)
                    for series in bundle.y_series
                )
                label_kind = "Doppler shift" if bundle.key == "doppler_shift" else "Thermal width"
                transformed_time_bundles.append(_clone_bundle(converted_bundle, y_label=f"{label_kind} [{shift_label}]", y_series=converted_series))
            elif "Velocity [" in bundle.y_label:
                transformed_time_bundles.append(self._bundle_velocity_y(converted_bundle))
            else:
                transformed_time_bundles.append(converted_bundle)
        for bundle in spectroscopy.profile_plots:
            converted_bundle = self._bundle_length_x(bundle) if "[um]" in bundle.x_label else bundle
            if bundle.key == "doppler_proxy_profile":
                converted_series = tuple(
                    np.asarray(_convert_shift_nm(np.asarray(series, dtype=np.float64), spectroscopy.line_wavelength_nm, shift_unit), dtype=np.float64)
                    for series in bundle.y_series
                )
                transformed_profile_bundles.append(_clone_bundle(converted_bundle, y_label=f"LOS Doppler proxy [{shift_label}]", y_series=converted_series))
            elif "Velocity [" in bundle.y_label:
                transformed_profile_bundles.append(self._bundle_velocity_y(converted_bundle))
            elif "temperature [" in bundle.y_label.lower():
                transformed_profile_bundles.append(self._bundle_temperature_y(converted_bundle))
            else:
                transformed_profile_bundles.append(converted_bundle)
        self.spectroscopy_summary_label.setText(
            f"Snapshot {spectroscopy.snapshot_index} @ {self._format_time(self._current_result.snapshot_time_s if self._current_result is not None else None)} | "
            f"LOS cos={spectroscopy.line_of_sight_cosine:.3f} | weighting={spectroscopy.weighting_mode} | "
            f"{spectroscopy.geometry_summary} | shift display={shift_label}"
        )
        self.spectroscopy_metrics.setPlainText(
            "\n".join(
                (
                    f"Bulk velocity {self._format_velocity(spectroscopy.bulk_velocity_cm_s, '{:.4g}')}",
                    f"LOS velocity  {self._format_velocity(spectroscopy.los_velocity_cm_s, '{:.4g}')}",
                    f"Shift         {_format_optional(float(np.asarray(shift_value, dtype=np.float64)), '{:.4g}')} {shift_label}",
                    f"Thermal width {_format_optional(float(np.asarray(width_value, dtype=np.float64)), '{:.4g}')} {shift_label}",
                    f"Ti            {self._format_temperature(spectroscopy.ion_temperature_ev, '{:.4g}')}",
                    f"mu            {_format_optional(spectroscopy.ion_mass_mu, '{:.4g}')}",
                )
            )
        )
        self.spectroscopy_plot_panel.set_bundles(
            tuple(transformed_time_bundles),
            self._with_snapshot_titles(tuple(transformed_profile_bundles), snapshot_index=spectroscopy.snapshot_index),
            view_scope="spectroscopy",
        )

    def _warning_brush(self, severity: str) -> tuple[QtGui.QBrush, QtGui.QBrush]:
        severity = str(severity).lower()
        if severity == "error":
            return QtGui.QBrush(QtGui.QColor("#7f1d1d")), QtGui.QBrush(QtGui.QColor("#fee2e2"))
        if severity == "warning":
            return QtGui.QBrush(QtGui.QColor("#92400e")), QtGui.QBrush(QtGui.QColor("#fef3c7"))
        if severity == "caution":
            return QtGui.QBrush(QtGui.QColor("#854d0e")), QtGui.QBrush(QtGui.QColor("#fef9c3"))
        return QtGui.QBrush(QtGui.QColor(self._theme.subtle_text)), QtGui.QBrush(QtGui.QColor(self._theme.panel_background))

    @staticmethod
    def _warning_counts(warnings: tuple[object, ...] | list[object]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for warning in warnings:
            severity = str(getattr(warning, "severity", "info") or "info").lower()
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _warning_summary_text(self, warnings: tuple[object, ...] | list[object]) -> str:
        counts = self._warning_counts(warnings)
        if not counts:
            return "Warnings: none"
        ordered = [severity for severity in ("error", "warning", "caution", "info") if severity in counts]
        parts = [f"{severity}={counts[severity]}" for severity in ordered]
        total = sum(counts.values())
        return f"Warnings: total={total} | " + ", ".join(parts)

    def _populate_warnings(self, result: DerivedAnalysisResult) -> None:
        self.warnings_tree.clear()
        self.warning_summary_label.setText(self._warning_summary_text(result.warnings))
        grouped: dict[str, list[tuple[str, str]]] = {}
        for warning in result.warnings:
            grouped.setdefault(str(warning.source), []).append((str(warning.severity), str(warning.message)))
        for module_name in sorted(grouped):
            module_items = grouped[module_name]
            counts = self._warning_counts([type("_WarningView", (), {"severity": severity})() for severity, _ in module_items])
            ordered = [severity for severity in ("error", "warning", "caution", "info") if severity in counts]
            summary = ", ".join(f"{severity}={counts[severity]}" for severity in ordered)
            parent = QtWidgets.QTreeWidgetItem([f"{module_name.capitalize()} ({len(module_items)})", summary])
            self.warnings_tree.addTopLevelItem(parent)
            for severity, message in module_items:
                item = QtWidgets.QTreeWidgetItem([severity.upper(), message])
                foreground, background = self._warning_brush(severity)
                item.setForeground(0, foreground)
                item.setForeground(1, foreground)
                item.setBackground(0, background)
                item.setBackground(1, background)
                parent.addChild(item)
            parent.setExpanded(True)
