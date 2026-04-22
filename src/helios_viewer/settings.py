"""Persistent viewer preferences and the viewer settings dialog.

Shell-level session state is stored separately in ``helios_app.session_state``.
This module only covers viewer-specific display and interaction preferences.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

from .style import THEME_MODES, configure_combo_box_interaction


DEFAULT_HOVER_INTERVAL_MS = 12


@dataclass(slots=True)
class ViewerSettings:
    theme_mode: str = "system"
    colormap: str = "cividis"
    map_scale_mode: str = "linear"
    line_scale_mode: str = "linear"
    diagnostic_scale_mode: str = "linear"
    clip_mode: str = "auto"
    show_boundaries: bool = True
    hover_interval_ms: int = DEFAULT_HOVER_INTERVAL_MS
    time_unit: str = "ns"
    length_unit: str = "um"
    pressure_unit: str = "GPa"
    density_unit: str = "g/cm3"
    temperature_unit: str = "eV"
    velocity_unit: str = "km/s"
    specific_energy_unit: str = "J/g"
    rate_unit: str = "J/g/s"
    heat_capacity_unit: str = "J/g/eV"
    number_density_unit: str = "1/cm3"
    angle_unit: str = "deg"
    photon_unit: str = "eV"
    default_profile_coordinate: str = "viewer_follow"
    wheel_guard_enabled: bool = True
    last_open_directory: str = ""


def _settings() -> QtCore.QSettings:
    return QtCore.QSettings("HeliosViewer", "HELIOS HDF5 Quick Look")


def default_viewer_settings() -> ViewerSettings:
    return ViewerSettings()


def load_viewer_settings() -> ViewerSettings:
    store = _settings()
    defaults = default_viewer_settings()
    theme_mode = str(store.value("theme_mode", defaults.theme_mode))
    if theme_mode not in THEME_MODES:
        theme_mode = defaults.theme_mode
    return ViewerSettings(
        theme_mode=theme_mode,
        colormap=str(store.value("colormap", defaults.colormap)),
        map_scale_mode=str(store.value("map_scale_mode", defaults.map_scale_mode)),
        line_scale_mode=str(store.value("line_scale_mode", defaults.line_scale_mode)),
        diagnostic_scale_mode=str(store.value("diagnostic_scale_mode", defaults.diagnostic_scale_mode)),
        clip_mode=str(store.value("clip_mode", defaults.clip_mode)),
        show_boundaries=store.value("show_boundaries", defaults.show_boundaries, type=bool),
        hover_interval_ms=int(store.value("hover_interval_ms", defaults.hover_interval_ms, type=int)),
        time_unit=str(store.value("time_unit", defaults.time_unit)),
        length_unit=str(store.value("length_unit", defaults.length_unit)),
        pressure_unit=str(store.value("pressure_unit", defaults.pressure_unit)),
        density_unit=str(store.value("density_unit", defaults.density_unit)),
        temperature_unit=str(store.value("temperature_unit", defaults.temperature_unit)),
        velocity_unit=str(store.value("velocity_unit", defaults.velocity_unit)),
        specific_energy_unit=str(store.value("specific_energy_unit", defaults.specific_energy_unit)),
        rate_unit=str(store.value("rate_unit", defaults.rate_unit)),
        heat_capacity_unit=str(store.value("heat_capacity_unit", defaults.heat_capacity_unit)),
        number_density_unit=str(store.value("number_density_unit", defaults.number_density_unit)),
        angle_unit=str(store.value("angle_unit", defaults.angle_unit)),
        photon_unit=str(store.value("photon_unit", defaults.photon_unit)),
        default_profile_coordinate=str(store.value("default_profile_coordinate", defaults.default_profile_coordinate)),
        wheel_guard_enabled=store.value("wheel_guard_enabled", defaults.wheel_guard_enabled, type=bool),
        last_open_directory=str(store.value("last_open_directory", defaults.last_open_directory)),
    )


def save_viewer_settings(settings: ViewerSettings) -> None:
    store = _settings()
    store.setValue("theme_mode", settings.theme_mode)
    store.setValue("colormap", settings.colormap)
    store.setValue("map_scale_mode", settings.map_scale_mode)
    store.setValue("line_scale_mode", settings.line_scale_mode)
    store.setValue("diagnostic_scale_mode", settings.diagnostic_scale_mode)
    store.setValue("clip_mode", settings.clip_mode)
    store.setValue("show_boundaries", settings.show_boundaries)
    store.setValue("hover_interval_ms", int(settings.hover_interval_ms))
    store.setValue("time_unit", settings.time_unit)
    store.setValue("length_unit", settings.length_unit)
    store.setValue("pressure_unit", settings.pressure_unit)
    store.setValue("density_unit", settings.density_unit)
    store.setValue("temperature_unit", settings.temperature_unit)
    store.setValue("velocity_unit", settings.velocity_unit)
    store.setValue("specific_energy_unit", settings.specific_energy_unit)
    store.setValue("rate_unit", settings.rate_unit)
    store.setValue("heat_capacity_unit", settings.heat_capacity_unit)
    store.setValue("number_density_unit", settings.number_density_unit)
    store.setValue("angle_unit", settings.angle_unit)
    store.setValue("photon_unit", settings.photon_unit)
    store.setValue("default_profile_coordinate", settings.default_profile_coordinate)
    store.setValue("wheel_guard_enabled", bool(settings.wheel_guard_enabled))
    store.setValue("last_open_directory", settings.last_open_directory)
    store.sync()


def reset_viewer_settings() -> ViewerSettings:
    store = _settings()
    store.clear()
    defaults = default_viewer_settings()
    save_viewer_settings(defaults)
    return defaults


class ViewerSettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings: ViewerSettings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Viewer Settings")
        self.setModal(True)
        self.resize(500, 460)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.addItem("System", "system")
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(settings.theme_mode)))
        form.addRow("Theme", self.theme_combo)

        self.hover_interval_spin = QtWidgets.QSpinBox()
        self.hover_interval_spin.setRange(0, 100)
        self.hover_interval_spin.setSingleStep(2)
        self.hover_interval_spin.setSuffix(" ms")
        self.hover_interval_spin.setValue(int(settings.hover_interval_ms))
        form.addRow("Hover update cadence", self.hover_interval_spin)

        self.time_unit_combo = QtWidgets.QComboBox()
        for value in ("s", "ms", "us", "ns", "ps", "fs"):
            self.time_unit_combo.addItem(value, value)
        self.time_unit_combo.setCurrentIndex(max(0, self.time_unit_combo.findData(settings.time_unit)))
        form.addRow("Time display unit", self.time_unit_combo)

        self.length_unit_combo = QtWidgets.QComboBox()
        for value in ("cm", "mm", "um", "nm"):
            self.length_unit_combo.addItem(value, value)
        self.length_unit_combo.setCurrentIndex(max(0, self.length_unit_combo.findData(settings.length_unit)))
        form.addRow("Length display unit", self.length_unit_combo)

        self.pressure_unit_combo = QtWidgets.QComboBox()
        for value in ("J/cm3", "GPa", "Mbar"):
            self.pressure_unit_combo.addItem(value, value)
        self.pressure_unit_combo.setCurrentIndex(max(0, self.pressure_unit_combo.findData(settings.pressure_unit)))
        form.addRow("Pressure display unit", self.pressure_unit_combo)

        self.density_unit_combo = QtWidgets.QComboBox()
        for value in ("g/cm3", "kg/m3"):
            self.density_unit_combo.addItem(value, value)
        self.density_unit_combo.setCurrentIndex(max(0, self.density_unit_combo.findData(settings.density_unit)))
        form.addRow("Density display unit", self.density_unit_combo)

        self.temperature_unit_combo = QtWidgets.QComboBox()
        for value in ("eV", "K"):
            self.temperature_unit_combo.addItem(value, value)
        self.temperature_unit_combo.setCurrentIndex(max(0, self.temperature_unit_combo.findData(settings.temperature_unit)))
        form.addRow("Temperature display unit", self.temperature_unit_combo)

        self.velocity_unit_combo = QtWidgets.QComboBox()
        for value in ("cm/s", "m/s", "km/s"):
            self.velocity_unit_combo.addItem(value, value)
        self.velocity_unit_combo.setCurrentIndex(max(0, self.velocity_unit_combo.findData(settings.velocity_unit)))
        form.addRow("Velocity display unit", self.velocity_unit_combo)

        self.specific_energy_unit_combo = QtWidgets.QComboBox()
        for value in ("J/g", "kJ/g", "MJ/kg"):
            self.specific_energy_unit_combo.addItem(value, value)
        self.specific_energy_unit_combo.setCurrentIndex(max(0, self.specific_energy_unit_combo.findData(settings.specific_energy_unit)))
        form.addRow("Specific-energy unit", self.specific_energy_unit_combo)

        self.rate_unit_combo = QtWidgets.QComboBox()
        for value in ("J/g/s", "TW/kg"):
            self.rate_unit_combo.addItem(value, value)
        self.rate_unit_combo.setCurrentIndex(max(0, self.rate_unit_combo.findData(settings.rate_unit)))
        form.addRow("Rate/power unit", self.rate_unit_combo)

        self.heat_capacity_unit_combo = QtWidgets.QComboBox()
        for value in ("J/g/eV", "J/kg/eV", "J/g/K", "J/kg/K"):
            self.heat_capacity_unit_combo.addItem(value, value)
        self.heat_capacity_unit_combo.setCurrentIndex(max(0, self.heat_capacity_unit_combo.findData(settings.heat_capacity_unit)))
        form.addRow("Heat-capacity unit", self.heat_capacity_unit_combo)

        self.number_density_unit_combo = QtWidgets.QComboBox()
        for value in ("1/cm3", "1/m3"):
            self.number_density_unit_combo.addItem(value, value)
        self.number_density_unit_combo.setCurrentIndex(max(0, self.number_density_unit_combo.findData(settings.number_density_unit)))
        form.addRow("Number-density unit", self.number_density_unit_combo)

        self.angle_unit_combo = QtWidgets.QComboBox()
        for value, label in (("deg", "deg"), ("rad", "rad")):
            self.angle_unit_combo.addItem(label, value)
        self.angle_unit_combo.setCurrentIndex(max(0, self.angle_unit_combo.findData(settings.angle_unit)))
        form.addRow("Angle display unit", self.angle_unit_combo)

        self.photon_unit_combo = QtWidgets.QComboBox()
        for value in ("keV", "eV", "nm"):
            self.photon_unit_combo.addItem(value, value)
        self.photon_unit_combo.setCurrentIndex(max(0, self.photon_unit_combo.findData(settings.photon_unit)))
        form.addRow("Photon display unit", self.photon_unit_combo)

        self.default_profile_coordinate_combo = QtWidgets.QComboBox()
        self.default_profile_coordinate_combo.addItem("Zone index", "zone")
        self.default_profile_coordinate_combo.addItem("Moving radius", "moving_radius")
        self.default_profile_coordinate_combo.addItem("Static x", "static_x")
        self.default_profile_coordinate_combo.addItem("Follow 2D map", "viewer_follow")
        self.default_profile_coordinate_combo.setCurrentIndex(
            max(0, self.default_profile_coordinate_combo.findData(settings.default_profile_coordinate))
        )
        form.addRow("Default profile coordinate", self.default_profile_coordinate_combo)

        self.wheel_guard_checkbox = QtWidgets.QCheckBox("Enable wheel guard on numeric controls")
        self.wheel_guard_checkbox.setChecked(bool(settings.wheel_guard_enabled))
        self.wheel_guard_checkbox.setToolTip(
            "When enabled, unfocused spin boxes ignore mouse-wheel changes to prevent accidental value edits."
        )
        form.addRow("", self.wheel_guard_checkbox)

        note = QtWidgets.QLabel(
            "Current colormap, scale modes, color-range mode, and boundary-overlay preference are saved automatically."
        )
        note.setWordWrap(True)
        form.addRow("", note)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        reset_button = buttons.addButton("Reset to Defaults", QtWidgets.QDialogButtonBox.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset_button.clicked.connect(self._reset_defaults)
        layout.addWidget(buttons)

        for combo in self.findChildren(QtWidgets.QComboBox):
            configure_combo_box_interaction(combo)

    def _reset_defaults(self) -> None:
        defaults = default_viewer_settings()
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(defaults.theme_mode)))
        self.hover_interval_spin.setValue(int(defaults.hover_interval_ms))
        self.time_unit_combo.setCurrentIndex(max(0, self.time_unit_combo.findData(defaults.time_unit)))
        self.length_unit_combo.setCurrentIndex(max(0, self.length_unit_combo.findData(defaults.length_unit)))
        self.pressure_unit_combo.setCurrentIndex(max(0, self.pressure_unit_combo.findData(defaults.pressure_unit)))
        self.density_unit_combo.setCurrentIndex(max(0, self.density_unit_combo.findData(defaults.density_unit)))
        self.temperature_unit_combo.setCurrentIndex(max(0, self.temperature_unit_combo.findData(defaults.temperature_unit)))
        self.velocity_unit_combo.setCurrentIndex(max(0, self.velocity_unit_combo.findData(defaults.velocity_unit)))
        self.specific_energy_unit_combo.setCurrentIndex(max(0, self.specific_energy_unit_combo.findData(defaults.specific_energy_unit)))
        self.rate_unit_combo.setCurrentIndex(max(0, self.rate_unit_combo.findData(defaults.rate_unit)))
        self.heat_capacity_unit_combo.setCurrentIndex(max(0, self.heat_capacity_unit_combo.findData(defaults.heat_capacity_unit)))
        self.number_density_unit_combo.setCurrentIndex(max(0, self.number_density_unit_combo.findData(defaults.number_density_unit)))
        self.angle_unit_combo.setCurrentIndex(max(0, self.angle_unit_combo.findData(defaults.angle_unit)))
        self.photon_unit_combo.setCurrentIndex(max(0, self.photon_unit_combo.findData(defaults.photon_unit)))
        self.default_profile_coordinate_combo.setCurrentIndex(
            max(0, self.default_profile_coordinate_combo.findData(defaults.default_profile_coordinate))
        )
        self.wheel_guard_checkbox.setChecked(bool(defaults.wheel_guard_enabled))

    def current_settings(self, base: ViewerSettings) -> ViewerSettings:
        return ViewerSettings(
            theme_mode=str(self.theme_combo.currentData()),
            colormap=base.colormap,
            map_scale_mode=base.map_scale_mode,
            line_scale_mode=base.line_scale_mode,
            diagnostic_scale_mode=base.diagnostic_scale_mode,
            clip_mode=base.clip_mode,
            show_boundaries=bool(base.show_boundaries),
            hover_interval_ms=int(self.hover_interval_spin.value()),
            time_unit=str(self.time_unit_combo.currentData()),
            length_unit=str(self.length_unit_combo.currentData()),
            pressure_unit=str(self.pressure_unit_combo.currentData()),
            density_unit=str(self.density_unit_combo.currentData()),
            temperature_unit=str(self.temperature_unit_combo.currentData()),
            velocity_unit=str(self.velocity_unit_combo.currentData()),
            specific_energy_unit=str(self.specific_energy_unit_combo.currentData()),
            rate_unit=str(self.rate_unit_combo.currentData()),
            heat_capacity_unit=str(self.heat_capacity_unit_combo.currentData()),
            number_density_unit=str(self.number_density_unit_combo.currentData()),
            angle_unit=str(self.angle_unit_combo.currentData()),
            photon_unit=str(self.photon_unit_combo.currentData()),
            default_profile_coordinate=str(self.default_profile_coordinate_combo.currentData()),
            wheel_guard_enabled=bool(self.wheel_guard_checkbox.isChecked()),
            last_open_directory=base.last_open_directory,
        )
