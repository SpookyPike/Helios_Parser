from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - viewer optional dependency
    pg = None


@dataclass(frozen=True)
class ViewerTheme:
    name: str
    window_background: str
    panel_background: str
    control_background: str
    popup_background: str
    plot_background: str
    mask_fill: str
    border_color: str
    text_color: str
    subtle_text: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_contrast: str
    selection_background: str
    selection_text: str
    overlay_color: str
    grid_color: str
    probe_live_color: str
    probe_frozen_color: str
    laser_entry_color: str


LIGHT_THEME = ViewerTheme(
    name="light",
    window_background="#eef3f8",
    panel_background="#ffffff",
    control_background="#ffffff",
    popup_background="#f8fbff",
    plot_background="#f7fafc",
    mask_fill="#dbe4efcc",
    border_color="#c7d2de",
    text_color="#0f172a",
    subtle_text="#475569",
    accent="#0f4c81",
    accent_hover="#145f9f",
    accent_pressed="#0b3c66",
    accent_contrast="#ffffff",
    selection_background="#d8ebff",
    selection_text="#0f172a",
    overlay_color="#475569",
    grid_color="#94a3b8",
    probe_live_color="#0f766e",
    probe_frozen_color="#b45309",
    laser_entry_color="#c2410c",
)

DARK_THEME = ViewerTheme(
    name="dark",
    window_background="#0f172a",
    panel_background="#111827",
    control_background="#1f2937",
    popup_background="#111827",
    plot_background="#0b1220",
    mask_fill="#020617c8",
    border_color="#334155",
    text_color="#e5edf7",
    subtle_text="#94a3b8",
    accent="#60a5fa",
    accent_hover="#93c5fd",
    accent_pressed="#3b82f6",
    accent_contrast="#0b1220",
    selection_background="#1e3a5f",
    selection_text="#e5edf7",
    overlay_color="#cbd5e1",
    grid_color="#475569",
    probe_live_color="#5eead4",
    probe_frozen_color="#fdba74",
    laser_entry_color="#fb923c",
)

THEME_MODES = ("light", "dark", "system")

UI_FONT_FAMILIES = [
    "DejaVu Sans",
    "Segoe UI",
    "Arial",
    "Liberation Sans",
    "Noto Sans",
]

MONO_FONT_FAMILIES = [
    "DejaVu Sans Mono",
    "Cascadia Mono",
    "Consolas",
    "Liberation Mono",
    "Courier New",
]

ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "fonts"

BUNDLED_FONT_FILES = [
    ASSET_ROOT / "DejaVuSans.ttf",
    ASSET_ROOT / "DejaVuSansMono.ttf",
]

FALLBACK_FONT_FILES = [
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/consola.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
]


def _load_application_fonts() -> None:
    for path in BUNDLED_FONT_FILES + FALLBACK_FONT_FILES:
        if path.exists():
            QtGui.QFontDatabase.addApplicationFont(str(path))


def _font_with_fallbacks(families: list[str], point_size: int, *, fixed_pitch: bool = False) -> QtGui.QFont:
    font = QtGui.QFont()
    if families:
        font.setFamily(families[0])
    if hasattr(font, "setFamilies"):
        font.setFamilies(families)
    elif families:
        font.setFamily(families[0])
    font.setPointSize(point_size)
    if fixed_pitch:
        font.setFixedPitch(True)
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return font


def build_ui_font() -> QtGui.QFont:
    return _font_with_fallbacks(UI_FONT_FAMILIES, 10)


def build_mono_font() -> QtGui.QFont:
    return _font_with_fallbacks(MONO_FONT_FAMILIES, 9, fixed_pitch=True)


def resolve_theme(mode: str, app: QtWidgets.QApplication | None = None) -> ViewerTheme:
    selected = str(mode or "system").lower()
    if selected == "dark":
        return DARK_THEME
    if selected == "system":
        application = app or QtWidgets.QApplication.instance()
        if application is not None:
            hints = QtGui.QGuiApplication.styleHints()
            if hasattr(hints, "colorScheme"):
                scheme = hints.colorScheme()
                if scheme == QtCore.Qt.ColorScheme.Dark:
                    return DARK_THEME
                if scheme == QtCore.Qt.ColorScheme.Light:
                    return LIGHT_THEME
            if application.palette().color(QtGui.QPalette.Window).lightnessF() < 0.5:
                return DARK_THEME
        return LIGHT_THEME
    return LIGHT_THEME


def build_palette(theme: ViewerTheme) -> QtGui.QPalette:
    palette = QtGui.QPalette()
    window = QtGui.QColor(theme.window_background)
    panel = QtGui.QColor(theme.panel_background)
    control = QtGui.QColor(theme.control_background)
    popup = QtGui.QColor(theme.popup_background)
    text = QtGui.QColor(theme.text_color)
    subtle = QtGui.QColor(theme.subtle_text)
    accent = QtGui.QColor(theme.selection_background)
    accent_text = QtGui.QColor(theme.selection_text)
    border = QtGui.QColor(theme.border_color)

    palette.setColor(QtGui.QPalette.Window, window)
    palette.setColor(QtGui.QPalette.Base, control)
    palette.setColor(QtGui.QPalette.AlternateBase, panel)
    palette.setColor(QtGui.QPalette.Button, panel)
    palette.setColor(QtGui.QPalette.ToolTipBase, popup)
    palette.setColor(QtGui.QPalette.ToolTipText, text)
    palette.setColor(QtGui.QPalette.Text, text)
    palette.setColor(QtGui.QPalette.ButtonText, text)
    palette.setColor(QtGui.QPalette.WindowText, text)
    palette.setColor(QtGui.QPalette.Highlight, accent)
    palette.setColor(QtGui.QPalette.HighlightedText, accent_text)
    palette.setColor(QtGui.QPalette.PlaceholderText, subtle)
    palette.setColor(QtGui.QPalette.Light, panel.lighter(110))
    palette.setColor(QtGui.QPalette.Midlight, border.lighter(105))
    palette.setColor(QtGui.QPalette.Dark, border.darker(115))
    palette.setColor(QtGui.QPalette.Mid, border)
    palette.setColor(QtGui.QPalette.Shadow, QtGui.QColor("#94a3b8"))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, subtle)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, subtle)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, subtle)
    return palette


def build_stylesheet(theme: ViewerTheme) -> str:
    return f"""
QWidget {{
    color: {theme.text_color};
}}
QMainWindow, QScrollArea, QStatusBar, QMenuBar {{
    background: {theme.window_background};
    color: {theme.text_color};
}}
QStatusBar::item {{
    border: none;
}}
QGroupBox {{
    font-weight: 600;
    border: 1px solid {theme.border_color};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    background: {theme.panel_background};
    color: {theme.text_color};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px 0 4px;
}}
QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QLineEdit, QDoubleSpinBox, QAbstractSpinBox, QTabWidget::pane {{
    background: {theme.control_background};
    color: {theme.text_color};
    border: 1px solid {theme.border_color};
    border-radius: 4px;
    selection-background-color: {theme.selection_background};
    selection-color: {theme.selection_text};
}}
QPlainTextEdit {{
    padding: 4px;
}}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    padding: 2px 6px;
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {theme.subtle_text};
    margin-right: 4px;
}}
QAbstractItemView, QListView, QListWidget, QComboBox QAbstractItemView {{
    background: {theme.popup_background};
    color: {theme.text_color};
    border: 1px solid {theme.border_color};
    outline: 0;
    selection-background-color: {theme.selection_background};
    selection-color: {theme.selection_text};
}}
QTreeWidget {{
    background: {theme.control_background};
    color: {theme.text_color};
    border: 1px solid {theme.border_color};
    border-radius: 4px;
}}
QMenu {{
    background: {theme.popup_background};
    color: {theme.text_color};
    border: 1px solid {theme.border_color};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 24px 5px 24px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {theme.selection_background};
    color: {theme.selection_text};
}}
QMenu::separator {{
    height: 1px;
    margin: 6px 8px;
    background: {theme.border_color};
}}
QMenuBar::item {{
    background: transparent;
    color: {theme.text_color};
    padding: 4px 8px;
}}
QMenuBar::item:selected {{
    background: {theme.selection_background};
    color: {theme.selection_text};
}}
QToolBar {{
    background: {theme.panel_background};
    border: 1px solid {theme.border_color};
    border-radius: 6px;
    spacing: 4px;
    padding: 3px 6px;
}}
QToolButton {{
    background: transparent;
    color: {theme.text_color};
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
}}
QToolButton:hover {{
    background: {theme.selection_background};
    color: {theme.selection_text};
    border-color: {theme.border_color};
}}
QToolButton:checked {{
    background: {theme.accent};
    color: {theme.accent_contrast};
    border-color: {theme.accent_pressed};
}}
QPushButton {{
    background: {theme.accent};
    color: {theme.accent_contrast};
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
}}
QPushButton:hover {{
    background: {theme.accent_hover};
}}
QPushButton:pressed {{
    background: {theme.accent_pressed};
}}
QPushButton:disabled {{
    background: #9aa9bb;
    color: #edf2f7;
}}
QLabel {{
    color: {theme.text_color};
}}
QLabel#SubtleLabel {{
    color: {theme.subtle_text};
}}
QTabWidget::pane {{
    background: {theme.panel_background};
}}
QTabBar::tab {{
    background: {theme.control_background};
    color: {theme.text_color};
    border: 1px solid {theme.border_color};
    border-bottom: none;
    padding: 6px 12px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar::tab:selected {{
    background: {theme.panel_background};
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}
QCheckBox::indicator:unchecked {{
    border: 1px solid {theme.border_color};
    background: {theme.control_background};
    border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    border: 1px solid {theme.accent};
    background: {theme.accent};
    border-radius: 3px;
}}
QSlider {{
    min-height: 24px;
}}
QSlider::groove:horizontal {{
    border: 1px solid {theme.border_color};
    background: {theme.window_background};
    height: 8px;
    margin: 0 4px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {theme.accent};
    border: 1px solid {theme.accent_pressed};
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QToolTip {{
    background: {theme.popup_background};
    color: {theme.text_color};
    border: 1px solid {theme.border_color};
    padding: 4px 6px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
"""


def configure_application(app: QtWidgets.QApplication | None) -> None:
    if app is None:
        return
    if bool(app.property("_helios_viewer_configured")):
        return
    _load_application_fonts()
    app.setOrganizationName("HeliosViewer")
    app.setApplicationName("HELIOS HDF5 Quick Look")
    app.setStyle("Fusion")
    app.setFont(build_ui_font())
    app.setProperty("_helios_viewer_configured", True)
    apply_theme(app, "system")


def configure_combo_box_interaction(combo: QtWidgets.QComboBox) -> None:
    """Normalize combo popup behavior across nested scientific panels."""

    combo.setFocusPolicy(QtCore.Qt.StrongFocus)
    combo.setMaxVisibleItems(max(12, int(combo.maxVisibleItems())))
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)

    view = combo.view()
    if not isinstance(view, QtWidgets.QListView):
        view = QtWidgets.QListView(combo)
        combo.setView(view)
    view.setFocusPolicy(QtCore.Qt.StrongFocus)
    view.setMouseTracking(True)
    view.setUniformItemSizes(True)
    view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
    view.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
    view.setAutoScroll(False)
    if view.viewport() is not None:
        view.viewport().setAttribute(QtCore.Qt.WA_Hover, True)


def apply_theme(app: QtWidgets.QApplication | None, mode: str) -> ViewerTheme:
    theme = resolve_theme(mode, app)
    if app is not None:
        app.setPalette(build_palette(theme))
        app.setStyleSheet(build_stylesheet(theme))
        app.setProperty("_helios_theme_mode", str(mode))
        app.setProperty("_helios_theme_name", theme.name)
    if pg is not None:
        pg.setConfigOptions(background=theme.plot_background, foreground=theme.text_color)
    return theme
