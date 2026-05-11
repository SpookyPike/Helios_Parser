"""Application icon loading helpers for HELIOS Analyzer."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtSvg, QtWidgets


def _asset_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _icon_dir() -> Path:
    return _asset_dir() / "icons"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_icon_png_path() -> Path | None:
    """Return the canonical PNG source for application icons.

    The corrected PNG is the source of truth for the window/taskbar icon. The
    `.ico` file is only a packaging convenience generated from the PNG when
    Pillow is available.
    """

    for candidate in (_repo_root() / "app_icon.png", _asset_dir() / "app_icon.png"):
        if candidate.exists():
            return candidate
    return None


def _icon_sizes() -> tuple[tuple[int, int], ...]:
    return ((16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256))


def ensure_packaging_icon() -> Path | None:
    """Regenerate the Windows `.ico` asset from the canonical PNG when needed."""

    source = canonical_icon_png_path()
    if source is None:
        return None
    target = _asset_dir() / "app_icon.ico"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    try:
        from PIL import Image
    except Exception:
        return target if target.exists() else None
    image = Image.open(source).convert("RGBA")
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="ICO", sizes=list(_icon_sizes()))
    return target


def icon_candidate_paths() -> tuple[Path, ...]:
    """Return icon candidates in preferred load order."""

    asset_dir = _asset_dir()
    candidates: list[Path] = []
    canonical_png = canonical_icon_png_path()
    if canonical_png is not None:
        candidates.append(canonical_png)
    packaging_icon = ensure_packaging_icon()
    if packaging_icon is not None:
        candidates.append(packaging_icon)
    candidates.extend(
        (
            asset_dir / "app_icon.png",
            asset_dir / "app_icon.ico",
            asset_dir / "three_icons.png",
        )
    )
    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        ordered.append(candidate)
        seen.add(candidate)
    return tuple(ordered)


def application_icon_path() -> Path | None:
    """Return the first existing icon asset path."""

    for candidate in icon_candidate_paths():
        if candidate.exists():
            return candidate
    return None


def load_application_icon() -> QtGui.QIcon:
    """Load the application icon, falling back safely when assets are missing."""

    candidate = application_icon_path()
    if candidate is None:
        return QtGui.QIcon()
    return QtGui.QIcon(str(candidate))


def load_ui_icon(name: str, *, color: str | None = None, size: int = 24) -> QtGui.QIcon:
    """Load a minimal UI SVG icon tinted to the current Qt palette."""

    path = _icon_dir() / f"{str(name).strip()}.svg"
    if not path.exists():
        return QtGui.QIcon()
    application = QtWidgets.QApplication.instance()
    palette = application.palette() if application is not None else QtGui.QPalette()
    normal_color = color or palette.color(QtGui.QPalette.ButtonText).name()
    selected_color = "#ffffff"
    disabled_color = palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText).name()
    raw_svg = path.read_text(encoding="utf-8")
    pixel_size = max(12, int(size))

    def _pixmap(icon_color: str) -> QtGui.QPixmap | None:
        svg = raw_svg.replace("currentColor", str(icon_color))
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
        if not renderer.isValid():
            return None
        pixmap = QtGui.QPixmap(pixel_size, pixel_size)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            renderer.render(painter)
        finally:
            painter.end()
        return pixmap

    normal = _pixmap(normal_color)
    if normal is None:
        return QtGui.QIcon(str(path))
    icon = QtGui.QIcon()
    icon.addPixmap(normal, QtGui.QIcon.Normal, QtGui.QIcon.Off)
    selected = _pixmap(selected_color)
    if selected is not None:
        icon.addPixmap(selected, QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon.addPixmap(selected, QtGui.QIcon.Active, QtGui.QIcon.On)
        icon.addPixmap(selected, QtGui.QIcon.Selected, QtGui.QIcon.On)
    disabled = _pixmap(disabled_color)
    if disabled is not None:
        icon.addPixmap(disabled, QtGui.QIcon.Disabled, QtGui.QIcon.Off)
        icon.addPixmap(disabled, QtGui.QIcon.Disabled, QtGui.QIcon.On)
    return icon


def set_button_icon(button: QtWidgets.QAbstractButton, name: str, *, size: int = 18) -> None:
    """Apply an icon to a command button only when the asset is available."""

    icon = load_ui_icon(name, size=size)
    if icon.isNull():
        return
    button.setIcon(icon)
    button.setIconSize(QtCore.QSize(size, size))


def apply_application_icon(target: QtWidgets.QApplication | QtWidgets.QWidget) -> None:
    """Apply the shared application icon if available."""

    icon = load_application_icon()
    if icon.isNull():
        return
    target.setWindowIcon(icon)
