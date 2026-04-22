from __future__ import annotations

import logging
import math
import time
from typing import Sequence
import warnings

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from helios.instrumentation import increment_counter, record_duration
from .style import LIGHT_THEME, ViewerTheme

LOGGER = logging.getLogger(__name__)


pg.setConfigOptions(
    antialias=True,
    imageAxisOrder="row-major",
    exitCleanup=False,
    background=LIGHT_THEME.plot_background,
    foreground=LIGHT_THEME.text_color,
)


def _tick_font() -> QtGui.QFont:
    font = QtWidgets.QApplication.font()
    font.setPointSize(max(9, font.pointSize()))
    return font


def signed_log10_transform(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.log10(1.0 + np.abs(array))


def signed_log10_inverse(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * (np.power(10.0, np.abs(array)) - 1.0)


def _scene_source_rect(scene: QtWidgets.QGraphicsScene) -> QtCore.QRectF:
    rect = scene.itemsBoundingRect()
    if not rect.isValid() or rect.isEmpty():
        rect = scene.sceneRect()
    if not rect.isValid() or rect.isEmpty():
        rect = QtCore.QRectF(0.0, 0.0, 960.0, 720.0)
    return rect.adjusted(-8.0, -8.0, 8.0, 8.0)


def _target_size_from_widget(widget: QtWidgets.QWidget, source_rect: QtCore.QRectF) -> QtCore.QSize:
    size = widget.size()
    width = max(1, int(size.width() or round(source_rect.width())))
    height = max(1, int(size.height() or round(source_rect.height())))
    return QtCore.QSize(width, height)


def _fit_rect_in_canvas(source_rect: QtCore.QRectF, width: int, height: int) -> QtCore.QRectF:
    if source_rect.width() <= 0.0 or source_rect.height() <= 0.0:
        return QtCore.QRectF(0.0, 0.0, float(width), float(height))
    scale = min(float(width) / source_rect.width(), float(height) / source_rect.height())
    target_width = source_rect.width() * scale
    target_height = source_rect.height() * scale
    return QtCore.QRectF(
        max(0.0, (float(width) - target_width) / 2.0),
        max(0.0, (float(height) - target_height) / 2.0),
        target_width,
        target_height,
    )


def _graphics_view_source_rect(view: QtWidgets.QGraphicsView) -> QtCore.QRectF:
    scene = view.scene()
    if scene is None:
        return QtCore.QRectF(0.0, 0.0, float(max(1, view.width())), float(max(1, view.height())))
    polygon = view.mapToScene(view.viewport().rect())
    rect = polygon.boundingRect()
    if not rect.isValid() or rect.isEmpty():
        rect = _scene_source_rect(scene)
    return rect


def _view_box_rect(view_box: pg.ViewBox) -> QtCore.QRectF:
    x_range, y_range = view_box.viewRange()
    x0 = float(min(x_range))
    x1 = float(max(x_range))
    y0 = float(min(y_range))
    y1 = float(max(y_range))
    return QtCore.QRectF(x0, y0, max(1.0e-12, x1 - x0), max(1.0e-12, y1 - y0))


def _render_graphics_view_to_image(
    view: QtWidgets.QGraphicsView,
    *,
    width: int,
    height: int,
    dpi: int,
    background_hex: str | None,
) -> tuple[QtGui.QImage, QtCore.QRectF]:
    scene = view.scene()
    if scene is None:
        raise RuntimeError("Graphics view has no scene to export.")
    target_width = max(1, int(width))
    target_height = max(1, int(height))
    image = QtGui.QImage(target_width, target_height, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent if background_hex is None else QtGui.QColor(background_hex))
    dots_per_meter = int(round(float(dpi) / 25.4 * 1000.0))
    image.setDotsPerMeterX(dots_per_meter)
    image.setDotsPerMeterY(dots_per_meter)
    source_rect = _graphics_view_source_rect(view)
    target_rect = _fit_rect_in_canvas(source_rect, target_width, target_height)
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        scene.render(painter, target_rect, source_rect)
    finally:
        painter.end()
    return image, source_rect


def _render_widget_to_image(
    widget: QtWidgets.QWidget,
    *,
    width: int,
    height: int,
    dpi: int,
    background_hex: str | None,
) -> QtGui.QImage:
    target_width = max(1, int(width))
    target_height = max(1, int(height))
    widget.resize(target_width, target_height)
    widget.show()
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    image = QtGui.QImage(target_width, target_height, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent if background_hex is None else QtGui.QColor(background_hex))
    dots_per_meter = int(round(float(dpi) / 25.4 * 1000.0))
    image.setDotsPerMeterX(dots_per_meter)
    image.setDotsPerMeterY(dots_per_meter)
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        widget.render(painter, QtCore.QPoint(0, 0))
    finally:
        painter.end()
    return image


def _make_background_transparent(image: QtGui.QImage, color_hex: str) -> QtGui.QImage:
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


_COLORMAP_ALIASES = {
    "gray": "gray",
    "grey": "gray",
    "hot": "hot",
    "jet": "jet",
}


def _custom_colormap(name: str) -> pg.ColorMap | None:
    normalized = _COLORMAP_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if normalized == "gray":
        return pg.ColorMap(
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.asarray([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte),
        )
    if normalized == "hot":
        return pg.ColorMap(
            np.asarray([0.0, 0.33, 0.66, 1.0], dtype=np.float64),
            np.asarray(
                [
                    [0, 0, 0, 255],
                    [230, 0, 0, 255],
                    [255, 210, 0, 255],
                    [255, 255, 255, 255],
                ],
                dtype=np.ubyte,
            ),
        )
    if normalized == "jet":
        return pg.ColorMap(
            np.asarray([0.0, 0.125, 0.375, 0.625, 0.875, 1.0], dtype=np.float64),
            np.asarray(
                [
                    [0, 0, 131, 255],
                    [0, 60, 170, 255],
                    [5, 255, 255, 255],
                    [255, 255, 0, 255],
                    [250, 0, 0, 255],
                    [128, 0, 0, 255],
                ],
                dtype=np.ubyte,
            ),
        )
    return None


def resolve_colormap(name: str) -> pg.ColorMap:
    normalized = str(name).strip().lower()
    candidates = [normalized]
    alias = _COLORMAP_ALIASES.get(normalized)
    if alias is not None and alias not in candidates:
        candidates.append(alias)
    for candidate in candidates:
        custom = _custom_colormap(candidate)
        if custom is not None:
            return custom
        try:
            colormap = pg.colormap.get(candidate)
        except FileNotFoundError:
            continue
        if colormap is not None:
            return colormap
    raise FileNotFoundError(f"Unsupported colormap: {name}")


def _render_scene_to_svg(
    scene: QtWidgets.QGraphicsScene,
    widget: QtWidgets.QWidget,
    path,
    *,
    background_hex: str | None,
    title: str,
) -> None:
    source_rect = _scene_source_rect(scene)
    target_size = _target_size_from_widget(widget, source_rect)
    generator = QtSvg.QSvgGenerator()
    generator.setFileName(str(path))
    generator.setTitle(title)
    generator.setSize(target_size)
    generator.setViewBox(QtCore.QRect(0, 0, target_size.width(), target_size.height()))
    painter = QtGui.QPainter(generator)
    try:
        if background_hex:
            painter.fillRect(QtCore.QRectF(0, 0, target_size.width(), target_size.height()), QtGui.QColor(background_hex))
        scene.render(
            painter,
            QtCore.QRectF(0.0, 0.0, float(target_size.width()), float(target_size.height())),
            source_rect,
        )
    finally:
        painter.end()


def _render_scene_to_pdf(
    scene: QtWidgets.QGraphicsScene,
    widget: QtWidgets.QWidget,
    path,
    *,
    background_hex: str | None,
) -> None:
    writer = QtGui.QPdfWriter(str(path))
    writer.setResolution(300)
    writer.setPageMargins(QtCore.QMarginsF(8.0, 8.0, 8.0, 8.0), QtGui.QPageLayout.Millimeter)
    source_rect = _scene_source_rect(scene)
    painter = QtGui.QPainter(writer)
    try:
        page_rect = QtCore.QRectF(writer.pageLayout().paintRectPixels(writer.resolution()))
        if background_hex:
            painter.fillRect(page_rect, QtGui.QColor(background_hex))
        target_rect = QtCore.QRectF(page_rect)
        source_size = source_rect.size()
        if source_size.width() > 0.0 and source_size.height() > 0.0:
            scale = min(page_rect.width() / source_size.width(), page_rect.height() / source_size.height())
            scaled_size = QtCore.QSizeF(source_size.width() * scale, source_size.height() * scale)
            target_rect = QtCore.QRectF(
                page_rect.x() + max(0.0, (page_rect.width() - scaled_size.width()) / 2.0),
                page_rect.y() + max(0.0, (page_rect.height() - scaled_size.height()) / 2.0),
                scaled_size.width(),
                scaled_size.height(),
            )
        scene.render(painter, target_rect, source_rect)
    finally:
        painter.end()


class ValueAxisItem(pg.AxisItem):
    def __init__(self, orientation: str) -> None:
        super().__init__(orientation)
        self._scale_mode = "linear"
        if hasattr(self, "enableAutoSIPrefix"):
            self.enableAutoSIPrefix(False)

    def set_scale_mode(self, mode: str) -> None:
        self._scale_mode = mode
        self.picture = None
        self.update()

    def tickStrings(self, values: Sequence[float], scale: float, spacing: float) -> list[str]:
        if self._scale_mode == "signed_log10":
            restored = signed_log10_inverse(np.asarray(values, dtype=np.float64))
            return [f"{value:.3g}" for value in restored]
        return super().tickStrings(values, scale, spacing)


class _ColorBarAxisProxy:
    def __init__(self, owner: "FixedColorBar") -> None:
        self._owner = owner

    def setLabel(self, text: str, color: str | None = None) -> None:
        self._owner.set_label(text)

    def setTextPen(self, pen) -> None:
        self._owner.set_text_pen(pen)

    def setTickFont(self, font: QtGui.QFont) -> None:
        self._owner.set_tick_font(font)

    def setWidth(self, width: int) -> None:
        self._owner.set_axis_width(width)


class FixedColorBar(QtWidgets.QWidget):
    """Standalone QWidget colorbar used outside the pyqtgraph scene."""

    def __init__(self, *, theme: ViewerTheme, color_map: pg.ColorMap, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._color_map = color_map
        self._levels = (0.0, 1.0)
        self._label = ""
        self._axis_font = _tick_font()
        self._axis_pen = QtGui.QPen(QtGui.QColor(theme.text_color))
        self._axis_width = 58
        self._source_item: object | None = None
        self._stops: list[tuple[float, QtGui.QColor]] = []
        self.axis = _ColorBarAxisProxy(self)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumWidth(132)
        self.setMaximumWidth(132)
        self.setColorMap(color_map)
        self.apply_theme(theme)

    def setImageItem(self, item: object) -> None:
        self._source_item = item

    def setColorMap(self, color_map: pg.ColorMap | str) -> None:
        if isinstance(color_map, str):
            color_map = resolve_colormap(color_map)
        self._color_map = color_map
        lut = color_map.getLookupTable(nPts=256, alpha=True)
        stop_denominator = max(1, len(lut) - 1)
        self._stops = [
            (
                index / stop_denominator,
                QtGui.QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])),
            )
            for index, rgba in enumerate(lut)
        ]
        self.update()

    def setLevels(self, values: tuple[float, float]) -> None:
        self._levels = FieldMapWidget._normalized_range(values)
        self.update()

    def levels(self) -> tuple[float, float]:
        return self._levels

    def set_label(self, label: str) -> None:
        self._label = label
        self.update()

    def set_text_pen(self, pen) -> None:
        if isinstance(pen, QtGui.QPen):
            self._axis_pen = QtGui.QPen(pen)
        else:
            self._axis_pen = QtGui.QPen(pen)
        self.update()

    def set_tick_font(self, font: QtGui.QFont) -> None:
        self._axis_font = QtGui.QFont(font)
        self.update()

    def set_axis_width(self, width: int) -> None:
        self._axis_width = int(width)
        self.update()

    def apply_theme(self, theme: ViewerTheme) -> None:
        self._theme = theme
        self._axis_pen = QtGui.QPen(QtGui.QColor(theme.text_color))
        self.update()

    @staticmethod
    def _format_tick(value: float) -> str:
        return f"{value:.3g}"

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(self._theme.plot_background))

        outer = self.rect().adjusted(8, 8, -8, -8)
        if outer.width() <= 0 or outer.height() <= 0:
            painter.end()
            return

        label_space = 22 if self._label else 6
        tick_text_space = max(34, self._axis_width)
        tick_len = 6
        bar_width = max(34, outer.width() - tick_text_space - tick_len - label_space - 12)
        bar_rect = QtCore.QRectF(outer.left(), outer.top(), float(bar_width), float(outer.height()))

        gradient = QtGui.QLinearGradient(bar_rect.left(), bar_rect.bottom(), bar_rect.left(), bar_rect.top())
        for stop, color in self._stops:
            gradient.setColorAt(float(stop), color)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.drawRect(bar_rect)

        painter.setPen(QtGui.QPen(QtGui.QColor(self._theme.text_color), 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(bar_rect)

        lower, upper = self._levels
        ticks = np.linspace(lower, upper, num=5, dtype=np.float64)
        painter.setPen(self._axis_pen)
        painter.setFont(self._axis_font)
        for value in ticks:
            fraction = 0.0 if upper == lower else float((value - lower) / (upper - lower))
            y = bar_rect.bottom() - fraction * bar_rect.height()
            tick_start = QtCore.QPointF(bar_rect.right() + 4.0, y)
            tick_end = QtCore.QPointF(bar_rect.right() + 4.0 + tick_len, y)
            painter.drawLine(tick_start, tick_end)
            text_rect = QtCore.QRectF(
                tick_end.x() + 4.0,
                y - 10.0,
                float(tick_text_space - 8),
                20.0,
            )
            painter.drawText(text_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, self._format_tick(float(value)))

        if self._label:
            painter.save()
            painter.setPen(self._axis_pen)
            painter.translate(self.width() - 4.0, self.height() / 2.0)
            painter.rotate(-90.0)
            label_rect = QtCore.QRectF(-(self.height() / 2.0), -18.0, float(self.height()), 20.0)
            painter.drawText(label_rect, QtCore.Qt.AlignCenter, self._label)
            painter.restore()

        painter.end()


class FieldMapWidget(QtWidgets.QWidget):
    probe_moved = QtCore.Signal(float, float)
    probe_clicked = QtCore.Signal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = LIGHT_THEME
        self.last_image_shape: tuple[int, int] | None = None
        self.current_title = ""
        self.current_x_label = ""
        self.current_y_label = ""
        self.current_colorbar_label = ""
        self.current_colormap = "viridis"
        self.current_orientation = "coord_x_time_y"
        self.current_render_mode = "image"
        self.current_boundary_count = 0
        self.current_reference_visible = False
        self.current_boundary_angle: float | None = None
        self.current_reference_angle: float | None = None
        self.current_time_marker_angle: float | None = None
        self.current_time_marker_visible = True
        self.last_raw_matrix: np.ndarray | None = None
        self.last_display_image: np.ndarray | None = None
        self.last_coordinate_values: np.ndarray | None = None
        self.last_coordinate_edges: np.ndarray | None = None
        self.last_time_values: np.ndarray | None = None
        self.last_time_edges: np.ndarray | None = None
        self.last_mesh_x: np.ndarray | None = None
        self.last_mesh_y: np.ndarray | None = None
        self.last_mesh_z: np.ndarray | None = None
        self.current_boundary_positions: tuple[float, ...] = ()
        self.current_boundary_curves: tuple[np.ndarray, ...] = ()
        self.current_reference_position: float | None = None
        self.current_reference_curve: np.ndarray | None = None
        self.current_inactive_ranges: tuple[tuple[float, float], ...] = ()
        self.current_laser_entry_visible = False
        self.current_laser_entry_position: float | None = None
        self.current_laser_entry_curve: np.ndarray | None = None
        self.current_levels: tuple[float, float] | None = None
        self.current_auto_levels = True
        self.render_call_count = 0
        self.mesh_render_count = 0
        self.last_render_elapsed_ms = 0.0
        self.current_navigation_mode = "pan"
        self.probe_visible = False
        self.probe_frozen = False
        self.current_probe_point: tuple[float, float] | None = None
        self._view_context_key: object | None = None
        self._view_preservation_key: object | None = None
        self.last_png_export_source_rect: tuple[float, float, float, float] | None = None

        self._probe_enabled = False
        self._label_style = {"color": self._theme.text_color, "font-size": "11pt"}
        self._title_style = {"color": self._theme.text_color, "size": "13pt"}
        self._active_color_map = resolve_colormap(self.current_colormap)

        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.setBackground(self._theme.plot_background)
        self._graphics.setViewportUpdateMode(QtWidgets.QGraphicsView.MinimalViewportUpdate)
        self._graphics.setOptimizationFlag(QtWidgets.QGraphicsView.DontSavePainterState, True)
        self._graphics.setOptimizationFlag(QtWidgets.QGraphicsView.DontAdjustForAntialiasing, True)
        self._graphics.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self._graphics.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        self._plot = self._graphics.addPlot(row=0, col=0)
        self._plot.setMenuEnabled(False)
        if hasattr(self._plot, "hideButtons"):
            self._plot.hideButtons()
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        self._plot.getViewBox().setBackgroundColor(self._theme.plot_background)
        self._plot.getAxis("left").setTickFont(_tick_font())
        self._plot.getAxis("bottom").setTickFont(_tick_font())
        self._plot.setLabel("left", "Time", **self._label_style)
        self._plot.setLabel("bottom", "Coordinate", **self._label_style)

        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._image_item.setNanPolicy("omit")
        if hasattr(self._image_item, "setAutoDownsample"):
            self._image_item.setAutoDownsample(True)
        self._mesh_item = pg.PColorMeshItem()
        self._mesh_item.hide()
        self._plot.addItem(self._image_item)
        self._plot.addItem(self._mesh_item)

        self._snapshot_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(self._theme.laser_entry_color, width=2))
        self._plot.addItem(self._snapshot_line)

        self._reference_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(self._theme.accent, width=2, style=QtCore.Qt.DashLine))
        self._plot.addItem(self._reference_line)
        self._reference_line.hide()
        self._reference_curve = pg.PlotCurveItem(pen=pg.mkPen(self._theme.accent, width=2, style=QtCore.Qt.DashLine))
        self._reference_curve.hide()
        self._plot.addItem(self._reference_curve)

        self._probe_x_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(self._theme.probe_live_color, width=1.5, style=QtCore.Qt.DotLine))
        self._probe_y_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(self._theme.probe_live_color, width=1.5, style=QtCore.Qt.DotLine))
        self._plot.addItem(self._probe_x_line)
        self._plot.addItem(self._probe_y_line)
        self._probe_x_line.hide()
        self._probe_y_line.hide()

        self._laser_entry_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(self._theme.laser_entry_color, width=2, style=QtCore.Qt.DotLine),
        )
        self._plot.addItem(self._laser_entry_line)
        self._laser_entry_line.hide()
        self._laser_entry_curve = pg.PlotCurveItem(pen=pg.mkPen(self._theme.laser_entry_color, width=2, style=QtCore.Qt.DotLine))
        self._plot.addItem(self._laser_entry_curve)
        self._laser_entry_curve.hide()

        self._boundary_lines: list[pg.InfiniteLine] = []
        self._boundary_curves: list[pg.PlotCurveItem] = []
        self._inactive_regions: list[pg.LinearRegionItem] = []
        self._inactive_region_pools: dict[str, list[pg.LinearRegionItem]] = {
            "horizontal": [],
            "vertical": [],
        }

        self._colorbar = FixedColorBar(theme=self._theme, color_map=resolve_colormap(self.current_colormap))
        self._colorbar.setFixedWidth(132)
        self._colorbar.axis.setTickFont(_tick_font())
        self._colorbar.setImageItem(self._image_item)
        self.set_colormap(self.current_colormap)
        self.apply_theme(self._theme)

        scene = self._graphics.scene()
        scene.sigMouseMoved.connect(self._on_scene_mouse_moved)
        scene.sigMouseClicked.connect(self._on_scene_mouse_clicked)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._graphics, 1)
        layout.addWidget(self._colorbar, 0)

    def set_colormap(self, name: str) -> None:
        self.current_colormap = name
        self._active_color_map = resolve_colormap(name)
        self._apply_active_colormap()

    def _apply_active_colormap(self) -> None:
        """Keep image and mesh rendering paths aligned on the same colormap.

        This widget can render the 2D field map either as a mesh (`PColorMeshItem`)
        or as a regular image (`ImageItem`). The original bug was that only the mesh
        path had the selected colormap reapplied, so switching from moving-mesh radius
        into zone/static image mode silently fell back to grayscale. Any render-path
        transition must update all three targets together: image item, mesh item, and
        the visible colorbar.
        """

        self._colorbar.setColorMap(self._active_color_map)
        self._mesh_item.setColorMap(self._active_color_map)
        if hasattr(self._image_item, "setColorMap"):
            self._image_item.setColorMap(self._active_color_map)

    def save_png(self, path, *, width: int, height: int, dpi: int, transparent: bool = False) -> None:
        export_widget = self._build_export_clone()
        try:
            image = _render_widget_to_image(
                export_widget,
                width=width,
                height=height,
                dpi=dpi,
                background_hex=None if transparent else self._theme.plot_background,
            )
        finally:
            export_widget.close()
            export_widget.deleteLater()
        view_rect = _view_box_rect(self._plot.getViewBox())
        self.last_png_export_source_rect = (
            float(view_rect.x()),
            float(view_rect.y()),
            float(view_rect.width()),
            float(view_rect.height()),
        )
        if transparent:
            image = _make_background_transparent(image, self._theme.plot_background)
        image.save(str(path))

    def _build_export_clone(self) -> "FieldMapWidget":
        clone = FieldMapWidget()
        clone.apply_theme(self._theme)
        clone.set_colormap(self.current_colormap)
        clone.set_field_map(
            np.asarray(self.last_raw_matrix if self.last_raw_matrix is not None else np.empty((0, 0)), dtype=np.float64),
            np.asarray(self.last_coordinate_values if self.last_coordinate_values is not None else np.empty(0), dtype=np.float64),
            np.asarray(self.last_time_values if self.last_time_values is not None else np.empty(0), dtype=np.float64),
            orientation=self.current_orientation,
            title=self.current_title,
            x_label=self.current_x_label,
            y_label=self.current_y_label,
            colorbar_label=self.current_colorbar_label,
            levels=self.current_levels,
            auto_levels=self.current_auto_levels,
            boundary_positions=self.current_boundary_positions,
            show_boundaries=bool(self.current_boundary_count),
            reference_position=self.current_reference_position,
            mesh_x=self.last_mesh_x,
            mesh_y=self.last_mesh_y,
            coordinate_edges=self.last_coordinate_edges,
            time_edges=self.last_time_edges,
            boundary_curves=self.current_boundary_curves,
            reference_curve=self.current_reference_curve,
            inactive_ranges=self.current_inactive_ranges,
            laser_entry_position=self.current_laser_entry_position,
            laser_entry_curve=self.current_laser_entry_curve,
            show_time_marker=self.current_time_marker_visible,
            show_reference_marker=self.current_reference_visible,
            preserve_view=False,
            view_context_key=None,
            view_preservation_key=None,
        )
        if self.probe_visible and self.current_probe_point is not None:
            clone.set_probe_point(self.current_probe_point[0], self.current_probe_point[1], frozen=self.probe_frozen)
        clone.set_time_marker(float(self._snapshot_line.value()))
        source_rect = _view_box_rect(self._plot.getViewBox())
        view_box = clone._plot.getViewBox()
        view_box.disableAutoRange()
        view_box.setXRange(float(source_rect.left()), float(source_rect.right()), padding=0.0)
        view_box.setYRange(float(source_rect.top()), float(source_rect.bottom()), padding=0.0)
        return clone

    def apply_theme(self, theme: ViewerTheme) -> None:
        self._theme = theme
        self._label_style = {"color": theme.text_color, "font-size": "11pt"}
        self._title_style = {"color": theme.text_color, "size": "13pt"}
        self._graphics.setBackground(theme.plot_background)
        self._plot.getViewBox().setBackgroundColor(theme.plot_background)
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        axis_pen = pg.mkPen(theme.text_color)
        grid_pen = pg.mkPen(theme.grid_color)
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setTextPen(axis_pen)
            axis.setPen(axis_pen)
            axis.setTickFont(_tick_font())
        self._plot.setTitle(self.current_title, **self._title_style)
        self._plot.setLabel("bottom", self.current_x_label or "Coordinate", **self._label_style)
        self._plot.setLabel("left", self.current_y_label or "Time", **self._label_style)
        self._colorbar.apply_theme(theme)
        self._colorbar.axis.setLabel(self.current_colorbar_label or "", color=theme.text_color)
        self._colorbar.axis.setTextPen(axis_pen)
        self._snapshot_line.setPen(pg.mkPen(theme.laser_entry_color, width=2))
        self._reference_line.setPen(pg.mkPen(theme.accent, width=2, style=QtCore.Qt.DashLine))
        self._reference_curve.setPen(pg.mkPen(theme.accent, width=2, style=QtCore.Qt.DashLine))
        probe_pen = pg.mkPen(theme.probe_live_color, width=1.4, style=QtCore.Qt.DotLine)
        self._probe_x_line.setPen(probe_pen)
        self._probe_y_line.setPen(probe_pen)
        self._laser_entry_line.setPen(pg.mkPen(theme.laser_entry_color, width=2, style=QtCore.Qt.DotLine))
        self._laser_entry_curve.setPen(pg.mkPen(theme.laser_entry_color, width=2, style=QtCore.Qt.DotLine))
        for line in self._boundary_lines:
            line.setPen(pg.mkPen(theme.overlay_color, width=1, style=QtCore.Qt.DashLine))
        for curve in self._boundary_curves:
            curve.setPen(pg.mkPen(theme.overlay_color, width=1, style=QtCore.Qt.DashLine))
        brush = pg.mkBrush(QtGui.QColor(theme.mask_fill))
        for region in self._inactive_regions:
            region.setBrush(brush)

    def save_vector_svg(self, path, *, transparent: bool = False) -> None:
        _render_scene_to_svg(
            self._graphics.scene(),
            self._graphics,
            path,
            background_hex=None if transparent else self._theme.plot_background,
            title=self.current_title or "HELIOS field map",
        )

    def save_vector_pdf(self, path) -> None:
        _render_scene_to_pdf(
            self._graphics.scene(),
            self._graphics,
            path,
            background_hex=self._theme.plot_background,
        )

    def set_probe_enabled(self, enabled: bool) -> None:
        self._probe_enabled = enabled
        if not enabled and not self.probe_frozen:
            self.clear_probe()

    def set_probe_point(self, x_value: float, y_value: float, *, frozen: bool = False) -> None:
        pen_color = self._theme.probe_frozen_color if frozen else self._theme.probe_live_color
        pen_style = QtCore.Qt.SolidLine if frozen else QtCore.Qt.DotLine
        pen = pg.mkPen(pen_color, width=1.6 if frozen else 1.4, style=pen_style)
        self._probe_x_line.setPen(pen)
        self._probe_y_line.setPen(pen)
        self._probe_x_line.setValue(float(x_value))
        self._probe_y_line.setValue(float(y_value))
        self._probe_x_line.show()
        self._probe_y_line.show()
        self.probe_visible = True
        self.probe_frozen = frozen
        self.current_probe_point = (float(x_value), float(y_value))

    def clear_probe(self) -> None:
        self._probe_x_line.hide()
        self._probe_y_line.hide()
        self.probe_visible = False
        self.probe_frozen = False
        self.current_probe_point = None

    def _set_active_colorbar_item(self, item: pg.GraphicsObject) -> None:
        self._colorbar.setImageItem(item)

    def _clear_boundary_lines(self) -> None:
        for line in self._boundary_lines:
            line.hide()
        self.current_boundary_positions = ()
        self.current_boundary_angle = None

    def _clear_boundary_curves(self) -> None:
        for curve in self._boundary_curves:
            curve.hide()
        self.current_boundary_curves = ()

    def _clear_inactive_regions(self) -> None:
        for region in self._inactive_regions:
            region.hide()
        self.current_inactive_ranges = ()

    def _set_boundary_lines(self, positions: Sequence[float], *, angle: float) -> None:
        pen = pg.mkPen(self._theme.overlay_color, width=1, style=QtCore.Qt.DashLine)
        while len(self._boundary_lines) < len(positions):
            line = pg.InfiniteLine(angle=angle, movable=False, pen=pen)
            line.hide()
            self._plot.addItem(line)
            self._boundary_lines.append(line)
        for index, line in enumerate(self._boundary_lines):
            if index < len(positions):
                line.setPen(pen)
                line.setAngle(angle)
                line.setValue(float(positions[index]))
                line.show()
            else:
                line.hide()
        self.current_boundary_positions = tuple(float(position) for position in positions)
        self.current_boundary_angle = float(angle)

    def _set_boundary_curves(self, curves: Sequence[np.ndarray]) -> None:
        pen = pg.mkPen(self._theme.overlay_color, width=1, style=QtCore.Qt.DashLine)
        normalized_curves: list[np.ndarray] = []
        for curve in curves:
            values = np.asarray(curve, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 2:
                continue
            normalized_curves.append(values)
        while len(self._boundary_curves) < len(normalized_curves):
            item = pg.PlotCurveItem(pen=pen)
            item.hide()
            self._plot.addItem(item)
            self._boundary_curves.append(item)
        for index, item in enumerate(self._boundary_curves):
            if index < len(normalized_curves):
                values = normalized_curves[index]
                item.setPen(pen)
                item.setData(values[:, 0], values[:, 1])
                item.show()
            else:
                item.hide()
        self.current_boundary_curves = tuple(normalized_curves)

    def _set_inactive_ranges(self, ranges: Sequence[tuple[float, float]], *, orientation: str) -> None:
        brush = pg.mkBrush(QtGui.QColor(self._theme.mask_fill))
        region_orientation = "horizontal" if orientation == "time_x_coord_y" else "vertical"
        pool = self._inactive_region_pools[region_orientation]
        while len(pool) < len(ranges):
            item = pg.LinearRegionItem(values=(0.0, 1.0), orientation=region_orientation, brush=brush, pen=None, movable=False)
            item.setZValue(10)
            item.hide()
            self._plot.addItem(item)
            pool.append(item)
            self._inactive_regions.append(item)
        for item in self._inactive_regions:
            item.hide()
        for index, (start, stop) in enumerate(ranges):
            item = pool[index]
            item.setBrush(brush)
            item.setRegion((float(start), float(stop)))
            item.show()
        self.current_inactive_ranges = tuple((float(start), float(stop)) for start, stop in ranges)

    def _set_reference_curve(self, curve: np.ndarray | None) -> None:
        if curve is None:
            self._reference_curve.hide()
            self.current_reference_curve = None
            return
        values = np.asarray(curve, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            self._reference_curve.hide()
            self.current_reference_curve = None
            return
        self._reference_curve.setData(values[:, 0], values[:, 1])
        self._reference_curve.show()
        self.current_reference_curve = values

    def _set_laser_entry_line(self, position: float, *, angle: float) -> None:
        self._laser_entry_curve.hide()
        self._laser_entry_line.setAngle(angle)
        self._laser_entry_line.setValue(float(position))
        self._laser_entry_line.show()
        self.current_laser_entry_visible = True
        self.current_laser_entry_position = float(position)
        self.current_laser_entry_curve = None

    def _set_laser_entry_curve(self, curve: np.ndarray | None) -> None:
        if curve is None:
            self._laser_entry_curve.hide()
            self._laser_entry_line.hide()
            self.current_laser_entry_visible = False
            self.current_laser_entry_position = None
            self.current_laser_entry_curve = None
            return
        values = np.asarray(curve, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            self._set_laser_entry_curve(None)
            return
        self._laser_entry_line.hide()
        self._laser_entry_curve.setData(values[:, 0], values[:, 1])
        self._laser_entry_curve.show()
        self.current_laser_entry_visible = True
        self.current_laser_entry_position = None
        self.current_laser_entry_curve = values

    def _current_data_bounds(self) -> tuple[float, float, float, float] | None:
        if self.current_render_mode == "mesh" and self.last_mesh_x is not None and self.last_mesh_y is not None:
            x_bounds = self._finite_min_max(self.last_mesh_x)
            y_bounds = self._finite_min_max(self.last_mesh_y)
            if x_bounds is None or y_bounds is None:
                return None
            return (
                float(x_bounds[0]),
                float(x_bounds[1]),
                float(y_bounds[0]),
                float(y_bounds[1]),
            )
        if self.last_coordinate_values is None or self.last_time_values is None:
            return None
        coordinate_edges = self.last_coordinate_edges
        time_edges = self.last_time_edges
        coordinate_values = np.asarray(self.last_coordinate_values, dtype=np.float64)
        time_values = np.asarray(self.last_time_values, dtype=np.float64)
        if coordinate_values.size == 0 or time_values.size == 0:
            return None
        if self.current_orientation == "time_x_coord_y":
            x_values = np.asarray(time_edges if time_edges is not None else self._centers_to_edges(time_values), dtype=np.float64)
            y_values = np.asarray(
                coordinate_edges if coordinate_edges is not None else self._centers_to_edges(coordinate_values),
                dtype=np.float64,
            )
        else:
            x_values = np.asarray(
                coordinate_edges if coordinate_edges is not None else self._centers_to_edges(coordinate_values),
                dtype=np.float64,
            )
            y_values = np.asarray(time_edges if time_edges is not None else self._centers_to_edges(time_values), dtype=np.float64)
        return (
            float(np.min(x_values)),
            float(np.max(x_values)),
            float(np.min(y_values)),
            float(np.max(y_values)),
        )

    @staticmethod
    def _centers_to_edges(values: np.ndarray) -> np.ndarray:
        centers = np.asarray(values, dtype=np.float64)
        if centers.size == 0:
            return np.array([], dtype=np.float64)
        if centers.size == 1:
            delta = max(abs(float(centers[0])) * 0.5, 0.5)
            return np.asarray([float(centers[0]) - delta, float(centers[0]) + delta], dtype=np.float64)
        edges = np.empty(centers.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - (edges[1] - centers[0])
        edges[-1] = centers[-1] + (centers[-1] - edges[-2])
        return edges

    @staticmethod
    def _normalized_range(bounds: tuple[float, float]) -> tuple[float, float]:
        lower, upper = float(bounds[0]), float(bounds[1])
        if lower == upper:
            scale = max(abs(lower), abs(upper), np.finfo(np.float64).eps)
            delta = max(scale * 0.02, np.finfo(np.float64).eps * 10.0)
            lower -= delta
            upper += delta
        return (min(lower, upper), max(lower, upper))

    @staticmethod
    def _finite_min_max(values: np.ndarray) -> tuple[float, float] | None:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(invalid="ignore", over="ignore"):
                lower = float(np.nanmin(array))
                upper = float(np.nanmax(array))
        if math.isfinite(lower) and math.isfinite(upper):
            return (lower, upper)
        flat = array.reshape(-1)
        chunk_size = 262_144
        found = False
        finite_min = 0.0
        finite_max = 0.0
        for start in range(0, flat.size, chunk_size):
            chunk = flat[start : start + chunk_size]
            finite_mask = np.isfinite(chunk)
            if not np.any(finite_mask):
                continue
            finite_chunk = chunk[finite_mask]
            chunk_min = float(np.min(finite_chunk))
            chunk_max = float(np.max(finite_chunk))
            if not found:
                finite_min = chunk_min
                finite_max = chunk_max
                found = True
                continue
            if chunk_min < finite_min:
                finite_min = chunk_min
            if chunk_max > finite_max:
                finite_max = chunk_max
        if not found:
            return None
        return (finite_min, finite_max)

    @staticmethod
    def _padded_range(bounds: tuple[float, float], *, fraction: float, floor_fraction: float = 0.01) -> tuple[float, float]:
        lower, upper = FieldMapWidget._normalized_range(bounds)
        span = upper - lower
        reference = max(abs(lower), abs(upper), span, np.finfo(np.float64).eps)
        padding = max(span * fraction, reference * floor_fraction * fraction, np.finfo(np.float64).eps * 10.0)
        return (lower - padding, upper + padding)

    def _capture_view_state(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float, float, float]] | None:
        data_bounds = self._current_data_bounds()
        if data_bounds is None:
            return None
        view_x, view_y = self._plot.getViewBox().viewRange()
        return (
            (float(view_x[0]), float(view_x[1])),
            (float(view_y[0]), float(view_y[1])),
            data_bounds,
        )

    @staticmethod
    def _restore_axis_range(
        view_box: pg.ViewBox,
        axis: str,
        view_range: tuple[float, float],
        previous_bounds: tuple[float, float],
        new_bounds: tuple[float, float],
    ) -> None:
        previous_min, previous_max = previous_bounds
        new_min, new_max = new_bounds
        previous_width = previous_max - previous_min
        new_width = new_max - new_min
        if previous_width == 0.0 or new_width == 0.0:
            return
        start_fraction = (view_range[0] - previous_min) / previous_width
        end_fraction = (view_range[1] - previous_min) / previous_width
        start_value = new_min + start_fraction * new_width
        end_value = new_min + end_fraction * new_width
        if axis == "x":
            view_box.setXRange(start_value, end_value, padding=0.0)
        else:
            view_box.setYRange(start_value, end_value, padding=0.0)

    def _restore_view_state(
        self,
        state: tuple[tuple[float, float], tuple[float, float], tuple[float, float, float, float]] | None,
    ) -> bool:
        if state is None:
            return False
        new_bounds = self._current_data_bounds()
        if new_bounds is None:
            return False
        x_view, y_view, previous_bounds = state
        view_box = self._plot.getViewBox()
        self._restore_axis_range(view_box, "x", x_view, previous_bounds[:2], new_bounds[:2])
        self._restore_axis_range(view_box, "y", y_view, previous_bounds[2:], new_bounds[2:])
        return True

    def _auto_fit_to_data(self) -> bool:
        bounds = self._current_data_bounds()
        if bounds is None:
            return False
        x_bounds = self._padded_range(bounds[:2], fraction=0.03)
        y_bounds = self._padded_range(bounds[2:], fraction=0.04)
        view_box = self._plot.getViewBox()
        view_box.disableAutoRange()
        view_box.setXRange(x_bounds[0], x_bounds[1], padding=0.0)
        view_box.setYRange(y_bounds[0], y_bounds[1], padding=0.0)
        return True

    def reset_dataset_state(self) -> None:
        """Clear stale plot state before a new dataset is rendered."""
        self.last_image_shape = None
        self.current_title = ""
        self.current_x_label = ""
        self.current_y_label = ""
        self.current_colorbar_label = ""
        self.current_render_mode = "image"
        self.current_boundary_count = 0
        self.current_reference_visible = False
        self.current_boundary_angle = None
        self.current_reference_angle = None
        self.current_time_marker_angle = None
        self.current_time_marker_visible = True
        self.last_raw_matrix = None
        self.last_display_image = None
        self.last_coordinate_values = None
        self.last_coordinate_edges = None
        self.last_time_values = None
        self.last_time_edges = None
        self.last_mesh_x = None
        self.last_mesh_y = None
        self.last_mesh_z = None
        self.current_boundary_positions = ()
        self.current_boundary_curves = ()
        self.current_reference_position = None
        self.current_reference_curve = None
        self.current_inactive_ranges = ()
        self.current_laser_entry_visible = False
        self.current_laser_entry_position = None
        self.current_laser_entry_curve = None
        self.current_levels = None
        self.current_auto_levels = True
        self._view_context_key = None
        self._view_preservation_key = None
        self._image_item.hide()
        self._mesh_item.hide()
        self._snapshot_line.hide()
        self._reference_line.hide()
        self._set_reference_curve(None)
        self._set_laser_entry_curve(None)
        self._clear_boundary_lines()
        self._clear_boundary_curves()
        self._clear_inactive_regions()
        self.clear_probe()
        self._plot.setTitle("", **self._title_style)
        self._plot.setLabel("bottom", "Coordinate", **self._label_style)
        self._plot.setLabel("left", "Time", **self._label_style)
        self._set_active_colorbar_item(self._image_item)
        self._colorbar.setLevels((0.0, 1.0))
        self._colorbar.axis.setLabel("", color=self._theme.text_color)
        self._plot.getViewBox().disableAutoRange()

    def set_field_map(
        self,
        data: np.ndarray,
        coordinate_values: np.ndarray,
        time_values: np.ndarray,
        *,
        orientation: str,
        title: str,
        x_label: str,
        y_label: str,
        colorbar_label: str,
        levels: tuple[float, float] | None = None,
        auto_levels: bool = True,
        boundary_positions: Sequence[float] | None = None,
        show_boundaries: bool = False,
        reference_position: float | None = None,
        mesh_x: np.ndarray | None = None,
        mesh_y: np.ndarray | None = None,
        coordinate_edges: np.ndarray | None = None,
        time_edges: np.ndarray | None = None,
        boundary_curves: Sequence[np.ndarray] | None = None,
        reference_curve: np.ndarray | None = None,
        inactive_ranges: Sequence[tuple[float, float]] | None = None,
        laser_entry_position: float | None = None,
        laser_entry_curve: np.ndarray | None = None,
        show_time_marker: bool = True,
        show_reference_marker: bool = False,
        preserve_view: bool = False,
        view_context_key: object | None = None,
        view_preservation_key: object | None = None,
    ) -> None:
        started = time.perf_counter()
        same_view_context = view_preservation_key is not None and view_preservation_key == self._view_preservation_key
        previous_view = self._capture_view_state() if preserve_view and same_view_context else None
        self.render_call_count += 1
        image = np.asarray(data, dtype=np.float64)
        self.last_image_shape = tuple(int(value) for value in image.shape)
        # Keep references to the current render inputs instead of cloning large
        # arrays on every refresh. Export/clone paths can materialize copies when
        # they need an isolated widget state.
        self.last_raw_matrix = image
        self.last_coordinate_values = np.asarray(coordinate_values, dtype=np.float64)
        self.last_coordinate_edges = None if coordinate_edges is None else np.asarray(coordinate_edges, dtype=np.float64)
        self.last_time_values = np.asarray(time_values, dtype=np.float64)
        self.last_time_edges = None if time_edges is None else np.asarray(time_edges, dtype=np.float64)
        self.current_title = title
        self.current_x_label = x_label
        self.current_y_label = y_label
        self.current_colorbar_label = colorbar_label
        self.current_orientation = orientation
        self.current_levels = levels
        self.current_auto_levels = bool(auto_levels)
        self._view_context_key = view_context_key
        self._view_preservation_key = view_preservation_key
        self.current_boundary_count = 0
        self.current_time_marker_visible = bool(show_time_marker)
        self.last_mesh_x = None
        self.last_mesh_y = None
        self.last_mesh_z = None

        if mesh_x is not None and mesh_y is not None:
            self.mesh_render_count += 1
            x_mesh = np.asarray(mesh_x, dtype=np.float64)
            y_mesh = np.asarray(mesh_y, dtype=np.float64)
            z_mesh = image
            self.current_render_mode = "mesh"
            self.last_mesh_x = x_mesh
            self.last_mesh_y = y_mesh
            self.last_mesh_z = z_mesh
            self.last_display_image = None
            self._image_item.hide()
            self._mesh_item.show()
            self._mesh_item.setData(x_mesh, y_mesh, z_mesh, autoLevels=auto_levels)
            self._apply_active_colormap()
            self._set_active_colorbar_item(self._mesh_item)
            if auto_levels:
                finite_bounds = self._finite_min_max(z_mesh)
                if finite_bounds is not None:
                    resolved_levels = self._normalized_range(finite_bounds)
                    self._mesh_item.setLevels(resolved_levels)
                    self._colorbar.setLevels(resolved_levels)
            elif levels is not None:
                resolved_levels = self._normalized_range(levels)
                self._mesh_item.setLevels(resolved_levels)
                self._colorbar.setLevels(resolved_levels)
        else:
            self.current_render_mode = "image"
            if orientation == "time_x_coord_y":
                display = image.T
                x_values = np.asarray(time_values, dtype=np.float64)
                y_values = np.asarray(coordinate_values, dtype=np.float64)
                self._snapshot_line.setAngle(90)
                self.current_time_marker_angle = 90.0
                boundary_angle = 0
                reference_angle = 0
            else:
                display = image
                x_values = np.asarray(coordinate_values, dtype=np.float64)
                y_values = np.asarray(time_values, dtype=np.float64)
                self._snapshot_line.setAngle(0)
                self.current_time_marker_angle = 0.0
                boundary_angle = 90
                reference_angle = 90
            self.last_display_image = np.asarray(display, dtype=np.float64)
            self._mesh_item.hide()
            self._image_item.show()
            self._image_item.setImage(display, autoLevels=auto_levels)
            # ImageItem may be recreated/reset internally by setImage(), so explicitly
            # reapply the selected ColorMap every time we rebuild the image render path.
            self._apply_active_colormap()
            self._set_active_colorbar_item(self._image_item)
            if not auto_levels and levels is not None:
                resolved_levels = self._normalized_range(levels)
                self._image_item.setLevels(resolved_levels)
                self._colorbar.setLevels(resolved_levels)
            elif auto_levels:
                finite_bounds = self._finite_min_max(display)
                if finite_bounds is not None:
                    self._colorbar.setLevels(self._normalized_range(finite_bounds))
            if display.size:
                resolved_coordinate_edges = (
                    np.asarray(coordinate_edges, dtype=np.float64)
                    if coordinate_edges is not None
                    else self._centers_to_edges(coordinate_values)
                )
                resolved_time_edges = (
                    np.asarray(time_edges, dtype=np.float64)
                    if time_edges is not None
                    else self._centers_to_edges(time_values)
                )
                if orientation == "time_x_coord_y":
                    x_edges = resolved_time_edges
                    y_edges = resolved_coordinate_edges
                else:
                    x_edges = resolved_coordinate_edges
                    y_edges = resolved_time_edges
                x0 = float(x_edges[0]) if x_edges.size else 0.0
                x1 = float(x_edges[-1]) if x_edges.size else float(max(1, display.shape[1] - 1))
                y0 = float(y_edges[0]) if y_edges.size else 0.0
                y1 = float(y_edges[-1]) if y_edges.size else float(max(1, display.shape[0] - 1))
                width = x1 - x0
                height = y1 - y0
                if width == 0.0:
                    width = 1.0
                if height == 0.0:
                    height = 1.0
                self._image_item.setRect(QtCore.QRectF(x0, y0, width, height))

        self._plot.setTitle(title, **self._title_style)
        self._plot.setLabel("bottom", x_label, **self._label_style)
        self._plot.setLabel("left", y_label, **self._label_style)
        self._colorbar.axis.setLabel(colorbar_label or "", color=self._theme.text_color)
        if show_time_marker:
            self._snapshot_line.show()
        else:
            self._snapshot_line.hide()

        self._clear_boundary_lines()
        self._clear_boundary_curves()
        if show_boundaries and boundary_curves:
            self._set_boundary_curves(boundary_curves)
            self.current_boundary_count = len(self.current_boundary_curves)
        elif show_boundaries and boundary_positions and mesh_x is None and mesh_y is None:
            self._set_boundary_lines(boundary_positions, angle=boundary_angle)
            self.current_boundary_count = len(self.current_boundary_positions)
        else:
            self.current_boundary_count = 0

        if inactive_ranges and mesh_x is None and mesh_y is None:
            self._set_inactive_ranges(inactive_ranges, orientation=orientation)
        else:
            self._clear_inactive_regions()

        self._reference_line.hide()
        self.current_reference_visible = False
        self.current_reference_position = None
        self.current_reference_angle = None
        self._set_reference_curve(None)
        if show_reference_marker and reference_curve is not None:
            self._set_reference_curve(reference_curve)
            self.current_reference_visible = True
        elif show_reference_marker and reference_position is not None and mesh_x is None and mesh_y is None:
            self._reference_line.setAngle(reference_angle)
            self._reference_line.setValue(float(reference_position))
            self._reference_line.show()
            self.current_reference_visible = True
            self.current_reference_position = float(reference_position)
            self.current_reference_angle = float(reference_angle)

        self._set_laser_entry_curve(None)
        if laser_entry_curve is not None:
            self._set_laser_entry_curve(laser_entry_curve)
        elif laser_entry_position is not None and mesh_x is None and mesh_y is None:
            self._set_laser_entry_line(float(laser_entry_position), angle=boundary_angle)

        if not self._restore_view_state(previous_view):
            self._auto_fit_to_data()
        elapsed_s = time.perf_counter() - started
        self.last_render_elapsed_ms = elapsed_s * 1.0e3
        record_duration("viewer.render.field_map", elapsed_s)
        increment_counter("viewer.render.field_map.calls")
        increment_counter(f"viewer.field_map.render.{self.current_render_mode}")
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "viewer.render.field_map took %.3f ms (mode=%s shape=%s)",
                self.last_render_elapsed_ms,
                self.current_render_mode,
                self.last_image_shape,
            )

    def set_time_marker(self, value: float) -> None:
        self._snapshot_line.setValue(value)

    def set_navigation_mode(self, mode: str) -> None:
        view_box = self._plot.getViewBox()
        if mode == "zoom":
            view_box.setMouseMode(view_box.RectMode)
            self.current_navigation_mode = "zoom"
        else:
            view_box.setMouseMode(view_box.PanMode)
            self.current_navigation_mode = "pan"

    def reset_view(self) -> None:
        if not self._auto_fit_to_data():
            self._plot.enableAutoRange()
            self._plot.getViewBox().autoRange()

    def _map_scene_to_view(self, scene_position) -> tuple[float, float] | None:
        if not self._plot.sceneBoundingRect().contains(scene_position):
            return None
        view_position = self._plot.getViewBox().mapSceneToView(scene_position)
        return float(view_position.x()), float(view_position.y())

    def _on_scene_mouse_moved(self, scene_position) -> None:
        if not self._probe_enabled or self.probe_frozen:
            return
        mapped = self._map_scene_to_view(scene_position)
        if mapped is None:
            return
        self.probe_moved.emit(mapped[0], mapped[1])

    def _on_scene_mouse_clicked(self, event) -> None:
        if not self._probe_enabled:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        mapped = self._map_scene_to_view(event.scenePos())
        if mapped is None:
            return
        self.probe_clicked.emit(mapped[0], mapped[1])


class CurvePlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = LIGHT_THEME
        self.current_colormap = "viridis"
        self.current_curve_count = 0
        self.current_primary_length = 0
        self.current_title = ""
        self.current_x_label = ""
        self.current_y_label = ""
        self.value_scale_mode = "linear"
        self.current_boundary_count = 0
        self.current_cursor_visible = False
        self.current_cursor_position: float | None = None
        self.last_x_values: np.ndarray | None = None
        self.last_raw_y_series: tuple[np.ndarray, ...] = ()
        self.last_y_series: tuple[np.ndarray, ...] = ()
        self.current_boundary_positions: tuple[float, ...] = ()
        self.set_curves_call_count = 0
        self.curve_item_update_count = 0
        self.current_navigation_mode = "pan"
        self._view_context_key: object | None = None
        self.last_png_export_source_rect: tuple[float, float, float, float] | None = None

        axis_items = {"left": ValueAxisItem("left")}
        self._plot = pg.PlotWidget(axisItems=axis_items)
        self._plot.setBackground(self._theme.plot_background)
        self._plot.setMenuEnabled(False)
        self._plot.setViewportUpdateMode(QtWidgets.QGraphicsView.MinimalViewportUpdate)
        self._plot.setOptimizationFlag(QtWidgets.QGraphicsView.DontSavePainterState, True)
        self._plot.setOptimizationFlag(QtWidgets.QGraphicsView.DontAdjustForAntialiasing, True)
        plot_item = self._plot.getPlotItem()
        plot_item.setMenuEnabled(False)
        if hasattr(plot_item, "hideButtons"):
            plot_item.hideButtons()
        self._plot.getViewBox().setBackgroundColor(self._theme.plot_background)
        self._value_axis: ValueAxisItem = self._plot.getAxis("left")  # type: ignore[assignment]
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        self._plot.getAxis("left").setTickFont(_tick_font())
        self._plot.getAxis("bottom").setTickFont(_tick_font())
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            if hasattr(axis, "enableAutoSIPrefix"):
                axis.enableAutoSIPrefix(False)
        self._legend: pg.LegendItem | None = None
        self._boundary_lines: list[pg.InfiniteLine] = []
        self._curve_items: list[pg.PlotCurveItem] = []
        self._curve_names: list[str | None] = []
        self._cursor_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(self._theme.accent, width=2, style=QtCore.Qt.DotLine),
        )
        self._plot.addItem(self._cursor_line)
        self._cursor_line.hide()
        self._label_style = {"color": self._theme.text_color, "font-size": "11pt"}
        self._title_style = {"color": self._theme.text_color, "size": "13pt"}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)
        self._plot.viewport().installEventFilter(self)
        self.apply_theme(self._theme)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self._plot.viewport() and event.type() == QtCore.QEvent.MouseButtonDblClick:
            mouse_event = event
            if isinstance(mouse_event, QtGui.QMouseEvent) and mouse_event.button() == QtCore.Qt.LeftButton:
                self.reset_view()
                mouse_event.accept()
                return True
        return super().eventFilter(watched, event)

    def set_colormap(self, name: str) -> None:
        """Update the line palette used for slice and diagnostic curves."""

        self.current_colormap = str(name)
        self._apply_curve_palette()

    def apply_theme(self, theme: ViewerTheme) -> None:
        self._theme = theme
        self._label_style = {"color": theme.text_color, "font-size": "11pt"}
        self._title_style = {"color": theme.text_color, "size": "13pt"}
        self._plot.setBackground(theme.plot_background)
        self._plot.getViewBox().setBackgroundColor(theme.plot_background)
        axis_pen = pg.mkPen(theme.text_color)
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setTextPen(axis_pen)
            axis.setPen(axis_pen)
            axis.setTickFont(_tick_font())
        self._plot.setTitle(self.current_title, **self._title_style)
        self._plot.setLabel("bottom", self.current_x_label or "Coordinate", **self._label_style)
        self._plot.setLabel("left", self.current_y_label or "Value", **self._label_style)
        self._cursor_line.setPen(pg.mkPen(theme.accent, width=2, style=QtCore.Qt.DotLine))
        if self._legend is not None:
            desired_names = list(self._curve_names[: self.current_curve_count])
            scene = self._legend.scene()
            if scene is not None:
                scene.removeItem(self._legend)
            self._legend = None
            self._curve_names = []
            self._rebuild_legend(desired_names, self.current_curve_count)
        for line in self._boundary_lines:
            line.setPen(pg.mkPen(theme.overlay_color, width=1, style=QtCore.Qt.DashLine))
        self._apply_curve_palette()

    def save_vector_svg(self, path, *, transparent: bool = False) -> None:
        _render_scene_to_svg(
            self._plot.scene(),
            self._plot,
            path,
            background_hex=None if transparent else self._theme.plot_background,
            title=self.current_title or "HELIOS plot",
        )

    def save_vector_pdf(self, path) -> None:
        _render_scene_to_pdf(
            self._plot.scene(),
            self._plot,
            path,
            background_hex=self._theme.plot_background,
        )

    def save_png(self, path, *, width: int, height: int, dpi: int, transparent: bool = False) -> None:
        export_widget = self._build_export_clone()
        try:
            image = _render_widget_to_image(
                export_widget,
                width=width,
                height=height,
                dpi=dpi,
                background_hex=None if transparent else self._theme.plot_background,
            )
        finally:
            export_widget.close()
            export_widget.deleteLater()
        view_rect = _view_box_rect(self._plot.getViewBox())
        self.last_png_export_source_rect = (
            float(view_rect.x()),
            float(view_rect.y()),
            float(view_rect.width()),
            float(view_rect.height()),
        )
        if transparent:
            image = _make_background_transparent(image, self._theme.plot_background)
        image.save(str(path))

    def _clear_boundary_lines(self) -> None:
        for line in self._boundary_lines:
            line.hide()
        self.current_boundary_count = 0
        self.current_boundary_positions = ()

    def _current_data_bounds(self) -> tuple[float, float, float, float] | None:
        if self.last_x_values is None or self.last_x_values.size == 0:
            return None
        x_bounds = FieldMapWidget._finite_min_max(self.last_x_values)
        if x_bounds is None:
            return None
        if not self.last_y_series:
            return None
        lower_y: float | None = None
        upper_y: float | None = None
        for series in self.last_y_series:
            y_bounds = FieldMapWidget._finite_min_max(series)
            if y_bounds is None:
                continue
            lower_y = float(y_bounds[0]) if lower_y is None else min(lower_y, float(y_bounds[0]))
            upper_y = float(y_bounds[1]) if upper_y is None else max(upper_y, float(y_bounds[1]))
        if lower_y is None or upper_y is None:
            return None
        return (
            float(x_bounds[0]),
            float(x_bounds[1]),
            float(lower_y),
            float(upper_y),
        )

    def _capture_view_state(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float, float, float]] | None:
        data_bounds = self._current_data_bounds()
        if data_bounds is None:
            return None
        view_x, view_y = self._plot.getViewBox().viewRange()
        return (
            (float(view_x[0]), float(view_x[1])),
            (float(view_y[0]), float(view_y[1])),
            data_bounds,
        )

    def _restore_view_state(
        self,
        state: tuple[tuple[float, float], tuple[float, float], tuple[float, float, float, float]] | None,
    ) -> bool:
        if state is None:
            return False
        new_bounds = self._current_data_bounds()
        if new_bounds is None:
            return False
        x_view, y_view, previous_bounds = state
        view_box = self._plot.getViewBox()
        FieldMapWidget._restore_axis_range(view_box, "x", x_view, previous_bounds[:2], new_bounds[:2])
        FieldMapWidget._restore_axis_range(view_box, "y", y_view, previous_bounds[2:], new_bounds[2:])
        return True

    def _auto_fit_to_data(self) -> bool:
        bounds = self._current_data_bounds()
        if bounds is None:
            return False
        x_bounds = FieldMapWidget._padded_range(bounds[:2], fraction=0.03)
        if self.value_scale_mode == "log10" and bounds[2] > 0.0:
            y_bounds = (
                max(np.nextafter(0.0, 1.0), bounds[2] / 1.2),
                bounds[3] * 1.2 if bounds[3] > 0.0 else bounds[3],
            )
        else:
            y_bounds = FieldMapWidget._padded_range(bounds[2:], fraction=0.08)
        view_box = self._plot.getViewBox()
        view_box.disableAutoRange()
        view_box.setXRange(x_bounds[0], x_bounds[1], padding=0.0)
        view_box.setYRange(y_bounds[0], y_bounds[1], padding=0.0)
        return True

    def reset_dataset_state(self) -> None:
        """Clear stale curve state before a new dataset is rendered."""
        self.current_curve_count = 0
        self.current_primary_length = 0
        self.current_title = ""
        self.current_x_label = ""
        self.current_y_label = ""
        self.value_scale_mode = "linear"
        self.current_boundary_count = 0
        self.current_cursor_visible = False
        self.current_cursor_position = None
        self.last_x_values = None
        self.last_raw_y_series = ()
        self.last_y_series = ()
        self.current_boundary_positions = ()
        self._view_context_key = None
        self._clear_boundary_lines()
        self._cursor_line.hide()
        self._ensure_curve_items(0)
        self._rebuild_legend(None, 0)
        self._plot.setTitle("", **self._title_style)
        self._plot.setLabel("bottom", "Coordinate", **self._label_style)
        self._plot.setLabel("left", "Value", **self._label_style)
        self._plot.getViewBox().disableAutoRange()

    def clear_plot(self) -> None:
        """Public clear helper that preserves internal widget lifecycle state."""

        self.reset_dataset_state()

    def _transform_y(self, values: np.ndarray, mode: str) -> np.ndarray:
        if mode == "signed_log10":
            return signed_log10_transform(values)
        if mode == "log10":
            return np.where(values > 0.0, values, np.nan)
        return values

    def _ensure_curve_items(self, count: int) -> None:
        while len(self._curve_items) < count:
            item = pg.PlotCurveItem(pen=pg.mkPen(self._theme.accent, width=2))
            self._plot.addItem(item)
            self._curve_items.append(item)
            self._curve_names.append(None)
        while len(self._curve_items) > count:
            item = self._curve_items.pop()
            self._plot.removeItem(item)
            self._curve_names.pop()
        self._apply_curve_palette()

    def _curve_pen_colors(self, count: int) -> list[str]:
        if count <= 0:
            return []
        color_map = resolve_colormap(self.current_colormap)
        if count == 1:
            samples = [0.78]
        else:
            samples = np.linspace(0.14, 0.86, num=count, dtype=np.float64)
        colors: list[str] = []
        for sample in samples:
            qcolor = QtGui.QColor(color_map.mapToQColor(float(sample)))
            colors.append(qcolor.name())
        return colors

    def _apply_curve_palette(self) -> None:
        colors = self._curve_pen_colors(len(self._curve_items))
        for item, color in zip(self._curve_items, colors):
            item.setPen(pg.mkPen(color, width=2))

    def _rebuild_legend(self, names: Sequence[str] | None, count: int) -> None:
        desired = [names[index] if names and index < len(names) else None for index in range(count)]
        if self._curve_names == desired and ((count > 1) == (self._legend is not None)):
            return
        if self._legend is not None:
            scene = self._legend.scene()
            if scene is not None:
                scene.removeItem(self._legend)
            self._legend = None
        self._curve_names = desired
        if count > 1:
            self._legend = self._plot.addLegend(offset=(10, 10), labelTextColor=self._theme.text_color)
            for item, name in zip(self._curve_items[:count], desired):
                self._legend.addItem(item, name or "")

    def set_curves(
        self,
        x: np.ndarray,
        ys: Sequence[np.ndarray],
        *,
        title: str,
        x_label: str,
        y_label: str,
        curve_names: Sequence[str] | None = None,
        value_scale_mode: str = "linear",
        boundary_positions: Sequence[float] | None = None,
        show_boundaries: bool = False,
        auto_range: bool = True,
        cursor_position: float | None = None,
        show_cursor: bool = False,
        preserve_view: bool = False,
        view_context_key: object | None = None,
    ) -> None:
        same_view_context = view_context_key is not None and view_context_key == self._view_context_key
        previous_view = self._capture_view_state() if preserve_view and same_view_context else None
        self.set_curves_call_count += 1
        self.current_curve_count = len(ys)
        self.current_primary_length = int(len(ys[0])) if ys else 0
        self.current_title = title
        self.current_x_label = x_label
        self.current_y_label = y_label
        self.value_scale_mode = value_scale_mode
        self._view_context_key = view_context_key
        # Hot curve updates now retain linear arrays by reference when safe.
        # Callers must treat x/y arrays as immutable after handoff; export/clone
        # paths still materialize isolated copies explicitly below.
        x_values = np.asarray(x, dtype=np.float64)
        self.last_x_values = x_values
        raw_series: list[np.ndarray] = []
        transformed_series: list[np.ndarray] = []

        self._plot.setTitle(title, **self._title_style)
        self._plot.setLabel("bottom", x_label, **self._label_style)
        self._plot.setLabel("left", y_label, **self._label_style)
        self._value_axis.set_scale_mode("signed_log10" if value_scale_mode == "signed_log10" else "linear")
        self._plot.setLogMode(False, value_scale_mode == "log10")
        self._ensure_curve_items(len(ys))
        self._rebuild_legend(curve_names, len(ys))
        for index, y in enumerate(ys):
            raw_values = np.asarray(y, dtype=np.float64)
            y_values = self._transform_y(raw_values, value_scale_mode)
            raw_series.append(raw_values)
            transformed_series.append(np.asarray(y_values, dtype=np.float64))
            self._curve_items[index].setData(x_values, y_values)
            self.curve_item_update_count += 1
        for item in self._curve_items[len(ys):]:
            item.clear()
        self.last_raw_y_series = tuple(raw_series)
        self.last_y_series = tuple(transformed_series)

        if show_boundaries and boundary_positions:
            pen = pg.mkPen(self._theme.overlay_color, width=1, style=QtCore.Qt.DashLine)
            while len(self._boundary_lines) < len(boundary_positions):
                line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
                line.hide()
                self._plot.addItem(line)
                self._boundary_lines.append(line)
            for index, line in enumerate(self._boundary_lines):
                if index < len(boundary_positions):
                    line.setPen(pen)
                    line.setValue(float(boundary_positions[index]))
                    line.show()
                else:
                    line.hide()
            active_boundary_count = len(boundary_positions)
        else:
            self._clear_boundary_lines()
            active_boundary_count = 0
        self.current_boundary_count = active_boundary_count
        self.current_boundary_positions = tuple(float(position) for position in (boundary_positions or ()))
        if show_cursor and cursor_position is not None:
            self._cursor_line.setValue(float(cursor_position))
            self._cursor_line.show()
            self.current_cursor_visible = True
            self.current_cursor_position = float(cursor_position)
        else:
            self._cursor_line.hide()
            self.current_cursor_visible = False
            self.current_cursor_position = None
        if auto_range and not self._restore_view_state(previous_view):
            self._auto_fit_to_data()

    def set_cursor_marker(self, value: float | None, *, visible: bool = True) -> None:
        if not visible or value is None or not np.isfinite(float(value)):
            self._cursor_line.hide()
            self.current_cursor_visible = False
            self.current_cursor_position = None
            return
        cursor_value = float(value)
        self._cursor_line.setValue(cursor_value)
        self._cursor_line.show()
        self.current_cursor_visible = True
        self.current_cursor_position = cursor_value

    def clear_cursor_marker(self) -> None:
        self.set_cursor_marker(None, visible=False)

    def _build_export_clone(self) -> "CurvePlotWidget":
        clone = CurvePlotWidget()
        clone.apply_theme(self._theme)
        clone.set_colormap(self.current_colormap)
        clone.set_curves(
            np.asarray(self.last_x_values if self.last_x_values is not None else np.empty(0), dtype=np.float64),
            [np.asarray(series, dtype=np.float64) for series in self.last_raw_y_series],
            title=self.current_title,
            x_label=self.current_x_label,
            y_label=self.current_y_label,
            curve_names=[name or "" for name in self._curve_names[: self.current_curve_count]],
            value_scale_mode=self.value_scale_mode,
            boundary_positions=self.current_boundary_positions,
            show_boundaries=bool(self.current_boundary_count),
            auto_range=False,
            cursor_position=self.current_cursor_position,
            show_cursor=self.current_cursor_visible,
            preserve_view=False,
            view_context_key=None,
        )
        source_rect = _view_box_rect(self._plot.getViewBox())
        view_box = clone._plot.getViewBox()
        view_box.disableAutoRange()
        view_box.setXRange(float(source_rect.left()), float(source_rect.right()), padding=0.0)
        view_box.setYRange(float(source_rect.top()), float(source_rect.bottom()), padding=0.0)
        return clone

    def set_navigation_mode(self, mode: str) -> None:
        view_box = self._plot.getViewBox()
        if mode == "zoom":
            view_box.setMouseMode(view_box.RectMode)
            self.current_navigation_mode = "zoom"
        else:
            view_box.setMouseMode(view_box.PanMode)
            self.current_navigation_mode = "pan"

    def reset_view(self) -> None:
        if not self._auto_fit_to_data():
            self._plot.enableAutoRange()
            self._plot.getViewBox().autoRange()
