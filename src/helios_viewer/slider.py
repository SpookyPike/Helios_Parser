"""Small Qt slider helpers shared across viewer and shell widgets."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class _AbsoluteClickSliderStyle(QtWidgets.QProxyStyle):
    """Use absolute click positioning so large snapshot jumps are deterministic."""

    def styleHint(
        self,
        hint: QtWidgets.QStyle.StyleHint,
        option: QtWidgets.QStyleOption | None = None,
        widget: QtWidgets.QWidget | None = None,
        returnData: QtWidgets.QStyleHintReturn | None = None,
    ) -> int:
        if hint == QtWidgets.QStyle.SH_Slider_AbsoluteSetButtons:
            return int(QtCore.Qt.MouseButton.LeftButton.value)
        return super().styleHint(hint, option, widget, returnData)


def apply_absolute_click_slider_behavior(slider: QtWidgets.QSlider) -> None:
    """Keep standard drag behavior while making click-to-jump predictable."""

    # Qt's built-in Fusion style may report an empty name on some platforms;
    # the desktop shell configures Fusion as its application style.
    base_style_name = slider.style().name() or "Fusion"
    proxy_style = _AbsoluteClickSliderStyle(base_style_name)
    proxy_style.setParent(slider)
    slider.setStyle(proxy_style)
