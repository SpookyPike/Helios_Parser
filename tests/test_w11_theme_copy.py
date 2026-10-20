from __future__ import annotations

import unittest
from unittest import mock

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import HDF5_ROOT, get_app, process_events, reset_test_settings
from PySide6 import QtCore, QtGui
from test_viewer_wave2 import _build_open_run_payload

from helios_viewer.main_window import HeliosViewerMainWindow
from helios_viewer.style import DARK_THEME, LIGHT_THEME


def _contrast_ratio(foreground: str, background: str) -> float:
    def luminance(color: QtGui.QColor) -> float:
        channels = []
        for channel in (color.redF(), color.greenF(), color.blueF()):
            channels.append(
                channel / 12.92
                if channel <= 0.04045
                else ((channel + 0.055) / 1.055) ** 2.4
            )
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    high, low = sorted(
        (luminance(QtGui.QColor(foreground)), luminance(QtGui.QColor(background))),
        reverse=True,
    )
    return (high + 0.05) / (low + 0.05)


class W11ThemeCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_theme_switch_refreshes_field_colors_and_keeps_filters(self) -> None:
        reset_test_settings()
        window = HeliosViewerMainWindow()
        try:
            window._set_theme_mode("light")
            payload = _build_open_run_payload(
                HDF5_ROOT / "Cu_0166_stabilized.h5",
                window.controller.run_generation,
            )
            payload.field_metadata = {
                name: {
                    "status": "validated",
                    "source": "test",
                    "dimensions": ("time", "zone"),
                }
                for name in payload.fields
            }
            with (
                mock.patch.object(window.controller, "load_field") as load_field,
                mock.patch.object(window.controller, "load_diagnostic"),
            ):
                window._on_run_opened(payload)
                first_field = payload.fields[0]
                window.current_field_name = first_field
                window._select_list_item_by_data(window.field_list, first_field)
                window.field_status_filter_combo.setCurrentIndex(
                    window.field_status_filter_combo.findData("validated")
                )
                window.show_alias_fields_checkbox.setChecked(True)
                process_events()
                load_field.reset_mock()

                window._set_theme_mode("dark")
                process_events()
                load_field.assert_not_called()
                item = window.field_list.currentItem()
                self.assertEqual(
                    item.data(QtCore.Qt.ItemDataRole.UserRole), first_field
                )
                self.assertEqual(
                    window.field_status_filter_combo.currentData(), "validated"
                )
                self.assertTrue(window.show_alias_fields_checkbox.isChecked())
                self.assertEqual(item.foreground().color().name(), "#5eead4")

                window._set_theme_mode("light")
                process_events()
                item = window.field_list.currentItem()
                self.assertEqual(
                    item.data(QtCore.Qt.ItemDataRole.UserRole), first_field
                )
                self.assertEqual(item.foreground().color().name(), "#0f766e")
        finally:
            window.close()
            process_events()

    def test_subtle_information_text_meets_contrast_target(self) -> None:
        for theme in (LIGHT_THEME, DARK_THEME):
            with self.subTest(theme=theme.name):
                self.assertGreaterEqual(
                    _contrast_ratio(theme.subtle_text, theme.panel_background), 4.5
                )


if __name__ == "__main__":
    unittest.main()
