from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import get_app, process_events, reset_test_settings, wait_until
from PySide6 import QtWidgets

from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state


class W11LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def open_window(self) -> HeliosParseViewMainWindow:
        reset_test_settings()
        reset_session_state()
        window = HeliosParseViewMainWindow()
        window.show()
        process_events()
        return window

    def test_requested_desktop_client_sizes_are_honored(self) -> None:
        window = self.open_window()
        try:
            process_events(200)
            for size in ((1024, 768), (1280, 800), (1860, 1120)):
                with self.subTest(size=size):
                    window.resize(*size)
                    try:
                        wait_until(lambda: (window.width(), window.height()) == size, timeout_s=2.0)
                    except AssertionError as exc:
                        screen = window.screen()
                        available = screen.availableGeometry() if screen is not None else None
                        self.fail(
                            f"Requested client size {size}, got "
                            f"{(window.width(), window.height())}; "
                            f"minimumSize={window.minimumSize().width()}x{window.minimumSize().height()}, "
                            f"minimumSizeHint={window.minimumSizeHint().width()}x{window.minimumSizeHint().height()}, "
                            f"sizeHint={window.sizeHint().width()}x{window.sizeHint().height()}, "
                            f"availableScreen={available}, mode={window.session_state.current_mode}: {exc}"
                        )
                    self.assertEqual((window.width(), window.height()), size)
        finally:
            window.close()
            process_events()

    def test_tall_derived_tabs_scroll_at_laptop_height(self) -> None:
        window = self.open_window()
        try:
            window._set_mode("derived")
            window.resize(1024, 768)
            workspace = window.derived_controller.widget()
            workspace.result_tabs.setCurrentIndex(
                workspace.result_tabs.indexOf(workspace.preheat_tab)
            )
            process_events()

            self.assertIs(workspace.result_tabs.currentWidget(), workspace.preheat_tab)
            scroll_area = workspace.result_tabs.widget(
                workspace.result_tabs.indexOf(workspace.preheat_tab)
            )
            self.assertIsInstance(scroll_area, QtWidgets.QScrollArea)
            self.assertGreater(scroll_area.verticalScrollBar().maximum(), 0)
            self.assertTrue(workspace.preheat_requested())
        finally:
            window.close()
            process_events()


if __name__ == "__main__":
    unittest.main()
