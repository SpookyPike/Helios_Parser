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
            screen = window.screen()
            available = screen.availableGeometry() if screen is not None else None
            if available is None:
                self.fail("The native desktop did not report available screen geometry.")

            # Hosted native runners can expose a 1024x720 desktop. Fit the
            # three laptop/desktop targets to that physical surface so the OS
            # does not clamp the top-level window before Qt receives its size.
            min_width = window.minimumSizeHint().width()
            min_height = window.minimumSizeHint().height()
            def fit_dimension(target: int, available_size: int, minimum: int, ratio: float) -> int:
                if available_size >= target:
                    return target
                return min(target, max(minimum, int(available_size * ratio)))

            requests = ((1024, 768, 0.85), (1280, 800, 0.90), (1860, 1120, 0.95))
            sizes = tuple(
                (
                    fit_dimension(width, available.width(), min_width, ratio),
                    fit_dimension(height, available.height(), min_height, ratio),
                )
                for width, height, ratio in requests
            )
            for size in sizes:
                with self.subTest(size=size):
                    window.resize(*size)
                    try:
                        wait_until(lambda: (window.width(), window.height()) == size, timeout_s=2.0)
                    except AssertionError as exc:
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
