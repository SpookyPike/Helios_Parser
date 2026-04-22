from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
from PySide6 import QtWidgets

from _viewer_test_utils import get_app, process_events, reset_test_settings
from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state


class AppPhase41Tests(unittest.TestCase):
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

    def test_mode_toolbar_and_menu_share_single_action_state(self) -> None:
        window = self.open_window()
        try:
            toolbars = window.findChildren(QtWidgets.QToolBar)
            self.assertTrue(toolbars)
            actions = {action.text(): action for action in toolbars[0].actions()}
            self.assertIs(actions["Parser Mode"], window.parser_mode_action)
            self.assertIs(actions["Viewer Mode"], window.viewer_mode_action)
            self.assertIs(actions["Derived / Analysis"], window.derived_mode_action)

            window.viewer_mode_action.trigger()
            process_events()
            self.assertEqual(window._current_mode_id(), "viewer")
            self.assertTrue(window.viewer_mode_action.isChecked())

            window.derived_mode_action.trigger()
            process_events()
            self.assertEqual(window._current_mode_id(), "derived")
            self.assertTrue(window.derived_mode_action.isChecked())
            self.assertFalse(window.viewer_mode_action.isChecked())

            window.parser_mode_action.trigger()
            process_events()
            self.assertEqual(window._current_mode_id(), "parser")
            self.assertTrue(window.parser_mode_action.isChecked())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
