from __future__ import annotations

import logging
import os
import sys

from PySide6 import QtWidgets

from .icon import apply_application_icon
from .main_window import HeliosViewerMainWindow
from .style import configure_application


def _configure_logging() -> None:
    level_name = os.environ.get("HELIOS_ANALYZER_LOG_LEVEL")
    if not level_name and os.environ.get("HELIOS_ANALYZER_DEBUG"):
        level_name = "DEBUG"
    if not level_name:
        return
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    else:
        root.setLevel(level)


def main() -> int:
    _configure_logging()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app)
    apply_application_icon(app)
    window = HeliosViewerMainWindow()
    apply_application_icon(window)
    window.show()
    return app.exec()
