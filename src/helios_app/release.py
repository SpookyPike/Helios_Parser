"""Release-facing metadata for HELIOS Parse / View.

The version and the application name live in the top-level ``helios_release``
module so the command line can read them without importing numpy, h5py or Qt.
This module stays the public face: every packaging script and the About dialog
import from here, and the names below are re-exported unchanged.
"""

from __future__ import annotations

from helios_release import APP_NAME, RELEASE_DATE, RELEASE_VERSION

__all__ = [
    "APP_NAME",
    "AUTHOR_AFFILIATION",
    "AUTHOR_NAME",
    "RELEASE_DATE",
    "RELEASE_VERSION",
    "RELEASE_YEAR",
    "authorship_line",
    "release_label",
]

RELEASE_YEAR = "2026"
AUTHOR_NAME = "Dmitrii Bespalov"
AUTHOR_AFFILIATION = "European XFEL"


def release_label() -> str:
    return f"{APP_NAME} {RELEASE_VERSION}"


def authorship_line() -> str:
    return f"Code developed by {AUTHOR_NAME} at {AUTHOR_AFFILIATION}."