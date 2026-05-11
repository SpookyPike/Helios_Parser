"""Release-facing metadata for HELIOS Parse / View."""

from __future__ import annotations

APP_NAME = "HELIOS Parse / View"
RELEASE_VERSION = "1.1.1"
RELEASE_DATE = "2026-05-11"
RELEASE_YEAR = "2026"
AUTHOR_NAME = "Dmitrii Bespalov"
AUTHOR_AFFILIATION = "European XFEL"


def release_label() -> str:
    return f"{APP_NAME} {RELEASE_VERSION}"


def authorship_line() -> str:
    return f"Code developed by {AUTHOR_NAME} at {AUTHOR_AFFILIATION}."
