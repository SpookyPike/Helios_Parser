"""Opt-in gate for the slow integration and GUI tests.

WHY.  ``python -m unittest discover`` is the gate a contributor runs before
pushing, and it has to finish in a time that keeps that habit alive.  A few
files here drive a real Qt event loop or parse the multi-hundred-megabyte
operator fixtures, and together they dominate the wall clock: measured on the
development host, ``test_app_phase412`` and ``test_validation`` each exceed
60 s on their own while the portable core runs in seconds.

They are not useless -- they are the only coverage of the GUI lifecycle and of
the historical log layouts -- so they are gated, not deleted:

    HELIOS_RUN_SLOW_TESTS=1 python -m unittest discover -p "test_*.py"

The environment variable is checked at import time, so discovery still counts
the tests and reports them as skipped rather than hiding them.
"""

from __future__ import annotations

import os
import unittest

SLOW_TESTS_ENV = "HELIOS_RUN_SLOW_TESTS"


def slow_tests_enabled() -> bool:
    return os.environ.get(SLOW_TESTS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def slow(reason: str):
    """Class decorator: skip unless HELIOS_RUN_SLOW_TESTS is set."""
    return unittest.skipUnless(
        slow_tests_enabled(),
        f"{reason} (set {SLOW_TESTS_ENV}=1 to run)",
    )