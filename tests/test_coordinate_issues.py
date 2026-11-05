"""Coordinate issues: an expected property of the file is not a warning.

HELIOS prints zone positions with fewer significant digits than the zone widths,
so `diff(edges)` cannot reproduce a thin cell even when the file is sound. The
old code raised the same WARNING for that case and for a genuinely inconsistent
table, and on production decks the first case fires on every conversion, which
is how a warning channel stops being read.

These tests pin the discrimination: rounding and width-based reconstruction are
`info`; a table that disagrees materially with its own widths stays `warning`.
"""
from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
import numpy as np

from helios_parser.coordinates import build_coordinate_edge_array

WIDTH = 8.33e-6


def _rounded_grid(count: int) -> tuple[np.ndarray, np.ndarray]:
    """A grid whose printed positions are rounded to 1e-7, as the log does."""
    widths = np.full(count, WIDTH)
    exact = np.arange(count + 1) * WIDTH
    printed = np.round(exact, 7)
    return widths, printed


class CoordinateIssueSeverityTests(unittest.TestCase):
    def test_rounded_positions_are_informational(self) -> None:
        widths, printed = _rounded_grid(300)
        issues: list = []
        build_coordinate_edge_array(
            printed[1:], widths, boundary_edge=float(printed[0]), issues=issues
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "edge_width_mismatch_rounded_positions")
        self.assertEqual(issues[0].severity, "info")

    def test_material_mismatch_stays_a_warning(self) -> None:
        widths = np.full(300, WIDTH)
        outer = np.linspace(WIDTH, 1.0e-2, 300)
        issues: list = []
        build_coordinate_edge_array(outer, widths, boundary_edge=0.0, issues=issues)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "edge_width_mismatch")
        self.assertEqual(issues[0].severity, "warning")

    def test_non_monotonic_reconstruction_is_informational(self) -> None:
        widths, printed = _rounded_grid(20)
        outer = printed[1:].copy()
        outer[5] = outer[4] - 1.0e-7
        issues: list = []
        build_coordinate_edge_array(
            outer, widths, boundary_edge=float(printed[0]), issues=issues
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "rounded_edge_fallback")
        self.assertEqual(issues[0].severity, "info")


if __name__ == "__main__":
    unittest.main()