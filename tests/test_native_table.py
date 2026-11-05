"""Contract tests for the native LOG table parser and its Python fallback.

The kernel is an optional accelerator: `native_table.parse_table_block` returns
None when the library is absent or the block does not fit the contract, and
`_parse_field_table_rows` then runs its `np.fromstring` path unchanged. These
tests pin both sides of that boundary, because a divergence between them would
show up as a silently different number in a converted file.

No fixture: every block is an inline literal, so the entry stays cheap enough for
the core gate.
"""
from __future__ import annotations

import logging
import unittest
from unittest import mock

import _test_bootstrap  # noqa: F401
import numpy as np

from helios_parser import native_table
from helios_parser.parser import _parse_field_table_rows

LOGGER = logging.getLogger(__name__)

HEADER = "#    Radius     Zone width Mass dens  Velocity   Rad temp   Ion temp   Elec temp  Ion press  Elec press Rad press   Compression Elec Dens Mean Chg   Art. Visc."
UNITS = "        (cm)       (cm)       (g/cm3)    (cm/s)     (eV)       (eV)       (eV)       (J/cm3)    (J/cm3)    (J/cm3)     (rho/rho0)  (1/cm3)              (J/cm3)   "


def _block(rows: int = 3, *, crlf: bool = False) -> list[str]:
    """A primary-shaped table: zone 0 row plus `rows` zone rows of 14 columns."""
    newline = "\r" if crlf else ""
    lines = [
        "     0  0.000E+00                        0.000E+00  0.000E+00" + newline,
    ]
    for zone in range(1, rows + 1):
        values = [f"{zone}.0E+00" for _ in range(14)]
        # Zone centres and a uniform width, so the coordinate guard accepts the
        # block: centre = (zone - 0.5) * width, boundary edge 0.
        values[0] = f"{(zone - 0.5) * 1.0E-08:.3E}"
        values[1] = "1.000E-08"
        lines.append(f"{zone:6d}  " + "  ".join(values) + newline)
    return lines


def _parse(data_lines: list[str], zones: int = 3) -> dict:
    arrays, _units, _raw, extras = _parse_field_table_rows(
        data_lines, HEADER, UNITS, zones, LOGGER
    )
    return {"arrays": arrays, "extras": extras}


class NativeKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        # The release bundles ship the library; a source checkout may not. Every
        # test here calls the kernel directly, so skip the class rather than
        # failing on a clean clone. `ParserFallbackTests` covers the other path.
        if not native_table.available():
            self.skipTest("native library not built; run scripts/build_native_table.py")

    def test_library_is_loaded(self) -> None:
        self.assertTrue(native_table.available())

    def test_describe_names_the_active_path(self) -> None:
        text = native_table.describe()
        self.assertTrue(text.startswith("on ("), text)
        self.assertIn("libhelios_native", text)

    def test_kernel_matches_fromstring(self) -> None:
        lines = _block()
        matrix, rows = native_table.parse_table_block("\n".join(lines), 3, 14)
        self.assertEqual(rows, 3)
        # Independent reference: numpy's own parser on the filtered text.
        filtered = [line.lstrip() for line in lines if not line.lstrip().startswith("0 ")]
        values = np.fromstring("\n".join(filtered), sep=" ", dtype=np.float64).reshape(-1, 15)
        expected = np.full((3, 14), np.nan)
        expected[values[:, 0].astype(int) - 1, :] = values[:, 1:]
        np.testing.assert_array_equal(matrix, expected)

    def test_zone_zero_and_out_of_range_rows_are_dropped(self) -> None:
        lines = _block()
        lines.append("     9  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00  1.0E+00")
        matrix, rows = native_table.parse_table_block("\n".join(lines), 3, 14)
        self.assertEqual(rows, 3)
        self.assertTrue(np.all(np.isfinite(matrix)))

    def test_width_mismatch_returns_none(self) -> None:
        lines = ["     1  1.0E+00", "     2  1.0E+00  2.0E+00"]
        self.assertIsNone(native_table.parse_table_block("\n".join(lines), 2, 2))

    def test_unparseable_token_returns_none(self) -> None:
        lines = ["     1  not-a-number  1.0E+00"]
        self.assertIsNone(native_table.parse_table_block("\n".join(lines), 1, 2))

    def test_absurd_shape_returns_none(self) -> None:
        self.assertIsNone(native_table.parse_table_block("1 2", 100_000, 100_000))


class ParserFallbackTests(unittest.TestCase):
    def test_native_path_matches_fallback(self) -> None:
        lines = _block()
        native_result = _parse(lines)
        with mock.patch("helios_parser.parser.parse_table_block", return_value=None):
            fallback_result = _parse(lines)
        for name, values in native_result["arrays"].items():
            np.testing.assert_array_equal(values, fallback_result["arrays"][name])
        self.assertEqual(
            native_result["extras"]["parsed_row_count"],
            fallback_result["extras"]["parsed_row_count"],
        )

    def test_fallback_handles_malformed_exponent(self) -> None:
        lines = [
            "     0  0.000E+00" + "  0.000E+00" * 13,
            "     1  5.000E-09  1.000E-08  2.978-116  4.000E+00  5.000E+00  6.000E+00"
            "  7.000E+00  8.000E+00  9.000E+00  1.000E+01  1.100E+01  1.200E+01  1.300E+01  1.400E+01",
        ]
        result = _parse(lines, zones=1)
        self.assertAlmostEqual(result["arrays"]["density"][0], 2.978e-116)

    def test_long_fractional_token_keeps_its_value(self) -> None:
        # Twenty fractional digits: a mantissa that stops at nineteen would
        # collapse this to zero, so the value is the guard on both paths.
        lines = [
            "     0  0.000E+00" + "  0.000E+00" * 13,
            "     1  5.000E-09  1.000E-08  1.000E-02  4.000E+00  5.000E+00  6.000E+00"
            "  7.000E+00  8.000E+00  9.000E+00  1.000E+01  1.100E+01  1.200E+01  1.300E+01  0.00000000000000000001",
        ]
        result = _parse(lines, zones=1)
        self.assertEqual(result["arrays"]["artificial_viscosity"][0], 1e-20)

    def test_oversized_zone_index_is_dropped_on_both_paths(self) -> None:
        # 2**32 + 1 narrows to zone 1 under an int32 cast, so this row must be
        # dropped rather than overwrite the real zone 1 row.
        lines = [
            "     0  0.000E+00" + "  0.000E+00" * 13,
            "     1  5.000E-09  1.000E-08  1.000E+00  4.000E+00  5.000E+00  6.000E+00"
            "  7.000E+00  8.000E+00  9.000E+00  1.000E+01  1.100E+01  1.200E+01  1.300E+01  1.400E+01",
            "4294967297  5.000E-09  1.000E-08  9.000E+00  4.000E+00  5.000E+00  6.000E+00"
            "  7.000E+00  8.000E+00  9.000E+00  1.000E+01  1.100E+01  1.200E+01  1.300E+01  1.400E+01",
        ]
        native_result = _parse(lines, zones=1)
        self.assertEqual(native_result["arrays"]["density"][0], 1.0)
        with mock.patch("helios_parser.parser.parse_table_block", return_value=None):
            fallback_result = _parse(lines, zones=1)
        self.assertEqual(fallback_result["arrays"]["density"][0], 1.0)
        self.assertEqual(fallback_result["extras"]["parsed_row_count"], 1)

    def test_crlf_block_parses(self) -> None:
        lines = _block(crlf=True)
        native_result = _parse(lines)
        with mock.patch("helios_parser.parser.parse_table_block", return_value=None):
            fallback_result = _parse(lines)
        for name, values in native_result["arrays"].items():
            np.testing.assert_array_equal(values, fallback_result["arrays"][name])

    def test_missing_library_is_not_an_import_error(self) -> None:
        with mock.patch.object(native_table, "_LIBRARY", None), mock.patch.object(
            native_table, "_DESCRIPTION", None
        ):
            self.assertFalse(native_table.available())
            self.assertIsNone(native_table.parse_table_block("     1  1.0\n", 1, 1))
            self.assertEqual(native_table.describe(), "off (Python fallback, np.fromstring)")


if __name__ == "__main__":
    unittest.main()