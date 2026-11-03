"""Validate the converter against the source log, not against itself.

Every other reader test opens a file the writer produced, so a writer bug that
is consistent with itself passes. This test parses the log's own numeric tables
with its own small parser and compares against the HDF5, which makes the log the
oracle.

The log's `Radius` column is a zone EDGE, one row per boundary, so a snapshot has
`n_zones + 1` radius rows. The converter stores those in
`grid/dynamic_coordinate_edge` and the zone centres in
`grid/dynamic_coordinate_center`. That distinction is what this test pins: the
old golden fixtures stored edges where centres belonged, and nothing noticed
because every test compared the writer with itself.

Precision note: the log prints four significant digits, so the coordinates are
compared with an absolute tolerance derived from the value scale rather than
bit-for-bit. The physics columns are printed with the same four digits but the
converter copies them verbatim, so those compare exactly.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
import h5py
import numpy as np

from helios_parser import write_hdf5

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "5Fe+4.9TW+light.log"

# Column order after `Radius` in the per-snapshot table header.
FIELD_COLUMNS = {
    "density": "Mass dens",
    "velocity": "Velocity",
    "temperature_e": "Elec temp",
    "temperature_i": "Ion temp",
    "electron_density": "Elec Dens",
    "mean_charge": "Mean Chg",
}
# Index into the row after the zone index and the radius, i.e. parts[2 + index].
# Order in the table: width, Mass dens, Velocity, Rad temp, Ion temp, Elec temp,
# Ion press, Elec press, Rad press, Compression, Elec Dens, Mean Chg, Art. Visc.
COLUMN_INDEX = {
    "density": 1,
    "velocity": 2,
    "temperature_i": 4,
    "temperature_e": 5,
    "electron_density": 10,
    "mean_charge": 11,
}


def _parse_snapshot(lines: list[str], header_index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (radius rows, {field: column}) for one snapshot table.

    Rows start at zone 0, which is the inner boundary and carries no field
    values, so the field arrays drop it and the radius array keeps it.
    """

    radius: list[float] = []
    columns: dict[str, list[float]] = {name: [] for name in FIELD_COLUMNS}
    for line in lines[header_index + 1 :]:
        parts = line.split()
        if not parts:
            continue
        if not parts[0].isdigit():
            if radius:
                break
            continue
        zone = int(parts[0])
        if zone == 0 and radius:
            break
        radius.append(float(parts[1]))
        if zone == 0:
            # The inner boundary row carries a radius and no field values.
            continue
        for name, index in COLUMN_INDEX.items():
            try:
                columns[name].append(float(parts[2 + index]))
            except (IndexError, ValueError):
                columns[name].append(np.nan)
    return np.asarray(radius, dtype=np.float64), {name: np.asarray(v, dtype=np.float64) for name, v in columns.items()}


def _snapshot_headers(lines: list[str]) -> list[int]:
    return [index for index, line in enumerate(lines) if line.strip().startswith("#    Radius")]


class LogIsTheOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SOURCE.is_file():
            raise unittest.SkipTest(f"source log missing: {SOURCE.name}")
        cls.lines = SOURCE.read_text(encoding="utf-8", errors="ignore").splitlines()
        cls.headers = _snapshot_headers(cls.lines)
        if len(cls.headers) < 2:
            raise unittest.SkipTest("source log carries fewer than two snapshot tables")
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.output = Path(cls._tmpdir.name) / "truth.h5"
        write_hdf5(SOURCE, cls.output, overwrite=True)
        cls.handle = h5py.File(cls.output, "r")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.handle.close()
        cls._tmpdir.cleanup()

    def test_snapshot_count_matches_the_log(self) -> None:
        self.assertEqual(int(self.handle["time/time"].shape[0]), len(self.headers))

    def test_every_physics_column_is_copied_verbatim(self) -> None:
        """The strongest statement available: no rounding, no reinterpretation."""

        for snapshot in (0, len(self.headers) - 1):
            radius, columns = _parse_snapshot(self.lines, self.headers[snapshot])
            n_zones = int(self.handle["fields/density"].shape[1])
            self.assertEqual(radius.size, n_zones + 1, "a snapshot has one radius row per boundary")
            for field_name in FIELD_COLUMNS:
                with self.subTest(snapshot=snapshot, field=field_name):
                    stored = np.asarray(self.handle[f"fields/{field_name}"][snapshot], dtype=np.float64)
                    expected = columns[field_name]
                    self.assertEqual(stored.shape, expected.shape)
                    np.testing.assert_array_equal(stored, expected)

    def test_the_log_radius_column_is_the_stored_edge_grid(self) -> None:
        for snapshot in (0, len(self.headers) - 1):
            radius, _ = _parse_snapshot(self.lines, self.headers[snapshot])
            with self.subTest(snapshot=snapshot):
                stored = np.asarray(self.handle["grid/dynamic_coordinate_edge"][snapshot], dtype=np.float64)
                self.assertEqual(stored.shape, radius.shape)
                scale = float(np.max(np.abs(radius)))
                # The log prints four significant digits, so the tolerance comes
                # from the printed precision, not from the code's arithmetic.
                np.testing.assert_allclose(stored, radius, rtol=0.0, atol=scale * 1e-3)

    def test_the_stored_centre_is_the_midpoint_of_the_log_edges(self) -> None:
        for snapshot in (0, len(self.headers) - 1):
            radius, _ = _parse_snapshot(self.lines, self.headers[snapshot])
            with self.subTest(snapshot=snapshot):
                centre = np.asarray(self.handle["grid/dynamic_coordinate_center"][snapshot], dtype=np.float64)
                midpoint = 0.5 * (radius[:-1] + radius[1:])
                scale = float(np.max(np.abs(radius)))
                np.testing.assert_allclose(centre, midpoint, rtol=0.0, atol=scale * 1e-3)
                # And the stored radius field is that centre, not the edge.
                radius_field = np.asarray(self.handle["fields/radius"][snapshot], dtype=np.float64)
                np.testing.assert_allclose(radius_field, centre, rtol=1e-12, atol=0.0)

    def test_the_centre_is_not_the_edge(self) -> None:
        """Guard against a regression to the old behaviour, which stored edges.

        Without this, a writer that went back to storing edges would still
        satisfy the midpoint test above only if the edges were also shifted,
        so pin the offset explicitly on the first zone.
        """

        radius, _ = _parse_snapshot(self.lines, self.headers[0])
        centre = np.asarray(self.handle["grid/dynamic_coordinate_center"][0], dtype=np.float64)
        self.assertNotAlmostEqual(float(centre[0]), float(radius[1]), places=12)
        self.assertAlmostEqual(float(centre[0]), float(0.5 * (radius[0] + radius[1])), places=12)


if __name__ == "__main__":
    unittest.main()