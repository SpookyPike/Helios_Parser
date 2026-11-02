"""HELIOS 12.0.0 output support: the log path, the Exodus path, and their agreement.

WHY THIS SUITE EXISTS.  12.0.0 changed two things that matter to a parser:

1. The log prints a whole ``Time Step Controller`` block and a ``Run Time:`` line
   between the cycle line and the state table.  A reader that assumes the table
   follows the cycle line immediately reads ZERO state dumps from every 12.0.0
   log and reports a successful parse with no data.  That failure is silent, so
   it gets a test rather than a comment.

2. The default plot output is an Exodus II container (``Plot file type = 0``),
   not a BPF.  An Exodus file written by HELIOS is NetCDF-4, i.e. HDF5, and its
   ``fluid_velocity`` is stored at the NODES while every other field is
   zone-centred.  The mapping used here is ``[:, 1:]`` -- the zone's upper node --
   which is what the log prints.  Averaging the two nodes is off by 27 %, so the
   mapping is asserted against the same run's log, not against itself.

The fixtures are a PREFIX and a SLICE of one real run, not synthetic data:
``cu_v12_prefix.log`` is the header plus the first ten snapshots of
``cu-bare-250um-6ns_eta0.1320.log``, and ``cu_v12_slice.exo`` is the first ten
time steps of the core hydrodynamic fields of the matching ``.exo``.  Because
they come from the same run, the log and the Exodus file are independent
encodings of the same numbers, so their agreement is a real check.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import _test_bootstrap  # noqa: F401
from helios_parser.exodus import (
    EDGE_DATASET,
    VELOCITY_DATASET,
    ExodusFormatError,
    looks_like_exodus,
    read_exodus,
)
from helios_parser.hdf5 import SUPPORTED_INPUT_SUFFIXES, resolve_parse_sources, write_hdf5
from helios_parser.parser import HeliosParser

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "v12"
LOG_FIXTURE = FIXTURES / "cu_v12_prefix.log"
EXO_FIXTURE = FIXTURES / "cu_v12_slice.exo"

EXPECTED_SNAPSHOTS = 10
EXPECTED_ZONES = 300


def _cycle_header_count(path: Path) -> int:
    total = 0
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if "Cycle" in line and "Time (s)" in line and "controlled by" in line:
                total += 1
    return total


class Helios12LogTests(unittest.TestCase):
    def test_the_fixture_really_carries_the_v12_controller_block(self) -> None:
        """Guard the guard: if the block were absent the snapshot test is vacuous."""
        text = LOG_FIXTURE.read_text(errors="replace")
        self.assertIn("(ver. 12.0.0)", text)
        self.assertIn("Time Step Controller:", text)
        self.assertIn("Run Time:", text)

    def test_every_snapshot_in_a_v12_log_is_parsed(self) -> None:
        """The 12.0.0 controller block must not swallow the state tables."""
        parser = HeliosParser()
        simulation = parser.parse(LOG_FIXTURE)
        self.assertEqual(simulation.metadata.get("helios_version"), "12.0.0")
        self.assertEqual(len(simulation.time["time"]), EXPECTED_SNAPSHOTS)
        self.assertEqual(_cycle_header_count(LOG_FIXTURE), EXPECTED_SNAPSHOTS)
        for name in ("density", "velocity", "temperature_i", "temperature_e"):
            self.assertIn(name, simulation.fields)
            self.assertEqual(simulation.fields[name].shape, (EXPECTED_SNAPSHOTS, EXPECTED_ZONES))

    def test_v12_run_status_is_not_reported_as_a_completed_run(self) -> None:
        """A prefix has no footer; the parser must say so instead of claiming success."""
        parser = HeliosParser()
        simulation = parser.parse(LOG_FIXTURE)
        status = simulation.run_status
        self.assertIsNotNone(status)
        self.assertNotEqual(status.state, "completed")


class ExodusDetectionTests(unittest.TestCase):
    def test_exodus_is_detected_and_a_log_is_not(self) -> None:
        self.assertTrue(looks_like_exodus(EXO_FIXTURE))
        self.assertFalse(looks_like_exodus(LOG_FIXTURE))

    def test_exodus_is_an_accepted_input_suffix(self) -> None:
        self.assertIn(".exo", SUPPORTED_INPUT_SUFFIXES)
        selection = resolve_parse_sources(EXO_FIXTURE)
        self.assertEqual(selection.mode, "log_only")

    def test_a_plain_hdf5_file_is_rejected_with_a_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bogus = Path(directory) / "not_helios.exo"
            with h5py.File(bogus, "w") as handle:
                handle.create_dataset("something", data=np.zeros(3))
            self.assertFalse(looks_like_exodus(bogus))
            with self.assertRaises(ExodusFormatError):
                read_exodus(bogus)


class ExodusReadTests(unittest.TestCase):
    def test_core_fields_and_geometry(self) -> None:
        run = read_exodus(EXO_FIXTURE)
        self.assertEqual(run.n_snapshots, EXPECTED_SNAPSHOTS)
        self.assertEqual(run.n_zones, EXPECTED_ZONES)
        self.assertEqual(run.coordinate_name, "x")
        self.assertEqual(run.geometry, "planar")
        self.assertEqual(run.coordinate_edge_cm.shape, (EXPECTED_SNAPSHOTS, EXPECTED_ZONES + 1))
        self.assertEqual(run.coordinate_center_cm.shape, (EXPECTED_SNAPSHOTS, EXPECTED_ZONES))
        for name in (
            "density",
            "velocity",
            "temperature_i",
            "temperature_e",
            "temperature_radiation",
            "pressure_i",
            "pressure_e",
            "pressure_radiation",
            "plasma_pressure_j_cm3",
            "total_pressure_j_cm3",
        ):
            self.assertIn(name, run.fields, name)
            self.assertEqual(run.fields[name].shape, (EXPECTED_SNAPSHOTS, EXPECTED_ZONES))

    def test_velocity_is_the_upper_node_not_the_node_mean(self) -> None:
        """The one non-obvious mapping.  Averaging is off by ~27 %, so assert both ways."""
        with h5py.File(EXO_FIXTURE, "r") as handle:
            nodal = np.asarray(handle[VELOCITY_DATASET][()], dtype=np.float64)
        run = read_exodus(EXO_FIXTURE)
        np.testing.assert_array_equal(run.fields["velocity"], nodal[:, 1:])
        mean = 0.5 * (nodal[:, :-1] + nodal[:, 1:])
        scale = np.abs(nodal).max()
        self.assertGreater(np.abs(mean - run.fields["velocity"]).max() / scale, 0.05)

    def test_derived_pressures_follow_the_log_path_definition(self) -> None:
        run = read_exodus(EXO_FIXTURE)
        plasma = run.fields["pressure_i"] + run.fields["pressure_e"]
        total = plasma + run.fields["pressure_radiation"]
        np.testing.assert_allclose(run.fields["plasma_pressure_j_cm3"], plasma)
        np.testing.assert_allclose(run.fields["total_pressure_j_cm3"], total)
        np.testing.assert_allclose(run.fields["pressure"], total)

    def test_cylindrical_geometry_maps_to_the_radial_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "cyl.exo"
            shutil.copyfile(EXO_FIXTURE, copy)
            with h5py.File(copy, "r+") as handle:
                handle.attrs["Geometry ID (0 = planar, 1 = cylindrical, 2 = spherical)"] = np.int32(1)
            run = read_exodus(copy)
            self.assertEqual(run.coordinate_name, "r")
            self.assertEqual(run.geometry, "r")


class ExodusAgainstLogTests(unittest.TestCase):
    """The two encodings of one run must agree to the log's printing precision.

    The log prints five significant digits, so ~5e-4 relative is the floor.  A
    larger difference means a mapping is wrong, not that the code is imprecise.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        directory = Path(cls._directory.name)
        cls._log_h5 = directory / "from_log.h5"
        cls._exo_h5 = directory / "from_exodus.h5"
        write_hdf5(LOG_FIXTURE, cls._log_h5)
        write_hdf5(EXO_FIXTURE, cls._exo_h5)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_time_axes_agree(self) -> None:
        with h5py.File(self._log_h5) as log, h5py.File(self._exo_h5) as exo:
            log_time = log["time/time"][()]
            exo_time = exo["time/time"][()]
            self.assertEqual(log_time.shape, exo_time.shape)
            np.testing.assert_allclose(log_time, exo_time, rtol=1e-12, atol=0.0)
            np.testing.assert_array_equal(log["time/cycle"][()], exo["time/cycle"][()])

    def test_hydrodynamic_fields_agree_to_the_log_precision(self) -> None:
        with h5py.File(self._log_h5) as log, h5py.File(self._exo_h5) as exo:
            for name in ("density", "velocity", "temperature_i", "temperature_e", "pressure_i", "pressure_e"):
                with self.subTest(field=name):
                    from_log = log[f"fields/{name}"][()]
                    from_exo = exo[f"fields/{name}"][()]
                    scale = np.abs(from_log).max()
                    relative = np.abs(from_log - from_exo).max() / scale
                    self.assertLess(relative, 1e-3)

    def test_the_published_file_records_where_it_came_from(self) -> None:
        with h5py.File(self._log_h5) as log, h5py.File(self._exo_h5) as exo:
            self.assertEqual(log.attrs["helios_version"], "12.0.0")
            # The Exodus container does not record the code version, so the field
            # is empty on purpose rather than guessed.
            self.assertEqual(exo.attrs["helios_version"], "")
            self.assertEqual(exo["metadata/run_status/state"][()].decode(), "exodus_container")
            self.assertEqual(int(exo["metadata/n_snapshots"][()]), EXPECTED_SNAPSHOTS)


if __name__ == "__main__":
    unittest.main()