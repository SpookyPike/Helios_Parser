"""Frequency-resolved radiation output: the reader, the HDF5 schema, the helpers.

WHY THIS SUITE EXISTS.  `Write freq-dependent radiation output = 1` adds 18
datasets to the Exodus container, and they split over two different time axes:
the four node-centred spectral arrays are sampled on `time_freq`, the boundary and
interface fluxes and every integrated scalar on `time_whole`.  Reading either
against the wrong axis produces a plausible number and no error, so the tests here
assert the axes and the shapes rather than a sample value.

The fixtures are real HELIOS 12.0.0 output from the campaign decks beside them
(`.rhc`), not synthetic data:

- `cu_freqrad_tiny_t0p02ns.exo`, 100 zones, one spectral dump, one region.
- `stack_freqrad_t0p2ns.exo`, 184 zones, two spectral dumps, three regions, so
  `n_bd = 4` and the interface ordering is testable.
- `cu_freqrad_t0p3ns.exo`, 300 zones, six spectral dumps over 31 hydro dumps. It
  is 26 MB, so it stays a local-only fixture and its test is behind the slow gate.

TWO MEASUREMENTS THAT PIN THE CONVENTIONS, both asserted below:

1. The node dimension of the spectral arrays is `n_zones + 2`, and rows 0 and
   `n_zones + 1` are exactly zero.  The interior rows are the zone centres, in
   Rmin to Rmax order: `sum over groups of energy_density` equals
   `rad_energy_density[interior]`, and that equals `3 * rad_pressure`.
2. Group integration uses rectangle weights.  `sum(FdIntgPowerLossRmin * dE)`
   reproduces `FreqIntgRadEnLossRmin` to 1e-16 relative; trapezoid weights are off
   by 1.5 % on that boundary, which is what the guard below measures.  The
   requirement document proposed trapezoid and cross-checked against
   `TotalFluxRmin`, but `Freq. resolved incident flux at Rmin` is identically zero
   in all three fixtures, so that check cannot hold.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import _test_bootstrap  # noqa: F401
from _slow_tests import slow
from helios.services.derived import integrate_over_groups, spectrum_at, spectrum_map
from helios_parser.exodus import (
    GROUP_BOUNDS_DATASET,
    RADIATION_TIME_DATASET,
    SPECTRAL_NODE_FIELDS,
    ExodusFormatError,
    read_exodus,
)
from helios_parser.hdf5 import write_hdf5

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RADIATION = FIXTURES / "radiation"
TINY_EXO = RADIATION / "cu_freqrad_tiny_t0p02ns.exo"
STACK_EXO = RADIATION / "stack_freqrad_t0p2ns.exo"
LONG_EXO = RADIATION / "cu_freqrad_t0p3ns.exo"
NO_RADIATION_EXO = FIXTURES / "v12" / "cu_v12_slice.exo"

ENERGY_DENSITY_DATASET = "Freq. resolved radiation energy density for nodes [J per (cm3.group)]"


def _convert(source: Path, directory: Path) -> Path:
    target = directory / f"{source.stem}.h5"
    write_hdf5(source, target)
    return target


class RadiationReadTests(unittest.TestCase):
    """The reader, on the 4 MB single-region fixture."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.exodus_run = read_exodus(TINY_EXO)

    def test_the_spectral_block_is_read_with_zone_centred_nodes(self) -> None:
        radiation = self.exodus_run.radiation
        self.assertIsNotNone(radiation)
        self.assertEqual(self.exodus_run.n_zones, 100)
        for name in ("energy_density", "heating_rate", "cooling_rate", "rosseland_opacity"):
            self.assertIn(name, radiation.spectral.fields)
            self.assertEqual(radiation.spectral.fields[name].shape, (1, 100, 250), name)
        self.assertEqual(radiation.spectral.node_axis_cm.shape, (100,))
        np.testing.assert_allclose(radiation.spectral.node_axis_cm, self.exodus_run.coordinate_center_cm[0])

    def test_the_two_time_axes_differ_and_are_not_prefix_equal(self) -> None:
        """The main design trap: `time_freq` is not a prefix of `time_whole`."""
        radiation = self.exodus_run.radiation
        freq = radiation.spectral.time_s
        whole = radiation.flux.time_s
        self.assertNotEqual(freq.size, whole.size)
        self.assertEqual(whole.size, self.exodus_run.n_snapshots)
        np.testing.assert_array_equal(radiation.flux.time_s, self.exodus_run.time_s)
        self.assertFalse(np.array_equal(freq, whole[: freq.size]))

    def test_the_spectral_axis_pairs_with_the_hydro_axis_by_index(self) -> None:
        radiation = self.exodus_run.radiation
        np.testing.assert_array_equal(
            radiation.flux.time_s[radiation.spectral.hydro_index],
            radiation.spectral.time_s,
        )

    def test_the_group_grid_brackets_the_group_values(self) -> None:
        radiation = self.exodus_run.radiation
        edges = radiation.spectral.group_edges_ev
        self.assertEqual(edges.shape, (1, 251))
        self.assertEqual(radiation.spectral.group_centre_ev.shape, (1, 250))
        self.assertTrue(np.all(np.diff(edges, axis=1) > 0))
        centre = radiation.spectral.group_centre_ev[0]
        np.testing.assert_allclose(centre, np.sqrt(edges[0, :-1] * edges[0, 1:]))
        self.assertAlmostEqual(float(edges[0, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(edges[0, -1]) / 1e6, 1.0, places=5)

    def test_a_single_region_has_two_boundaries_and_one_region_index(self) -> None:
        radiation = self.exodus_run.radiation
        self.assertEqual(radiation.n_boundaries, 2)
        self.assertEqual(radiation.n_regions, 1)
        self.assertEqual(radiation.flux.fields["net_interface"].shape, (4, 2, 250))
        self.assertEqual(radiation.flux.region_boundary_node_index.shape, (1,))
        # 1-based node count 101 -> 0-based index of the last node.
        self.assertEqual(int(radiation.flux.region_boundary_node_index[0]), 100)

    def test_a_run_without_the_flag_carries_no_radiation_block(self) -> None:
        run = read_exodus(NO_RADIATION_EXO)
        self.assertIsNone(run.radiation)
        self.assertNotIn(RADIATION_TIME_DATASET, h5py.File(NO_RADIATION_EXO, "r"))

    def test_a_spectral_array_on_the_wrong_axis_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "transposed.exo"
            shutil.copyfile(TINY_EXO, copy)
            with h5py.File(copy, "r+") as handle:
                values = np.asarray(handle[ENERGY_DENSITY_DATASET][()])
                del handle[ENERGY_DENSITY_DATASET]
                handle.create_dataset(ENERGY_DENSITY_DATASET, data=np.transpose(values, (1, 0, 2)))
            with self.assertRaises(ExodusFormatError):
                read_exodus(copy)

    def test_a_non_empty_node_padding_row_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "padded.exo"
            shutil.copyfile(TINY_EXO, copy)
            with h5py.File(copy, "r+") as handle:
                dataset = handle[ENERGY_DENSITY_DATASET]
                dataset[0, 0, :] = 1.0
            with self.assertRaises(ExodusFormatError):
                read_exodus(copy)

    def test_a_missing_photon_grid_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "nogrid.exo"
            shutil.copyfile(TINY_EXO, copy)
            with h5py.File(copy, "r+") as handle:
                del handle[GROUP_BOUNDS_DATASET]
                del handle["photon_energy_group_bounds_singleArrayNonCR"]
            with self.assertRaises(ExodusFormatError):
                read_exodus(copy)


class RadiationZoneAlignmentTests(unittest.TestCase):
    """The interior rows really are the zone centres, on both small fixtures."""

    def test_the_group_sum_reproduces_the_zone_radiation_energy_density(self) -> None:
        for source in (TINY_EXO, STACK_EXO):
            with self.subTest(fixture=source.name):
                run = read_exodus(source)
                radiation = run.radiation
                edges = radiation.spectral.group_edges_ev
                grouped = radiation.spectral.fields["energy_density"].sum(axis=2)
                with h5py.File(source, "r") as handle:
                    reference = np.asarray(handle["rad_energy_density"][()], dtype=np.float64)
                np.testing.assert_allclose(
                    grouped,
                    reference[radiation.spectral.hydro_index][:, 1:-1],
                    rtol=1e-12,
                    atol=0.0,
                )
                self.assertEqual(edges.shape[0], radiation.n_freq_snapshots)

    def test_the_spectral_energy_density_is_three_times_the_radiation_pressure(self) -> None:
        run = read_exodus(TINY_EXO)
        radiation = run.radiation
        grouped = radiation.spectral.fields["energy_density"].sum(axis=2)
        pressure = run.fields["pressure_radiation"][radiation.spectral.hydro_index]
        np.testing.assert_allclose(grouped, 3.0 * pressure, rtol=1e-6)


class MultiRegionRadiationTests(unittest.TestCase):
    """`n_bd = 4` on the three-region stack, and the Rmin to Rmax ordering."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.exodus_run = read_exodus(STACK_EXO)
        cls.radiation = cls.exodus_run.radiation

    def test_three_regions_give_four_boundaries(self) -> None:
        self.assertEqual(self.radiation.n_regions, 3)
        self.assertEqual(self.radiation.n_boundaries, 4)
        self.assertEqual(self.radiation.flux.fields["net_interface"].shape, (6, 4, 250))
        self.assertEqual(self.radiation.integrated["interface_flux"].shape, (6, 4))
        self.assertEqual(self.radiation.integrated["net_cooling_by_region"].shape, (6, 3))

    def test_the_interface_rows_run_from_rmin_to_rmax(self) -> None:
        """Measured identity: first row is -TotalFluxRmin, last is +TotalFluxRmax."""
        flux = self.radiation.integrated["interface_flux"]
        np.testing.assert_allclose(flux[:, 0], -self.radiation.integrated["total_flux_rmin"])
        np.testing.assert_allclose(flux[:, -1], self.radiation.integrated["total_flux_rmax"])

    def test_the_region_boundary_indices_are_node_indices(self) -> None:
        index = self.radiation.flux.region_boundary_node_index
        self.assertEqual(index.shape, (3,))
        self.assertTrue(np.all(np.diff(index) > 0))
        self.assertEqual(int(index[-1]), self.exodus_run.n_zones)


class RadiationHdf5Tests(unittest.TestCase):
    """The published schema: two groups, the on/off marker, the helper inputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        directory = Path(cls._directory.name)
        cls.tiny_h5 = _convert(TINY_EXO, directory)
        cls.off_h5 = _convert(NO_RADIATION_EXO, directory)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_both_groups_are_published_with_the_documented_shapes(self) -> None:
        with h5py.File(self.tiny_h5) as handle:
            self.assertEqual(handle["radiation/time"].shape, (1,))
            self.assertEqual(handle["radiation/group_edges"].shape, (1, 251))
            self.assertEqual(handle["radiation/group_centre"].shape, (1, 250))
            self.assertEqual(handle["radiation/node_axis"].shape, (100,))
            for name in ("energy_density", "heating_rate", "cooling_rate", "rosseland_opacity"):
                self.assertEqual(handle[f"radiation/{name}"].shape, (1, 100, 250), name)
            self.assertEqual(handle["flux/time"].shape, (4,))
            self.assertEqual(handle["flux/group_edges"].shape, (4, 251))
            self.assertEqual(handle["flux/incident_rmax"].shape, (4, 250))
            self.assertEqual(handle["flux/net_interface"].shape, (4, 2, 250))
            self.assertEqual(handle["flux/region_boundary_node_index"].shape, (1,))
            for name in (
                "total_energy",
                "loss_rmax",
                "loss_rmin",
                "total_flux_rmax",
                "total_flux_rmin",
                "loss_at_boundaries",
                "loss_at_boundaries_time_integral",
                "electron_to_radiation_integral",
            ):
                self.assertEqual(handle[f"radiation/integrated/{name}"].shape, (4,), name)
            self.assertEqual(handle["radiation/integrated/net_cooling_by_region"].shape, (4, 1))
            self.assertEqual(handle["radiation/integrated/interface_flux"].shape, (4, 2))

    def test_units_are_carried_as_dataset_attributes(self) -> None:
        with h5py.File(self.tiny_h5) as handle:
            expected = {
                "radiation/time": "s",
                "radiation/group_edges": "eV",
                "radiation/node_axis": "cm",
                "radiation/energy_density": "J/cm3/group",
                "radiation/rosseland_opacity": "cm2/g",
                "flux/incident_rmax": "J/(cm2 s eV)",
                "flux/net_interface": "J/(cm2 s eV)",
                "radiation/integrated/interface_flux": "J/(cm2 s)",
            }
            for path, unit in expected.items():
                self.assertEqual(handle[path].attrs["unit"], unit, path)

    def test_the_metadata_sizes_the_arrays_without_opening_them(self) -> None:
        with h5py.File(self.tiny_h5) as handle:
            self.assertEqual(handle["metadata/radiation_output"][()].decode(), "on")
            self.assertEqual(int(handle["metadata/n_groups"][()]), 250)
            self.assertEqual(int(handle["metadata/n_freq_snapshots"][()]), 1)
            self.assertEqual(int(handle["metadata/n_boundaries"][()]), 2)
            self.assertEqual(int(handle["metadata/n_regions"][()]), 1)

    def test_a_run_without_the_flag_says_so_and_publishes_no_groups(self) -> None:
        with h5py.File(self.off_h5) as handle:
            self.assertEqual(handle["metadata/radiation_output"][()].decode(), "off")
            self.assertNotIn("radiation", handle)
            self.assertNotIn("flux", handle)
            self.assertNotIn("n_groups", handle["metadata"])

    def test_the_published_file_still_satisfies_the_reader(self) -> None:
        """The new groups must not break the required-group contract."""
        with h5py.File(self.tiny_h5) as handle:
            for group in ("grid", "time", "fields", "regions", "materials", "diagnostics", "metadata"):
                self.assertIn(group, handle)


class RadiationHelperTests(unittest.TestCase):
    """`integrate_over_groups` reproduces HELIOS's own integrated scalars."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        directory = Path(cls._directory.name)
        cls.stack_h5 = _convert(STACK_EXO, directory)
        cls.tiny_h5 = _convert(TINY_EXO, directory)
        cls.off_h5 = _convert(NO_RADIATION_EXO, directory)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_group_integration_reproduces_the_stored_scalar(self) -> None:
        with h5py.File(self.stack_h5) as handle:
            edges = handle["flux/group_edges"][()]
            for side, stored in (("rmin", "loss_rmin"), ("rmax", "loss_rmax")):
                with self.subTest(side=side):
                    values = handle[f"flux/power_loss_{side}_integrated"][()]
                    reference = handle[f"radiation/integrated/{stored}"][()]
                    np.testing.assert_allclose(
                        integrate_over_groups(values, edges),
                        reference,
                        rtol=1e-12,
                        atol=0.0,
                    )

    def test_trapezoid_weights_would_be_measurably_wrong(self) -> None:
        """Guard the guard: the convention is measured, not assumed.

        Measured on the reference fixtures, the rectangle rule is exact to 1e-16
        and trapezoid weights are off by 1.5 % on the Rmin boundary.  Asserting
        both directions keeps the choice from being read as arbitrary.
        """
        with h5py.File(self.tiny_h5) as handle:
            edges = handle["flux/group_edges"][()]
            values = handle["flux/power_loss_rmin_integrated"][()]
            reference = handle["radiation/integrated/loss_rmin"][()]
        scale = np.abs(reference).max()
        rectangle = integrate_over_groups(values, edges)
        self.assertLess(np.abs(rectangle - reference).max() / scale, 1e-12)
        widths = np.diff(edges, axis=1)
        trapezoid = 0.5 * (values[:, :-1] * widths[:, :-1] + values[:, 1:] * widths[:, 1:]).sum(axis=1)
        self.assertGreater(np.abs(trapezoid - reference).max() / scale, 1e-3)

    def test_a_mismatched_edge_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            integrate_over_groups(np.zeros((2, 250)), np.zeros(250))

    def test_spectrum_at_returns_the_sample_time_it_used(self) -> None:
        with h5py.File(self.stack_h5) as handle:
            times = handle["flux/time"][()]
            edges = handle["flux/group_edges"][()]
            expected = handle["flux/incident_rmax"][()]
        sample_time, group_edges, spectrum = spectrum_at(self.stack_h5, 1.0e-12, side="rmax")
        index = int(np.abs(times - 1.0e-12).argmin())
        self.assertEqual(sample_time, float(times[index]))
        np.testing.assert_array_equal(group_edges, edges[index])
        np.testing.assert_array_equal(spectrum, expected[index])
        self.assertNotEqual(sample_time, 1.0e-12)

    def test_spectrum_map_returns_the_axes_a_contour_needs(self) -> None:
        time_s, group_edges, values = spectrum_map(self.stack_h5, side="rmin")
        self.assertEqual(values.shape, (6, 250))
        self.assertEqual(time_s.shape, (6,))
        self.assertEqual(group_edges.shape, (6, 251))

    def test_the_net_quantity_reads_the_boundary_row(self) -> None:
        """`net` is the informative one: the incident arrays are zero in the fixtures."""
        time_s, group_edges, values = spectrum_map(self.stack_h5, side="rmax", quantity="net")
        with h5py.File(self.stack_h5) as handle:
            expected = handle["flux/net_interface"][:, -1, :]
            self.assertEqual(time_s.shape, (6,))
            self.assertEqual(group_edges.shape, (6, 251))
        np.testing.assert_array_equal(values, expected)
        _, _, rmin_values = spectrum_map(self.stack_h5, side="rmin", quantity="net")
        with h5py.File(self.stack_h5) as handle:
            np.testing.assert_array_equal(rmin_values, handle["flux/net_interface"][:, 0, :])

    def test_an_unknown_quantity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            spectrum_map(self.stack_h5, side="rmax", quantity="gross")

    def test_a_bad_side_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            spectrum_at(self.stack_h5, 0.0, side="left")

    def test_a_run_without_radiation_is_rejected_with_a_reason(self) -> None:
        with self.assertRaises(ValueError):
            spectrum_map(self.off_h5)


@slow("reads the 26 MB six-spectral-dump fixture")
@unittest.skipUnless(LONG_EXO.exists(), "local-only fixture tests/fixtures/radiation/cu_freqrad_t0p3ns.exo")
class LongRadiationRunTests(unittest.TestCase):
    """The milestone gate: six spectral dumps over 31 hydro dumps."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.exodus_run = read_exodus(LONG_EXO)

    def test_the_axis_asymmetry_is_real_at_production_length(self) -> None:
        radiation = self.exodus_run.radiation
        self.assertEqual(self.exodus_run.n_snapshots, 31)
        self.assertEqual(radiation.n_freq_snapshots, 6)
        self.assertEqual(radiation.spectral.fields["energy_density"].shape, (6, 300, 250))
        self.assertEqual(radiation.flux.fields["net_interface"].shape, (31, 2, 250))
        np.testing.assert_array_equal(radiation.spectral.hydro_index, np.arange(4, 30, 5))

    def test_the_spectrum_versus_time_map_has_the_spectral_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = _convert(LONG_EXO, Path(directory))
            time_s, group_edges, values = spectrum_map(target, side="rmax")
            self.assertEqual(values.shape, (31, 250))
            self.assertEqual(time_s.shape, (31,))
            self.assertEqual(group_edges.shape, (31, 251))
            with h5py.File(target) as handle:
                self.assertEqual(int(handle["metadata/n_freq_snapshots"][()]), 6)
                self.assertEqual(int(handle["metadata/n_groups"][()]), 250)


if __name__ == "__main__":
    unittest.main()