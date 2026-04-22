from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import h5py
import numpy as np

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosParser, HeliosPreview, inspect, parse, preview, write_hdf5
from helios_parser.document import normalize_diagnostic_value, reconcile_diagnostic_width


ROOT = Path(__file__).resolve().parents[1]
CORE_FIELDS = {"density", "pressure", "velocity", "temperature_e", "temperature_i"}
DIAGNOSTIC_SECTIONS = {"radiation_boundary_fluxes", "energy_summary", "energy_exchange", "energy_balance"}
INPUT_SECTIONS = {"hydro", "laser_source", "radiation_source", "radiative_transfer", "time_control"}
EXPECTED_FIELDS = {
    "5Fe+4.9TW+light.log": {
        "radius",
        "zone_width",
        "density",
        "velocity",
        "temperature_radiation",
        "temperature_i",
        "temperature_e",
        "pressure_i",
        "pressure_e",
        "pressure_radiation",
        "compression",
        "electron_density",
        "mean_charge",
        "artificial_viscosity",
        "ion_energy",
        "electron_energy",
        "ion_heat_capacity",
        "electron_heat_capacity",
        "radiation_energy",
        "kinetic_energy",
        "radiation_heating",
        "radiation_cooling",
        "radiation_net_heating",
        "laser_source",
        "laser_deposition",
        "pressure",
    },
    "Cu_0166.log": {
        "radius",
        "zone_width",
        "density",
        "velocity",
        "temperature_radiation",
        "temperature_i",
        "temperature_e",
        "pressure_i",
        "pressure_e",
        "pressure_radiation",
        "compression",
        "electron_density",
        "mean_charge",
        "artificial_viscosity",
        "ion_energy",
        "electron_energy",
        "ion_heat_capacity",
        "electron_heat_capacity",
        "radiation_energy",
        "kinetic_energy",
        "radiation_heating",
        "radiation_cooling",
        "radiation_sink",
        "radiation_net_heating",
        "laser_source",
        "laser_deposition",
        "pressure",
    },
    "10ns+10Si+60Al+15Si+4.27TW.log": {
        "radius",
        "zone_width",
        "density",
        "velocity",
        "temperature_radiation",
        "temperature_i",
        "temperature_e",
        "pressure_i",
        "pressure_e",
        "pressure_radiation",
        "compression",
        "electron_density",
        "mean_charge",
        "artificial_viscosity",
        "ion_energy",
        "electron_energy",
        "ion_heat_capacity",
        "electron_heat_capacity",
        "radiation_energy",
        "kinetic_energy",
        "radiation_heating",
        "radiation_cooling",
        "radiation_net_heating",
        "laser_source",
        "laser_deposition",
        "pressure",
    },
}
EXPECTED_COUNTS = {
    "5Fe+4.9TW+light.log": {"zones": 500, "snapshots": 8, "regions": 1, "materials": 1},
    "Cu_0166.log": {"zones": 300, "snapshots": 461, "regions": 1, "materials": 1},
    "10ns+10Si+60Al+15Si+4.27TW.log": {"zones": 1300, "snapshots": 701, "regions": 3, "materials": 2},
    "25Cu+1.4TW.log": {"zones": 50, "snapshots": 901},
    "10ns+15Si+70Al+10Si+4.27TW.log": {"zones": 1300, "snapshots": 701},
}


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = HeliosParser()
        cls.simulations = {
            name: cls.parser.parse(ROOT / name)
            for name in ("5Fe+4.9TW+light.log", "Cu_0166.log", "10ns+10Si+60Al+15Si+4.27TW.log")
        }

    def test_inspect_and_preview_modes(self) -> None:
        for name in ("5Fe+4.9TW+light.log", "Cu_0166.log", "10ns+10Si+60Al+15Si+4.27TW.log"):
            with self.subTest(example=name):
                header = self.parser.inspect(ROOT / name)
                preview = self.parser.preview(ROOT / name)
                self.assertEqual(header.n_zones, EXPECTED_COUNTS[name]["zones"])
                self.assertEqual(header.n_regions, EXPECTED_COUNTS[name]["regions"])
                self.assertEqual(header.n_materials, EXPECTED_COUNTS[name]["materials"])
                self.assertIsInstance(preview, HeliosPreview)
                self.assertEqual(preview.header.n_zones, header.n_zones)
                self.assertIsNotNone(preview.snapshot)
                assert preview.snapshot is not None
                self.assertTrue(CORE_FIELDS.issubset(preview.snapshot.fields))
                self.assertTrue(DIAGNOSTIC_SECTIONS.issubset(preview.snapshot.diagnostics))
                self.assertEqual(preview.snapshot.fields["density"].shape, (EXPECTED_COUNTS[name]["zones"],))

    def test_package_level_backend_entry_points(self) -> None:
        path = ROOT / "5Fe+4.9TW+light.log"
        header = inspect(path)
        first_preview = preview(path)
        simulation = parse(path)
        self.assertEqual(header.n_zones, 500)
        self.assertIsInstance(first_preview, HeliosPreview)
        self.assertIsNotNone(first_preview.snapshot)
        self.assertEqual(simulation.fields["density"].shape, (8, 500))

    def test_memory_and_mmap_modes_match_on_smallest_file(self) -> None:
        path = ROOT / "5Fe+4.9TW+light.log"
        memory_parser = HeliosParser(access_mode="memory")
        mmap_parser = HeliosParser(access_mode="mmap")
        memory_header = memory_parser.inspect(path)
        mmap_header = mmap_parser.inspect(path)
        memory_preview = memory_parser.preview(path)
        mmap_preview = mmap_parser.preview(path)
        self.assertEqual(memory_header.n_zones, mmap_header.n_zones)
        self.assertEqual(memory_header.n_regions, mmap_header.n_regions)
        self.assertEqual(memory_header.n_materials, mmap_header.n_materials)
        self.assertEqual(memory_header.regions["region_index"].tolist(), mmap_header.regions["region_index"].tolist())
        assert memory_preview.snapshot is not None
        assert mmap_preview.snapshot is not None
        np.testing.assert_allclose(memory_preview.snapshot.fields["density"], mmap_preview.snapshot.fields["density"])
        np.testing.assert_allclose(
            memory_preview.snapshot.diagnostics["energy_summary"]["current"]["ions"],
            mmap_preview.snapshot.diagnostics["energy_summary"]["current"]["ions"],
        )

    def test_cu_parse_emits_no_probe_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulation = HeliosParser().parse(ROOT / "Cu_0166.log")
        self.assertIn("radiation_sink", simulation.fields)
        deprecations = [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
        self.assertEqual(deprecations, [])

    def test_counts_fields_and_core_diagnostics(self) -> None:
        for name, simulation in self.simulations.items():
            with self.subTest(example=name):
                expected = EXPECTED_COUNTS[name]
                self.assertEqual(simulation.metadata["n_zones"], expected["zones"])
                self.assertEqual(simulation.metadata["n_snapshots"], expected["snapshots"])
                self.assertEqual(set(simulation.fields), EXPECTED_FIELDS[name])
                self.assertTrue(CORE_FIELDS.issubset(simulation.fields))
                self.assertEqual(set(simulation.diagnostics), DIAGNOSTIC_SECTIONS)
                self.assertEqual(simulation.regions["region_index"].shape[0], expected["regions"])
                self.assertEqual(simulation.materials["index"].shape[0], expected["materials"])

    def test_structured_input_parameters_and_diagnostic_shapes(self) -> None:
        for name, simulation in self.simulations.items():
            with self.subTest(example=name):
                self.assertEqual(set(simulation.input_parameters), INPUT_SECTIONS)
                self.assertTrue(np.all(np.diff(simulation.time["time"]) >= 0.0))
                self.assertEqual(
                    simulation.diagnostics["energy_summary"]["initial"]["ions"].shape,
                    (EXPECTED_COUNTS[name]["snapshots"],),
                )
                self.assertEqual(
                    simulation.diagnostics["energy_exchange"]["sources_to_plasma"]["laser_deposition"]["total"].shape,
                    (EXPECTED_COUNTS[name]["snapshots"],),
                )
                self.assertEqual(
                    simulation.diagnostics["radiation_boundary_fluxes"]["region_net_cooling_rate"].shape,
                    (EXPECTED_COUNTS[name]["snapshots"], EXPECTED_COUNTS[name]["regions"]),
                )
                self.assertEqual(
                    simulation.diagnostics["radiation_boundary_fluxes"]["terminal_net_flux_at_boundary"].shape,
                    (EXPECTED_COUNTS[name]["snapshots"],),
                )
                if name == "10ns+10Si+60Al+15Si+4.27TW.log":
                    self.assertEqual(
                        simulation.input_parameters["radiative_transfer"]["frequency_gridding"]["section"].shape,
                        (2,),
                    )
                    self.assertEqual(simulation.regions["material_table_index"].tolist(), [1, 2, 1])

    def test_hdf5_round_trip_includes_new_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "5Fe_extended.h5"
            write_hdf5(ROOT / "5Fe+4.9TW+light.log", output, overwrite=True, parser=self.parser)
            with h5py.File(output, "r") as handle:
                self.assertIn("regions", handle)
                self.assertIn("materials", handle)
                self.assertIn("diagnostics", handle)
                self.assertIn("input_parameters", handle["metadata"])
                self.assertEqual(tuple(handle["regions"]["region_index"].shape), (1,))
                self.assertEqual(tuple(handle["materials"]["index"].shape), (1,))
                self.assertEqual(
                    tuple(handle["diagnostics"]["radiation_boundary_fluxes"]["region_net_cooling_rate"].shape),
                    (8, 1),
                )
                self.assertEqual(
                    tuple(handle["diagnostics"]["energy_summary"]["current"]["ions"].shape),
                    (8,),
                )
                self.assertEqual(handle["diagnostics"]["energy_summary"]["current"]["ions"].attrs["units"], "J/cm**2")
                self.assertEqual(handle["diagnostics"]["radiation_boundary_fluxes"]["region_net_cooling_rate"].attrs["units"], "J/s/cm2")
                self.assertEqual(handle["metadata"]["input_parameters"]["laser_source"]["wavelength"].attrs["units"], "microns")

    def test_diagnostic_normalization_pads_and_truncates_to_schema_width(self) -> None:
        np.testing.assert_allclose(
            normalize_diagnostic_value(np.asarray([1.0, 2.0], dtype=np.float64), 4),
            np.asarray([1.0, 2.0, np.nan, np.nan], dtype=np.float64),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            normalize_diagnostic_value(np.asarray([1.0, 2.0, 3.0], dtype=np.float64), 2),
            np.asarray([1.0, 2.0], dtype=np.float64),
            equal_nan=True,
        )
        self.assertTrue(np.isnan(float(normalize_diagnostic_value(None, None))))

    def test_diagnostic_width_reconciliation_is_explicit(self) -> None:
        self.assertEqual(reconcile_diagnostic_width(3, np.asarray([1.0, 2.0], dtype=np.float64)), (3, False, None))
        self.assertEqual(reconcile_diagnostic_width(2, np.asarray([1.0, 2.0, 3.0], dtype=np.float64)), (3, True, "vector_widened"))
        self.assertEqual(reconcile_diagnostic_width(None, np.asarray([1.0, 2.0], dtype=np.float64)), (2, True, "scalar_to_vector"))
        self.assertEqual(reconcile_diagnostic_width(None, None), (None, False, None))

    def test_all_example_files_have_consistent_header_counts(self) -> None:
        for name, counts in EXPECTED_COUNTS.items():
            with self.subTest(example=name):
                header = self.parser.inspect(ROOT / name)
                self.assertEqual(header.n_zones, counts["zones"])
                self.assertEqual(self.parser.count_snapshots(ROOT / name), counts["snapshots"])
                first_snapshot = next(self.parser.iter_snapshots(ROOT / name, header=header))
                self.assertTrue(CORE_FIELDS.issubset(first_snapshot.fields))
                self.assertTrue(DIAGNOSTIC_SECTIONS.issubset(first_snapshot.diagnostics))


if __name__ == "__main__":
    unittest.main()
