from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
import re

import h5py
import numpy as np

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosParser, HeliosRun, write_hdf5
from helios_parser.coordinates import build_coordinate_edge_array, build_coordinate_edge_grid, build_coordinate_model
from helios_parser.parser import _parse_field_table_rows


ROOT = Path(__file__).resolve().parents[1]
HDF5_ROOT = ROOT / "outputs" / "hdf5"


def _assert_coordinate_invariants(
    testcase: unittest.TestCase,
    edge_values: np.ndarray,
    center_values: np.ndarray,
    widths: np.ndarray,
    *,
    require_width_match: bool = True,
) -> None:
    edges = np.asarray(edge_values, dtype=np.float64)
    centers = np.asarray(center_values, dtype=np.float64)
    zone_widths = np.asarray(widths, dtype=np.float64)
    testcase.assertEqual(edges.shape, (zone_widths.size + 1,))
    testcase.assertEqual(centers.shape, zone_widths.shape)
    testcase.assertTrue(np.all(np.diff(edges) > 0.0))
    np.testing.assert_allclose(
        centers,
        0.5 * (edges[:-1] + edges[1:]),
        rtol=1.0e-12,
        atol=1.0e-18,
        equal_nan=True,
    )
    if require_width_match:
        width_error = np.abs((edges[1:] - edges[:-1]) - zone_widths)
        testcase.assertLessEqual(float(np.nanmax(width_error)), max(float(np.nanmax(np.abs(zone_widths))) * 5.0e-3, 1.0e-9))


def _first_row_zero_boundary_from_log(path: Path) -> float | None:
    pattern = re.compile(r"^\s*0\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)")
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                token = match.group(1).replace("D", "E").replace("d", "e")
                return float(token)
    return None


def _primary_table_lines(second_velocity_token: str) -> list[str]:
    return [
        " 0 0.000E+00",
        " 1 2.500E-06 2.500E-06 8.920E+00 1.000E+05 1.000E+00 2.000E+00 3.000E+00 4.000E+00 5.000E+00 5.500E+00 1.000E+00 6.000E+00 7.000E+00 8.000E+00",
        f" 2 5.000E-06 2.500E-06 8.921E+00 {second_velocity_token} 1.100E+00 2.100E+00 3.100E+00 4.100E+00 5.100E+00 5.600E+00 1.100E+00 6.100E+00 7.100E+00 8.100E+00",
    ]


class CoordinateSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = HeliosParser()
        cls.logger = logging.getLogger("coordinate-semantics-tests")

    def test_primary_table_parser_handles_clean_and_malformed_exponents(self) -> None:
        header_line = "Radius Zone width Mass dens Velocity Rad temp Ion temp Elec temp Ion press Elec press Rad press Compression Elec Dens Mean Chg Art. Visc."
        units_line = "cm cm g/cm3 cm/s eV eV eV J/cm3 J/cm3 J/cm3 rho/rho0 1/cm3 - J/cm3"
        cases = {
            "well_formed": "4.973E-174",
            "malformed_missing_e": "4.973-174",
            "mixed_block": "1.000D+05",
        }
        for label, velocity_token in cases.items():
            with self.subTest(case=label):
                arrays, _, raw_field_map, extras = _parse_field_table_rows(
                    _primary_table_lines(velocity_token),
                    header_line,
                    units_line,
                    2,
                    self.logger,
                )
                self.assertIn("radius", arrays)
                self.assertEqual(raw_field_map["radius"], "Radius (zone-center alias derived from HELIOS edge coordinates)")
                self.assertAlmostEqual(float(extras["boundary_edge"]), 0.0)
                _assert_coordinate_invariants(
                    self,
                    np.asarray(extras["coordinate_edge"], dtype=np.float64),
                    np.asarray(extras["coordinate_center"], dtype=np.float64),
                    np.asarray(arrays["zone_width"], dtype=np.float64),
                )
                expected_velocity = 4.973e-174 if label != "mixed_block" else 1.0e5
                self.assertTrue(np.isfinite(float(arrays["velocity"][1])))
                self.assertAlmostEqual(float(arrays["velocity"][1]), expected_velocity)

    def test_coordinate_helper_preserves_nonzero_boundary_edge_exactly(self) -> None:
        edges, centers = build_coordinate_model(
            np.asarray([3.0, 5.0], dtype=np.float64),
            np.asarray([2.0, 2.0], dtype=np.float64),
            boundary_edge=1.25,
        )
        np.testing.assert_allclose(edges, np.asarray([1.25, 3.0, 5.0], dtype=np.float64))
        np.testing.assert_allclose(centers, np.asarray([2.125, 4.0], dtype=np.float64))

    def test_cumulative_width_fallback_is_only_used_for_non_monotonic_rounded_edges(self) -> None:
        fallback_issues = []
        fallback_edges = build_coordinate_edge_array(
            np.asarray([1.0e-4, 2.0e-4, 2.0e-4], dtype=np.float64),
            np.asarray([1.0e-4, 1.0e-4, 1.0e-8], dtype=np.float64),
            boundary_edge=0.0,
            issues=fallback_issues,
        )
        np.testing.assert_allclose(fallback_edges, np.asarray([0.0, 1.0e-4, 2.0e-4, 2.0001e-4], dtype=np.float64))
        self.assertIn("rounded_edge_fallback", {issue.code for issue in fallback_issues})

        mismatch_issues = []
        preserved_edges = build_coordinate_edge_array(
            np.asarray([1.0, 2.1, 3.1], dtype=np.float64),
            np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
            boundary_edge=0.0,
            issues=mismatch_issues,
        )
        np.testing.assert_allclose(preserved_edges, np.asarray([0.0, 1.0, 2.1, 3.1], dtype=np.float64))
        self.assertIn("edge_width_mismatch", {issue.code for issue in mismatch_issues})
        self.assertNotIn("rounded_edge_fallback", {issue.code for issue in mismatch_issues})

    def test_cylindrical_tiny_negative_dynamic_boundary_is_clipped_explicitly(self) -> None:
        issues = []
        edges = build_coordinate_edge_grid(
            np.asarray([[1.0e-6, 2.0e-6]], dtype=np.float64),
            np.asarray([[1.0e-6, 1.0e-6]], dtype=np.float64),
            boundary_edges=np.asarray([-1.0e-19], dtype=np.float64),
            geometry="CYLINDRICAL",
            issues=issues,
        )
        self.assertEqual(float(edges[0, 0]), 0.0)
        self.assertIn("cylindrical_boundary_noise_clipped", {issue.code for issue in issues})

    def test_coordinate_model_is_explicit_for_planar_and_cylindrical_logs(self) -> None:
        cases = (
            ("5Fe+4.9TW+light.log", "PLANAR"),
            ("Cu1e17.log", "PLANAR"),
            ("Cu1e17_cyl.log", "CYLINDRICAL"),
        )
        for name, geometry in cases:
            with self.subTest(example=name):
                path = ROOT / name
                header = self.parser.inspect(path)
                preview = self.parser.preview(path)
                simulation = self.parser.parse(path)

                self.assertEqual(str(header.metadata.get("geometry")).upper(), geometry)
                self.assertEqual(str(header.metadata["coordinate_model"]["coordinate_name"]), "x" if geometry == "PLANAR" else "radius")

                header_edges = np.asarray(header.grid["coordinate_edge"], dtype=np.float64)
                header_centers = np.asarray(header.grid["coordinate_center"], dtype=np.float64)
                header_widths = np.asarray(header.grid["zone_width"], dtype=np.float64)
                _assert_coordinate_invariants(self, header_edges, header_centers, header_widths)
                np.testing.assert_allclose(np.asarray(header.grid["x"], dtype=np.float64), header_centers)
                parsed_boundary = _first_row_zero_boundary_from_log(path)
                if parsed_boundary is not None:
                    self.assertAlmostEqual(float(header_edges[0]), float(parsed_boundary))
                if geometry == "CYLINDRICAL":
                    self.assertGreaterEqual(float(header_edges[0]), 0.0)

                self.assertIsNotNone(preview.snapshot)
                assert preview.snapshot is not None
                self.assertEqual(preview.snapshot.coordinate_name, "x" if geometry == "PLANAR" else "radius")
                self.assertIsNotNone(preview.snapshot.coordinate_edge)
                self.assertIsNotNone(preview.snapshot.coordinate_center)
                _assert_coordinate_invariants(
                    self,
                    np.asarray(preview.snapshot.coordinate_edge, dtype=np.float64),
                    np.asarray(preview.snapshot.coordinate_center, dtype=np.float64),
                    np.asarray(preview.snapshot.fields["zone_width"], dtype=np.float64),
                )
                if parsed_boundary is not None:
                    self.assertAlmostEqual(float(preview.snapshot.coordinate_edge[0]), float(parsed_boundary))
                if geometry == "CYLINDRICAL":
                    self.assertGreaterEqual(float(preview.snapshot.coordinate_edge[0]), 0.0)
                np.testing.assert_allclose(
                    np.asarray(preview.snapshot.fields["radius"], dtype=np.float64),
                    np.asarray(preview.snapshot.coordinate_center, dtype=np.float64),
                    rtol=1.0e-12,
                    atol=1.0e-18,
                )

                sim_edges = np.asarray(simulation.grid["coordinate_edge"], dtype=np.float64)
                sim_centers = np.asarray(simulation.grid["coordinate_center"], dtype=np.float64)
                sim_widths = np.asarray(simulation.grid["zone_width"], dtype=np.float64)
                _assert_coordinate_invariants(self, sim_edges, sim_centers, sim_widths)
                np.testing.assert_allclose(sim_edges, header_edges, rtol=1.0e-12, atol=1.0e-18)
                np.testing.assert_allclose(sim_centers, header_centers, rtol=1.0e-12, atol=1.0e-18)

    def test_hdf5_roundtrip_preserves_explicit_coordinate_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("Cu1e17.log", "Cu1e17_cyl.log"):
                with self.subTest(example=name):
                    source = ROOT / name
                    target = Path(tmpdir) / f"{source.stem}.h5"
                    write_hdf5(source, target, overwrite=True, parser=self.parser)

                    with h5py.File(target, "r") as handle:
                        self.assertIn("coordinate_center", handle["grid"])
                        self.assertIn("coordinate_edge", handle["grid"])
                        self.assertIn("dynamic_coordinate_center", handle["grid"])
                        self.assertIn("dynamic_coordinate_edge", handle["grid"])
                        self.assertEqual(handle["grid"]["coordinate_center"].attrs["coordinate_location"], "center")
                        self.assertEqual(handle["grid"]["coordinate_edge"].attrs["coordinate_location"], "edge")
                        self.assertEqual(handle["grid"]["dynamic_coordinate_center"].attrs["coordinate_location"], "center")
                        self.assertEqual(handle["grid"]["dynamic_coordinate_edge"].attrs["coordinate_location"], "edge")
                        self.assertTrue(bool(handle["grid"]["x"].attrs["legacy_coordinate_alias"]))
                        self.assertTrue(bool(handle["fields"]["radius"].attrs["legacy_coordinate_alias"]))
                        self.assertIn("coordinate_model", handle["metadata"])

                    with HeliosRun(target) as run:
                        static_edges = run.get_static_coordinate(location="edge")
                        static_centers = run.get_static_coordinate(location="center")
                        np.testing.assert_allclose(run.get_static_coordinate(), static_centers, rtol=1.0e-12, atol=1.0e-18)
                        np.testing.assert_allclose(run.get_coordinate(prefer_dynamic=False), static_centers, rtol=1.0e-12, atol=1.0e-18)
                        static_widths = np.asarray(run.get_grid("zone_width"), dtype=np.float64)
                        _assert_coordinate_invariants(self, static_edges, static_centers, static_widths)
                        dynamic_edges = run.get_dynamic_coordinate(snapshot_index=0, location="edge")
                        dynamic_centers = run.get_dynamic_coordinate(snapshot_index=0, location="center")
                        np.testing.assert_allclose(run.get_dynamic_coordinate(snapshot_index=0), dynamic_centers, rtol=1.0e-12, atol=1.0e-18)
                        np.testing.assert_allclose(run.get_coordinate(snapshot_index=0), dynamic_centers, rtol=1.0e-12, atol=1.0e-18)
                        dynamic_widths = np.asarray(run.get_snapshot_field("zone_width", 0), dtype=np.float64)
                        assert dynamic_edges is not None
                        assert dynamic_centers is not None
                        _assert_coordinate_invariants(self, dynamic_edges, dynamic_centers, dynamic_widths)
                        np.testing.assert_allclose(run.get_grid("x"), static_centers, rtol=1.0e-12, atol=1.0e-18)
                        np.testing.assert_allclose(run.get_grid("coordinate_edge"), static_edges, rtol=1.0e-12, atol=1.0e-18)
                        np.testing.assert_allclose(run.get_field("radius", time_slice=0), dynamic_centers, rtol=1.0e-12, atol=1.0e-18)

    def test_existing_hdf5_files_load_through_coordinate_compatibility_path(self) -> None:
        path = HDF5_ROOT / "Cu_0166_stabilized.h5"
        with HeliosRun(path) as run:
            static_edges = run.get_static_coordinate(location="edge")
            static_centers = run.get_static_coordinate(location="center")
            static_widths = np.asarray(run.get_grid("zone_width"), dtype=np.float64)
            _assert_coordinate_invariants(self, static_edges, static_centers, static_widths, require_width_match=False)
            if run.has_dynamic_coordinate():
                dynamic_edges = run.get_dynamic_coordinate(snapshot_index=0, location="edge")
                dynamic_centers = run.get_dynamic_coordinate(snapshot_index=0, location="center")
                dynamic_widths = np.asarray(run.get_snapshot_field("zone_width", 0), dtype=np.float64)
                assert dynamic_edges is not None
                assert dynamic_centers is not None
                _assert_coordinate_invariants(self, dynamic_edges, dynamic_centers, dynamic_widths, require_width_match=False)


if __name__ == "__main__":
    unittest.main()
