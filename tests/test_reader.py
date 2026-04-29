from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosRun


ROOT = Path(__file__).resolve().parents[1]
HDF5_ROOT = ROOT / "outputs" / "hdf5"
EXPECTED = {
    "5Fe+4.9TW+light_stabilized.h5": {"zones": 500, "snapshots": 8, "regions": 1, "materials": 1},
    "Cu_0166_stabilized.h5": {"zones": 300, "snapshots": 461, "regions": 1, "materials": 1},
    "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5": {"zones": 1300, "snapshots": 701, "regions": 3, "materials": 2},
}


class _FakeDataset:
    def __init__(self, values: np.ndarray, *, attrs: dict[str, object] | None = None) -> None:
        self.values = np.asarray(values)
        self.attrs = {} if attrs is None else dict(attrs)
        self.shape = self.values.shape
        self.calls: list[Any] = []

    def __getitem__(self, index):
        self.calls.append(index)
        if index == ():
            raise AssertionError("test dataset was read as a full array")
        return self.values[index]


class ReaderTests(unittest.TestCase):
    def test_summary_field_access_and_diagnostics(self) -> None:
        for name, expected in EXPECTED.items():
            with self.subTest(example=name):
                with HeliosRun(HDF5_ROOT / name) as run:
                    summary = run.summary()
                    self.assertEqual(summary["n_zones"], expected["zones"])
                    self.assertEqual(summary["n_snapshots"], expected["snapshots"])
                    self.assertEqual(summary["n_regions"], expected["regions"])
                    self.assertEqual(summary["n_materials"], expected["materials"])
                    self.assertIn("density", run.list_fields())
                    self.assertIn("energy_summary/current/ions", run.list_diagnostics())

                    density = run.get_field("density")
                    self.assertEqual(density.shape, (expected["snapshots"], expected["zones"]))
                    density_slice = run.get_field("density", time_slice=slice(0, 2), zone_slice=slice(0, 3))
                    self.assertEqual(density_slice.shape, (2, 3))

                    temperature = run.get_snapshot_field("temperature_e", -1)
                    self.assertEqual(temperature.shape, (expected["zones"],))
                    np.testing.assert_allclose(temperature, run.get_lineout("temperature_e", -1))

                    self.assertEqual(run.get_time().shape, (expected["snapshots"],))
                    self.assertEqual(run.get_radius().shape, (expected["zones"],))
                    self.assertEqual(run.get_radius(snapshot_index=0).shape, (expected["zones"],))

                    energy_summary = run.get_diagnostic("energy_summary/current/ions")
                    self.assertEqual(energy_summary.shape, (expected["snapshots"],))
                    boundary_flux = run.get_diagnostic("radiation_boundary_fluxes/region_net_cooling_rate")
                    self.assertEqual(boundary_flux.shape, (expected["snapshots"], expected["regions"]))

    def test_snapshot_dynamic_coordinate_read_does_not_materialize_full_grid(self) -> None:
        run = HeliosRun.__new__(HeliosRun)
        center = _FakeDataset(np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float64))
        edge = _FakeDataset(np.asarray([[0.5, 1.5, 2.5, 3.5], [1.0, 2.0, 3.0, 4.0]], dtype=np.float64))
        run.path = Path("synthetic.h5")
        run._grid_datasets = {
            "zone_id": _FakeDataset(np.asarray([1, 2, 3], dtype=np.int32)),
            "zone_width": _FakeDataset(np.asarray([1.0, 1.0, 1.0], dtype=np.float64)),
            "dynamic_coordinate_center": center,
            "dynamic_coordinate_edge": edge,
        }
        run._time_datasets = {"time": _FakeDataset(np.asarray([0.0, 1.0], dtype=np.float64))}
        run._field_datasets = {}
        run._metadata_cache = {"geometry": "PLANAR"}
        run._coordinate_model_cache = {
            "coordinate_name": "x",
            "dynamic_center_dataset": "dynamic_coordinate_center",
            "dynamic_edge_dataset": "dynamic_coordinate_edge",
            "width_dataset": "zone_width",
        }

        np.testing.assert_allclose(run.get_dynamic_coordinate(snapshot_index=1, location="center"), np.asarray([1.5, 2.5, 3.5]))
        np.testing.assert_allclose(run.get_dynamic_coordinate(snapshot_index=1, location="edge"), np.asarray([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(center.calls, [(1, slice(None, None, None))])
        self.assertEqual(edge.calls, [(1, slice(None, None, None))])
        self.assertFalse(hasattr(run, "_dynamic_coordinate_center_cache") and run._dynamic_coordinate_center_cache is not None)

    def test_radius_snapshot_field_read_uses_row_slice_without_full_dynamic_cache(self) -> None:
        run = HeliosRun.__new__(HeliosRun)
        radius = _FakeDataset(
            np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float64),
            attrs={"coordinate_location": "center"},
        )
        run.path = Path("synthetic_radius.h5")
        run._grid_datasets = {
            "zone_id": _FakeDataset(np.asarray([1, 2, 3], dtype=np.int32)),
            "zone_width": _FakeDataset(np.asarray([1.0, 1.0, 1.0], dtype=np.float64)),
        }
        run._time_datasets = {"time": _FakeDataset(np.asarray([0.0, 1.0], dtype=np.float64))}
        run._field_datasets = {"radius": radius}
        run._metadata_cache = {"geometry": "PLANAR"}
        run._coordinate_model_cache = {
            "coordinate_name": "x",
            "legacy_dynamic_center_alias": "radius",
            "width_dataset": "zone_width",
        }

        np.testing.assert_allclose(run.get_snapshot_field("radius", 1), np.asarray([1.5, 2.5, 3.5]))
        self.assertEqual(radius.calls, [(1, slice(None, None, None))])
        self.assertFalse(hasattr(run, "_dynamic_coordinate_center_cache") and run._dynamic_coordinate_center_cache is not None)

    def test_time_trace_read_uses_column_slice(self) -> None:
        run = HeliosRun.__new__(HeliosRun)
        density = _FakeDataset(np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float64))
        run.path = Path("synthetic_trace.h5")
        run._grid_datasets = {"zone_id": _FakeDataset(np.asarray([1, 2, 3], dtype=np.int32))}
        run._time_datasets = {"time": _FakeDataset(np.asarray([0.0, 1.0], dtype=np.float64))}
        run._field_datasets = {"density": density}
        run._field_names = ("density",)

        np.testing.assert_allclose(run.get_time_trace("density", 2), np.asarray([3.0, 3.5]))
        self.assertEqual(density.calls, [(slice(None, None, None), 2)])

    def test_radius_time_trace_read_uses_dynamic_coordinate_column_slice(self) -> None:
        run = HeliosRun.__new__(HeliosRun)
        center = _FakeDataset(np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float64))
        run.path = Path("synthetic_radius_trace.h5")
        run._grid_datasets = {
            "zone_id": _FakeDataset(np.asarray([1, 2, 3], dtype=np.int32)),
            "dynamic_coordinate_center": center,
        }
        run._time_datasets = {"time": _FakeDataset(np.asarray([0.0, 1.0], dtype=np.float64))}
        run._field_datasets = {}
        run._metadata_cache = {"geometry": "PLANAR"}
        run._coordinate_model_cache = {
            "coordinate_name": "x",
            "dynamic_center_dataset": "dynamic_coordinate_center",
        }

        np.testing.assert_allclose(run.get_time_trace("radius", 1), np.asarray([2.0, 2.5]))
        self.assertEqual(center.calls, [(slice(None, None, None), 1)])
        self.assertFalse(hasattr(run, "_dynamic_coordinate_center_cache") and run._dynamic_coordinate_center_cache is not None)

    def test_region_and_material_masks_match_hdf5_metadata(self) -> None:
        for name, expected in EXPECTED.items():
            with self.subTest(example=name):
                with HeliosRun(HDF5_ROOT / name) as run:
                    regions = run.get_regions()
                    self.assertEqual(len(regions["region_index"]), expected["regions"])
                    for region_id, start, stop in zip(
                        regions["region_index"],
                        regions["min_zone_index"],
                        regions["max_zone_index"],
                    ):
                        mask = run.get_region_mask(int(region_id))
                        self.assertEqual(mask.shape, (expected["zones"],))
                        self.assertEqual(int(mask.sum()), int(stop - start + 1))

                    materials = run.get_materials()
                    self.assertEqual(len(materials["index"]), expected["materials"])
                    zone_material_index = np.abs(run.get_grid("zone_material_index"))
                    for material_id in materials["index"]:
                        mask = run.get_material_mask(int(material_id))
                        self.assertEqual(mask.shape, (expected["zones"],))
                        self.assertEqual(int(mask.sum()), int(np.count_nonzero(zone_material_index == material_id)))

    def test_visar_readiness_surface_reports_status_and_boundaries(self) -> None:
        for name in ("5Fe+4.9TW+light_stabilized.h5", "Cu_0166_stabilized.h5", "10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"):
            with self.subTest(example=name):
                with HeliosRun(HDF5_ROOT / name) as run:
                    readiness = run.check_visar_readiness()
                    self.assertTrue(readiness.ready)
                    self.assertEqual(readiness.support.velocity_field_name, "velocity")
                    self.assertEqual(readiness.support.time_axis_name, "time")
                    self.assertGreaterEqual(len(readiness.support.candidate_boundaries), 2)
                    self.assertEqual(readiness.support.event_timing_source, "derived.shock_tracking.track_shock_front")

    def test_visar_readiness_flags_missing_velocity_and_inconsistent_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "minimal_invalid.h5"
            with h5py.File(path, "w") as handle:
                grid = handle.create_group("grid")
                grid.create_dataset("zone_id", data=np.asarray([1, 2, 3], dtype=np.int32))
                grid.create_dataset("x", data=np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
                grid.create_dataset("zone_region_id", data=np.asarray([1, 1, 2], dtype=np.int32))
                grid.create_dataset("zone_material_index", data=np.asarray([1, 1, 1], dtype=np.int32))
                time = handle.create_group("time")
                time.create_dataset("time", data=np.asarray([0.0, 1.0e-9], dtype=np.float64))
                fields = handle.create_group("fields")
                fields.create_dataset("density", data=np.ones((2, 3), dtype=np.float64))
                regions = handle.create_group("regions")
                regions.create_dataset("region_index", data=np.asarray([1, 2], dtype=np.int32))
                regions.create_dataset("min_zone_index", data=np.asarray([2, 2], dtype=np.int32))
                regions.create_dataset("max_zone_index", data=np.asarray([2, 3], dtype=np.int32))
                materials = handle.create_group("materials")
                materials.create_dataset("index", data=np.asarray([1], dtype=np.int32))
                diagnostics = handle.create_group("diagnostics")
                metadata = handle.create_group("metadata")
                metadata.create_dataset("available_fields", data=np.asarray(["density"], dtype=h5py.string_dtype("utf-8")))
            with HeliosRun(path) as run:
                readiness = run.check_visar_readiness()
            self.assertFalse(readiness.ready)
            self.assertIn("velocity field is missing.", readiness.reasons)
            self.assertTrue(any("region/interface indexing" in reason for reason in readiness.reasons))


if __name__ == "__main__":
    unittest.main()
