from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import _test_bootstrap  # noqa: F401
from helios.cache import AnalyzerCacheSet
from helios.runtime import RunContext
from helios_viewer.models import OpenRunPayload


class PlatformPhase34Tests(unittest.TestCase):
    def test_run_context_tracks_loaded_run_state(self) -> None:
        payload = OpenRunPayload(
            run_generation=0,
            path=Path("example.h5"),
            summary={"n_zones": 4, "n_snapshots": 3},
            metadata={"simulation_name": "example"},
            fields=["density", "velocity"],
            field_units={"density": "g/cm3", "velocity": "cm/s"},
            diagnostics=["energy_summary/current/ions"],
            diagnostic_units={"energy_summary/current/ions": "J/cm**2"},
            regions={},
            materials={},
            time=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            time_unit="s",
            static_x=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            static_x_edges=np.asarray([0.05, 0.15, 0.25, 0.35, 0.45], dtype=np.float64),
            static_x_unit="cm",
            zone_region_id=np.asarray([1, 1, 2, 2], dtype=np.int32),
            zone_material_index=np.asarray([1, 1, 2, 2], dtype=np.int32),
            has_dynamic_radius=True,
            radius_unit="cm",
        )

        context = RunContext.from_payload(payload)
        self.assertTrue(context.has_run)
        self.assertEqual(context.map_coordinate, "moving_radius")
        self.assertEqual(context.slice_coordinate, "zone")
        context.set_snapshot_index(10)
        self.assertEqual(context.snapshot_index, 2)
        context.set_coordinate_modes(map_coordinate="zone", slice_coordinate="static_x")
        self.assertEqual(context.map_coordinate, "zone")
        self.assertEqual(context.slice_coordinate, "static_x")
        context.set_subset(region_ids=(1, 2), material_ids=(2,))
        self.assertEqual(context.selected_region_ids, (1, 2))
        self.assertEqual(context.selected_material_ids, (2,))

    def test_cache_layers_have_explicit_bucket_invalidation(self) -> None:
        caches = AnalyzerCacheSet()
        view_bucket = caches.view_cache.bucket("display")
        view_bucket["density"] = "cached"
        self.assertEqual(caches.view_cache.bucket("display")["density"], "cached")
        caches.view_cache.clear_bucket("display")
        self.assertEqual(len(caches.view_cache.bucket("display")), 0)
        caches.raw_data_cache.bucket("raw")["field"] = 1
        caches.derived_cache.bucket("derived")["shock"] = 2
        caches.raw_data_cache.clear()
        self.assertEqual(len(caches.raw_data_cache.bucket("raw")), 0)
        self.assertEqual(caches.derived_cache.bucket("derived")["shock"], 2)


if __name__ == "__main__":
    unittest.main()
