from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
import h5py
import numpy as np

from helios_parser import HeliosParser, HeliosRun, write_hdf5
from helios_parser.exodus import BoundaryFlux, RadiationOutput, SpectralRadiation
from helios_parser.hdf5 import (
    _create_bpf_field_dataset,
    _field_chunk_shape,
    _spectral_chunk_shape,
    _time_major_chunk_shape,
    _write_radiation_groups,
)

ROOT = Path(__file__).resolve().parents[1]

# Chunk geometry is invisible through the HDF5 API, so nothing else in the suite
# would notice if these extents silently reverted. The 2.2 layout exists for one
# reason: field chunks must match the writer's 32-snapshot flush so a flush lands
# in one chunk, and radiation chunks must be one dump so a single-dump read does
# not inflate sixteen.


def _synthetic_radiation(n_dumps: int, n_nodes: int, n_groups: int) -> RadiationOutput:
    """Build the smallest radiation payload the writer accepts."""

    # The group grid is per dump on both axes, not one static grid.
    group_edges = np.tile(np.linspace(0.1, 1000.0, n_groups + 1), (n_dumps, 1))
    group_centre = 0.5 * (group_edges[:, 1:] + group_edges[:, :-1])
    shape = (n_dumps, n_nodes, n_groups)
    flux_shape = (n_dumps, 3, n_groups)
    return RadiationOutput(
        spectral=SpectralRadiation(
            time_s=np.arange(n_dumps, dtype=np.float64),
            cycle=np.arange(n_dumps, dtype=np.int64),
            hydro_index=np.arange(n_dumps, dtype=np.int64),
            group_edges_ev=group_edges,
            group_centre_ev=group_centre,
            node_axis_cm=np.linspace(0.0, 1.0, n_nodes),
            fields={"spectral_energy_density": np.ones(shape, dtype=np.float64)},
            field_units={"spectral_energy_density": "J/cm3/eV"},
        ),
        flux=BoundaryFlux(
            time_s=np.arange(n_dumps, dtype=np.float64),
            cycle=np.arange(n_dumps, dtype=np.int64),
            group_edges_ev=group_edges,
            region_boundary_node_index=np.asarray([1, n_nodes], dtype=np.int64),
            fields={"net_interface": np.ones(flux_shape, dtype=np.float64)},
            field_units={"net_interface": "W/eV"},
        ),
        integrated={"total_flux_rmin": np.ones(n_dumps, dtype=np.float64)},
        integrated_units={"total_flux_rmin": "W"},
    )


class ChunkHelperTest(unittest.TestCase):
    """The helpers are the single source of truth, so assert them directly."""

    def test_field_chunks_span_the_writer_flush_batch(self) -> None:
        self.assertEqual(_time_major_chunk_shape((461, 300), time_chunk=32), (32, 300))
        # Capped by the snapshot count, never larger than the dataset.
        self.assertEqual(_time_major_chunk_shape((7, 300), time_chunk=32), (7, 300))
        self.assertEqual(_time_major_chunk_shape((461, 300)), (16, 300))

    def test_spectral_chunks_hold_exactly_one_dump(self) -> None:
        self.assertEqual(_spectral_chunk_shape((40, 1202, 250)), (1, 1202, 250))
        self.assertEqual(_spectral_chunk_shape((5, 251)), (1, 251))
        # A 1-D array has no dump axis, so one-element chunks would be
        # pathological; it falls back to the time-major extent.
        self.assertEqual(_spectral_chunk_shape((9,)), (9,))
        self.assertEqual(_spectral_chunk_shape((900,)), (16,))

    def test_field_chunk_shape_spans_32_snapshots(self) -> None:
        """The log writer and the dynamic-coordinate writer both route here."""

        self.assertEqual(_field_chunk_shape(461, 300), (32, 300))
        self.assertEqual(_field_chunk_shape(1501, 1202), (32, 1202))
        # Capped by the run length, never wider than the dataset.
        self.assertEqual(_field_chunk_shape(7, 300), (7, 300))
        self.assertEqual(_field_chunk_shape(0, 300), (1, 300))

    def test_the_bpf_writer_uses_32_not_16(self) -> None:
        """The tracked log fixture has 8 snapshots, so prove 32 on the writer itself."""

        with tempfile.TemporaryDirectory() as tmpdir:
            with h5py.File(Path(tmpdir) / "bpf_chunks.h5", "w") as handle:
                fields = handle.create_group("fields")
                meta = handle.create_group("field_metadata")
                sample = np.zeros((50,), dtype=np.float64)
                dataset = _create_bpf_field_dataset(fields, meta, "density", sample, 461, None, {"unit": "g/cm3"})
                self.assertEqual(tuple(dataset.chunks), (32, 50))
                # Capped by the run length, never wider than the dataset.
                short = _create_bpf_field_dataset(fields, meta, "velocity", sample, 7, None, {"unit": "cm/s"})
                self.assertEqual(tuple(short.chunks), (7, 50))

    def test_a_production_spectral_chunk_is_one_dump(self) -> None:
        # 1202 nodes x 250 groups float64 is 2.4 MB, against 38.5 MB for the
        # sixteen-dump chunk it replaced. The reader keeps HDF5's 1 MiB default
        # cache, so this is not a residency claim: it is a claim that a
        # single-dump read inflates one dump instead of sixteen.
        extent = _spectral_chunk_shape((1501, 1202, 250))
        one_dump = 1202 * 250 * 8
        old_chunk = 16 * one_dump
        self.assertEqual(int(np.prod(extent)) * 8, one_dump)
        self.assertEqual(tuple(extent), (1, 1202, 250))
        # The extent this replaced was 38.5 MB, larger than any HDF5 block
        # cache, so one dump read inflated sixteen.
        self.assertGreater(old_chunk, 32 * 1024 * 1024)


class ConvertedFieldChunkTest(unittest.TestCase):
    """Assert the geometry a real conversion actually writes."""

    @classmethod
    def setUpClass(cls) -> None:
        # The tracked BPF fixture, not a local-only log: this test runs in the
        # core gate, which materialises a tree containing only the fixtures
        # declared for the entry.
        cls.parser = HeliosParser()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.output = Path(cls._tmpdir.name) / "chunk_layout.h5"
        write_hdf5(ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf", cls.output, overwrite=True)
        cls.handle = h5py.File(cls.output, "r")
        cls.helios_run = HeliosRun(cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.helios_run.close()
        cls.handle.close()
        cls._tmpdir.cleanup()

    def test_schema_version_is_two_point_two(self) -> None:
        self.assertEqual(self.handle.attrs["schema_version"], "2.2")

    def test_field_datasets_span_32_snapshots(self) -> None:
        n_snapshots = int(self.helios_run.n_snapshots)
        expected = (min(32, n_snapshots), int(self.helios_run.n_zones))
        checked = 0
        for name in ("density", "velocity", "temperature_e", "pressure_i"):
            dataset = self.handle[f"fields/{name}"]
            self.assertEqual(tuple(dataset.chunks), expected, f"fields/{name}")
            checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_dynamic_coordinate_datasets_match_the_field_layout(self) -> None:
        n_snapshots = int(self.helios_run.n_snapshots)
        for name in ("dynamic_coordinate_center", "dynamic_coordinate_edge"):
            if name not in self.handle["grid"]:
                continue
            dataset = self.handle["grid"][name]
            self.assertEqual(int(dataset.chunks[0]), min(32, n_snapshots), name)
            self.assertEqual(int(dataset.chunks[1]), int(dataset.shape[1]), name)

    def test_time_datasets_are_unchanged(self) -> None:
        dataset = self.handle["time/time"]
        self.assertEqual(tuple(dataset.chunks), (min(1024, int(self.helios_run.n_snapshots)),))

    def test_axis_order_stays_time_major(self) -> None:
        dataset = self.handle["fields/density"]
        self.assertEqual(int(dataset.shape[0]), int(self.helios_run.n_snapshots))
        self.assertEqual(int(dataset.shape[1]), int(self.helios_run.n_zones))


class RadiationChunkTest(unittest.TestCase):
    """Radiation and flux must chunk differently; only radiation is one-per-dump."""

    def test_radiation_is_one_dump_per_chunk_and_flux_is_not(self) -> None:
        n_dumps, n_nodes, n_groups = 40, 12, 6
        radiation = _synthetic_radiation(n_dumps, n_nodes, n_groups)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "radiation.h5"
            with h5py.File(path, "w") as handle:
                _write_radiation_groups(handle, radiation, compression=None)
                spectral = handle["radiation/spectral_energy_density"]
                self.assertEqual(tuple(spectral.chunks), (1, n_nodes, n_groups))
                self.assertEqual(tuple(handle["radiation/group_edges"].chunks), (1, n_groups + 1))
                self.assertEqual(tuple(handle["radiation/group_centre"].chunks), (1, n_groups))
                # flux sits on the whole-run axis and keeps the 16-dump chunk.
                flux = handle["flux/net_interface"]
                self.assertEqual(tuple(flux.chunks), (min(16, n_dumps), 3, n_groups))
                self.assertEqual(tuple(handle["flux/group_edges"].chunks), (min(16, n_dumps), n_groups + 1))

    def test_values_survive_the_new_geometry(self) -> None:
        n_dumps, n_nodes, n_groups = 40, 12, 6
        radiation = _synthetic_radiation(n_dumps, n_nodes, n_groups)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "radiation_roundtrip.h5"
            with h5py.File(path, "w") as handle:
                _write_radiation_groups(handle, radiation, compression="gzip")
                stored = handle["radiation/spectral_energy_density"][()]
            np.testing.assert_array_equal(stored, radiation.spectral.fields["spectral_energy_density"])


if __name__ == "__main__":
    unittest.main()