from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import _test_bootstrap  # noqa: F401

from helios_parser import HeliosParser, HeliosRun, write_hdf5


ROOT = Path(__file__).resolve().parents[1]
HDF5_ROOT = ROOT / "outputs" / "hdf5"


class Wave4IngestTests(unittest.TestCase):
    def test_default_parser_uses_mmap_and_matches_memory_preview(self) -> None:
        source = ROOT / "5Fe+4.9TW+light.log"
        default_parser = HeliosParser()
        memory_parser = HeliosParser(access_mode="memory")

        with default_parser.open_document(source) as document:
            self.assertEqual(document.buffer.access_mode, "mmap")
            preview_default = document.preview()
        preview_memory = memory_parser.preview(source)

        self.assertEqual(preview_default.header.n_zones, preview_memory.header.n_zones)
        assert preview_default.snapshot is not None
        assert preview_memory.snapshot is not None
        np.testing.assert_allclose(
            preview_default.snapshot.fields["density"],
            preview_memory.snapshot.fields["density"],
            equal_nan=True,
        )

    def test_hdf5_field_chunking_is_snapshot_major(self) -> None:
        source = ROOT / "5Fe+4.9TW+light.log"
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "5Fe_chunked.h5"
            write_hdf5(source, output, overwrite=True)
            with h5py.File(output, "r") as handle:
                density = handle["fields"]["density"]
                self.assertIsNotNone(density.chunks)
                assert density.chunks is not None
                self.assertEqual(density.chunks[0], min(16, density.shape[0]))
                self.assertEqual(density.chunks[1], density.shape[1])

    def test_legacy_hdf5_compatibility_still_loads_after_ingest_changes(self) -> None:
        with HeliosRun(HDF5_ROOT / "Cu_0166_stabilized.h5") as run:
            self.assertEqual(run.get_metadata().get("geometry"), "PLANAR")
            density = run.get_field("density")
            self.assertEqual(density.shape[0], run.n_snapshots)


if __name__ == "__main__":
    unittest.main()
