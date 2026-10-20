from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import h5py

import _test_bootstrap  # noqa: F401
from helios.cache import clear_session_raw_data_cache
from helios.runtime import RunRevision
from helios.services.derived.common import load_run_data
from helios_parser import write_hdf5


TRACKED_BPF = (
    Path(__file__).resolve().parents[1]
    / "new_data"
    / "5Fe+4.9TW+light"
    / "5Fe+4.9TW+light.bpf"
)


class RunRevisionTests(unittest.TestCase):
    def test_each_explicit_open_has_a_distinct_key_for_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.h5"
            path.write_bytes(b"fixture")

            first = RunRevision.create(path)
            second = RunRevision.create(path)

            self.assertEqual(first.canonical_path, second.canonical_path)
            self.assertEqual(first.file_size, second.file_size)
            self.assertEqual(first.modified_ns, second.modified_ns)
            self.assertEqual(first.file_device, second.file_device)
            self.assertEqual(first.file_inode, second.file_inode)
            self.assertNotEqual(first.open_token, second.open_token)
            self.assertNotEqual(first.cache_key, second.cache_key)

    def test_revision_is_canonical_and_keeps_conversion_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.h5"
            path.write_bytes(b"fixture")
            revision = RunRevision.create(path.parent / "." / path.name, conversion_id="convert-7")

            self.assertEqual(str(path.resolve()), revision.canonical_path)
            self.assertEqual("convert-7", revision.conversion_id)
            self.assertEqual(7, len(revision.cache_key))

    def test_same_path_reopen_loads_replaced_values_when_size_and_mtime_match(self) -> None:
        clear_session_raw_data_cache(reason="same-path-revision-test")
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                path = Path(temporary_directory) / "same-path-replacement.h5"
                write_hdf5(TRACKED_BPF, path)
                original_stat = path.stat()
                first_revision = RunRevision.create(path)
                first = load_run_data(path, revision=first_revision, requested_fields=())
                original_value = float(first.density_g_cm3[0, 0])

                with h5py.File(path, "r+") as handle:
                    density = handle["fields/density"]
                    density[0, 0] = original_value + 1.0
                os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                replaced_stat = path.stat()

                self.assertEqual(original_stat.st_size, replaced_stat.st_size)
                self.assertEqual(original_stat.st_mtime_ns, replaced_stat.st_mtime_ns)
                self.assertEqual(original_stat.st_dev, replaced_stat.st_dev)
                self.assertEqual(original_stat.st_ino, replaced_stat.st_ino)
                second_revision = RunRevision.create(path)
                self.assertNotEqual(first_revision.cache_key, second_revision.cache_key)

                reopened = load_run_data(path, revision=second_revision, requested_fields=())
                self.assertEqual(original_value + 1.0, float(reopened.density_g_cm3[0, 0]))
        finally:
            clear_session_raw_data_cache(reason="same-path-revision-test-complete")


if __name__ == "__main__":
    unittest.main()
