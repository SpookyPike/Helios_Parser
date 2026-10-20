from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helios.runtime import RunRevision


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
            self.assertNotEqual(first.open_token, second.open_token)
            self.assertNotEqual(first.cache_key, second.cache_key)

    def test_revision_is_canonical_and_keeps_conversion_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.h5"
            path.write_bytes(b"fixture")
            revision = RunRevision.create(path.parent / "." / path.name, conversion_id="convert-7")

            self.assertEqual(str(path.resolve()), revision.canonical_path)
            self.assertEqual("convert-7", revision.conversion_id)
            self.assertEqual(5, len(revision.cache_key))


if __name__ == "__main__":
    unittest.main()
