"""``helios_diag diff`` and ``validate``: the two gate commands.

Both exit 7 when they find something, which is what makes them usable in a
pipeline, so the exit code is asserted rather than the prose.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
import h5py

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOG_SAMPLE = ROOT / "tests" / "fixtures" / "v12" / "cu_v12_prefix.log"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "helios_diag", *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        text=True,
        capture_output=True,
        check=False,
    )


class DiffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.base = Path(cls._tmpdir.name) / "base.h5"
        assert _run("validate", "x").returncode != 0  # warm the module path
        result = subprocess.run(
            [sys.executable, "-m", "helios_parser", str(LOG_SAMPLE), str(cls.base), "--quiet"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC)},
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_identical_files_exit_zero(self) -> None:
        result = _run("diff", str(self.base), str(self.base))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"status": "ok"', result.stdout.replace("\n", " "))

    def test_a_perturbed_field_exits_seven_and_names_the_dataset(self) -> None:
        changed = Path(self._tmpdir.name) / "changed.h5"
        shutil.copy2(self.base, changed)
        with h5py.File(changed, "r+") as handle:
            handle["fields/density"][0, 0] = handle["fields/density"][0, 0] + 1.0
        result = _run("diff", str(self.base), str(changed))
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("fields/density", result.stdout)
        self.assertIn("differs", result.stdout)

    def test_a_unit_change_fails_even_when_every_value_matches(self) -> None:
        """Identical numbers in different units are not the same result."""
        changed = Path(self._tmpdir.name) / "reunited.h5"
        shutil.copy2(self.base, changed)
        with h5py.File(changed, "r+") as handle:
            handle["fields/density"].attrs["unit"] = "kg/m3"
            handle["fields/density"].attrs["units"] = "kg/m3"
        result = _run("diff", str(self.base), str(changed), "-f", "density")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("unit_mismatch", result.stdout)

    def test_a_shape_mismatch_is_reported_not_raised(self) -> None:
        changed = Path(self._tmpdir.name) / "reshaped.h5"
        shutil.copy2(self.base, changed)
        with h5py.File(changed, "r+") as handle:
            data = handle["fields/density"][()][:, :10]
            del handle["fields/density"]
            handle["fields"].create_dataset("density", data=data)
        result = _run("diff", str(self.base), str(changed))
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("shape_mismatch", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_dataset_missing_on_one_side_is_structural(self) -> None:
        changed = Path(self._tmpdir.name) / "trimmed.h5"
        shutil.copy2(self.base, changed)
        with h5py.File(changed, "r+") as handle:
            del handle["fields/velocity"]
        result = _run("diff", str(self.base), str(changed))
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("only_in_left", result.stdout)


class ValidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.good = Path(cls._tmpdir.name) / "good.h5"
        result = subprocess.run(
            [sys.executable, "-m", "helios_parser", str(LOG_SAMPLE), str(cls.good), "--quiet"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC)},
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_a_clean_conversion_passes(self) -> None:
        result = _run("validate", str(self.good))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"fail": 0', result.stdout)

    def test_a_missing_required_group_fails(self) -> None:
        broken = Path(self._tmpdir.name) / "broken.h5"
        shutil.copy2(self.good, broken)
        with h5py.File(broken, "r+") as handle:
            del handle["materials"]
        result = _run("validate", str(broken))
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("materials", result.stdout)

    def test_the_zero_extent_check_does_not_false_positive(self) -> None:
        """A chunked zero-extent dataset cannot exist: HDF5 refuses to create one.
        The 2.0.1 invariant is therefore enforced by the library, and the check
        here only has to stay quiet on a file that has no empty families."""
        result = _run("validate", str(self.good))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("zero_extent_families", result.stdout)


if __name__ == "__main__":
    unittest.main()