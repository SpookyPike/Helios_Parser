"""Self-description: an output must stand alone, on either file era.

The legacy case is produced by stripping the modern-only parts from a real
conversion, which is a more faithful pre-2.0 shape than a hand-built tree and
cannot drift from what the writer actually emits.
"""

from __future__ import annotations

import json
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

REQUIRED_KEYS = ("command", "tool_version", "generated_at", "file", "notes")


def _reject_constant(name: str):
    raise ValueError(f"non-finite literal in JSON: {name}")


def _run(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        text=True,
        capture_output=True,
        check=False,
    )


def _strict_json(text: str):
    """Parse with a parser that refuses NaN and Infinity, unlike Python's default."""
    return json.loads(text, parse_constant=_reject_constant)


def _make_legacy(path: Path) -> None:
    """Strip everything 2.0 added, leaving the shape a pre-2.0 file has.

    A pre-2.0 file records only the unit on a field dataset, and no schema or
    parse metadata, so the label has to come back as inferred.
    """
    with h5py.File(path, "r+") as handle:
        if "field_metadata" in handle:
            del handle["field_metadata"]
        for attribute in ("schema_version", "format", "geometry"):
            handle.attrs.pop(attribute, None)
        for name in handle["fields"]:
            dataset = handle["fields"][name]
            for attribute in ("label", "dimensions", "source", "status", "field_name", "quantity"):
                dataset.attrs.pop(attribute, None)
        metadata = handle["metadata"]
        for name in (
            "parse_mode",
            "coordinate_model",
            "run_status",
            "source_precedence",
            "source_files",
            "schema_version",
        ):
            if name in metadata:
                del metadata[name]


class SelfDescriptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        modern = Path(cls._tmpdir.name) / "modern.h5"
        assert _run("helios_parser", str(LOG_SAMPLE), str(modern), "--quiet").returncode == 0
        cls.legacy_source = Path(cls._tmpdir.name) / "legacy_source.h5"
        modern.rename(cls.legacy_source)
        _make_legacy(cls.legacy_source)

    def setUp(self) -> None:
        # Each test gets its own copy: several strip attributes, and a shared file
        # would make the result depend on test order.
        self.legacy = Path(self._tmpdir.name) / f"{self._testMethodName}.h5"
        shutil.copy2(self.legacy_source, self.legacy)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_every_subcommand_carries_the_envelope(self) -> None:
        commands = {
            "inspect": ["inspect", str(self.legacy)],
            "stats": ["stats", str(self.legacy), "-f", "density"],
            "query": ["query", str(self.legacy), "-f", "density", "--snapshot", "0"],
            "validate": ["validate", str(self.legacy)],
            "diff": ["diff", str(self.legacy), str(self.legacy)],
        }
        for name, argv in commands.items():
            result = _run("helios_diag", *argv)
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")
            payload = _strict_json(result.stdout)
            for key in REQUIRED_KEYS:
                self.assertIn(key, payload, f"{name} is missing {key}")

    def test_a_legacy_file_reports_what_it_could_not_determine(self) -> None:
        result = _run("helios_diag", "inspect", str(self.legacy))
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = _strict_json(result.stdout)
        file_block = payload["file"]
        self.assertIsNone(file_block["schema_version"])
        self.assertIsNone(file_block["parse_mode"])
        self.assertEqual(file_block["run_status"]["source"], "legacy_hdf5")
        joined = " ".join(payload["notes"])
        self.assertIn("schema_version not recorded", joined)

    def test_a_legacy_field_label_is_marked_inferred_but_keeps_its_unit(self) -> None:
        payload = _strict_json(_run("helios_diag", "inspect", str(self.legacy)).stdout)
        density = next(field for field in payload["fields"] if field["name"] == "density")
        self.assertEqual(density["label_origin"], "inferred")
        # Legacy files do record the unit, so it must survive the degradation.
        self.assertEqual(density["unit"], "g/cm3")

    def test_geometry_is_normalized_and_the_raw_value_is_kept(self) -> None:
        with h5py.File(self.legacy, "r+") as handle:
            handle.attrs["geometry"] = "PLANAR"
        payload = _strict_json(_run("helios_diag", "inspect", str(self.legacy)).stdout)
        self.assertEqual(payload["file"]["geometry"], "planar")
        self.assertEqual(payload["file"]["geometry_raw"], "PLANAR")

    def test_a_field_without_a_recorded_unit_reports_null_and_a_note(self) -> None:
        with h5py.File(self.legacy, "r+") as handle:
            del handle["fields/density"].attrs["unit"]
            if "units" in handle["fields/density"].attrs:
                del handle["fields/density"].attrs["units"]
        payload = _strict_json(_run("helios_diag", "inspect", str(self.legacy)).stdout)
        density = next(field for field in payload["fields"] if field["name"] == "density")
        self.assertIsNone(density["unit"])
        self.assertTrue(any("unit not recorded" in note for note in payload["notes"]))

    def test_statistics_report_their_accounting(self) -> None:
        payload = _strict_json(_run("helios_diag", "stats", str(self.legacy), "-f", "density").stdout)
        accounting = payload["accounting"]
        self.assertEqual(accounting["n_used"] + accounting["n_nan"], accounting["n_values"])

    def test_no_output_contains_a_non_finite_literal(self) -> None:
        for argv in (
            ["inspect", str(self.legacy)],
            ["stats", str(self.legacy), "-f", "density"],
            ["validate", str(self.legacy)],
        ):
            result = _run("helios_diag", *argv)
            self.assertNotIn("NaN", result.stdout, argv)
            self.assertNotIn("Infinity", result.stdout, argv)
            _strict_json(result.stdout)  # raises if one slipped through


if __name__ == "__main__":
    unittest.main()