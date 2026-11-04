from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
BPF_SAMPLE = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"


class CliErrorTests(unittest.TestCase):
    def test_truncated_bpf_reports_concise_parse_error_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            truncated = tmpdir_path / "truncated.bpf"
            truncated.write_bytes(BPF_SAMPLE.read_bytes()[:8192])
            output = tmpdir_path / "bad.h5"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "helios_parser",
                    str(truncated),
                    str(output),
                    "--compression",
                    "lzf",
                    "--overwrite",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 4, result.stderr)
            # The message names the input and what was wrong with it, not a blanket
            # "Parse error:" that also appeared for a missing file or an existing output.
            self.assertIn(str(truncated), result.stderr)
            self.assertIn("cannot read as HELIOS output", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_a_missing_input_is_not_reported_as_a_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-m", "helios_parser", str(Path(tmpdir) / "nope.log"), str(Path(tmpdir) / "x.h5")],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertIn("no such file or directory", result.stderr)
            self.assertNotIn("Parse error", result.stderr)

    def test_a_missing_output_directory_is_a_write_error_not_an_input_error(self) -> None:
        """The writer raises FileNotFoundError for a missing output directory too,
        so the code has to disambiguate rather than blame the input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "no-such-dir" / "out.h5"
            result = subprocess.run(
                [sys.executable, "-m", "helios_parser", str(BPF_SAMPLE), str(missing)],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 6, result.stderr)
            self.assertIn("output directory does not exist", result.stderr)
            self.assertNotIn("no such file or directory", result.stderr)

    def test_an_existing_output_is_not_reported_as_a_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "taken.h5"
            output.write_bytes(b"")
            result = subprocess.run(
                [sys.executable, "-m", "helios_parser", str(BPF_SAMPLE), str(output)],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 5, result.stderr)
            self.assertIn("--overwrite", result.stderr)
            self.assertNotIn("Parse error", result.stderr)


if __name__ == "__main__":
    unittest.main()
