from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import os
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
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Parse error:", result.stderr)
            self.assertIn("Reason:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
