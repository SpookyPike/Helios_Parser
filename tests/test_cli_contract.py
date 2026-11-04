"""The CLI contract: exit codes, streams, and the version flag.

These run the commands as a subprocess, because exit codes and stream
separation are the contract and cannot be asserted in-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BPF_SAMPLE = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"
LOG_SAMPLE = ROOT / "tests" / "fixtures" / "v12" / "cu_v12_prefix.log"
EXO_SAMPLE = ROOT / "tests" / "fixtures" / "radiation" / "zm0_nofreqrad.exo"


def _run(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        text=True,
        capture_output=True,
        check=False,
    )


class ConvertContractTests(unittest.TestCase):
    def test_version_exits_zero_and_prints_the_release(self) -> None:
        result = _run("helios_parser", "--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("helios_to_hdf5", result.stdout)
        sys.path.insert(0, str(SRC))
        from helios_release import RELEASE_VERSION

        self.assertIn(RELEASE_VERSION, result.stdout)

    def test_help_names_every_supported_input_type(self) -> None:
        result = _run("helios_parser", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for suffix in (".log", ".bpf", ".exo"):
            self.assertIn(suffix, result.stdout)

    def test_a_successful_conversion_writes_json_to_stdout_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.h5"
            result = _run("helios_parser", str(LOG_SAMPLE), str(output), "--json", "--quiet")
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["n_snapshots"], 10)
            self.assertEqual(payload["n_zones"], 300)
            self.assertGreater(payload["bytes"], 0)
            self.assertTrue(output.exists())
            # Nothing but the JSON object belongs on stdout.
            self.assertEqual(result.stdout.count("\n"), 1)

    def test_exodus_is_convertible_though_the_help_text_used_to_omit_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "exo.h5"
            result = _run("helios_parser", str(EXO_SAMPLE), str(output), "--quiet")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())

    def test_usage_errors_exit_two(self) -> None:
        result = _run("helios_parser")
        self.assertEqual(result.returncode, 2)

    def test_a_batch_keeps_going_and_reports_the_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run(
                "helios_parser",
                str(Path(tmpdir) / "missing.log"),
                str(LOG_SAMPLE),
                "--outdir",
                tmpdir,
                "--json",
                "--quiet",
            )
            self.assertEqual(result.returncode, 3, result.stderr)
            lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["status"], "error")
            self.assertEqual(lines[0]["error_kind"], "input_missing")
            self.assertEqual(lines[1]["status"], "ok")


class DiagContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.run_h5 = Path(cls._tmpdir.name) / "contract.h5"
        result = _run("helios_parser", str(LOG_SAMPLE), str(cls.run_h5), "--quiet")
        assert result.returncode == 0, result.stderr

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_version_exits_zero(self) -> None:
        result = _run("helios_diag", "--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("helios_diag", result.stdout)

    def test_help_lists_every_subcommand(self) -> None:
        result = _run("helios_diag", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("inspect", "stats", "query", "diff", "validate", "analyse"):
            self.assertIn(name, result.stdout)

    def test_a_missing_file_exits_three_with_a_json_error(self) -> None:
        result = _run("helios_diag", "inspect", "/tmp/definitely-not-here.h5")
        self.assertEqual(result.returncode, 3, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "error")
        self.assertIn("no such file", result.stderr)

    def test_stats_writes_one_json_object_and_logs_only_to_stderr(self) -> None:
        result = _run("helios_diag", "stats", str(self.run_h5), "-f", "density")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "stats")
        self.assertEqual(len(payload["rows"]), 300)

    def test_an_experimental_module_needs_the_flag_and_is_a_usage_error_without_it(self) -> None:
        """The backend hides plasmon and transmission; the CLI must say so and exit 2
        rather than report the file as unreadable."""
        result = _run("helios_diag", "analyse", str(self.run_h5), "--module", "plasmon")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--include-experimental", result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["error_kind"], "usage")

    def test_the_experimental_flag_actually_runs_the_module(self) -> None:
        result = _run("helios_diag", "analyse", str(self.run_h5), "--module", "plasmon", "--include-experimental")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        plasmon = payload["result"]["plasmon"]
        self.assertNotEqual(plasmon.get("regime_label"), "disabled")
        self.assertTrue(any("experimental" in note for note in payload["notes"]))

    def test_a_preset_flag_does_not_defeat_include_experimental(self) -> None:
        """setdefault would let a preset HELIOS_ENABLE_EXPERIMENTAL=0 win, returning a
        successful payload with the module disabled."""
        result = subprocess.run(
            [sys.executable, "-m", "helios_diag", "analyse", str(self.run_h5),
             "--module", "plasmon", "--include-experimental"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC), "HELIOS_ENABLE_EXPERIMENTAL": "0"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        plasmon = json.loads(result.stdout)["result"]["plasmon"]
        self.assertNotEqual(plasmon.get("regime_label"), "disabled")

    def test_include_experimental_does_not_leak_into_the_caller(self) -> None:
        import types

        from helios_diag import analyse

        os.environ.pop("HELIOS_ENABLE_EXPERIMENTAL", None)
        try:
            analyse.run(
                types.SimpleNamespace(
                    path=str(self.run_h5), module=["xrd"], snapshot=None,
                    max_array=1000, include_experimental=True,
                )
            )
            self.assertNotIn("HELIOS_ENABLE_EXPERIMENTAL", os.environ)
        finally:
            os.environ.pop("HELIOS_ENABLE_EXPERIMENTAL", None)

    def test_include_experimental_does_not_leak_when_analysis_raises(self) -> None:
        import types

        from helios_diag import analyse

        os.environ["HELIOS_ENABLE_EXPERIMENTAL"] = "0"
        try:
            with self.assertRaises(FileNotFoundError):
                analyse.run(
                    types.SimpleNamespace(
                        path="/tmp/definitely-not-a-run.h5", module=["xrd"], snapshot=None,
                        max_array=1000, include_experimental=True,
                    )
                )
            self.assertEqual(os.environ.get("HELIOS_ENABLE_EXPERIMENTAL"), "0")
        finally:
            os.environ.pop("HELIOS_ENABLE_EXPERIMENTAL", None)

    def test_the_headless_path_never_imports_qt(self) -> None:
        """A diagnostics run on a compute node must not need a display."""
        probe = (
            "import sys; sys.argv=['helios_diag','inspect',sys.argv[1]];"
            "from helios_diag.cli import main; main();"
            "print('QT' if 'PySide6' in sys.modules else 'HEADLESS')"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe, str(self.run_h5)],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC)},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HEADLESS", result.stdout)


if __name__ == "__main__":
    unittest.main()