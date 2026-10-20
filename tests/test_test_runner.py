from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts import run_test_suite


class TestSuiteRunnerTests(unittest.TestCase):
    def test_discovery_includes_every_test_module_and_smoke(self) -> None:
        entries = run_test_suite.discover_entries()
        names = {entry.name for entry in entries}
        expected = {path.name for path in (run_test_suite.ROOT / "tests").glob("test_*.py")}
        self.assertTrue(expected.issubset(names))
        self.assertIn("smoke_test.py", names)
        self.assertEqual(len(names), len(entries))

    def test_core_is_portable_subset_and_full_is_discovered(self) -> None:
        entries = run_test_suite.discover_entries()
        core = run_test_suite.select_entries(entries, "core", [])
        full = run_test_suite.select_entries(entries, "full", [])
        self.assertLess(len(core), len(full))
        self.assertEqual({entry.name for entry in core}, run_test_suite.CORE_NAMES)
        self.assertEqual({entry.name for entry in full}, {entry.name for entry in entries})

    def test_unknown_only_entry_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does_not_exist.py"):
            run_test_suite.select_entries(
                [run_test_suite.TestEntry("smoke_test.py", (), "smoke")],
                "core",
                ["does_not_exist.py"],
            )

    def test_cli_rejects_unknown_only_entry(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            run_test_suite.main(["--only", "does_not_exist.py"])
        self.assertEqual(raised.exception.code, 2)

    def test_fixture_preflight_reports_missing_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fixture.dat").write_text("current", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "fixtures": {
                            "present": {"path": "fixture.dat", "size_bytes": 7, "sha256": "wrong"},
                            "missing": {"path": "absent.dat", "provenance": "historical local data"},
                        },
                        "entry_requirements": {"one.py": ["present"], "two.py": ["missing"]},
                    }
                ),
                encoding="utf-8",
            )
            entries = [
                run_test_suite.TestEntry("one.py", (), "test"),
                run_test_suite.TestEntry("two.py", (), "test"),
            ]
            errors = run_test_suite.check_fixtures(entries, manifest, root)
        self.assertEqual(len(errors), 2)
        self.assertIn("hash mismatch", errors[0])
        self.assertIn("missing fixture", errors[1])

    def test_each_entry_gets_explicit_flags_and_private_settings_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = run_test_suite.build_env(
                run_test_suite.TestEntry("test_v1_release_gate.py", (), "test"), Path(tmp)
            )
            self.assertEqual(env["HELIOS_ENABLE_EXPERIMENTAL"], "")
            self.assertEqual(env["HELIOS_DEV_MODE"], "")
            self.assertTrue(env["HELIOS_TEST_SETTINGS_DIR"].startswith(tmp))
            self.assertTrue(Path(env["HELIOS_TEST_SETTINGS_DIR"]).is_dir())

    def test_known_heavy_qt_suites_have_named_timeout_overrides(self) -> None:
        self.assertEqual(run_test_suite.TIMEOUT_OVERRIDES["test_app_phase412.py"], 300)
        self.assertEqual(run_test_suite.TIMEOUT_OVERRIDES["test_wave6_derived.py"], 300)

    def test_runner_records_success_and_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "reports"
            report.mkdir()
            success = run_test_suite.TestEntry(
                "success.py", (sys.executable, "-c", "print('Ran 1 test, skipped=0')"), "test"
            )
            signaled = run_test_suite.TestEntry(
                "signal.py",
                (sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"),
                "test",
            )
            success_result = run_test_suite.run_entry(success, report)
            signal_result = run_test_suite.run_entry(signaled, report)
        self.assertEqual(success_result["return_code"], 0)
        self.assertEqual(success_result["skipped"], 0)
        self.assertEqual(signal_result["signal"], 15)
        self.assertNotEqual(signal_result["return_code"], 0)

    def test_assertion_failure_remains_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "reports"
            report.mkdir()
            failed = run_test_suite.TestEntry(
                "assertion.py", (sys.executable, "-c", "raise AssertionError('expected failure')"), "test"
            )
            result = run_test_suite.run_entry(failed, report)
            output = Path(result["log"]).read_text(encoding="utf-8")
        self.assertEqual(result["return_code"], 1)
        self.assertIn("AssertionError", output)

    def test_payload_wrapper_uses_unittest_method_discovery(self) -> None:
        from scripts.run_payload_audit import TEST_NAMES
        from test_viewer_payloads import ViewerPayloadTests

        expected = unittest.defaultTestLoader.getTestCaseNames(ViewerPayloadTests)
        self.assertEqual(TEST_NAMES, expected)

    def test_runner_times_out_and_keeps_partial_output(self) -> None:
        original = run_test_suite.TIMEOUT_OVERRIDES.get("slow.py")
        run_test_suite.TIMEOUT_OVERRIDES["slow.py"] = 0.05
        try:
            with tempfile.TemporaryDirectory() as tmp:
                report = Path(tmp) / "reports"
                report.mkdir()
                slow = run_test_suite.TestEntry(
                    "slow.py", (sys.executable, "-c", "import time; print('started', flush=True); time.sleep(2)"), "test"
                )
                result = run_test_suite.run_entry(slow, report)
                log = Path(result["log"]).read_text(encoding="utf-8")
        finally:
            if original is None:
                run_test_suite.TIMEOUT_OVERRIDES.pop("slow.py", None)
            else:
                run_test_suite.TIMEOUT_OVERRIDES["slow.py"] = original
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["return_code"], 124)
        self.assertIn("TIMEOUT", log)

    def test_runner_timeout_terminates_wrapper_descendants_before_continuing(self) -> None:
        original = run_test_suite.TIMEOUT_OVERRIDES.get("wrapper-child.py")
        run_test_suite.TIMEOUT_OVERRIDES["wrapper-child.py"] = 0.1
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                report = root / "reports"
                report.mkdir()
                marker = root / "late-child-write.txt"
                child_code = (
                    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    f"time.sleep(0.4); open({str(marker)!r}, 'w').write('late')"
                )
                wrapper_code = (
                    "import subprocess, sys; "
                    f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
                    "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                    "import time; time.sleep(2)"
                )
                wrapper = run_test_suite.TestEntry(
                    "wrapper-child.py", (sys.executable, "-c", wrapper_code), "test"
                )
                result = run_test_suite.run_entry(wrapper, report, root=root)
                next_entry = run_test_suite.TestEntry(
                    "after-timeout.py", (sys.executable, "-c", "print('next entry ran')"), "test"
                )
                next_result = run_test_suite.run_entry(next_entry, report, root=root)
                time.sleep(0.5)
                self.assertTrue(result["timed_out"])
                self.assertFalse(marker.exists(), "descendant wrote after timeout cleanup")
                self.assertEqual(0, next_result["return_code"])
        finally:
            if original is None:
                run_test_suite.TIMEOUT_OVERRIDES.pop("wrapper-child.py", None)
            else:
                run_test_suite.TIMEOUT_OVERRIDES["wrapper-child.py"] = original


if __name__ == "__main__":
    unittest.main()
