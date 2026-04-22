from __future__ import annotations

"""Reliable fresh-process test runner for HELIOS Analyzer.

Qt-heavy viewer tests are more stable when each module runs in its own process.
This script provides a CI-friendly execution model without changing the existing
unittest-based test modules.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TEST_COMMANDS: list[tuple[str, list[str]]] = [
    ("smoke_test.py", ["python", "tests/smoke_test.py"]),
    ("test_reader.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_reader.py"]),
    ("test_validation.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_validation.py"]),
    ("test_viewer_mvp.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_viewer_mvp.py"]),
    ("test_viewer_payloads.py", ["python", "scripts/run_payload_audit.py"]),
    ("test_viewer_interactions.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_viewer_interactions.py"]),
    ("test_viewer_phase2.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_viewer_phase2.py"]),
    ("test_viewer_phase2b.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_viewer_phase2b.py"]),
    ("test_viewer_phase2c.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_viewer_phase2c.py"]),
    ("test_app_phase3.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_app_phase3.py"]),
    ("test_app_phase31.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_app_phase31.py"]),
    ("test_app_phase32.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_app_phase32.py"]),
    ("test_app_phase32r.py", ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_app_phase32r.py"]),
]

CORE_TESTS = [
    "smoke_test.py",
    "test_reader.py",
    "test_viewer_mvp.py",
    "test_viewer_interactions.py",
    "test_viewer_phase2.py",
    "test_viewer_phase2b.py",
    "test_viewer_phase2c.py",
    "test_app_phase3.py",
    "test_app_phase31.py",
    "test_app_phase32.py",
    "test_app_phase32r.py",
]

FULL_TESTS = [name for name, _command in TEST_COMMANDS]


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = env.get("PYTHONPATH", "")
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HELIOS Analyzer tests in isolated processes.")
    parser.add_argument("--list", action="store_true", help="List the configured test modules and exit.")
    parser.add_argument(
        "--profile",
        choices=("core", "full"),
        default="core",
        help="Select the default module set. 'core' is the recommended CI path; 'full' also runs the heavy validation suite.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only test entries whose filename matches one of these values. Can be repeated.",
    )
    args = parser.parse_args(argv)

    allowed_names = set(CORE_TESTS if args.profile == "core" else FULL_TESTS)
    selected = [entry for entry in TEST_COMMANDS if entry[0] in allowed_names]
    if args.only:
        allowed = set(args.only)
        selected = [entry for entry in TEST_COMMANDS if entry[0] in allowed]

    if args.list:
        for name, _command in selected:
            print(name)
        return 0

    env = build_env()
    for name, command in selected:
        resolved_command = [sys.executable if part == "python" else part for part in command]
        print(f"=== {name} ===", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(resolved_command, cwd=str(ROOT), env=env)
        elapsed = time.perf_counter() - started
        print(f"--- {name}: {elapsed:.2f}s", flush=True)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
