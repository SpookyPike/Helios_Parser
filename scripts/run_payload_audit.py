from __future__ import annotations

"""Run the heavy viewer payload audit in isolated subprocesses.

``tests/test_viewer_payloads.py`` is numerically valuable but has proven less
stable as a single long-lived Qt unittest process. This helper keeps that audit
in the test suite while executing each test case in a fresh interpreter.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_viewer_payloads import ViewerPayloadTests


ROOT = Path(__file__).resolve().parents[1]

TEST_NAMES = unittest.defaultTestLoader.getTestCaseNames(ViewerPayloadTests)


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = env.get("PYTHONPATH", "")
    values = [str(ROOT / "src"), str(ROOT / "tests")]
    if existing:
        values.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(values)
    return env


def main() -> int:
    env = build_env()
    outcomes = []
    for name in TEST_NAMES:
        command = [
            sys.executable,
            "-m",
            "unittest",
            f"test_viewer_payloads.ViewerPayloadTests.{name}",
        ]
        print(f"=== {name} ===", flush=True)
        completed = subprocess.run(command, cwd=str(ROOT), env=env)
        outcomes.append((name, completed.returncode))
        print(f"--- {name}: {'PASS' if completed.returncode == 0 else 'FAIL'} ({completed.returncode})", flush=True)
    return 0 if all(code == 0 for _name, code in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
