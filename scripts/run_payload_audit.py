from __future__ import annotations

"""Run the heavy viewer payload audit in isolated subprocesses.

``tests/test_viewer_payloads.py`` is numerically valuable but has proven less
stable as a single long-lived Qt unittest process. This helper keeps that audit
in the test suite while executing each test case in a fresh interpreter.
"""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TEST_NAMES = [
    "test_dynamic_radius_mode_stays_disabled_until_radius_payload_is_loaded",
    "test_2d_orientation_and_coordinate_payloads_match_reader",
    "test_snapshot_lineouts_and_time_traces_match_reader",
    "test_region_and_material_masks_apply_only_along_coordinate_dimension",
    "test_boundary_overlays_and_scale_modes_match_expected_payloads",
    "test_new_helios_format_field_and_diagnostic_loading_remain_consistent",
]


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
    for name in TEST_NAMES:
        command = [
            sys.executable,
            "-m",
            "unittest",
            f"test_viewer_payloads.ViewerPayloadTests.{name}",
        ]
        print(f"=== {name} ===", flush=True)
        completed = subprocess.run(command, cwd=str(ROOT), env=env)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
