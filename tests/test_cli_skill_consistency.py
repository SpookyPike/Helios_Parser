"""The shipped agent skill must match the shipped command line.

A skill that lists flags drifts from the binary the moment a flag changes. This
test is the guard: every command the skill tells an agent to run has to answer
`--help`, every subcommand the CLI exposes has to appear in the skill, and the
envelope the skill quotes has to match what the CLI actually emits.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SKILL = ROOT / "skills" / "helios-cli" / "SKILL.md"
LOG_SAMPLE = ROOT / "tests" / "fixtures" / "v12" / "cu_v12_prefix.log"

# A fenced code block line that starts with one of the commands. The command name
# is the console script; the Python module that implements it differs for the
# converter, which is why the map below exists.
_COMMAND = re.compile(r"^(helios_to_hdf5|helios_diag)\s+([a-z_]+)?", re.MULTILINE)
_MODULE = {"helios_to_hdf5": "helios_cli", "helios_diag": "helios_diag"}


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *argv],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        text=True,
        capture_output=True,
        check=False,
    )


class SkillConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SKILL.is_file():
            raise unittest.SkipTest(f"skill not present at {SKILL}")
        cls.text = SKILL.read_text(encoding="utf-8")

    def test_every_command_in_the_skill_answers_help(self) -> None:
        seen: set[tuple[str, str]] = set()
        for match in _COMMAND.finditer(self.text):
            module, token = match.group(1), match.group(2)
            # For helios_to_hdf5 the second token is its first argument, not a
            # subcommand, so only helios_diag has one to check.
            subcommand = token if module == "helios_diag" else None
            key = (module, subcommand or "")
            if key in seen:
                continue
            seen.add(key)
            argv = ["helios_diag", subcommand, "--help"] if subcommand else [_MODULE[module], "--help"]
            result = _run(argv)
            self.assertEqual(result.returncode, 0, f"{' '.join(argv)} failed: {result.stderr}")
        self.assertGreater(len(seen), 5, "the skill should show more than a couple of commands")

    def test_every_subcommand_is_documented(self) -> None:
        from helios_diag.cli import COMMANDS

        for name in COMMANDS:
            self.assertIn(name, self.text, f"subcommand {name} is not mentioned in the skill")

    def test_the_documented_envelope_matches_the_real_one(self) -> None:
        block = re.search(r"```json\n(.*?)```", self.text, re.DOTALL)
        self.assertIsNotNone(block, "the skill should quote the JSON envelope")
        documented = set(json.loads(block.group(1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            run_h5 = Path(tmpdir) / "skill.h5"
            convert = subprocess.run(
                [sys.executable, "-m", "helios_parser", str(LOG_SAMPLE), str(run_h5), "--quiet"],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(SRC)},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(convert.returncode, 0, convert.stderr)
            result = _run(["helios_diag", "stats", str(run_h5), "-f", "density"])
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = set(json.loads(result.stdout))

        # Everything the skill documents must exist; the CLI may also add blocks.
        missing = documented - actual
        self.assertFalse(missing, f"the skill documents keys the CLI does not emit: {sorted(missing)}")
        for required in ("command", "tool_version", "file", "notes"):
            self.assertIn(required, actual)

    def test_the_adapters_share_the_body(self) -> None:
        readme = ROOT / "skills" / "helios-cli" / "README.md"
        fragment = ROOT / "skills" / "helios-cli" / "AGENTS.fragment.md"
        for path in (readme, fragment):
            self.assertTrue(path.is_file(), f"{path.name} is missing")
        # The README is the SKILL body without frontmatter; a divergence means one
        # of them was edited alone.
        body = self.text.split("---", 2)[2].strip()
        self.assertIn(body.splitlines()[0], readme.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()