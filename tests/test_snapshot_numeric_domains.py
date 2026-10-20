from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
from helios_parser import HeliosParser
from helios_parser.parser import SnapshotDomainError


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "5Fe+4.9TW+light.log"


class SnapshotNumericDomainTests(unittest.TestCase):
    def test_fractional_cycle_is_rejected_before_integer_conversion(self) -> None:
        parser = HeliosParser()
        with parser.open_document(SAMPLE) as document:
            header = document.inspect()
            block = document.index.snapshot_blocks[0]
            text = document.buffer.slice(block.span)
            lines = text.splitlines()
            cycle_header = next(index for index, line in enumerate(lines) if re.match(r"^\s*Cycle\s+Time\s+\(s\)", line))
            fields = lines[cycle_header + 1].split()
            fields[0] = "1.5"
            lines[cycle_header + 1] = "  ".join(fields)
            with self.assertRaisesRegex(SnapshotDomainError, "finite integer"):
                document.snapshot_parser.parse("\n".join(lines), header)

    def test_decreasing_snapshot_time_is_fatal(self) -> None:
        text = SAMPLE.read_text(encoding="utf-8", errors="replace")
        headers = list(re.finditer(r"(?m)^\s*Cycle\s+Time\s+\(s\)", text))
        self.assertGreaterEqual(len(headers), 2)
        lines = text.splitlines()
        header_lines = [index for index, line in enumerate(lines) if re.match(r"^\s*Cycle\s+Time\s+\(s\)", line)]
        cycle_row = lines[header_lines[1] + 1].split()
        cycle_row[1] = "-1.00000e+000"
        lines[header_lines[1] + 1] = "  ".join(cycle_row)
        with tempfile.TemporaryDirectory() as tmpdir:
            malformed = Path(tmpdir) / "decreasing.log"
            malformed.write_text("\n".join(lines) + "\n", encoding="utf-8")
            parser = HeliosParser()
            with parser.open_document(malformed) as document:
                iterator = document.iter_snapshots_streaming()
                next(iterator)
                with self.assertRaisesRegex(ValueError, "nondecreasing"):
                    next(iterator)


if __name__ == "__main__":
    unittest.main()
