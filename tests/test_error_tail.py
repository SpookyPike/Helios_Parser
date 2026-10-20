from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
from helios_app.parser_controller import _tail_snippet


class ErrorTailTests(unittest.TestCase):
    def test_log_tail_is_bounded_by_bytes_and_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "large.log"
            log.write_bytes(("old line\n" * 20000 + "tail-1\ntail-2\n").encode())
            snippet = _tail_snippet(log, line_count=2, max_bytes=1024)
            self.assertEqual(snippet.splitlines(), ["tail-1", "tail-2"])

    def test_bpf_never_renders_binary_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bpf = Path(tmpdir) / "broken.bpf"
            bpf.write_bytes(bytes(range(256)) * 256)
            self.assertEqual(_tail_snippet(bpf), "")


if __name__ == "__main__":
    unittest.main()
