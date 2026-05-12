from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401

from helios_viewer.main_window import _metadata_text


class ViewerFieldLabelUxTests(unittest.TestCase):
    def test_legacy_unknown_fallback_is_not_shown_as_user_label(self) -> None:
        self.assertEqual(_metadata_text({"status": "legacy", "source": "unknown"}), "")

    def test_known_bpf_metadata_remains_visible(self) -> None:
        self.assertEqual(_metadata_text({"status": "unknown_bpf_record", "source": "bpf"}), "raw BPF")


if __name__ == "__main__":
    unittest.main()
