from __future__ import annotations

import re
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SCREENSHOT_DIR = DOCS / "assets" / "screenshots"


class RepositoryConsistencyTests(unittest.TestCase):
    def test_manual_references_current_screenshot_assets(self) -> None:
        manual = (DOCS / "index.html").read_text(encoding="utf-8")
        refs = sorted(set(re.findall(r'assets/screenshots/([A-Za-z0-9_.-]+\.png)', manual)))
        self.assertEqual(
            refs,
            [
                "derived_plasmon_current.png",
                "parser_preview_current.png",
                "viewer_cylindrical_radius_current.png",
                "viewer_mouse_probe_current.png",
                "viewer_zone_index_current.png",
            ],
        )
        for name in refs:
            self.assertTrue((SCREENSHOT_DIR / name).exists(), name)

    def test_docs_screenshot_directory_no_longer_contains_old_phase_assets(self) -> None:
        names = [path.name for path in SCREENSHOT_DIR.glob("*.png")]
        self.assertFalse(any(name.startswith("phase31_") for name in names))
        self.assertFalse(any(name.startswith("phase41_") for name in names))

    def test_capture_script_was_replaced_with_current_name(self) -> None:
        self.assertTrue((ROOT / "scripts" / "capture_docs_assets.py").exists())
        self.assertFalse((ROOT / "scripts" / "capture_phase41_docs.py").exists())


if __name__ == "__main__":
    unittest.main()
