from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
from _viewer_test_utils import ROOT, get_app, process_events, reset_test_settings, wait_until

from helios_app.main_app import HeliosParseViewMainWindow
from helios_app.session_state import reset_session_state


class AppBpfWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def open_window(self) -> HeliosParseViewMainWindow:
        reset_test_settings()
        reset_session_state()
        window = HeliosParseViewMainWindow()
        window.show()
        process_events()
        return window

    def test_bpf_only_preview_shows_unknown_metadata_and_conversion_source(self) -> None:
        source = ROOT / "new_data/5Fe+4.9TW+light/5Fe+4.9TW+light.bpf"
        window = self.open_window()
        try:
            self.assertEqual(window.open_log_action.text(), "Open HELIOS Input...")
            window._open_path(source)
            wait_until(lambda: window.current_preview is not None, 30.0)
            preview = window.current_preview
            assert preview is not None
            self.assertEqual(preview.input_format, "bpf")
            self.assertEqual(preview.primary_source, source)
            self.assertIsNone(preview.companion_source)
            self.assertIsNone(preview.geometry)
            self.assertIsNone(preview.n_materials)
            self.assertEqual(preview.companion_status, "not present")
            self.assertIn("Unknown", window.preview_summary.toPlainText())
            self.assertIn("complete frame validation has not run", preview.validation_scope)
            self.assertTrue(window.parse_button.isEnabled())
        finally:
            window.close()

    def test_latest_input_preview_wins_and_bpf_log_pair_is_explicit(self) -> None:
        window = self.open_window()
        try:
            cu_bpf = ROOT / "new_data/25Cu+1.87TW/25Cu+1.87TW.bpf"
            fe_bpf = ROOT / "new_data/5Fe+4.9TW+light/5Fe+4.9TW+light.bpf"
            window._open_path(cu_bpf)
            window._open_path(fe_bpf)
            wait_until(lambda: window.current_preview is not None, 30.0)
            preview = window.current_preview
            assert preview is not None
            self.assertEqual(preview.source, fe_bpf)

            window._open_path(cu_bpf)
            wait_until(lambda: window.current_preview is not None and window.current_preview.source == cu_bpf, 30.0)
            preview = window.current_preview
            assert preview is not None
            self.assertEqual(preview.primary_source, cu_bpf)
            self.assertEqual(preview.companion_source, cu_bpf.with_suffix(".log"))
            self.assertEqual(preview.companion_status, "alignment not checked")
            self.assertIn("full frame validation runs during conversion", " ".join(preview.warnings or []))
        finally:
            window.close()

    def test_bpf_only_input_can_convert_from_desktop_parser(self) -> None:
        source = ROOT / "new_data/5Fe+4.9TW+light/5Fe+4.9TW+light.bpf"
        temp_dir = Path(tempfile.mkdtemp())
        output = temp_dir / "bpf_only.h5"
        window = self.open_window()
        try:
            window._open_path(source)
            wait_until(lambda: window.current_preview is not None, 30.0)
            window.output_path_edit.setText(str(output))
            window.overwrite_checkbox.setChecked(True)
            window._start_parse_from_controls()
            wait_until(lambda: output.exists() and window.last_parse_result is not None, 60.0)
            self.assertEqual(window.last_parse_result.source, source)
            self.assertEqual(window.last_parse_result.output, output)
        finally:
            window.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_malformed_optional_log_is_reported_after_bpf_fallback(self) -> None:
        source = ROOT / "new_data/5Fe+4.9TW+light/5Fe+4.9TW+light.bpf"
        temp_dir = Path(tempfile.mkdtemp())
        bpf_copy = temp_dir / source.name
        shutil.copy2(source, bpf_copy)
        bpf_copy.with_suffix(".log").write_text("malformed optional companion\n", encoding="utf-8")
        output = temp_dir / "fallback.h5"
        window = self.open_window()
        try:
            window._open_path(bpf_copy)
            wait_until(lambda: window.current_preview is not None, 30.0)
            preview = window.current_preview
            assert preview is not None
            self.assertEqual(preview.companion_status, "optional companion unavailable")
            window.output_path_edit.setText(str(output))
            window.overwrite_checkbox.setChecked(True)
            window._start_parse_from_controls()
            wait_until(lambda: output.exists() and window.last_parse_result is not None, 60.0)
            self.assertTrue(any("could not be validated" in warning for warning in (window.last_parse_result.warnings or [])))
        finally:
            window.close()
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
