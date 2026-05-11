from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


class ViewerBpfAxesSmokeTests(unittest.TestCase):
    def test_bpf_non_zone_axes_viewer_smoke(self) -> None:
        script = textwrap.dedent(
            r"""
            import json
            import os
            import sys
            import tempfile
            import traceback
            from pathlib import Path

            import numpy as np

            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            root = Path.cwd()
            sys.path.insert(0, str(root / "src"))
            sys.path.insert(0, str(root / "tests"))

            from PySide6 import QtCore, QtWidgets
            from _viewer_test_utils import find_row_by_data, wait_until
            from helios_parser import write_hdf5
            from helios_viewer.main_window import HeliosViewerMainWindow

            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            errors = []

            def hook(exc_type, exc, tb):
                errors.append("".join(traceback.format_exception(exc_type, exc, tb)))

            sys.excepthook = hook

            def process_events(count=3):
                for _ in range(count):
                    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
                    QtCore.QThread.msleep(20)

            def select_field(window, field_name, wait_for_payload=True):
                if find_row_by_data(window.field_list, field_name) < 0:
                    window.show_alias_fields_checkbox.setChecked(True)
                    process_events()
                row = find_row_by_data(window.field_list, field_name)
                assert row >= 0, field_name
                window.field_list.setCurrentRow(row)
                if wait_for_payload:
                    wait_until(
                        lambda: window.current_field_payload is not None and window.current_field_payload.field_name == field_name,
                        timeout_s=30.0,
                    )

            result = {"field_checks": {}, "errors": errors}
            tmpdir = tempfile.TemporaryDirectory()
            try:
                output = Path(tmpdir.name) / "bpf_schema2.h5"
                write_hdf5(root / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf", output, overwrite=True, compression="lzf")
                window = HeliosViewerMainWindow()
                window.load_file(output)
                wait_until(lambda: window.run_payload is not None and window.field_list.count() > 0, timeout_s=30.0)

                result["density_visible_default"] = find_row_by_data(window.field_list, "density") >= 0
                result["mass_density_visible_default"] = find_row_by_data(window.field_list, "mass_density_g_cm3") >= 0
                window.show_alias_fields_checkbox.setChecked(True)
                process_events()
                result["density_visible_with_aliases"] = find_row_by_data(window.field_list, "density") >= 0
                window.field_status_filter_combo.setCurrentIndex(window.field_status_filter_combo.findData("unknown_bpf_record"))
                process_events()
                result["raw_visible_with_filter"] = find_row_by_data(window.field_list, "bpf_record_03") >= 0
                result["canonical_hidden_with_raw_filter"] = find_row_by_data(window.field_list, "mass_density_g_cm3") < 0
                window.field_status_filter_combo.setCurrentIndex(window.field_status_filter_combo.findData("__all__"))
                process_events()

                for field_name, axis_label, expected_width in [
                    ("radiation_net_flux_rmin_j_s_cm2_eV", "Frequency group", 200),
                    ("bpf_record_03", "BPF record value index", 50),
                    ("node_position_cm", "Node index", 501),
                    ("boundary_net_flux_pair_j_s_cm2", "Boundary index", 2),
                ]:
                    select_field(window, field_name)
                    result["field_checks"][field_name] = {
                        "map_shape": list(window.field_map_widget.last_display_image.shape),
                        "map_y_label": window.field_map_widget.current_y_label,
                        "line_x_shape": list(np.asarray(window.lineout_plot.last_x_values).shape),
                        "line_y_shape": list(np.asarray(window.lineout_plot.last_y_series[0]).shape),
                        "line_x_label": window.lineout_plot.current_x_label,
                        "expected_width": expected_width,
                        "expected_axis": axis_label,
                    }

                select_field(window, "ionization_fractions_by_zone_charge", wait_for_payload=False)
                wait_until(lambda: "not plotted" in window.status_message.text(), timeout_s=30.0)
                result["unsupported_message"] = window.status_message.text()
                result["errors"] = errors
                window.controller.shutdown()
                window.close()
                process_events(8)
            finally:
                tmpdir.cleanup()

            print("VIEWER_SMOKE_JSON=" + json.dumps(result, sort_keys=True), flush=True)
            os._exit(0)
            """
        )
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "tests")}
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        marker = "VIEWER_SMOKE_JSON="
        line = next((line for line in completed.stdout.splitlines() if line.startswith(marker)), "")
        self.assertTrue(line, completed.stdout + completed.stderr)
        data = json.loads(line[len(marker) :])
        self.assertFalse(data["density_visible_default"])
        self.assertTrue(data["mass_density_visible_default"])
        self.assertTrue(data["density_visible_with_aliases"])
        self.assertTrue(data["raw_visible_with_filter"])
        self.assertTrue(data["canonical_hidden_with_raw_filter"])
        for check in data["field_checks"].values():
            width = int(check["expected_width"])
            self.assertEqual(check["map_shape"], [width, 8])
            self.assertEqual(check["map_y_label"], check["expected_axis"])
            self.assertEqual(check["line_x_shape"], [width])
            self.assertEqual(check["line_y_shape"], [width])
            self.assertEqual(check["line_x_label"], check["expected_axis"])
        self.assertIn("not plotted", data["unsupported_message"])
        self.assertEqual(data["errors"], [])


if __name__ == "__main__":
    unittest.main()
