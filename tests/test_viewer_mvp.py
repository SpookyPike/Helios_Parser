from __future__ import annotations

import unittest

from _viewer_test_utils import EXPECTED, HDF5_ROOT, find_row_by_data, get_app, reset_test_settings, wait_until

import _test_bootstrap  # noqa: F401
from helios_viewer.main_window import HeliosViewerMainWindow


class ViewerMVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_viewer_opens_representative_files(self) -> None:
        for name, expected in EXPECTED.items():
            with self.subTest(example=name):
                reset_test_settings()
                window = HeliosViewerMainWindow()
                try:
                    self.assertNotEqual(window.controller.worker_thread, self.app.thread())
                    window.load_file(HDF5_ROOT / name)
                    wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None, timeout_s=30.0)
                    wait_until(
                        lambda: (
                            window.field_map_widget.last_display_image is not None
                            and tuple(window.field_map_widget.last_display_image.shape) == (expected["zones"], expected["snapshots"])
                        )
                        or (
                            window.field_map_widget.last_mesh_z is not None
                            and tuple(window.field_map_widget.last_mesh_z.shape) == (expected["snapshots"], expected["zones"])
                        ),
                        timeout_s=30.0,
                    )

                    self.assertIn(f"Zones: {expected['zones']}", window.summary_text.toPlainText())
                    self.assertIn(f"Snapshots: {expected['snapshots']}", window.summary_text.toPlainText())
                    self.assertEqual(window.field_list.count(), len(window.run_payload.fields))
                    self.assertEqual(window.snapshot_slider.maximum(), expected["snapshots"] - 1)
                    self.assertIn("Density", window.field_label.text())
                    self.assertEqual(window.field_map_widget.current_orientation, "time_x_coord_y")
                    self.assertIn("Time [", window.field_map_widget.current_x_label)
                    self.assertTrue(
                        "Static x [" in window.field_map_widget.current_y_label
                        or "Moving-mesh x [" in window.field_map_widget.current_y_label
                        or "Moving-mesh radius [" in window.field_map_widget.current_y_label
                        or "Radius [" in window.field_map_widget.current_y_label
                        or "Zone index" in window.field_map_widget.current_y_label
                    )
                    self.assertIn("Density map", window.field_map_widget.current_title)
                    self.assertGreaterEqual(window.field_map_widget.current_boundary_count, max(expected["regions"] - 1, 0))

                    diagnostic_row = find_row_by_data(window.diagnostic_list, "energy_summary/current/ions")
                    if diagnostic_row >= 0:
                        window.diagnostic_list.setCurrentRow(diagnostic_row)
                        wait_until(lambda: window.current_diagnostic_payload is not None, timeout_s=30.0)
                        self.assertGreaterEqual(window.diagnostic_plot.current_curve_count, 1)
                        self.assertIn("Energy summary", window.diagnostic_plot.current_title)
                finally:
                    window.close()


if __name__ == "__main__":
    unittest.main()
