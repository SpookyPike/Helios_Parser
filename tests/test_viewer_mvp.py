from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
from _slow_tests import slow
from _viewer_test_utils import (
    EXPECTED,
    HDF5_ROOT,
    close_window,
    find_row_by_data,
    get_app,
    process_events,
    reset_test_settings,
    wait_until,
)
from PySide6 import QtCore

from helios_viewer.main_window import HeliosViewerMainWindow


@slow("drives a real Qt event loop")
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
                    # `pressure` is a compatibility alias, hidden by default.
                    window.show_alias_fields_checkbox.setChecked(True)
                    process_events()
                    listed_fields = {
                        str(window.field_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole))
                        for index in range(window.field_list.count())
                        if window.field_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole) is not None
                    }
                    self.assertTrue(set(window.run_payload.fields).issubset(listed_fields))
                    self.assertEqual(len(listed_fields), sum(
                        window.field_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole) is not None
                        for index in range(window.field_list.count())
                    ))
                    self.assertIn("pressure", listed_fields)
                    self.assertEqual(window.snapshot_slider.maximum(), expected["snapshots"] - 1)
                    # The label prefers the field's own metadata label when the
                    # file records one ("Mass dens" for density), so assert that
                    # the label identifies the selected field by either form.
                    label_text = window.field_label.text().casefold()
                    selected = window.current_field_payload.field_name
                    metadata = window.run_payload.field_metadata.get(selected) or {}
                    metadata_label = str(metadata.get("label", "") or "").strip().casefold()
                    self.assertTrue(
                        selected.casefold() in label_text or (metadata_label and metadata_label in label_text),
                        f"field label {label_text!r} identifies neither {selected!r} nor {metadata_label!r}",
                    )
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
                    close_window(window)


if __name__ == "__main__":
    unittest.main()
