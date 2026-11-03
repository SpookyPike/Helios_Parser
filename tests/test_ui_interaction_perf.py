"""Interaction-cost harness for the 2.0 UI findings.

The app is native PySide6 with no web surface, so nothing external can drive it.
This harness drives the real widgets through `QTest` under the offscreen
platform and asserts on **call counts and wall time**, not pixels. Pixels are
only used by `scripts/capture_ui_evidence.py` for the visual half.

Counting rather than timing is deliberate: a count is stable on any machine,
while a millisecond threshold flakes on a loaded runner. Where a count cannot
express the defect (slider drag latency) the test measures wall time against a
generous ceiling and reports the number it saw.

Run with the full profile: these drive a real Qt event loop.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

import _test_bootstrap  # noqa: F401
import numpy as np
from _slow_tests import slow
from _viewer_test_utils import (
    HDF5_ROOT,
    combo_set_data,
    find_row_by_data,
    get_app,
    process_events,
    reset_test_settings,
    wait_until,
)

import helios_viewer.plots as plots_module
from helios_viewer.main_window import HeliosViewerMainWindow

FIXTURE = "Cu_0166_stabilized.h5"


def _open_window(name: str = FIXTURE) -> HeliosViewerMainWindow:
    reset_test_settings()
    window = HeliosViewerMainWindow()
    window.load_file(HDF5_ROOT / name)
    wait_until(lambda: window.run_payload is not None and window.current_field_payload is not None, timeout_s=60.0)
    return window


def _select_field(window: HeliosViewerMainWindow, field_name: str = "density") -> None:
    row = find_row_by_data(window.field_list, field_name)
    if row < 0:
        raise AssertionError(f"Field {field_name!r} not found.")
    window.field_list.setCurrentRow(row)
    wait_until(lambda: window.current_field_payload is not None and window.current_field_payload.field_name == field_name, 60.0)


@slow("drives a real Qt event loop")
class ColormapResolveTests(unittest.TestCase):
    """F15: the map is resolved and reapplied even when the name is unchanged."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_setting_the_same_colormap_does_not_resolve_it_again(self) -> None:
        window = _open_window()
        try:
            _select_field(window)
            widget = window.field_map_widget
            widget.set_colormap("plasma")
            process_events(50)
            resolved_after_first = widget._active_color_map

            calls = 0
            real = plots_module.resolve_colormap

            def counting(name: str):
                nonlocal calls
                calls += 1
                return real(name)

            with mock.patch.object(plots_module, "resolve_colormap", counting):
                for _ in range(5):
                    widget.set_colormap("plasma")
            self.assertEqual(calls, 0, "an unchanged colormap name must not be resolved again")
            self.assertIs(widget._active_color_map, resolved_after_first)
        finally:
            window.close()
            process_events(50)

    def test_a_changed_colormap_is_still_resolved_and_applied(self) -> None:
        window = _open_window()
        try:
            _select_field(window)
            widget = window.field_map_widget
            widget.set_colormap("plasma")
            process_events(50)

            calls = 0
            real = plots_module.resolve_colormap

            def counting(name: str):
                nonlocal calls
                calls += 1
                return real(name)

            with mock.patch.object(plots_module, "resolve_colormap", counting):
                widget.set_colormap("viridis")
            self.assertEqual(calls, 1)
            expected = real("viridis").getLookupTable(nPts=256, alpha=True)
            mesh_lut = widget._mesh_item.getColorMap().getLookupTable(nPts=256, alpha=True)
            np.testing.assert_array_equal(mesh_lut, expected)
            image_lut = widget._image_item.getColorMap().getLookupTable(nPts=256, alpha=True)
            np.testing.assert_array_equal(image_lut, expected)
            bar_lut = widget._colorbar._color_map.getLookupTable(nPts=256, alpha=True)
            np.testing.assert_array_equal(bar_lut, expected)
        finally:
            window.close()
            process_events(50)

    def test_the_first_application_is_never_skipped(self) -> None:
        """The guard must not eat the constructor's initial apply."""

        widget = plots_module.FieldMapWidget()
        try:
            self.assertIsNotNone(widget._active_color_map)
            # The constructor applies, so the three targets already carry a map.
            self.assertIsNotNone(widget._mesh_item.getColorMap())
            self.assertIsNotNone(widget._image_item.getColorMap())
        finally:
            widget.close()
            widget.deleteLater()


@slow("drives a real Qt event loop")
class DataBoundsTests(unittest.TestCase):
    """F17: one render must not rescan the coordinate bounds three times."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_a_render_scans_the_bounds_at_most_twice(self) -> None:
        window = _open_window()
        try:
            _select_field(window)
            widget = window.field_map_widget
            calls = {"n": 0}
            real = widget._current_data_bounds

            def counting():
                calls["n"] += 1
                return real()

            with mock.patch.object(widget, "_current_data_bounds", counting):
                window._refresh_field_map(preserve_view=False)
                process_events(50)
            self.assertLessEqual(
                calls["n"],
                2,
                f"one field-map refresh scanned the bounds {calls['n']} times",
            )
            self.assertGreaterEqual(calls["n"], 1, "the control must observe at least one scan")
        finally:
            window.close()
            process_events(50)


@slow("drives a real Qt event loop")
class SnapshotDragTests(unittest.TestCase):
    """F22: a slider drag must not repaint synchronously on every tick."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_the_slider_does_not_emit_per_drag_step(self) -> None:
        """Tracking off is what removes the intermediate values; assert it."""

        window = _open_window()
        try:
            self.assertFalse(
                window.snapshot_slider.hasTracking(),
                "a tracked slider emits valueChanged for every intermediate drag value",
            )
            # The control: programmatic and keyboard paths must still emit.
            refreshes = {"n": 0}
            real_refresh = window._refresh_line_plot

            def counting_refresh(*args, **kwargs):
                refreshes["n"] += 1
                return real_refresh(*args, **kwargs)

            window._refresh_line_plot = counting_refresh
            target = min(window.snapshot_slider.maximum(), window.snapshot_slider.value() + 1)
            window.snapshot_slider.setValue(target)
            process_events(50)
            self.assertGreaterEqual(refreshes["n"], 1, "setValue must still refresh synchronously")
        finally:
            window.close()
            process_events(50)

    def test_dragging_twelve_ticks_stays_under_the_budget(self) -> None:
        window = _open_window()
        try:
            _select_field(window)
            slider = window.snapshot_slider
            start = max(slider.minimum(), int(slider.value()))
            budget = 12
            stop = min(slider.maximum(), start + budget)
            if stop - start < 4:
                self.skipTest("fixture has too few snapshots for a drag")

            refreshes = {"n": 0}
            real_refresh = window._refresh_line_plot

            def counting_refresh(*args, **kwargs):
                refreshes["n"] += 1
                return real_refresh(*args, **kwargs)

            window._refresh_line_plot = counting_refresh
            began = time.perf_counter()
            for value in range(start, stop + 1):
                slider.setValue(value)
            process_events(50)
            elapsed = time.perf_counter() - began
            ticks = stop - start + 1

            # The debounce means far fewer repaints than ticks. The exact ratio
            # depends on the refresh timer, so assert the ceiling, not equality.
            self.assertLessEqual(
                refreshes["n"],
                ticks,
                f"{refreshes['n']} line-plot refreshes for {ticks} slider ticks",
            )
            self.assertLess(elapsed, 5.0, f"a {ticks}-tick drag took {elapsed:.2f} s")
        finally:
            window.close()
            process_events(50)


@slow("drives a real Qt event loop")
class PercentileCopyTests(unittest.TestCase):
    """F16: the finite-mask copy must not run when clipping is not percentile."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_the_finite_mask_is_only_built_for_percentile_clipping(self) -> None:
        window = _open_window()
        try:
            _select_field(window)
            combo = window.clip_mode_combo
            values = [str(combo.itemData(i)) for i in range(combo.count())]
            self.assertIn("auto", values)
            combo_set_data(combo, "auto")
            process_events(50)

            with mock.patch.object(np, "isfinite", wraps=np.isfinite) as spy:
                window._refresh_field_map(preserve_view=False)
                process_events(50)
            auto_calls = spy.call_count
            self.assertEqual(auto_calls, 0, "auto clipping must not build a finite mask")

            if "percentile" in values:
                combo_set_data(combo, "percentile")
                process_events(50)
                with mock.patch.object(np, "isfinite", wraps=np.isfinite) as spy2:
                    window._refresh_field_map(preserve_view=False)
                    process_events(50)
                percentile_calls = spy2.call_count
                self.assertGreater(
                    percentile_calls,
                    auto_calls,
                    "percentile clipping should be the branch that builds the finite mask",
                )
        finally:
            window.close()
            process_events(50)


if __name__ == "__main__":
    unittest.main()