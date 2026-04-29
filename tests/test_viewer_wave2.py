from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import HDF5_ROOT, get_app, process_events, reset_test_settings, wait_until
from helios_parser import HeliosRun
from helios_viewer.controller import RunController
from helios_viewer.main_window import HeliosViewerMainWindow, _REFRESH_FIELD_MAP, _REFRESH_LINE
from helios_viewer.models import DiagnosticPayload, FieldPayload, FieldTracePayload, OpenRunPayload, SnapshotFieldPayload


def _build_open_run_payload(path: Path, generation: int) -> OpenRunPayload:
    with HeliosRun(path) as run:
        fields = run.list_fields()
        diagnostics = run.list_diagnostics()
        return OpenRunPayload(
            run_generation=int(generation),
            path=path,
            summary=run.summary(),
            metadata=run.get_metadata(),
            fields=fields,
            field_units={name: run.get_field_unit(name) for name in fields},
            diagnostics=diagnostics,
            diagnostic_units={name: run.get_diagnostic_unit(name) for name in diagnostics},
            regions=run.get_regions(),
            materials=run.get_materials(),
            time=np.asarray(run.get_time(), dtype=np.float64),
            time_unit=run.get_time_unit(),
            static_x=np.asarray(run.get_static_coordinate(location="center"), dtype=np.float64),
            static_x_edges=np.asarray(run.get_static_coordinate(location="edge"), dtype=np.float64),
            static_x_unit=run.get_grid_unit("x"),
            zone_region_id=np.asarray(run.get_grid("zone_region_id"), dtype=np.int32),
            zone_material_index=np.asarray(run.get_grid("zone_material_index"), dtype=np.int32),
            has_dynamic_radius="radius" in fields,
            radius_unit=run.get_field_unit("radius") if "radius" in fields else "",
        )


def _build_field_payload(path: Path, field_name: str, generation: int) -> FieldPayload:
    with HeliosRun(path) as run:
        edge_data = None
        if field_name == "radius":
            data = np.asarray(run.get_dynamic_coordinate(location="center"), dtype=np.float64)
            dynamic_edge = run.get_dynamic_coordinate(location="edge")
            edge_data = None if dynamic_edge is None else np.asarray(dynamic_edge, dtype=np.float64)
        else:
            data = np.asarray(run.get_field(field_name), dtype=np.float64)
        return FieldPayload(
            run_generation=int(generation),
            field_name=field_name,
            unit=run.get_field_unit(field_name),
            data=data,
            edge_data=edge_data,
        )


def _build_diagnostic_payload(path: Path, diagnostic_path: str, generation: int) -> DiagnosticPayload:
    with HeliosRun(path) as run:
        return DiagnosticPayload(
            run_generation=int(generation),
            path=diagnostic_path,
            unit=run.get_diagnostic_unit(diagnostic_path),
            data=np.asarray(run.get_diagnostic(diagnostic_path)),
        )


class ViewerWave2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def setUp(self) -> None:
        reset_test_settings()

    def test_run_controller_bounded_caches_expose_stats(self) -> None:
        controller = RunController()
        try:
            for index in range(18):
                controller.field_cache[f"field_{index}"] = FieldPayload(
                    run_generation=0,
                    field_name=f"field_{index}",
                    unit="",
                    data=np.asarray([float(index)], dtype=np.float64),
                )
            for index in range(17):
                controller.diagnostic_cache[f"diag_{index}"] = DiagnosticPayload(
                    run_generation=0,
                    path=f"diag_{index}",
                    unit="",
                    data=np.asarray([float(index)], dtype=np.float64),
                )

            self.assertIsNotNone(controller.field_cache.get("field_17"))
            self.assertIsNone(controller.field_cache.get("missing-field"))
            self.assertIsNotNone(controller.diagnostic_cache.get("diag_16"))
            self.assertIsNone(controller.diagnostic_cache.get("missing-diagnostic"))

            stats = controller.cache_stats()
            self.assertLessEqual(stats["field_cache"].size, stats["field_cache"].capacity)
            self.assertLessEqual(stats["snapshot_field_cache"].size, stats["snapshot_field_cache"].capacity)
            self.assertLessEqual(stats["field_trace_cache"].size, stats["field_trace_cache"].capacity)
            self.assertLessEqual(stats["diagnostic_cache"].size, stats["diagnostic_cache"].capacity)
            self.assertGreater(stats["field_cache"].evictions, 0)
            self.assertGreater(stats["diagnostic_cache"].evictions, 0)
            self.assertGreater(stats["field_cache"].hits, 0)
            self.assertGreater(stats["field_cache"].misses, 0)
            self.assertGreater(stats["diagnostic_cache"].hits, 0)
            self.assertGreater(stats["diagnostic_cache"].misses, 0)
        finally:
            controller.shutdown()

    def test_snapshot_and_trace_payloads_are_cached_without_full_field_payload(self) -> None:
        controller = RunController()
        try:
            snapshot_payloads: list[SnapshotFieldPayload] = []
            trace_payloads: list[FieldTracePayload] = []
            controller.snapshot_field_loaded.connect(snapshot_payloads.append)
            controller.field_trace_loaded.connect(trace_payloads.append)
            controller._run_generation = 3

            snapshot_payload = SnapshotFieldPayload(
                run_generation=3,
                field_name="density",
                snapshot_index=4,
                unit="g/cm3",
                data=np.asarray([1.0, 2.0], dtype=np.float64),
            )
            trace_payload = FieldTracePayload(
                run_generation=3,
                field_name="density",
                zone_index=1,
                unit="g/cm3",
                data=np.asarray([1.0, 1.5, 2.0], dtype=np.float64),
            )
            controller._handle_snapshot_field_loaded(snapshot_payload)
            controller._handle_field_trace_loaded(trace_payload)

            self.assertEqual(snapshot_payloads, [snapshot_payload])
            self.assertEqual(trace_payloads, [trace_payload])
            self.assertIs(controller.snapshot_field_cache[("density", 4)], snapshot_payload)
            self.assertIs(controller.field_trace_cache[("density", 1)], trace_payload)
            self.assertIsNone(controller.field_cache.get("density"))
        finally:
            controller.shutdown()

    def test_stale_controller_results_are_discarded_after_run_switch(self) -> None:
        controller = RunController()
        try:
            controller.worker_thread.quit()
            controller.worker_thread.wait(2000)

            run_a = HDF5_ROOT / "Cu_0166_stabilized.h5"
            run_b = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
            opened_paths: list[Path] = []
            field_names: list[str] = []
            diagnostic_paths: list[str] = []
            status_messages: list[str] = []
            controller.run_opened.connect(lambda payload: opened_paths.append(payload.path))
            controller.field_loaded.connect(lambda payload: field_names.append(payload.field_name))
            controller.diagnostic_loaded.connect(lambda payload: diagnostic_paths.append(payload.path))
            controller.status_changed.connect(status_messages.append)

            controller.open_file(run_a)
            generation_a = controller.run_generation
            controller.open_file(run_b)
            generation_b = controller.run_generation
            self.assertGreater(generation_b, generation_a)

            controller._handle_run_opened(_build_open_run_payload(run_a, generation_a))
            controller._handle_field_loaded(_build_field_payload(run_a, "density", generation_a))
            controller._handle_diagnostic_loaded(
                _build_diagnostic_payload(run_a, "energy_summary/current/ions", generation_a)
            )

            controller._handle_run_opened(_build_open_run_payload(run_b, generation_b))
            controller._handle_field_loaded(_build_field_payload(run_b, "density", generation_b))
            controller._handle_diagnostic_loaded(
                _build_diagnostic_payload(run_b, "energy_summary/current/ions", generation_b)
            )

            self.assertEqual(opened_paths, [run_b])
            self.assertEqual(field_names, ["density"])
            self.assertEqual(diagnostic_paths, ["energy_summary/current/ions"])
            self.assertEqual(controller.run_payload.path, run_b)
            self.assertEqual(controller.field_cache["density"].run_generation, generation_b)
            self.assertEqual(controller.diagnostic_cache["energy_summary/current/ions"].run_generation, generation_b)
            self.assertTrue(any("Discarded stale" in message for message in status_messages))
        finally:
            controller.shutdown()

    def test_main_window_ignores_stale_field_payloads_at_ui_boundary(self) -> None:
        window = HeliosViewerMainWindow()
        try:
            run_path = HDF5_ROOT / "Cu_0166_stabilized.h5"
            stale_generation = 1
            active_generation = 2
            window.controller.worker_thread.quit()
            window.controller.worker_thread.wait(2000)
            window.controller._run_generation = active_generation
            window._on_run_opened(_build_open_run_payload(run_path, active_generation))
            window.current_field_name = "density"

            stale_payload = _build_field_payload(run_path, "density", stale_generation)
            active_payload = _build_field_payload(run_path, "density", active_generation)

            render_before = window.field_map_widget.render_call_count
            window._on_field_loaded(stale_payload)
            process_events(20)
            self.assertIsNone(window.current_field_payload)
            self.assertEqual(window.field_map_widget.render_call_count, render_before)

            window._on_field_loaded(active_payload)
            wait_until(lambda: window.current_field_payload is not None, timeout_s=5.0)
            self.assertEqual(window.current_field_payload.run_generation, active_generation)
            self.assertGreater(window.field_map_widget.render_call_count, render_before)
        finally:
            window.close()

    def test_refresh_batching_still_coalesces_rapid_updates(self) -> None:
        window = HeliosViewerMainWindow()
        try:
            window.load_file(HDF5_ROOT / "Cu_0166_stabilized.h5")
            wait_until(
                lambda: window.run_payload is not None and window.current_field_payload is not None,
                timeout_s=30.0,
            )
            field_renders_before = window.field_map_widget.render_call_count
            line_renders_before = window.lineout_plot.set_curves_call_count

            for _ in range(8):
                window._schedule_refresh(_REFRESH_FIELD_MAP | _REFRESH_LINE, preserve_view=True)
            wait_until(
                lambda: window.field_map_widget.render_call_count > field_renders_before,
                timeout_s=5.0,
            )
            process_events(120)

            self.assertLessEqual(window.field_map_widget.render_call_count - field_renders_before, 2)
            self.assertLessEqual(window.lineout_plot.set_curves_call_count - line_renders_before, 2)
            self.assertEqual(window.current_field_payload.field_name, "density")
        finally:
            window.controller.shutdown()
            window.close()


if __name__ == "__main__":
    unittest.main()
