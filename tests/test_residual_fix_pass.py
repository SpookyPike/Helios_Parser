from __future__ import annotations

from dataclasses import replace
import importlib
import sys
import time
import unittest
from pathlib import Path

import numpy as np

import _test_bootstrap  # noqa: F401

from _viewer_test_utils import get_app, process_events, wait_until
from helios.cache import AnalyzerCacheSet, BoundedCacheBucket
from helios.cancellation import CancellationToken
from helios.runtime import RunContext
from helios.tasks import AnalysisTaskManager
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.common import load_run_data
from helios.services.derived.models import DerivedAnalysisResult, DerivedPlotBundle
from helios_analysis.controller import DerivedController
from helios_viewer.plots import CurvePlotWidget


HDF5_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "hdf5"


def _context_from_dataset(path: Path, dataset, *, snapshot_index: int | None = None) -> RunContext:
    return RunContext(
        path=path,
        summary=dict(dataset.summary),
        metadata=dict(dataset.metadata),
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64),
        static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32),
        zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32),
        has_dynamic_radius=dataset.radius_cm is not None,
        snapshot_index=min(20, max(0, dataset.time_s.size - 1)) if snapshot_index is None else int(snapshot_index),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="zone",
        selected_region_ids=tuple(int(value) for value in np.asarray(dataset.regions["region_index"], dtype=np.int32)),
        selected_material_ids=tuple(int(value) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


class ResidualFixPassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def test_bounded_cache_bucket_evicts_oldest_entry(self) -> None:
        bucket = BoundedCacheBucket(max_items=2)
        bucket["a"] = 1
        bucket["b"] = 2
        _ = bucket["a"]
        bucket["c"] = 3
        self.assertIn("a", bucket)
        self.assertIn("c", bucket)
        self.assertNotIn("b", bucket)

    def test_analyzer_cache_set_uses_bounded_buckets(self) -> None:
        caches = AnalyzerCacheSet()
        bucket = caches.derived_cache.bucket("derived")
        for index in range(40):
            bucket[index] = index
        self.assertLessEqual(len(bucket), 24)
        stats = caches.stats()["derived_cache"].buckets["derived"]
        self.assertGreater(stats.evictions, 0)
        self.assertEqual(stats.capacity, 24)

    def test_cache_observability_tracks_hits_misses_and_clear_reason(self) -> None:
        caches = AnalyzerCacheSet()
        bucket = caches.view_cache.bucket("display", max_items=2)
        bucket["density"] = 1
        self.assertEqual(bucket.get("density"), 1)
        self.assertIsNone(bucket.get("missing"))
        stats = bucket.stats()
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)
        caches.view_cache.clear_bucket("display", reason="test-reset")
        self.assertEqual(caches.view_cache.stats().last_clear_reason, "test-reset")

    def test_run_context_copy_shares_immutable_arrays_read_only(self) -> None:
        context = RunContext(
            path=Path("demo.h5"),
            summary={"n_snapshots": 3, "n_zones": 4},
            metadata={},
            fields=("density",),
            diagnostics=(),
            time_values=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            static_x_values=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            zone_region_id=np.asarray([1, 1, 2, 2], dtype=np.int32),
            zone_material_index=np.asarray([1, 1, 1, 1], dtype=np.int32),
        )
        cloned = context.copy()
        self.assertTrue(np.shares_memory(context.time_values, cloned.time_values))
        self.assertTrue(np.shares_memory(context.static_x_values, cloned.static_x_values))
        self.assertFalse(cloned.time_values.flags.writeable)
        self.assertFalse(cloned.static_x_values.flags.writeable)

    def test_backend_import_of_helios_does_not_pull_qt_tasks(self) -> None:
        sys.modules.pop("helios", None)
        sys.modules.pop("helios.tasks", None)
        module = importlib.import_module("helios")
        self.assertNotIn("helios.tasks", sys.modules)
        _ = module.RunContext
        self.assertNotIn("helios.tasks", sys.modules)
        _ = module.AnalysisTaskManager
        self.assertIn("helios.tasks", sys.modules)

    def test_task_manager_cooperative_cancellation_stops_running_work(self) -> None:
        manager = AnalysisTaskManager()
        cancelled = {"value": False}
        finished = {"value": False}
        failed = {"value": False}
        token = CancellationToken()

        manager.task_cancelled.connect(lambda task_id: cancelled.__setitem__("value", True))
        manager.result_ready.connect(lambda result: finished.__setitem__("value", True))
        manager.task_failed.connect(lambda task_id, message: failed.__setitem__("value", True))

        def _work() -> int:
            for _ in range(1000):
                token.check_cancelled()
                time.sleep(0.001)
            return 1

        handle = manager.submit(context_key=("cancel",), fn=_work, cancellation_token=token)
        process_events(20)
        manager.cancel(handle.task_id)
        wait_until(lambda: cancelled["value"], timeout_s=5.0)
        process_events(20)
        self.assertTrue(cancelled["value"])
        self.assertFalse(finished["value"])
        self.assertFalse(failed["value"])
        manager.shutdown()

    def test_task_manager_uses_single_persistent_pool_configuration(self) -> None:
        manager = AnalysisTaskManager()
        completed: list[int] = []
        for value in (1, 2):
            manager.submit(context_key=("pool", value), fn=lambda value=value: value)
        wait_until(lambda: manager.stats().completed == 2, timeout_s=5.0)
        self.assertEqual(manager.stats().max_thread_count, 1)
        manager.shutdown()

    def test_curve_plot_widget_reuses_linear_arrays_without_copying(self) -> None:
        widget = CurvePlotWidget()
        try:
            x = np.linspace(0.0, 1.0, 8, dtype=np.float64)
            y = np.linspace(1.0, 2.0, 8, dtype=np.float64)
            widget.set_curves(
                x,
                [y],
                title="demo",
                x_label="x",
                y_label="y",
                preserve_view=False,
            )
            self.assertTrue(np.shares_memory(widget.last_x_values, x))
            self.assertTrue(np.shares_memory(widget.last_raw_y_series[0], y))
            self.assertTrue(np.shares_memory(widget.last_y_series[0], y))
        finally:
            widget.close()

    def test_derived_controller_rejects_invalid_result_before_ui_apply(self) -> None:
        path = HDF5_ROOT / "5Fe+4.9TW+light_stabilized.h5"
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset)
        result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=("residual", "valid"),
        )
        invalid_bundle = DerivedPlotBundle(
            key="invalid",
            title="invalid",
            x_label="x",
            y_label="y",
            x_values=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            y_series=(np.asarray([0.0, 1.0], dtype=np.float64),),
        )
        invalid_result: DerivedAnalysisResult = replace(
            result,
            xrd=replace(result.xrd, profile_plots=(invalid_bundle,)),
        )

        controller = DerivedController()
        workspace = controller.widget()
        apply_calls = {"count": 0}
        original_set_result = workspace.set_result
        try:
            controller._workspace_alive = True
            controller._active = True
            controller._context = context.copy()
            request_key = ("derived-analysis-core", "demo", context.snapshot_index)
            controller._active_request_key = request_key
            controller._active_task_id = "task-1"
            workspace.set_result(result)
            prior_summary = workspace.xrd_summary_label.text()
            prior_result = workspace._current_result

            def _set_result_guard(*args, **kwargs):
                apply_calls["count"] += 1
                return original_set_result(*args, **kwargs)

            workspace.set_result = _set_result_guard  # type: ignore[method-assign]
            controller._on_task_result(type("Result", (), {"task_id": "task-1", "context_key": request_key, "result": invalid_result})())
            self.assertEqual(apply_calls["count"], 0)
            self.assertIn("failed", workspace.result_status_label.text().lower())
            self.assertIs(workspace._current_result, prior_result)
            self.assertEqual(workspace.xrd_summary_label.text(), prior_summary)
        finally:
            workspace.set_result = original_set_result  # type: ignore[method-assign]
            controller.shutdown()
            workspace.close()


if __name__ == "__main__":
    unittest.main()
