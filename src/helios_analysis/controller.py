"""Controller for HELIOS Derived / Analysis mode.

This layer connects the embeddable workspace to the derived-service backend. It
uses the shared `RunContext`, task model, and cache layers prepared in earlier
phases so that analysis updates remain non-blocking and stale results can be
discarded safely.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import logging
import time

import numpy as np
from PySide6 import QtCore

from helios.cache import AnalyzerCacheSet
from helios.cancellation import CancellationToken
from helios.instrumentation import increment_counter, snapshot_metrics
from helios.runtime import RunContext
from helios.services.derived import (
    DerivedAnalysisParameters,
    analysis_result_time_plot_modules,
    compute_analysis_result,
    load_run_data,
    normalize_time_plot_modules,
    refresh_analysis_result_for_snapshot,
)
from helios.services.derived.common import aggregate_warnings
from helios.services.derived.models import DerivedAnalysisResult, DerivedPlotBundle
from helios.services.derived.selection import build_analysis_geometry
from helios.services.derived.transmission import apply_transmission_model, normalize_transmission_mode
from helios.tasks import AnalysisTaskManager, AnalysisTaskResult
from helios_viewer.style import resolve_theme

from .workspace import HeliosDerivedWorkspace


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingAnalysisRequest:
    context: RunContext
    parameters: DerivedAnalysisParameters
    reuse_request_key: tuple[object, ...]
    core_request_key: tuple[object, ...]
    request_key: tuple[object, ...]
    requested_time_plot_modules: frozenset[str]
    include_wavefront: bool
    update_kind: str
    generation: int
    base_request_key: tuple[object, ...] | None = None
    transmission_model_energy_kev: float | None = None
    transmission_model_mode: str | None = None


@dataclass(slots=True)
class _ComputedAnalysisPayload:
    result: object
    core_request_key: tuple[object, ...]
    update_kind: str


class DerivedController(QtCore.QObject):
    """Own derived-analysis state, background work, and result caching."""

    status_changed = QtCore.Signal(str)
    busy_changed = QtCore.Signal(bool)
    analysis_ready = QtCore.Signal(object)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace = HeliosDerivedWorkspace()
        self._workspace.destroyed.connect(self._on_workspace_destroyed)
        self._workspace_alive = True
        self._context = RunContext.empty()
        self._cache_layers = AnalyzerCacheSet()
        self._dataset_cache = self._cache_layers.raw_data_cache.bucket("derived_run_data", max_items=3)
        self._analysis_cache = self._cache_layers.derived_cache.bucket("analysis_results", max_items=24)
        self._analysis_core_cache = self._cache_layers.derived_cache.bucket("analysis_core_results", max_items=12)
        self._tasks = AnalysisTaskManager(self)
        self._tasks.result_ready.connect(self._on_task_result)
        self._tasks.task_failed.connect(self._on_task_failed)
        self._tasks.task_cancelled.connect(self._on_task_cancelled)
        self._workspace.parameters_changed.connect(self._schedule_recompute)
        self._workspace.refresh_requested.connect(self._schedule_recompute)
        self._workspace.time_plot_modules_changed.connect(self._schedule_recompute)
        self._workspace.transmission_refine_requested.connect(self._schedule_transmission_refine)
        self._workspace.cancel_requested.connect(self.cancel_active_request)
        self._recompute_timer = QtCore.QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.timeout.connect(self._start_recompute)
        self._recompute_debounce_ms = 90
        self._active_request_key: tuple[object, ...] | None = None
        self._active_task_id: str | None = None
        self._active_task_request_key: tuple[object, ...] | None = None
        self._active_task_token: CancellationToken | None = None
        self._active_reuse_request_key: tuple[object, ...] | None = None
        self._displayed_result_key: tuple[object, ...] | None = None
        self._displayed_reuse_request_key: tuple[object, ...] | None = None
        self._displayed_result: DerivedAnalysisResult | None = None
        self._current_generation = 0
        self._active_generation = 0
        self._pending_request: _PendingAnalysisRequest | None = None
        self._active = False
        self._shutting_down = False
        self._busy = False
        self._last_completed_update_kind = "idle"
        self._last_started_update_kind = "idle"

    def widget(self) -> HeliosDerivedWorkspace:
        return self._workspace

    @QtCore.Slot()
    def _on_workspace_destroyed(self) -> None:
        self._workspace_alive = False
        self._active_task_id = None
        self._active_task_request_key = None
        self._active_task_token = None
        self._active_reuse_request_key = None

    def set_theme_mode(self, mode: str) -> None:
        if self._workspace_alive:
            self._workspace.apply_theme(resolve_theme(mode))

    def set_display_settings(self, settings: object) -> None:
        if self._workspace_alive:
            self._workspace.set_display_settings(settings)

    def set_default_profile_coordinate_mode(self, mode: str) -> None:
        if self._workspace_alive:
            self._workspace.set_default_profile_coordinate_mode(mode)
        if self._active and self._context.has_run and not self._shutting_down:
            self._schedule_recompute()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            self._recompute_timer.stop()
            self._tasks.cancel(self._active_task_id)
            if self._workspace_alive and self._context.has_run:
                self._workspace.clear_results("Derived results will update when Derived / Analysis is active.")
            self._set_busy(False, "Derived mode synchronized to the active run.")
            return
        if self._context.has_run:
            self._schedule_recompute()

    def set_run_context(self, context: RunContext) -> None:
        self._context = context.copy()
        if self._workspace_alive:
            self._workspace.set_context(self._context)
        if not self._context.has_run:
            self._tasks.cancel(self._active_task_id)
            self._active_request_key = None
            self._active_task_id = None
            self._active_task_request_key = None
            self._active_task_token = None
            self._active_reuse_request_key = None
            self._displayed_result_key = None
            self._displayed_reuse_request_key = None
            self._displayed_result = None
            self._pending_request = None
            self._set_busy(False, "Derived mode ready.")
            if self._workspace_alive:
                self._workspace.clear_results("Load a HELIOS run in Viewer Mode, then switch to Derived / Analysis.")
            return
        if not self._active:
            if self._workspace_alive:
                self._workspace.clear_results("Derived results will update when Derived / Analysis is active.")
            self._set_busy(False, "Derived mode synchronized to the active run.")
            return
        self._schedule_recompute()

    def current_run_context(self) -> RunContext:
        return self._context

    def cache_stats(self) -> dict[str, object]:
        return self._cache_layers.stats()

    def shutdown(self) -> None:
        self._shutting_down = True
        self._recompute_timer.stop()
        self._tasks.cancel(self._active_task_id)
        self._active_task_id = None
        self._active_task_request_key = None
        self._active_task_token = None
        self._active_reuse_request_key = None
        self._pending_request = None
        self._tasks.shutdown()
        if self._workspace_alive:
            self._workspace.close()

    @QtCore.Slot()
    def cancel_active_request(self) -> None:
        if self._active_task_id is None:
            return
        self._pending_request = None
        self._tasks.cancel(self._active_task_id)
        self._set_busy(False, "Cancelled the active derived-analysis task.")

    def _schedule_recompute(self) -> None:
        if not self._context.has_run or not self._active or self._shutting_down:
            return
        self._recompute_timer.start(self._recompute_debounce_ms)

    def _build_request_key(self, parameters: DerivedAnalysisParameters) -> tuple[object, ...]:
        requested_time_plot_modules = normalize_time_plot_modules(self._workspace.requested_time_plot_modules())
        return (
            *self._build_core_request_key(parameters),
            parameters.preheat_target_region_id,
            int(self._context.snapshot_index),
            tuple(sorted(requested_time_plot_modules)),
            bool(self._workspace.advanced_requested()),
        )

    def _build_reuse_request_key(
        self,
        parameters: DerivedAnalysisParameters,
        *,
        context: RunContext | None = None,
    ) -> tuple[object, ...]:
        active_context = self._context if context is None else context
        return (
            "derived-analysis-reuse",
            *active_context.context_key,
            str(active_context.map_coordinate),
            str(active_context.slice_coordinate),
            tuple(int(value) for value in active_context.selected_region_ids),
            tuple(int(value) for value in active_context.selected_material_ids),
            *parameters.core_key(),
        )

    def _build_core_request_key(self, parameters: DerivedAnalysisParameters) -> tuple[object, ...]:
        return (
            "derived-analysis-core",
            *self._build_reuse_request_key(parameters),
            int(self._context.snapshot_index),
        )

    def _set_busy(self, busy: bool, message: str) -> None:
        self._busy = bool(busy)
        if self._workspace_alive:
            self._workspace.set_busy(self._busy, message)
        self.busy_changed.emit(self._busy)
        self.status_changed.emit(message)

    @staticmethod
    def _result_has_wavefront(result: DerivedAnalysisResult | None) -> bool:
        return bool(result is not None and result.wave_tracking is not None and result.interface_events is not None)

    @staticmethod
    def _result_has_preheat(result: DerivedAnalysisResult | None) -> bool:
        return bool(result is not None and result.wave_tracking is not None and result.interface_events is not None and result.preheat is not None)

    @staticmethod
    def _preheat_target_matches(result: DerivedAnalysisResult | None, parameters: DerivedAnalysisParameters) -> bool:
        if result is None:
            return False
        requested_region_id = None if parameters.preheat_target_region_id is None else int(parameters.preheat_target_region_id)
        if result.preheat is None:
            return False
        existing_region_id = None
        if str(result.preheat.target_selection_mode or "auto") == "user_selected":
            existing_region_id = None if result.preheat.target_region_id is None else int(result.preheat.target_region_id)
        return existing_region_id == requested_region_id

    @staticmethod
    def _merge_cached_features(previous: DerivedAnalysisResult | None, current: DerivedAnalysisResult) -> DerivedAnalysisResult:
        if previous is None:
            return current
        wave_tracking = current.wave_tracking if current.wave_tracking is not None else previous.wave_tracking
        interface_events = current.interface_events if current.interface_events is not None else previous.interface_events
        preheat = current.preheat if current.preheat is not None else previous.preheat
        if wave_tracking is current.wave_tracking and interface_events is current.interface_events and preheat is current.preheat:
            return current
        return replace(
            current,
            wave_tracking=wave_tracking,
            interface_events=interface_events,
            preheat=preheat,
        )

    @staticmethod
    def _format_duration_ms(value_s: float | None) -> str:
        if value_s is None:
            return "-"
        return f"{float(value_s) * 1.0e3:.1f} ms"

    def _set_performance_summary(self, *, update_kind: str, from_cache: bool) -> None:
        if not self._workspace_alive:
            return
        timers = snapshot_metrics().get("timers", {})
        derived_full = timers.get("derived.compute.full")
        shock = timers.get("derived.compute.shock")
        wavefront = timers.get("derived.compute.wavefront")
        wave_tracking = timers.get("derived.compute.wave_tracking")
        preheat = timers.get("derived.compute.preheat")
        transmission_xcom = timers.get("derived.compute.transmission_xcom")
        summary_parts = [f"Source: {'cache' if from_cache else 'background compute'}"]
        if update_kind == "wavefront":
            if wavefront is not None:
                summary_parts.append(f"WaveFront {self._format_duration_ms(wavefront.last_s)}")
            if wave_tracking is not None:
                summary_parts.append(f"tracker {self._format_duration_ms(wave_tracking.last_s)}")
            if preheat is not None:
                summary_parts.append(f"preheat {self._format_duration_ms(preheat.last_s)}")
            if shock is not None:
                summary_parts.append(f"legacy Shock {self._format_duration_ms(shock.last_s)}")
        elif update_kind == "preheat":
            if preheat is not None:
                summary_parts.append(f"Preheat {self._format_duration_ms(preheat.last_s)}")
            if wave_tracking is not None:
                summary_parts.append(f"tracker {self._format_duration_ms(wave_tracking.last_s)}")
            if shock is not None:
                summary_parts.append(f"legacy Shock {self._format_duration_ms(shock.last_s)}")
        elif update_kind == "transmission_model":
            if transmission_xcom is not None:
                summary_parts.append(f"Transmission model {self._format_duration_ms(transmission_xcom.last_s)}")
            if shock is not None:
                summary_parts.append(f"legacy Shock {self._format_duration_ms(shock.last_s)}")
        elif update_kind == "snapshot":
            refresh = timers.get("derived.compute.snapshot_refresh")
            if refresh is not None:
                summary_parts.append(f"snapshot refresh {self._format_duration_ms(refresh.last_s)}")
        else:
            if derived_full is not None:
                summary_parts.append(f"legacy refresh {self._format_duration_ms(derived_full.last_s)}")
            if shock is not None:
                summary_parts.append(f"legacy Shock {self._format_duration_ms(shock.last_s)}")
        summary = " | ".join(summary_parts)
        LOGGER.info("Derived performance summary: %s", summary)
        self._workspace.set_performance_summary(
            summary,
            wavefront=(update_kind == "wavefront"),
            preheat=(update_kind in {"wavefront", "preheat"}),
        )

    def _show_failure_state(self, message: str) -> None:
        if self._workspace_alive and getattr(self._workspace, "_current_result", None) is None:
            self._workspace.clear_results(message)
        self._set_busy(False, message)

    def _next_generation(self) -> int:
        self._current_generation += 1
        return self._current_generation

    @staticmethod
    def _bundle_is_valid(bundle: DerivedPlotBundle) -> bool:
        x_values = np.asarray(bundle.x_values, dtype=np.float64)
        if x_values.ndim != 1:
            return False
        for series in bundle.y_series:
            y_values = np.asarray(series, dtype=np.float64)
            if y_values.ndim != 1 or y_values.shape[0] != x_values.shape[0]:
                return False
        return True

    def _validate_result(self, result: object, *, request_key: tuple[object, ...]) -> DerivedAnalysisResult:
        if not isinstance(result, DerivedAnalysisResult):
            raise TypeError("Derived analysis produced an unexpected result object.")
        expected_snapshot = int(result.snapshot_index)
        if len(request_key) >= 3:
            if len(request_key) >= 7 and request_key[-3] == "transmission_model":
                expected_snapshot = int(request_key[-6])
            else:
                expected_snapshot = int(request_key[-3])
        if int(result.snapshot_index) != expected_snapshot:
            raise ValueError(
                f"Derived analysis returned snapshot {int(result.snapshot_index)} for request snapshot {expected_snapshot}."
            )
        for module_result in (result.xrd, result.plasmon, result.transmission, result.spectroscopy):
            for bundle in (*module_result.time_plots, *module_result.profile_plots):
                if not self._bundle_is_valid(bundle):
                    raise ValueError(f"Derived plot bundle {bundle.key!r} is structurally invalid.")
        return result

    def _apply_result(self, result: DerivedAnalysisResult, *, request_key: tuple[object, ...]) -> None:
        if not self._workspace_alive or self._shutting_down:
            LOGGER.debug("Skipping derived result apply because the workspace is no longer active.")
            return
        active_request_key = self._active_request_key
        if active_request_key is None or request_key != active_request_key:
            LOGGER.debug("Skipping derived result apply for stale request %s", request_key)
            return
        self._displayed_result_key = request_key
        self._displayed_reuse_request_key = self._active_reuse_request_key
        self._displayed_result = result
        self._workspace.set_result(result)
        self.analysis_ready.emit(result)

    def _requested_time_plot_modules(self) -> frozenset[str]:
        if not self._workspace_alive:
            return frozenset()
        return normalize_time_plot_modules(self._workspace.requested_time_plot_modules())

    def _compatible_displayed_result(self, reuse_request_key: tuple[object, ...]) -> DerivedAnalysisResult | None:
        if self._displayed_result is None or self._displayed_reuse_request_key != reuse_request_key:
            return None
        if self._context.path is None or self._displayed_result.dataset_path != Path(self._context.path):
            return None
        return self._displayed_result

    def _schedule_transmission_refine(self) -> None:
        if not self._context.has_run or self._context.path is None or not self._workspace_alive or self._shutting_down or not self._active:
            return
        parameters = self._workspace.parameters()
        reuse_request_key = self._build_reuse_request_key(parameters, context=self._context)
        request_key_seed = self._build_request_key(parameters)
        base_request_key = request_key_seed
        mode = normalize_transmission_mode(parameters.transmission_mode)
        energy_kev = float(parameters.transmission_photon_energy_kev)
        request_key = (*request_key_seed, "transmission_model", mode, round(energy_kev, 12))
        request = _PendingAnalysisRequest(
            context=self._context.copy(),
            parameters=parameters,
            reuse_request_key=reuse_request_key,
            core_request_key=self._build_core_request_key(parameters),
            request_key=request_key,
            requested_time_plot_modules=self._requested_time_plot_modules(),
            include_wavefront=False,
            update_kind="transmission_model",
            generation=self._next_generation(),
            base_request_key=base_request_key,
            transmission_model_energy_kev=energy_kev,
            transmission_model_mode=mode,
        )
        cached = self._analysis_cache.get(request_key)
        if cached is not None and self._active_task_id is None:
            try:
                self._active_request_key = request_key
                self._active_reuse_request_key = reuse_request_key
                cached_result = self._validate_result(cached, request_key=request_key)
                self._apply_result(cached_result, request_key=request_key)
            except Exception as exc:  # pragma: no cover - defensive UI path
                LOGGER.exception("Failed to apply cached transmission-model result.")
                self._show_failure_state(f"Derived analysis failed: {exc}")
                return
            increment_counter("derived.cache.transmission_model.hit")
            self._set_performance_summary(update_kind="transmission_model", from_cache=True)
            self._set_busy(False, "Loaded cached Transmission model result.")
            return
        if self._active_task_id is not None:
            self._pending_request = request
            self._active_request_key = request_key
            self._tasks.cancel(self._active_task_id)
            self._set_busy(True, "Queued latest Transmission model update...")
            return
        self._launch_request(request)

    def _launch_request(self, request: _PendingAnalysisRequest) -> None:
        context_snapshot = request.context.copy()
        request_key = request.request_key
        displayed_base_result = self._compatible_displayed_result(request.reuse_request_key)
        self._active_request_key = request_key
        self._active_reuse_request_key = request.reuse_request_key
        self._active_generation = int(request.generation)
        self._pending_request = None
        self._last_started_update_kind = str(request.update_kind)
        if request.update_kind == "snapshot":
            snapshot_time = (
                float(context_snapshot.time_values[context_snapshot.snapshot_index])
                if context_snapshot.time_values.size and 0 <= context_snapshot.snapshot_index < context_snapshot.time_values.size
                else float("nan")
            )
            self._set_busy(
                True,
                f"Updating snapshot {context_snapshot.snapshot_index} @ {snapshot_time * 1.0e9:.6g} ns...",
            )
        elif request.update_kind == "time_plots":
            self._set_busy(True, "Loading derived time traces for the active module...")
        elif request.update_kind == "preheat":
            self._set_busy(True, "Evaluating target preheat diagnostics...")
        elif request.update_kind == "wavefront":
            self._set_busy(True, "Tracking WaveFront branches and interface events...")
        elif request.update_kind == "transmission_model":
            energy_text = "-" if request.transmission_model_energy_kev is None else f"{float(request.transmission_model_energy_kev):.4g} keV"
            mode_text = str(request.transmission_model_mode or "transmission").replace("_", " ")
            self._set_busy(True, f"Applying Transmission {mode_text} model at {energy_text}...")
        else:
            self._set_busy(True, "Computing derived metrics...")

        def _compute() -> object:
            path = Path(context_snapshot.path) if context_snapshot.path is not None else None
            if path is None:
                raise RuntimeError("No active run is available for derived analysis.")
            dataset = self._dataset_cache.get(path)
            if dataset is None:
                started = time.perf_counter()
                dataset = load_run_data(path)
                self._dataset_cache[path] = dataset
                LOGGER.info("Loaded derived dataset %s in %.3fs", path.name, time.perf_counter() - started)
            if request.update_kind == "snapshot":
                base_result = displayed_base_result
                if base_result is None:
                    base_result = self._analysis_core_cache.get(request.core_request_key)
                if base_result is not None:
                    return _ComputedAnalysisPayload(
                        result=refresh_analysis_result_for_snapshot(
                            dataset,
                            context_snapshot,
                            parameters=request.parameters,
                            context_key=request_key,
                            base_result=base_result,
                            progress_check=cancellation_token.check_cancelled,
                        ),
                        core_request_key=request.core_request_key,
                        update_kind="snapshot",
                    )
            if request.update_kind == "transmission_model":
                base_result = self._analysis_cache.get(request.base_request_key) if request.base_request_key is not None else None
                if (
                    base_result is None
                    and displayed_base_result is not None
                    and int(displayed_base_result.snapshot_index) == int(context_snapshot.snapshot_index)
                ):
                    base_result = displayed_base_result
                if base_result is None:
                    base_result = self._analysis_core_cache.get(request.core_request_key)
                if base_result is None or not request.requested_time_plot_modules.issubset(analysis_result_time_plot_modules(base_result)):
                    base_result = compute_analysis_result(
                        dataset,
                        context_snapshot,
                        parameters=request.parameters,
                        context_key=request.request_key,
                        requested_time_plot_modules=request.requested_time_plot_modules,
                        include_wavefront=False,
                        progress_check=cancellation_token.check_cancelled,
                    )
                current_geometry = build_analysis_geometry(
                    dataset,
                    context_snapshot,
                    observation_side=request.parameters.observation_side,
                    line_of_sight_angle_deg=request.parameters.line_of_sight_angle_deg,
                    line_of_sight_impact_parameter_cm=request.parameters.line_of_sight_impact_parameter_cm,
                    profile_coordinate_mode=request.parameters.profile_coordinate_mode,
                )
                updated_transmission = apply_transmission_model(
                    base_result.transmission,
                    dataset,
                    context_snapshot,
                    snapshot_index=int(context_snapshot.snapshot_index),
                    parameters=request.parameters,
                    geometry=current_geometry,
                    progress_check=cancellation_token.check_cancelled,
                )
                updated_result = replace(
                    base_result,
                    snapshot_index=int(context_snapshot.snapshot_index),
                    snapshot_time_s=(
                        float(dataset.time_s[int(context_snapshot.snapshot_index)])
                        if 0 <= int(context_snapshot.snapshot_index) < int(dataset.time_s.size)
                        else float("nan")
                    ),
                    geometry=current_geometry,
                    transmission=updated_transmission,
                    warnings=aggregate_warnings(
                        base_result.shock.warnings,
                        base_result.xrd.warnings,
                        base_result.plasmon.warnings,
                        updated_transmission.warnings,
                        base_result.spectroscopy.warnings,
                        (() if base_result.wave_tracking is None else base_result.wave_tracking.warnings),
                        (() if base_result.interface_events is None else base_result.interface_events.warnings),
                        (() if base_result.preheat is None else base_result.preheat.warnings),
                    ),
                )
                return _ComputedAnalysisPayload(
                    result=updated_result,
                    core_request_key=request.core_request_key,
                    update_kind="transmission_model",
                )
            base_result = displayed_base_result
            if base_result is None:
                base_result = self._analysis_core_cache.get(request.core_request_key)
            return _ComputedAnalysisPayload(
                result=compute_analysis_result(
                    dataset,
                    context_snapshot,
                    parameters=request.parameters,
                    context_key=request_key,
                    requested_time_plot_modules=request.requested_time_plot_modules,
                    base_result=(base_result if request.update_kind in {"time_plots", "wavefront", "preheat"} else None),
                    include_wavefront=bool(request.include_wavefront),
                    progress_check=cancellation_token.check_cancelled,
                ),
                core_request_key=request.core_request_key,
                update_kind=request.update_kind,
            )

        cancellation_token = CancellationToken()
        handle = self._tasks.submit(context_key=request_key, fn=_compute, cancellation_token=cancellation_token)
        self._active_task_id = handle.task_id
        self._active_task_request_key = request_key
        self._active_task_token = handle.cancellation_token

    def _start_pending_request_if_needed(self) -> None:
        if self._shutting_down:
            return
        pending = self._pending_request
        if pending is None:
            if self._active_task_id is None:
                self._set_busy(False, "Derived mode ready.")
            return
        self._pending_request = None
        if not self._active or not self._context.has_run or not self._workspace_alive:
            self._set_busy(False, "Derived mode synchronized to the active run.")
            return
        cached_core = self._analysis_core_cache.get(pending.core_request_key)
        displayed_base = self._compatible_displayed_result(pending.reuse_request_key)
        base_result = displayed_base if displayed_base is not None else cached_core
        if (
            pending.update_kind != "transmission_model"
            and
            base_result is not None
            and pending.requested_time_plot_modules.issubset(analysis_result_time_plot_modules(base_result))
            and (not pending.include_wavefront or self._result_has_wavefront(base_result))
            and (not pending.include_wavefront or self._preheat_target_matches(base_result, pending.parameters))
            and (pending.update_kind != "preheat" or self._result_has_preheat(base_result))
        ):
            pending.update_kind = "snapshot"
        self._launch_request(pending)

    def _start_recompute(self) -> None:
        if not self._context.has_run or self._context.path is None or not self._workspace_alive or self._shutting_down or not self._active:
            return
        parameters = self._workspace.parameters()
        reuse_request_key = self._build_reuse_request_key(parameters, context=self._context)
        core_request_key = self._build_core_request_key(parameters)
        request_key = self._build_request_key(parameters)
        context_snapshot = self._context.copy()
        requested_time_plot_modules = self._requested_time_plot_modules()
        advanced_kind = self._workspace.active_advanced_request_kind()
        include_wavefront = advanced_kind is not None
        transmission_mode = normalize_transmission_mode(self._workspace.selected_transmission_mode()) if self._workspace_alive else "thomson"
        transmission_special = self._workspace.transmission_requested() and transmission_mode != "thomson"
        cached_core = self._analysis_core_cache.get(core_request_key)
        displayed_base = self._compatible_displayed_result(reuse_request_key)
        feature_base = displayed_base if displayed_base is not None else cached_core
        available_time_plot_modules = analysis_result_time_plot_modules(feature_base) if feature_base is not None else frozenset()
        if transmission_special:
            update_kind = "transmission_model"
        elif feature_base is None:
            update_kind = advanced_kind or "full"
        elif advanced_kind == "preheat" and (not self._result_has_preheat(feature_base) or not self._preheat_target_matches(feature_base, parameters)):
            update_kind = "preheat"
        elif advanced_kind == "wavefront" and (not self._result_has_wavefront(feature_base) or not self._preheat_target_matches(feature_base, parameters)):
            update_kind = "wavefront"
        elif requested_time_plot_modules.issubset(available_time_plot_modules):
            update_kind = "snapshot"
        else:
            update_kind = "time_plots"
            requested_time_plot_modules = frozenset(set(available_time_plot_modules) | set(requested_time_plot_modules))
        request = _PendingAnalysisRequest(
            context=context_snapshot,
            parameters=parameters,
            reuse_request_key=reuse_request_key,
            core_request_key=core_request_key,
            request_key=(
                request_key
                if update_kind != "transmission_model"
                else (*request_key, "transmission_model", transmission_mode, round(float(parameters.transmission_photon_energy_kev), 12))
            ),
            requested_time_plot_modules=requested_time_plot_modules,
            include_wavefront=include_wavefront,
            update_kind=update_kind,
            generation=self._next_generation(),
            base_request_key=(request_key if update_kind == "transmission_model" else None),
            transmission_model_energy_kev=(float(parameters.transmission_photon_energy_kev) if update_kind == "transmission_model" else None),
            transmission_model_mode=(transmission_mode if update_kind == "transmission_model" else None),
        )
        final_request_key = request.request_key
        if self._active_task_id is not None:
            self._pending_request = request
            self._active_request_key = final_request_key
            self._active_reuse_request_key = request.reuse_request_key
            self._tasks.cancel(self._active_task_id)
            if update_kind == "snapshot":
                queue_message = "Queued latest snapshot update..."
            elif update_kind == "time_plots":
                queue_message = "Queued latest derived time-trace update..."
            elif update_kind == "preheat":
                queue_message = "Queued latest preheat update..."
            elif update_kind == "wavefront":
                queue_message = "Queued latest WaveFront branch-tracking update..."
            elif update_kind == "transmission_model":
                queue_message = "Queued latest Transmission model update..."
            else:
                queue_message = "Queued latest derived analysis update..."
            self._set_busy(True, queue_message)
            return
        if final_request_key == self._displayed_result_key and final_request_key in self._analysis_cache:
            try:
                self._active_request_key = final_request_key
                self._active_reuse_request_key = reuse_request_key
                cached_result = self._validate_result(self._analysis_cache[final_request_key], request_key=final_request_key)
                self._apply_result(cached_result, request_key=final_request_key)
            except Exception as exc:  # pragma: no cover - defensive UI path
                LOGGER.exception("Failed to apply cached derived result.")
                self._show_failure_state(f"Derived analysis failed: {exc}")
                return
            if update_kind == "transmission_model":
                increment_counter("derived.cache.transmission_model.hit")
                self._set_performance_summary(update_kind="transmission_model", from_cache=True)
                self._set_busy(False, "Loaded cached Transmission model result.")
            elif advanced_kind == "wavefront":
                increment_counter("derived.cache.wavefront.hit")
                self._set_performance_summary(update_kind="wavefront", from_cache=True)
                self._set_busy(False, "WaveFront analysis already up to date.")
            elif advanced_kind == "preheat":
                increment_counter("derived.cache.preheat.hit")
                self._set_performance_summary(update_kind="preheat", from_cache=True)
                self._set_busy(False, "Preheat analysis already up to date.")
            else:
                self._set_performance_summary(update_kind=update_kind, from_cache=True)
                self._set_busy(False, "Derived metrics already up to date.")
            return
        cached = self._analysis_cache.get(final_request_key)
        if cached is not None:
            try:
                self._active_request_key = final_request_key
                self._active_reuse_request_key = reuse_request_key
                cached_result = self._validate_result(cached, request_key=final_request_key)
                self._apply_result(cached_result, request_key=final_request_key)
            except Exception as exc:  # pragma: no cover - defensive UI path
                LOGGER.exception("Failed to apply cached derived result.")
                self._show_failure_state(f"Derived analysis failed: {exc}")
                return
            if update_kind == "transmission_model":
                increment_counter("derived.cache.transmission_model.hit")
                self._set_performance_summary(update_kind="transmission_model", from_cache=True)
                self._set_busy(False, "Loaded cached Transmission model result.")
            elif advanced_kind == "wavefront":
                increment_counter("derived.cache.wavefront.hit")
                self._set_performance_summary(update_kind="wavefront", from_cache=True)
                self._set_busy(False, "Loaded cached WaveFront analysis.")
            elif advanced_kind == "preheat":
                increment_counter("derived.cache.preheat.hit")
                self._set_performance_summary(update_kind="preheat", from_cache=True)
                self._set_busy(False, "Loaded cached preheat analysis.")
            else:
                self._set_performance_summary(update_kind=update_kind, from_cache=True)
                self._set_busy(False, "Loaded derived metrics from cache.")
            return
        if update_kind == "transmission_model":
            increment_counter("derived.cache.transmission_model.miss")
        elif advanced_kind == "wavefront":
            increment_counter("derived.cache.wavefront.miss")
        elif advanced_kind == "preheat":
            increment_counter("derived.cache.preheat.miss")
        self._launch_request(request)

    @QtCore.Slot(object)
    def _on_task_result(self, result: AnalysisTaskResult) -> None:
        if not self._workspace_alive:
            LOGGER.debug("Discarding derived result because the workspace was destroyed.")
            return
        if result.task_id != self._active_task_id:
            LOGGER.debug("Discarding stale derived result for task %s", result.task_id)
            return
        self._active_task_id = None
        self._active_task_request_key = None
        self._active_task_token = None
        payload = result.result
        try:
            if isinstance(payload, _ComputedAnalysisPayload):
                computed_result = self._validate_result(payload.result, request_key=result.context_key)
                if payload.update_kind in {"full", "time_plots", "wavefront", "preheat"}:
                    computed_result = self._merge_cached_features(self._analysis_core_cache.get(payload.core_request_key), computed_result)
                    self._analysis_core_cache[payload.core_request_key] = computed_result
                self._last_completed_update_kind = payload.update_kind
            else:
                computed_result = self._validate_result(payload, request_key=result.context_key)
                self._last_completed_update_kind = "full"
            self._analysis_cache[result.context_key] = computed_result
        except Exception as exc:  # pragma: no cover - defensive result-validation path
            LOGGER.exception("Derived analysis returned an invalid result payload.")
            self._show_failure_state(f"Derived analysis failed: {exc}")
            if self._pending_request is not None:
                self._start_pending_request_if_needed()
            return
        active_request_key = self._active_request_key
        if active_request_key is None or result.context_key != active_request_key:
            LOGGER.debug("Discarding stale derived result for context %s", result.context_key)
            self._start_pending_request_if_needed()
            return
        try:
            self._apply_result(computed_result, request_key=result.context_key)
        except Exception as exc:  # pragma: no cover - defensive UI path
            LOGGER.exception("Failed to apply derived result.")
            self._show_failure_state(f"Derived analysis failed: {exc}")
            if self._pending_request is not None:
                self._start_pending_request_if_needed()
            return
        if self._last_completed_update_kind == "snapshot":
            self._set_performance_summary(update_kind="snapshot", from_cache=False)
            self._set_busy(False, "Snapshot-local derived views updated.")
        elif self._last_completed_update_kind == "time_plots":
            self._set_performance_summary(update_kind="time_plots", from_cache=False)
            self._set_busy(False, "Derived time traces updated for the active module.")
        elif self._last_completed_update_kind == "transmission_model":
            self._set_performance_summary(update_kind="transmission_model", from_cache=False)
            self._set_busy(False, "Transmission model updated.")
        elif self._last_completed_update_kind == "preheat":
            self._set_performance_summary(update_kind="preheat", from_cache=False)
            self._set_busy(False, "Preheat diagnostics updated.")
        elif self._last_completed_update_kind == "wavefront":
            self._set_performance_summary(update_kind="wavefront", from_cache=False)
            self._set_busy(False, "WaveFront branches updated.")
        else:
            self._set_performance_summary(update_kind="full", from_cache=False)
            self._set_busy(False, "Derived metrics updated.")
        self._start_pending_request_if_needed()

    @QtCore.Slot(str, str)
    def _on_task_failed(self, task_id: str, message: str) -> None:
        if not self._workspace_alive:
            LOGGER.debug("Discarding derived failure because the workspace was destroyed.")
            return
        if task_id != self._active_task_id:
            LOGGER.debug("Discarding stale derived failure for task %s", task_id)
            return
        self._active_task_id = None
        self._active_task_request_key = None
        self._active_task_token = None
        self._active_reuse_request_key = None
        if self._pending_request is not None:
            LOGGER.warning("Derived analysis task failed for an outdated selection: %s", message)
            self._start_pending_request_if_needed()
            return
        self._show_failure_state(f"Derived analysis failed: {message}")

    @QtCore.Slot(str)
    def _on_task_cancelled(self, task_id: str) -> None:
        if task_id != self._active_task_id:
            LOGGER.debug("Discarding stale derived cancellation for task %s", task_id)
            return
        self._active_task_id = None
        self._active_task_request_key = None
        self._active_task_token = None
        self._active_reuse_request_key = None
        self._start_pending_request_if_needed()
