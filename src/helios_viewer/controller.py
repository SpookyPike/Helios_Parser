from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6 import QtCore

from helios.cache import AnalyzerCacheSet, CacheBucketStats
from helios.services.derived.common import publish_field_payload, publish_open_run_payload
from .models import DiagnosticPayload, FieldPayload, OpenRunPayload
from .workers import RunWorker

LOGGER = logging.getLogger(__name__)


class RunController(QtCore.QObject):
    run_opened = QtCore.Signal(object)
    field_loaded = QtCore.Signal(object)
    diagnostic_loaded = QtCore.Signal(object)
    status_changed = QtCore.Signal(str)
    error_occurred = QtCore.Signal(str, str)
    busy_changed = QtCore.Signal(bool)

    _open_requested = QtCore.Signal(str, int)
    _field_requested = QtCore.Signal(str, int)
    _diagnostic_requested = QtCore.Signal(str, int)
    _close_requested = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QtCore.QThread(self)
        self._worker = RunWorker()
        self._worker.moveToThread(self._thread)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

        self._open_requested.connect(self._worker.open_run, QtCore.Qt.QueuedConnection)
        self._field_requested.connect(self._worker.load_field, QtCore.Qt.QueuedConnection)
        self._diagnostic_requested.connect(self._worker.load_diagnostic, QtCore.Qt.QueuedConnection)
        self._close_requested.connect(self._worker.close_run, QtCore.Qt.QueuedConnection)

        self._worker.run_opened.connect(self._handle_run_opened)
        self._worker.field_loaded.connect(self._handle_field_loaded)
        self._worker.diagnostic_loaded.connect(self._handle_diagnostic_loaded)
        self._worker.status.connect(self.status_changed)
        self._worker.error.connect(self._handle_error)

        self.run_payload: OpenRunPayload | None = None
        self._cache_layers = AnalyzerCacheSet()
        self.field_cache = self._cache_layers.raw_data_cache.bucket("viewer_fields", max_items=12)
        self.diagnostic_cache = self._cache_layers.raw_data_cache.bucket("viewer_diagnostics", max_items=12)
        self._pending_jobs = 0
        self._run_generation = 0
        self._shutdown_started = False

    @property
    def run_generation(self) -> int:
        return int(self._run_generation)

    def cache_stats(self) -> dict[str, CacheBucketStats]:
        return {
            "field_cache": self.field_cache.stats(),
            "diagnostic_cache": self.diagnostic_cache.stats(),
        }

    @property
    def worker_thread(self) -> QtCore.QThread:
        return self._thread

    def open_file(self, path: str | Path) -> None:
        self._run_generation += 1
        self.run_payload = None
        self.field_cache.clear(reason=f"open_file:generation:{self._run_generation}")
        self.diagnostic_cache.clear(reason=f"open_file:generation:{self._run_generation}")
        self._begin_job()
        self._open_requested.emit(str(Path(path)), self._run_generation)

    def load_field(self, field_name: str) -> None:
        cached = self.field_cache.get(field_name)
        if cached is not None:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("Viewer field cache hit for %s: %s", field_name, self.field_cache.stats())
            self.field_loaded.emit(cached)
            return
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Viewer field cache miss for %s: %s", field_name, self.field_cache.stats())
        self._begin_job()
        self._field_requested.emit(field_name, self._run_generation)

    def load_diagnostic(self, path: str) -> None:
        cached = self.diagnostic_cache.get(path)
        if cached is not None:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("Viewer diagnostic cache hit for %s: %s", path, self.diagnostic_cache.stats())
            self.diagnostic_loaded.emit(cached)
            return
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Viewer diagnostic cache miss for %s: %s", path, self.diagnostic_cache.stats())
        self._begin_job()
        self._diagnostic_requested.emit(path, self._run_generation)

    def get_zone_mask(self, filter_kind: str, filter_value: int | None = None) -> np.ndarray | None:
        if self.run_payload is None or filter_kind == "all":
            return None
        if filter_kind == "region" and filter_value is not None:
            return self.run_payload.zone_region_id == int(filter_value)
        if filter_kind == "material" and filter_value is not None:
            return np.abs(self.run_payload.zone_material_index) == abs(int(filter_value))
        return None

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._pending_jobs = 0
        self.busy_changed.emit(False)
        if not self._thread.isRunning():
            return
        try:
            QtCore.QMetaObject.invokeMethod(self._worker, "close_run", QtCore.Qt.BlockingQueuedConnection)
        except (RuntimeError, TypeError):
            LOGGER.debug("Viewer worker close_run invoke failed during shutdown.", exc_info=True)
        for signal, slot in (
            (self._worker.run_opened, self._handle_run_opened),
            (self._worker.field_loaded, self._handle_field_loaded),
            (self._worker.diagnostic_loaded, self._handle_diagnostic_loaded),
            (self._worker.status, self.status_changed),
            (self._worker.error, self._handle_error),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._thread.quit()
        if not self._thread.wait(5000):
            LOGGER.warning("Viewer worker thread did not stop within 5 s during shutdown.")

    def _begin_job(self) -> None:
        self._pending_jobs += 1
        if self._pending_jobs == 1:
            self.busy_changed.emit(True)

    def _end_job(self) -> None:
        if self._pending_jobs > 0:
            self._pending_jobs -= 1
        if self._pending_jobs == 0:
            self.busy_changed.emit(False)

    @QtCore.Slot(object)
    def _handle_run_opened(self, payload: OpenRunPayload) -> None:
        if int(payload.run_generation) != self._run_generation:
            LOGGER.debug(
                "Discarding stale run payload for generation %s; current generation is %s.",
                payload.run_generation,
                self._run_generation,
            )
            self.status_changed.emit(
                f"Discarded stale run payload for generation {payload.run_generation}; current generation is {self._run_generation}."
            )
            self._end_job()
            return
        self.run_payload = payload
        self.field_cache.clear(reason=f"run_opened:generation:{self._run_generation}")
        self.diagnostic_cache.clear(reason=f"run_opened:generation:{self._run_generation}")
        publish_open_run_payload(
            payload.path,
            summary=payload.summary,
            metadata=payload.metadata,
            regions=payload.regions,
            materials=payload.materials,
            fields=payload.fields,
            diagnostics=payload.diagnostics,
            time_values=payload.time,
            static_x_center=payload.static_x,
            static_x_edge=payload.static_x_edges,
            zone_region_id=payload.zone_region_id,
            zone_material_index=payload.zone_material_index,
            has_dynamic_radius=payload.has_dynamic_radius,
            run_status=payload.run_status,
            visar_support_metadata=payload.visar_support_metadata,
        )
        self._end_job()
        self.run_opened.emit(payload)

    @QtCore.Slot(object)
    def _handle_field_loaded(self, payload: FieldPayload) -> None:
        if int(payload.run_generation) != self._run_generation:
            LOGGER.debug(
                "Discarding stale field payload %s for generation %s; current generation is %s.",
                payload.field_name,
                payload.run_generation,
                self._run_generation,
            )
            self.status_changed.emit(
                f"Discarded stale field payload for generation {payload.run_generation}; current generation is {self._run_generation}."
            )
            self._end_job()
            return
        self.field_cache[payload.field_name] = payload
        if self.run_payload is not None:
            publish_field_payload(
                self.run_payload.path,
                field_name=payload.field_name,
                data=payload.data,
                edge_data=payload.edge_data,
            )
        self._end_job()
        self.field_loaded.emit(payload)

    @QtCore.Slot(object)
    def _handle_diagnostic_loaded(self, payload: DiagnosticPayload) -> None:
        if int(payload.run_generation) != self._run_generation:
            LOGGER.debug(
                "Discarding stale diagnostic payload %s for generation %s; current generation is %s.",
                payload.path,
                payload.run_generation,
                self._run_generation,
            )
            self.status_changed.emit(
                f"Discarded stale diagnostic payload for generation {payload.run_generation}; current generation is {self._run_generation}."
            )
            self._end_job()
            return
        self.diagnostic_cache[payload.path] = payload
        self._end_job()
        self.diagnostic_loaded.emit(payload)

    @QtCore.Slot(str, str)
    def _handle_error(self, message: str, details: str) -> None:
        self._end_job()
        self.error_occurred.emit(message, details)
