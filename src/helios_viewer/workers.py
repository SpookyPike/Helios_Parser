from __future__ import annotations

from dataclasses import asdict
import logging
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore

from helios_parser import HeliosRun
from helios.runtime import RunRevision

from .models import DiagnosticPayload, FieldPayload, FieldTracePayload, OpenRunPayload, SnapshotFieldPayload


LOGGER = logging.getLogger(__name__)


class RunWorker(QtCore.QObject):
    run_opened = QtCore.Signal(object)
    field_loaded = QtCore.Signal(object)
    snapshot_field_loaded = QtCore.Signal(object)
    field_trace_loaded = QtCore.Signal(object)
    diagnostic_loaded = QtCore.Signal(object)
    status = QtCore.Signal(str, int, object)
    error = QtCore.Signal(str, str, int, object, object)

    def __init__(self) -> None:
        super().__init__()
        self._run: HeliosRun | None = None
        self._revision: RunRevision | None = None

    def _emit_status(self, message: str, generation: int) -> None:
        self.status.emit(str(message), int(generation), self._revision)

    def _emit_error(self, exc: Exception, generation: int, request_key: tuple[object, ...]) -> None:
        self.error.emit(str(exc), traceback.format_exc(), int(generation), self._revision, request_key)

    @QtCore.Slot(str, int, object, object)
    def open_run(self, path_text: str, generation: int, revision: RunRevision, request_key: tuple[object, ...]) -> None:
        try:
            self._close_current_run()
            self._revision = revision
            path = Path(path_text)
            self._emit_status(f"Opening {path.name}...", generation)
            started = time.perf_counter()
            run = HeliosRun(path)
            self._run = run
            fields = run.list_fields()
            diagnostics = run.list_diagnostics()
            payload = OpenRunPayload(
                run_generation=int(generation),
                path=path,
                summary=run.summary(),
                metadata=run.get_metadata(),
                fields=fields,
                field_units={name: run.get_field_unit(name) for name in fields},
                field_metadata={name: asdict(run.get_field_metadata(name)) for name in fields},
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
                run_status=run.get_run_status(),
                visar_support_metadata=asdict(run.get_visar_support_metadata()),
                revision=revision,
                request_key=request_key,
            )
            self.run_opened.emit(payload)
            LOGGER.info("Opened %s in %.3f s", path.name, time.perf_counter() - started)
            self._emit_status(f"Opened {path.name}", generation)
        except Exception as exc:
            self._emit_error(exc, generation, request_key)

    @QtCore.Slot(str, int, object)
    def load_field(self, field_name: str, generation: int, request_key: tuple[object, ...]) -> None:
        try:
            if self._run is None:
                raise RuntimeError(f"Cannot load field {field_name}: no run is open.")
            self._emit_status(f"Loading field: {field_name}", generation)
            started = time.perf_counter()
            edge_data: np.ndarray | None = None
            if field_name == "radius":
                dynamic_center = self._run.get_dynamic_coordinate(location="center")
                if dynamic_center is not None:
                    data = np.asarray(dynamic_center, dtype=np.float64)
                    dynamic_edge = self._run.get_dynamic_coordinate(location="edge")
                    edge_data = None if dynamic_edge is None else np.asarray(dynamic_edge, dtype=np.float64)
                else:
                    data = np.asarray(self._run.get_field(field_name), dtype=np.float64)
            else:
                data = np.asarray(self._run.get_field(field_name), dtype=np.float64)
            payload = FieldPayload(
                run_generation=int(generation),
                field_name=field_name,
                unit=self._run.get_field_unit(field_name),
                data=data,
                edge_data=edge_data,
                revision=self._revision,
                request_key=request_key,
            )
            self.field_loaded.emit(payload)
            LOGGER.info("Loaded field %s in %.3f s", field_name, time.perf_counter() - started)
            self._emit_status(f"Loaded field: {field_name}", generation)
        except Exception as exc:
            self._emit_error(exc, generation, request_key)

    @QtCore.Slot(str, int, int, object)
    def load_snapshot_field(self, field_name: str, snapshot_index: int, generation: int, request_key: tuple[object, ...]) -> None:
        try:
            if self._run is None:
                raise RuntimeError(f"Cannot load snapshot field {field_name}[{snapshot_index}]: no run is open.")
            normalized_snapshot = self._run._normalize_snapshot_index(int(snapshot_index))
            self._emit_status(f"Loading snapshot field: {field_name}[{normalized_snapshot}]", generation)
            started = time.perf_counter()
            edge_data: np.ndarray | None = None
            if field_name == "radius":
                dynamic_center = self._run.get_dynamic_coordinate(normalized_snapshot, location="center")
                if dynamic_center is not None:
                    data = np.asarray(dynamic_center, dtype=np.float64)
                    dynamic_edge = self._run.get_dynamic_coordinate(normalized_snapshot, location="edge")
                    edge_data = None if dynamic_edge is None else np.asarray(dynamic_edge, dtype=np.float64)
                else:
                    data = np.asarray(self._run.get_snapshot_field(field_name, normalized_snapshot), dtype=np.float64)
            else:
                data = np.asarray(self._run.get_snapshot_field(field_name, normalized_snapshot), dtype=np.float64)
            payload = SnapshotFieldPayload(
                run_generation=int(generation),
                field_name=field_name,
                snapshot_index=normalized_snapshot,
                unit=self._run.get_field_unit(field_name),
                data=data,
                edge_data=edge_data,
                revision=self._revision,
                request_key=request_key,
            )
            self.snapshot_field_loaded.emit(payload)
            LOGGER.info("Loaded snapshot field %s[%s] in %.3f s", field_name, normalized_snapshot, time.perf_counter() - started)
            self._emit_status(f"Loaded snapshot field: {field_name}[{normalized_snapshot}]", generation)
        except Exception as exc:
            self._emit_error(exc, generation, request_key)

    @QtCore.Slot(str, int, int, object)
    def load_field_trace(self, field_name: str, zone_index: int, generation: int, request_key: tuple[object, ...]) -> None:
        try:
            if self._run is None:
                raise RuntimeError(f"Cannot load field trace {field_name}[:,{zone_index}]: no run is open.")
            self._emit_status(f"Loading field trace: {field_name} zone {int(zone_index) + 1}", generation)
            started = time.perf_counter()
            data = np.asarray(self._run.get_time_trace(field_name, int(zone_index)), dtype=np.float64)
            payload = FieldTracePayload(
                run_generation=int(generation),
                field_name=field_name,
                zone_index=int(zone_index),
                unit=self._run.get_field_unit(field_name),
                data=data,
                revision=self._revision,
                request_key=request_key,
            )
            self.field_trace_loaded.emit(payload)
            LOGGER.info("Loaded field trace %s[:,%s] in %.3f s", field_name, zone_index, time.perf_counter() - started)
            self._emit_status(f"Loaded field trace: {field_name} zone {int(zone_index) + 1}", generation)
        except Exception as exc:
            self._emit_error(exc, generation, request_key)

    @QtCore.Slot(str, int, object)
    def load_diagnostic(self, path: str, generation: int, request_key: tuple[object, ...]) -> None:
        try:
            if self._run is None:
                raise RuntimeError(f"Cannot load diagnostic {path}: no run is open.")
            self._emit_status(f"Loading diagnostic: {path}", generation)
            started = time.perf_counter()
            payload = DiagnosticPayload(
                run_generation=int(generation),
                path=path,
                unit=self._run.get_diagnostic_unit(path),
                data=np.asarray(self._run.get_diagnostic(path)),
                revision=self._revision,
                request_key=request_key,
            )
            self.diagnostic_loaded.emit(payload)
            LOGGER.info("Loaded diagnostic %s in %.3f s", path, time.perf_counter() - started)
            self._emit_status(f"Loaded diagnostic: {path}", generation)
        except Exception as exc:
            self._emit_error(exc, generation, request_key)

    @QtCore.Slot()
    def close_run(self) -> None:
        self._close_current_run()

    def _close_current_run(self) -> None:
        if self._run is not None:
            self._run.close()
            self._run = None
            self._revision = None
