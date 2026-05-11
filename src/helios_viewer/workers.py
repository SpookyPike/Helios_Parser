from __future__ import annotations

from dataclasses import asdict
import logging
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore

from helios_parser import HeliosRun

from .models import DiagnosticPayload, FieldPayload, FieldTracePayload, OpenRunPayload, SnapshotFieldPayload


LOGGER = logging.getLogger(__name__)


class RunWorker(QtCore.QObject):
    run_opened = QtCore.Signal(object)
    field_loaded = QtCore.Signal(object)
    snapshot_field_loaded = QtCore.Signal(object)
    field_trace_loaded = QtCore.Signal(object)
    diagnostic_loaded = QtCore.Signal(object)
    status = QtCore.Signal(str)
    error = QtCore.Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._run: HeliosRun | None = None

    @QtCore.Slot(str, int)
    def open_run(self, path_text: str, generation: int) -> None:
        try:
            self._close_current_run()
            path = Path(path_text)
            self.status.emit(f"Opening {path.name}...")
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
            )
            self.run_opened.emit(payload)
            LOGGER.info("Opened %s in %.3f s", path.name, time.perf_counter() - started)
            self.status.emit(f"Opened {path.name}")
        except Exception as exc:
            self.error.emit(str(exc), traceback.format_exc())

    @QtCore.Slot(str, int)
    def load_field(self, field_name: str, generation: int) -> None:
        try:
            if self._run is None:
                LOGGER.debug("Ignoring field load %s for generation %s because no run is open.", field_name, generation)
                self.status.emit(f"Ignored stale field request: {field_name}")
                return
            self.status.emit(f"Loading field: {field_name}")
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
            )
            self.field_loaded.emit(payload)
            LOGGER.info("Loaded field %s in %.3f s", field_name, time.perf_counter() - started)
            self.status.emit(f"Loaded field: {field_name}")
        except Exception as exc:
            self.error.emit(str(exc), traceback.format_exc())

    @QtCore.Slot(str, int, int)
    def load_snapshot_field(self, field_name: str, snapshot_index: int, generation: int) -> None:
        try:
            if self._run is None:
                LOGGER.debug(
                    "Ignoring snapshot field load %s[%s] for generation %s because no run is open.",
                    field_name,
                    snapshot_index,
                    generation,
                )
                self.status.emit(f"Ignored stale snapshot field request: {field_name}")
                return
            normalized_snapshot = self._run._normalize_snapshot_index(int(snapshot_index))
            self.status.emit(f"Loading snapshot field: {field_name}[{normalized_snapshot}]")
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
            )
            self.snapshot_field_loaded.emit(payload)
            LOGGER.info("Loaded snapshot field %s[%s] in %.3f s", field_name, normalized_snapshot, time.perf_counter() - started)
            self.status.emit(f"Loaded snapshot field: {field_name}[{normalized_snapshot}]")
        except Exception as exc:
            self.error.emit(str(exc), traceback.format_exc())

    @QtCore.Slot(str, int, int)
    def load_field_trace(self, field_name: str, zone_index: int, generation: int) -> None:
        try:
            if self._run is None:
                LOGGER.debug("Ignoring field trace load %s[:,%s] for generation %s because no run is open.", field_name, zone_index, generation)
                self.status.emit(f"Ignored stale field trace request: {field_name}")
                return
            self.status.emit(f"Loading field trace: {field_name} zone {int(zone_index) + 1}")
            started = time.perf_counter()
            data = np.asarray(self._run.get_time_trace(field_name, int(zone_index)), dtype=np.float64)
            payload = FieldTracePayload(
                run_generation=int(generation),
                field_name=field_name,
                zone_index=int(zone_index),
                unit=self._run.get_field_unit(field_name),
                data=data,
            )
            self.field_trace_loaded.emit(payload)
            LOGGER.info("Loaded field trace %s[:,%s] in %.3f s", field_name, zone_index, time.perf_counter() - started)
            self.status.emit(f"Loaded field trace: {field_name} zone {int(zone_index) + 1}")
        except Exception as exc:
            self.error.emit(str(exc), traceback.format_exc())

    @QtCore.Slot(str, int)
    def load_diagnostic(self, path: str, generation: int) -> None:
        try:
            if self._run is None:
                LOGGER.debug("Ignoring diagnostic load %s for generation %s because no run is open.", path, generation)
                self.status.emit(f"Ignored stale diagnostic request: {path}")
                return
            self.status.emit(f"Loading diagnostic: {path}")
            started = time.perf_counter()
            payload = DiagnosticPayload(
                run_generation=int(generation),
                path=path,
                unit=self._run.get_diagnostic_unit(path),
                data=np.asarray(self._run.get_diagnostic(path)),
            )
            self.diagnostic_loaded.emit(payload)
            LOGGER.info("Loaded diagnostic %s in %.3f s", path, time.perf_counter() - started)
            self.status.emit(f"Loaded diagnostic: {path}")
        except Exception as exc:
            self.error.emit(str(exc), traceback.format_exc())

    @QtCore.Slot()
    def close_run(self) -> None:
        self._close_current_run()

    def _close_current_run(self) -> None:
        if self._run is not None:
            self._run.close()
            self._run = None
