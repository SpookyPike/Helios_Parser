"""Background parser preview/parse controller for the unified shell.

The production parser remains in ``helios_parser``. This module only packages
that parser into a Qt-friendly worker/controller pair so Parser Mode can:

- preview log structure without blocking the UI
- run the real HDF5 conversion pipeline in the background
- return compact payloads for shell rendering
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore

from helios_parser import HeliosParser, write_hdf5
from helios_parser.hdf5 import WriteProgress


LOGGER = logging.getLogger(__name__)


def _pretty_field(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _field_families(field_names: list[str]) -> list[str]:
    categories: dict[str, list[str]] = {
        "density": ["density"],
        "temperature": ["temperature"],
        "pressure": ["pressure", "viscosity"],
        "velocity": ["velocity"],
        "radiation": ["radiation", "rad"],
        "energy": ["energy", "heat", "laser"],
        "charge / ionization": ["charge", "electron_density"],
        "geometry": ["radius", "zone_width", "compression"],
    }
    detected: list[str] = []
    lowered = [name.lower() for name in field_names]
    for label, needles in categories.items():
        if any(any(needle in name for needle in needles) for name in lowered):
            detected.append(label)
    return detected


def _tail_snippet(path: Path, *, line_count: int = 20) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except Exception:
        return ""
    snippet = "".join(lines[-line_count:]).strip()
    return snippet


@dataclass(slots=True)
class ParsePreviewPayload:
    """Fast preview summary returned by Parser Mode."""

    source: Path
    simulation_name: str
    geometry: str | None
    n_snapshots: int
    n_zones: int
    n_regions: int
    n_materials: int
    fields: list[str]
    field_units: dict[str, str]
    field_families: list[str]
    first_cycle: int | None
    first_time: float | None
    first_time_step: float | None
    last_cycle: int | None
    last_time: float | None
    approx_numeric_bytes: int
    header_sections: tuple[str, ...]
    code_version: str | None
    calculation_datetime: str | None


@dataclass(slots=True)
class ParseResultPayload:
    """Result metadata emitted after a successful HDF5 conversion."""

    source: Path
    output: Path
    elapsed_s: float


@dataclass(slots=True)
class ParseProgressPayload:
    """Granular progress update for a running parse job."""

    stage: str
    current: int
    total: int
    fraction: float
    message: str
    elapsed_s: float
    eta_s: float | None


class ParserWorker(QtCore.QObject):
    """Worker object that runs parser preview/parse jobs off the UI thread."""

    preview_ready = QtCore.Signal(object)
    parse_succeeded = QtCore.Signal(object)
    progress = QtCore.Signal(object)
    status = QtCore.Signal(str)
    error = QtCore.Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._parser = HeliosParser()

    @QtCore.Slot(str)
    def preview_log(self, path_text: str) -> None:
        path = Path(path_text)
        try:
            started_at = time.perf_counter()
            self.status.emit(f"Previewing {path.name}...")
            with self._parser.open_document(path) as document:
                header = document.inspect()
                snapshot_count = document.index.snapshot_count
                preview = document.preview()
                first_snapshot = preview.snapshot
                last_snapshot = None
                if snapshot_count > 1:
                    block = document.index.snapshot_blocks[-1]
                    last_snapshot = document.snapshot_parser.parse(document.buffer.slice(block.span), header)

            fields = list(first_snapshot.fields.keys()) if first_snapshot is not None else []
            field_units = dict(first_snapshot.field_units) if first_snapshot is not None else {}
            approx_numeric_bytes = snapshot_count * header.n_zones * max(1, len(fields)) * 8
            payload = ParsePreviewPayload(
                source=path,
                simulation_name=header.simulation_name,
                geometry=str(header.metadata.get("geometry") or ""),
                n_snapshots=snapshot_count,
                n_zones=header.n_zones,
                n_regions=header.n_regions,
                n_materials=header.n_materials,
                fields=fields,
                field_units=field_units,
                field_families=_field_families(fields),
                first_cycle=first_snapshot.cycle if first_snapshot is not None else None,
                first_time=first_snapshot.time if first_snapshot is not None else None,
                first_time_step=first_snapshot.time_step if first_snapshot is not None else None,
                last_cycle=last_snapshot.cycle if last_snapshot is not None else (first_snapshot.cycle if first_snapshot is not None else None),
                last_time=last_snapshot.time if last_snapshot is not None else (first_snapshot.time if first_snapshot is not None else None),
                approx_numeric_bytes=approx_numeric_bytes,
                header_sections=header.header_sections,
                code_version=header.code_version,
                calculation_datetime=header.calculation_datetime,
            )
            self.preview_ready.emit(payload)
            LOGGER.info("Previewed %s in %.3f s", path.name, time.perf_counter() - started_at)
            self.status.emit(f"Preview ready: {path.name}")
        except Exception as exc:
            details = traceback.format_exc()
            snippet = _tail_snippet(path)
            if snippet:
                details += f"\n\nLog tail:\n{snippet}"
            self.error.emit(f"Failed to preview {path.name}", details)

    @QtCore.Slot(str, str, object, bool)
    def parse_log(self, source_text: str, output_text: str, compression: object, overwrite: bool) -> None:
        source = Path(source_text)
        output = Path(output_text)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.status.emit(f"Parsing {source.name} -> {output.name}...")
            started = QtCore.QElapsedTimer()
            started.start()
            started_perf = time.perf_counter()

            def on_progress(progress: WriteProgress) -> None:
                elapsed_s = started.elapsed() / 1000.0
                eta_s: float | None = None
                if progress.current > 0 and progress.total > 0 and progress.current < progress.total:
                    rate = elapsed_s / progress.current
                    eta_s = max(0.0, rate * (progress.total - progress.current))
                payload = ParseProgressPayload(
                    stage=progress.stage,
                    current=progress.current,
                    total=progress.total,
                    fraction=progress.fraction,
                    message=progress.message,
                    elapsed_s=elapsed_s,
                    eta_s=eta_s,
                )
                self.progress.emit(payload)

            write_hdf5(
                source,
                output,
                compression=None if compression in {None, "", "none"} else str(compression),
                overwrite=bool(overwrite),
                parser=self._parser,
                progress_callback=on_progress,
            )
            elapsed_s = started.elapsed() / 1000.0
            self.parse_succeeded.emit(ParseResultPayload(source=source, output=output, elapsed_s=elapsed_s))
            LOGGER.info("Parsed %s to %s in %.3f s", source.name, output.name, time.perf_counter() - started_perf)
            self.status.emit(f"Parse complete: {output.name}")
        except Exception as exc:
            details = traceback.format_exc()
            snippet = _tail_snippet(source)
            if snippet:
                details += f"\n\nLog tail:\n{snippet}"
            self.error.emit(f"Failed to parse {source.name}", details)


class ParserController(QtCore.QObject):
    """Threaded facade used by the app shell's Parser Mode."""

    preview_ready = QtCore.Signal(object)
    parse_succeeded = QtCore.Signal(object)
    progress_changed = QtCore.Signal(object)
    status_changed = QtCore.Signal(str)
    error_occurred = QtCore.Signal(str, str)
    busy_changed = QtCore.Signal(bool)

    _preview_requested = QtCore.Signal(str)
    _parse_requested = QtCore.Signal(str, str, object, bool)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QtCore.QThread(self)
        self._worker = ParserWorker()
        self._worker.moveToThread(self._thread)
        self._thread.start()

        self._preview_requested.connect(self._worker.preview_log, QtCore.Qt.QueuedConnection)
        self._parse_requested.connect(self._worker.parse_log, QtCore.Qt.QueuedConnection)
        self._worker.preview_ready.connect(self._handle_preview_ready)
        self._worker.parse_succeeded.connect(self._handle_parse_succeeded)
        self._worker.progress.connect(self.progress_changed)
        self._worker.status.connect(self.status_changed)
        self._worker.error.connect(self._handle_error)

        self.preview_payload: ParsePreviewPayload | None = None
        self.parse_result: ParseResultPayload | None = None
        self._pending_jobs = 0

    def preview_log(self, path: str | Path) -> None:
        self.preview_payload = None
        self._begin_job()
        self._preview_requested.emit(str(Path(path)))

    def parse_log(
        self,
        source: str | Path,
        output: str | Path,
        *,
        compression: str | None = None,
        overwrite: bool = False,
    ) -> None:
        self.parse_result = None
        self._begin_job()
        self._parse_requested.emit(str(Path(source)), str(Path(output)), compression, bool(overwrite))

    def shutdown(self) -> None:
        self._thread.quit()
        self._thread.wait(5000)

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
    def _handle_preview_ready(self, payload: ParsePreviewPayload) -> None:
        self.preview_payload = payload
        self._end_job()
        self.preview_ready.emit(payload)

    @QtCore.Slot(object)
    def _handle_parse_succeeded(self, payload: ParseResultPayload) -> None:
        self.parse_result = payload
        self._end_job()
        self.parse_succeeded.emit(payload)

    @QtCore.Slot(str, str)
    def _handle_error(self, message: str, details: str) -> None:
        self._end_job()
        self.error_occurred.emit(message, details)
