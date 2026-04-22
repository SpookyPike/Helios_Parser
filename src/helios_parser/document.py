from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .buffer import HeliosBuffer
from .indexing import StructuralIndexer
from .model import HeliosBlockIndex, HeliosHeader, HeliosPreview, RunStatusInfo, Simulation, Snapshot
from .parser import HeaderSemanticParser, SnapshotBlockSemanticParser


RUN_COMPLETED_RE = re.compile(r"#\s*Simulation completed on\s*:\s*(.+)", re.IGNORECASE)
RUN_ABORTED_RE = re.compile(r"#\s*Simulation (?:aborted|terminated|stopped|ended)(?:\s+on|\s+at)?\s*:?\s*(.+)?", re.IGNORECASE)


def flatten_snapshot_diagnostics(diagnostics: dict[str, Any]) -> list[tuple[tuple[str, ...], Any]]:
    flattened: list[tuple[tuple[str, ...], Any]] = []

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if path == () and key == "units":
                    continue
                walk(child, path + (key,))
            return
        flattened.append((path, value))

    walk(diagnostics, ())
    return flattened


def diagnostic_value_width(value: Any) -> int | None:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim == 0:
            return None
        return int(array.shape[0])
    return None


def normalize_diagnostic_value(value: Any, width: int | None) -> np.ndarray | float:
    """Normalize snapshot diagnostics to a stable numeric schema.

    The normal hot path keeps a stable width. If a caller explicitly widens the
    schema first via `reconcile_diagnostic_width`, this function pads into that
    widened shape without renegotiating it here.
    """

    if width is None:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            return float(array[0]) if array.size else float("nan")
    padded = np.full(int(width), np.nan, dtype=np.float64)
    if value is None:
        return padded
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size:
        padded[: min(int(width), int(array.size))] = array[: min(int(width), int(array.size))]
    return padded


def reconcile_diagnostic_width(current_width: int | None, value: Any) -> tuple[int | None, bool, str | None]:
    """Return the stable schema width for a diagnostic path.

    Normal case: widths stay unchanged after the first observation. If a later
    vector widens, we upgrade the width explicitly instead of truncating. If a
    path flips from scalar to vector after the schema was locked as scalar, the
    caller must surface that as a validation issue rather than silently
    reshaping already-written scalar data.
    """

    observed_width = diagnostic_value_width(value)
    if current_width is None:
        if observed_width is None:
            return None, False, None
        return int(observed_width), True, "scalar_to_vector"
    if observed_width is None or int(observed_width) <= int(current_width):
        return int(current_width), False, None
    return int(observed_width), True, "vector_widened"


def assign_nested(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = mapping
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


@dataclass(slots=True)
class HeliosDocument:
    buffer: HeliosBuffer
    index: HeliosBlockIndex
    header_parser: HeaderSemanticParser
    snapshot_parser: SnapshotBlockSemanticParser
    _header: HeliosHeader | None = field(default=None, init=False, repr=False)
    _snapshot_cache: tuple[Snapshot, ...] | None = field(default=None, init=False, repr=False)
    _run_status: RunStatusInfo | None = field(default=None, init=False, repr=False)

    def _base_run_status(self, header: HeliosHeader) -> RunStatusInfo:
        intended_end_time = None
        time_control = header.input_parameters.get("time_control", {})
        if isinstance(time_control, dict):
            raw_end_time = time_control.get("max_simulation_time")
            if raw_end_time is not None:
                try:
                    intended_end_time = float(raw_end_time)
                except (TypeError, ValueError):
                    intended_end_time = None
        tail_text = self.buffer.tail_text(20000)
        completed = RUN_COMPLETED_RE.search(tail_text)
        aborted = RUN_ABORTED_RE.search(tail_text) if completed is None else None
        if completed is not None:
            state = "completed"
            source = "footer"
            footer_message = completed.group(0).strip()
            footer_datetime = completed.group(1).strip()
        elif aborted is not None:
            state = "aborted"
            source = "footer"
            footer_message = aborted.group(0).strip()
            footer_datetime = (aborted.group(1) or "").strip() or None
        else:
            state = "unknown"
            source = "footer_absent"
            footer_message = None
            footer_datetime = None
        return RunStatusInfo(
            state=state,
            source=source,
            footer_message=footer_message,
            footer_datetime=footer_datetime,
            intended_end_time_s=intended_end_time,
            last_valid_snapshot_time_s=None,
            indexed_snapshot_count=int(self.index.snapshot_count),
            valid_snapshot_count=int(self.index.snapshot_count),
            notes=(),
        )

    @staticmethod
    def _run_status_metadata(status: RunStatusInfo) -> dict[str, Any]:
        return {
            "state": status.state,
            "source": status.source,
            "footer_message": status.footer_message,
            "footer_datetime": status.footer_datetime,
            "intended_end_time_s": status.intended_end_time_s,
            "last_valid_snapshot_time_s": status.last_valid_snapshot_time_s,
            "indexed_snapshot_count": status.indexed_snapshot_count,
            "valid_snapshot_count": status.valid_snapshot_count,
            "dropped_partial_final_block": status.dropped_partial_final_block,
            "damaged_final_block_reason": status.damaged_final_block_reason,
            "notes": list(status.notes),
        }

    def _apply_run_status_to_header(self, header: HeliosHeader, status: RunStatusInfo) -> None:
        header.run_status = status
        header.metadata["run_status"] = self._run_status_metadata(status)

    def _load_snapshots(self, header: HeliosHeader) -> tuple[tuple[Snapshot, ...], RunStatusInfo]:
        if self._snapshot_cache is not None and self._run_status is not None:
            return self._snapshot_cache, self._run_status

        base_status = self._base_run_status(header)
        notes = list(base_status.notes)
        snapshots: list[Snapshot] = []
        blocks = self.index.snapshot_blocks
        if not blocks:
            raise ValueError(f"No HELIOS snapshots found in {self.buffer.source}.")

        for block_offset, block in enumerate(blocks):
            try:
                snapshot = self.snapshot_parser.parse(self.buffer.slice(block.span), header)
            except Exception as exc:
                if block_offset == len(blocks) - 1 and snapshots:
                    notes.append(
                        "The final indexed snapshot block was structurally incomplete and was dropped. "
                        "The scientific dataset ends at the last fully parseable snapshot."
                    )
                    self._snapshot_cache = tuple(snapshots)
                    self._run_status = RunStatusInfo(
                        state="truncated",
                        source="parser_partial_final_block",
                        footer_message=base_status.footer_message,
                        footer_datetime=base_status.footer_datetime,
                        intended_end_time_s=base_status.intended_end_time_s,
                        last_valid_snapshot_time_s=float(snapshots[-1].time),
                        indexed_snapshot_count=int(len(blocks)),
                        valid_snapshot_count=int(len(snapshots)),
                        dropped_partial_final_block=True,
                        damaged_final_block_reason=str(exc),
                        notes=tuple(notes),
                    )
                    self._apply_run_status_to_header(header, self._run_status)
                    return self._snapshot_cache, self._run_status
                raise ValueError(
                    f"Failed to parse HELIOS snapshot block {int(block.index)} from {self.buffer.source}: {exc}"
                ) from exc
            snapshots.append(snapshot)

        last_valid_time = float(snapshots[-1].time)
        intended = base_status.intended_end_time_s
        if intended is not None and np.isfinite(float(intended)):
            last_dt = float(snapshots[-1].time_step) if np.isfinite(float(snapshots[-1].time_step)) else 0.0
            tolerance = max(abs(last_dt), abs(float(intended)) * 1.0e-6, 1.0e-15)
            if abs(last_valid_time - float(intended)) > tolerance:
                notes.append(
                    "The last fully parseable snapshot time differs from the header-declared maximum simulation time; "
                    "the parser keeps the last valid snapshot without treating that mismatch as a parser failure."
                )

        self._snapshot_cache = tuple(snapshots)
        self._run_status = RunStatusInfo(
            state=base_status.state,
            source=base_status.source,
            footer_message=base_status.footer_message,
            footer_datetime=base_status.footer_datetime,
            intended_end_time_s=base_status.intended_end_time_s,
            last_valid_snapshot_time_s=last_valid_time,
            indexed_snapshot_count=base_status.indexed_snapshot_count,
            valid_snapshot_count=int(len(snapshots)),
            dropped_partial_final_block=False,
            damaged_final_block_reason=None,
            notes=tuple(notes),
        )
        self._apply_run_status_to_header(header, self._run_status)
        return self._snapshot_cache, self._run_status

    def _finalize_stream_run_status(
        self,
        *,
        header: HeliosHeader,
        base_status: RunStatusInfo,
        valid_snapshot_count: int,
        last_valid_snapshot: Snapshot | None,
        truncated_final_block: bool,
        damaged_final_block_reason: str | None,
    ) -> RunStatusInfo:
        if valid_snapshot_count <= 0 or last_valid_snapshot is None:
            raise ValueError(f"No HELIOS snapshots found in {self.buffer.source}.")

        notes = list(base_status.notes)
        if truncated_final_block:
            notes.append(
                "The final indexed snapshot block was structurally incomplete and was dropped. "
                "The scientific dataset ends at the last fully parseable snapshot."
            )

        last_valid_time = float(last_valid_snapshot.time)
        intended = base_status.intended_end_time_s
        if intended is not None and np.isfinite(float(intended)):
            last_dt = float(last_valid_snapshot.time_step) if np.isfinite(float(last_valid_snapshot.time_step)) else 0.0
            tolerance = max(abs(last_dt), abs(float(intended)) * 1.0e-6, 1.0e-15)
            if abs(last_valid_time - float(intended)) > tolerance:
                notes.append(
                    "The last fully parseable snapshot time differs from the header-declared maximum simulation time; "
                    "the parser keeps the last valid snapshot without treating that mismatch as a parser failure."
                )

        state = "truncated" if truncated_final_block else base_status.state
        source = "parser_partial_final_block" if truncated_final_block else base_status.source
        return RunStatusInfo(
            state=state,
            source=source,
            footer_message=base_status.footer_message,
            footer_datetime=base_status.footer_datetime,
            intended_end_time_s=base_status.intended_end_time_s,
            last_valid_snapshot_time_s=last_valid_time,
            indexed_snapshot_count=int(self.index.snapshot_count),
            valid_snapshot_count=int(valid_snapshot_count),
            dropped_partial_final_block=bool(truncated_final_block),
            damaged_final_block_reason=damaged_final_block_reason,
            notes=tuple(notes),
        )

    @dataclass(slots=True)
    class StreamingSnapshotIterator:
        document: "HeliosDocument"
        header: HeliosHeader
        base_status: RunStatusInfo
        _blocks: tuple[Any, ...]
        _position: int = 0
        _valid_snapshot_count: int = 0
        _last_valid_snapshot: Snapshot | None = None
        _run_status: RunStatusInfo | None = None
        _damaged_final_block_reason: str | None = None
        _finished: bool = False

        @property
        def run_status(self) -> RunStatusInfo | None:
            return self._run_status

        def __iter__(self) -> "HeliosDocument.StreamingSnapshotIterator":
            return self

        def __next__(self) -> Snapshot:
            while self._position < len(self._blocks):
                block_offset = self._position
                block = self._blocks[block_offset]
                self._position += 1
                try:
                    snapshot = self.document.snapshot_parser.parse(self.document.buffer.slice(block.span), self.header)
                except Exception as exc:
                    if block_offset == len(self._blocks) - 1 and self._valid_snapshot_count > 0:
                        self._damaged_final_block_reason = str(exc)
                        self._finish(truncated_final_block=True)
                        raise StopIteration
                    raise ValueError(
                        f"Failed to parse HELIOS snapshot block {int(block.index)} from {self.document.buffer.source}: {exc}"
                    ) from exc
                self._valid_snapshot_count += 1
                self._last_valid_snapshot = snapshot
                return snapshot
            self._finish(truncated_final_block=False)
            raise StopIteration

        def _finish(self, *, truncated_final_block: bool) -> None:
            if self._finished:
                return
            self._finished = True
            self._run_status = self.document._finalize_stream_run_status(
                header=self.header,
                base_status=self.base_status,
                valid_snapshot_count=self._valid_snapshot_count,
                last_valid_snapshot=self._last_valid_snapshot,
                truncated_final_block=truncated_final_block,
                damaged_final_block_reason=self._damaged_final_block_reason,
            )
            self.document._run_status = self._run_status
            self.document._apply_run_status_to_header(self.header, self._run_status)

    def inspect(self) -> HeliosHeader:
        if self._header is None:
            self._header = self.header_parser.parse(self.buffer.source, self.buffer.slice(self.index.header_span))
            self._apply_run_status_to_header(self._header, self._base_run_status(self._header))
        return self._header

    def preview(self) -> HeliosPreview:
        header = self.inspect()
        snapshot = None
        if self.index.snapshot_blocks:
            snapshot = self.snapshot_parser.parse(self.buffer.slice(self.index.snapshot_blocks[0].span), header)
        return HeliosPreview(source=self.buffer.source, header=header, snapshot=snapshot)

    def iter_snapshots(self, *, header: HeliosHeader | None = None) -> Iterator[Snapshot]:
        active_header = header or self.inspect()
        snapshots, _ = self._load_snapshots(active_header)
        for snapshot in snapshots:
            yield snapshot

    def iter_snapshots_streaming(self, *, header: HeliosHeader | None = None) -> "StreamingSnapshotIterator":
        active_header = header or self.inspect()
        blocks = self.index.snapshot_blocks
        if not blocks:
            raise ValueError(f"No HELIOS snapshots found in {self.buffer.source}.")
        return HeliosDocument.StreamingSnapshotIterator(
            document=self,
            header=active_header,
            base_status=self._base_run_status(active_header),
            _blocks=blocks,
        )

    def parsed_snapshot_count(self, *, header: HeliosHeader | None = None) -> int:
        active_header = header or self.inspect()
        snapshots, _ = self._load_snapshots(active_header)
        return len(snapshots)

    def get_run_status(self, *, header: HeliosHeader | None = None) -> RunStatusInfo:
        active_header = header or self.inspect()
        _, status = self._load_snapshots(active_header)
        return status

    def parse_full(self) -> Simulation:
        header = self.inspect()
        snapshots, run_status = self._load_snapshots(header)
        n_snapshots = len(snapshots)
        if n_snapshots == 0:
            raise ValueError(f"No HELIOS snapshots found in {self.buffer.source}.")
        first_snapshot = snapshots[0]

        field_names = list(first_snapshot.fields.keys())
        field_units = dict(first_snapshot.field_units)
        raw_field_map = dict(first_snapshot.raw_field_map)
        stacked_fields = {name: np.full((n_snapshots, header.n_zones), np.nan, dtype=np.float64) for name in field_names}
        time = {
            "time": np.empty(n_snapshots, dtype=np.float64),
            "cycle": np.empty(n_snapshots, dtype=np.int64),
            "time_step": np.empty(n_snapshots, dtype=np.float64),
            "time_step_control": np.empty(n_snapshots, dtype=object),
        }
        diagnostic_buffers: dict[tuple[str, ...], np.ndarray] = {}
        diagnostic_schema: dict[tuple[str, ...], int | None] = {}
        diagnostic_schema_notes: list[str] = []

        def assign_snapshot(index: int, snapshot: Snapshot) -> None:
            time["time"][index] = snapshot.time
            time["cycle"][index] = snapshot.cycle
            time["time_step"][index] = snapshot.time_step
            time["time_step_control"][index] = snapshot.time_step_control
            for name, values in snapshot.fields.items():
                if name not in stacked_fields:
                    stacked_fields[name] = np.full((n_snapshots, header.n_zones), np.nan, dtype=np.float64)
                    field_names.append(name)
                    field_units[name] = snapshot.field_units.get(name, "")
                    raw_field_map[name] = snapshot.raw_field_map.get(name, name)
                stacked_fields[name][index, :] = values
            for path_key, value in flatten_snapshot_diagnostics(snapshot.diagnostics):
                if path_key not in diagnostic_buffers:
                    width = diagnostic_value_width(value)
                    diagnostic_schema[path_key] = width
                    diagnostic_buffers[path_key] = (
                        np.full((n_snapshots, int(width)), np.nan, dtype=np.float64)
                        if width is not None
                        else np.full(n_snapshots, np.nan, dtype=np.float64)
                    )
                else:
                    current_width = diagnostic_schema[path_key]
                    resolved_width, widened, reason = reconcile_diagnostic_width(current_width, value)
                    if widened and current_width is None and reason == "scalar_to_vector":
                        upgraded = np.full((n_snapshots, int(resolved_width)), np.nan, dtype=np.float64)
                        previous = np.asarray(diagnostic_buffers[path_key], dtype=np.float64).reshape(-1)
                        upgraded[:, 0] = previous
                        diagnostic_buffers[path_key] = upgraded
                        diagnostic_schema[path_key] = resolved_width
                        diagnostic_schema_notes.append(
                            f"Diagnostic {'/'.join(path_key)} widened from scalar to width {int(resolved_width)}; "
                            "earlier scalar values were preserved in column 0 and remaining elements padded with NaN."
                        )
                    elif widened and current_width is not None and resolved_width is not None and int(resolved_width) > int(current_width):
                        upgraded = np.full((n_snapshots, int(resolved_width)), np.nan, dtype=np.float64)
                        upgraded[:, : int(current_width)] = np.asarray(diagnostic_buffers[path_key], dtype=np.float64)
                        diagnostic_buffers[path_key] = upgraded
                        diagnostic_schema[path_key] = resolved_width
                        diagnostic_schema_notes.append(
                            f"Diagnostic {'/'.join(path_key)} widened from width {int(current_width)} to {int(resolved_width)}; "
                            "later snapshots were preserved without truncation."
                        )
                buffer = diagnostic_buffers[path_key]
                width = diagnostic_schema[path_key]
                normalized = normalize_diagnostic_value(value, width)
                if width is not None:
                    buffer[index, :] = normalized
                else:
                    buffer[index] = float(normalized)

        assign_snapshot(0, first_snapshot)
        for index, snapshot in enumerate(snapshots[1:], start=1):
            assign_snapshot(index, snapshot)

        diagnostics: dict[str, Any] = {}
        for path_key, values in diagnostic_buffers.items():
            assign_nested(diagnostics, path_key, values)

        metadata = dict(header.metadata)
        metadata["n_snapshots"] = n_snapshots
        metadata["available_fields"] = list(field_names)
        metadata["raw_field_map"] = raw_field_map
        metadata["run_status"] = self._run_status_metadata(run_status)
        if diagnostic_schema_notes:
            metadata["diagnostic_schema_notes"] = diagnostic_schema_notes

        return Simulation(
            source=header.source,
            grid=header.grid,
            grid_units=header.grid_units,
            regions=header.regions,
            region_units=header.region_units,
            materials=header.materials,
            material_units=header.material_units,
            time=time,
            time_units={"time": "s", "cycle": "", "time_step": "s", "time_step_control": ""},
            fields=stacked_fields,
            field_units=field_units,
            diagnostics=diagnostics,
            input_parameters=header.input_parameters,
            metadata=metadata,
            raw_field_map=raw_field_map,
            run_status=run_status,
        )

    def close(self) -> None:
        self.buffer.close()

    def __enter__(self) -> "HeliosDocument":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_document(
    path: str | Path,
    *,
    access_mode: str,
    indexer: StructuralIndexer,
    header_parser: HeaderSemanticParser,
    snapshot_parser: SnapshotBlockSemanticParser,
) -> HeliosDocument:
    buffer = HeliosBuffer.from_path(path, access_mode=access_mode)
    try:
        index = indexer.build(buffer)
    except Exception:
        buffer.close()
        raise
    return HeliosDocument(
        buffer=buffer,
        index=index,
        header_parser=header_parser,
        snapshot_parser=snapshot_parser,
    )
