"""Output writers shared by every ``helios_diag`` subcommand.

JSON is the contract and goes to stdout by default.  ``table`` and ``csv`` are
opt-in and meant for a human reading a terminal.  Rows are streamed rather than
collected, because 1 200 rows cost 0.4 ms to serialize while 100 000 cost 32.5 ms
and a large run must not build the whole list first.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

from .envelope import dump

FORMATS = ("json", "table", "csv")


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _preamble(payload: dict[str, Any]) -> list[str]:
    """The few facts a human needs before reading the numbers."""
    lines: list[str] = []
    command = payload.get("command")
    file_block = payload.get("file") or {}
    if command:
        lines.append(f"# {command}  {payload.get('tool_version', '')}".rstrip())
    for key in ("path", "format", "geometry", "helios_version", "schema_version", "source_file"):
        if file_block.get(key) is not None:
            lines.append(f"# {key}: {file_block[key]}")
    for key in ("n_snapshots", "n_zones", "n_regions", "n_materials"):
        if file_block.get(key) is not None:
            lines.append(f"# {key}: {file_block[key]}")
    field = payload.get("field") or {}
    if field:
        unit = field.get("unit") or "no unit recorded"
        lines.append(f"# field: {field.get('name')}  [{unit}]  label={field.get('label')}")
        if field.get("label_origin") == "inferred":
            lines.append("# field metadata inferred from the dataset name")
    request = payload.get("request") or {}
    for key, value in request.items():
        lines.append(f"# {key}: {value}")
    accounting = payload.get("accounting") or {}
    if accounting:
        lines.append(
            f"# values: {accounting.get('n_used')} used, "
            f"{accounting.get('n_nan')} NaN skipped, {accounting.get('n_values')} total"
        )
    for note in payload.get("notes") or []:
        lines.append(f"# note: {note}")
    if lines:
        lines.append("")
    return lines


def _table(rows: Iterable[dict[str, Any]], columns: Sequence[str], handle: TextIO) -> None:
    widths = [len(name) for name in columns]
    buffered: list[list[str]] = []
    for row in rows:
        cells = [_cell(row.get(name)) for name in columns]
        buffered.append(cells)
        widths = [max(width, len(cell)) for width, cell in zip(widths, cells, strict=True)]
    handle.write("  ".join(name.ljust(width) for name, width in zip(columns, widths, strict=True)) + "\n")
    handle.write("  ".join("-" * width for width in widths) + "\n")
    for cells in buffered:
        handle.write("  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)) + "\n")


def _csv(rows: Iterable[dict[str, Any]], columns: Sequence[str], handle: TextIO) -> None:
    writer = csv.writer(handle)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if row.get(name) is None else row.get(name) for name in columns])


LARGE_ROW_WARNING = 50_000


def emit(
    payload: dict[str, Any],
    *,
    rows: Iterable[dict[str, Any]] | None = None,
    columns: Sequence[str] = (),
    fmt: str = "json",
    out: str | Path | None = None,
) -> None:
    """Write the envelope as JSON, or the rows as a table or CSV."""
    if fmt == "json":
        document = dict(payload)
        if rows is not None:
            materialized = rows if isinstance(rows, list) else list(rows)
            document["rows"] = materialized
            if len(materialized) > LARGE_ROW_WARNING:
                document.setdefault("notes", []).append(
                    f"{len(materialized)} rows serialized at once; use --format csv for large exports"
                )
        text = dump(document) + "\n"
        if out is None:
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            Path(out).write_text(text, encoding="utf-8")
        return

    handle = sys.stdout if out is None else open(out, "w", encoding="utf-8", newline="")
    try:
        # A CSV file is data: a '#' preamble would break every naive reader, so
        # the context lines belong to the table format only.
        if fmt == "table":
            for line in _preamble(payload):
                handle.write(line + "\n")
        if rows is not None:
            if fmt == "csv":
                _csv(rows, columns, handle)
            else:
                _table(rows, columns, handle)
        handle.flush()
    finally:
        if out is not None:
            handle.close()