"""The JSON envelope every ``helios_diag`` subcommand emits.

WHY THIS MODULE EXISTS.  An output has to be readable by a human or an agent who
has never seen the source simulation, on a fresh conversion and on a file
converted years ago.  The two eras record different things: a modern file
carries ``schema_version``, a ``field_metadata`` group and per-dataset labels,
while a pre-2.0 file carries a ``unit`` attribute and little else.  Every block
below therefore resolves in a fixed order and reports what it could not
determine instead of guessing.
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from helios_release import RELEASE_VERSION

# A field whose recorded status is "legacy" carries no field_metadata entry, so
# its label and dimensions came from the dataset name and shape.
_INFERRED_STATUS = "legacy"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNREADABLE = 3
EXIT_ISSUES = 7


class UsageError(ValueError):
    """A request the CLI cannot honour, as opposed to a file it cannot read."""


class Result(NamedTuple):
    """What a subcommand returns: the envelope, the rows for table/csv, and a verdict."""

    payload: dict[str, Any]
    rows: list[dict[str, Any]] | None = None
    columns: tuple[str, ...] = ()
    exit_code: int = EXIT_OK


def jsonable(value: Any) -> Any:
    """Convert numpy and dataclass values to strict-JSON types.

    Deliberately does not import numpy: this module is loaded when the command
    line builds its subcommand table, and ``--help`` must not pay 140 ms for an
    import it will not use. Numpy arrays and scalars are recognised by their
    ``tolist`` and ``item`` methods instead.

    Non-finite floats become ``None``: Python's ``json`` emits ``NaN`` by default,
    which is not valid JSON and which strict parsers reject.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # numpy arrays, numpy scalars and anything else that knows how to unwrap itself
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    return str(value)


def dump(payload: dict[str, Any]) -> str:
    """Serialize with ``allow_nan=False`` so a leak becomes an error, not bad JSON."""
    return json.dumps(jsonable(payload), allow_nan=False)


def _normalized_geometry(value: Any) -> tuple[str | None, str | None]:
    """Return (lower-case geometry, raw value). The two file eras disagree on case."""
    if value is None:
        return None, None
    raw = str(value)
    if not raw:
        return None, None
    return raw.strip().lower(), raw


def file_block(run: Any) -> tuple[dict[str, Any], list[str]]:
    """Describe the file itself: provenance, versions, size and counts."""
    attributes = run.get_file_attributes()
    metadata = run.get_metadata()
    geometry, geometry_raw = _normalized_geometry(attributes.get("geometry") or metadata.get("geometry"))
    helios_version = attributes.get("helios_version") or metadata.get("helios_version") or None
    source_file = attributes.get("source_file") or metadata.get("source_file") or None
    schema_version = attributes.get("schema_version") or metadata.get("schema_version") or None
    parse_mode = metadata.get("parse_mode") or None
    coordinate_model = metadata.get("coordinate_model") or None

    raw_input = str(attributes.get("format") or "") in {"log", "bpf", "exodus"}
    notes: list[str] = []
    if schema_version is None:
        notes.append(
            "a raw HELIOS deck records no schema_version; the writer contract applies only "
            "after conversion"
            if raw_input
            else "schema_version not recorded: the file predates 2.0 or was written by another tool, "
            "so layout claims cannot be checked against it"
        )
    if parse_mode is None:
        notes.append("parse_mode not recorded")
    if coordinate_model is None:
        notes.append(
            "a raw deck carries no coordinate_model attribute; coordinate semantics come from the "
            "parsed coordinate arrays"
            if raw_input
            else "coordinate_model not recorded; coordinate semantics are reported per accessor"
        )
    if helios_version is None:
        notes.append("helios_version not recorded")

    block = {
        "path": str(run.path),
        "bytes": Path(run.path).stat().st_size,
        "format": attributes.get("format") or None,
        "schema_version": schema_version,
        "geometry": geometry,
        "geometry_raw": geometry_raw,
        "source_file": source_file,
        "helios_version": helios_version,
        "parse_mode": parse_mode,
        "coordinate_model": coordinate_model,
        "run_status": run.get_run_status(),
        "n_snapshots": run.n_snapshots,
        "n_zones": run.n_zones,
        "n_regions": run.n_regions,
        "n_materials": run.n_materials,
    }
    return block, notes


def file_block_for_path(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    """Describe a converted run or a raw deck, then close it again."""
    from .source import detect_input, raw_header  # local: envelope stays importable without numpy

    if detect_input(path) != "hdf5":
        view = raw_header(path)
        try:
            return file_block(view)
        finally:
            view.close()

    from helios_parser import HeliosRun  # local

    run = HeliosRun(path)
    try:
        return file_block(run)
    finally:
        run.close()


def field_block(run: Any, name: str) -> tuple[dict[str, Any], list[str]]:
    """Describe one field: unit, label, dimensions and where each came from."""
    metadata = run.get_field_metadata(name)
    inferred = metadata.status == _INFERRED_STATUS
    notes: list[str] = []
    if inferred:
        notes.append(
            f"field {name}: label and dimensions inferred from the dataset name and shape; "
            "the file records no field_metadata entry"
        )
    if not metadata.unit:
        notes.append(f"field {name}: unit not recorded")

    block = {
        "name": metadata.name,
        "unit": metadata.unit or None,
        "label": metadata.label,
        "label_origin": "inferred" if inferred else "recorded",
        "dimensions": list(metadata.dimensions),
        "source": metadata.source,
        "status": metadata.status,
        "shape": list(metadata.shape),
        "dtype": metadata.dtype,
        "description": metadata.description or None,
        "alias_of": metadata.alias_of,
        "quantity": metadata.quantity or None,
    }
    return block, notes


def envelope(command: str, *, file: dict[str, Any], notes: list[str], **blocks: Any) -> dict[str, Any]:
    """Assemble the standard envelope. Every subcommand uses exactly this shape."""
    payload: dict[str, Any] = {
        "command": command,
        "tool_version": RELEASE_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file": file,
    }
    payload.update(blocks)
    payload["notes"] = notes
    return payload