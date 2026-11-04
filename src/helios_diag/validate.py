"""``helios_diag validate``: is this converted file internally consistent.

Exit 0 when clean, 7 when a check fails, 3 when the file cannot be opened.  The
checks are the invariants the test suite already asserts, gathered in one place
so a pipeline can gate on them without running pytest.

A pre-2.0 file is a legitimate input, so checks that only apply to the modern
layout report ``skip`` with a reason rather than ``fail``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import envelope

# What the READER needs to open a run at all (helios_parser.reader.REQUIRED_GROUPS
# plus the time axis). A pre-2.0 file satisfies exactly this and nothing more, and
# it is a legitimate input, so this is the only list a missing entry can fail on.
REQUIRED_ENTRIES = ("grid", "time/time", "fields", "regions", "materials", "diagnostics", "metadata")

# What the 2.0 writer promises to publish. A file without them is still readable,
# so their absence is reported as a skip rather than a failure.
WRITER_ENTRIES = ("metadata/n_snapshots", "metadata/run_status/state")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="Converted HDF5 run to check.")


def _row(check: str, status: str, detail: str) -> dict[str, Any]:
    return {"check": check, "status": status, "detail": detail}


def _required(handle) -> list[dict[str, Any]]:
    missing = [name for name in REQUIRED_ENTRIES if name not in handle]
    if missing:
        return [_row("required_entries", "fail", "missing: " + ", ".join(missing))]
    absent = [name for name in WRITER_ENTRIES if name not in handle]
    if absent:
        return [
            _row("required_entries", "ok", f"{len(REQUIRED_ENTRIES)} entries the reader needs are present"),
            _row(
                "writer_entries",
                "skip",
                "not recorded: " + ", ".join(absent) + " (the file predates 2.0)",
            ),
        ]
    return [_row("required_entries", "ok", f"{len(REQUIRED_ENTRIES)} entries present, writer entries too")]


def _counts(handle) -> list[dict[str, Any]]:
    if "metadata/n_snapshots" in handle:
        declared = int(handle["metadata/n_snapshots"][()])
        actual = int(handle["time/time"].shape[0])
        if declared != actual:
            return [_row("snapshot_count", "fail", f"metadata says {declared}, time/time holds {actual}")]
    actual = int(handle["time/time"].shape[0])
    if actual <= 0:
        return [_row("snapshot_count", "fail", "no snapshots")]
    rows = [_row("snapshot_count", "ok", f"{actual} snapshots")]
    if "metadata/run_status/state" in handle:
        state = handle["metadata/run_status/state"][()]
        if isinstance(state, bytes):
            state = state.decode("utf-8", errors="replace")
        if not str(state):
            rows.append(_row("run_status", "fail", "run_status/state is empty"))
        else:
            rows.append(_row("run_status", "ok", f"state={state}"))
    else:
        rows.append(_row("run_status", "skip", "no run_status group; the reader reports unknown/legacy_hdf5"))
    return rows


def _zero_extent(handle) -> list[dict[str, Any]]:
    """A zero-extent dataset cannot be chunked, so the writer must not have tried."""
    offenders: list[str] = []
    checked = 0
    for name, item in handle.items():
        if not hasattr(item, "shape"):
            continue
        if 0 in item.shape:
            checked += 1
            if item.chunks is not None or item.compression is not None:
                offenders.append(f"{name} (chunks={item.chunks}, compression={item.compression})")
    if offenders:
        return [_row("zero_extent_families", "fail", "chunked or compressed: " + ", ".join(offenders))]
    if not checked:
        return [_row("zero_extent_families", "skip", "this file has no zero-extent family")]
    return [_row("zero_extent_families", "ok", f"{checked} empty families written contiguously")]


def _field_metadata(handle) -> list[dict[str, Any]]:
    if "field_metadata" not in handle:
        return [_row("field_metadata", "skip", "no field_metadata group; the file predates 2.0")]
    declared = set(handle["field_metadata"].keys())
    present = set(handle["fields"].keys())
    orphans = sorted(declared - present)
    if orphans:
        return [_row("field_metadata", "fail", "entries without a dataset: " + ", ".join(orphans))]
    undescribed = sorted(present - declared)
    if undescribed:
        return [_row("field_metadata", "fail", "datasets without an entry: " + ", ".join(undescribed))]
    return [_row("field_metadata", "ok", f"{len(declared)} entries match the field datasets")]


def _radiation_marker(handle) -> list[dict[str, Any]]:
    marker = handle.get("metadata/radiation_output")
    if marker is None:
        return [_row("radiation_marker", "skip", "no radiation_output marker")]
    value = marker[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = str(value).strip().lower()
    if "radiation" not in handle:
        return [_row("radiation_marker", "fail", f"marker says {value!r} but there is no radiation group")]
    empty = [name for name, item in handle["radiation"].items() if hasattr(item, "shape") and item.shape[:1] == (0,)]
    if value == "off" and not empty:
        return [_row("radiation_marker", "fail", "marker says off but every radiation family holds data")]
    if value == "on" and empty:
        return [_row("radiation_marker", "fail", "marker says on but " + ", ".join(empty) + " are empty")]
    return [_row("radiation_marker", "ok", f"marker={value}, {len(empty)} empty families")]


def _chunk_layout(handle) -> list[dict[str, Any]]:
    if "schema_version" not in handle.attrs:
        return [_row("chunk_layout", "skip", "no schema_version; the file predates 2.2")]
    schema = handle.attrs["schema_version"]
    if isinstance(schema, bytes):
        schema = schema.decode("utf-8", errors="replace")
    if str(schema) != "2.2":
        return [_row("chunk_layout", "skip", f"schema {schema} is not 2.2")]
    if "fields" not in handle:
        return [_row("chunk_layout", "fail", "schema 2.2 without a fields group")]
    n_zones = int(handle["metadata/n_zones"][()])
    bad: list[str] = []
    for name, item in handle["fields"].items():
        if not hasattr(item, "chunks") or item.chunks is None:
            continue
        chunks = tuple(int(value) for value in item.chunks)
        if len(chunks) == 2 and chunks[1] != n_zones:
            bad.append(f"{name} zone extent {chunks[1]} != {n_zones}")
        if len(chunks) == 2 and chunks[0] > 32:
            bad.append(f"{name} spans {chunks[0]} snapshots per chunk, above the 32-snapshot flush")
    if bad:
        return [_row("chunk_layout", "fail", "; ".join(bad[:5]))]
    return [_row("chunk_layout", "ok", "field chunks match the schema 2.2 contract")]


def run(args: argparse.Namespace) -> envelope.Result:
    import h5py  # local

    with h5py.File(args.path, "r") as handle:
        rows: list[dict[str, Any]] = []
        rows += _required(handle)
        if not any(row["status"] == "fail" for row in rows):
            rows += _counts(handle)
            rows += _field_metadata(handle)
            rows += _zero_extent(handle)
            rows += _radiation_marker(handle)
            rows += _chunk_layout(handle)

    failed = [row for row in rows if row["status"] == "fail"]
    skipped = [row for row in rows if row["status"] == "skip"]
    try:
        file_block, notes = envelope.file_block_for_path(args.path)
    except (KeyError, OSError, ValueError) as exc:
        # The file is broken enough that the reader will not open it. That is
        # exactly what this command is for, so report it rather than crash.
        file_block = {"path": str(args.path), "bytes": Path(args.path).stat().st_size}
        notes = [f"the reader cannot open this file: {exc}"]
        if not any(row["check"] == "required_entries" for row in rows):
            rows.insert(0, _row("required_entries", "fail", str(exc)))
            failed.append(rows[0])
    if failed:
        notes.append(f"{len(failed)} checks failed")
    if skipped:
        notes.append(f"{len(skipped)} checks did not apply to this file")
    payload = envelope.envelope(
        "validate",
        file=file_block,
        notes=notes,
        request={"checks": len(rows)},
        summary={
            "ok": sum(1 for row in rows if row["status"] == "ok"),
            "fail": len(failed),
            "skip": len(skipped),
        },
    )
    return envelope.Result(
        payload, rows, ("check", "status", "detail"), envelope.EXIT_ISSUES if failed else envelope.EXIT_OK
    )