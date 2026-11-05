"""``helios_diag inspect``: what simulation is this file, and what does it hold.

This is the command a human runs first and the one an agent calls before asking
any question, because it is the only way to learn which fields exist, in what
units, and what the drive was.  No field data is read, so it costs metadata
access only.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from . import envelope, units


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="Converted HDF5 run, or a raw .log, .bpf or HDF5 .exo deck.")

# Fields are listed compactly: the full metadata is available per field, but a
# 29-field run should stay readable.
_FIELD_KEYS = ("name", "unit", "label", "label_origin", "dimensions", "status", "source", "shape")


def _flatten(prefix: str, node: Any, out: dict[str, Any]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            _flatten(child, value, out)
        return
    out[prefix] = units.annotate(prefix, envelope.jsonable(node))


def _physics_block(run: Any) -> tuple[dict[str, Any], list[str]]:
    parameters = run.get_metadata().get("input_parameters")
    notes: list[str] = []
    if not isinstance(parameters, dict) or not parameters:
        notes.append("input_parameters not recorded: the deck setup is unavailable in this file")
        return {"sections": {}, "laser": None}, notes

    sections: dict[str, Any] = {}
    for section, payload in parameters.items():
        flat: dict[str, Any] = {}
        _flatten(str(section), payload, flat)
        sections[str(section)] = flat

    laser = sections.get("laser_source")
    if laser:
        cosine = laser.get("laser_source/incident_angle_cosine", {}).get("value")
        if isinstance(cosine, (int, float)) and -1.0 <= cosine <= 1.0:
            laser["laser_source/incident_angle_deg"] = {
                "value": math.degrees(math.acos(cosine)),
                "unit": "deg",
                "unit_origin": "derived",
                "derived": "acos(incident_angle_cosine)",
            }
    return {"sections": sections, "laser": laser}, notes


def _fields_block(run: Any) -> tuple[list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    notes: list[str] = []
    inferred = 0
    for name in run.list_fields():
        block, field_notes = envelope.field_block(run, name)
        if block["label_origin"] == "inferred":
            inferred += 1
        entries.append({key: block[key] for key in _FIELD_KEYS})
        notes.extend(field_notes)
    if inferred:
        notes.append(
            f"{inferred} of {len(entries)} fields carry no recorded metadata; "
            "their labels come from the dataset names"
        )
    return entries, notes


def _visar_block(run: Any) -> dict[str, Any] | None:
    try:
        readiness = run.check_visar_readiness()
    except Exception:  # noqa: BLE001 - absence of the structure is not an error here
        return None
    return envelope.jsonable(readiness)


def collect(run: Any) -> dict[str, Any]:
    file_block, file_notes = envelope.file_block(run)
    physics, physics_notes = _physics_block(run)
    fields, field_notes = _fields_block(run)
    notes = file_notes + physics_notes + field_notes + list(getattr(run, "notes", ()))
    return envelope.envelope(
        "inspect",
        file=file_block,
        notes=notes,
        physics=physics,
        visar=_visar_block(run),
        fields=fields,
    )


def run(args: argparse.Namespace) -> envelope.Result:
    from helios_parser import HeliosRun  # local: --help must not import numpy

    from .source import detect_input, raw_header

    kind = detect_input(args.path)
    if kind != "hdf5":
        view = raw_header(args.path)
        try:
            return envelope.Result(collect(view))
        finally:
            view.close()

    run_handle = HeliosRun(args.path)
    try:
        return envelope.Result(collect(run_handle))
    finally:
        run_handle.close()