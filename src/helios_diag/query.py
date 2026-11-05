"""``helios_diag query``: pull numbers out of a run.

Statistics tell you that zone 900 peaked at 4.69e-9 s; this tells you what the
curve actually looked like.  Each form uses the narrowest reader accessor
available, so a single profile costs one row read rather than the whole field.
"""

from __future__ import annotations

import argparse

from . import envelope

MODE_SNAPSHOT = "profile_at_snapshot"
MODE_ZONE = "trace_for_zone"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="Converted HDF5 run, or a raw .log, .bpf or HDF5 .exo deck.")
    parser.add_argument("-f", "--field", required=True, help="Field to read, e.g. density.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=int, metavar="N", help="Profile across zones at this snapshot index.")
    mode.add_argument("--zone", type=int, metavar="Z", help="Time trace for this zone.")
    mode.add_argument("--time", type=float, metavar="SECONDS", help="Profile at the snapshot nearest this time.")
    parser.add_argument("--region", type=int, help="Restrict a profile to one region id.")
    parser.add_argument("--material", type=int, help="Restrict a profile to one material id.")


def _profile(run_handle, field: str, snapshot: int, args) -> envelope.Result:
    import numpy as np  # local

    profile = np.asarray(run_handle.get_snapshot_field(field, snapshot), dtype=np.float64)
    mask = None
    if args.region is not None:
        mask = run_handle.get_region_mask(args.region)
    if args.material is not None:
        material_mask = run_handle.get_material_mask(args.material)
        mask = material_mask if mask is None else (mask & material_mask)
    zones = np.arange(profile.size) if mask is None else np.nonzero(mask)[0]
    if mask is not None:
        profile = profile[mask]
    times = np.asarray(run_handle.get_time(), dtype=np.float64)

    file_block, file_notes = envelope.file_block(run_handle)
    field_block, field_notes = envelope.field_block(run_handle, field)
    notes = list(file_notes) + list(field_notes)
    if args.region is not None or args.material is not None:
        notes.append(f"restricted to {zones.size} zones by region {args.region} / material {args.material}")

    rows = [{"zone": int(zone), "value": float(value)} for zone, value in zip(zones, profile, strict=True)]
    payload = envelope.envelope(
        "query",
        file=file_block,
        notes=notes,
        field=field_block,
        request={
            "mode": MODE_SNAPSHOT,
            "snapshot": snapshot,
            "time_s": float(times[snapshot]),
            "axis": "zone",
        },
        coordinate={
            "kind": "centre" if run_handle.get_static_coordinate(location="center").size == profile.size else "edge",
            "dynamic": bool(run_handle.has_dynamic_coordinate()),
            "unit": run_handle.get_grid_unit("x"),
        },
        accounting={"n_values": int(profile.size), "n_nan": int(np.isnan(profile).sum()), "n_used": int(profile.size)},
    )
    return envelope.Result(payload, rows, ("zone", "value"))


def _trace(run_handle, field: str, zone: int, args) -> envelope.Result:
    import numpy as np  # local

    trace = np.asarray(run_handle.get_time_trace(field, zone), dtype=np.float64)
    times = np.asarray(run_handle.get_time(), dtype=np.float64)
    if trace.size != times.size:
        raise ValueError(f"time trace has {trace.size} samples but the run has {times.size} snapshots")

    file_block, file_notes = envelope.file_block(run_handle)
    field_block, field_notes = envelope.field_block(run_handle, field)
    notes = list(file_notes) + list(field_notes)
    rows = [
        {"snapshot": int(index), "time_s": float(times[index]), "value": float(value)}
        for index, value in enumerate(trace)
    ]
    payload = envelope.envelope(
        "query",
        file=file_block,
        notes=notes,
        field=field_block,
        request={"mode": MODE_ZONE, "zone": int(zone), "axis": "time"},
        accounting={
            "n_values": int(trace.size),
            "n_nan": int(np.isnan(trace).sum()),
            "n_used": int(trace.size),
        },
    )
    return envelope.Result(payload, rows, ("snapshot", "time_s", "value"))


def run(args: argparse.Namespace) -> envelope.Result:
    import numpy as np  # local

    from .source import open_any

    run_handle = open_any(args.path, fields=(args.field,))
    try:
        if args.zone is not None:
            return _trace(run_handle, args.field, int(args.zone), args)
        if args.snapshot is not None:
            return _profile(run_handle, args.field, int(args.snapshot), args)
        times = np.asarray(run_handle.get_time(), dtype=np.float64)
        snapshot = int(np.argmin(np.abs(times - float(args.time))))
        result = _profile(run_handle, args.field, snapshot, args)
        result.payload["notes"].append(
            f"--time {args.time:g} resolved to snapshot {snapshot} at t={times[snapshot]:g} s"
        )
        return result
    finally:
        run_handle.close()