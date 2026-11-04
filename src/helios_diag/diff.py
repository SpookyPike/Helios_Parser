"""``helios_diag diff``: compare two conversions, dataset by dataset.

This is the check the performance work in this repository did by hand.  It is
what would have caught the 2.0.1 fixture drift and the 2.0.2 shuffle change in
one command, and it doubles as the correctness half of an A/B measurement.

Memory is bounded on purpose: two datasets are in memory at a time, never two
whole files.
"""

from __future__ import annotations

import argparse
from typing import Any

from . import envelope


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("left", help="Reference HDF5 run.")
    parser.add_argument("right", help="Candidate HDF5 run.")
    parser.add_argument("--rtol", type=float, default=1e-9, help="Relative tolerance. Default 1e-9.")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance. Default 0.")
    parser.add_argument(
        "-f",
        "--field",
        action="append",
        default=[],
        metavar="NAME",
        help="Restrict to datasets whose path contains this name. May be repeated.",
    )


def _numeric_paths(path: str) -> dict[str, tuple[int, ...]]:
    import h5py  # local

    found: dict[str, tuple[int, ...]] = {}
    with h5py.File(path, "r") as handle:

        def visit(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset) and item.dtype.kind in "fiu" and item.size:
                found[name] = tuple(int(value) for value in item.shape)

        handle.visititems(visit)
    return found


def _compare(left: str, right: str, name: str, rtol: float, atol: float) -> dict[str, Any]:
    import h5py  # local
    import numpy as np  # local

    with h5py.File(left, "r") as left_handle, h5py.File(right, "r") as right_handle:
        a = np.asarray(left_handle[name][()], dtype=np.float64)
        b = np.asarray(right_handle[name][()], dtype=np.float64)
    if a.shape != b.shape:
        return {
            "dataset": name,
            "shape_left": list(a.shape),
            "shape_right": list(b.shape),
            "status": "shape_mismatch",
            "max_abs_diff": None,
            "max_rel_diff": None,
            "n_differing": None,
        }
    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    differing = int((~close).sum())
    delta = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
    return {
        "dataset": name,
        "shape_left": list(a.shape),
        "shape_right": list(b.shape),
        "status": "ok" if differing == 0 else "differs",
        "max_abs_diff": float(np.nanmax(delta)) if delta.size else 0.0,
        "max_rel_diff": float(np.nanmax(relative)) if relative.size else 0.0,
        "n_differing": differing,
    }


def run(args: argparse.Namespace) -> envelope.Result:
    left_block, left_notes = envelope.file_block_for_path(args.left)
    right_block, right_notes = envelope.file_block_for_path(args.right)

    left_paths = _numeric_paths(args.left)
    right_paths = _numeric_paths(args.right)
    if args.field:
        wanted = [name.lower() for name in args.field]
        keep = {path for path in set(left_paths) | set(right_paths) if any(token in path.lower() for token in wanted)}
        left_paths = {path: shape for path, shape in left_paths.items() if path in keep}
        right_paths = {path: shape for path, shape in right_paths.items() if path in keep}

    notes = left_notes + right_notes
    rows: list[dict[str, Any]] = []
    for name in sorted(set(left_paths) | set(right_paths)):
        if name not in left_paths:
            rows.append({"dataset": name, "shape_left": None, "shape_right": list(right_paths[name]),
                         "status": "only_in_right", "max_abs_diff": None, "max_rel_diff": None, "n_differing": None})
            continue
        if name not in right_paths:
            rows.append({"dataset": name, "shape_left": list(left_paths[name]), "shape_right": None,
                         "status": "only_in_left", "max_abs_diff": None, "max_rel_diff": None, "n_differing": None})
            continue
        rows.append(_compare(args.left, args.right, name, args.rtol, args.atol))

    differing = [row for row in rows if row["status"] != "ok"]
    if differing:
        notes.append(
            f"{len(differing)} of {len(rows)} datasets differ beyond rtol={args.rtol:g}, atol={args.atol:g}"
        )
    payload = envelope.envelope(
        "diff",
        file=left_block,
        notes=notes,
        candidate=right_block,
        request={"rtol": args.rtol, "atol": args.atol, "fields": args.field, "n_datasets": len(rows)},
    )
    columns = ("dataset", "shape_left", "shape_right", "status", "max_abs_diff", "max_rel_diff", "n_differing")
    return envelope.Result(payload, rows, columns, envelope.EXIT_ISSUES if differing else envelope.EXIT_OK)