"""``helios_diag analyse``: run the derived analysis without the Qt application.

The derived layer imports no Qt, and three existing functions compose into a
headless run: ``load_run_data``, ``make_run_context`` and
``compute_analysis_result``.  This module is the thin adapter that maps a
``--module`` list onto the fields those modules need.

Honest limit: ``load_run_data`` always reads the six core fields (density,
velocity, both temperatures, electron density, mean charge).  ``requested_fields``
bounds only the *optional* ones, so ``--module`` saves the optional set rather
than the core.  Saying otherwise would be a claim the loader does not honour.
"""

from __future__ import annotations

import argparse
from typing import Any

from . import envelope

# Which optional fields each module can use. Anything not listed here runs on the
# core six alone. "radiation" reads the radiation group, not a field.
MODULE_OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {
    "xrd": (),
    "spectroscopy": (),
    "plasmon": ("temperature_radiation",),
    "transmission": (),
    "preheat": ("temperature_radiation",),
    "shock": (),
    "wavefront": (),
}

ALL_MODULES = tuple(MODULE_OPTIONAL_FIELDS)

# Modules the derived backend hides unless HELIOS_DEV_MODE or
# HELIOS_ENABLE_EXPERIMENTAL is set (helios.feature_flags). They are reachable from
# here with --include-experimental rather than by exporting a variable first.
EXPERIMENTAL_MODULES = frozenset({"plasmon", "transmission"})


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="Converted HDF5 run.")
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        metavar="NAME",
        help=f"Module to run. May be repeated. Known: {', '.join(ALL_MODULES)}.",
    )
    parser.add_argument("--snapshot", type=int, default=None, help="Snapshot index. Default: the last snapshot.")
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help=(
            "Also run the modules the backend hides by default (plasmon, transmission). "
            "Equivalent to setting HELIOS_ENABLE_EXPERIMENTAL=1 for this invocation."
        ),
    )
    parser.add_argument(
        "--max-array",
        type=int,
        default=1000,
        help="Arrays larger than this are summarised instead of dumped. Default 1000 elements.",
    )


def _summarize_arrays(value: Any, limit: int) -> Any:
    """Replace large arrays with a description, so the payload stays readable."""
    import numpy as np  # local

    if isinstance(value, dict):
        return {str(key): _summarize_arrays(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_arrays(item, limit) for item in value]
    if isinstance(value, np.ndarray):
        if value.size <= limit:
            return envelope.jsonable(value)
        finite = value[np.isfinite(value)] if value.dtype.kind == "f" else value
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
        }
    return envelope.jsonable(value)


def run(args: argparse.Namespace) -> envelope.Result:
    import dataclasses  # local

    import numpy as np  # local

    from helios.services.derived import analysis as derived_analysis
    from helios.services.derived.common import load_run_data, make_run_context

    requested = [name for name in args.module if name]
    unknown = [name for name in requested if name not in MODULE_OPTIONAL_FIELDS]
    if unknown:
        raise envelope.UsageError(f"unknown module(s): {', '.join(unknown)}; known: {', '.join(ALL_MODULES)}")

    gated = sorted(set(requested) & EXPERIMENTAL_MODULES)
    if gated and not args.include_experimental:
        raise envelope.UsageError(
            f"{', '.join(gated)} is hidden in the production backend; "
            "pass --include-experimental (or set HELIOS_ENABLE_EXPERIMENTAL=1) to run it"
        )
    if args.include_experimental:
        # The backend reads the flag at call time, so this is enough and it does not
        # leak past this process.
        import os  # local

        os.environ.setdefault("HELIOS_ENABLE_EXPERIMENTAL", "1")

    optional: set[str] = set()
    for name in requested:
        optional.update(MODULE_OPTIONAL_FIELDS[name])

    dataset = load_run_data(args.path, requested_fields=optional or None)
    n_snapshots = int(np.asarray(dataset.time_s).size)
    snapshot = n_snapshots - 1 if args.snapshot is None else int(args.snapshot)
    if not 0 <= snapshot < n_snapshots:
        raise ValueError(f"snapshot {snapshot} is outside 0..{n_snapshots - 1}")

    context = make_run_context(dataset, args.path, snapshot_index=snapshot)
    result = derived_analysis.compute_analysis_result(
        dataset,
        context,
        parameters=derived_analysis.DerivedAnalysisParameters(),
        context_key=("helios_diag", str(args.path), snapshot),
        requested_modules=frozenset(requested) or None,
    )

    file_block, notes = envelope.file_block_for_path(args.path)
    notes = list(notes)
    if not requested:
        notes.append("no --module given, so every available module ran")
    notes.append(
        "the six core fields are always loaded; --module bounds only the optional set"
    )
    if gated:
        notes.append(
            f"ran {', '.join(gated)} with the experimental backend enabled by --include-experimental"
        )
    modules = {
        name: _summarize_arrays(dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value, args.max_array)
        for name, value in sorted(dataclasses.asdict(result).items())
    }
    payload = envelope.envelope(
        "analyse",
        file=file_block,
        notes=notes,
        request={"modules": requested or list(ALL_MODULES), "snapshot": snapshot},
        result=modules,
    )
    return envelope.Result(payload)