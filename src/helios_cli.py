"""Command line for the HELIOS converter.

WHY THIS IS TOP-LEVEL.  Importing anything under ``helios_parser`` executes the
package ``__init__``, which pulls numpy and h5py: measured 228 ms, against 10 ms
for a module that imports nothing.  Keeping the entry point here lets ``--help``
and ``--version`` answer in about 25 ms instead of a quarter of a second, which
matters because an agent probes those two flags constantly.

``helios_parser.cli`` re-exports ``main`` and ``build_parser``, so
``python -m helios_parser`` keeps working.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from helios_release import RELEASE_VERSION

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_PARSE = 4
EXIT_EXISTS = 5
EXIT_WRITE = 6
EXIT_INTERRUPTED = 130

SUPPORTED_INPUTS = (".log", ".bpf", ".exo")
INPUT_HELP = "One or more HELIOS inputs: a .log deck output, a .bpf plot file, or a .exo Exodus II plot file."

LOGGER = logging.getLogger("helios_cli")


class _WarningCounter(logging.Handler):
    """Count WARNING and above so the summary can report them."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helios_to_hdf5",
        description=(
            "Convert HELIOS hydrodynamics logs, BPF plot files or Exodus II plot files "
            "to a stabilized HDF5 file."
        ),
        epilog=(
            "Usage forms: 'helios_to_hdf5 INPUT OUTPUT' as before, "
            "'helios_to_hdf5 INPUT -o OUTPUT', or 'helios_to_hdf5 INPUT... --outdir DIR'. "
            "A .log and a .bpf beside each other are read as one run: the log supplies the "
            "metadata and the BPF the bulk fields. An .exo carries both."
        ),
    )
    # One positional list, not two, so both the historical "INPUT OUTPUT" form and
    # a batch "--outdir" form are expressible. _plan_jobs decides which one it is.
    parser.add_argument("paths", nargs="+", type=Path, metavar="PATH", help=INPUT_HELP)
    parser.add_argument("-o", "--output", type=Path, help="Output HDF5 path for a single input.")
    parser.add_argument("--outdir", type=Path, help="Directory for one output per input, named after the input stem.")
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip"),
        default=None,
        help=(
            "Optional HDF5 compression. Off by default. lzf is the fast option and "
            "enables the shuffle filter; gzip is the smaller one and does not."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write one JSON object per input to stdout. Logs stay on stderr.",
    )
    parser.add_argument("--quiet", action="store_true", help="Log warnings and errors only.")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Write one machine-readable progress line per stage to stderr.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log debug output and re-raise errors.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {RELEASE_VERSION}")
    return parser


def _plan_jobs(
    parser: argparse.ArgumentParser,
    paths: list[Path],
    output: Path | None,
    outdir: Path | None,
) -> list[tuple[Path, Path]]:
    """Resolve the three accepted calling forms into (input, output) pairs."""
    if outdir is not None:
        if output is not None:
            parser.error("--outdir and --output cannot be combined")
        return [(path, outdir / f"{path.stem}.h5") for path in paths]
    if output is not None:
        if len(paths) != 1:
            parser.error("--output takes exactly one input; use --outdir to convert several")
        return [(paths[0], output)]
    if len(paths) != 2:
        parser.error(
            "expected INPUT OUTPUT, or INPUT with --output, or INPUT... with --outdir"
        )
    return [(paths[0], paths[1])]


def _describe_failure(kind: str, source: Path, target: Path, exc: BaseException) -> str:
    """Frame the failure by what actually went wrong, not by the parse alone."""
    if kind == "input_missing":
        return f"{source}: no such file or directory"
    if kind == "output_exists":
        return f"{target} already exists; pass --overwrite to replace it"
    if kind == "parse_error":
        return f"{source}: cannot read as HELIOS output: {exc}"
    if kind == "write_error":
        return f"{target}: cannot write: {exc}"
    if kind == "interrupted":
        return f"{source}: interrupted"
    return f"{source}: {exc}"


def _classify(exc: BaseException) -> tuple[int, str]:
    if isinstance(exc, FileNotFoundError):
        return EXIT_INPUT, "input_missing"
    if isinstance(exc, FileExistsError):
        return EXIT_EXISTS, "output_exists"
    if isinstance(exc, (ValueError, EOFError)):
        return EXIT_PARSE, "parse_error"
    if isinstance(exc, OSError):
        return EXIT_WRITE, "write_error"
    return EXIT_WRITE, "write_error"


def _summarize(source: Path, target: Path, compression: str | None, duration_ms: float, warnings: int) -> dict:
    import h5py  # local: keeps --help and --version free of numpy and h5py

    with h5py.File(target, "r") as handle:
        n_snapshots = int(handle["metadata/n_snapshots"][()]) if "metadata/n_snapshots" in handle else None
        n_zones = int(handle["metadata/n_zones"][()]) if "metadata/n_zones" in handle else None
    return {
        "command": "convert",
        "version": RELEASE_VERSION,
        "input": str(source),
        "output": str(target),
        "bytes": target.stat().st_size,
        "n_snapshots": n_snapshots,
        "n_zones": n_zones,
        "compression": compression,
        "duration_ms": round(duration_ms, 1),
        "warnings": warnings,
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    jobs = _plan_jobs(parser, args.paths, args.output, args.outdir)

    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    counter = _WarningCounter()
    logging.getLogger().addHandler(counter)

    from helios_parser.hdf5 import write_hdf5  # local: 210 ms of numpy and h5py

    progress = None
    if args.progress:

        def progress(payload) -> None:  # noqa: ANN001 - WriteProgress, imported lazily
            print(
                f"progress stage={payload.stage} current={payload.current} "
                f"total={payload.total} fraction={payload.fraction:.3f}",
                file=sys.stderr,
                flush=True,
            )

    exit_code = EXIT_OK
    try:
        for source, target in jobs:
            started = time.perf_counter()
            counter.count = 0
            try:
                write_hdf5(
                    source,
                    target,
                    compression=args.compression,
                    overwrite=args.overwrite,
                    progress_callback=progress,
                )
            except KeyboardInterrupt:
                code, kind = EXIT_INTERRUPTED, "interrupted"
            except BaseException as exc:  # noqa: BLE001 - classified below
                code, kind = _classify(exc)
                if args.verbose:
                    raise
                message = _describe_failure(kind, source, target, exc)
                print(f"helios_to_hdf5: {message}", file=sys.stderr)
                if args.json:
                    print(
                        json.dumps(
                            {
                                "command": "convert",
                                "version": RELEASE_VERSION,
                                "input": str(source),
                                "output": str(target),
                                "status": "error",
                                "error_kind": kind,
                                "error": message,
                            },
                            allow_nan=False,
                        ),
                        flush=True,
                    )
                exit_code = exit_code or code
                continue

            duration_ms = (time.perf_counter() - started) * 1000.0
            if args.json:
                print(
                    json.dumps(
                        _summarize(source, target, args.compression, duration_ms, counter.count),
                        allow_nan=False,
                    ),
                    flush=True,
                )
    finally:
        logging.getLogger().removeHandler(counter)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())