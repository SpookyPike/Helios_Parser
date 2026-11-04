"""``helios_diag``: quick diagnostics over a converted HELIOS run.

Top-level imports are deliberately limited to the standard library, so
``--help`` and ``--version`` answer without pulling numpy or h5py.  Each
subcommand owns its own flags and its own heavy imports.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys

from helios_release import RELEASE_VERSION

from .envelope import UsageError

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNREADABLE = 3
EXIT_WRITE = 6
EXIT_ISSUES = 7
EXIT_INTERRUPTED = 130

# name -> (module, one-line help). The handler signature is
# ``run(args) -> Result``; each module also exposes ``add_arguments(parser)``.
COMMANDS: dict[str, tuple[str, str]] = {
    "inspect": ("helios_diag.inspect", "what simulation is this file, and which fields does it hold"),
    "stats": ("helios_diag.stats", "per-zone or per-snapshot statistics for one field"),
    "query": ("helios_diag.query", "extract a profile, a time trace or a snapshot as numbers"),
    "diff": ("helios_diag.diff", "compare two conversions dataset by dataset"),
    "validate": ("helios_diag.validate", "check a converted file against the writer contract"),
    "analyse": ("helios_diag.analyse", "run the derived analysis modules headlessly"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helios_diag",
        description=(
            "Quick diagnostics over a HELIOS run converted to HDF5. Every subcommand "
            "writes JSON to stdout by default, so the output is machine-readable "
            "without flags."
        ),
        epilog=(
            "Start with 'helios_diag inspect run.h5': it reports the simulation setup, "
            "the laser pulse, the geometry and every available field with its unit."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {RELEASE_VERSION}")
    parser.add_argument("--quiet", action="store_true", help="Log warnings and errors only.")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    for name, (module_name, summary) in COMMANDS.items():
        module = importlib.import_module(module_name)
        subparser = subparsers.add_parser(name, help=summary, description=summary)
        subparser.add_argument("--format", choices=("json", "table", "csv"), default="json", help="Output format.")
        subparser.add_argument("--out", help="Write the output here instead of stdout.")
        module.add_arguments(subparser)
    return parser


def _describe(exc: BaseException, path: str | None) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"{path or exc}: no such file or directory"
    return f"{path or 'input'}: cannot read as a converted HELIOS run: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    module = importlib.import_module(COMMANDS[args.command][0])
    path = getattr(args, "path", None)
    try:
        result = module.run(args)
    except KeyboardInterrupt:
        print("helios_diag: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except UsageError as exc:
        # A bad request is not an unreadable file, and a caller branches on the code.
        print(f"helios_diag: {exc}", file=sys.stderr)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "command": args.command,
                        "tool_version": RELEASE_VERSION,
                        "path": path,
                        "status": "error",
                        "error_kind": "usage",
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
        return EXIT_USAGE
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"helios_diag: {_describe(exc, path)}", file=sys.stderr)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "command": args.command,
                        "tool_version": RELEASE_VERSION,
                        "path": path,
                        "status": "error",
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
        return EXIT_UNREADABLE

    from helios_diag.formatting import emit  # local: keeps --help light

    try:
        emit(result.payload, rows=result.rows, columns=result.columns, fmt=args.format, out=args.out)
    except OSError as exc:
        print(f"helios_diag: cannot write {args.out}: {exc}", file=sys.stderr)
        return EXIT_WRITE
    return result.exit_code

if __name__ == "__main__":
    raise SystemExit(main())
