from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from .hdf5 import write_hdf5
from .parser import HeliosParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HELIOS hydrodynamics logs or BPF plot files to HDF5.")
    parser.add_argument("input", type=Path, help="Path to a HELIOS .log file or .bpf plot file.")
    parser.add_argument("output", type=Path, help="Path to the HDF5 file to create.")
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf"),
        default=None,
        help="Optional HDF5 compression filter.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = HeliosParser()
    try:
        write_hdf5(
            args.input,
            args.output,
            compression=args.compression,
            overwrite=args.overwrite,
            parser=parser,
        )
    except (FileNotFoundError, OSError, ValueError, EOFError) as exc:
        if args.verbose:
            raise
        print(f"Parse error: {args.input}", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        print("Hint: verify that the input is a complete HELIOS .log or .bpf file and that any companion file is not truncated.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
