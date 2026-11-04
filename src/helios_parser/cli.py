"""Backwards-compatible entry point for the HELIOS converter.

The implementation lives in the top-level ``helios_cli`` module so that
``--help`` and ``--version`` do not pay for this package's ``__init__``, which
imports numpy and h5py. This module stays so ``python -m helios_parser`` and the
existing CLI tests keep working.
"""

from __future__ import annotations

from helios_cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())