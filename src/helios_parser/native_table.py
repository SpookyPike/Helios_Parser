"""Native LOG table-block parser, loaded on demand.

``native/table_parser`` builds a small ``cdylib`` that parses one HELIOS field
table in a single pass. This module is the only place that knows about it. When
the library is missing, or the block does not fit the fast-path contract,
:func:`parse_table_block` returns ``None`` and the caller keeps its existing
``np.fromstring`` path. A source checkout without a Rust toolchain is therefore
slower, never broken.
"""
from __future__ import annotations

import ctypes
import hashlib
import sys
from pathlib import Path

import numpy as np

# A corrupt header must not ask for an unbounded allocation. The largest deck in
# the reference set is 1300 zones by 14 columns, so this is generous.
_MAX_ELEMENTS = 50_000_000

_LIBRARY_NAMES = {
    "linux": "libhelios_native.so",
    "darwin": "libhelios_native.dylib",
    "win32": "helios_native.dll",
}


def _library_name() -> str:
    if sys.platform.startswith("linux"):
        return _LIBRARY_NAMES["linux"]
    if sys.platform == "darwin":
        return _LIBRARY_NAMES["darwin"]
    if sys.platform in {"win32", "cygwin", "msys"}:
        return _LIBRARY_NAMES["win32"]
    return _LIBRARY_NAMES["linux"]


def _load() -> ctypes.CDLL | None:
    path = Path(__file__).with_name(_library_name())
    if not path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(path))
    except OSError:
        return None
    library.parse_table_block.restype = ctypes.c_longlong
    library.parse_table_block.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    return library


_LIBRARY = _load()


def available() -> bool:
    """True when the native library loaded and the fast path can be used."""
    return _LIBRARY is not None


_DESCRIPTION: str | None = None


def describe() -> str:
    """One line naming the active path, for the CLI's file block.

    A silent fallback produces correct output at the old speed, so without this
    there is no way to answer "is the native path live on this machine" from the
    command line.
    """
    global _DESCRIPTION
    if _DESCRIPTION is None:
        if _LIBRARY is None:
            _DESCRIPTION = "off (Python fallback, np.fromstring)"
        else:
            path = Path(__file__).with_name(_library_name())
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            except OSError:
                digest = "unreadable"
            _DESCRIPTION = f"on ({path.name}, sha256 {digest})"
    return _DESCRIPTION


def parse_table_block(
    text: str,
    n_zones: int,
    n_cols: int,
) -> tuple[np.ndarray, int] | None:
    """Parse one table block into a ``(n_zones, n_cols)`` matrix.

    Returns ``(matrix, parsed_rows)`` or ``None`` when the native path cannot
    answer: no library, bad shape, or a block the kernel rejects (a row with the
    wrong width, or a token it cannot parse). The caller then uses its Python
    path, so the two implementations never have to agree on malformed input.
    """
    if _LIBRARY is None:
        return None
    zones = int(n_zones)
    columns = int(n_cols)
    if zones <= 0 or columns <= 0 or zones * columns > _MAX_ELEMENTS:
        return None
    buffer = text.encode("utf-8", "replace")
    matrix = np.full((zones, columns), np.nan, dtype=np.float64)
    rows = _LIBRARY.parse_table_block(
        buffer,
        len(buffer),
        zones,
        columns,
        matrix.ctypes.data_as(ctypes.c_void_p),
    )
    if rows < 0:
        return None
    return matrix, int(rows)