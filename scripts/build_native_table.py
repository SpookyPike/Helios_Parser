#!/usr/bin/env python3
"""Build the native LOG table parser and install it beside the package.

The crate lives in ``native/table_parser`` and is pinned by
``native/table_parser/rust-toolchain.toml``. The built library is copied to
``src/helios_parser/libhelios_native.<ext>`` where ``native_table.py`` loads it.
Without that file the parser runs its pure-Python fallback, so a source checkout
is usable before anyone has a Rust toolchain.

Usage:
    python3 scripts/build_native_table.py
    python3 scripts/build_native_table.py --check
    python3 scripts/build_native_table.py --output-dir dist/native
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "native" / "table_parser"
INSTALL_DIR = ROOT / "src" / "helios_parser"

# The crate is a cdylib, so the artifact carries the crate name. The loader
# expects one fixed name per platform.
BUILT_NAMES = {
    "linux": "libhelios_table_parser.so",
    "darwin": "libhelios_table_parser.dylib",
    "win32": "helios_table_parser.dll",
}
INSTALLED_NAMES = {
    "linux": "libhelios_native.so",
    "darwin": "libhelios_native.dylib",
    "win32": "helios_native.dll",
}


def _platform_key() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform in {"win32", "cygwin", "msys"}:
        return "win32"
    raise SystemExit(f"Unsupported build platform: {sys.platform}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_channel() -> str:
    for line in (CRATE / "rust-toolchain.toml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("channel"):
            return stripped.split("=", 1)[1].strip().strip('"')
    raise SystemExit("native/table_parser/rust-toolchain.toml has no channel pin")


def _rustc_version() -> str:
    # cwd=CRATE so the rustup proxy selects the toolchain pinned by
    # rust-toolchain.toml, the same one `cargo build` will use.
    result = subprocess.run(
        ["rustc", "--version"], cwd=CRATE, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            "rustc is not available. Install it with "
            "`rustup toolchain install %s`, or ship without the native parser."
            % _pinned_channel()
        )
    return result.stdout.strip()


def _lock_digest() -> str:
    lock = CRATE / "Cargo.lock"
    if not lock.is_file():
        return "missing"
    return _sha256(lock)


def build(*, output_dir: Path | None = None) -> Path:
    platform_key = _platform_key()
    version = _rustc_version()
    pinned = _pinned_channel()
    if pinned not in version:
        raise SystemExit(
            f"rustc {version} does not match the pin {pinned}. "
            "Run `rustup toolchain install %s` or update the pin deliberately." % pinned
        )
    subprocess.run(
        ["cargo", "build", "--release", "--locked"],
        cwd=CRATE,
        check=True,
    )
    built = CRATE / "target" / "release" / BUILT_NAMES[platform_key]
    if not built.is_file():
        raise SystemExit(f"cargo reported success but {built} is missing")
    target_dir = output_dir if output_dir is not None else INSTALL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    installed = target_dir / INSTALLED_NAMES[platform_key]
    shutil.copy2(built, installed)
    print(f"rustc        {version}")
    print(f"cargo.lock   {_lock_digest()}")
    print(f"installed    {installed}")
    return installed


def ensure_native_library() -> Path:
    """Return the installed library, building it when it is missing.

    Release scripts call this before copying ``src`` so a bundle never ships
    without the accelerator by accident.
    """
    platform_key = _platform_key()
    installed = INSTALL_DIR / INSTALLED_NAMES[platform_key]
    if installed.is_file():
        return installed
    return build()


def check() -> int:
    platform_key = _platform_key()
    installed = INSTALL_DIR / INSTALLED_NAMES[platform_key]
    if not installed.is_file():
        print(f"missing      {installed}")
        return 1
    print(f"rustc        {_rustc_version()}")
    print(f"cargo.lock   {_lock_digest()}")
    print(f"installed    {installed} ({installed.stat().st_size} bytes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the installed artifact and toolchain without building.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Copy the library here instead of src/helios_parser.",
    )
    args = parser.parse_args(argv)
    if args.check:
        return check()
    build(output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())