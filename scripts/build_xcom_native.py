#!/usr/bin/env python3
"""Build the bundled XCOM 3.1 runtime payload for the current native platform."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x-com_fallback"))
from build_xcom_fallback_from_xcom import COMPILE_FLAGS, precision_source  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output_directory: Path, *, compiler: str = "gfortran") -> dict[str, object]:
    archive = ROOT / "XCOM.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(f"Bundled XCOM source archive is missing: {archive}")
    output_directory.mkdir(parents=True, exist_ok=True)
    executable_name = "xcom.exe" if os.name == "nt" else "xcom"
    with tempfile.TemporaryDirectory(prefix="helios-xcom-build-") as temporary:
        source_root = Path(temporary)
        with tarfile.open(archive) as bundle:
            members = [item for item in bundle.getmembers()
                       if "/.rsrc/" not in item.name and not item.name.endswith("/.rsrc")]
            bundle.extractall(source_root, members=members, filter="data")
        vendor_directory = source_root / "XCOM"
        source_file = vendor_directory / "XCOM.f"
        patched = precision_source(source_file.read_text(encoding="ascii"))
        source_file.write_text(patched, encoding="ascii", newline="\n")
        staged_executable = source_root / executable_name
        native_flags = [*COMPILE_FLAGS, "-static"]
        command = [compiler, *native_flags, "XCOM.f", "-o", str(staged_executable)]
        compiled = subprocess.run(command, cwd=vendor_directory, capture_output=True,
                                  text=True, timeout=90, check=False)
        if compiled.returncode:
            raise RuntimeError(compiled.stdout + compiled.stderr)
        files = sorted(vendor_directory.glob("MDATX3.*"))
        if len(files) != 100:
            raise ValueError(f"Expected 100 XCOM element files; found {len(files)}.")
        shutil.copyfile(staged_executable, output_directory / executable_name)
        if os.name != "nt":
            (output_directory / executable_name).chmod(0o755)
        for source in files:
            shutil.copyfile(source, output_directory / source.name)
        machine = platform.machine().lower()
        machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
        system = "windows" if os.name == "nt" else platform.system().lower()
        manifest = {
            "engine": "NIST XCOM 3.1",
            "platform": f"{system}-{machine}",
            "compiler": compiler,
            "compiler_flags": native_flags,
            "source_archive_sha256": sha256(archive),
            "precision_source_sha256": hashlib.sha256(patched.encode()).hexdigest(),
            "executable": executable_name,
            "executable_sha256": sha256(output_directory / executable_name),
            "element_data_files": 100,
            "element_data_sha256": {
                item.name: sha256(output_directory / item.name) for item in files
            },
            "energy_range_kev": [1.0, 35.0],
        }
        (output_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (output_directory / "XCOM_NOTICE.txt").write_text(
            "XCOM 3.1, 23 June 1999\n"
            "Copyright 1999, Martin J. Berger.\n"
            "Permission is given to use the program for non-commercial purposes.\n"
            "The bundled executable and element databases are derived from the\n"
            "XCOM source archive. This notice does not grant commercial\n"
            "redistribution rights. Review the applicable NIST distribution\n"
            "terms before distributing a package containing this payload.\n",
            encoding="utf-8",
            newline="\n",
        )
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", default=os.environ.get("FC", "gfortran"))
    args = parser.parse_args()
    result = build(args.output_dir.resolve(), compiler=args.compiler)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "element_data_sha256"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
