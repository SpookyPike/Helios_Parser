from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helios_app.release import RELEASE_VERSION  # noqa: E402


ZIP_PATH = ROOT / "outputs" / "distributables" / f"helios-parser-viewer-v{RELEASE_VERSION}-windows-portable.zip"
ROOT_MARKERS = ("PYTHONPATH",)


def verify_portable_scientific_resources(app_root: Path, source_root: Path = ROOT) -> None:
    """Check that the portable bundle contains the authoritative JSON tables."""

    source_package = source_root / "src" / "helios" / "services" / "derived"
    resource_roots = (
        source_package / "resources",
        source_package / "reference_data" / "plasmon",
    )
    expected = {
        path.relative_to(source_root / "src")
        for root in resource_roots
        for path in root.rglob("*.json")
        if path.is_file()
    }
    if not expected:
        raise RuntimeError("no authoritative packaged scientific JSON resources found")

    for relative_path in sorted(expected):
        source = source_root / "src" / relative_path
        bundled_matches = [
            path
            for path in app_root.rglob(relative_path.name)
            if path.is_file()
            and path.as_posix().endswith(relative_path.as_posix())
        ]
        if len(bundled_matches) != 1:
            raise RuntimeError(
                f"expected one bundled resource {relative_path}, found {len(bundled_matches)}"
            )
        source_digest = hashlib.sha256(source.read_bytes()).digest()
        bundled_digest = hashlib.sha256(bundled_matches[0].read_bytes()).digest()
        if bundled_digest != source_digest:
            raise RuntimeError(f"portable scientific resource hash mismatch: {relative_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a built Windows portable bundle.")
    parser.add_argument("--zip-path", type=Path, default=ZIP_PATH)
    args = parser.parse_args()
    zip_path = args.zip_path.resolve()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    with tempfile.TemporaryDirectory(prefix="helios_windows_portable_") as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        app_root = tmp / f"helios-parser-viewer-v{RELEASE_VERSION}-windows-portable"
        verify_portable_scientific_resources(app_root)
        native_bundle_required = bool(os.environ.get("HELIOS_XCOM_BUNDLE_DIR"))
        env = os.environ.copy()
        env.pop("HELIOS_XCOM_BUNDLE_DIR", None)
        if native_bundle_required:
            manifests = [
                path for path in app_root.rglob("manifest.json")
                if path.parent.name == "windows-x86_64"
                and path.parent.parent.name == "helios_xcom_native"
            ]
            if len(manifests) != 1:
                raise RuntimeError(f"expected one packaged Windows XCOM manifest, found {len(manifests)}")
            native_root = manifests[0].parent
            if (
                not (native_root / "xcom.exe").is_file()
                or not (native_root / "XCOM_NOTICE.txt").is_file()
                or len(list(native_root.glob("MDATX3.*"))) != 100
            ):
                raise RuntimeError("portable XCOM payload is incomplete")
            env["HELIOS_XCOM_EXECUTABLE"] = str(native_root / "xcom.exe")
            env["HELIOS_XCOM_DATA_DIR"] = str(native_root)
        cli = app_root / "helios_to_hdf5.exe"
        gui = app_root / "HeliosParseView.exe"
        sample = app_root / "sample_data" / "5Fe+4.9TW+light.bpf"
        output = tmp / "parsed_from_portable.h5"
        for key in ROOT_MARKERS:
            env.pop(key, None)
        subprocess.run(
            [str(cli), str(sample), str(output), "--compression", "lzf", "--overwrite"],
            cwd=tmp,
            env=env,
            check=True,
            timeout=120,
        )
        if not output.exists() or output.stat().st_size <= 0:
            raise RuntimeError("portable CLI did not create an HDF5 output")
        with h5py.File(output, "r") as handle:
            schema = handle.attrs.get("schema_version")
            field_count = len(handle["fields"].keys())
        if str(schema) != "2.1" or field_count < 20:
            raise RuntimeError(f"unexpected portable output schema={schema!r} fields={field_count}")
        env["QT_QPA_PLATFORM"] = "offscreen"
        process = subprocess.Popen([str(gui), str(output)], cwd=tmp, env=env)
        try:
            time.sleep(8)
            if process.poll() is not None and process.returncode != 0:
                raise RuntimeError(f"portable GUI exited early with {process.returncode}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        print(f"portable parse ok: schema={schema} fields={field_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
