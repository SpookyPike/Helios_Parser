from __future__ import annotations

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


def main() -> int:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(ZIP_PATH)
    with tempfile.TemporaryDirectory(prefix="helios_windows_portable_") as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(tmp)
        app_root = tmp / f"helios-parser-viewer-v{RELEASE_VERSION}-windows-portable"
        cli = app_root / "helios_to_hdf5.exe"
        gui = app_root / "HeliosParseView.exe"
        sample = app_root / "sample_data" / "5Fe+4.9TW+light.bpf"
        output = tmp / "parsed_from_portable.h5"
        env = os.environ.copy()
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
        if str(schema) != "2.0" or field_count < 20:
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
