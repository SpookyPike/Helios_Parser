from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helios_app.release import RELEASE_VERSION  # noqa: E402
from helios_parser import write_hdf5  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "distributables"
ARTIFACT_NAME = f"helios-parser-viewer-v{RELEASE_VERSION}-windows-portable"
PORTABLE_DIR = OUTPUT_ROOT / ARTIFACT_NAME
ZIP_PATH = OUTPUT_ROOT / f"{ARTIFACT_NAME}.zip"
PYINSTALLER_SPEC = ROOT / "packaging" / "pyinstaller" / "helios_parser_viewer.spec"
PYINSTALLER_DIST = ROOT / "dist" / "HeliosParserViewer"


def _copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_readme() -> None:
    text = f"""# HELIOS Parse / View {RELEASE_VERSION} Windows portable

This folder is a no-install Windows build. It bundles the Python runtime and
the application dependencies needed for normal parser/viewer use.

Launch:
- Double-click `HeliosParseView.exe` for the desktop parser/viewer workflow.
- Use `helios_to_hdf5.exe <input.log-or-bpf> <output.h5>` for command-line conversion.

Writable outputs are created wherever the user chooses in the app or CLI. The
application does not require administrator privileges and does not need a source
checkout or `pip install`.

Sample inputs are in `sample_data/`. A schema-2.0 BPF-derived HDF5 example is
in `examples/`.
"""
    (PORTABLE_DIR / "README_PORTABLE.txt").write_text(text, encoding="utf-8")


def _copy_samples_and_examples() -> None:
    _copy_file(
        ROOT / "new_data" / "25Cu+1.87TW" / "25Cu+1.87TW.log",
        PORTABLE_DIR / "sample_data" / "25Cu+1.87TW.log",
    )
    bpf_source = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"
    _copy_file(bpf_source, PORTABLE_DIR / "sample_data" / "5Fe+4.9TW+light.bpf")
    (PORTABLE_DIR / "examples").mkdir(parents=True, exist_ok=True)
    write_hdf5(
        bpf_source,
        PORTABLE_DIR / "examples" / "5Fe+4.9TW+light_bpf_schema2.h5",
        overwrite=True,
        compression="lzf",
    )


def _zip_portable() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PORTABLE_DIR.rglob("*")):
            archive.write(path, path.relative_to(PORTABLE_DIR.parent))


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(PYINSTALLER_SPEC),
        ],
        cwd=ROOT,
        check=True,
    )
    if PORTABLE_DIR.exists():
        shutil.rmtree(PORTABLE_DIR)
    shutil.copytree(PYINSTALLER_DIST, PORTABLE_DIR)
    _copy_samples_and_examples()
    _write_readme()
    _zip_portable()
    print(PORTABLE_DIR)
    print(ZIP_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
