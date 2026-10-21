from __future__ import annotations

import argparse
import os
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


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "distributables"
ARTIFACT_NAME = f"helios-parser-viewer-v{RELEASE_VERSION}-windows-portable"
PYINSTALLER_SPEC = ROOT / "packaging" / "pyinstaller" / "helios_parser_viewer.spec"


def _copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_readme(portable_dir: Path) -> None:
    native_note = (
        "This build also includes the optional XCOM 3.1 neutral-atom reference "
        "provider (Z=1–100, 1–35 keV).\n"
        if list(portable_dir.rglob("manifest.json"))
        else "The optional native XCOM provider is not included in this build.\n"
    )
    text = f"""# HELIOS Parse / View {RELEASE_VERSION} Windows portable

This folder is a no-install Windows build. It bundles the Python runtime and
the application dependencies needed for normal parser/viewer use.

Launch:
- Double-click `HeliosParseView.exe` for the desktop parser/viewer workflow.
- Use `helios_to_hdf5.exe <input.log-or-bpf> <output.h5>` for command-line conversion.

Writable outputs are created wherever the user chooses in the app or CLI. The
application does not require administrator privileges and does not need a source
checkout or `pip install`.

Sample inputs are in `sample_data/`. A schema-2.1 BPF-derived HDF5 example is
in `examples/`.
{native_note}
"""
    (portable_dir / "README_PORTABLE.txt").write_text(text, encoding="utf-8")


def _copy_samples_and_examples(portable_dir: Path) -> None:
    _copy_file(
        ROOT / "new_data" / "25Cu+1.87TW" / "25Cu+1.87TW.log",
        portable_dir / "sample_data" / "25Cu+1.87TW.log",
    )
    bpf_source = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"
    _copy_file(bpf_source, portable_dir / "sample_data" / "5Fe+4.9TW+light.bpf")
    (portable_dir / "examples").mkdir(parents=True, exist_ok=True)
    write_hdf5(
        bpf_source,
        portable_dir / "examples" / "5Fe+4.9TW+light_bpf_schema2.h5",
        overwrite=True,
        compression="lzf",
    )


def _zip_portable(portable_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(portable_dir.rglob("*")):
            archive.write(path, path.relative_to(portable_dir.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Windows portable bundle.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    work_root = args.work_root.resolve()
    portable_dir = output_root / ARTIFACT_NAME
    zip_path = output_root / f"{ARTIFACT_NAME}.zip"
    pyinstaller_dist = work_root / "dist" / "HeliosParserViewer"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    build_environment = os.environ.copy()
    configured_native_bundle = os.environ.get("HELIOS_XCOM_BUNDLE_DIR")
    if configured_native_bundle:
        native_bundle = Path(configured_native_bundle).resolve()
        if not (native_bundle / "manifest.json").is_file():
            raise FileNotFoundError(f"XCOM bundle manifest is missing: {native_bundle}")
        build_environment["HELIOS_XCOM_BUNDLE_DIR"] = str(native_bundle)
    else:
        build_environment.pop("HELIOS_XCOM_BUNDLE_DIR", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(work_root / "dist"),
            "--workpath",
            str(work_root / "build"),
            str(PYINSTALLER_SPEC),
        ],
        cwd=ROOT,
        env=build_environment,
        check=True,
    )
    if portable_dir.exists():
        shutil.rmtree(portable_dir)
    shutil.copytree(pyinstaller_dist, portable_dir)
    _copy_samples_and_examples(portable_dir)
    _write_readme(portable_dir)
    _zip_portable(portable_dir, zip_path)
    print(portable_dir)
    print(zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
