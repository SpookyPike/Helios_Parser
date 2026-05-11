from __future__ import annotations

import gzip
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helios_app.release import APP_NAME, RELEASE_DATE, RELEASE_VERSION  # noqa: E402
from helios_parser import write_hdf5  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "distributables"
PACKAGE_NAME = "helios-parser-viewer"
DEB_PATH = OUTPUT_ROOT / f"{PACKAGE_NAME}_{RELEASE_VERSION}_all.deb"
INSTALL_ROOT = Path("opt") / PACKAGE_NAME
SYSTEM_BIN = Path("usr") / "bin"


def _ignore_runtime_junk(_: str, names: list[str]) -> set[str]:
    patterns = {"__pycache__", ".pytest_cache", "*.pyc", "*.pyo", "*.tmp", "*.temp"}
    ignored: set[str] = set()
    for name in names:
        if name in {"__pycache__", ".pytest_cache"}:
            ignored.add(name)
        if name.endswith((".pyc", ".pyo", ".tmp", ".temp")):
            ignored.add(name)
        if any(Path(name).match(pattern) for pattern in patterns):
            ignored.add(name)
    return ignored


def _copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, ignore=_ignore_runtime_junk)


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _launcher(module: str) -> str:
    return f"""#!/usr/bin/env sh
set -eu
export PYTHONPATH="/opt/{PACKAGE_NAME}/src${{PYTHONPATH:+:$PYTHONPATH}}"
exec python3 -m {module} "$@"
"""


def _populate_data_tree(data_root: Path) -> None:
    app_root = data_root / INSTALL_ROOT
    _copy_tree(SRC, app_root / "src")
    _copy_tree(ROOT / "docs", app_root / "docs")
    shutil.copy2(ROOT / "README.md", app_root / "README.md")
    shutil.copy2(ROOT / "pyproject.toml", app_root / "pyproject.toml")

    sample_dir = app_root / "sample_data"
    sample_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "new_data" / "25Cu+1.87TW" / "25Cu+1.87TW.log", sample_dir / "25Cu+1.87TW.log")
    bpf_source = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"
    shutil.copy2(bpf_source, sample_dir / "5Fe+4.9TW+light.bpf")
    (app_root / "examples").mkdir(parents=True, exist_ok=True)
    write_hdf5(
        bpf_source,
        app_root / "examples" / "5Fe+4.9TW+light_bpf_schema2.h5",
        overwrite=True,
        compression="lzf",
    )

    _write_executable(data_root / SYSTEM_BIN / "helios-parse-view", _launcher("helios_app"))
    _write_executable(data_root / SYSTEM_BIN / "helios-to-hdf5", _launcher("helios_parser"))
    _write_executable(data_root / SYSTEM_BIN / "helios-hdf5-viewer", _launcher("helios_viewer"))

    desktop = data_root / "usr" / "share" / "applications" / "helios-parse-view.desktop"
    desktop.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text(
        """[Desktop Entry]
Type=Application
Name=HELIOS Parse / View
Comment=Parse and inspect HELIOS LOG/BPF/HDF5 data
Exec=helios-parse-view %f
Icon=helios-parse-view
Terminal=false
Categories=Science;Education;
MimeType=application/x-hdf;
""",
        encoding="utf-8",
        newline="\n",
    )
    icon_target = data_root / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps" / "helios-parse-view.png"
    icon_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "helios_viewer" / "assets" / "app_icon.png", icon_target)

    doc_dir = data_root / "usr" / "share" / "doc" / PACKAGE_NAME
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "README.Debian").write_text(
        f"""{APP_NAME} {RELEASE_VERSION}

This package installs the HELIOS parser/viewer Python source under
/opt/{PACKAGE_NAME} and exposes system launchers in /usr/bin.

Runtime dependencies are expected from Debian/Ubuntu packages. No editable
source checkout or end-user pip install is required.

Launch:
- helios-parse-view
- helios-to-hdf5 INPUT OUTPUT
- helios-hdf5-viewer FILE

Release date: {RELEASE_DATE}
""",
        encoding="utf-8",
        newline="\n",
    )
    with gzip.open(doc_dir / "changelog.gz", "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"""{PACKAGE_NAME} ({RELEASE_VERSION}) unstable; urgency=medium

  * Package HELIOS Parse / View desktop runtime and CLI launchers.

 -- Dmitrii Bespalov <maintainer@example.invalid>  {RELEASE_DATE}
"""
        )


def _control_text() -> str:
    return f"""Package: {PACKAGE_NAME}
Version: {RELEASE_VERSION}
Section: science
Priority: optional
Architecture: all
Maintainer: Dmitrii Bespalov <maintainer@example.invalid>
Depends: python3 (>= 3.10), python3-numpy, python3-h5py, python3-matplotlib, python3-pyqtgraph, python3-pyside6.qtcore, python3-pyside6.qtgui, python3-pyside6.qtwidgets, python3-pyside6.qtsvg
Description: HELIOS parser and scientific HDF5 viewer
 {APP_NAME} is a desktop parser/viewer workflow for HELIOS LOG/BPF data.
 It converts representative HELIOS outputs to schema-versioned HDF5 and
 opens them in a PySide-based scientific viewer.
"""


def _make_tar_gz(source: Path, target: Path, arcname: str = ".") -> None:
    with tarfile.open(target, "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            archive.add(path, arcname=Path(arcname) / path.relative_to(source), recursive=False)


def _write_ar_member(handle, name: str, data: bytes, mode: int = 0o100644) -> None:
    encoded_name = (name + "/").encode("ascii")
    header = (
        encoded_name.ljust(16, b" ")
        + b"0".ljust(12, b" ")
        + b"0".ljust(6, b" ")
        + b"0".ljust(6, b" ")
        + oct(mode)[2:].encode("ascii").ljust(8, b" ")
        + str(len(data)).encode("ascii").ljust(10, b" ")
        + b"`\n"
    )
    handle.write(header)
    handle.write(data)
    if len(data) % 2:
        handle.write(b"\n")


def _make_deb(control_tar: Path, data_tar: Path) -> None:
    if DEB_PATH.exists():
        DEB_PATH.unlink()
    with DEB_PATH.open("wb") as handle:
        handle.write(b"!<arch>\n")
        _write_ar_member(handle, "debian-binary", b"2.0\n")
        _write_ar_member(handle, "control.tar.gz", control_tar.read_bytes())
        _write_ar_member(handle, "data.tar.gz", data_tar.read_bytes())


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="helios_deb_") as tmpdir:
        tmp = Path(tmpdir)
        data_root = tmp / "data"
        control_root = tmp / "control"
        data_root.mkdir()
        control_root.mkdir()
        _populate_data_tree(data_root)
        (control_root / "control").write_text(_control_text(), encoding="utf-8", newline="\n")
        data_tar = tmp / "data.tar.gz"
        control_tar = tmp / "control.tar.gz"
        _make_tar_gz(data_root, data_tar)
        _make_tar_gz(control_root, control_tar)
        _make_deb(control_tar, data_tar)
    print(DEB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
