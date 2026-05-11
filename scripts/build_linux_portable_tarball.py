from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helios_app.release import RELEASE_VERSION  # noqa: E402
from helios_parser import write_hdf5  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "distributables"
BUNDLE_NAME = f"helios-parser-viewer-v{RELEASE_VERSION}-linux-portable"
BUNDLE_DIR = OUTPUT_ROOT / BUNDLE_NAME
TARBALL_PATH = OUTPUT_ROOT / f"{BUNDLE_NAME}.tar.gz"


def _ignore_runtime_junk(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name in {"__pycache__", ".pytest_cache"} or name.endswith((".pyc", ".pyo"))}


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _launcher(module: str) -> str:
    return f"""#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${{PYTHONPATH:+:$PYTHONPATH}}"
exec python3 -m {module} "$@"
"""


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    shutil.copytree(SRC, BUNDLE_DIR / "src", ignore=_ignore_runtime_junk)
    shutil.copytree(ROOT / "docs", BUNDLE_DIR / "docs", ignore=_ignore_runtime_junk)
    shutil.copy2(ROOT / "README.md", BUNDLE_DIR / "README.md")
    shutil.copy2(ROOT / "pyproject.toml", BUNDLE_DIR / "pyproject.toml")
    _write_executable(BUNDLE_DIR / "bin" / "helios-parse-view", _launcher("helios_app"))
    _write_executable(BUNDLE_DIR / "bin" / "helios-to-hdf5", _launcher("helios_parser"))
    _write_executable(BUNDLE_DIR / "bin" / "helios-hdf5-viewer", _launcher("helios_viewer"))
    (BUNDLE_DIR / "sample_data").mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / "examples").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "new_data" / "25Cu+1.87TW" / "25Cu+1.87TW.log",
        BUNDLE_DIR / "sample_data" / "25Cu+1.87TW.log",
    )
    bpf_source = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"
    shutil.copy2(bpf_source, BUNDLE_DIR / "sample_data" / "5Fe+4.9TW+light.bpf")
    write_hdf5(
        bpf_source,
        BUNDLE_DIR / "examples" / "5Fe+4.9TW+light_bpf_schema2.h5",
        overwrite=True,
        compression="lzf",
    )
    (BUNDLE_DIR / "README_LINUX_PORTABLE.txt").write_text(
        f"""# HELIOS Parse / View {RELEASE_VERSION} Linux portable bundle

This is a source-runtime portable bundle, not an AppImage. It does not require
an editable checkout and does not run pip, but it expects Python 3.10+ and the
runtime Python packages to be installed by the operating system:

python3, numpy, h5py, matplotlib, pyqtgraph, and PySide6 QtCore/Gui/Widgets/Svg.

Launch from the extracted directory:
- `bin/helios-parse-view`
- `bin/helios-to-hdf5 sample_data/5Fe+4.9TW+light.bpf output.h5`
- `bin/helios-hdf5-viewer examples/5Fe+4.9TW+light_bpf_schema2.h5`

Use the Debian package when you want dependency installation handled by apt.
""",
        encoding="utf-8",
        newline="\n",
    )
    if TARBALL_PATH.exists():
        TARBALL_PATH.unlink()
    with tarfile.open(TARBALL_PATH, "w:gz") as archive:
        archive.add(BUNDLE_DIR, arcname=BUNDLE_NAME)
    print(TARBALL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
