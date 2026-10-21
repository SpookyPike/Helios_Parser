from __future__ import annotations

import fnmatch
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

from helios_app.release import RELEASE_VERSION  # noqa: E402
from helios_parser import write_hdf5  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "distributables"
BUNDLE_NAME = f"helios-parser-viewer-v{RELEASE_VERSION}-linux-portable"
BUNDLE_DIR = OUTPUT_ROOT / BUNDLE_NAME
TARBALL_PATH = OUTPUT_ROOT / f"{BUNDLE_NAME}.tar.gz"


def _ignore_runtime_junk(_: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in {"__pycache__", ".pytest_cache"}
        or name.endswith((".pyc", ".pyo", ".egg-info"))
    }


def _ignore_internal_doc_artifacts(_: str, names: list[str]) -> set[str]:
    ignored = _ignore_runtime_junk("", names)
    patterns = (
        "*audit*.md",
        "*plan*.md",
        "*review*.md",
        "*validation*.md",
        "*bug*report*.md",
        "helios_performance_*.json",
    )
    for name in names:
        normalized = name.lower()
        if normalized == "performance" or any(fnmatch.fnmatchcase(normalized, pattern) for pattern in patterns):
            ignored.add(name)
    return ignored


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _launcher(module: str) -> str:
    return f"""#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${{PYTHONPATH:+:$PYTHONPATH}}"
if [ -x "$ROOT/helios_xcom_native/linux-x86_64/xcom" ]; then
  export HELIOS_XCOM_EXECUTABLE="$ROOT/helios_xcom_native/linux-x86_64/xcom"
  export HELIOS_XCOM_DATA_DIR="$ROOT/helios_xcom_native/linux-x86_64"
fi
exec python3 -m {module} "$@"
"""


def _populate_bundle(bundle_dir: Path) -> None:
    shutil.copytree(SRC, bundle_dir / "src", ignore=_ignore_runtime_junk)
    shutil.copytree(ROOT / "docs", bundle_dir / "docs", ignore=_ignore_internal_doc_artifacts)
    shutil.copy2(ROOT / "README.md", bundle_dir / "README.md")
    shutil.copy2(ROOT / "pyproject.toml", bundle_dir / "pyproject.toml")
    native_bundle_dir = os.environ.get("HELIOS_XCOM_BUNDLE_DIR")
    if native_bundle_dir:
        source = Path(native_bundle_dir).resolve()
        required = (source / "xcom", source / "manifest.json", source / "XCOM_NOTICE.txt")
        if any(not path.is_file() for path in required) or len(list(source.glob("MDATX3.*"))) != 100:
            raise ValueError(f"XCOM bundle is incomplete: {source}")
        shutil.copytree(source, bundle_dir / "helios_xcom_native" / "linux-x86_64")
    _write_executable(bundle_dir / "bin" / "helios-parse-view", _launcher("helios_app"))
    _write_executable(bundle_dir / "bin" / "helios-to-hdf5", _launcher("helios_parser"))
    _write_executable(bundle_dir / "bin" / "helios-hdf5-viewer", _launcher("helios_viewer"))
    (bundle_dir / "sample_data").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "examples").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "new_data" / "25Cu+1.87TW" / "25Cu+1.87TW.log",
        bundle_dir / "sample_data" / "25Cu+1.87TW.log",
    )
    bpf_source = ROOT / "new_data" / "5Fe+4.9TW+light" / "5Fe+4.9TW+light.bpf"
    shutil.copy2(bpf_source, bundle_dir / "sample_data" / "5Fe+4.9TW+light.bpf")
    write_hdf5(
        bpf_source,
        bundle_dir / "examples" / "5Fe+4.9TW+light_bpf_schema2.h5",
        overwrite=True,
        compression="lzf",
    )
    (bundle_dir / "README_LINUX_PORTABLE.txt").write_text(
        f"""# HELIOS Parse / View {RELEASE_VERSION} Linux portable bundle

This is a source-runtime portable bundle, not an AppImage. It does not require
an editable checkout and does not run pip, but it expects Python 3.10+ and the
runtime Python packages to be installed by the operating system:

python3, numpy, h5py, matplotlib, pyqtgraph, and PySide6 QtCore/Gui/Widgets/Svg.
When an approved XCOM native bundle is supplied at package-build time, it
supports neutral-element coefficients from 1 to 35 keV. The app continues to
use precomputed tables first.

Launch from the extracted directory:
- `bin/helios-parse-view`
- `bin/helios-to-hdf5 sample_data/5Fe+4.9TW+light.bpf output.h5`
- `bin/helios-hdf5-viewer examples/5Fe+4.9TW+light_bpf_schema2.h5`

Use the Debian package when you want dependency installation handled by apt.
Extract the tarball on a filesystem that permits execution.
""",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="helios-linux-portable-") as tmpdir:
        staged_bundle = Path(tmpdir) / BUNDLE_NAME
        _populate_bundle(staged_bundle)
        if TARBALL_PATH.exists():
            TARBALL_PATH.unlink()
        with tarfile.open(TARBALL_PATH, "w:gz") as archive:
            archive.add(staged_bundle, arcname=BUNDLE_NAME)
        if BUNDLE_DIR.exists():
            shutil.rmtree(BUNDLE_DIR)
        shutil.copytree(staged_bundle, BUNDLE_DIR)
    print(TARBALL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
