from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helios_app.release import APP_NAME, RELEASE_DATE, RELEASE_VERSION  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "release"
BUNDLE_NAME = f"helios-parser-viewer-v{RELEASE_VERSION}"
BUNDLE_DIR = OUTPUT_ROOT / BUNDLE_NAME
ZIP_PATH = OUTPUT_ROOT / f"{BUNDLE_NAME}.zip"
RELEASE_ASSETS = ROOT / "release_assets"
VALIDATION_OUTPUTS = ROOT / "outputs" / "validation_outputs"

DIRECTORIES_TO_COPY = (
    "src",
    "docs",
    "x-com_fallback",
)

OUTPUT_DIRECTORIES_TO_COPY: tuple[tuple[str, Path], ...] = ()

SCRIPT_FILES_TO_COPY = (
    "scripts/create_release_bundle.py",
    "scripts/inspect_bpf.py",
    "scripts/plot_bpf_extra_fields.py",
)

FILES_TO_COPY = (
    "README.md",
    "pyproject.toml",
    "app_icon.png",
    "XCOM.tar.gz",
    "5Fe+4.9TW+light.log",
    "25Cu+1.4TW.log",
)

OUTPUT_FILES_TO_COPY: tuple[tuple[str, Path], ...] = ()

IGNORE_PATTERNS = (
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "*.tmp",
    "*.temp",
    "*_old.py",
    "v1_code_health_audit.md",
    "plasmon_xrts_observable.md",
    "derived_plasmon_current.png",
    "derived_transmission_current.png",
)

FORBIDDEN_BUNDLE_MARKERS = (
    "/__pycache__/",
    "/.pytest_cache/",
    "/tests/",
    "/outputs/reports/",
    "/outputs/validation_outputs/",
    "_old.py",
    "v1_code_health_audit.md",
    "plasmon_xrts_observable.md",
    "derived_plasmon_current.png",
    "derived_transmission_current.png",
)

EXAMPLE_FILES = (
    ("5Fe+4.9TW+light_stabilized.h5", ROOT / "5Fe+4.9TW+light_stabilized.h5"),
    ("Cu1e17_cyl_stabilized.h5", ROOT / "Cu1e17_cyl_stabilized.h5"),
    ("Cu_0166_stabilized.h5", ROOT / "Cu_0166_stabilized.h5"),
)


def copy_tree(relative: str) -> None:
    source = ROOT / relative
    target = BUNDLE_DIR / relative
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
    )


def copy_file(relative: str) -> None:
    source = ROOT / relative
    target = BUNDLE_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_output_directory(relative: str, source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing validation output directory: {source}")
    target = BUNDLE_DIR / relative
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
    )


def copy_output_file(relative: str, source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing validation output file: {source}")
    target = BUNDLE_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_examples_readme() -> None:
    text = f"""# Demo files included in {APP_NAME} {RELEASE_VERSION}

These files are included because they are practical for onboarding and small or moderate enough to redistribute in a shareable source bundle.

- `5Fe+4.9TW+light_stabilized.h5`: very small planar example for basic open, Viewer, and quick legacy Shock checks
- `Cu1e17_cyl_stabilized.h5`: cylindrical example for radius-aware viewer semantics
- `Cu_0166_stabilized.h5`: moderate quick-look derived example for Shock, XRD, Spectroscopy, Preheat, and WaveFront comparison

Larger layered advanced-analysis runs are documented with screenshots, but are not bundled here to keep the archive practical.
Experimental Plasmon/XRTS and Transmission GUI panels are hidden in production unless HELIOS_DEV_MODE=1 or HELIOS_ENABLE_EXPERIMENTAL=1 is set.
Release date: {RELEASE_DATE}
"""
    path = BUNDLE_DIR / "examples" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_release_assets() -> None:
    for asset in RELEASE_ASSETS.iterdir():
        if not asset.is_file():
            continue
        shutil.copy2(asset, BUNDLE_DIR / asset.name)


def copy_examples() -> None:
    examples_dir = BUNDLE_DIR / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    for target_name, source in EXAMPLE_FILES:
        if not source.exists():
            raise FileNotFoundError(f"Missing example file: {source}")
        shutil.copy2(source, examples_dir / target_name)
    write_examples_readme()


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(BUNDLE_DIR.rglob("*")):
            archive.write(path, path.relative_to(BUNDLE_DIR.parent))


def assert_clean_bundle() -> None:
    offenders: list[str] = []
    paths = [path.relative_to(BUNDLE_DIR.parent).as_posix() for path in BUNDLE_DIR.rglob("*")]
    with zipfile.ZipFile(ZIP_PATH) as archive:
        paths.extend(archive.namelist())
    for name in paths:
        normalized = "/" + name.replace("\\", "/")
        if any(marker in normalized for marker in FORBIDDEN_BUNDLE_MARKERS):
            offenders.append(name)
    if offenders:
        joined = "\n".join(f"- {name}" for name in sorted(set(offenders))[:50])
        raise RuntimeError(f"Release bundle contains forbidden test, report, cache, or legacy artifacts:\n{joined}")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    for directory in DIRECTORIES_TO_COPY:
        copy_tree(directory)
    for relative in SCRIPT_FILES_TO_COPY:
        copy_file(relative)
    for relative in FILES_TO_COPY:
        copy_file(relative)
    for relative, source in OUTPUT_DIRECTORIES_TO_COPY:
        copy_output_directory(relative, source)
    for relative, source in OUTPUT_FILES_TO_COPY:
        copy_output_file(relative, source)
    copy_release_assets()
    copy_examples()
    make_zip()
    assert_clean_bundle()

    print(BUNDLE_DIR)
    print(ZIP_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
