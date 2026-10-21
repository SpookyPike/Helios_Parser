# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os


ROOT = Path.cwd()
SRC = ROOT / "src"
ICON = SRC / "helios_viewer" / "assets" / "app_icon.ico"
NATIVE_BUNDLE = os.environ.get("HELIOS_XCOM_BUNDLE_DIR")
NATIVE_BINARIES = []
NATIVE_DATA = []
NATIVE_HIDDEN_IMPORTS = [
    "helios.services.derived.xcom_native",
    "helios.services.derived.xcom_composition",
]
if NATIVE_BUNDLE:
    native_root = Path(NATIVE_BUNDLE).resolve()
    native_platform = native_root.name
    native_destination = f"helios_xcom_native/{native_platform}"
    native_executable = native_root / ("xcom.exe" if os.name == "nt" else "xcom")
    if (
        not native_executable.is_file()
        or len(list(native_root.glob("MDATX3.*"))) != 100
        or not (native_root / "manifest.json").is_file()
        or not (native_root / "XCOM_NOTICE.txt").is_file()
    ):
        raise RuntimeError(
            "HELIOS_XCOM_BUNDLE_DIR must contain the native executable, manifest, "
            "vendor notice, and all 100 MDATX3 files"
        )
    NATIVE_BINARIES.append((str(native_executable), native_destination))
    NATIVE_DATA.extend((str(item), native_destination) for item in native_root.glob("MDATX3.*"))
    manifest = native_root / "manifest.json"
    NATIVE_DATA.append((str(manifest), native_destination))
    NATIVE_DATA.append((str(native_root / "XCOM_NOTICE.txt"), native_destination))

COMMON_EXCLUDES = [
    "Cryptodome",
    "PIL.ImageTk",
    "fastapi",
    "huggingface_hub",
    "lxml",
    "matplotlib",
    "pandas",
    "pydantic",
    "pytest",
    "scipy",
    "sklearn",
    "sympy",
    "tensorflow",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
]

gui_analysis = Analysis(
    [str(ROOT / "packaging" / "entrypoints" / "helios_parse_view.py")],
    pathex=[str(SRC)],
    binaries=NATIVE_BINARIES,
    datas=[
        (str(SRC / "helios_viewer" / "assets"), "helios_viewer/assets"),
        (str(SRC / "helios" / "services" / "derived" / "resources"), "helios/services/derived/resources"),
        (str(SRC / "helios" / "services" / "derived" / "reference_data" / "plasmon"), "helios/services/derived/reference_data/plasmon"),
        *NATIVE_DATA,
    ],
    hiddenimports=["PySide6.QtSvg", "PySide6.QtOpenGLWidgets", *NATIVE_HIDDEN_IMPORTS],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=COMMON_EXCLUDES,
    noarchive=False,
    optimize=0,
)
gui_pyz = PYZ(gui_analysis.pure)

gui_exe = EXE(
    gui_pyz,
    gui_analysis.scripts,
    [],
    exclude_binaries=True,
    name="HeliosParseView",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
)

cli_analysis = Analysis(
    [str(ROOT / "packaging" / "entrypoints" / "helios_to_hdf5.py")],
    pathex=[str(SRC)],
    binaries=NATIVE_BINARIES,
    datas=[
        (str(SRC / "helios" / "services" / "derived" / "resources"), "helios/services/derived/resources"),
        (str(SRC / "helios" / "services" / "derived" / "reference_data" / "plasmon"), "helios/services/derived/reference_data/plasmon"),
        *NATIVE_DATA,
    ],
    hiddenimports=NATIVE_HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=COMMON_EXCLUDES,
    noarchive=False,
    optimize=0,
)
cli_pyz = PYZ(cli_analysis.pure)

cli_exe = EXE(
    cli_pyz,
    cli_analysis.scripts,
    [],
    exclude_binaries=True,
    name="helios_to_hdf5",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON),
)

coll = COLLECT(
    gui_exe,
    cli_exe,
    gui_analysis.binaries,
    gui_analysis.datas,
    cli_analysis.binaries,
    cli_analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HeliosParserViewer",
)
