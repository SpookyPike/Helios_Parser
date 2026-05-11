# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path.cwd()
SRC = ROOT / "src"
ICON = SRC / "helios_viewer" / "assets" / "app_icon.ico"

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
    binaries=[],
    datas=[(str(SRC / "helios_viewer" / "assets"), "helios_viewer/assets")],
    hiddenimports=["PySide6.QtSvg", "PySide6.QtOpenGLWidgets"],
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
    binaries=[],
    datas=[],
    hiddenimports=[],
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
