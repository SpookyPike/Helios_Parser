# Packaging and Distributable Artifacts

HELIOS Parse / View ships three release-oriented artifact types:

- Windows portable ZIP: no installer, bundled Python runtime, double-click GUI.
- Debian package: apt-managed runtime dependencies, system launchers.
- Linux portable tarball: source-runtime fallback for Linux systems where an
  AppImage is not built.

The real desktop entry point is `helios_parse_view`, implemented by
`python -m helios_app`. The parser CLI is `helios_to_hdf5`, implemented by
`python -m helios_parser`. The HDF5-only viewer is `helios_hdf5_viewer`,
implemented by `python -m helios_viewer`.

## Maintainer Build Requirements

From a clean checkout on the target build machine:

```bash
python -m pip install --upgrade pip wheel build pyinstaller
python -m pip install -e .[desktop]
```

The Windows portable ZIP must be built on Windows. The Debian package builder is
pure Python and can assemble the package layout on any platform, but install/run
testing should be done on a Debian or Ubuntu system.

## Windows Portable Build

```powershell
python scripts\build_windows_portable.py
python scripts\validate_windows_portable.py
```

Output:

```text
outputs/distributables/helios-parser-viewer-v1.1.2-windows-portable.zip
```

Usage after extraction:

```text
HeliosParseView.exe
helios_to_hdf5.exe sample_data\5Fe+4.9TW+light.bpf output.h5
```

This build bundles Python and the desktop dependencies. Users do not need a
source checkout, editable install, or administrator privileges.

## Debian Package

```bash
python scripts/build_debian_package.py
```

Output:

```text
outputs/distributables/helios-parser-viewer_1.1.2_all.deb
```

Install on Debian/Ubuntu-like systems:

```bash
sudo apt install ./outputs/distributables/helios-parser-viewer_1.1.2_all.deb
```

Launchers:

```bash
helios-parse-view
helios-to-hdf5 input.bpf output.h5
helios-hdf5-viewer output.h5
```

The package installs application files under `/opt/helios-parser-viewer` and
launchers under `/usr/bin`. Python runtime dependencies are declared as Debian
package dependencies instead of being installed with pip.

## Linux Portable Fallback

An AppImage is not produced by the current Windows-based packaging workflow
because AppImage validation requires a Linux build host and Linux-native Qt/Python
runtime collection. The fallback artifact is:

```bash
python scripts/build_linux_portable_tarball.py
```

Output:

```text
outputs/distributables/helios-parser-viewer-v1.1.2-linux-portable.tar.gz
```

This tarball is portable in the sense that it does not require an editable
checkout and does not run pip, but it expects the OS to provide Python 3.10+,
NumPy, h5py, matplotlib, pyqtgraph, and PySide6. Use the Debian package when
you want apt to install those dependencies.
