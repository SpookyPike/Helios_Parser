# Packaging and Distributable Artifacts

HELIOS Parse / View ships three release-oriented artifact types:

- Windows portable ZIP: no installer, bundled Python runtime, double-click GUI.
- Debian package: apt-managed runtime dependencies, system launchers.
- Linux portable tarball: source-runtime fallback for Linux systems where an
  AppImage is not built.

Five console scripts are installed. The desktop shell is `helios_parse_view`
(`python -m helios_app`), the HDF5-only viewer is `helios_hdf5_viewer`
(`python -m helios_viewer`), and the command line is three scripts:

| script | module | what it does |
|---|---|---|
| `helios_to_hdf5` | `helios_cli` | convert `.log`, `.bpf` or `.exo` to HDF5 |
| `helios_diag` | `helios_diag.cli` | inspect, stats, query and analyse a run (converted or raw); diff and validate a converted run |
| `helios_plot_radiation` | `helios_viewer.radiation_plot` | radiation figures; needs the `plot` extra |

`helios_to_hdf5` and `helios_diag` live in top-level modules on purpose: importing
anything under `helios_parser` or `helios_app` runs that package's `__init__`, which
pulls numpy, h5py or Qt, and `--version` would cost a quarter of a second instead of
30 ms. `python -m helios_parser` still works and re-exports the same entry point.
See [the command line reference](cli.md).

## v2.2.0 availability

The v2.2.0 release ships the source archive; GitHub generates the source ZIP and
tarball for the tag. The Debian package, Linux portable tarball and Windows
portable ZIP are built from the same tree with the commands below and are not
attached to this release. Public packages include the precomputed cold-XCOM
tables. They do not include a compiled native XCOM runtime or the vendor source
archive; native payload bundling remains opt-in pending confirmation of
redistribution terms.

## The native table parser

`native/table_parser` is a Rust `cdylib` that parses one LOG field table in a single
pass. It is optional: `helios_parser/native_table.py` falls back to the existing
`np.fromstring` path when the library is absent, so a source checkout needs no Rust
toolchain and an unsupported platform loses speed rather than function.

`scripts/build_native_table.py` builds it with the toolchain pinned in
`native/table_parser/rust-toolchain.toml` and copies the artifact to
`src/helios_parser/libhelios_native.<ext>`. The Debian, Linux portable, Windows portable
and source-bundle scripts call it before copying `src`, so a release either ships the
library or fails to build. The PyInstaller spec adds it to `NATIVE_BINARIES`.

The two shipped crates are licence-compatible with the rest of the package:
`fast-float` is MIT OR Apache-2.0, `memchr` is Unlicense OR MIT.

## Maintainer Build Requirements

A fresh virtual environment needs the build backend before an editable install:

```bash
python3 -m venv venv
venv/bin/pip install --upgrade pip setuptools
venv/bin/pip install -e .
```

Without `setuptools`, `pip install -e .` fails with
`Cannot import build backend 'setuptools.build_meta'` while the previous editable
install keeps working through its `.pth`. The result is a stale
`helios_parser-<old>.dist-info`: `pip show helios-parser` reports the old version
and `helios_to_hdf5 --version` reports the new one, on the same interpreter.

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
outputs/distributables/helios-parser-viewer-v2.2.0-windows-portable.zip
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
outputs/distributables/helios-parser-viewer_2.2.0_all.deb
```

Install on Debian/Ubuntu-like systems:

```bash
sudo apt install ./outputs/distributables/helios-parser-viewer_2.2.0_all.deb
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
outputs/distributables/helios-parser-viewer-v2.2.0-linux-portable.tar.gz
```

This tarball is portable in the sense that it does not require an editable
checkout and does not run pip, but it expects the OS to provide Python 3.10+,
NumPy, h5py, scipy, matplotlib, pyqtgraph, and PySide6. Use the Debian package
when you want apt to install those dependencies.

`scipy` is a required runtime dependency from 2.0.0 and is bundled in the
PyInstaller build rather than excluded. Without it the plasma dispersion function
falls back to a complex quadrature that is about 280 times slower, so a build that
drops it looks correct and is slow. The About dialog names the active backend.
