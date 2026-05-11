# Startup Manual

This repository ships as a source-based desktop workflow for HELIOS parsing,
viewing, and derived analysis.

## What is included

- source code in [`src/`](src/)
- scripts in [`scripts/`](scripts/)
- documentation in [`docs/`](docs/)
- tests in [`tests/`](tests/)
- release ZIP bundles in [`outputs/release/`](outputs/release/)

The current shareable bundle is:

- `outputs/release/helios-parser-viewer-v1.1.1.zip` after building the current patch release

## Quick start on Windows

1. Open PowerShell in the project root.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install the desktop dependencies and launch:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .[desktop]
python -m helios_app
```

## Quick start on Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[desktop]
python -m helios_app
```

## Release bundle startup

If you unpack the release archive instead of working from the repo:

- Windows: run `Run_HELIOS_Analyzer.ps1`
- Linux: run `run_helios_analyzer.sh`

Those launcher scripts are also stored in [`release_assets/`](release_assets/).

## Where to start reading

- [`README.md`](README.md)
- [`docs/index.html`](docs/index.html)
- [`docs/bpf_h5d_schema.md`](docs/bpf_h5d_schema.md)

## Current packaging scope

The GitHub repo keeps the code, docs, scripts, tests, release assets, and ZIP
bundles. Large raw HELIOS logs and heavyweight local HDF5 scratch files are kept
out of git so the repository remains pushable on standard GitHub limits.
