# HELIOS Parse / View

HELIOS Parse / View is the current cross-platform desktop workflow in this
repository for HELIOS post-processing.

- Release: `1.1.0`
- Release date: `2026-05-11`
- Code developed by `Dmitrii Bespalov` at `European XFEL`

Top-level application modes:

- `Parser`: preview HELIOS `.log` / `.bpf` files and convert them to stabilized HDF5
- `Viewer`: inspect stabilized HDF5 runs with explicit center/edge coordinate semantics
- `Derived / Analysis`: fast legacy Shock plus lazy WaveFront, interface-event, Preheat, XRD, and Spectroscopy diagnostics

## What the project does

This is no longer a minimal parser prototype. The current tree includes:

- explicit center/edge coordinate semantics across parser, HDF5, runtime, viewer, and derived analysis
- streaming HDF5 conversion with partial-final-block protection and run-status metadata
- BPF-primary full-data parsing with LOG metadata/fallback support and optional EXO validation
- self-describing HDF5/H5D field metadata for dynamic reader/viewer discovery
- bounded caches and shared raw-data reuse between viewer and derived mode
- persistent background task execution with cooperative cancellation and latest-wins UI apply
- a fast legacy `Shock` quick-look path
- lazy/cached `WaveFront` advanced branch tracking
- interface-aware diagnostics and a separate `Preheat` advanced tab
- module-local XRD, Spectroscopy, and Warnings workflows in the production GUI
- experimental Plasmon/XRTS and Transmission workflows retained behind `HELIOS_DEV_MODE=1` or `HELIOS_ENABLE_EXPERIMENTAL=1`

## Repository contents

This GitHub repo is set up as a release-oriented source tree. It keeps:

- source code
- scripts
- tests
- documentation
- release assets
- packaged release ZIP archives

It does **not** keep the largest local raw HELIOS logs and heavyweight scratch
HDF5 products, because several of those exceed normal GitHub file-size limits.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[desktop]
python -m helios_app
```

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[desktop]
python -m helios_app
```

## Documentation

Start with the documentation index:

- [Documentation index](docs/index.html)
- [User guide](docs/user-guide.html)
- [Developer guide](docs/developer-guide.html)
- [Developer testing guide](docs/developer-testing.html)
- [Maintenance guide](docs/maintenance.html)
- [Future development](docs/future-development.html)
- [Release notes](docs/release-notes.html)
- [BPF H5D schema](docs/bpf_h5d_schema.md)
- [BPF deep forensic analysis](docs/bpf_deep_forensic_analysis.md)
- [Architecture reference](docs/architecture_extension.md)
- [Plasmon XRTS observable note](docs/plasmon_xrts_observable.md)
- [Startup manual](STARTUP_MANUAL.md)

## Release bundle

The shareable Windows/Linux source bundle is generated to:

- `outputs/release/helios-parser-viewer-v1.1.0.zip`

The bundle includes:

- source code
- updated documentation
- launch/bootstrap scripts for Windows and Linux
- a top-level quick-start file
- selected demo HDF5 examples

Use:

```powershell
python scripts\create_release_bundle.py
```

## Changelog 1.1.0

- added production BPF parsing and BPF-primary HDF5 conversion with transparent `.log` / `.bpf` routing
- added schema-version 2.0 field metadata and dynamic reader field discovery for sparse LOG-only and rich BPF-derived files
- cross-validated overlapping LOG/BPF fields, including velocity convention, radiation pressure/energy, net radiation heating, total pressure, and cumulative laser source handling
- preserved unresolved BPF records under stable raw names with honest metadata and documented confidence levels
- updated release bundle generation and documentation for the production parser/reader pipeline

## Changelog 1.0.1

- production backend now skips hidden Plasmon/XRTS and Transmission module computation unless `HELIOS_DEV_MODE=1` or `HELIOS_ENABLE_EXPERIMENTAL=1`
- added snapshot-row and zone-trace reader/viewer payload paths to avoid full-field materialization for targeted reads
- cleaned the release archive to exclude tests, legacy shadow modules, caches, old reports, and development audit outputs

## Changelog 1.0.0

- production GUI now hides physics pipelines whose archived outputs fail order-of-magnitude sanity: Plasmon/XRTS and Transmission
- added centralized feature flags: `HELIOS_DEV_MODE=1` or `HELIOS_ENABLE_EXPERIMENTAL=1`
- added lightweight physical sanity checks for kept Shock, XRD, and Spectroscopy paths
- refreshed release metadata and bundle name for v1.0.0

## Important current behavior

- `.bpf` is the primary full-data source when present; `.log` remains the setup metadata, diagnostics, and fallback path.
- `.exo` remains optional validation/subset support, not a requirement for full parsing.
- HDF5/H5D outputs are self-describing: readers discover available fields, units, axes, labels, and source provenance from metadata.
- Some BPF records are still preserved as raw `bpf_record_XX` fields when the evidence is insufficient for a physics name.
- `Shock` is the fast default derived path.
- `WaveFront` is advanced, lazy, and cached.
- `Preheat` is a separate advanced tab and does not live inside legacy Shock.
- `Plasmon` and `Transmission` remain available only as experimental/development GUI panels.
- snapshot browsing should stay lightweight even after advanced analysis has been loaded.
- derived outputs remain quick-look tools, not publication-grade forward models.

## Testing

Focused examples:

```powershell
python -m unittest discover -v -s tests -p "test_coordinate_semantics.py"
python -m unittest discover -v -s tests -p "test_run_status.py"
python -m unittest discover -v -s tests -p "test_app_phase412.py"
python -m unittest discover -v -s tests -p "test_derived_backend_optimization.py"
python -m unittest discover -v -s tests -p "test_derived_stability.py"
python -m unittest discover -v -s tests -p "test_wave6_derived.py"
python tests\smoke_test.py
```

See the full [Developer Testing Guide](docs/developer-testing.html) for category-specific guidance, manual QA expectations, and performance checks.

## Current honesty limits

- derived outputs remain quick-look and benchmark tools, not publication-grade forward models
- the article-native Al observable layer is a real material-specific reconstruction seam, but it still lacks full bound-free atomic physics and article-side detector/background assumptions
- another generic dielectric tweak is not the current priority; the remaining XRTS residual is now mainly material-specific
