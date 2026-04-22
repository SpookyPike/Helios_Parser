# HELIOS Parse / View

HELIOS Parse / View is the current cross-platform desktop workflow in this
repository for HELIOS post-processing.

- Release: `0.9.1`
- Release date: `2026-03-20`
- Code developed by `Dmitrii Bespalov` at `European XFEL`

Top-level application modes:

- `Parser`: preview HELIOS `.log` files and convert them to stabilized HDF5
- `Viewer`: inspect stabilized HDF5 runs with explicit center/edge coordinate semantics
- `Derived / Analysis`: fast legacy Shock plus lazy advanced WaveFront, interface-event, Preheat, Transmission, XRD, and Plasmon/XRTS diagnostics

## What the project does

This is no longer a minimal parser prototype. The current tree includes:

- explicit center/edge coordinate semantics across parser, HDF5, runtime, viewer, and derived analysis
- streaming HDF5 conversion with partial-final-block protection and run-status metadata
- bounded caches and shared raw-data reuse between viewer and derived mode
- persistent background task execution with cooperative cancellation and latest-wins UI apply
- a fast legacy `Shock` quick-look path
- lazy/cached `WaveFront` advanced branch tracking
- interface-aware diagnostics and a separate `Preheat` advanced tab
- module-local XRD, Plasmon, Transmission, Spectroscopy, and Warnings workflows
- model-aware `Transmission` with explicit `Auto hybrid`, `Thomson`, `Free-free`, `Free-free + Thomson`, and `XCOM` selection
- optional cold-material XCOM seam with canonical material resolution from parsed EOS/opacity metadata
- deterministic precomputed XCOM fallback-table path for Windows or other environments where native XCOM cannot execute
- expanded plasmon/XRTS quick-look modes including RPA, Mermin, static-LFC, and Lindhard-family paths with explicit warnings when a selected state is degenerate, non-collective, or outside the validated quick-look domain
- article-facing plasmon backends including quantum hydrodynamic and finite-T self-consistent STLS controls plus explicit `dielectric`, minimal `XRTS observable`, and material-specific `XRTS article-native Al` comparison layers

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
- [Architecture reference](docs/architecture_extension.md)
- [Plasmon XRTS observable note](docs/plasmon_xrts_observable.md)
- [Startup manual](STARTUP_MANUAL.md)

## Release bundle

The shareable Windows/Linux source bundle is generated to:

- `outputs/release/helios-parse-view-0.9.1.zip`

The bundle includes:

- source code
- updated documentation
- launch/bootstrap scripts for Windows and Linux
- a top-level quick-start file
- selected demo HDF5 examples
- curated observable-benchmark artifacts for the article-facing Al XRTS passes

Use:

```powershell
python scripts\create_release_bundle.py
```

## Changelog 0.9.1

- updated user and developer documentation to match the current parser/viewer/derived feature set
- documented the current `Transmission` workflow as a model-aware panel rather than a Thomson-only quick look
- documented the XCOM seam, canonical material resolution, precomputed fallback table, and current limits of cold-opacity usage
- refreshed release notes, bundle metadata, and shareable archive contents for the current code state

## Important current behavior

- `Shock` is the fast default derived path.
- `WaveFront` is advanced, lazy, and cached.
- `Preheat` is a separate advanced tab and does not live inside legacy Shock.
- `Plasmon` now supports `dielectric`, minimal `XRTS observable`, and material-specific `XRTS article-native Al` modes.
- The article-native Al mode keeps the backend dielectric response fixed and upgrades the observable assembly with explicit free / elastic / bound-core bookkeeping, Al form-factor diagnostics, and inelastic-branch peak extraction after elastic subtraction.
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
