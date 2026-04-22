# HELIOS Analyzer Validation Report

## Scope

This report covers the Phase 4.1 continuation pass:

- repository/state inspection
- dataset and archive discovery
- Windows-safe archive handling
- application icon extraction/integration
- offline validation of current Derived modules
- optional XCOM adapter inspection
- focused test/documentation hardening

## Repository state detected at start

The repository already contained a working three-mode application:

- Parser mode in `src/helios_app` + `src/helios_parser`
- Viewer mode in `src/helios_viewer`
- Derived / Analysis mode in `src/helios_analysis` + `src/helios/services/derived`

Also present before this continuation:

- top-toolbar mode switching for Parser / Viewer / Derived
- geometry/filtering/weighting controls in Derived mode
- warning severities
- Phase 4 derived services
- docs rebuilt for the three-mode application

This continuation therefore stayed incremental and avoided rewriting working code.

## Datasets and archives discovered

Registry outputs:

- `outputs/reports/dataset_registry.json`
- `outputs/reports/dataset_registry.md`
- `outputs/reports/preferred_hdf5_datasets.md`

Current registry summary after hardening:

| Artifact type | Count |
| --- | ---: |
| `helios_log` | 8 |
| `stabilized_hdf5` | 5 |
| `hdf5` | 11 |
| `derived_hdf5_artifact` | 7 |
| `archive` | 2 |
| `pdf_reference` | 7 |
| `image_asset` | 107 |
| `inspection_error` | 0 |

Representative directly usable datasets:

- `5Fe+4.9TW+light_stabilized.h5` — 500 zones, 8 snapshots, planar
- `Cu_0166_stabilized.h5` — 300 zones, 461 snapshots, planar, `radiation_sink` present
- `10ns+10Si+60Al+15Si+4.27TW_stabilized.h5` — 1300 zones, 701 snapshots, planar, multilayer

Discovered archives:

- `helios_xcom_integration.zip`
- `XCOM.tar.gz`

## Windows/archive handling status

New platform helpers:

- `src/helios/platform/archive_utils.py`
- `src/helios/platform/registry.py`

Validated behavior:

- `.zip` inspection works
- `.tar.gz` inspection works
- pure-Python extraction works
- no WSL/bash-only path assumptions were required
- path traversal is blocked during extraction

Current archive/XCOM bundle counts from validation:

- `helios_xcom_integration.zip`: 273 members
- `XCOM.tar.gz`: 216 members

## Icon asset integration status

Source asset discovered automatically:

- `three_icons.png`

Generated assets:

- `src/helios_viewer/assets/app_icon.png`
- `src/helios_viewer/assets/app_icon.ico`
- `docs/assets/app_icon.png`

Behavior:

- the app now uses the extracted rightmost icon where supported
- icon loading falls back safely if the generated icon assets are missing
- docs now show the real shipping icon and describe its source/replacement path

## Derived validation artifacts generated

Validation output root:

- `outputs/validation_outputs/`

Scripts used:

- `scripts/validate_shock_tracker.py`
- `scripts/validate_xrd_model.py`
- `scripts/validate_plasmon_regime.py`
- `scripts/validate_transmission.py`
- `scripts/validate_spectroscopy.py`
- `scripts/validate_selection_filters.py`
- `scripts/validate_weighting_modes.py`
- `scripts/validate_xcom_adapter.py`

### Shock tracker

Artifacts:

- `outputs/validation_outputs/shock_tracker/*_shock_validation.png`
- `outputs/validation_outputs/shock_tracker/summary.json`

Observed current results:

- `5Fe+4.9TW+light_stabilized.h5`
  - activation snapshot: 1
  - no interface crossings
  - max speed: `8.55e5 cm/s`
- `Cu_0166_stabilized.h5`
  - activation snapshot: 4
  - breakout time inferred at `4.079e-09 s`
  - no interface crossings
  - max speed: `7.25e5 cm/s`
- `10ns+10Si+60Al+15Si+4.27TW_stabilized.h5`
  - activation snapshot: 1
  - no >50-zone jump discontinuities
  - no signed-velocity sign flips
  - region 2->3 crossing at `1.125e-09 s`
  - region 1->2 crossing at `6.211e-09 s`
  - max speed: `1.70e6 cm/s`

### XRD

Artifacts:

- `outputs/validation_outputs/xrd/*_time.png`
- `outputs/validation_outputs/xrd/*_profile.png`
- `outputs/validation_outputs/xrd/summary.json`

Current status:

- time traces and profile plots generated for all three canonical datasets
- region-aware layer counting works
- warning remains explicit that the model is isotropic-compression only

### Plasmon / XRTS

Artifacts:

- `outputs/validation_outputs/plasmon/*_time.png`
- `outputs/validation_outputs/plasmon/*_profile.png`
- `outputs/validation_outputs/plasmon/summary.json`

Current status:

- time traces and profile plots generated for all three canonical datasets
- geometry summary and weighting mode are explicit
- warning severity is preserved for non-collective regimes

### Transmission

Artifacts:

- `outputs/validation_outputs/transmission/*_time.png`
- `outputs/validation_outputs/transmission/*_profile.png`
- `outputs/validation_outputs/transmission/summary.json`

Current status:

- scalar quick-look + time/profile views generated for all three canonical datasets
- per-region budgets present on the multilayer run
- shipping behavior remains Thomson-only and explicitly labeled

### Spectroscopy

Artifacts:

- `outputs/validation_outputs/spectroscopy/*_time.png`
- `outputs/validation_outputs/spectroscopy/*_profile.png`
- `outputs/validation_outputs/spectroscopy/summary.json`

Current status:

- time traces and profile plots generated for all three canonical datasets
- LOS geometry is explicit
- quick-look limitation remains explicit

### Geometry, filtering, and weighting

Artifacts:

- `outputs/validation_outputs/selection_filters/summary.json`
- `outputs/validation_outputs/weighting_modes/summary.json`

Observed behavior:

- front/back propagation and LOS cosine propagate into derived services
- blowoff/low-density filters are recorded explicitly in result metadata
- opposite-velocity exclusion changes selection where expected
- weighting modes remain numerically stable and explicit:
  - `simple_mean`
  - `width`
  - `mass`
  - `electron_density`
  - `electron_column`

## XCOM adapter status

Validation artifact:

- `outputs/validation_outputs/xcom_adapter/summary.json`

Observed status:

- archive discovery: working
- extraction on Windows: working
- wrapper import: working
- real query execution: not working in this environment

Current blocking issue:

- `gfortran is required to build the vendor XCOM backend but was not found`

Important result:

- the future optional backend seam is now validated up to request construction
- derived transmission can already build a structured cold-attenuation request with:
  - zone indices
  - region/material ids
  - density
  - LOS path length
  - photon energies

## Issues found and fixes applied

### Fixed

- dataset registry no longer classifies older logs/HDF5 files as inspection errors just because they lack newer header fields or `/regions`
- docs now cover the real three-mode app plus the continuation work
- docs now include the actual shipping icon
- maintenance docs now describe current archive/validation commands instead of stale phase-specific notes
- focused tests now cover:
  - archive helpers
  - registry fallback behavior
  - icon fallback behavior
  - optional XCOM request seam behavior

### Intentionally not forced into production

- XCOM is still optional and not enabled by default
- no parser/viewer scientific behavior was rewritten in this continuation
- no HDF5 schema changes were made

## Focused tests run

- `python -m unittest discover -s tests -p "test_phase41_continuation.py"`
- `python -m unittest discover -s tests -p "test_app_phase41.py"`
- `python -m unittest discover -s tests -p "test_derived_phase41.py"`
- `python -m unittest discover -s tests -p "test_app_phase4.py"`
- `python -m unittest discover -s tests -p "test_viewer_phase4.py"`
- `python tests\smoke_test.py`

## Documentation updated

- `docs/index.html`
- `docs/future-development.html`
- `docs/maintenance.html`
- `docs/architecture_extension.md`
- `status.md`

These now describe:

- all three top-level modes
- Derived workflow and limitations
- Windows-safe archive handling
- dataset/artifact registry
- icon source and replacement path
- optional XCOM preparation path

## Remaining limitations

- shock tracking is still heuristic, though now characterized better across more datasets
- transmission remains Thomson-only in the shipping app
- the optional XCOM backend still depends on external toolchain availability
- moving-mesh edges remain reconstructed from zone-center geometry
- physical-coordinate traces still use nearest-zone mapping where labeled

## Recommended next steps

1. Keep the current Thomson baseline as the default shipping path.
2. Treat XCOM as an optional backend integration phase only after a reproducible Windows toolchain story exists.
3. Extend registry coverage if more external bundles or calculation types are added.
4. Use the new offline validation scripts as the first debug path before modifying Derived GUI behavior.
