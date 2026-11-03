# HELIOS Parse / View

HELIOS Parse / View is the current cross-platform desktop workflow in this
repository for HELIOS post-processing.

- Release: `1.4.0`
- Release date: `2026-11-03`
- Code developed by `Dmitrii Bespalov` at `European XFEL`

Top-level application modes:

- `Parser`: preview HELIOS `.log` / `.bpf` / `.exo` files and convert them to stabilized HDF5
- `Viewer`: inspect stabilized HDF5 runs with explicit center/edge coordinate semantics
- `Derived / Analysis`: fast legacy Shock plus lazy WaveFront, interface-event, Preheat, XRD, and Spectroscopy diagnostics

## What the project does

This is no longer a minimal parser prototype. The current tree includes:

- explicit center/edge coordinate semantics across parser, HDF5, runtime, viewer, and derived analysis
- streaming HDF5 conversion with partial-final-block protection and run-status metadata
- BPF-primary full-data parsing with LOG metadata/fallback support, and direct reading of the Exodus II (`.exo`) plot output that HELIOS 12 writes by default
- self-describing HDF5/H5D field metadata for dynamic reader/viewer discovery
- bounded caches and shared raw-data reuse between viewer and derived mode
- persistent background task execution with cooperative cancellation and latest-wins UI apply
- a fast legacy `Shock` quick-look path
- lazy/cached `WaveFront` advanced branch tracking
- interface-aware diagnostics and a separate `Preheat` advanced tab
- module-local XRD, Spectroscopy, and Warnings workflows in the production GUI
- experimental Plasmon/XRTS and Transmission workflows retained behind `HELIOS_DEV_MODE=1` or `HELIOS_ENABLE_EXPERIMENTAL=1`
- Transmission uses validated cold-XCOM lookup tables for covered requests and can use an optional native XCOM 3.1 backend for supported materials and energies

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
- [Packaging and distributable artifacts](docs/packaging.md)
- [XCOM tables, accuracy limits, and regeneration](docs/xcom-tables.md)
- [Architecture reference](docs/architecture_extension.md)
- [Plasmon XRTS observable note](docs/plasmon_xrts_observable.md)
- [Startup manual](STARTUP_MANUAL.md)

## Release bundle

The shareable Windows/Linux source bundle is generated to:

- `outputs/release/helios-parser-viewer-v1.4.0.zip`

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

## Runtime artifacts

Windows portable, Debian package, and Linux portable fallback builds are
documented in [Packaging and Distributable Artifacts](docs/packaging.md).

```powershell
python scripts\build_windows_portable.py
python scripts\validate_windows_portable.py
python scripts\build_debian_package.py
python scripts\build_linux_portable_tarball.py
```

Expected artifact names:

- `outputs/distributables/helios-parser-viewer-v1.4.0-windows-portable.zip`
- `outputs/distributables/helios-parser-viewer_1.4.0_all.deb`
- `outputs/distributables/helios-parser-viewer-v1.4.0-linux-portable.tar.gz`

The v1.4.0 release provides the source bundle, Debian package, Linux portable tarball, standalone lookup tables, and checksums. Windows users can run the source bundle or build the portable ZIP locally using the instructions in the packaging guide. Public packages omit the vendor XCOM archive and compiled native payload; the cold-XCOM tables remain included and usable.

## Changelog 1.4.0

- the Exodus reader now reads the frequency-resolved radiation output, which it previously dropped. `Write freq-dependent radiation output = 1` adds 18 datasets to the `.exo`; a converted run used to carry none of them and no marker saying whether radiation was off or was discarded.
- published them as two HDF5 groups, not one, because the arrays do not share a time axis: `radiation/` holds the four node-centred spectral arrays on the `time_freq` axis, `flux/` holds the boundary and interface spectra and the power losses on the `time_whole` axis. Flattening them would let a consumer index a boundary spectrum against the spectral axis and get a plausible wrong number. `radiation/hydro_index` records which hydro dump each spectral dump belongs to.
- `radiation/integrated/` carries the ten scalars HELIOS writes per hydro dump: total radiation energy, the loss and flux per boundary, the interface fluxes, net cooling by region, and the electron-to-radiation integral.
- `metadata/radiation_output` is `"on"` or `"off"` on every converted `.exo`, with `n_groups`, `n_freq_snapshots`, `n_boundaries` and `n_regions` when it is on. Absence is now distinguishable from a failed read.
- the four spectral arrays are stored zone-centred, not with HELIOS's raw node dimension. The container writes `n_zones + 2` node rows and the two outer rows are exactly zero on every dump of every reference fixture; the interior rows are the zone centres in Rmin to Rmax order, which two measured identities confirm: the group sum of `energy_density` equals `rad_energy_density[interior]`, and that equals `3 * rad_pressure`. The padding is dropped so the spectral arrays index the same zones as every other field, and a file whose padding rows are not empty raises rather than losing data.
- added `integrate_over_groups`, `spectrum_at` and `spectrum_map` to `helios.services.derived` for the three questions the campaign asks of this data: one integrated number per time, one spectrum at an instant, and the spectrum-versus-time map a contour plot needs. Group integration uses rectangle weights because that reproduces HELIOS's own integrated scalars to 1e-16, while trapezoid weights are off by 1.5 % on the Rmin boundary.
- the spectral axis is never resampled onto the hydro axis. A consumer that wants a spectrum at a hydro time interpolates, with its own documented rule.
- added `tests/test_radiation_output.py` with three real HELIOS 12.0.0 fixtures: 100 zones and one spectral dump for the commit gate, 184 zones over three regions for the interface ordering, and a 26 MB six-dump run behind the slow gate.
- `Freq. resolved incident flux at Rmin` and `at Rmax` are mapped but are identically zero in all three fixtures, so they carry nothing there yet.

## Changelog 1.3.0

- integrated an optional native XCOM 3.1 subprocess backend for supported neutral elements and mixtures up to Z=100
- use the in-process, validated cold-XCOM tables for covered requests; native execution is reserved for supported requests outside the table domain
- added bounded batching, exact-result caching, cancellation, and subprocess cleanup to keep repeated Transmission calculations responsive
- clarified Transmission energy units, native-provider availability, and the scientific limits of neutral-atom data in the interface and docs
- added cross-platform packaging hooks and Windows CI validation; native payload inclusion remains opt-in and is not part of public release assets

## Changelog 1.2.1

- regenerated cold XCOM tables with binary64 arithmetic and adaptive energy sampling
- preserved both limits at absorption edges, correcting large Cu/Fe interpolation errors
- checked independent native-XCOM holdouts with sampled numerical error below 0.01%
- disabled extrapolation by default and cached table fingerprints for repeated calculations
- documented table provenance, precision limits, regeneration, and local validation

## Changelog 1.2.0

- hardened LOG/BPF parsing, partial-record recovery, and atomic HDF5 conversion
- introduced explicit run status, pressure semantics, and source-energy accounting
- improved cache revision safety, bounded memory retention, and demand-driven derived loading
- added independent module availability with clearer missing-input explanations
- improved desktop accessibility, layout, theme handling, and BPF workflows
- added performance regression tooling and reproducible benchmark evidence
- refreshed Linux and Windows packaging, launchers, and scientific resources

## Changelog 1.1.3

- cleaned field-browser labels for normal legacy-schema HDF5 files so fallback metadata no longer appears as `[legacy unknown]`
- brightened field-browser status colors in dark mode so unselected fields remain readable
- rebuilt the Windows portable executable bundle with the viewer labeling/color fix

## Changelog 1.1.2

- added a real Windows portable PyInstaller runtime with GUI and parser CLI executables
- added a Debian package builder with `/opt/helios-parser-viewer` layout and `/usr/bin` launchers
- added a documented Linux portable tarball fallback for environments where AppImage is not built
- documented distributable artifact build, usage, and validation steps
- fixed packaged viewer resource inclusion for SVG toolbar/action icons

## Changelog 1.1.1

- hardened the viewer for rich BPF schema metadata: non-zone 2D fields now render on node, frequency, boundary, or raw BPF axes instead of zone coordinates
- added practical field inventory filters by source, status, and axes, with compatibility aliases hidden by default
- changed expected malformed-input CLI failures to concise parse errors without Python tracebacks
- cleaned startup documentation for portable paths
- made smoke/BPF tests rely on tracked fixtures instead of untracked root-level logs
- refreshed release packaging to include tracked sample inputs and a schema-2.0 BPF-derived HDF5 example

## Running the tests

Two gates, because one number cannot serve both purposes.

**The commit gate: about 25 s.** The Qt and event-loop modules are skipped, so
the portable core runs fast enough to use on every change.

```bash
cd tests && python -m unittest discover -p "test_*.py"
```

**The milestone gate: about 14 min, with a 70 % coverage floor.** Every module
runs in its own process, which is what keeps the suite at minutes rather than
the hour it takes when the Qt modules share one process and pile up event loops.

```bash
python scripts/run_test_suite.py --profile full --coverage
```

It writes per-entry logs, a JSON outcome manifest and `coverage.txt` to the
report directory, and exits 3 when coverage is under the floor. The floor is
`fail_under = 70` in `pyproject.toml`; the measured value on the reference host
is 72.7 %. Raise the floor at a milestone boundary, deliberately: a floor pinned
just under today's number fails on the next commit that adds a module before its
tests, and then it gets lowered rather than met.

Both gates pin to the six P-cores (`taskset -c 0-11`, the SMT siblings included)
and cap the BLAS and OpenMP pools at 12 threads. The reference host is a hybrid
12th-gen Intel; on the E-cores the same suite roughly doubles.

## Changelog 1.3.1

- HELIOS 12.0.0 support: every snapshot of a 12.0.0 log is parsed. The release prints a whole `Time Step Controller` block and a `Run Time:` line between the cycle line and the state table, and a reader that assumes the table follows the cycle line immediately reads **zero** state dumps while still reporting a successful parse.
- added the Exodus II plot-output reader (`helios_to_hdf5 run.exo run.h5`). HELIOS 12 writes `.exo` by default (`Plot file type = 0`) and only writes `.bpf` when that control is changed, so an upgrade silently changes the output format. The reader maps the core hydrodynamic fields onto the same names and units as the log path, so a converted file is interchangeable downstream.
- documented the one non-obvious Exodus mapping: `fluid_velocity` is stored at the **nodes** while every other field is zone-centred, and the zone velocity the log prints is `fluid_velocity[:, 1:]`, the zone's upper node. The two-node mean is 27 % off, which would bias rear-face velocities and therefore breakout times.
- `.exo` is now an accepted input suffix rather than an unsupported-type error.
- added `tests/test_v12_outputs.py` with real 12.0.0 fixtures: a ten-snapshot log prefix and a ten-step Exodus slice cut from the same run, so the log path and the Exodus path are checked against each other rather than against themselves.
- the Exodus container does not record the HELIOS version, so `helios_version` is written empty rather than guessed.
- the test suite is split into a ~25 s commit gate and a ~14 min milestone gate with a 70 % coverage floor; see *Running the tests*.
- fixed five tests that failed only because they resolved paths against the working directory instead of the repository root (`scripts.*` imports, `README.md`, `docs/*.html`). They read as broken tests and were a path miss.

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
- `.exo` (Exodus II, which is NetCDF-4 and therefore HDF5) is read directly: the core hydrodynamic fields are mapped onto the log path's names and units, and the frequency-resolved radiation output is published under `radiation/` and `flux/` when the deck asked for it.
- the spectral radiation arrays live on their own time axis. `radiation/` is sampled on `time_freq` and `flux/` on `time_whole`; `radiation/hydro_index` maps one onto the other, and nothing resamples between them.
- `.exo` files do not record the HELIOS version, so a converted `.exo` carries an empty `helios_version` and `metadata/run_status/state = exodus_container` instead of a claimed completion state.
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
