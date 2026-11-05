# HELIOS Parse / View

HELIOS Parse / View is the current cross-platform desktop workflow in this
repository for HELIOS post-processing.

- Release: `2.2.1`
- Release date: `2026-11-04`
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

## Command line

Two commands, both writing JSON to stdout and logs to stderr. Full reference in
[docs/cli.md](docs/cli.md); `just` lists the repository recipes.

```bash
helios_to_hdf5 run.log run.h5 --json          # convert; .log, .bpf and .exo
helios_diag inspect run.h5                    # the deck setup, the laser, the fields
helios_diag stats run.h5 -f density           # per zone, with the peak time
helios_diag query run.h5 -f density --zone 900
helios_diag diff before.h5 after.h5           # exits 7 when they differ
helios_diag validate run.h5                   # exits 7 on a failed check
helios_diag analyse run.h5 --module xrd       # derived modules, no Qt
```

Every output states its own context: geometry, versions, units and their origin, the
request that produced the numbers, the NaN accounting, and a note for anything it could
not determine. A unit is never guessed.

An agent skill ships in [skills/helios-cli](skills/helios-cli/SKILL.md).

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

- `outputs/release/helios-parser-viewer-v2.2.1.zip`

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

- `outputs/distributables/helios-parser-viewer-v2.2.1-windows-portable.zip`
- `outputs/distributables/helios-parser-viewer_2.2.1_all.deb`
- `outputs/distributables/helios-parser-viewer-v2.2.1-linux-portable.tar.gz`

The v2.2.1 release provides the source bundle and checksums. The Debian package, Linux portable tarball and Windows portable ZIP are built from the same tree with the commands above and are not attached to this release. Public packages omit the vendor XCOM archive and compiled native payload; the cold-XCOM tables remain included and usable.

## Changelog 2.2.1

Seven findings from real use of 2.2.0 on the Maxwell cluster
(`docs/findings-2026-11-05-cluster-use.md`).

### Fixed

- `--breakout-fraction` returned the fraction itself while the tracker compares it against an interface index, so 0.2, 0.5 and 0.8 all meant "reach the rear-most cell". It is now a position along the track: 0.5 is the midpoint, 0.95 near the rear face. **The default reading changes**: 1.995e-9 s instead of 4.079e-9 s on `Cu_0166_stabilized.h5`, with 3.877e-9 s at 0.95.
- a front that stops on a material interface reports that interface, its zone and both regions at `caution`, instead of an `info` note that reads like "no breakout".
- the coordinate mismatch that fired on every conversion is `info` with an explanation when the cumulative width grid reproduces the parsed edges, because it is the rounding of the printed positions; a material mismatch stays a `warning`.
- `analyse --help` names the raw formats; `stats` documents that its peak time is not a threshold crossing.

### Added

- every file block names the parser in use: `native table parser: on (libhelios_native.so, sha256 …)` or `off (Python fallback)`.

## Changelog 2.2.0

The diagnostics commands read a deck that was never converted, and the LOG parse
itself got faster. On the 530.7 MB reference deck, 1501 snapshots by 1200 zones,
the full parse drops from 9.03 s to 2.67 s and `write_hdf5` from 8.43 s to 3.93 s,
with a byte-identical output digest on four decks.

### Command line

- `inspect`, `stats`, `query` and `analyse` accept a raw `.log`, `.bpf` or HDF5 `.exo`. Nothing has to be converted first, and the answers match the converted file, which `test_cli_raw_equivalence.py` asserts command by command.
- `validate` and `diff` report a usage error for a raw input: both check the writer contract, which only exists after conversion.
- `analyse --breakout-fraction F` replaces the fixed 0.5 shock breakout threshold. At 2.2.0 the default reproduced the previous breakout time exactly; 2.2.1 changed the fraction to a position on the track, so the default moved to the midpoint.

### Performance

- a native field-table parser parses one LOG table in a single pass: 0.213 ms against 1.318 ms for a 1200 by 14 block, 6.2x, correctly rounded including `1.234-123` and `1.0D+00`.
- the cycle-header scan uses `bytes.find` plus an anchored pattern instead of one MULTILINE regex over the whole buffer: 2.12 s to 0.32 s, with identical block spans.
- `parse_full` takes a field filter and stacks the per-snapshot coordinates: a single-field statistic stacks one field, and a raw analysis stacks the same set the converted path would.

### Packaging

- the native parser ships as a small library built by `scripts/build_native_table.py` and copied into every bundle. It is optional: without it the parser runs the existing Python path at the old speed.
- the core gate grows from ten entries to thirteen.

## Changelog 2.1.0

The library could already answer questions about a converted run; the command line
could only convert. This release puts that surface on a command line, with an output
contract an agent can consume.

On the largest tracked run, 1501 snapshots by 1200 zones, a per-zone statistic takes
52 ms and reading one field 3 ms. `--version` costs 30 ms because the version constant
moved out of the package whose `__init__` imports numpy and Qt.

### Command line

- `helios_diag` with `inspect`, `stats`, `query`, `diff`, `validate` and `analyse`. JSON on stdout by default, `--format table|csv` for humans.
- every output carries the file provenance, the field unit and label with the origin of each, the request, the NaN accounting, and a note for every degradation.
- `inspect` decodes the deck setup and the laser pulse from `metadata/input_parameters` without reading field data.
- `stats` reports the time of each zone's maximum and no peak for a flat zone, rather than the t=0 that `argmax` returns.
- the converter gains exit codes 3 to 6, `--json`, `--progress`, `--quiet`, `--outdir`, and names `.exo` in its help.

### Performance

- quantiles in one numpy call and the non-NaN reduction path when the field holds no NaN: 31.2 ms against 71.0 ms for the per-zone set.
- `--at` reads one snapshot row rather than the whole field: 9.6 KB instead of 14.4 MB.
- `--version` and `--help` answer in about 30 ms, against 230 ms before.

### Packaging and agents

- the radiation plotter ships as `helios_plot_radiation`.
- an agent skill ships in `skills/helios-cli/` with adapters for harnesses without a skill loader; a test asserts every command it names answers `--help`.
- a `justfile` collects the workflows.
- the core gate grows from five entries to ten.

### Fixed

- `helios_diag validate` found a defect on its first run: a flag-off 12.0 deck published `radiation_output = on` with `n_freq_snapshots = 0` and every family empty. The marker now follows the dump count.
- the radiation plotter crashed on a run with flux dumps but an empty spectral axis; it now skips that figure and says why.

## Changelog 2.0.2

A compressed conversion was leaving a third of the saving on the table because
the HDF5 shuffle filter was never enabled. The filter reorders bytes inside each
chunk so the high bytes of a float64 column become a long run of identical
values, which is what a compressor can exploit.

Measured end to end on the largest tracked deck, a 530 MB log of 1501 snapshots
by 1200 zones, pinned to the P-cores: LZF goes from 184.4 MB to 118.6 MB with
the filter, a 35.7 % saving, for 2.6 % more write time. Under gzip the filter
costs 3 %, from 102.2 MB to 105.2 MB, so it stays off. A default conversion is
still uncompressed and every stored value is identical.

### Converter

- `_shuffle_for()` decides the flag from the filter name and is wired into every site that creates a compressed dataset: field datasets, dynamic coordinates, the streaming coordinate pair, the BPF field builder, and the generic array writer.
- `--compression lzf` now enables the shuffle filter; `--compression gzip` does not.
- a run whose datasets are a single small chunk loses 8.5 %: the 8-snapshot tracked fixture goes from 1.53 MB to 1.66 MB. That is 130 kB against 66 MB on the large deck, so the trade is deliberate and recorded in the helper's docstring.
- an empty family stays uncompressed and unshuffled, because h5py refuses the filter without chunked storage and a zero-extent axis cannot be chunked.
- added `CompressionShuffleTest` to the core gate: the filter mapping, the flag on every compressed dataset of a real conversion, and the empty-family case.

## Changelog 2.0.1

The HDF5 writer refused every Exodus run from a deck that leaves the
frequency-resolved radiation output off. That is the default deck and every
production deck in the campaign, so this blocked ordinary runs.

A flag-off run still carries the nine radiation families, and they are empty:
`(0,)` for the per-dump arrays and `(0, n_zones, 250)` for the spectral ones.
HDF5 requires every chunk dimension to be at least 1 and at most the data
extent, and the chunk helpers clamp the zero axis to 1, so `create_dataset`
refused with `Chunk shape must not be greater than data shape in any dimension.
(1, 251) is not compatible with (0, 251)` and the writer aborted the whole file
rather than the family.

### Converter

- `_write_array_dataset` writes a zero-extent array contiguous and uncompressed. It is the single function every array dataset routes through, so the guard covers the radiation, flux and diagnostic families together. The axes are kept, so a consumer reads "this family is empty" rather than "this family is missing". Non-empty arrays are unchanged and still chunked and compressed.
- added `tests/fixtures/radiation/zm0_nofreqrad.exo` with its deck, real HELIOS 12.0.0 output from the default deck: 300 zones, 2 snapshots, 1 region, 250 photon groups. The pre-fix writer reproduces the production error on it exactly.
- added `test_a_flag_off_run_writes_its_empty_families_contiguously`, which asserts shape 0 on all nine families plus `chunks is None` and `compression is None`.
- the previous "no radiation" fixture could not have caught this. `cu_v12_slice.exo` was cut from a run whose radiation group was already absent, so it never reaches the empty-family path.

## Changelog 2.0.0

This release is a performance and correctness pass over the converter, the
reader, the derived analysis and the viewer. Three changes alter what ships:
`scipy` is now a required dependency, the HDF5 chunk layout changed under schema
`2.2`, and the golden fixtures were regenerated because they predated the
centre/edge coordinate contract.

### Measured

Per stage on the largest tracked deck, `10ns+10Si+60Al+15Si+4.27TW`
(268 MB log, 701 snapshots, 1300 zones, 213 MB HDF5). Medians of three runs, warm
page cache, pinned to the six P-cores, same machine, `origin/master` against this
branch:

| stage | before | after | speedup |
|---|---|---|---|
| `.log` to HDF5 | 16.84 s | 6.01 s | 2.8x |
| `.bpf` to HDF5 | 4.72 s | 0.87 s | 5.4x |
| derived analysis | 12.41 s | 1.94 s | 6.4x |
| full `.log` to analysis | 29.25 s | 7.95 s | 3.7x |

Reader operations on the same file (open, 701-snapshot scrub, whole-field read,
dynamic-coordinate scrub) are between 1 and 8 ms before and after. They are
unchanged, and several reader findings from the audit were measured and dropped
rather than fixed because they could not move a millisecond-scale operation.

### Converter

- the `.log` table parser no longer scans every block with a regex before parsing
  it. The scan cost 1.70 s of a 3.72 s conversion; a `try/except ValueError`
  around the parse produces byte-identical output. `np.fromstring` raises on a
  malformed token rather than truncating, so the guard has to be an exception.
- the `.bpf` writer buffers 32 snapshots and writes them in one call instead of
  one dataset write per field per snapshot. That was 81,090 `__setitem__` calls
  per conversion, 68 % of the profile. Buffers are sized from each dataset's own
  trailing shape, because node, group-boundary and per-material fields are not all
  the same width.
- field, dynamic-coordinate and `laser_source_*` datasets now span 32 snapshots
  per chunk, matching the writer's flush batch. Radiation spectral datasets span
  one dump per chunk instead of 16, which cuts the bytes a single-dump read
  inflates from 38.5 MB to 2.4 MB at production size. `flux/*` and `time/*` are
  unchanged. Axis order never changes, only extent, so schema `2.2` is a minor
  bump and the reader needs no change: chunk layout is invisible through the HDF5
  API.

### Derived analysis

- the wavefront grid search batched 55,200 `np.linalg.lstsq` calls on 5x2 systems
  into stacked normal equations solved with `np.linalg.solve`, and hoisted an
  amplitude that was recomputed 20 times per field. `lstsq` cannot take stacked
  input, and 2x2 normal equations square the condition number, so a
  `LinAlgError` fallback to the original per-window solve keeps a saturated
  window from aborting the run. 148 of 149 payload values are bit-identical; the
  one that moved did so by 4.1e-16 relative.
- the interface position grid is built once per run instead of once per candidate
  snapshot, and the candidate maxima scan is vectorised.
- the derived cache key split in two. `core_key` covers geometry and the
  shock/wave selection, so a plasmon or XRD edit reuses the wavefront work
  instead of recomputing it. Because the full request key derives from the core
  key, the split needed explicit guards: cached results carry a module key and are
  reused only on equality, and `_can_reuse_preheat` now runs on all three reuse
  paths. A plasmon-only edit drops from 0.94 s to 0.002 s.
- the radiation container is cached per run revision, and the primary propagation
  direction no longer copies a finite-masked array to answer a question about
  ordering.
- plasmon cluster aggregation uses `np.bincount` and two computations whose
  results were discarded are gone.

### Viewer

- an unchanged colormap name no longer re-resolves and reapplies the map on every
  refresh: 0.564 ms to 0.075 ms per push. Render-path transitions still reapply
  directly, because there the map is unchanged and the item that needs it is new.
- automatic clipping no longer builds a full-field finite mask that only the
  percentile branch reads.
- the snapshot slider emits once per drag instead of once per intermediate value.
  The index, the label and the run context stay synchronous; only the repaint was
  ever the problem.
- the display caches pin their source array, so a freed array's `id()` cannot be
  recycled by a different array and served a stale conversion. Two write-only
  cache-key attributes are removed.

### Dependencies and fixtures

- `scipy` is now a required dependency. It was preferred but undeclared, so a
  clean install silently ran a 1537-point complex quadrature at 25.4 ms per call
  instead of `wofz` at 0.090 ms. It was also in the packaging exclude list, so the
  Windows portable would have shipped without it while the metadata claimed
  otherwise. **This changes plasmon numeric values for anyone who was on the
  fallback.** The About dialog now names the active backend.
- the six golden fixtures were regenerated. They stored the log's `Radius` column
  where zone centres belong; that column is a zone edge. Verified against the
  source logs rather than against the old files: every physics column matches
  exactly, and the old value is recovered as centre plus half the outer zone
  width, to the digit, on `Cu_0166`. `tests/test_log_truth.py` now parses the log
  itself and makes it the oracle.
- `ruff` is configured and enforced. The tree went from 9,463 findings to zero
  under `select = ["E4","E7","E9","F","I"]` at `line-length = 120`. Two real
  misplaced-import defects surfaced in `src`. `ruff format` is deliberately not
  applied.

## Changelog 1.4.0

- the Exodus reader now reads the frequency-resolved radiation output, which it previously dropped. `Write freq-dependent radiation output = 1` adds 18 datasets to the `.exo`; a converted run used to carry none of them and no marker saying whether radiation was off or was discarded.
- published them as two HDF5 groups, not one, because the arrays do not share a time axis: `radiation/` holds the four node-centred spectral arrays on the `time_freq` axis, `flux/` holds the boundary and interface spectra and the power losses on the `time_whole` axis. Flattening them would let a consumer index a boundary spectrum against the spectral axis and get a plausible wrong number. `radiation/hydro_index` records which hydro dump each spectral dump belongs to.
- `radiation/integrated/` carries the ten scalars HELIOS writes per hydro dump: total radiation energy, the loss and flux per boundary, the interface fluxes, net cooling by region, and the electron-to-radiation integral.
- `metadata/radiation_output` is `"on"` or `"off"` on every converted `.exo`, with `n_groups`, `n_freq_snapshots`, `n_boundaries` and `n_regions` when it is on. Absence is now distinguishable from a failed read.
- the four spectral arrays are stored zone-centred, not with HELIOS's raw node dimension. The container writes `n_zones + 2` node rows and the two outer rows are exactly zero on every dump of every reference fixture; the interior rows are the zone centres in Rmin to Rmax order, which two measured identities confirm: the group sum of `energy_density` equals `rad_energy_density[interior]`, and that equals `3 * rad_pressure`. The padding is dropped so the spectral arrays index the same zones as every other field, and a file whose padding rows are not empty raises rather than losing data.
- added a **Radiation** tab to Derived / Analysis: one boundary-spectrum profile per flux dump with Rmin and Rmax in the same frame, the boundary flux and the radiation loss against time, and a metrics block with the group count, the dump counts and the sample time. A run without radiation says so instead of showing an empty plot.
- the two per-group array families use opposite normalizations and the container labels both "per eV". `FdIntgPowerLoss*` is a density, so it needs the group width: `sum(FdIntgPowerLossRmin * dE)` reproduces `FreqIntgRadEnLossRmin` to 1e-16. `net_interface` is a bin content: its plain group sum reproduces `TotalFluxRmin` exactly, and weighting it by the group width is off by 1e4. The writer records which is which in a `normalization` dataset attribute.
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

**The commit gate: about 8 s.** Five fixed entries that cover conversion, the
on-disk layout, the CLI and the runner itself. Add an entry only if it runs in a
second or two and would catch a mistake you would otherwise ship.

```bash
.venv/bin/python scripts/run_test_suite.py --profile core
```

**The milestone gate: about 6 min, with a 75 % coverage floor.** Every module
runs in its own process, which is what keeps the suite at minutes rather than
the hour it takes when the Qt modules share one process and pile up event loops.

```bash
python scripts/run_test_suite.py --profile full --coverage
```

It writes per-entry logs, a JSON outcome manifest and `coverage.txt` to the
report directory, and exits 3 when coverage is under the floor. The floor is
`fail_under = 75` in `pyproject.toml`; the measured value on the reference host
is 82.7 %. Raise the floor at a milestone boundary, deliberately: a floor pinned
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
