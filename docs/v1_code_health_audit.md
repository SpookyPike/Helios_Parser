# HELIOS Parser / Viewer v1.0.0 Code Health Audit

Audit date: 2026-04-29

Scope: parser ingest, HDF5 read/write, cache layers, derived-analysis execution,
viewer workers/UI, release bundle contents, existing benchmark/report artifacts,
and visible production physics surfaces.

## Executive Verdict

The v1.0.0 release is usable for the production-visible workflows, but the code
base is carrying avoidable performance and maintenance risk.

The biggest issue is not parser throughput. The parser already has a streaming
HDF5 conversion path with measured multi-x speedups. The main remaining risks
are:

1. Production hidden physics modules are still computed by the derived backend.
2. Large full-field arrays are eagerly materialized in reader, viewer, and
   derived paths.
3. Cache limits are item-count based, not memory-size based.
4. Two large legacy shadow modules are packaged in the release archive.
5. Several time-series derived routines still use Python loops over snapshots
   where matrix-style NumPy reductions would be faster and simpler to bound.

Recommended release stance: keep v1.0.0 visible GUI filtering, but make the
high-priority fixes below before treating this as a long-lived maintenance
baseline.

## Evidence Reviewed

Repository structure and file scale:

- `src/helios_analysis/workspace.py`: 5,314 LOC, 78 loops.
- `src/helios_analysis/workspace_old.py`: 3,852 LOC, unused legacy shadow.
- `src/helios_viewer/main_window.py`: 3,545 LOC.
- `src/helios/services/derived/plasmon.py`: 3,039 LOC.
- `src/helios/services/derived/shock_tracking.py`: 2,129 LOC.
- `src/helios/services/derived/transmission.py`: 1,619 LOC.
- `src/helios/services/derived/transmission_old.py`: 1,619 LOC, unused legacy shadow.

Release and performance artifacts:

- `outputs/reports/optimization_pass2_final_comparison.json`
- `outputs/reports/viewer_phase2_performance.json`
- `outputs/reports/phase412_validation.json`
- `outputs/reports/phase412c_snapshot_validation.json`
- `outputs/release/helios-parser-viewer-v1.0.0.zip`

Measured parser/write performance from existing reports:

- `Cu_0166.log`: parse 6.14 s -> 2.04 s, write 11.09 s -> 3.99 s.
- `10ns+10Si+60Al+15Si+4.27TW.log`: parse 33.41 s -> 15.93 s,
  write 41.50 s -> 12.89 s.

Measured viewer performance from existing reports:

- Moving-mesh hover on the large file is around 5 ms mean after optimization.
- Snapshot derived refresh reports around 0.32-0.39 s for representative files.
- `phase412_validation.json` confirms max inflight derived worker count of 1.

## Priority Findings

### P0: Hidden Production Physics Modules Are Still Computed

Files:

- `src/helios_analysis/workspace.py`
- `src/helios/feature_flags.py`
- `src/helios/services/derived/analysis.py`

Status: fixed after this audit. Production backend compute now filters hidden
Plasmon/XRTS and Transmission contracts and returns empty disabled result
objects for those result slots. Dev/experimental mode keeps the full module
set.

What is good:

- The production GUI gates Plasmon/XRTS and Transmission with
  `production_feature_visible()`.
- Tests in `tests/test_v1_release_gate.py` confirm hidden production tabs and
  dev/experimental restoration.

Problem:

- `src/helios/services/derived/analysis.py` still registers XRD, Plasmon,
  Transmission, and Spectroscopy in `_MODULE_CONTRACTS`.
- `_compute_module_results()` loops over every contract and computes every
  module unless a cached exact snapshot result can be reused.
- This means hidden production physics still costs CPU, memory, warning
  aggregation, validation surface, and cache pressure.

Impact:

- Wastes derived refresh time on modules the production user cannot see.
- Keeps physically unsafe outputs alive in backend result objects.
- Makes performance tuning harder because v1.0 visible workflows are
  benchmarked with hidden work included.

Recommendation:

- Add a production-visible module filter in `analysis.py`, not just
  `workspace.py`.
- In production, compute only: `xrd`, `spectroscopy`, `shock`, `wave_tracking`,
  `interface_events`, and `preheat`.
- In `HELIOS_DEV_MODE=1` or `HELIOS_ENABLE_EXPERIMENTAL=1`, include Plasmon and
  Transmission.
- Add a test that production `compute_analysis_result()` does not call hidden
  module compute functions.

Expected benefit:

- Faster first derived refresh.
- Less memory churn.
- Cleaner physics release boundary.

### P1: Release Archive Includes Large Legacy Shadow Modules

Files:

- `src/helios_analysis/workspace_old.py`
- `src/helios/services/derived/transmission_old.py`
- `scripts/create_release_bundle.py`
- `outputs/release/helios-parser-viewer-v1.0.0.zip`

Observed:

- `workspace_old.py` and `transmission_old.py` are not imported by active source
  or tests.
- The v1.0.0 release ZIP includes both files:
  - `helios-parser-viewer-v1.0.0/src/helios_analysis/workspace_old.py`
  - `helios-parser-viewer-v1.0.0/src/helios/services/derived/transmission_old.py`
- `transmission_old.py` is the same scale as active `transmission.py`.

Impact:

- Package bloat.
- Confusing maintenance surface.
- Higher risk that a future patch edits the wrong file.
- Legacy code can be discovered by users even though it is not production
  supported.

Recommendation:

- Exclude `*_old.py` from release packaging immediately.
- After v1.0.0 tagging, either delete them or move them to an explicit archive
  outside importable `src`.
- Add a release packaging test that asserts no `*_old.py`, `__pycache__`, or
  `.pytest_cache` paths are present in the ZIP.

### P1: Cache Limits Bound Item Count, Not Memory

Files:

- `src/helios/cache.py`
- `src/helios/services/derived/common.py`
- `src/helios_analysis/controller.py`
- `src/helios_viewer/workers.py`

Current behavior:

- `BoundedCacheBucket` evicts by `max_items`.
- Session raw cache has a default bucket size of 64.
- Derived controller caches `DerivedRunData` with `max_items=3`.
- Derived common loads and shares many full arrays: density, velocity, electron
  and ion temperatures, electron density, mean charge, zone width, radius,
  optional pressure/energy/radiation fields.

Problem:

- One cache item can be a small metadata dict or a multi-MB/GB field array.
- A 64-item raw array bucket can be far too large for bigger HELIOS runs.
- Viewer field payloads and derived arrays can duplicate the same HDF5 data in
  memory.

Impact:

- Memory use is predictable by item count but not by bytes.
- Large runs can stay resident after the user switches fields or datasets.
- Cache stats do not expose byte footprint, so performance diagnosis is blind.

Recommendation:

- Add optional byte accounting to `BoundedCacheBucket`.
- Estimate item size with `np.ndarray.nbytes`, plus recursive dict/list support
  for common metadata.
- Add per-layer byte caps:
  - raw data cache: conservative cap, for example 512 MB default.
  - derived cache: separate cap, for example 256 MB default.
  - view cache: smaller cap, for example 128 MB default.
- Include byte size, byte capacity, and byte evictions in cache stats.
- Prefer sharing read-only arrays between viewer and derived caches by run
  identity instead of copying through separate payloads.

### P1: Full-Field Materialization Is Still The Default Read Path

Files:

- `src/helios_parser/reader.py`
- `src/helios_viewer/workers.py`
- `src/helios/services/derived/common.py`

Status: partially fixed after this audit. `HeliosRun` now has explicit
snapshot-row and zone-trace access that does not materialize the full dynamic
coordinate grid. The viewer worker/controller now expose snapshot-field and
field-trace payload paths. Full map rendering and derived full-run analysis
still intentionally use full 2D fields.

Current behavior:

- `HeliosRun.get_field()` supports `time_slice` and `zone_slice`.
- The viewer worker still loads full fields for field display:
  `data = np.asarray(self._run.get_field(field_name), dtype=np.float64)`.
- Dynamic coordinate access caches full dynamic center/edge arrays.
- Derived `load_run_data()` loads a broad full-field subset for every derived
  request.

What is good:

- The HDF5 reader already exposes sliced field access.
- HDF5 chunks are snapshot-major, which is appropriate for snapshot reads and
  short scrubs.

Problem:

- Higher layers rarely use the sliced read API.
- A simple snapshot/profile display can pull an entire `(time, zone)` field.
- Dynamic radius/edge access is particularly expensive because it can load both
  center and edge grids for all times.

Recommendation:

- Add explicit accessors and payload types for:
  - active snapshot row
  - time trace for one or a few zones
  - visible viewport/time window
- In viewer mode, load full 2D fields only when the 2D map needs them.
- In slice/profile mode, use `get_snapshot_field()`.
- For dynamic coordinates, add snapshot-scoped edge/center read methods that do
  not populate the full coordinate cache unless needed by a moving mesh map.
- Add tests proving snapshot reads do not call full-field reads.

Expected benefit:

- Better startup and field-switch latency for large files.
- Lower memory pressure.
- Better scaling beyond current representative datasets.

## Parser and HDF5 Audit

### Streaming Conversion

Files:

- `src/helios_parser/hdf5.py`
- `src/helios_parser/document.py`
- `src/helios_parser/parser.py`

What is good:

- `write_hdf5()` uses `iter_snapshots_streaming()`.
- Field datasets are created lazily when fields appear.
- Snapshot batches are buffered, currently with `batch_size <= 32`.
- Datasets are resized to actual valid snapshot count at finalize.
- Chunk shape is snapshot-major: up to 16 time rows by all zones.

Remaining bottlenecks:

- HDF5 writes are single-writer, which is appropriate, but parse and write are
  still coupled in one loop.
- `document.parse_full()` still materializes all snapshots and then stacks all
  fields into dense arrays. This remains valid for compatibility but should not
  be used by large-run production flows.
- Header/diagnostic parsing still has several Python regex and line loops.
- Snapshot table parsing has some `np.fromstring()` use, which is good, but
  fallback normalization is still expensive.

Recommendations:

- Keep `parse_full()` as a compatibility/API path, but mark it as not suitable
  for large production conversion.
- Add a parser regression test or benchmark that fails if conversion accidentally
  routes through `parse_full()` for CLI/GUI HDF5 conversion.
- Consider a producer/consumer pipeline:
  - one worker parses snapshot blocks into NumPy arrays
  - one ordered writer thread writes HDF5 batches
  - HDF5 writes remain single-threaded
- Pre-normalize D/E scientific notation at block level and favor
  `np.fromstring()` over per-token `float()` wherever table width is known.
- Add configurable conversion batch size. Current 32 is safe; larger batches
  may improve write throughput on SSDs for medium files.

Parallelization opportunity:

- Parallelize snapshot parsing by block ranges only if ordering and diagnostics
  schema widening are controlled. Keep HDF5 writing ordered and single-threaded.

## Viewer Audit

Files:

- `src/helios_viewer/main_window.py`
- `src/helios_viewer/workers.py`
- `src/helios_viewer/plots.py`

What is good:

- Worker thread separation prevents the GUI from blocking on open/load.
- Hover and moving-mesh interaction performance is already measured.
- Existing benchmark shows hover profile mean around 4-6 ms.

Problems:

- `main_window.py` is very large and mixes state, plotting, UI controls,
  cache behavior, coordinate semantics, export, and interaction policy.
- `RunWorker.load_field()` loads full fields for every selected field.
- Field map and slice modes likely share the same heavy payload even when a
  slice view only needs one snapshot.
- Table resizing such as `resizeColumnsToContents()` appears in derived
  population paths and can become expensive for large tables.

Recommendations:

- Split `main_window.py` after v1.0.0 into targeted controllers:
  - run/session state
  - field map view
  - slice/profile view
  - export
  - coordinate controls
- Introduce lazy field payloads or snapshot payloads for slice-only view.
- Keep full-field payload only for active 2D map field.
- Add a small viewer performance smoke test for:
  - opening run metadata without loading a field
  - loading a snapshot profile
  - loading a full map field
- Avoid automatic `resizeColumnsToContents()` on every update for large tables;
  set fixed or content-sampled widths where possible.

## Derived Analysis Audit

### Orchestration and Tasking

Files:

- `src/helios/services/derived/analysis.py`
- `src/helios_analysis/controller.py`
- `src/helios/tasks.py`

What is good:

- Derived computation is off the GUI thread.
- `AnalysisTaskManager` has latest-wins cancellation semantics.
- Thread pool is deliberately capped at one worker, which reduces stale result
  races and HDF5 contention.
- Snapshot refresh can reuse existing full-run traces.

Problems:

- Hidden modules still compute, as noted in P0.
- Independent visible modules are always computed serially.
- The single worker is correct at the GUI task level, but inside one task there
  is no module-level parallelism.

Recommendations:

- Keep GUI-level max worker count at 1 for stability.
- Inside one derived request, parallelize independent NumPy-heavy visible
  modules only after hidden modules are filtered:
  - XRD time traces
  - Spectroscopy time traces
  - optional Preheat summary
- Use threads only for NumPy-heavy sections that release the GIL.
- Use processes only for pure-Python heavy loops if serialization cost is
  measured to be lower than compute time.
- Keep shock/wave branch association sequential unless it is redesigned, because
  branch continuity is time ordered.

### XRD

File: `src/helios/services/derived/xrd.py`

Current behavior:

- Region aggregation uses `np.bincount()`, which is good.
- Time-series generation still loops over every snapshot and calls
  `_region_metrics_for_snapshot()`.

Vectorization opportunities:

- Build a `(time, zone)` valid mask once.
- For contiguous region blocks, use `np.add.reduceat()` across zones.
- For arbitrary region IDs, use matrix-style reductions with a precomputed
  region membership matrix or repeated `np.bincount()` over time in a tighter
  vectorized helper.
- Compute density sums, weight sums, thickness, compression, Bragg shift, and
  Q arrays as full matrices instead of one snapshot at a time.

Expected benefit:

- Faster time-plot activation.
- Less repeated cache lookup and selection reconstruction.

### Spectroscopy

File: `src/helios/services/derived/spectroscopy.py`

Current behavior:

- It already uses `shared_time_series_weighted_means()` for core velocity,
  temperature, and mass means.
- Thermal width is then calculated in a Python loop over time.

Vectorization opportunities:

- Replace scalar `doppler_width_fraction()` loop with a vectorized formula over
  `ti_series` and `ion_mass_series`.
- Keep scalar helper for single-snapshot/profile use; add vector helper for
  time series.

Expected benefit:

- Small but low-risk speedup.
- Simpler NaN handling with one finite mask.

### Selection and Shared Weighted Means

File: `src/helios/services/derived/selection.py`

Current behavior:

- `shared_time_series_selection_state()` builds a `(time, zone)` boolean mask by
  looping over time and calling `build_analysis_mask()`.
- `shared_time_series_weighted_means()` loops over time and fields, then calls
  `weighted_means()`.

Problem:

- This is a central hot path reused by XRD and Spectroscopy.
- Any improvement here benefits multiple kept modules.

Vectorization opportunities:

- Split masks into static and dynamic components.
- Region/material/zone-index filters are static and can be built once.
- Low-density and opposite-velocity filters are dynamic and can be applied as
  matrix operations.
- For common weighting modes, compute full `(time, zone)` weights once:
  - path length
  - zone width
  - density/path
- Then compute weighted means with matrix reductions:
  `np.nansum(values * weights * mask, axis=1) / np.nansum(weights * mask, axis=1)`.

Expected benefit:

- High leverage for all visible derived time traces.

### Shock and WaveFront

File: `src/helios/services/derived/shock_tracking.py`

Current behavior:

- Legacy shock detector matrix is built over snapshots.
- Wave tracking extracts candidates per frame, then associates branches in time
  order.
- Branch association is inherently sequential.

Vectorization and parallelization opportunities:

- Vectorize evidence precomputation across `(time, zone)`:
  - density gradients
  - velocity gradients
  - pressure/temperature cues
  - finite masks and thresholds
- Candidate extraction per frame can be parallelized after evidence matrices are
  built.
- Branch association should remain sequential, but it can operate on a compact
  precomputed candidate list instead of recomputing frame evidence.

Expected benefit:

- Faster WaveFront activation on large files.
- Lower repeated work between legacy Shock and WaveFront.

### Preheat

File: `src/helios/services/derived/preheat.py`

Current behavior:

- Preheat includes loops over selected snapshot indices and all time indices.
- It is lazy/advanced, which protects normal refresh.

Vectorization opportunities:

- Precompute target-region mask and energy/temperature arrays once.
- Convert per-snapshot reductions into matrix reductions.
- Share path/selection masks with the common selection cache.

Expected benefit:

- Faster advanced tab activation.

### Plasmon and Transmission

Files:

- `src/helios/services/derived/plasmon.py`
- `src/helios/services/derived/transmission.py`

Release status:

- These are hidden in production GUI because previous physical QA classified
  them as unsafe for v1.0 user-facing exposure.

Performance status:

- They should not be on the production critical path.
- They remain valid dev/experimental targets.

Recommendations:

- First remove them from production compute.
- For dev mode:
  - parallelize independent Plasmon scan points/model comparisons
  - cache response functions by state tuple
  - vectorize Transmission time-series optical-depth reductions
  - keep XCOM/refinement calls isolated behind explicit user action

## Cache and Data Ownership

Current cache surfaces:

- `AnalyzerCacheSet.raw_data_cache`
- `AnalyzerCacheSet.derived_cache`
- `AnalyzerCacheSet.view_cache`
- process-level `get_session_raw_data_cache()`

Good:

- Cache layers are explicit.
- Cache stats exist.
- Run identity includes path, mtime, and size.

Gaps:

- No byte accounting.
- No central ownership decision for full HDF5 field arrays.
- No visible cache pressure reporting in diagnostics.
- No policy distinction between metadata, small vectors, full fields, and
  derived plots.

Recommended cache policy:

- Metadata: cache broadly.
- Small coordinate/time vectors: cache broadly.
- Full 2D fields: cache narrowly by active run and active field.
- Derived plots: cache by request key, but evict by bytes.
- Experimental module results: never populate production cache.

## Dead or Legacy Code

High-confidence unused:

- `src/helios_analysis/workspace_old.py`
- `src/helios/services/derived/transmission_old.py`

Intentionally retained compatibility code:

- Legacy coordinate aliases in HDF5 reader/writer.
- Legacy HDF5 layout support in reader/registry tests.
- Legacy Shock compatibility adapters used by WaveFront and current UI.

Do not delete without replacement:

- Legacy coordinate handling.
- Legacy HDF5 compatibility.
- Legacy Shock path, because it remains production-visible and fast.

Cleanup recommendation:

- Remove only shadow modules from release packaging now.
- Keep compatibility logic that is exercised by tests.

## Documentation and Release Bundle Drift

Observed:

- The release ZIP excludes caches, but includes legacy shadow modules.
- The release ZIP includes a Plasmon screenshot under docs assets.
- Some docs still describe advanced/legacy behavior broadly.

Recommendation:

- Add release bundle assertions:
  - no `__pycache__`
  - no `*_old.py`
  - no large debug outputs
  - no screenshots for hidden production-only flows unless clearly labeled
    experimental/dev
- Update docs to state that Plasmon/XRTS and Transmission are dev-only in
  v1.0.0 production.

## Optimization Roadmap

### Immediate, Low-Risk

1. Filter hidden modules in `analysis.py` production compute.
2. Exclude `*_old.py` from release bundle.
3. Add ZIP-content tests for legacy/cache/debug exclusions.
4. Vectorize Spectroscopy thermal width time series.
5. Add cache byte stats without changing eviction policy yet.

### Medium Effort

1. Convert shared weighted time-series means to matrix reductions.
2. Vectorize XRD time-series region reductions.
3. Add snapshot-scoped viewer field payloads.
4. Add byte-capped LRU eviction.
5. Split `main_window.py` into smaller controllers.

### Larger Effort

1. Lazy dataset-backed derived data access instead of eager `DerivedRunData`
   full-field loading.
2. Module-level parallel execution inside one derived task.
3. Parser producer/consumer pipeline for snapshot parsing and HDF5 writing.
4. Vectorized WaveFront evidence precomputation.

## Suggested Tests To Add

Production feature boundary:

- Production `compute_analysis_result()` does not call Plasmon or Transmission.
- Dev/experimental mode restores those compute calls.

Release bundle:

- ZIP excludes `*_old.py`.
- ZIP excludes caches and debug outputs.
- ZIP includes expected launch assets and example files.

Cache:

- Cache stats expose byte totals.
- Full-field arrays evict under a small byte cap.
- Switching between large fields does not retain unbounded arrays.

Reader:

- `get_snapshot_field()` reads a single row and returns correct shape.
- Viewer slice/profile path does not call full `get_field()` for map-size data.

Derived:

- XRD vectorized time traces match current scalar loop within tolerance.
- Spectroscopy vectorized thermal width matches scalar helper within tolerance.
- Sanity checks still reject non-finite or physically impossible outputs.

Performance:

- Keep lightweight benchmark baselines in JSON:
  - open metadata
  - load full field
  - load snapshot profile
  - first derived refresh
  - snapshot refresh
  - XRD time plot activation
  - Spectroscopy time plot activation

## Final Assessment

The parser and viewer are already past the first round of optimization work.
The strongest next speedups will come from reducing work, not from micro-tuning:

- do not compute hidden v1.0-invalid physics modules in production
- do not read full arrays for snapshot-only work
- do not keep large arrays in item-count-only caches
- vectorize shared time-series reductions once instead of optimizing every
  module separately

The code base has enough instrumentation and benchmark scripts to support this
work without guessing. The next pass should focus on production-path trimming
and memory-bounded data access before adding more derived physics.
