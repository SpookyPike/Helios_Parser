# HELIOS Parse / View audit

**Date:** 2026-10-20 · **Version:** 1.1.3 · **HEAD:** `7663446` plus the existing working-tree changes.

**Assessment:** Fix data preservation and calculation consistency before further optimization. The architecture already supports streaming, caching, and background work, but several paths can silently change scientific results or retain stale data. The desktop interface also has reproducible accessibility and small-window problems.

**25 findings: 14 P1, 11 P2; no P0 or P3 findings.** P1 means fix before release because of data integrity, incorrect results, blocked workflows, or major usability problems. P2 means a bounded performance, compatibility, or usability improvement. Findings concern the audited working tree; they are not necessarily regressions introduced by its current edits.

## Scope and method

Reviewed the LOG tokenizer/indexer/parser, streaming document model, BPF records and field mappings, HDF5 conversion and reading, shared caches, worker/controller lifecycles, desktop shell/viewer, and derived analysis. Production calculations reviewed include Shock, WaveFront/interface events, Preheat, XRD, and Spectroscopy. Experimental Plasmon/XRTS and Transmission were inspected and their existing suites exercised; experimental findings are identified below.

Used Impeccable's audit/critique guidance, structural source outlines, targeted source tracing, controlled corruption/edge-case probes, a wheel build, fresh-process tests, and current Qt renders. No Momo skill/tool was available. Ix could not produce a usable graph for this repository, so findings were established from source and execution instead.

Tests and generated outputs ran in `/tmp/helios-audit-20261020/repo`, a copy of the current source and fixtures. Existing project edits and reports were left intact. Scratch scripts, JSON measurements, screenshots, and test logs are under `/tmp/helios-audit-20261020/`; the essential evidence is recorded here so this report stands alone.

GUI review used the actual Qt widgets with the offscreen platform, in light/dark themes and at requested 1280×800 and 1024×768 sizes. Accessibility names were queried through `QAccessible`. Native screen-reader behavior, Windows/macOS rendering, and high-DPI behavior still require platform testing. Numerical review establishes the implementation defects below; passing tests do not establish research-level accuracy of every experimental physical model.

## Highest-priority findings

### F01 · P1 · Conversion can delete the previous output or the input itself

**Location:** [hdf5.py:758](../src/helios_parser/hdf5.py#L758), especially `target.unlink()` at line 777.

The existing target is removed before opening/validating the input. A malformed LOG with `overwrite=True` deleted a sentinel output and then raised `ValueError`. Passing the same LOG path as input and output deleted the source and raised `FileNotFoundError`. Later failures can also leave a partially written HDF5 at the final filename.

- **Suggested fix:** Reject input/output identity, including symlink/hardlink aliases and companion inputs. Write to a temporary sibling and atomically replace the target only after successful finalization.
- **Test direction:** Invalid input, injected mid-write failure, same path, symlink alias, and valid overwrite. Assert original bytes survive failures and no partial final output is published.

### F02 · P1 · Preheat integrates cumulative laser energy a second time

**Location:** [common.py:69](../src/helios/services/derived/common.py#L69), [common.py:543](../src/helios/services/derived/common.py#L543), [preheat.py:462](../src/helios/services/derived/preheat.py#L462), [preheat.py:803](../src/helios/services/derived/preheat.py#L803), [bpf.py:189](../src/helios_parser/bpf.py#L189).

`laser_source` is cumulative **J/g**, but the loader calls it `laser_source_j_g_s` and `_source_term_budget` integrates it as a rate. For a unit areal mass and cumulative values `[0, 10, 20] J/g` at `[0, 1, 2] ns`, the function returned `2e-8`, while the energy increase is `20 J/cm²`. This corrupts the displayed source budget and the fallback residual/source attribution when instantaneous deposition is unavailable.

- **Suggested fix:** Represent cumulative energy and instantaneous power separately. Obtain cumulative-source budgets from endpoint energy differences with the appropriate geometry/mass weighting; integrate only J/g/s fields.
- **Test direction:** An analytic constant-deposition case must give matching budgets from instantaneous deposition and cumulative source, including a nonzero initial cumulative value and a selected time window.

### F03 · P1 · Reopening a replaced file can reuse stale derived data

**Location:** [controller.py:493](../src/helios_analysis/controller.py#L493), [controller.py:220](../src/helios_analysis/controller.py#L220), [runtime.py:109](../src/helios/runtime.py#L109).

The derived dataset cache is keyed only by `Path`; context/result identities contain path and dimensions, without a file revision. After changing a copied HDF5 density from `7.874` to `15.748` and delivering its reopened context, the controller still retained the dataset containing `7.874`. Shared raw-cache mtime/size identity does not fix this separate cache.

- **Suggested fix:** Carry one run revision/generation through dataset, context, analysis-result, and task keys. Invalidate dependent caches when a file is reopened or replaced.
- **Test direction:** Analyze a file, replace it at the same path with equal dimensions, reopen, and assert both viewer and derived values update. Include replacement while an old task is in flight.

### F04 · P1 · BPF conversion bypasses important validity checks

**Location:** [bpf.py:491](../src/helios_parser/bpf.py#L491), [bpf.py:520](../src/helios_parser/bpf.py#L520), [hdf5.py:686](../src/helios_parser/hdf5.py#L686).

`BpfFile.iter_snapshots()` checks time ordering, but the writer calls `extract_snapshot()` directly. Patching the second timestamp to `-1 s` made the iterator reject the file while conversion accepted it and marked the output complete. Separate record-aligned probes with a reversed node coordinate or negative density were also accepted. Invalid time/geometry can contaminate gradients, integration, and LOS weights.

- **Suggested fix:** Route conversion through a common validated iterator. Validate time and cycle policy, strictly increasing finite node edges, and density/domain constraints appropriate to each field. Preserve explicitly allowed EOS behavior rather than rejecting every negative pressure.
- **Test direction:** Corrupt individual records while keeping Fortran markers valid. Reject decreasing/nonfinite times, inverted cells, and invalid density; exercise duplicate-time policy before any derivative calculation.

### F05 · P1 · A partial final BPF snapshot is reported as complete

**Location:** [bpf.py:369](../src/helios_parser/bpf.py#L369), [hdf5.py:736](../src/helios_parser/hdf5.py#L736).

Truncating the eight-snapshot fixture after five complete records of its final snapshot produced seven output snapshots and five trailing records, but `state="complete"` and `dropped_partial_final_block=False`. This differs from the LOG truncation protections and hides missing final data.

- **Suggested fix:** Recognize documented trailers separately from incomplete snapshot records. Preserve valid frames but record truncation honestly; distinguish an intact export from evidence that the simulation completed.
- **Test direction:** Truncate at record boundaries and inside records, including the first/final snapshot and valid trailers. Verify retained counts, status, warnings, and CLI behavior.

### F06 · P1 · The meaning of `pressure` depends on the input source

**Location:** [bpf.py:446](../src/helios_parser/bpf.py#L446), [bpf.py:226](../src/helios_parser/bpf.py#L226), [parser.py:1038](../src/helios_parser/parser.py#L1038), [common.py:209](../src/helios/services/derived/common.py#L209).

BPF `pressure_j_cm3`, aliased to `pressure`, is ion plus electron pressure. LOG `pressure` adds ion, electron, and radiation pressure. Derived loading labels the alias as total pressure and checks it against all three components. The same public field therefore changes meaning depending on whether a companion BPF exists. Small radiation contributions in current fixtures can conceal the discrepancy.

- **Suggested fix:** Define explicit plasma-pressure and total-pressure contracts, then make aliases, metadata, consistency checks, and derived consumers agree across sources.
- **Test direction:** Use a synthetic radiation-dominated case, e.g. components `1, 2, 4`. Assert identical documented semantics through LOG-only and BPF-primary conversion; do not rely only on a global relative tolerance.

### F07 · P1 · LOG/BPF laser-source alignment checks only the endpoints

**Location:** [hdf5.py:569](../src/helios_parser/hdf5.py#L569).

The cumulative LOG source overrides BPF-derived values when shapes and first/last times match. Interior snapshot times and cycle identities are not checked. A probe changed the LOG's second time to `7.5339e-12 s`; the BPF time remained `5.022593e-12 s`, yet the override was accepted and its source marked `log`.

- **Suggested fix:** Match all timestamps and cycles with justified tolerances. If alignment fails, retain the BPF-derived source with an explicit warning or require a documented resampling policy.
- **Test direction:** Equal endpoints with shifted, reordered, or duplicated interior frames must not silently copy rows. Verify accepted alignment within LOG text precision.

### F08 · P1 · Scientific notation can be silently misread in LOG metadata

**Location:** [tokenizer.py:43](../src/helios_parser/tokenizer.py#L43), [tokenizer.py:116](../src/helios_parser/tokenizer.py#L116), [parser.py:975](../src/helios_parser/parser.py#L975).

Numeric extraction recognizes uppercase `E`, while normalization of `D/d` happens after matching. Thus `1.25D-09` and `1.25e-09` split into `1.25` and `-09`. Changing only cycle-line exponents in the Cu fixture from `E` to `D` changed initial times from `[1e-15, 5.0226e-12, 1.0021e-11]` to `[1, 5.0226, 1.0021]` and produced negative timesteps, without rejecting the parse. Field-table normalization already handles D notation, making behavior inconsistent.

- **Suggested fix:** Match complete `[EeDd]` exponents or normalize numeric notation before extraction; validate parsed time/timestep domains.
- **Test direction:** Parameterize header, cycle, and diagnostic numbers over E/e/D/d and omitted-E notation. Compare complete parsed simulations, not just token strings.

### F09 · P1 · Missing Shock results abort unrelated analyses

**Location:** [analysis.py:345](../src/helios/services/derived/analysis.py#L345), [physical_sanity.py:27](../src/helios/services/derived/physical_sanity.py#L27).

Any nonempty selection invokes `validate_shock_result`, which raises when no finite shock track exists. Uniform synthetic states used by the Plasmon UI tests consequently fail before their requested analysis runs. An absence of detectable shock is a valid state, not evidence that XRD or spectroscopy cannot be evaluated. Other model-domain handling has similar coupling: zero ion temperature yields a NaN Doppler width and a fatal validator error; an unreachable Bragg angle creates NaN Q and can also abort assembly.

- **Suggested fix:** Separate invalid input/results from physically unavailable module estimates. Return an unavailable status and warning for the affected module while keeping independent results usable.
- **Test direction:** Uniform no-shock runs must still yield valid independent analyses. Cover empty selections, zero-temperature policy, and unreachable reflections without suppressing genuine invalid-data errors.

### F10 · P1 · Valid standalone BPF outputs cannot enter Derived mode reliably

**Location:** [hdf5.py:382](../src/helios_parser/hdf5.py#L382), [common.py:317](../src/helios/services/derived/common.py#L317), [common.py:480](../src/helios/services/derived/common.py#L480).

BPF-only conversion creates a minimal region/material schema, while `load_run_data` assumes `atomic_weight`, `initial_mass_density`, and `initial_temperature` exist. Converting the standalone BPF fixture succeeded; loading it for derived analysis raised `KeyError('atomic_weight')`. Sparse LOG schemas can similarly encounter mandatory field reads before module capabilities are considered.

- **Suggested fix:** Load and validate capabilities per module. Disable estimates requiring missing physical metadata with a specific explanation; do not invent atomic weights or initial densities.
- **Test direction:** Open BPF-only and sparse HDF5 fixtures in Derived mode. Available modules should work, unavailable ones should explain the missing inputs, and the application must remain responsive.

### F11 · P1 · The desktop Parser cannot open the BPF inputs supported by the backend

**Location:** [main_app.py:759](../src/helios_app/main_app.py#L759), [main_app.py:771](../src/helios_app/main_app.py#L771), [parser_controller.py:122](../src/helios_app/parser_controller.py#L122).

The dialog accepts only `*.log`, routing handles `.log`/HDF5, and preview uses the text LOG parser. A user with only a BPF file cannot complete the advertised desktop parsing workflow, although the CLI converter supports it. A LOG preview also describes the text fields without clearly announcing that conversion may select its BPF companion as primary.

- **Suggested fix:** Add “Open HELIOS input” with LOG/BPF routing and a BPF-aware preview. Show selected primary source, companions, and the actual output field inventory before conversion.
- **Test direction:** Exercise BPF-only, LOG-only, LOG+BPF, and a malformed optional companion from the GUI. Assert preview and conversion agree on provenance and available fields.

### F12 · P1 · Numeric analysis controls lack accessible names

**Location:** [workspace.py:987](../src/helios_analysis/workspace.py#L987), [workspace.py:1008](../src/helios_analysis/workspace.py#L1008).

Qt accessibility inspection found empty names for the visible minimum-density control and both zone-bound controls. Their adjacent labels are plain layout widgets without buddies or explicit accessible names. A screen-reader user cannot reliably identify which value is being edited. “LOS angle” did expose a name, demonstrating that the issue is control-specific rather than an unavailable accessibility API.

- **Suggested fix:** Set label buddies or explicit names/descriptions, including lower/upper zone bounds and units. Check keyboard order and state announcements.
- **Test direction:** Assert nonempty meaningful `QAccessible.Name` values, then complete filtering using a native screen reader and keyboard only. Suggested Impeccable command: `/impeccable harden`.

### F13 · P1 · Common desktop window sizes clip controls and navigation

**Location:** [main_app.py:170](../src/helios_app/main_app.py#L170), [main_app.py:502](../src/helios_app/main_app.py#L502), [workspace.py:900](../src/helios_analysis/workspace.py#L900), [workspace.py:1969](../src/helios_analysis/workspace.py#L1969).

In fresh Qt renders, requesting 1280×800 resulted in an 833 px parser window, an 879 px viewer, and a 948 px derived window. At a requested 1024×768, the window remained 879 px high; the derived sidebar showed truncated labels/editors and horizontal scrolling, plot labels clipped, and bottom snapshot text ran past the edge. These are layout constraints, not a need for a mobile version.

- **Suggested fix:** Reduce minimum-size propagation, bound splitter sizes, allow form rows to wrap, and collapse secondary metadata/controls. Keep essential navigation visible at laptop sizes.
- **Test direction:** Check actual window geometry and control visibility at 1024×768, 1280×800, and larger sizes in both themes, then repeat with enlarged text. Suggested command: `/impeccable adapt`.

### F14 · P1 · The release test gate is incomplete and currently unhealthy

**Location:** [run_test_suite.py:20](../scripts/run_test_suite.py#L20), [run_test_suite.py:50](../scripts/run_test_suite.py#L50), current tests listed below.

The named `full` profile contains 13 entries, including smoke, and omits **47 of the 59 current `test_*.py` modules**. BPF, truncation, coordinates, release gates, and newer numerical tests are among the omissions. Independent discovery exposed assertion failures, numerical/module errors, native process crashes, and timeouts. Some failures are obsolete expectations rather than application defects; they still make the gate ineffective.

- **Suggested fix:** Discover all supported modules in fresh processes, preserve exit codes, and make exclusions explicit. Reconcile stale units/field-count expectations, fix module failures, and isolate the Qt teardown crash on the supported dependency matrix.
- **Test direction:** Assert discovery covers every test module or a documented exclusion. Require clean process exit, not just printed `OK`; investigate both timeout modules and retain per-module logs.

## Performance, numerical edge cases, and interface consistency

### F15 · P2 · Canonical BPF fields bypass display-unit preferences

**Location:** [units.py:117](../src/helios_viewer/units.py#L117), [main_window.py:1115](../src/helios_viewer/main_window.py#L1115).

Unit-family selection recognizes legacy field names. With temperature set to K, `temperature_e=1 eV` became `11604.518 K`, but canonical `electron_temperature_eV` remained `1 eV`. Likewise `pressure` converted to GPa while `pressure_j_cm3` stayed native. Labels remain honest, but equivalent fields behave differently, particularly when compatibility aliases are hidden.

- **Suggested fix:** Resolve unit families from validated field metadata/aliases and physical units, including canonical BPF names.
- **Test direction:** For every alias/canonical pair, assert equal converted values and units in maps, lineouts, probes, and exports. Suggested command: `/impeccable clarify`.

### F16 · P2 · Spectrum caching silently replaces nonuniform energy grids

**Location:** [plasmon_spectrum.py:46](../src/helios/services/derived/plasmon_spectrum.py#L46). **Experimental.**

For a nonuniform grid longer than 32 points, the cache key falls back to point count and endpoints, then reconstructs a linear grid. `linspace(0,1,40)` and its square generated the same key; reconstruction changed interior coordinates by up to `0.249836`. Adaptive/nonuniform inputs can therefore receive spectra evaluated at different energies. Current GUI-generated linear grids avoid this case.

- **Suggested fix:** Encode/hash all nonuniform grid values, or explicitly reject unsupported grids. Preserve the input grid in computation.
- **Test direction:** Compare cached/uncached kernels on linear, logarithmic, and adaptive grids with identical endpoints/counts; verify distinct keys and matching spectra.

### F17 · P2 · Radian controls round physical parameters too aggressively

**Location:** [workspace.py:2294](../src/helios_analysis/workspace.py#L2294), [workspace.py:3845](../src/helios_analysis/workspace.py#L3845). **Plasmon control is experimental.**

The general angle conversion uses six radian decimals, but the Plasmon configurator resets its control to three. The existing display-settings test entered π/2 and received `90.011669615°` instead of `90°`. This adds avoidable errors and can alter analysis/cache keys when changing representation.

- **Suggested fix:** Preserve canonical physical values separately from rounded presentation, and make angle precision consistent.
- **Test direction:** Degree/radian and angle/k round trips must preserve requested geometry within a documented physical tolerance; retain the failing stability regression.

### F18 · P2 · BPF-primary conversion still fully parses the LOG companion

**Location:** [hdf5.py:555](../src/helios_parser/hdf5.py#L555), [parser_controller.py:54](../src/helios_app/parser_controller.py#L54).

The BPF path opens LOG metadata, then parses the entire LOG and retains every field primarily to recover cumulative `laser_source`. A Cu fixture run took 4.61 s with a companion versus 3.35 s without; full LOG parsing accounted for 1.09 s and 9.37 MB of field arrays. These are single-run observations, not general speedup estimates. Error-tail capture also uses `readlines()` on the entire LOG to return only 20 lines.

- **Suggested fix:** Stream the required companion field/diagnostics and alignment metadata. Read a bounded file tail. Put optional companion inspection inside a fallback/error policy.
- **Test direction:** Compare peak RSS and conversion time over increasing snapshot counts; assert BPF conversion does not materialize unrelated LOG fields. Test errors on a large malformed companion.

### F19 · P2 · Eager derived loading and item-count caches do not bound memory

**Location:** [common.py:317](../src/helios/services/derived/common.py#L317), [common.py:29](../src/helios/services/derived/common.py#L29), [cache.py:41](../src/helios/cache.py#L41), [controller.py:82](../src/helios_analysis/controller.py#L82).

Opening Derived loads all available optional hydro arrays even for basic Shock use. The small fixture required 25 full field reads. Session arrays are limited to 192 entries, viewer fields to their own count, and derived datasets to three runs; none imposes a byte budget. A few large matrices can exhaust memory well before any item limit is reached.

- **Suggested fix:** Load fields by requested module, add byte accounting/budgets, share alias storage, and release old-run ownership deliberately.
- **Test direction:** Measure peak RSS for Shock-only, Preheat, and repeated run switching on large synthetic files. Verify byte eviction and that activating one module does not load unrelated fields.

### F20 · P2 · Clearing the shared cache detaches its live buckets

**Location:** [cache.py:139](../src/helios/cache.py#L139), [cache.py:187](../src/helios/cache.py#L187), [common.py:28](../src/helios/services/derived/common.py#L28).

`CacheLayer.clear()` clears buckets and removes them from the registry. `common.py` retains module-level bucket references. After a clear, new writes went to its old bucket while `layer.bucket("session_run_arrays")` returned a different empty object. Subsequent global clearing/statistics can miss the live data.

- **Suggested fix:** Preserve registered bucket identities during clearing, or resolve buckets dynamically everywhere instead of retaining references.
- **Test direction:** Publish, clear, publish again, inspect layer statistics, and clear again. All clients must share one registered bucket and the second clear must release its entries.

### F21 · P2 · Cold attenuation repeats identical lookups per zone

**Location:** [xcom_hook.py:724](../src/helios/services/derived/xcom_hook.py#L724), [xcom_hook.py:646](../src/helios/services/derived/xcom_hook.py#L646). **Experimental.**

A 1,300-zone single-material/energy request called `lookup_fallback_mu_rho` 1,300 times. Each lookup rebuilds table vectors and searches the energy grid. Grouping the same material's areal density into one zone gave transmission `0.9824927444988333`, versus `0.9824927444988334` originally. The repeated cost grows with zones × energies × table rows, despite few unique materials.

- **Suggested fix:** Aggregate cold areal density by canonical material, evaluate each material/energy coefficient once, and retain required budget metadata. Precompute searchable table arrays.
- **Test direction:** Assert lookup count scales with unique materials × energies and compare transmission/budgets against the original per-zone sum across mixtures and boundary energies.

### F22 · P2 · Identical in-flight reader requests are not deduplicated

**Location:** [controller.py:94](../src/helios_viewer/controller.py#L94), [controller.py:106](../src/helios_viewer/controller.py#L106).

Caches avoid completed reads, but pending requests are not tracked. Five identical snapshot requests before processing completions emitted five worker requests. Rapid refreshes can queue redundant HDF5 reads; generation checks discard stale results after the work has already occurred.

- **Suggested fix:** Maintain generation-scoped pending keys and coalesce identical requests. Drop superseded queued work where practical without breaking busy/error accounting.
- **Test direction:** A burst of identical field/snapshot/trace requests should produce one physical read; test errors and generation changes so pending keys cannot become stuck. Suggested command: `/impeccable optimize`.

### F23 · P2 · Reading one field materializes metadata for every field

**Location:** [reader.py:210](../src/helios_parser/reader.py#L210), [reader.py:757](../src/helios_parser/reader.py#L757).

The first `get_snapshot_field('density', 0)` on a BPF-derived run invoked metadata parsing for all 104 fields because axis lookup calls the whole-registry loader. This is a one-time metadata fan-out, not 104 full data reads, but it undermines narrow reader access for rich schemas.

- **Suggested fix:** Cache metadata per requested field; enumerate everything only when the field browser explicitly requests the inventory.
- **Test direction:** A single-field read should load only its metadata, a repeated read none, and explicit inventory discovery all fields once.

### F24 · P2 · Switching to dark mode leaves low-contrast field headings

**Location:** [main_window.py:1394](../src/helios_viewer/main_window.py#L1394), [main_window.py:1472](../src/helios_viewer/main_window.py#L1472).

Field-group headings retain the foreground assigned when the list was built. After loading in light mode and switching to dark, “Physics and mapped fields” remained `#475569` on `#111827`: measured contrast **2.34:1**. The theme-switch code refreshes plots/icons but not the existing item colors.

- **Suggested fix:** Reapply semantic foreground colors to all existing list items on theme changes; preserve selection/filter state.
- **Test direction:** Compare fresh light/dark startup with light→dark→light transitions and check informational heading contrast as well as field rows. Suggested command: `/impeccable colorize`.

### F25 · P2 · Built wheels omit scientific reference resources

**Location:** [pyproject.toml:47](../pyproject.toml#L47), [plasmon_reference_data.py:19](../src/helios/services/derived/plasmon_reference_data.py#L19), [plasmon_electron_policy.py:230](../src/helios/services/derived/plasmon_electron_policy.py#L230), [xcom_hook.py:121](../src/helios/services/derived/xcom_hook.py#L121). **Experimental resources.**

A fresh wheel contained no JSON files. From its extracted package, importing `plasmon_reference_data` raised `FileNotFoundError`; the material-policy registry had zero entries and XCOM fallback data was absent. Source-tree/editable execution conceals these omissions. The basic application import still succeeded; this is not a demonstrated production startup failure.

- **Suggested fix:** Package required data under the owning Python package and resolve it with package-resource APIs. Declare optional-resource absence explicitly.
- **Test direction:** Build/install a wheel outside the checkout, with no source-tree fallback. Exercise reference loading, policy lookup, and cold attenuation, not just application imports.

## N+1 and speed assessment

No classic ORM/database N+1 was established in the parser/viewer path. The measured problems are repeated in-memory lookup work, duplicate queued I/O, whole-registry metadata access, and overfetching:

| Path | Observed work | Target after fix |
|---|---|---|
| Cold XCOM, 1,300 zones / one material / one energy | 1,300 coefficient lookups | One material/energy lookup |
| Five identical pending snapshot requests | Five worker signals | One physical read |
| First access to one BPF field | Metadata for all 104 fields | Metadata for the requested field |
| Small run, initial Derived load | 25 full field reads | Fields needed by active modules |
| Same run, repeated Derived load | **Zero additional field reads** | Preserve this successful reuse |
| BPF with LOG companion | Full companion parse, 26 field arrays retained in measured case | Stream only required companion data |

For the small fixture, a measured cold derived load was 22.4 ms and the warm load 0.40 ms. These timings demonstrate working cache reuse, not large-run performance. Conversion figures above were single warm-filesystem observations with default compression and should not become hard CI thresholds.

Prioritize fixing correctness and memory ownership, then profile representative large runs. Use read/lookup counts and peak RSS as stable regression criteria; use repeated median/p95 timings for performance comparisons. Benchmark HDF5 chunk layouts for both snapshot reads and zone traces before changing them. Existing HDF5 caches mean per-row writes should not automatically be described as repeated disk decompression/compression without measurement.

## Impeccable desktop interface assessment

**Implementation-integrity verdict:** The interface has a coherent product-specific structure: Parser → Viewer → Derived, explicit coordinates, scientific units, and familiar Qt controls. It needs workflow and accessibility corrections; a visual redesign is not the first priority.

These are provisional desktop implementation scores, adapting the Impeccable rubric to Qt rather than applying web/mobile-only rules:

| Dimension | Score / 4 | Evidence |
|---|---:|---|
| Accessibility | 2 | Some Qt form labels resolve correctly; several numeric controls are unnamed; native screen-reader pass remains outstanding |
| Performance | 2 | Background tasks, lazy advanced results, and useful caches exist; duplicate reads and eager loading remain |
| Window adaptivity | 1 | Requested laptop-sized geometry expands vertically and clips important controls |
| Appearance/theming | 3 | Shared light/dark tokens and coherent plots; stale field-item colors on theme switch |
| Implementation integrity | 3 | Clear scientific workflow; BPF routing and unit-family behavior diverge across layers |
| **Total** | **11 / 20** | **Significant targeted improvements needed; this is not a scientific-accuracy score** |

First-time users encounter LOG-only wording despite BPF capability, ambiguous preview/conversion provenance, and raw context strings such as `map=static_x`, `slice=zone`, and `weighting=auto`. Expert users benefit from linked plots and snapshot navigation but pay for duplicate reads. Accessibility-dependent users face unnamed inputs and clipped small-window controls.

Recommended UI sequence: `/impeccable harden` for names/recovery, `/impeccable adapt` for window constraints, `/impeccable clarify` for input/provenance/unit language, `/impeccable optimize` for request coalescing, `/impeccable colorize` for theme transitions, then `/impeccable polish`. These can be addressed individually or together; rerun `/impeccable audit` after fixes.

## Validation results

Environment: Python **3.12.13**, NumPy **2.5.4**, h5py **3.16.0**, PySide6 **6.12.0**, pyqtgraph **0.14.0**. SciPy was absent from the project venv, so experimental numerical tests exercised available fallback paths. Qt used `QT_QPA_PLATFORM=offscreen`; test subprocesses limited OpenMP/OpenBLAS threads to one.

All **59** discovered `test_*.py` modules were attempted in separate processes with a **100-second timeout per module**, plus the standalone smoke test:

| Outcome | Module count | Interpretation |
|---|---:|---|
| Clean exit 0 | 30 | 212 tests reported by these modules; successful process termination |
| SIGSEGV / return code -11 | 25 | 11 printed unittest `OK` before the native crash; others also had assertions/errors |
| Ordinary nonzero exit | 2 | `test_plasmon_ui_phase3.py`, `test_repository_consistency.py` |
| Timed out | 2 | `test_app_phase412.py`, `test_wave6_derived.py`; neither is counted as passed |
| Standalone smoke | Separate pass | `tests/smoke_test.py` completed successfully |

Available unittest summaries reported **342 tests, 27 failures, and 16 errors** across completed summaries. These counts include processes that later crashed; they are not a green-suite claim. Timeout processes may have completed additional tests without a final summary.

Cleanly passing modules included BPF pipeline, CLI errors, coordinates, reader, run status, streaming conversion, derived backend optimization/services, all fourteen Plasmon numerical phase modules, Transmission/XCOM, validation, BPF viewer axes, field-label UX, and several platform/ingest/performance checks.

| Failing area | Evidence and triage |
|---|---|
| Shock prerequisite | Plasmon UI phase 2/3/7 calculations raise `Shock result contains no finite track samples`; concrete application issue F09 |
| Units precision | `test_derived_stability.py` observes 90.011669615° for a π/2 request; F17 |
| Display-coordinate assertions | Payload/interaction suites compare displayed micrometers against native centimeters, showing a factor of 10,000; reconcile unit-aware expectations before treating these as calculation errors |
| Field counts / group labels | Historical viewer/shell assertions expect old item counts; derived warning test expects `selection` but UI uses `selection (1)` |
| Other UI contracts | Failures also concern lazy plot options/counts, empty-selection enablement, moving-mesh expectations, probe payloads, and stale-payload teardown; require targeted triage, not blanket tolerance relaxation |
| Documentation inventory | Repository-consistency test expects an older screenshot set than current `docs/index.html` |
| Qt lifetime | A faulthandler rerun of `test_app_phase41.py` again printed `OK` then segfaulted with no Python frame; root cause between application lifecycle and this Qt/runtime combination remains unresolved |

The existing full runner has a special payload-audit wrapper; this audit additionally ran that module directly to expose its current assertions. Do not equate the direct result with having executed every historical wrapper workflow.

### Reproduction commands

Run test modules against a disposable copy because historical tests generate outputs/settings. The following reproduces fresh-process discovery; the original audit retained each subprocess's full output separately.

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY'
import os, subprocess, sys
from pathlib import Path
env = dict(os.environ, QT_QPA_PLATFORM='offscreen',
           PYTHONPATH=str(Path('src').resolve()),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
failed = False
for module in sorted(Path('tests').glob('test_*.py')):
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'unittest', 'discover',
             '-s', 'tests', '-p', module.name], env=env, timeout=100)
        print(module.name, result.returncode, flush=True)
        failed |= result.returncode != 0
    except subprocess.TimeoutExpired:
        print(module.name, 'TIMEOUT', flush=True)
        failed = True
raise SystemExit(int(failed))
PY
```

## Calculation-quality test directions

Use independent expected values and conservation properties, in addition to source-to-source comparisons:

| Contract | Required regression |
|---|---|
| Units | 1 eV → 11604.518 K; 1 J/cm³ → 0.001 GPa; 1 cm → 10,000 μm; alias/canonical equality and display round trips |
| Time | Strictly defined duplicate policy, finite monotonic time, nonnegative timesteps; E/e/D/d extraction equivalence |
| Pressure | Thermal versus total pressure named explicitly; radiation-dominated LOG/BPF parity |
| Energy | Constant-power integral equals cumulative-energy difference; nonzero initial energy, restricted windows, missing-data policy |
| Geometry | Analytic planar thickness and cylindrical chord cases, material subsets, mesh deformation, and invalid-cell rejection |
| XRD | Known isotropic compression → d/d0 and Q; unreachable reflection reported unavailable, independent modules preserved |
| Spectroscopy | Known signed LOS velocity and thermal width; observation-side symmetry, zero-temperature policy, invalid input isolation |
| Shock/WaveFront | Analytic moving front on stationary and moving meshes; no-front case; irregular time spacing and sparse observations |
| Experimental spectra | Grid preservation, normalization area, detailed-balance/conservation checks where applicable, convergence with quadrature/grid refinement, and cached/uncached equivalence |
| Provenance | Full-frame LOG/BPF alignment, file-replacement invalidation, packaged resource identity, experimental policy/model labels in exports |

The source already records NRL formula provenance. Use the [NRL Plasma Formulary publication page](https://www.nrl.navy.mil/News-Media/Publications/NRL-PlasmaFormulary/) and the repository's `NRL_Formulary_2023.pdf` when defining independent coefficient/domain tests. This audit does not infer experimental model accuracy from agreement with its own fixtures or article-specific calibration.

## Practices to preserve

- LOG conversion already has a streaming iterator and partial-final-block metadata; extend that protection consistently to BPF.
- BPF aliases use HDF5 hard links, avoiding duplicate field storage.
- Reader hyperslabs support snapshots and time traces; keep narrow I/O paths narrow.
- Explicit coordinate centers/edges, units, and field provenance are a strong basis for trustworthy results.
- Warm raw-data reuse eliminated repeated field reads in the measured case.
- Background execution, generation checks, cooperative cancellation, and lazy advanced modules are useful foundations.
- Experimental Plasmon/Transmission features are hidden by default, and quick-look assumptions are surfaced in result warnings.

## Suggested implementation order

1. **Preserve data:** F01; add failure-injection tests before changing conversion behavior.
2. **Make calculations/source identity consistent:** F02–F10; fix source contracts and availability handling before tuning numerical performance.
3. **Restore an effective release gate:** F14, including native process exits and the current units/UI contracts.
4. **Make the desktop workflow usable:** F11–F13, F15, F24.
5. **Reduce measured repeated work and memory:** F18–F23; retain cold/warm read counts, RSS, and timing baselines.
6. **Harden experimental numerical/distribution paths:** F16–F17 and F25, then validate wheel-installed behavior and model convergence.

Source-size and duplication observations: `helios_analysis/workspace.py` is about 5,700 lines, viewer `main_window.py` about 4,400, and old workspace/transmission copies remain in `src`. Extracting form/unit adapters and module presentation contracts after the correctness fixes would reduce drift; this is follow-up architecture work, not a prerequisite for resolving the concrete defects above.
