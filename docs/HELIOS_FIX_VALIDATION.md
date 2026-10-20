# HELIOS audit fix validation

**Updated:** 2026-10-20
**Branch:** `fix/helios-parser-audit`
**Plan:** [HELIOS_FIX_IMPLEMENTATION_PLAN.md](HELIOS_FIX_IMPLEMENTATION_PLAN.md)
**Audit:** [HELIOS_AUDIT_2026-10-20.md](HELIOS_AUDIT_2026-10-20.md)

This is validation evidence, not a release sign-off. The user asked to prefer focused tests and run the full suite only when needed. After the two earlier full-profile failures were fixed and verified with focused tests, the full profile was rerun because W14 requires post-fix integration evidence. The latest run, after the cache-retirement lifecycle change, completed all 78 entries with zero failures, zero timeouts, zero missing entries, and two explicit test skips. The Linux full-profile gate passes. Native Linux accessibility/lifecycle/layout/theme checks, the paired UI no-regression comparison, and Linux PyInstaller frozen-runtime smoke pass. Windows build/runtime validation and active-task/view memory limits remain open release work.

## Focused validation

All tests below ran from the disposable current-source tree created by `scripts/run_test_suite.py`, using Python 3.12.13, NumPy 2.5.4, h5py 3.16.0, PySide6 6.12.0, pyqtgraph 0.14.0, and Linux x86_64. The complete targeted performance manifest is `/tmp/helios-w14-performance-final/manifest.json` in the validation environment.

| Target | Result | Evidence |
|---|---|---|
| Native Wayland UI | PASS | `test_accessibility_controls.py`, `test_qt_lifecycle.py`, `test_w11_layout.py`, and `test_w11_theme_copy.py`: 12 tests passed with `QT_QPA_PLATFORM=wayland`; exact resize assertions wait for compositor acknowledgement. The size request test also passed with `QT_QPA_PLATFORM=xcb` (1 test). |
| Linux frozen runtime | PASS | PyInstaller 6.22.3 built the repository spec into `/tmp/helios-frozen-dist-20261020`; the GUI stayed alive for a 5-second offscreen startup smoke, and the frozen CLI converted the BPF fixture to H5D schema 2.1. Both bundled scientific-resource JSON files had hashes matching their source files. This is Linux evidence; the Windows build/runtime remains untested. |
| `test_cache_lifecycle.py` | PASS | Stable bucket identity and clear lifecycle |
| `test_derived_backend_optimization.py` | PASS | Field-read and derived-backend work counts |
| `test_plasmon_energy_grid_cache.py` | PASS | Exact spectrum-grid cache behavior |
| `test_reader.py` | PASS | Reader metadata and narrow-read behavior |
| `test_run_revision.py` | PASS | Same-path replacement/revision behavior |
| `test_transmission_xcom.py` | PASS | Grouped cold attenuation, scalar agreement, and XCOM contracts |
| `test_viewer_wave2.py` | PASS | Pending-read coalescing, stale payload rejection, and render batching |
| `test_app_phase3.py` | PASS | Viewer field-list payload and UI behavior |
| `test_wave6_derived.py` | PASS (earlier W13 run) | 48 tests; 271.124 s; clean exit |
| `test_app_phase412.py` | PASS (earlier W13 run) | 11 tests; 221.097 s; clean exit |
| `test_app_phase3.AppPhase3Tests.test_open_existing_hdf5_loads_viewer_and_mode_switching_preserves_viewer_state` | PASS (earlier W13 run) | Focused app/viewer lifecycle case |

The first W14 targeted attempt exposed two test-harness issues, both corrected and rerun successfully: `test_transmission_xcom.py` lacked the `local_h5_50al_root` fixture declaration, and the stale-field UI test omitted field-axis metadata while deleting its Qt thread before normal window shutdown. A follow-up focused run also passed `test_bpf_pipeline.py` (14 tests, 34.289 s), `test_wave4_ingest.py` (3 tests, 0.260 s), and `test_performance_comparator.py` (9 tests, 0.011 s).

A later W14 review batch ran ten targeted suites in fresh disposable processes: numeric notation, snapshot numeric domains, pressure semantics, derived-module contracts, accessibility controls, BPF desktop workflow, packaged resources, Qt lifecycle, W11 layout, and theme refresh. All ten passed after correcting the packaged-resource test to compare against the actual source under `src/helios/services/derived/resources/`; its first run had failed because it expected a nonexistent repository-root copy. The final per-suite manifest is `/tmp/helios-pr-review-targeted/manifest.json`, and the corrected resource-suite rerun is `/tmp/helios-pr-review-packaged/manifest.json` in this validation environment.

The pre-fix full-profile manifest is `/tmp/helios-final-reverification-20261020/manifest.json`: 78 entries executed, 76 passed, and 2 failed. `test_viewer_phase2.py` sampled the redraw counter before a queued refresh had settled; an explicit idle wait was added. `test_wave6_derived.py` exposed a conflict over manual preheat navigation controls; manual mode now enables its slider/spin/time controls, consistent with the mode-sync contract. The moving-mesh hover, manual preheat-mode, and phase-7 initial-selection tests all passed in focused follow-up runs. The post-fix full run then passed all 78 entries; after the subsequent cache-retirement change, a second full run also passed all 78 entries. The latest authoritative manifest is `/tmp/helios-final-retirement-20261020/manifest.json` (76 unittest modules, one payload wrapper, one smoke entry; zero failures/timeouts/missing entries; two explicit skips).

## Paired performance evidence and limits

Paired before/after profiles and comparator outputs are archived under [`docs/performance/w14/`](performance/w14/). Measurements ran on the same Linux x86_64 host (12th Gen Intel Core i7-12700H, Python 3.12.13, NumPy 2.5.4, h5py 3.16.0, PySide6 6.12.0; 64 GiB RAM where the harness recorded it). Short cases used ten alternating pairs; slower loading and conversion cases used five pairs. Baseline source was `origin/master` at `76634468`. The first candidate profiles used branch commit `a319dca`; the P05 fast-path rerun used the working-tree source fingerprint based on that commit, before the optimization commit was created. Source fingerprints, fixture records and raw samples are in the archived JSON.

| Case | Result | Evidence |
|---|---|---|
| P01 grouped XCOM (1,300 zones, 3 materials × 4 energies) | PASS | 10 pairs; median 60.817 → 1.148 ms; lookup calls 5,200 → 12; optical-depth scalar-reference error 2.842e-13; exact comparator output in `xcom-comparison-10.json` |
| P04 demand-driven derived loading (50Al stabilized HDF5) | PASS | 5 pairs; retained arrays 203.1 → 70.4 MB; optional arrays 125.0 MB → 0; peak RSS delta 473.0 → 205.5 MB; elapsed 0.546 → 0.247 s; all 12 output checksums equal |
| P05 BPF companion conversion (Cu BPF + same-stem LOG) | PASS; timing improvement inconclusive | After a narrow parser fast path for cycle/time and the laser-source column, 5 paired fresh-process runs produced exact output equivalence and no metric exceeded the 10% slowdown threshold. Median total time was 5.359 → 5.316 s (paired 95% interval 0.971–1.174, inconclusive), first-write 1.399 → 1.167 s (0.814–1.180, inconclusive), and peak RSS delta 124.0 → 112.2 MB (0.904 ratio, improved). The earlier pre-fast-path regression remains archived for traceability. The LOG fixture hash is recorded in the fixture JSON; this is the locally edited fixture version present during the run. |
| P06 cache budgets / retention | Cache cap and synchronous open/analyze/retire cache ownership PASS; active-task/view RSS remains outside this bound | The existing 30-cycle profile recorded the 512 MiB aggregate cache cap and first eviction at cycle 9. A new ten-cycle profile uses the same `retire_run_revision()` API called by Derived on run change/close, loading and analyzing the 226 MB stabilized 10ns fixture each cycle. Each cycle removed 24 cache entries, including 65.7 MB of dataset arrays and 29.2 MB of analysis results; matching retained bytes and aggregate cache bytes were zero after every retirement. From cycles 2–10, post-retirement RSS stayed within a 0.26 MiB range; the first cycle was 8.2 MiB higher during initialization. The report, fixture hash, environment, Git revision, and working-tree source fingerprint are in `cache-retirement-analysis-10.json`. Active task/view references remain outside this cache-only measurement. |
| P07 HDF5 access | PASS for comparator; timing inconclusive | 10 pairs; all representative median ratios were within about 1.3% of baseline and classified inconclusive. The earlier exploratory zone-trace result remains a one-off and is not a speed claim. This profile predates the RAM-capture fallback, so `ram_bytes` is null in its environment record. |
| P08 UI responsiveness | PASS for latest-render correctness and proposed 10% slowdown gate; timing improvement inconclusive | 10 alternating fresh-process pairs on native Wayland. Each process performs 100 warm slider bursts (8 values each) and 165 tab switches against the same 701-snapshot, 226 MB fixture. All 20 processes exited zero and every final render matched the last requested snapshot. Candidate/baseline median ratios: settling p50 0.956 (95% paired bootstrap interval 0.889–1.032), settling p95 0.981 (0.916–1.059), event-loop gap p50 0.969 (0.915–1.040), and event-loop gap p95 0.989 (0.864–1.044); all are inconclusive, with no median slowdown exceeding 10%. The original baseline had a reproducible Qt shutdown crash: the queued-open close test failed after `OK` in 3/3 fresh native runs. Applying only the `c9a4372` slider proxy-style ownership fix to that baseline passed 3/3; the candidate lifecycle module passed all four tests on Wayland. The corrected comparison baseline overlay and fingerprints are recorded in the JSON profiles. |

The paired artifacts are evidence for their named fixtures and source fingerprints only; they are not CI timing limits. P05 now passes the proposed regression threshold with exact output equivalence, though speed and first-write improvements remain inconclusive. P06 now verifies cache-owned release across repeated open/analyze/retire cycles; this does not claim an RSS ceiling for arrays retained by active tasks or views.

## Earlier exploratory performance evidence

The work-count tests above passed, including the 1,300-zone grouped-XCOM case and duplicate reader-request coverage. An exploratory `bench_hdf5_access.py` run compared the existing Cu HDF5 layout (`64 × 256` chunks) to a newly converted candidate (`16 × 300` chunks), at 461 snapshots × 300 zones. Median measurements (32 accesses per category, one process/run) were:

| Access pattern | Existing layout | New conversion layout | Candidate / existing |
|---|---:|---:|---:|
| Snapshot read | 0.00776 ms | 0.00677 ms | 0.87× |
| Zone trace | 0.01192 ms | 0.02503 ms | 2.10× |
| 64-snapshot scrub | 0.479 ms | 0.454 ms | 0.95× |

This single exploratory run shows a zone-trace regression for the candidate layout and is too small to support a speed claim. It is not a paired baseline/candidate acceptance run: the script does not record ten alternating process-level pairs, uncertainty, memory, or source fingerprints. No application HDF5 chunking change is proposed from this result. Native UI responsiveness remains open. Deterministic work-count improvements and paired P01/P04/P05 measurements are recorded separately above.

## Finding map

| Finding | Implementation area / representative change | Regression evidence | Status |
|---|---|---|---|
| F01 Atomic conversion | W01 atomic HDF5 publication (`46afa14`) | `test_streaming_conversion.py`, conversion atomicity coverage | Targeted implementation tests landed; W14 cross-platform overwrite-failure gate pending |
| F02 Source-energy budget | W04 cumulative source semantics (`936b3fb`) | `test_wave6_derived.py`, Preheat energy-budget cases | Focused suite passed |
| F03 Stale run revision | W05 revision-scoped caches/tasks (`54e6255`, `18879d0`) | `test_run_revision.py`, viewer stale-result tests | Targeted suites passed |
| F04 BPF validity | W02 frame/domain validation (`06a8215`, `840d147`) | `test_bpf_pipeline.py`, `test_snapshot_numeric_domains.py` | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F05 Partial final snapshot | W02 partial-frame reporting (`b13c542`) | `test_bpf_pipeline.py` truncation cases | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F06 Pressure semantics | W03 explicit plasma/total pressure (`365b8b9`) | BPF/reader/derived pressure parity cases | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F07 Companion alignment | W04 full-cycle alignment (`936b3fb`) | `test_wave4_ingest.py`, companion alignment cases | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F08 Scientific notation | W02 numeric parsing (`06f8c6d`) | Numeric-domain and LOG parser cases | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F09 Independent modules | W06 capability-aware analysis (`764ec90`, `7ee6a35`), no-detection validation (`0ffa1c4`) | `test_wave6_derived.py`, `test_derived_stability.py`, `test_v1_release_gate.py` | No-detection and independent-analysis checks passed; the full run's separate manual-preheat assertion passed after focused correction |
| F10 Standalone BPF analysis | W06 prerequisite/partial-data handling (`4cd5cf8`, `1e03e84`) | BPF partial metadata and analysis cases | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F11 Desktop BPF workflow | W09 preview/convert routing (`e1bb5a1`) | BPF desktop workflow tests | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F12 Accessible names | W11 named controls (`b588c31`, `efae132`) | UI accessibility checks | Full/offscreen suite and native Wayland accessibility checks passed; Windows native check pending |
| F13 Window sizes | W11 scrollable tall pages (`f8f2eca`) | Layout/accessibility checks | Native Wayland layout/lifecycle suite passed after waiting for compositor resize acknowledgement; XCB size request passed; Windows native check pending |
| F14 Release test gate | W00 runner and W13 lifecycle (`bbe0cb5`, `c9a4372`), fixture declarations (`b1712a6`, `8e57270`) | `test_test_runner.py`, `test_qt_lifecycle.py`, full profile | Post-fix full profile executed 78/78 entries with 0 failures, 0 timeouts, 0 missing entries, and 2 explicit skips; Linux full-profile gate passes |
| F15 Display units | W10 canonical/display conversion (`bb65a93`, `f83e1fe`) | Viewer unit tests and app phase3 | Focused `test_app_phase3.py` passed |
| F16 Spectrum grids | W12 exact axes (`33407e1`) | `test_plasmon_energy_grid_cache.py` | Passed |
| F17 Radian precision | W10 canonical controls (`e875a8c`) | Unit round-trip tests | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F18 Streamed companion | W04 streamed source and alignment (`936b3fb`), narrow LOG companion parser | `test_wave4_ingest.py`, `test_bpf_pipeline.py` | 17 focused tests passed; follow-up P05 comparator passed exact output and the 10% regression threshold; runtime and first-write comparisons are inconclusive |
| F19 Demand loading/memory | W07 lazy fields and budgets (`03fa549`, `df0a422`), revision retirement | `test_derived_backend_optimization.py`, `test_cache_lifecycle.py`, ten-cycle cache-retirement benchmark | Work-count/budget tests and paired derived-load/RSS comparison passed; ten-cycle open/analyze/retire removed all revision-owned cache entries. Active task/view arrays remain outside the cache budget |
| F20 Cache clearing | W05 stable bucket identity (`aa4b475`) | `test_cache_lifecycle.py` | Passed |
| F21 Grouped XCOM | W12b grouped lookups (`5048a9c`) | `test_transmission_xcom.py`, including 1,300 zones | Passed; paired P01 confirms 5,200 → 12 lookups/run and a 52.9× median speedup on the archived fixture |
| F22 Pending I/O | W08 request coalescing (`25f66e7`) | `test_viewer_wave2.py` | Passed |
| F23 Per-field metadata | W08 lazy metadata (`d99955f`) | `test_reader.py` | Passed |
| F24 Theme contrast | W11 theme refresh (`6d5effb`) | Theme/accessibility tests | Full/offscreen suite and native Wayland theme checks passed; Windows native theme review pending |
| F25 Packaged resources | W12c package data (`8fc08e9`) | Installed wheel/sdist smoke and Linux frozen-runtime resource check | Installed package smoke passed; Linux PyInstaller spec bundled both scientific-resource files with matching hashes and the frozen CLI smoke passed. Windows frozen-runtime resource check pending |

## Remaining release checks

- Full dynamically discovered release suite and smoke: post-fix run passed all 78 entries (76 modules, one wrapper, one smoke); two tests explicitly skipped. This satisfies the Linux full-profile gate, but is not by itself release sign-off.
- P05 BPF companion paired timing review: passed after the narrow parser path; speedup remains unproven because the paired confidence intervals are inconclusive.
- Cache-retirement evidence: ten large-fixture open/analyze/retire cycles cleared all 24 revision-owned entries per cycle and retained zero cache bytes. After the initial cycle, post-retirement RSS stayed within a 0.26 MiB range over cycles 2–10; RSS remains above its starting point because native allocators retain pages. The benchmark does not claim an RSS ceiling for arrays retained by active tasks or views.
- The P01/P04/P05/P07 paired profiles, comparisons and fixture manifests are archived; P01/P04 passed, P07 was inconclusive, and the final P05 comparison passed after optimization. The earlier P05 failure is retained as a before-optimization profile.
- Native Linux Wayland accessibility/lifecycle/layout/theme checks passed (12 tests), and the size request test passed on XCB. P08 passed the ten-pair comparison and latest-render check; all timing intervals remain inconclusive, but none showed a median slowdown above the plan's 10% review threshold. The original baseline shutdown crash is isolated to Qt proxy-style ownership: the self-contained queued-open test failed after `OK` with SIGSEGV in 3/3 original-baseline processes, passed 3/3 with only the `c9a4372` slider-style file applied, and passed 3/3 on the candidate. The full candidate lifecycle module passed 4/4. Reproduction details are in `ui-shutdown-lifecycle-reproduction-linux-wayland.json`; benchmark diagnostics and paired profiles are in the adjacent `ui-shutdown-diagnostic-*` and `ui-responsiveness-*-paired-10-*` files. Windows native checks remain pending.
- A focused Wayland rerun after making the queued-open lifecycle fixture self-contained passed 17 tests with two platform-specific skips. The Windows-only locked-destination overwrite test is now explicit, and `.github/workflows/windows-validation.yml` runs it with native accessibility/layout/lifecycle tests plus an isolated portable build/GUI/CLI smoke. The workflow has not yet run on a Windows host.
- The Windows portable README and validator now expect the current H5D schema 2.1. `build_windows_portable.py` and its validator accept external output paths so CI does not overwrite tracked release artifacts.
- Linux PyInstaller frozen-runtime smoke passed: GUI startup, BPF-to-H5D CLI conversion, and scientific-resource hash checks. Windows frozen-runtime build/smoke remains pending.

The current branch has passing focused evidence for the performance-sensitive reader, cache, derived, viewer, spectrum-cache, and XCOM paths. The post-fix Linux W14 full-profile gate, P06 cache-retirement gate, P08 paired UI no-regression comparison, and Linux frozen-runtime smoke pass. Overall release completion remains open because Windows native/build checks and active-task/view memory limits remain unverified; the P08 timing improvement itself is inconclusive.
