# HELIOS audit fix validation

**Updated:** 2026-10-20
**Branch:** `fix/helios-parser-audit`
**Plan:** [HELIOS_FIX_IMPLEMENTATION_PLAN.md](HELIOS_FIX_IMPLEMENTATION_PLAN.md)
**Audit:** [HELIOS_AUDIT_2026-10-20.md](HELIOS_AUDIT_2026-10-20.md)

This is targeted validation evidence, not a release sign-off. The user asked to prefer focused tests and run the full suite only when needed. Accordingly, the W14 full-profile run was interrupted before completion and has not been restarted. The interrupted runner produced no complete manifest and is not counted as a pass or fail.

## Focused validation

All tests below ran from the disposable current-source tree created by `scripts/run_test_suite.py`, using Python 3.12.13, NumPy 2.5.4, h5py 3.16.0, PySide6 6.12.0, pyqtgraph 0.14.0, and Linux x86_64. The complete targeted performance manifest is `/tmp/helios-w14-performance-final/manifest.json` in the validation environment.

| Target | Result | Evidence |
|---|---|---|
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

## Paired performance evidence and limits

Paired before/after profiles and comparator outputs are archived under [`docs/performance/w14/`](performance/w14/). Measurements ran on the same Linux x86_64 host (12th Gen Intel Core i7-12700H, Python 3.12.13, NumPy 2.5.4, h5py 3.16.0, PySide6 6.12.0; 64 GiB RAM where the harness recorded it). Short cases used ten alternating pairs; slower loading and conversion cases used five pairs. Baseline source was `origin/master` at `76634468`. The first candidate profiles used branch commit `a319dca`; the P05 fast-path rerun used the working-tree source fingerprint based on that commit, before the optimization commit was created. Source fingerprints, fixture records and raw samples are in the archived JSON.

| Case | Result | Evidence |
|---|---|---|
| P01 grouped XCOM (1,300 zones, 3 materials × 4 energies) | PASS | 10 pairs; median 60.817 → 1.148 ms; lookup calls 5,200 → 12; optical-depth scalar-reference error 2.842e-13; exact comparator output in `xcom-comparison-10.json` |
| P04 demand-driven derived loading (50Al stabilized HDF5) | PASS | 5 pairs; retained arrays 203.1 → 70.4 MB; optional arrays 125.0 MB → 0; peak RSS delta 473.0 → 205.5 MB; elapsed 0.546 → 0.247 s; all 12 output checksums equal |
| P05 BPF companion conversion (Cu BPF + same-stem LOG) | PASS; timing improvement inconclusive | After a narrow parser fast path for cycle/time and the laser-source column, 5 paired fresh-process runs produced exact output equivalence and no metric exceeded the 10% slowdown threshold. Median total time was 5.359 → 5.316 s (paired 95% interval 0.971–1.174, inconclusive), first-write 1.399 → 1.167 s (0.814–1.180, inconclusive), and peak RSS delta 124.0 → 112.2 MB (0.904 ratio, improved). The earlier pre-fast-path regression remains archived for traceability. The LOG fixture hash is recorded in the fixture JSON; this is the locally edited fixture version present during the run. |
| P06 cache retention (10 repeated loads) | Diagnostic only; lifecycle gate still open | RSS was available, but inspection shows this benchmark retained ten distinct raw-data revisions and did not execute an application run-retirement hook. Its rising cache/RSS values cannot be treated as a valid open/analyze/retire result. The archived profile exposes this gap; rerun after the lifecycle benchmark models actual retirement. |
| P07 HDF5 access | PASS for comparator; timing inconclusive | 10 pairs; all representative median ratios were within about 1.3% of baseline and classified inconclusive. The earlier exploratory zone-trace result remains a one-off and is not a speed claim. This profile predates the RAM-capture fallback, so `ram_bytes` is null in its environment record. |

The paired artifacts are evidence for their named fixtures and source fingerprints only; they are not CI timing limits. P05 now passes the proposed regression threshold with exact output equivalence, though speed and first-write improvements remain inconclusive. P06 needs a valid retirement benchmark before its memory gate can pass.

## Earlier exploratory performance evidence

The work-count tests above passed, including the 1,300-zone grouped-XCOM case and duplicate reader-request coverage. An exploratory `bench_hdf5_access.py` run compared the existing Cu HDF5 layout (`64 × 256` chunks) to a newly converted candidate (`16 × 300` chunks), at 461 snapshots × 300 zones. Median measurements (32 accesses per category, one process/run) were:

| Access pattern | Existing layout | New conversion layout | Candidate / existing |
|---|---:|---:|---:|
| Snapshot read | 0.00776 ms | 0.00677 ms | 0.87× |
| Zone trace | 0.01192 ms | 0.02503 ms | 2.10× |
| 64-snapshot scrub | 0.479 ms | 0.454 ms | 0.95× |

This single exploratory run shows a zone-trace regression for the candidate layout and is too small to support a speed claim. It is not a paired baseline/candidate acceptance run: the script does not record ten alternating process-level pairs, uncertainty, memory, or source fingerprints. No application HDF5 chunking change is proposed from this result. Full W14 performance acceptance remains open for paired timings, conversion peak RSS, cache-retention RSS, and native UI responsiveness. Deterministic work-count improvements are verified by the targeted suites; timing and memory claims remain unproven.

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
| F09 Independent modules | W06 capability-aware analysis (`764ec90`, `7ee6a35`) | `test_wave6_derived.py`, unavailable-analysis cases | `test_wave6_derived.py` passed |
| F10 Standalone BPF analysis | W06 prerequisite/partial-data handling (`4cd5cf8`, `1e03e84`) | BPF partial metadata and analysis cases | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F11 Desktop BPF workflow | W09 preview/convert routing (`e1bb5a1`) | BPF desktop workflow tests | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F12 Accessible names | W11 named controls (`b588c31`, `efae132`) | UI accessibility checks | Existing targeted work passed during implementation; native platform check pending |
| F13 Window sizes | W11 scrollable tall pages (`f8f2eca`) | Layout/accessibility checks | Existing targeted work passed during implementation; native platform check pending |
| F14 Release test gate | W00 runner and W13 lifecycle (`bbe0cb5`, `c9a4372`) | Targeted runner; clean process tests | Full profile intentionally incomplete; release gate remains open |
| F15 Display units | W10 canonical/display conversion (`bb65a93`, `f83e1fe`) | Viewer unit tests and app phase3 | Focused `test_app_phase3.py` passed |
| F16 Spectrum grids | W12 exact axes (`33407e1`) | `test_plasmon_energy_grid_cache.py` | Passed |
| F17 Radian precision | W10 canonical controls (`e875a8c`) | Unit round-trip tests | Existing targeted work passed during implementation; not rerun in this W14 pass |
| F18 Streamed companion | W04 streamed source and alignment (`936b3fb`), narrow LOG companion parser | `test_wave4_ingest.py`, `test_bpf_pipeline.py` | 17 focused tests passed; follow-up P05 comparator passed exact output and the 10% regression threshold; runtime and first-write comparisons are inconclusive |
| F19 Demand loading/memory | W07 lazy fields and budgets (`03fa549`, `df0a422`) | `test_derived_backend_optimization.py`, cache budget tests | Work-count/budget tests and paired derived-load/RSS comparison passed; cache-retirement RSS gate still needs a lifecycle-valid run |
| F20 Cache clearing | W05 stable bucket identity (`aa4b475`) | `test_cache_lifecycle.py` | Passed |
| F21 Grouped XCOM | W12b grouped lookups (`5048a9c`) | `test_transmission_xcom.py`, including 1,300 zones | Passed; paired P01 confirms 5,200 → 12 lookups/run and a 52.9× median speedup on the archived fixture |
| F22 Pending I/O | W08 request coalescing (`25f66e7`) | `test_viewer_wave2.py` | Passed |
| F23 Per-field metadata | W08 lazy metadata (`d99955f`) | `test_reader.py` | Passed |
| F24 Theme contrast | W11 theme refresh (`6d5effb`) | Theme/accessibility tests | Existing targeted work passed during implementation; native theme review pending |
| F25 Packaged resources | W12c package data (`8fc08e9`) | Installed wheel/sdist resource smoke | Earlier packaging smoke passed; frozen-runtime resource check unavailable/not run |

## Remaining release checks

- Full dynamically discovered release suite and smoke: incomplete by user preference; rerun only if required for a release decision.
- P05 BPF companion paired timing review: passed after the narrow parser path; speedup remains unproven because the paired confidence intervals are inconclusive.
- Cache-retirement RSS: pending a corrected ten-cycle benchmark that actually retires each run revision. The first diagnostic retained distinct raw revisions and is not considered an acceptance measurement.
- The P01/P04/P05/P07 paired profiles, comparisons and fixture manifests are archived; P01/P04 passed, P07 was inconclusive, and the final P05 comparison passed after optimization. The earlier P05 failure is retained as a before-optimization profile.
- Native GUI latency/accessibility and lifecycle checks on supported desktop platforms: pending; offscreen Linux tests only.
- PyInstaller/frozen-runtime resource smoke: pending because the supported frozen build environment was not exercised.

The current branch has passing focused evidence for the performance-sensitive reader, cache, derived, viewer, spectrum-cache, and XCOM paths. W14 and the release completion gate are not marked complete because the required paired timing/memory and native/frozen checks remain outstanding.
