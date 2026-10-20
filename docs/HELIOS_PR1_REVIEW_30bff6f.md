# PR #1 review at 30bff6f

**Recommendation: request changes. Four actionable findings remain in changed code: one P1 and three P2.** The original reproductions are largely corrected, but the timeout change introduces a new hang and adjacent parser, capability, and cache cases remain uncovered. A separately identified, pre-existing XRD calculation defect also leaves the plan's W06 acceptance incomplete.

- PR: <https://github.com/SpookyPike/Helios_Parser/pull/1>
- Reviewed head: `30bff6fd9b40da1d42348e3153d46501f043af31`
- Base: `76634468fb41fe9f3ca044a76879d91ee8ab70af` (`master`)
- Scope: current 150-path PR inventory, the preceding full reviews, all four follow-up commits after `339abf8`, and their surrounding parser, reader, cache, derived-analysis, calculation, desktop UI, and test-tooling paths. This pass independently ran 16 affected and neighboring modules and targeted failure reproductions; it does not claim a fresh independent validation of every physical model or a native screen-reader walkthrough.
- GitHub: current-head [Windows targeted/native/portable validation](https://github.com/SpookyPike/Helios_Parser/actions/runs/39476594689) passed; PR is open and mergeable. The hosted workflow does not run the POSIX timeout scenario below.

## Findings in changed code

### R1 — P1 — Kill the process group before the unbounded communicate call

**Location:** [scripts/run_test_suite.py:283–286](../scripts/run_test_suite.py#L283).

After SIGTERM's one-second grace period expires, the exception handler calls `process.communicate()` without a timeout. SIGKILL is now below that call. A direct test process that ignores SIGTERM, or a surviving descendant that retains the output pipe, prevents the call from returning and makes the kill unreachable. A permanently hung process consequently defeats the runner's module deadline and blocks subsequent entries and fixture cleanup.

**Verified reproduction:** run a process that ignores SIGTERM, keeps stdout, and finishes naturally after 2.5 seconds, with an entry timeout of 0.15 seconds. The runner takes **2.519 seconds**, and its log contains the process's natural-completion message. It should have killed the group after approximately 1.15 seconds. The reproduction deliberately uses a finite process so the review itself cannot hang or leave an orphan. Replacing that finite sleep with an infinite wait would leave the runner blocked.

The new regression test redirects the descendant's streams to `DEVNULL`, allowing `communicate()` to return as soon as the wrapper exits; it therefore bypasses this exception branch.

**Suggested fix:** on grace expiry, send SIGKILL to the group before draining/reaping the process. Preserve the separate cleanup for descendants that survive a promptly exiting wrapper. Every wait on this failure path must remain bounded until termination has actually been requested.

**Test direction:** cover a TERM-resistant direct process, a TERM-resistant descendant retaining inherited stdout, and the redirected-stream child already covered. Put the regression under an independent outer deadline so a broken runner cannot hang its own test. Assert timely return, no natural late completion/write, a passing next entry, and successful fixture cleanup.

### R2 — P2 — Reject contradictory bytes in incomplete BPF record markers

**Location:** [src/helios_parser/bpf.py:563–569](../src/helios_parser/bpf.py#L563).

An opening-marker fragment shorter than four bytes is accepted unconditionally. Once the opening marker is complete, the function validates its declared length but accepts any remaining tail shorter than the full record, without checking available closing-marker bytes. A fragment already contradicting the expected marker is corruption, even when its remaining bytes are absent.

**Verified reproduction:** use second-snapshot relative record 1 of the tracked Fe BPF. Retain one byte of its opening marker and flip that byte: accepted, recovering one frame. Separately retain the complete opening marker and payload plus one closing-marker byte, and flip that closing byte: also accepted. Passing the latter file to `write_hdf5(..., overwrite=True)` **replaces an existing sentinel destination** instead of rejecting the corrupt input.

The valid partial ionization-fraction cases at relative records 70 and 72 now recover correctly; this is a distinct corruption-detection gap in the same recovery path.

**Suggested fix:** compare every available opening-marker and closing-marker prefix with the corresponding prefix of `struct.pack('<i', expected)`. Missing bytes may be recoverable; existing contradictory bytes must fail.

**Test direction:** test every one-, two-, and three-byte marker prefix at several record types. Genuine truncations must recover complete preceding frames. A mutation in any available marker byte must fail and preserve a sentinel destination. Include fraction records and fixed-width records.

### R3 — P2 — Apply weighting prerequisites to XRD as well as Spectroscopy

**Location:** [src/helios/services/derived/analysis.py:581–598](../src/helios/services/derived/analysis.py#L581); static [XRD contract:206](../src/helios/services/derived/analysis.py#L206).

The new parameter-dependent capability checks run only for `contract.name == 'spectroscopy'`. XRD uses the same selectable electron-density/electron-column weighting, but its contract requires only density and zone width. When electron density is absent, the new loader supplies a NaN placeholder, all weights become zero, and XRD is still advertised as available without a missing-input reason.

**Verified reproduction:** remove `fields/electron_density` from a temporary copy of the Cu HDF5 and request only XRD through `compute_analysis_result()`. Width weighting produces `available=True` with one layer. Electron-column weighting produces **`available=True`, zero layers, and `reason=None`**. Module warnings can mention invalid weights, but the capability state fails to identify the missing electron-density input.

**Suggested fix:** resolve required inputs from the module and selected weighting/filter settings through a shared capability resolver, before computation. Gate every affected module independently with a specific missing-input reason; preserve modules using supported weightings.

**Test direction:** cross missing density/electron density with all supported weightings for XRD, Spectroscopy, and applicable Plasmon modes. Assert module availability, missing-input reasons, finite supported results, and the UI's unavailable state. Retain the current independent-module tests.

### R4 — P2 — Distinguish an absent optional coordinate from an evicted coordinate

**Location:** [src/helios/services/derived/common.py:446–447](../src/helios/services/derived/common.py#L446).

`has_dynamic_radius` means that some dynamic coordinate exists, but the new cache-completeness predicate requires both center and edge arrays whenever that flag is true. `HeliosRun` explicitly supports a canonical dynamic-center dataset without an edge dataset or legacy `radius` alias, returning `None` for the unavailable edge coordinate. Such a file can never satisfy the warm-cache predicate.

**Verified reproduction:** in a temporary copy of the Cu HDF5, store its existing center coordinates as `grid/dynamic_coordinate_center` and remove the legacy `fields/radius` alias. The reader supports this file and returns valid centers with no dynamic edges. After one initial `load_run_data(..., requested_fields=())`, **three warm loads reopen HDF5 three times** despite no eviction or requested-field change.

This concerns repeated shared-loader calls/reassembly. It does not assert that every UI snapshot change necessarily reloads the dataset; the controller also has a dataset cache.

**Suggested fix:** record availability/loading state separately for each coordinate, including a cached known-absent result. Reopen for an evicted source-present array, while treating a source-absent optional array as complete. Alternatively derive the missing coordinate when the reader's geometry contract permits it.

**Test direction:** retain the actual-eviction tests for width, centers, and edges, and add center-only, edge-only, legacy-alias, and static-only input cases. Warm repeats should perform zero reader opens and reads. An evicted existing coordinate should be refilled without rereading retained histories.

## Verified existing calculation issue still outside the promised acceptance

### XRD reciprocal-space Q is still tied to reflection-angle reachability

**Location:** [src/helios/services/derived/xrd.py:101–112](../src/helios/services/derived/xrd.py#L101), public-service validation at [analysis.py:709](../src/helios/services/derived/analysis.py#L709), and the helper-only [test:269](../tests/test_derived_services_phase4.py#L269).

This defect predates the PR; it is recorded separately from the four changed-code findings. The implementation plan explicitly promises that Q remains defined from the valid spacing ratio when a requested reflection angle is unreachable. The current XRD implementation computes Q only under `valid_bragg`, leaving it NaN when `sin(theta') > 1`. Public result validation then raises `ValueError: XRD Q is not finite.` rather than returning an unavailable angle with a valid Q and warning.

**Verified reproduction:** use the existing synthetic derived fixture with matching field capabilities at snapshot 4, where one layer has compression ratio 4. At initial angle 5°, the public service returns two layers and finite Q values. At 85°, the same service raises `XRD Q is not finite`. The PR's new unreachable-angle test invokes only the private metrics helper and checks its angle/warning; it never exercises this public validation path.

**Suggested fix:** compute `Q = Q0 / (d/d0)` wherever the compression/spacing ratio is valid. Gate only the angle-dependent products on the Bragg condition and keep the warning. Exercise the assembled service result and unrelated requested modules.

**Test direction:** test reachable and unreachable angles through `compute_analysis_result()`, with finite Q, `None` angle/shift when unreachable, and successful result validation. Include a snapshot refresh and the full-run Q trace.

## Status of the previous eight reproductions

| Previous finding at 339abf8 | Current evidence |
|---|---|
| Geometry eviction | Width/center/edge eviction regressions pass. R4 above concerns a different, genuinely absent-coordinate case. |
| Valid partial fraction records | Relative records 70 and 72 now recover one complete frame. R2 above concerns contradictory marker bytes. |
| Explicit LOG final-frame domain error | Fractional final cycle now raises the domain error rather than selecting fallback. |
| In-flight revision retirement | The real blocked-loader reproduction now leaves zero retired raw-array entries and zero retained raw-array bytes after resuming. |
| Same-snapshot placeholder reuse | A newly activated Spectroscopy module is recomputed, has profile plots, and matches fresh velocity 0. |
| Spectroscopy weighting inputs | Missing electron density under electron-column weighting is now unavailable with a specific reason; default mass/density regression passes. R3 above concerns the same dependency rule omitted for XRD. |
| Redirected-stream timeout child | That original child no longer writes after timeout. R1 above is a new hang in the inherited-stream/direct-process path. |
| Missing-source residual budget | Assembled residual tests pass: absent source channels yield unavailable residual; explicitly zero, observed source histories yield an available residual with coverage 1. |

## Other remaining plan and acceptance limits

- **Negative reported LOG timestep:** the unchanged reproduction still accepts `-1e-9`. Both parsers check finiteness without enforcing the plan's nonnegative-domain contract. A domain violation must also propagate from an explicit final companion.
- **BPF completion provenance:** [hdf5.py:874–880](../src/helios_parser/hdf5.py#L874) still maps structural integrity to run state `complete`. The plan requires structural completeness separately from simulation-completion evidence, using `unknown` when that evidence is absent. Existing tests assert the weaker contract.
- **Demand-driven loading:** the controller still selects optional-field sets by wavefront activation, while the loader eagerly reads all six core histories. This does not meet the plan's complete union-of-module-requirements/read-count contract for Shock-only work. No new runtime speedup claim is established by this review.
- **F12:** automated labels, offscreen layout/theme checks, and hosted Windows checks do not replace a native Orca/keyboard walkthrough. This remains explicitly open.
- **Performance evidence:** earlier P05/P07/P08 speedup conclusions remain inconclusive. The current grouped XCOM and viewer request-coalescing tests pass; no new coefficient-lookup N+1 regression was found in those checked paths.

## Executed validation

**All 16 isolated modules passed: 154 tests reported, 152 passed and two explicitly skipped.** The skips require Windows destination-lock behavior and an unavailable optional XCOM compute backend. No module failed, timed out, or crashed. The full suite was not rerun, following the targeted-test preference.

| Module | Tests | Skips |
|---|---:|---:|
| `test_accessibility_controls.py` | 4 | 0 |
| `test_bpf_pipeline.py` | 17 | 0 |
| `test_cache_lifecycle.py` | 14 | 0 |
| `test_conversion_atomicity.py` | 7 | 1 |
| `test_derived_backend_optimization.py` | 13 | 0 |
| `test_derived_module_contract.py` | 2 | 0 |
| `test_derived_services_phase4.py` | 14 | 0 |
| `test_preheat_energy_budget.py` | 6 | 0 |
| `test_pressure_contract.py` | 4 | 0 |
| `test_reader.py` | 10 | 0 |
| `test_snapshot_numeric_domains.py` | 4 | 0 |
| `test_test_runner.py` | 12 | 0 |
| `test_transmission_xcom.py` | 34 | 1 |
| `test_viewer_wave2.py` | 9 | 0 |
| `test_w11_layout.py` | 2 | 0 |
| `test_w11_theme_copy.py` | 2 | 0 |

Manifest and logs: `/tmp/helios-pr1-review-30bff6f/manifest.json`.

Executed reproductions:

- `/tmp/helios_review_30bff6f.py`: TERM-resistant direct process, corrupt partial markers and sentinel conversion, unreachable-angle public service, XRD missing weighting input, and repeated warm opens for center-only coordinates.
- `/tmp/helios_review_339_repros.py`: rerun on the current head for placeholder activation, valid fraction truncation, real in-flight retirement, partial core fields, and negative timestep.
- `/tmp/helios_review_339_edges.py`: rerun on the current head for explicit final-companion errors, redirected-stream timeout cleanup, and Spectroscopy's missing weighting input.

`git diff --check origin/master...HEAD` passed. Local HEAD, remote branch, and GitHub PR head match. Existing scientific-input edits and generated output directories were preserved. This review adds only this report, changes no production code, and posts no GitHub review/comments.

## Follow-up implementation

The five calculation/runtime issues above were fixed in the working tree after the reviewed head. These changes are intended as follow-up commits on the existing PR branch; the review findings remain anchored to `30bff6f`.

- **R1:** the runner sends SIGKILL to the process group as soon as the TERM grace period expires, then drains output with a timeout. A POSIX regression exercises a TERM-resistant process with inherited output under an independent watchdog.
- **R2:** incomplete opening and closing marker prefixes are checked against the expected record marker. Valid truncation remains recoverable; contradictory marker bytes are rejected. The regression covers one-, two-, and three-byte prefixes and confirms valid preceding snapshots remain readable.
- **R3:** XRD now applies the selected weighting and density-filter prerequisites used by Spectroscopy. The missing-electron-density electron-column case reports XRD unavailable with a specific reason.
- **R4:** center and edge coordinate source presence are cached separately. A center-only source is considered warm after its available center array is loaded, while source-present evicted arrays still trigger refill.
- **W06 XRD acceptance:** Q is computed from the valid spacing ratio even when the shifted Bragg angle is unreachable. A public assembled-result test verifies finite Q and unavailable angle/shift values.

Targeted validation through `scripts/run_test_suite.py` passed four isolated modules: `test_bpf_pipeline.py` (18 tests), `test_derived_backend_optimization.py` (14 tests), `test_derived_services_phase4.py` (15 tests), and `test_test_runner.py` (13 tests). No full suite was run. Detailed logs and manifest: `/tmp/helios-pr-fix-review-final/`.
