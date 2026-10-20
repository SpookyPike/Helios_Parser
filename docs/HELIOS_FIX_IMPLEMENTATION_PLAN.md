# HELIOS audit fix implementation plan

**Prepared:** 2026-10-20. **Source:** [HELIOS audit](HELIOS_AUDIT_2026-10-20.md). **Scope:** all 25 findings, F01–F25.

**Repository review:** 2026-10-20, branch `master`, HEAD `76634468fb41fe9f3ca044a76879d91ee8ab70af`. This revision incorporates a fresh source/Git review and the requested Windows-to-Linux Git hygiene step. The audit's executed test results remain historical evidence; this plan review did not rerun the full suite.

This is an execution plan, with proposed contracts and acceptance criteria. Its APIs and test filenames are implementation targets, not changes already made. Preserve the current working-tree edits. Deliver the work as small, independently reviewable commits in the order below; do not combine it with a general rewrite of the large UI modules.

## Execution status — 2026-10-20

The implementation packages and W14 evidence are on [PR #1](https://github.com/SpookyPike/Helios_Parser/pull/1); the finding-by-finding results are in [HELIOS_FIX_VALIDATION.md](HELIOS_FIX_VALIDATION.md). The Linux launcher was exercised directly from a temporary Unix-filesystem copy because this workspace is on a FUSE NTFS mount that reports the indexed executable as mode 0644 and does not persist `chmod`; the release-assets shell launcher also passed through `bash`. The Windows PowerShell launcher smoke, focused native checks, and portable build/validation passed in [Windows run #39465724852](https://github.com/SpookyPike/Helios_Parser/actions/runs/39465724852). P05 and P08 timing improvements are inconclusive, which meets the plan's acceptance when no repeatable median slowdown exceeds 10%.

F12 remains open: the automated Qt accessibility-name checks pass, but a native Orca/keyboard walkthrough has not been verified. The remaining unchecked finding is intentional; do not interpret the other checked findings as a release sign-off while this acceptance item is outstanding.

## 0. Git hygiene pass — G00, before implementation

The application was previously developed under Windows. Treat line-ending noise and generated artifacts explicitly before making functional changes.

### Verified Git state

| Item | Observed state |
|---|---|
| Tracked modifications | **438**, all unstaged |
| Content comparison | All 438 are **CRLF/LF-only differences** from their HEAD blobs; no other byte differences after normalizing CRLF to LF |
| Staged changes | None |
| Untracked entries | Six directory/file entries: the audit, this plan, three output directories, and `src/helios_parser.egg-info/` |
| Attributes/editor policy | No `.gitattributes` or `.editorconfig` |
| Local Git configuration | `core.filemode=false`; the queried `core.autocrlf`, `core.eol`, and `core.safecrlf` settings were absent |
| Mode/rename changes | None reported by `git diff --summary` |

This is a line-ending-dirty checkout, not evidence that hundreds of functional fixes have landed. Do not use ordinary diff line counts as an implementation baseline.

**Owner/files:** repository policy; proposed `.gitattributes`, `.editorconfig`, targeted `.gitignore` additions, and a hygiene manifest outside generated release trees.

1. Capture HEAD, porcelain status, the index/staged state, and original file hashes before normalization. Save a recoverable copy/patch of existing changes and the two untracked planning documents. Never use `git reset --hard`, `git clean`, blanket checkout, or `git add .` for this pass.
2. Classify tracked source/document text, Windows/Linux launchers, raw scientific fixtures, binary assets, release archives, and generated reports. The tracked `.rhc` files are ASCII text with CRLF, but they are scientific input fixtures; byte changes still need explicit fixture handling.
3. Add repository-controlled attributes: LF for Python, Markdown, HTML, SVG, JSON, CSV, TOML, YAML, and shell source; LF for PowerShell source unless a tested launcher requires CRLF; CRLF for any future `.bat`/`.cmd`. Keep raw `.log`/`.rhc` fixtures byte-preserving (`-text`) initially, and binary formats such as BPF/HDF5, fonts, raster images, ICO icons, PDFs, executables, and compressed archives `-text`. Use explicit overrides after the general text rule. Do not change global Git configuration.
4. Add matching editor defaults. Confirm the two tracked `.sh` launchers have the intended executable bit in the index: `core.filemode=false` can hide local mode differences. Change index modes only for launchers demonstrated to need direct execution, and test their documented invocation.
5. Apply `git add --renormalize` only to reviewed text paths after reviewing attributes. Inspect the staged diff with and without end-of-line ignoring. Keep normalization/policy changes separate from functional commits. For the three raw-fixture modifications, preserve their current bytes and record the remaining differences until a dedicated LF/CRLF fixture test establishes the intended canonical bytes; do not silently bundle fixture conversion into source normalization.
6. Add an ignore rule for generated `*.egg-info/`. Classify the three untracked output directories before adding narrow ignore rules or moving them to external scratch storage. Do not delete user artifacts. Existing tracked reports/archives remain tracked unless a separately reviewed retention change removes them; `.gitignore` alone does not untrack them.
7. Leave audit/plan documents explicitly reviewable and include them only in their intended documentation change. Verify both content and Git status after the pass; every remaining dirty path must have an explanation. A fully blank status is not required if preserving raw fixtures/user artifacts leaves classified changes.

**Acceptance:** the functional source baseline is distinguishable from line-ending noise; no original bytes are lost; binary/fixture hashes are unchanged unless a dedicated fixture change was tested; the staged policy/normalization diff has no hidden functional edits; Windows and Linux launcher invocations remain valid. The policy commit must not contain unrelated reports or release ZIP changes.

**Commit boundary:** `chore: define cross-platform text and artifact policies`, followed by a separate normalization-only commit if normalized tracked blobs actually change. This document requests the hygiene work; the plan-review turn itself edits only this plan.

## 1. Delivery order

| Package | Work | Findings | Prerequisites |
|---|---|---|---|
| G00 | Cross-platform Git hygiene and recoverable baseline | Repository prerequisite | None |
| W00 | Reliable discovery, fixture preflight, baseline reporting | F14, first part | G00 |
| W01 | Atomic, non-destructive conversion | F01 | W00 |
| W02 | Numeric notation, BPF validation, truncation status | F04, F05, F08 | W01 |
| W03 | Consistent pressure fields and legacy interpretation | F06 | W01; coordinate shared writer edits with W02 |
| W04 | Correct laser budgets and streamed companion alignment | F02, F07, F18 | W01, W02; serialize shared writer/schema edits with W03 |
| W05 | Run revisions and cache clearing | F03, F20 | W00 |
| W06 | Capability-aware loading and independent module results | F09, F10 | W03, W04, W05 |
| W07 | Demand-driven fields and memory budgets | F19 | W05, W06 |
| W08 | Narrow reader metadata and coalesced I/O | F22, F23 | W05 |
| W09 | BPF desktop workflow and provenance | F11 | W01–W06 |
| W10 | Display-unit parity and canonical control values | F15, F17 | W03, W06 |
| W11 | Accessible, adaptable, theme-consistent desktop UI | F12, F13, F24 | W09, W10 |
| W12 | Experimental grid, attenuation, and resource fixes | F16, F21, F25 | W12a/W12c: G00 + W00; W12b: W00, coordinate with loader changes |
| W13 | Qt lifecycle recovery and remaining test triage | F14, second part | Instrumentation starts at W00; finalize after UI/controller changes |
| W14 | Full acceptance and release evidence | All | W01–W13 |

Execute sequentially by default. W05, W08, and the W12 subcommits have fewer dependencies, but shared-file edits must still be coordinated. W03/W04 share conversion code; W06/W07 share loading code; W09/W10/W11 share UI code. Keep one integration owner for each of those file groups.

**First implementation slice:** G00 → W00 → W01 → W02. Start W13's minimal crash diagnosis during W00; do not defer all runtime investigation until the end.

## 2. Contracts to implement

These choices resolve the audit's cross-layer ambiguities. Record them in developer/schema documentation alongside the corresponding code change.

### Conversion and normalized time

- A failure before the publication commit point never changes an existing destination or any selected input/companion. A new final output becomes visible only after successful validation and close. Post-publication notification failure cannot undo that commit or truthfully be reported as a failed conversion.
- Preserve the existing BPF iterator's **finite, nondecreasing** time contract while extending it to the writer. The current check uses `< previous_time`, so duplicates are currently accepted. Reject decreases/nonfinite values; do not invent a new absolute-time origin restriction. Validate integral/nondecreasing cycles and finite/nonnegative reported timesteps separately.
- Preserve duplicate-time frames without silently dropping/averaging them. Differentiation and other operations requiring positive time spacing must gate that module/time series and explain the issue. Snapshot-local analyses remain usable. A stricter export mode could be added separately; it is not required to fix the bypassed check.
- Keep corruption distinct from an incomplete final write. Earlier corrupt records, marker mismatches, and invalid lengths are errors. A recoverable incomplete final snapshot may preserve preceding valid snapshots with explicit truncation status.
- An intact BPF export does not by itself prove the simulation completed. LOG currently uses `completed` from footer evidence; BPF currently writes `complete` unconditionally. Record structural completeness separately, use `unknown` without completion evidence and `truncated` for a recognized partial final snapshot, and preserve raw legacy status/source for provenance. Do not blindly map BPF `complete` to simulation `completed`.

### Pressure and schema compatibility

The current writer declares **H5D schema 2.0**. Target **2.1** for the explicit new fields and corrected legacy alias, but describe this accurately as a **versioned semantic correction**, not a purely additive change: existing BPF `pressure` changes meaning. Update release/schema compatibility notes and every alias consumer/test together.

| Name | Meaning in new output |
|---|---|
| `plasma_pressure_j_cm3` | `ion_pressure + electron_pressure` |
| `total_pressure_j_cm3` | `ion_pressure + electron_pressure + radiation_pressure` |
| `pressure` | Compatibility alias of **total pressure**, consistent with current LOG calculation and derived total-pressure use |
| Existing BPF `pressure_j_cm3` | Retains its documented **plasma-pressure** meaning; it must not silently become total pressure |

Use HDF5 hard links for equivalent names. Preserve source components and attach explicit quantity/semantics metadata. For legacy files, derive total pressure from all three available components or trust an explicitly documented total-pressure field. Do not reinterpret an ambiguous `pressure` field solely from its name. If necessary inputs are absent, expose an unavailable total-pressure capability and a reason. Do not assume radiation pressure is zero without explicit evidence.

### Cumulative source and energy budgets

- Introduce `DerivedRunData.laser_source_j_g` for cumulative energy. Keep `laser_deposition_j_g_s` for instantaneous power.
- Migrate constructors, fixtures, plots, exports, and consumers in the same package. Retire the misleading internal `laser_source_j_g_s` attribute; do not keep a live alias that continues to advertise the wrong units. The external HDF5 `laser_source` name remains supported with correct J/g metadata.
- For a planar Lagrangian target, define reference cell areal mass at the selected interval's first snapshot, `m_i = rho_i * dx_i`. The cumulative-source budget is `sum(m_i * (E_i(t) - E_i(t_start)))`, in J/cm². Integrate instantaneous deposition over the same interval and reference masses. Do not count the two estimates twice.
- Missing selected values produce an unavailable/incomplete budget and coverage information, not a silently successful zero. If mass changes exceed a documented numerical tolerance, flag the conservation inconsistency.
- Keep geometry explicit. If the current budget result cannot express a geometry's correct energy measure, mark that budget unavailable with an explanation rather than reporting a slab J/cm² value for it. Broader cylindrical/spherical energy-budget modeling is a separate extension.
- The sampled BPF integral is explicitly energy accumulated since the first exported snapshot; do not claim knowledge of pre-export deposition. Preserve the exact LOG cumulative quantity only when every frame is aligned.

### Run identity, capabilities, and result availability

- Add one immutable `RunRevision` shared by viewer payloads, `RunContext`, derived datasets, result keys, and tasks. It contains canonical path, file fingerprint, optional conversion ID, and a session open token generated once for each explicit open/reload.
- The fingerprint includes available device/inode, size, and nanosecond modification time. The open token prevents old cache reuse on explicit reopen even if dimensions/stat values match. It stays stable during snapshot changes and module activation so warm reuse still works.
- Generate a conversion ID for newly written files. Legacy files use fingerprint plus open token. Do not compute whole-file hashes during normal opens.
- Distinguish `not_loaded`, `missing_input`, `outside_domain`, `no_detection`, `available`, and `error`. Available-in-file and loaded-in-memory are separate capabilities.
- A module lacking prerequisites returns an unavailable typed result and a reason. Invalid shared input and unexpected programming errors remain errors; do not turn every exception into an empty result.
- No shock detection is a normal `no_detection` result. Zero ion temperature with positive ion mass yields zero thermal Doppler width. For XRD, compute reciprocal-space Q from the valid spacing ratio even if the requested reflection angle is unreachable; report the angle as unavailable with a warning.

## 3. Work packages

### W00 — Establish trustworthy test reporting

**Owner/files:** test infrastructure; `scripts/run_test_suite.py`, `scripts/run_payload_audit.py`, `tests/_test_bootstrap.py`, `tests/_viewer_test_utils.py`; new `tests/test_test_runner.py` and a fixture manifest/helper.

**Verified constraint:** there are 59 `test_*.py` modules, but the three commonly used HDF5 files under `outputs/hdf5/` and root `5Fe+4.9TW+light.log` / `10ns+10Si+60Al+15Si+4.27TW.log` are ignored, local-only files. No HDF5 files are tracked. The Cu LOG under `new_data/25Cu+1.87TW/` and two BPF inputs are tracked. A clean checkout cannot reproduce the audit simply by discovering all tests.

1. Replace the hard-coded full-module list with discovery of `tests/test_*.py`, plus the smoke entry. Preserve the existing CLI's `--profile`, `--only`, and `--list` behavior; reject unknown `--only` names rather than succeeding with zero work.
2. Keep each module in a fresh process and retain the payload wrapper as an explicit execution override. Verify that its test-method inventory matches discovery so new methods cannot disappear from coverage.
3. Continue after failures. Write one log per entry and a JSON manifest recording command, runtime, return code/signal, timeout, skips, and environment versions. Return nonzero if any required entry fails, crashes, times out, or was not executed.
4. Inventory fixture requirements by suite, with paths, dimensions, provenance, and hashes. Make the portable `core` suites use small tracked or deterministically generated fixtures. Preflight the historical large-run fixtures for `full`; absent required data produces a distinct missing-fixture failure, never a passing skip. Preserve large scientific validations until equivalent coverage is demonstrated. Materialize a disposable test tree from the reviewed working tree plus the explicit fixture manifest, not only from `git clone` or `git archive`.
5. Give each subprocess isolated output/settings/application-data directories and restore the same baseline settings before every test. `_viewer_test_utils.py` already redirects QSettings but `reset_test_settings()` clears only one application name; inspect shell session settings too. Preserve inputs read-only; copy only files a test mutates. Tests hard-code `ROOT/outputs`, so environment variables alone do not isolate their writes: update their path helpers or run them in a disposable tree.
6. Make feature flags explicit per test entry. `_test_bootstrap.py` defaults `HELIOS_ENABLE_EXPERIMENTAL=1`; a supposed production run with the variable merely absent is therefore experimental. Set both flags explicitly to empty/false for production-gate entries and enable them for historical experimental suites. Do not require every experimental UI assertion to pass with its tab intentionally hidden.
7. Add a configurable 120-second default module timeout and named heavy-test overrides, provisionally up to 300 seconds. These are proposed runner defaults, not measured runtime guarantees. Keep heavy suites in `full`; record and diagnose overrides instead of treating a longer timeout as a fix.

**Acceptance:** runner tests cover success, assertion failure, signal termination, timeout, unknown selection, missing fixtures, feature-flag selection, and omitted discovery. Report current results using the same outcome categories, without assuming the old counts will repeat or requiring all historical suites to turn green before W01. The audit's historical baseline was 30 clean module exits, 25 crashes, two ordinary failures, two timeouts, and a separate passing smoke test.

**Commit:** `test: discover all suites and report subprocess outcomes`.

### W01 — Preserve inputs and publish outputs atomically

**Owner/files:** conversion; `src/helios_parser/hdf5.py`, `cli.py`; new `tests/test_conversion_atomicity.py`, existing CLI/streaming tests.

1. Resolve the full input set first. Compare destination with requested source and all selected companions, using resolved paths and existing-file identity to catch symlinks/hardlinks.
2. Replace early unlinking with a temporary sibling output. Both LOG and BPF writers operate on that temporary path while metadata/progress continue to describe the intended final output.
3. Close all HDF5 handles, validate required groups/counts/status without rereading all field arrays, and only then publish. For overwrite, use atomic replacement. For no-overwrite, use a no-clobber atomic publication primitive; a same-filesystem hard link followed by temporary unlink is an option. If unsupported, fail clearly instead of falling back to partial copying into the final path. Document symlink handling and retain appropriate destination/umask permissions rather than unintentionally publishing a temporary file's mode.
4. Internal writers currently emit `done` themselves. Move the authoritative completion notification to the publisher; suppress the internal success events. Pre-publication progress-callback exceptions abort and clean the temporary file. After publication, log/contain completion-notification exceptions instead of reporting that a successfully committed conversion failed.
5. `write_hdf5()` currently has no cancellation-token argument. W01 must handle existing exceptions and failure injection without claiming cooperative cancellation already exists. Add explicit cancellation checkpoints separately if W13 needs them, preserving atomic cleanup; otherwise defer close until the parse finishes.

**Acceptance:** sentinel destination bytes survive malformed input and injected pre-publication failures. A callback error after publication preserves the committed output and successful conversion outcome, with the notification error logged. Input/companion identity is rejected before writing. A destination appearing during no-overwrite conversion is preserved. Check permissions and Windows replacement behavior, including a destination opened with sharing modes that deny replacement; not every open Windows handle prevents replacement.

**Commit:** `fix: publish converted HDF5 without destroying existing data`.

### W02 — Validate numeric input and BPF completeness consistently

**Owner/files:** parser; `tokenizer.py`, `parser.py`, `document.py`, `bpf.py`, `hdf5.py`, `model.py`, `reader.py`; new `tests/test_numeric_notation.py`; extend BPF/run-status/coordinate tests.

1. Extract `[EeDd]` exponents as complete tokens; retain omitted-E support. Apply common numeric-domain checks after parsing cycle metadata.
2. Make the writer consume the validated BPF iterator. Reuse its first yielded snapshot for schema discovery, avoiding a second extraction.
3. Validate time/cycle sequence, finite strictly increasing node positions, positive cell widths, finite nonnegative density, and field-specific domains. Zero density is permitted only where safe division/domain handling already treats vacuum explicitly. Do not blanket-reject negative EOS pressures.
4. Add scan/layout outcomes distinguishing valid snapshots, final partial bytes/records, recognized trailer, and corruption. Verify the documented all-zero 240-byte trailer against fixture payloads, not length alone.
5. Recover only an incomplete final snapshot following at least one complete valid frame. Mid-record EOF can be recoverable; invalid markers or corruption earlier in the stream remain fatal. An incomplete first snapshot fails.
6. Persist honest normalized status/counts and warnings through reader and GUI. Preserve existing complete-frame data bit-for-bit.

**Acceptance:** E/e/D/d fixtures parse identically; decreasing/nonfinite timestamps fail, while duplicate-time frames are retained and time-dependent module guards are tested. The eight-frame fixture cut after five records of frame eight produces seven retained frames, truncation metadata, and no false completion claim. Test inside-record EOF, corrupt middle records, valid trailer, and valid one-frame input. Trace `extract_snapshot` cycle rounding too: reject fractional/nonfinite cycle values before rounding can hide invalid input.

**Commit sequence:** notation/domain checks → shared validated iterator → truncation/status handling.

### W03 — Make pressure semantics source-independent

**Owner/files:** field contracts; `bpf.py`, `parser.py`, `hdf5.py`, `reader.py`, derived `common.py`, `models.py`; schema documentation and BPF/derived tests.

1. Implement the pressure naming table and schema 2.1 metadata. Keep BPF plasma pressure unchanged under its existing canonical name; point the legacy `pressure` alias to the new total-pressure field.
2. Update derived loading and consistency checks to select explicit total pressure or reconstruct it from components. Introduce a separate plasma-pressure capability if consumers need it.
3. Implement the legacy interpretation path without rewriting files. A legacy ambiguity produces a visible warning/capability limitation.
4. Update field labels, quantity metadata, display-unit registration inputs, and schema/release documentation together.

**Acceptance:** components `[1, 2, 4]` produce plasma `3` and total `7` through LOG and BPF conversion, and the public alias resolves identically. Test legacy LOG/BPF files, missing radiation components, and an explicitly documented total field. Compare pointwise absolute and relative error; avoid a single global scale that can hide radiation differences.

**Commit:** `fix: define explicit plasma and total pressure contracts`.

### W04 — Correct source budgets and stream aligned companion data

**Owner/files:** energy/companion integration; derived `models.py`, `common.py`, `preheat.py`; parser `document.py`, `parser.py`, `hdf5.py`; app `parser_controller.py`; relevant fixtures/tests.

1. Migrate the cumulative-energy attribute and split budget helpers into rate integration and cumulative-energy difference. Update every caller and label in the same change.
2. Implement reference-mass/window/missing-data rules from section 2. The current observed-energy helper uses time-varying `rho * width`, while source budgets integrate that same factor. Update and test both sides together if adopting a fixed reference mass; changing only cumulative-source weighting would create another inconsistent residual. Report unsupported budget geometry explicitly while preserving supported non-budget Preheat diagnostics.
3. Replace `parser.parse(log_path)` with a companion snapshot stream carrying only cycle/time, source precision, cumulative laser source, and needed metadata. Keeping one semantic snapshot in memory is acceptable; retaining the full unrelated field history is not.
4. Compare all times using the rounding interval implied by each original LOG time token, plus floating-point roundoff; compare cycles exactly. Do not use a broad absolute tolerance or align only endpoints. Do not interpolate in this fix.
5. Always compute the BPF sampled-source dataset. Stream LOG source candidates to a temporary HDF5 staging dataset; select them only after complete alignment succeeds. Alignment also gates companion-derived run metadata: the current writer imports LOG region/material/header information before the source override check. Do not discard only the laser candidate while retaining metadata from an unrelated run. Stage companion metadata or perform a bounded first validation pass before publishing grid/region/material groups.
6. Preserve the current explicit-input error policy: a malformed requested LOG fails, even if a BPF was discovered; a malformed automatically discovered LOG for a requested BPF may fall back to standalone BPF with a warning. Move optional header inspection inside that fallback handling. On valid but unaligned companions, discard candidate fields and unverified companion metadata; use standalone BPF metadata/capability limitations and record why. Never silently reinterpret the requested file or reuse wrong-run EOS/initial-state metadata. A more permissive requested-LOG fallback would be a separate explicit product option.
7. Replace error `readlines()` with a bounded tail read, initially limited to 64 KiB/20 displayed lines. A binary BPF error never attempts text-tail rendering.

**Acceptance:** cumulative `[0,10,20] J/g` at `[0,1,2] ns` and unit areal mass gives `20 J/cm²`; equivalent rate integration agrees. Cover a nonzero starting cumulative value, subwindows, nonuniform dt, missing zones, and mass-conservation warnings. Shift one interior time/cycle while preserving endpoints and assert no LOG override. Assert `HeliosParser.parse()` is not called by BPF-primary conversion and peak companion memory does not grow with unrelated field histories.

**Commit sequence:** energy contract/budgets → full alignment → streamed companion/fallback/tail handling.

### W05 — Give every open run a revision and preserve cache identities

**Owner/files:** runtime/caches; `helios/cache.py`, `runtime.py`, `tasks.py`; viewer `models.py`, `workers.py`, `controller.py`; app viewer bridge; derived `common.py` and analysis `controller.py`.

1. Add and propagate `RunRevision` using the contract above. Create the open token at the initiating controller, pass it to workers, and return the same token in payloads; do not independently regenerate it in the loader.
2. Replace path-only dataset/result/task keys with revision keys. Pass revision explicitly to `load_run_data`; preserve a fingerprint-based path-only compatibility entry point for existing direct callers, with documented refresh behavior.
3. On explicit reopen, retire previous-run ownership, cancel its derived tasks, and reject every late result/error/status that would affect the new run. Preserve warm reuse within one revision.
4. Change cache clearing to empty existing bucket objects without unregistering them. Give bucket removal a separate explicit operation if needed. Apply the same lifecycle policy to `clear_bucket`.

**Acceptance:** same-path/equal-shape replacement updates viewer and derived values, including a deliberately restored size/mtime on explicit reopen. Old tasks cannot apply to the new revision. Snapshot changes do not reload full fields. Publish→clear→publish→clear operates on the same registered bucket and reports correct stats.

**Commit sequence:** stable clearing → revision model/payloads → dataset/result/task migration.

### W06 — Isolate unavailable analyses and missing physical inputs

**Owner/files:** derived orchestration; `models.py`, `module_contract.py`, `common.py`, `analysis.py`, `physical_sanity.py`, `shock_tracking.py`, `xrd.py`, `spectroscopy.py`; analysis controller/workspace.

1. Extend `DerivedModuleContract` with explicit required fields and required material/initial-state metadata. The existing `capabilities_met()` method is a reusable seam, but active `_MODULE_CONTRACTS` do not declare the relevant requirements and `_compute_module_results()` does not enforce them. Populate and enforce those contracts; prepared WaveFront/interface/Preheat entries are placeholders, not completed implementations.
2. Add typed module availability/reason metadata to assembled results with backward-compatible defaults. `DerivedAnalysisResult` currently requires typed Shock/XRD/Plasmon/Transmission/Spectroscopy results, so do not casually replace them with `None`. Only hidden Plasmon/Transmission helpers already exist; add explicit unavailable-result constructors for other required types and update every renderer/export/validator that consumes them.
3. Make the loader construct a valid partial dataset when scientific inputs are missing. `DerivedRunData` currently requires density, velocity, temperatures, electron density, and charge arrays, and consumers call `np.asarray` directly. Update its type/absence contract and guard each module before any such conversion; empty or NaN-filled fabricated physical arrays are not a capability system. Never invent atomic weights, initial densities, or temperatures. W06 introduces absence handling; W07 distinguishes not-yet-loaded fields from missing fields.
4. Validate shared grid/loaded-array shapes before dispatch. Check geometry inference and standalone `track_shock_front` calls as well as `_compute_module_results`, since they run before the active module loop. Gate operations requiring strictly positive time intervals locally rather than rejecting all duplicate-time input. An absent shock does not stop XRD/spectroscopy; malformed shared shapes still fail explicitly.
5. Handle zero-temperature Doppler width and unreachable XRD angle according to section 2. Keep validators strict for truly impossible values.
6. Show the availability reason in the affected tab and exports. Prevent stale plots from a previous available result from remaining visible when the current result is unavailable.

**Acceptance:** BPF-only loading no longer raises `KeyError('atomic_weight')`; supported diagnostics work and other tabs state their missing inputs. Uniform no-shock input still computes independent modules. Test empty selection, partial metadata, zero temperature, unreachable angle, invalid shapes, and transitions between available/unavailable results.

**Commit sequence:** availability contract → capability-aware loading/dispatch → module-domain fixes/UI states.

### W07 — Load only requested fields and enforce cache byte budgets

**Owner/files:** derived loading/memory; `common.py`, `module_contract.py`, `models.py`, analysis `controller.py` and `workspace.py`; shared `cache.py`; viewer `controller.py`/`main_window.py`; app ownership/bridges if injecting a shared budget coordinator; benchmark helpers.

1. Add an explicit module/field requirement request to `load_run_data`. Internal callers provide the requested module set; retain an explicit eager compatibility mode for existing external callers/tests. Include `HeliosDerivedWorkspace._apply_plasmon_benchmark_preset`, which currently calls the full loader synchronously on the UI thread: reuse the loaded revision or perform its necessary metadata/field request in background work rather than leaving a hidden eager/UI-blocking path.
2. Load metadata once, then the union of requested required/optional fields. When a tab activates, extend/rebuild an immutable dataset snapshot sharing existing arrays; do not mutate a dataset still used by an in-flight task.
3. Keep field consistency checks lazy: evaluate a relation only when its operands are loaded. Opening Shock must not pull pressure, energy, or radiation histories solely to populate checks.
4. Add array/result byte accounting to cache statistics and eviction. Deduplicate shared ndarray base allocations within each accounting scope; do not charge a hard-linked alias twice after it has been canonicalized in memory.
5. Implement configurable byte limits and test them with small injected budgets. Choose shipping defaults from measured retained allocations on the manifest's representative runs and the intended desktop memory budget; the previous 512/128/128 MiB figures were unmeasured proposals, not repository facts or acceptance thresholds. Reject individually oversized cache entries. Report active arrays/task working memory separately because eviction cannot free externally retained objects.
6. Wire budget ownership explicitly. The repository currently creates separate `AnalyzerCacheSet` objects in viewer `main_window.py`, viewer `controller.py`, and analysis `controller.py`, plus the module-global session raw cache in `common.py`. A limit attached to one instance is not an application limit. Keep their ownership APIs compatible while registering all four scopes with an injected/shared session coordinator, or report per-owner limits honestly until that wiring is complete. Deduplicate allocations across registered owners when calculating the aggregate. Release retired run references and cancelled-task payloads; do not claim an RSS bound from cache limits alone.

**Acceptance:** instrument reads: Shock-only activation does not read unrelated energy/radiation/temperature histories unless its declared detector needs them. Repeated activation performs zero additional reads for already loaded requirements. Byte-budget tests use tiny configurable limits and shared views. Large-file RSS stabilizes after task completion and run eviction; temporary computation memory is reported separately.

**Commit sequence:** requirement-driven loading → byte accounting/coordinator → retention cleanup and measurements.

### W08 — Make narrow reads cheap and coalesce pending requests

**Owner/files:** reader/viewer I/O; parser `reader.py`; viewer `controller.py`, `workers.py`, `models.py`; reader/payload/performance tests.

1. Cache `FieldMetadata` per field. `get_field_metadata(name)` loads one entry; `list_field_metadata()` explicitly fills the registry. Avoid truthiness checks that mistake a partially filled cache for a complete inventory.
2. Track pending requests by `(revision, request_kind, field, normalized_selection)`. Coalesce equivalent full-field, snapshot, trace, and diagnostic requests; preserve one completion notification path per physical request.
3. Include request identity in error/completion signals so pending entries and busy counts are cleared exactly once. Clear pending state on success, failure, cancellation, and revision retirement.
4. Keep HDF5 access inside its owning worker thread. Do not add concurrent access to a shared handle to mask queueing costs.

**Acceptance:** first single-field access loads one metadata record; second access none; explicit inventory discovers all entries. Five equivalent pending requests cause one actual reader call. Error→retry works; stale completions cannot clear a newer revision's pending key or leave busy state stuck.

**Commit sequence:** per-field metadata → request identity/coalescing.

### W09 — Expose the actual BPF desktop conversion workflow

**Owner/files:** app workflow; `helios_app/main_app.py`, `parser_controller.py`, `session_state.py`; parser source-selection/preview helpers; app/BPF GUI tests and user guide.

1. Add “Open HELIOS input…” accepting LOG/BPF; route by actual suffix. Preserve the existing CLI entry points and existing recent-file behavior.
2. Introduce a backend preview helper shared with conversion source selection. Report primary source, companions, expected field registry, indexed frame count, validation scope, and whether geometry/material metadata is known. A fast header/first/last-frame preview must not label every frame valid or claim full companion alignment before scanning it; use a nullable validated count and `alignment not checked` state until conversion/full validation supplies evidence.
3. Extend preview payloads instead of faking LOG fields for BPF. Display “Unknown” for absent metadata, “Not yet validated” for unchecked claims, and explicit warnings for observed companion fallback/truncation. Keep the distinction between index count and valid count through status rendering.
4. Make preview results revision/request scoped so an older preview cannot enable conversion of the wrong file. Convert the source shown in the accepted current preview.
5. Ensure output naming, auto-open, overwrite errors, parse progress, and last-result navigation remain consistent for both formats.

**Acceptance:** GUI flows for LOG-only, BPF-only, aligned pair, mismatched pair, and malformed optional LOG. Preview provenance matches the selection policy and makes unchecked validation explicit; conversion updates it if full validation rejects the companion. BPF-only conversion proceeds without fabricated material metadata. Update startup/user documentation in this commit.

**Commit:** `feat: preview and convert LOG and BPF inputs in the desktop shell`.

### W10 — Unify displayed units and preserve physical control values

**Owner/files:** unit adapters; viewer `units.py`, `main_window.py`, `settings.py`; analysis `workspace.py`; unit/stability/BPF viewer tests.

1. Resolve quantity families from explicit field metadata and canonical aliases, validating native units. Unit spelling alone must not assign a physical meaning to an unknown raw BPF record.
2. Keep existing helper signatures usable by legacy callers, adding optional metadata/quantity arguments. Pass field metadata from the actual viewer pipeline.
3. Store canonical physical parameter values separately from spin-box formatting. Re-rendering/changing units does not write rounded values back into physical state or trigger scientific recomputation. User edits update canonical state once.
4. Standardize angle precision to at least six radian decimals; implement consistent angle/k and photon-energy/wavelength adapters. Where values are already persisted, keep canonical values rather than display strings; do not expand this fix into a new session-persistence feature.

**Acceptance:** alias/canonical parity for temperature, pressure, density, velocity, energy, and rates across maps/probes/lineouts/exports. π/2 entered in a six-decimal radian editor resolves to 90° within `5e-5°`; changing display units alone preserves the canonical value exactly. Test repeated unit cycles and unknown/raw quantities remaining native.

**Commit sequence:** metadata-based units → canonical control state/precision.

### W11 — Fix accessibility, window constraints, and theme transitions

**Owner/files:** desktop presentation; app `main_app.py`; viewer `main_window.py`, `style.py`; analysis `workspace.py`; GUI accessibility/layout tests.

1. Give minimum density, lower/upper zone bounds, and every other editable control a meaningful accessible name, units/description where needed, and label buddies. Include accessible names for icon-only actions and snapshot controls.
2. Keep the existing visual identity. Make essential actions/snapshot navigation fit the available client area; collapse secondary summaries, allow form-row wrapping, and set splitter proportions from available width with usable minimums.
3. Permit vertical scrolling in long control forms. Avoid horizontal form scrolling at the tested laptop sizes. Do not solve clipping by shrinking text below the established size.
4. Reapply semantic colors to existing field items on theme changes while preserving selection/filter state. Informational headings should meet a 4.5:1 text-contrast target in both themes.
5. Replace raw mode tokens in visible summaries with short labels and explain resolved automatic weighting/provenance. Keep detailed diagnostic values available without making them the main workflow.

**Acceptance:** requested 1024×768, 1280×800, and 1860×1120 client sizes are honored without minimum-size expansion; primary controls remain reachable and labels do not clip. Test light/dark startup and both switch directions, plus enlarged text. Assert accessibility names through `QAccessible`; perform one native keyboard/screen-reader walkthrough. Use one batched visual review and at most one confirmation pass, following Impeccable's bounded review guidance.

**Commit sequence:** accessible names → layout adaptation → theme/copy consistency.

### W12 — Fix experimental numerical/performance/distribution paths

Deliver as three separate commits; keep experimental visibility gates enabled only by the existing explicit flags.

**W12a / F16 — Preserve arbitrary energy grids.** Owner: `plasmon_spectrum.py` and spectral tests. Keep compact linear-grid keys only when reconstruction preserves the actual input values; the current near-linearity tolerance/rounded endpoints must not erase interior differences. Encode full canonical float64 values for other grids, including count/shape, and reconstruct exactly. A digest is acceptable only if the original immutable grid is also retained for evaluation. Reject nonfinite or unsupported multidimensional inputs explicitly. Test same-endpoint/count linear, nearly linear, quadratic, logarithmic, and adaptive grids above 32 points; cached and uncached spectra must agree.

**W12b / F21 — Group cold attenuation.** Owner: `xcom_hook.py` and transmission tests. Validate finite nonnegative density/path inputs, aggregate areal density by canonical material, and resolve coefficients once per unique material/energy. Cache parsed table vectors and use a searchable index, preserving the existing exact-energy choice and any repeated-energy absorption-edge entries; do not sort/deduplicate away discontinuities. Preserve interpolation/extrapolation warnings and material budgets. The 1,300-zone/single-material/single-energy case must use one coefficient lookup and match the scalar sum within `rtol=1e-12, atol=1e-15`. Cover zero paths, mixtures, exact table entries, edges, and extrapolation policy; do not alter edge physics in an optimization commit.

**W12c / F25 — Package scientific resources.** Owner: `pyproject.toml`, `plasmon_reference_data.py`, `plasmon_electron_policy.py`, `xcom_hook.py`, package-owned resource directories, `packaging/pyinstaller/helios_parser_viewer.spec`, and affected release builders/validators. Move/copy authoritative data into package resources with provenance; avoid divergent independently edited copies. Load through package-resource APIs, with explicit configuration overrides only where supported. Update both wheel/sdist data declarations and frozen-runtime data collection: the current PyInstaller GUI spec explicitly collects only viewer assets, so a wheel fix alone does not fix the portable application. The source-bundle copier copies `src` independently of Git ignores; exclude `*.egg-info` there too, rather than relying on G00's ignore rule. Verify resource hashes/registry contents and fallbacks from an installed wheel, an sdist-built wheel, and the supported frozen/portable artifact outside the checkout, with no editable-source path. PyInstaller currently excludes SciPy; keep and verify the numerical fallback path rather than silently adding SciPy as a mandatory dependency. Test production imports and explicitly enabled experimental calculations; record native artifact tests that require a different build host.

### W13 — Recover Qt lifecycle reliability and triage remaining failures

**Owner/files:** Qt lifecycle/test integration; `helios/tasks.py`; parser/viewer/analysis controllers and worker ownership; app/window close handlers; Qt test helpers; dependency constraints only if demonstrated necessary.

The audit established a native crash after printed `OK`; it did not establish its cause. Treat diagnosis as a bounded engineering task, not a known single-line fix.

1. Starting at W00, reproduce the smallest failing process with faulthandler and retain logs. Compare explicit close, repeated close, close while idle, close during read/parse/analysis, and rapid reopen.
2. Trace QObject/QThread/QRunnable ownership and destruction. Make shutdown idempotent, stop submissions, cancel work, close worker-owned resources in their thread, wait for cooperative completion, and destroy objects only after their work is drained. Do not clear signal owners while runnables can still emit.
3. Remove unbounded blocking-close calls where they can hang behind long queued work. Keep the owning objects alive until shutdown completes; if necessary defer final window close with a visible finishing/cancelling state. Viewer reads/parser conversion currently use QThreads, while derived tasks have cooperative cancellation tokens; do not assume the same cancellation mechanism exists in all three. For parser shutdown, either add an explicit conversion cancellation seam with safe checkpoints or defer teardown until work finishes.
4. Diagnose the two timed-out modules at test-case level using their real fixture costs. Separate slow finite work from deadlocks and apply explicit time budgets.
5. Compare a candidate supported Qt/Python/NumPy matrix only when lifecycle traces warrant it. Select and document a tested compatible range; do not guess version pins or remove dependency support merely to silence failures.
6. Triage each remaining historical failure: implementation bug, changed intended behavior, stale expected units/labels/counts, or runtime incompatibility. Update assertions only after documenting the correct contract. Preserve numerical coverage when replacing fixed UI row-count checks with semantic checks.

**Acceptance:** all required Qt test processes exit zero, including after their final `OK`. Repeated minimal lifecycle processes and close-during-work scenarios complete cleanly. No blanket skip, forced thread termination, `os._exit(0)`, or discarded native exit code may satisfy this gate. The payload wrapper executes every discovered intended test case. Both previously timed-out modules complete within their declared budgets.

**Commit sequence:** reproducible lifecycle diagnostics → justified ownership/shutdown fixes → contract-aware historical test updates.

### W14 — Prove the fixes together

**Owner/files:** integration/release; test runner and manifests, regression suites, benchmark scripts, schema/user/developer/release documentation.

1. Run targeted package tests after each commit. After integration, run the dynamically discovered full suite and smoke from a disposable current-source tree populated from W00's fixture manifest. Require the portable core profile to run from tracked/generated inputs; require explicit provision of the historical large fixtures for the full release gate. Keep production and experimental feature-flag entries explicit, accounting for bootstrap defaults.
2. Run an end-to-end matrix: LOG-only, BPF-only, aligned/mismatched companions, truncated final input, malformed input, overwrite failure, same-path replacement, sparse metadata, and uniform no-shock data.
3. Run wheel/sdist-installed and supported frozen-runtime smoke/resource checks outside the source tree. Verify legacy schema reads and the documented schema 2.1 semantic correction/provenance/aliases. Test readers against both versions rather than treating the alias change as merely additional fields.
4. Run the performance-improvement tests in section 5. Collect numerical residuals, I/O/lookup counts, cache bytes, peak RSS, and paired before/after timings. Archive the fixture/source/environment identities and comparator exit status. Do not convert the audit's one-off timing observations into CI limits.
5. Complete the desktop visual/accessibility checks and native lifecycle checks on the supported desktop platforms; label any untested platform in release evidence.
6. Write `docs/HELIOS_FIX_VALIDATION.md` mapping every finding to implementation commit, regression test, result, and any remaining limitation. Preserve the original audit as historical evidence.

**Completion gate:** every audit finding has an implementation and passing verification; no required suite is missing, crashing, timed out, or silently skipped. Experimental resource/numerical fixes must pass their explicit-feature gate before being marked resolved. Do not publish a release merely because P1 fixes landed while the full gate is still red.

## 4. Regression-test allocation

Use existing unittest infrastructure. Add focused suites only where the current ones do not express the contract; filenames below are proposed. Prefer synthetic fixtures for invariants and small tracked real inputs for integration.

| Test destination | Required coverage | Package |
|---|---|---|
| Repository-policy checks and fixture tests | Text/binary attributes, launcher modes, preserved fixture hashes, LF/CRLF parsing equivalence | G00 |
| New `test_test_runner.py` | Discovery, overrides, signals, timeouts, skip accounting, nonzero aggregate exit | W00 |
| New `test_conversion_atomicity.py` | Destination/input preservation, publication races, cleanup, callbacks | W01 |
| New `test_numeric_notation.py`; BPF/run-status/coordinate suites | Numeric variants, invalid time/cells, corruption versus truncation, trailers | W02 |
| New `test_pressure_contract.py`; BPF/reader suites | Source parity, radiation-dominated values, schema/legacy handling | W03 |
| New `test_source_energy_budget.py`; derived Preheat suites | Dimensional correctness, intervals, missing data, reference mass, source parity | W04 |
| New `test_companion_alignment.py`; streaming tests | All-frame/cycle matching, precision, fallback, bounded retention | W04 |
| New `test_run_revision.py`; cache/viewer/derived tests | Same-path replacements, stale tasks, explicit reopen, repeated clearing | W05 |
| New `test_module_availability.py`; release/derived/stability tests | No shock, missing metadata, invalid domains, independent results | W06 |
| Derived backend/performance suites | Exact field-read sets, incremental loading, byte eviction, retained RSS | W07 |
| Reader/viewer payload suites | Per-field metadata, physical I/O count, error/retry/coalescing | W08 |
| App/BPF pipeline suites | Actual desktop source routing, provenance, preview generations | W09 |
| Unit/derived-stability suites | Alias parity, display adapters, canonical precision and round trips | W10 |
| New desktop accessibility/layout suite | Names, buddy/focus relationships, actual geometry, theme transitions | W11 |
| Plasmon/Transmission suites; new wheel-resource suite | Exact grid preservation, grouped coefficients, installed resource behavior | W12 |
| Subprocess lifecycle suite; existing timed-out modules | Native clean exit, repeated close/open, shutdown during work | W13 |
| New performance regression suites and benchmark comparator | Count/scaling invariants, paired speed results, memory improvements, numerical equivalence | W00 baseline; W04/W07/W08/W12 comparisons; W14 gate |

Tests that inspect displayed coordinates must convert their independently calculated native expectation into the selected display unit. Assert both the value and label. Tests for field inventories should assert required fields/group semantics, not an obsolete fixed QListWidget item count.

## 5. Performance-improvement tests and acceptance

Performance verification is an implementation deliverable, not just an optional profiling session. Use deterministic regression tests for repeated work, then controlled before/after benchmarks for speed, memory, and interactive responsiveness. Targets below are proposed acceptance criteria; they are not claims that improvements have already been measured.

### 5.1 Reuse existing tools and add a reproducible comparison entry point

The repository already has `scripts/bench_hdf5_access.py`, `bench_streaming_conversion.py`, `bench_wave5_reuse.py`, `bench_wave6_derived.py`, and `microbench_wave1.py`, plus `tests/test_wave1_performance.py` and `test_derived_backend_optimization.py`. Extend useful helpers instead of replacing them wholesale. Several scripts currently hard-code local fixture paths, and the streaming RSS sampler returns no memory measurement when optional `psutil` is absent; resolve both limitations explicitly.

Add a proposed `scripts/run_performance_suite.py` facade with `baseline`, `candidate`, and `compare` operations, explicit fixture manifest/output paths, and a machine-readable result schema. Commands below are **target interfaces to implement**, not existing runnable commands:

```bash
.venv/bin/python scripts/run_performance_suite.py baseline --fixtures /tmp/helios-fixtures.json --output /tmp/helios-perf-before
.venv/bin/python scripts/run_performance_suite.py candidate --fixtures /tmp/helios-fixtures.json --output /tmp/helios-perf-after
.venv/bin/python scripts/run_performance_suite.py compare --baseline /tmp/helios-perf-before --candidate /tmp/helios-perf-after
```

W00 records the initial baseline. Each optimization package also records its direct correctness-fixed parent as a comparison baseline; an incorrect old pressure/energy result is not a numerical golden that the corrected implementation must reproduce. Freeze the fixture identities, feature flags, compression, and numerical resolution for a pair. Store Git revision plus working-tree source fingerprints, Python/library versions, OS, CPU, available RAM, and benchmark settings. Comparisons with incompatible manifests fail validation instead of printing a misleading speedup.

### 5.2 Required cases

| Test / benchmark | Setup and metrics | Pass criterion | Owner |
|---|---|---|---|
| P01 — Grouped cold attenuation | 1,300 zones, one material/energy; repeat with several materials/energies; coefficient calls, runtime, budgets/transmission | Exactly one lookup per unique material/energy; scalar-reference agreement within W12b tolerances; representative large case has a demonstrated lower paired median runtime | W12b |
| P02 — Pending I/O coalescing | 5 and 100 equivalent field/snapshot/trace requests before completion; actual reader calls, queue depth, busy transitions, final data | One physical read per unique revision/request key; retries work; balanced busy accounting; bounded queue instead of growth with duplicate requests | W08 |
| P03 — Metadata fan-out | Synthetic 10-, 104-, and 1,000-field HDF5 files; count attribute parses for one requested field and full inventory | One parse on first narrow read, zero on repeat; explicit inventory parses each remaining entry once; no O(field-count) metadata work for a narrow request | W08 |
| P04 — Demand-driven derived loading | Shock only → optional tab → repeat; field identities, read counts/bytes, cold/warm time, retained memory | Only declared requirements read; activating a tab reads only its additional requirements; repeating already loaded requirements reads zero additional arrays | W07 |
| P05 — Companion streaming | Equal-zone LOG/BPF pairs with increasing snapshot counts; conversion time, peak RSS, retained arrays, first-write latency | No full `Simulation` materialization; memory growth attributable to unrelated field histories is removed; allow indexed block offsets and bounded HDF5 cache costs; demonstrate reduced peak retained array bytes against the old full-parse path | W04 |
| P06 — Cache budgets and retention | Tiny injected limits for unit tests; representative runs and ten open/analyze/retire cycles for integration; unique allocation bytes, evictions, peak/post-retirement RSS | Aggregate registered cache allocations respect configured limits; oversized items not retained; retired results/tasks release ownership; post-retirement retained-array bytes plateau rather than grow per cycle | W05/W07 |
| P07 — Snapshot, trace, and scrub access | Run the existing HDF5 access benchmark at fixed chunking/compression; full numerical checksums and paired latency/throughput | No unintended extra full-field reads; representative unchanged access paths have no repeatable regression above the agreed 10% median review threshold | W08/W14 |
| P08 — UI responsiveness and latest result | Warm slider bursts and tab switches on a manifest-listed large run; event-loop gap samples, request-to-correct-render latency, task/read counts | Final render belongs to the latest request/revision; queue work is coalesced; native timing run demonstrates no regression in event-loop gaps/settling latency against the matched baseline | W08/W11/W13 |
| P09 — Cache reuse correctness | Compare cold, warm, cleared, and same-path-reopened results; values, provenance, source revision, cache/read counts | Warm reuse returns equivalent data without rereading; reopening changed data invalidates reuse; eviction/clearing never changes scientific outputs or revives an old revision | W05/W07/W12a |

Add focused proposed suites such as `tests/test_performance_work_counts.py`, `test_cache_memory_budget.py`, and `test_performance_comparator.py`. Reuse the existing numerical/backend suites for equivalence checks. Comparator tests cover incompatible environments, missing metrics, insufficient samples, an intentional slowdown, and an intentional memory-retention regression.

### 5.3 Measurement rules and improvement claims

1. Run correctness and count/scaling tests in ordinary CI. Timing tests run on a fixed benchmark host or a documented comparable environment; do not impose absolute laptop timings on arbitrary CI workers.
2. For short CPU/read cases, run warmups and at least ten paired measurements, alternating baseline/candidate order. Report every sample, medians, the candidate/baseline ratio, and dispersion. Where the benchmark supports it, report a paired uncertainty interval. Claim a timing improvement only when the difference exceeds measured variability; otherwise report **inconclusive**, while retaining any proven work-count improvement.
3. For slower full conversion/RSS cases, use at least five fresh processes per side. These characterize median/peak behavior; do not label five samples a reliable p95. Sample at least 100 interactions for UI tail-latency reporting and name the statistic/window used. Offscreen runs validate behavior; native runs are required for user-visible latency claims.
4. Separate application-cache cold/warm from OS-filesystem cold/warm. A fresh Python process is not proof of a cold OS cache. Record cache conditions; do not flush global OS caches as part of routine tests.
5. Require a real process-memory measurement for a memory gate. If `psutil` or a supported platform measurement backend is unavailable, report **measurement unavailable** and fail that required memory check; do not substitute `tracemalloc` alone for NumPy/HDF5/native RSS. Keep measurement dependencies in the test/benchmark environment rather than the production runtime.
6. Assert exact cache/retained-array budgets separately from RSS. Allocators may retain freed pages, so lack of immediate RSS return is not by itself a leak; repeated growth plus surviving references requires investigation. Record and test byte-accounting deduplication across cache owners.
7. For intended speedups, require the structural improvement and a representative matched timing comparison; for intended memory reductions, require retained-allocation evidence and a fresh-process peak-memory comparison. An inconclusive timing result cannot be described as a measured speed improvement. Pure correctness changes may carry an explained cost rather than being mislabeled optimizations.
8. Treat a repeatable median slowdown above 10% on unchanged scenarios as a review failure until it is fixed or explicitly justified in validation evidence. This is a proposed review threshold, not an audit measurement. Missing results, changed fixtures/resolution, failed equivalence, native crashes, or timeouts make the performance comparison fail.

Do not change HDF5 chunking, downsample scientific fields, lower numerical resolution, or change physical kernels merely to improve a headline timing. Any later chunk-layout change needs snapshot/trace benchmarks and independent numerical equivalence. W14 must retain baseline/candidate JSON, comparison results, and a short explanation of every improvement or justified regression in `docs/HELIOS_FIX_VALIDATION.md`.

## 6. Finding-to-package checklist

- [x] F01 → W01: atomic conversion and input identity.
- [x] F02 → W04: cumulative source units and energy budgets.
- [x] F03 → W05: run revisions and stale result rejection.
- [x] F04 → W02: common BPF validity checks.
- [x] F05 → W02: honest partial-final-snapshot status.
- [x] F06 → W03: explicit pressure semantics.
- [x] F07 → W04: every-frame companion alignment.
- [x] F08 → W02: scientific-notation extraction.
- [x] F09 → W06: independent module availability/domain handling.
- [x] F10 → W06: partial datasets and prerequisite checks.
- [x] F11 → W09: BPF-aware desktop workflow/provenance.
- [ ] F12 → W11: accessible control names and navigation.
- [x] F13 → W11: usable desktop client sizes.
- [x] F14 → W00 + W13 + W14: complete discovery, reliable runtime exits, green integration gate.
- [x] F15 → W10: canonical/alias display-unit parity.
- [x] F16 → W12a: nonuniform spectrum grids.
- [x] F17 → W10: canonical angle values and precision.
- [x] F18 → W04: streamed companion data and bounded error tails.
- [x] F19 → W07: demand-driven arrays and byte budgets.
- [x] F20 → W05: stable cache bucket lifecycle.
- [x] F21 → W12b: material-grouped attenuation lookups.
- [x] F22 → W08: pending I/O coalescing.
- [x] F23 → W08: per-field metadata loading.
- [x] F24 → W11: refreshed theme colors and contrast.
- [x] F25 → W12c: packaged scientific data resources.

Mark a finding complete only when its listed regression and integration checks pass. Keep schema changes, corrected numerical values, and unsupported-geometry/module reasons explicit in the final validation document.
