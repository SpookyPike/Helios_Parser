# HELIOS Plasmon Stability And Benchmark Pass

Date: 2026-03-27

This note summarizes the end-to-end plasmon stability pass, UI/state-flow fixes, runtime diagnostics, and the all-model aluminium benchmark on `50Al+10E+25CH+3.5TW_stabilized.h5`.

## A. Software / State-Flow Findings

### A1. Root causes fixed

- `study_mode="spectrum"` previously still built expensive dispersion/comparison branches in `plasmon.py`. That was the main cause of pathological runtime and made the UI look hung.
- Heavy plasmon numeric controls still reacted while typing. This caused destructive intermediate recompute and made probe-energy edits feel broken.
- Stale plasmon results could still resync controls in `workspace.py` if the result arrived after the user had already changed model/study settings.
- The cancel path existed structurally but lacked direct regression proof from the workspace button through the controller.
- Invalid benchmark branches in the Lindhard+Mermin family could return `invalid_for_benchmark` without a clear collision-closure reason in `warnings`.
- Even when a spectral bundle was unavailable, the plasmon tab could fall back to a legacy `Electron temperature vs time` default, which undermined the semantics of the plasmon tool.
- Preheat manual navigation still behaved like a local snapshot island in the UI layer instead of transparently following the global snapshot selector.

### A2. Behaviour after the fix

- Spectrum-mode requests no longer secretly build dispersion bundles.
- Dispersion-mode requests no longer build spectral comparison bundles.
- Heavy plasmon numeric inputs no longer fire recompute while the user is still typing.
- Stale plasmon results no longer overwrite the current control state.
- The cancel button is wired through the controller and clears the busy state.
- Invalid Lindhard+Mermin benchmark results now surface the collision-rate failure reason explicitly and keep the correct backend tag (`finite_t_lindhard`).
- Quicklook and invalid spectral states now default to the plasmon-centric `Plasma frequency vs time` trace rather than `Electron temperature vs time`.
- Preheat manual navigation now follows the global snapshot and keeps the local snapshot controls disabled.

## B. Runtime / Performance Findings

### B1. Before / after

- Earlier pathological runtime for `test_plasmon_phase5.py`: about `790.8 s` for 3 tests.
- After the branch-gating fix in `plasmon.py`: `17.189 s` for the same 3 tests.

### B2. Current test runtime

- `test_plasmon_phase*.py`: `74` tests in `31.541 s`, `OK (skipped=1)`.
- `test_plasmon_ui_phase*.py`: `12` tests in `4.390 s`, unittest output `OK`.

### B3. Benchmark runtime cost by model

Cold Al, ambient selection:

- `quicklook`: `0.031 s`
- `rpa`: `0.210 s`
- `rpa_static_lfc`: `0.272 s`
- `lindhard`: `0.148 s`
- `lindhard_static_lfc`: `0.040 s`
- `auto_best`: `0.048 s`

Driven dense Al slab near 6.3 ns:

- `quicklook`: `0.034 s`
- `rpa`: `6.315 s`
- `rpa_static_lfc`: `6.352 s`
- `lindhard`: `3.074 s`
- `lindhard_static_lfc`: `0.051 s`
- `auto_best`: `0.052 s`

Interpretation:

- The main remaining runtime hotspot is classical LOS-integrated `RPA` / `RPA + static LFC` on the driven dense slab.
- `lindhard_static_lfc` is very cheap numerically, but that does not make it physically best.
- `auto_best` is practical now because it resolves to the cheap/valid branch rather than forcing the most expensive one.

## C. Benchmark Methodology Findings

### C1. Common settings

- Dataset: `50Al+10E+25CH+3.5TW_stabilized.h5`
- Probe energy: `8.307 keV`
- Electron policy: `benchmark_valence_aware`
- Peak-extraction instrument broadening: `0.20 eV`
- Representative shape overlay broadening: `3.50 eV`

### C2. Ambient case

- Snapshot: `0`
- Time: `0.0000 ns`
- Material filter: `Al only` (`material_id = 1`)
- Zone span: `1-1000`
- Selected zones: `1000`
- Weighted density: `2.700 g/cm^3`
- Weighted electron temperature: `0.025 eV`
- Raw HELIOS weighted `Zbar`: `1.289e-13`

Interpretation:

- The raw HELIOS ionization state is not usable for literature-facing ambient Al dispersion.
- The benchmark therefore relies on the existing `benchmark_valence_aware` electron policy rather than raw `ne/zbar`.

### C3. Driven case

- Target probe time: `6.30 ns`
- Nearest hydro snapshot: `630` at `6.3001 ns`
- Material filter: `Al only`
- Additional clip: contiguous `rho >= 3.75 g/cm^3` slab
- Zone span: `561-973`
- Selected zones: `413`
- Weighted density: `4.196 g/cm^3`
- Weighted electron temperature: `0.483 eV`
- Weighted ion temperature: `0.489 eV`
- Raw HELIOS weighted `Zbar`: `0.1918`

Interpretation:

- This is within the article-facing density window and close to the quoted `~0.6 eV` driven regime, but somewhat cooler than the nominal article state.
- The results still depend materially on slab-selection and electron-mapping assumptions.

## D. Model-By-Model Physical Findings

### D1. Practical ranking now

Most practical now:

1. `auto_best`
2. `rpa_static_lfc`
3. `lindhard`

Scientifically closest to the current reference scaffold:

- Ambient experiment: `lindhard`
- Ambient Gawne reference: `rpa_static_lfc` / `auto_best`
- Driven experiment: `auto_best` / `rpa_static_lfc`
- Driven RPA reference: `lindhard`

### D2. Per-model findings

#### Quick look

- Ambient experiment MAE: `6.528 eV`
- Driven experiment MAE: `5.085 eV`
- Status: `not_applicable`

Judgement:

- Still useful as a fast heuristic trend line.
- Not suitable as a benchmark spectral model.
- Should stay clearly separated from benchmark-grade branches in user-facing comparison language.

#### RPA

- Ambient experiment MAE: `6.480 eV`
- Ambient Gawne MAE: `3.993 eV`
- Driven experiment MAE: `4.111 eV`
- Driven runtime mean: `6.315 s`

Judgement:

- Numerically stable.
- Too stiff / too high in the driven dense-Al state versus experiment.
- Expensive in LOS-integrated driven mode.

Needs:

- Better electron mapping and/or beyond-classical closure if it is to be used as a driven WDM benchmark branch.

#### Mermin

- Ambient: `invalid_for_benchmark`
- Driven: `invalid_for_benchmark`

Judgement:

- Not currently benchmark-ready for these Al selections.
- The problem is not “mysterious failure”; it is the collision-rate closure: the selected constant-ν path does not resolve a finite non-negative local collision rate for the LOS states in these cases.

Needs:

- A more credible collision model for these dense partially degenerate states, or explicit UI gating that marks the branch unavailable earlier.

#### RPA + static LFC

- Ambient experiment MAE: `4.502 eV`
- Ambient Gawne MAE: `2.439 eV`
- Driven experiment MAE: `2.679 eV`
- Driven runtime mean: `6.352 s`

Judgement:

- Best classical practical model in the current code.
- Best agreement with the current driven experimental scaffold.
- Best agreement with the ambient literature reference series already encoded in the repo.

Needs:

- Performance work on LOS-integrated driven runs.
- Better degenerate/WDM handling if closer TDDFT agreement is required.

#### Mermin + static LFC

- Ambient: `invalid_for_benchmark`
- Driven: `invalid_for_benchmark`

Judgement:

- Same collision-closure failure as Mermin, now with the extra ESA static-LFC dependency on top.

Needs:

- Same collision-model improvement as Mermin, plus continued ESA-domain diagnostics.

#### Finite-T Lindhard

- Ambient experiment MAE: `3.485 eV`
- Ambient Gawne MAE: `4.120 eV`
- Driven experiment MAE: `6.955 eV`
- Driven RPA-reference MAE: `2.017 eV`
- Driven runtime mean: `3.074 s`

Judgement:

- Best match to the blue ambient experiment points among the currently benchmarked branches.
- In the driven dense slab it drifts toward the digitized RPA-like reference rather than experiment/TDDFT.
- Numerically stable and diagnostically honest.

Needs:

- Better physical treatment of the driven compressed WDM state if it is to match experiment rather than just the RPA-like trend.

#### Finite-T Lindhard + Mermin

- Ambient: `invalid_for_benchmark`
- Driven: `invalid_for_benchmark`

Judgement:

- The backend tag is correct and the failure reason is now explicit.
- The branch is not user-ready for these benchmark states because the selected Mermin collision closure still fails.

Needs:

- Same collision-model work as the classical Mermin branch.

#### Finite-T Lindhard + static LFC

- Ambient experiment MAE: `5.482 eV`
- Ambient Gawne MAE: `5.362 eV`
- Driven experiment MAE: `7.525 eV`
- Driven runtime mean: `0.051 s`

Judgement:

- Very fast.
- Physically too high / too stiff on the driven benchmark compared with experiment.
- Useful as a cheap comparison branch, not as the current best benchmark answer.

Needs:

- Reassessment of the static-LFC usage/domain and electron mapping for the driven compressed slab.

#### Finite-T Lindhard + Mermin + static LFC

- Ambient: `invalid_for_benchmark`
- Driven: `invalid_for_benchmark`

Judgement:

- Not benchmark-ready for the same collision-closure reason as the other Mermin variants.

#### Auto best per state

- Ambient experiment MAE: `4.502 eV`
- Ambient Gawne MAE: `2.439 eV`
- Driven experiment MAE: `2.679 eV`
- Driven runtime mean: `0.052 s`

Judgement:

- Currently the best practical user-facing benchmark branch.
- For these Al cases it resolves to the same answer as `rpa_static_lfc`, but at much lower runtime cost because it avoids unnecessary heavy branch work.

Needs:

- Better surfacing of which local branch was chosen and why, especially for literature-facing comparisons.

## E. Recommended Next Improvements

### Highest impact

1. Improve or replace the constant-ν collision closure for all Mermin-family branches.
2. Keep `benchmark_valence_aware` explicit and documented as a benchmark-only electron-mapping policy; do not silently let users mistake it for raw hydro truth.
3. Add a user-facing benchmark preset for the Al article geometry and slab selection, so the same state can be rerun from the UI reproducibly.

### Medium impact

4. Optimize driven LOS-integrated `RPA` / `RPA + static LFC` runtime further.
5. Tighten benchmark reports around representative line-shape overlays once higher-quality digitized spectral references exist.
6. Surface model-unavailable reasons directly in the plasmon tab without requiring the user to infer them from missing curves.

### Lower impact

7. Separate quicklook benchmarking from spectral benchmarking more explicitly in the UI and docs.
8. Add a compact “diagnostics” disclosure block in the plasmon tab for runtime breakdown and invalid-zone counts, instead of relying only on the metrics text area.

## Generated Artifacts

- `outputs/validation_outputs/plasmon_article_cases/report.md`
- `outputs/validation_outputs/plasmon_article_cases/all_model_results.json`
- `outputs/validation_outputs/plasmon_article_cases/benchmark_points.csv`
- `outputs/validation_outputs/plasmon_article_cases/benchmark_summary.csv`
- `outputs/validation_outputs/plasmon_article_cases/ambient_dataset_overlay.png`
- `outputs/validation_outputs/plasmon_article_cases/driven_dataset_overlay.png`
- `outputs/validation_outputs/plasmon_article_cases/ambient_representative_spectra.png`
- `outputs/validation_outputs/plasmon_article_cases/driven_representative_spectra.png`
- `outputs/validation_outputs/plasmon_ui_live/report.md`
