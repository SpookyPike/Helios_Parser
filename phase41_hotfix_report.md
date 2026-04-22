# Phase 4.1.1 Hotfix Report

## Scope

This pass was limited to Derived / Analysis stability, cache correctness, plotting usability, unit consistency, and validation. No new scientific modules or parser/HDF5 architecture changes were introduced.

## Crash Root Cause

The Derived-mode instability came from a combination of presentation-layer and task-lifecycle issues:

1. `DerivedPlotPanel` was reaching inside `CurvePlotWidget` and calling `plot._plot.clear()`, which bypassed the widget's own lifecycle and could leave internal plot items disconnected.
2. Derived plots were always rendered with `preserve_view=True`, so recomputes reused stale zoom/view state even when the underlying curves had changed materially.
3. The Derived controller could submit duplicate in-flight recomputes for the same request key.
4. The shell shut down parser/viewer controllers on exit, but not the Derived controller, even though it owned worker threads.
5. The plotting layer still allowed pyqtgraph auto-SI axis scaling to add implicit `x0.001`-style presentation changes, which made some plasmon summaries look inconsistent with the plots even when the backend numbers matched.

## Fixes Implemented

### Runtime / caching / lifecycle

- Added duplicate-request suppression in [controller.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/controller.py).
- Guarded result/failure application when the Derived workspace has already been destroyed in [controller.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/controller.py).
- Added a real Derived shutdown path in:
  - [controller.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/controller.py)
  - [derived_controller.py](C:/Users/dimab/Documents/Helios_parser/src/helios_app/derived_controller.py)
  - [tasks.py](C:/Users/dimab/Documents/Helios_parser/src/helios/tasks.py)
  - [main_app.py](C:/Users/dimab/Documents/Helios_parser/src/helios_app/main_app.py)

### Plot usability / stability

- Removed the private `_plot.clear()` misuse and replaced it with a safe public clear path in [plots.py](C:/Users/dimab/Documents/Helios_parser/src/helios_viewer/plots.py) and [workspace.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/workspace.py).
- Added explicit `Reset View` buttons to Derived plot panels and shock plots in [workspace.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/workspace.py).
- Added double-click autoscale/reset for `CurvePlotWidget` in [plots.py](C:/Users/dimab/Documents/Helios_parser/src/helios_viewer/plots.py).
- Disabled auto-SI prefix scaling on curve axes in [plots.py](C:/Users/dimab/Documents/Helios_parser/src/helios_viewer/plots.py) so summary values and plot values stay in the same stated units.
- Derived recomputes now auto-fit the new data instead of restoring stale zoom by default.

### Derived semantics / unit presentation

- XRD weighting now actually uses the selected weighting mode instead of always width-averaging in [xrd.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/xrd.py).
- XRD now exposes a real `Q [1/A]` time-trace and the UI can switch between `Bragg shift [deg]`, `Q [1/A]`, and `k-space [1/A]` in [xrd.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/xrd.py) and [workspace.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/workspace.py).
- XRD profile titles were clarified to `Zone-resolved density profile` and `Zone-resolved compression ratio profile` in [xrd.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/xrd.py).
- Spectroscopy now supports display-layer shift units `nm`, `eV`, `meV`, and `ueV` in [workspace.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/workspace.py), backed by new conversion helpers in [conversions.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/units/conversions.py).
- Shock velocity labels were clarified to distinguish `Speed magnitude |v|` from `Signed shock velocity`, and the summary now states that both are computed from the smoothed primary trajectory in [workspace.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/workspace.py).

### Runtime warnings

- Added explicit warning/error generation for:
  - degenerate LOS geometry
  - empty selections
  - non-finite effective plasmon states
  - non-finite effective spectroscopy states
  - zero-effective transmission path length

These changes are in:
- [analysis.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/analysis.py)
- [plasmon.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/plasmon.py)
- [transmission.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/transmission.py)
- [spectroscopy.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/spectroscopy.py)

### Validation harness cleanup

- Offline validation now only selects stabilized HDF5 files that match the Derived-service schema in [scripts/_validation_common.py](C:/Users/dimab/Documents/Helios_parser/scripts/_validation_common.py).
- Fixed the plasmon validation summary to report `k_lambda_d` and `collectivity_parameter` under the correct names in [scripts/validate_plasmon_regime.py](C:/Users/dimab/Documents/Helios_parser/scripts/validate_plasmon_regime.py).

## Automated Tests Run

Passed:

- `python -m unittest discover -v -s tests -p "test_derived_stability.py"`
- `python -m unittest discover -s tests -p "test_derived_phase41.py"`
- `python -m unittest discover -s tests -p "test_app_phase41.py"`
- `python -m unittest discover -s tests -p "test_app_phase4.py"`
- `python -m unittest discover -s tests -p "test_viewer_phase4.py"`
- `python tests\\smoke_test.py`

New focused coverage lives in [test_derived_stability.py](C:/Users/dimab/Documents/Helios_parser/tests/test_derived_stability.py) and covers:

- duplicate in-flight recompute suppression
- filter/weighting propagation
- empty-selection panel disabling
- XRD display switching
- spectroscopy shift-unit switching

## GUI Stress Validation

Offscreen GUI stress script:
- [phase41_hotfix_validation.py](C:/Users/dimab/Documents/Helios_parser/scripts/phase41_hotfix_validation.py)

Generated report:
- [phase41_hotfix_validation.json](C:/Users/dimab/Documents/Helios_parser/outputs/reports/phase41_hotfix_validation.json)

Observed outcomes:

- `5Fe+4.9TW+light_stabilized.h5`
  - empty-selection path reproduced by excluding the only entry region
  - selected zones dropped from `500` to `0`
  - XRD / plasmon / spectroscopy plot controls disabled cleanly
  - no crash when switching across all Derived tabs

- `Cu_0166_stabilized.h5`
  - tab switching across `Shock / XRD / Plasmon / Transmission / Spectroscopy / Warnings` remained stable
  - XRD display switch selected `q_compressed`
  - spectroscopy summary/metrics switched to `eV` consistently

- `10ns+10Si+60Al+15Si+4.27TW_stabilized.h5`
  - repeated weighting/filter/geometry changes recomputed without crash
  - selection changed from `1300` to `900` zones under the filtered scenario
  - manual zoom, explicit reset button, and double-click reset all restored the original XRD time-plot viewport

Representative screenshots:

- ![Small empty selection](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase411_small_empty_selection_light.png)
- ![Cu Derived Q view](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase411_cu_derived_q_light.png)
- ![Large filtered Derived view](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase411_large_filtered_dark.png)

## Offline Scientific Sanity Validation

Refreshed offline validators:

- [shock tracker summary](C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/shock_tracker/summary.json)
- [xrd summary](C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/xrd/summary.json)
- [plasmon summary](C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/plasmon/summary.json)
- [transmission summary](C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/transmission/summary.json)
- [spectroscopy summary](C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/spectroscopy/summary.json)
- [selection filter summary](C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/selection_filters/summary.json)
- [weighting mode summary](C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/weighting_modes/summary.json)

Shock tracker debug overlays were preserved in:

- [5Fe shock debug](C:/Users/dimab/Documents/Helios_parser/outputs/derived_debug/5Fe+4.9TW+light_stabilized_shock_debug.png)
- [Cu shock debug](C:/Users/dimab/Documents/Helios_parser/outputs/derived_debug/Cu_0166_stabilized_shock_debug.png)
- [10ns shock debug](C:/Users/dimab/Documents/Helios_parser/outputs/derived_debug/10ns+10Si+60Al+15Si+4.27TW_stabilized_shock_debug.png)

Notable current offline results:

- shock validation shows `jump_count_gt_50_zones = 0` and `signed_velocity_sign_flips = 0` on all three representative stabilized datasets
- XRD now exposes `6` time plots and `2` profile plots on the validated datasets
- weighting validation shows real mode-dependent changes in effective plasmon state, especially on the heavy layered target

## Before / After Summary

Before this hotfix pass:

- Derived recomputes could reuse stale plot state and leave panels frozen or visually inconsistent.
- Background recomputes could duplicate work for the same request.
- Empty selections could leave stale-looking plots/controls visible.
- XRD claimed configurable weighting but ignored it internally.
- Spectroscopy was locked to `nm`.
- Plot presentation could imply scaled units that did not match the summary text.

After this hotfix pass:

- repeated tab switching, recompute, filter changes, geometry changes, and plot resets complete without a reproduced crash in the scripted GUI stress run
- empty selections degrade to warnings plus disabled Derived plot panels
- XRD weighting affects the backend calculation path
- spectroscopy shift display is selectable in practical experimental units
- Derived plots have explicit reset controls and double-click autoscale
- summary values and curve axes stay in the stated units

## Remaining Limitations

- Shock tracking is still a deterministic heuristic tracker, not a full shock-fit model.
- Transmission remains Thomson-only in production behavior.
- Spectroscopy remains a bulk Doppler/broadening quick look, not a line-formation model.
- Some weighting/filter combinations will only move results slightly on uniform single-region runs; this is expected and not a cache bug.
- The GUI validation here is scripted offscreen Qt interaction. It is good for determinism and crash catching, but it is not a substitute for subjective desktop-interaction feel on local hardware.
