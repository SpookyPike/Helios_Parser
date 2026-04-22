# Phase 4.1.2 Production Gate Report

## Scope

This pass was limited to release-blocking stabilization, global snapshot/time ownership, Viewer no-regression, and scientific-presentation cleanup. No parser redesign, HDF5-schema changes, or new physics modules were introduced.

## Concrete reproduced failures

The current tree had at least three real Phase 4.1.2 blocking issues:

1. Immediate open -> Derived could crash during widget construction.
2. Rapid control changes on large moving-radius runs could enqueue overlapping Derived tasks.
3. The new shell snapshot/time control could become unusable on short-time datasets because it displayed raw seconds in a fixed-point spin box.

I also audited the reported Viewer colorbar regression and XRD presentation state during the same pass.

## Root causes found

### 1. Immediate Derived crash

Deterministic crash reproduced from the shell:

- launch app
- open `Cu_0166_stabilized.h5`
- switch to `Derived / Analysis`

Concrete root cause:

- [workspace.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/workspace.py) called `_install_wheel_guard()` before `self.result_tabs` existed.
- That raised `AttributeError: 'HeliosDerivedWorkspace' object has no attribute 'result_tabs'` during workspace construction.

### 2. Rapid control-change crash / hang risk

On the large `50Al+10E+25CH+3.5TW_stabilized.h5` benchmark, rapid weighting/profile/filter changes were previously creating overlapping background tasks.

Concrete symptom reproduced before the controller fix:

- repeated control changes every ~70 ms caused `len(ctrl._tasks._threads)` to climb from `1` to `7`
- stale tasks could finish after the UI state had already moved on

Concrete root cause:

- [controller.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/controller.py) allowed multiple in-flight Derived recomputes instead of coalescing to a single latest request.

### 3. Global time entry unusable on small-time runs

After the new shell snapshot bar was introduced, small-run times were displayed in raw viewer time units. On `5Fe+4.9TW+light_stabilized.h5` that meant the time entry effectively rounded to `0.000000 s`.

Concrete root cause:

- [main_app.py](C:/Users/dimab/Documents/Helios_parser/src/helios_app/main_app.py) was feeding the shell control through the Viewer display-time path instead of a shell-stable physical time representation.

### 4. Viewer colorbar instability source

The Viewer colorbar regression was traced to degenerate level ranges and item rebinding across dataset resets.

Relevant fix path:

- [plots.py](C:/Users/dimab/Documents/Helios_parser/src/helios_viewer/plots.py) now rebinds the colorbar on dataset reset and normalizes identical min/max level pairs before applying them.

## Exact code paths changed

Primary runtime fixes:

- [workspace.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/workspace.py)
- [controller.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/controller.py)
- [main_app.py](C:/Users/dimab/Documents/Helios_parser/src/helios_app/main_app.py)
- [main_window.py](C:/Users/dimab/Documents/Helios_parser/src/helios_viewer/main_window.py)
- [plots.py](C:/Users/dimab/Documents/Helios_parser/src/helios_viewer/plots.py)

Supporting state/semantics fixes:

- [settings.py](C:/Users/dimab/Documents/Helios_parser/src/helios_viewer/settings.py)
- [runtime.py](C:/Users/dimab/Documents/Helios_parser/src/helios/runtime.py)
- [analysis.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/analysis.py)
- [selection.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/selection.py)
- [models.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/models.py)
- [xrd.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/xrd.py)
- [viewer_controller.py](C:/Users/dimab/Documents/Helios_parser/src/helios_app/viewer_controller.py)
- [derived_controller.py](C:/Users/dimab/Documents/Helios_parser/src/helios_app/derived_controller.py)

New validation/test coverage:

- [test_app_phase412.py](C:/Users/dimab/Documents/Helios_parser/tests/test_app_phase412.py)
- [test_derived_phase412.py](C:/Users/dimab/Documents/Helios_parser/tests/test_derived_phase412.py)
- [phase412_production_gate_validation.py](C:/Users/dimab/Documents/Helios_parser/scripts/phase412_production_gate_validation.py)

## What changed

### Derived crash hardening

- Moved wheel-guard installation to the end of Derived workspace construction so widget setup is deterministic.
- Coalesced Derived recomputes to a single in-flight request plus one pending latest request.
- Stale late results are ignored unless they still match the current request/context.
- Hidden/inactive Derived mode no longer recomputes in the background.
- Rapid combobox/wheel changes are debounced and protected by wheel-guard filtering.

### One global snapshot/time control

- The shell now owns one global snapshot bar at the bottom of the main window.
- It exposes:
  - slider
  - numeric snapshot selector
  - physical time entry
- It is the single active-snapshot source for Viewer and Derived.
- The shell time control now uses `ns`, not raw viewer display seconds, so it remains usable on short-time runs.
- Viewer’s embedded local snapshot bar is hidden while the shell owns snapshot state.

### Viewer stability / colorbar

- Dataset reset explicitly rebinds the colorbar to the active image item.
- Degenerate color ranges are expanded safely instead of collapsing the colorbar.
- Colorbar label/levels now survive:
  - Viewer open
  - Mouse Mode
  - Derived -> Viewer roundtrip
  - field changes
  - theme changes

### Profile coordinate defaults and semantics

- Default profile coordinate is now a real persisted setting in Viewer settings.
- The default is `Zone index`.
- `Moving radius` remains available but is no longer the silent default.
- Derived profile-coordinate `"viewer"` now follows the Viewer slice/profile coordinate, not the 2D map coordinate.

### XRD scientific/UI cleanup

- Removed the fake separate `k-space` option from the XRD display selector.
- XRD display now switches cleanly between:
  - degrees
  - `Q [1/A]`
- XRD table headers and time-trace selection now follow the active display mode.
- XRD summary and titles explicitly state that this is an effective isotropic-compression quick look.
- Snapshot profile titles now include the active snapshot/time.

## Before / after behavior

### Immediate open -> Derived

Before:

- opening `Cu_0166_stabilized.h5` and switching to Derived could crash during widget construction

After:

- same workflow completes
- Derived result becomes available
- snapshot/profile titles follow the active global snapshot

### Rapid control changes on large runs

Before:

- repeated weighting/profile/filter changes could create multiple in-flight tasks

After:

- validated maximum in-flight task count is `1`
- pending updates coalesce to the latest request
- no reproduced crash on the `10ns...` multilayer run or `50Al+10E+25CH+3.5TW`

### Global snapshot semantics

Before:

- Viewer and Derived snapshot/profile semantics were still fragmented

After:

- shell bottom-bar snapshot/time controls drive both Viewer and Derived
- Derived snapshot profiles and tables now correspond to the same active snapshot

## Validation performed

### Compile and smoke

Passed:

- `python -m py_compile` on the touched shell/viewer/derived modules
- `python tests\\smoke_test.py`

### Focused automated tests run

Passed:

- `python -m unittest discover -v -s tests -p "test_app_phase412.py"`
- `python -m unittest discover -v -s tests -p "test_derived_phase412.py"`
- `python -m unittest discover -v -s tests -p "test_app_phase4.py"`
- `python -m unittest discover -v -s tests -p "test_derived_stability.py"`
- `python -m unittest discover -v -s tests -p "test_viewer_phase4.py"`
- `python -m unittest discover -v -s tests -p "test_app_phase41.py"`

### Offscreen GUI production-gate validation

Validation script:

- [phase412_production_gate_validation.py](C:/Users/dimab/Documents/Helios_parser/scripts/phase412_production_gate_validation.py)

Generated report:

- [phase412_validation.json](C:/Users/dimab/Documents/Helios_parser/outputs/reports/phase412_validation.json)

Observed results:

- `5Fe+4.9TW+light_stabilized.h5`
  - shell snapshot bar drove Viewer and Derived coherently
  - validated state: `Snapshot 5 | t = 0.50004 ns`
  - Derived XRD profile title: `Zone-resolved density profile | snapshot 5 @ 0.500 ns`

- `Cu_0166_stabilized.h5`
  - immediate Viewer -> Derived switch completed
  - Derived selected zones: `300`
  - XRD degree-mode headers remained synchronized
  - plasmon and spectroscopy multi-curve legends remained visible
  - Viewer colorbar remained valid through field/theme/mode changes

- `10ns+10Si+60Al+15Si+4.27TW_stabilized.h5`
  - repeated control changes completed without reproduced crash
  - final validated snapshot label: `175 @ 1.750 ns`
  - max in-flight task count: `1`

- `50Al+10E+25CH+3.5TW_stabilized.h5`
  - used as the primary stress benchmark for this pass
  - rapid weighting/profile/filter changes completed without reproduced crash
  - final validated snapshot label: `187 @ 1.870 ns`
  - max in-flight task count: `1`

Representative screenshots:

- ![Small global snapshot sync](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase412_small_global_snapshot.png)
- ![Cu Derived](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase412_cu_derived.png)
- ![Cu Viewer colorbar](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase412_cu_viewer_colorbar.png)
- ![Heavy multilayer](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase412_heavy_multilayer.png)
- ![50Al stress run](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase412_stress_al_epoxy_ch.png)

## Release-gate status

### Fixed in this pass

- immediate open -> Derived crash: fixed
- wheel/combo-driven recompute storm: fixed
- large-run multi-control overlap path: fixed
- Viewer embedded local/global snapshot split: fixed
- global time entry usability on short-time runs: fixed
- Viewer colorbar degeneration / disappearance path: fixed in current validation
- XRD degree/Q UI mismatch: fixed

### Scientifically safe to show now

- Viewer with the global snapshot bar and colorbar restored
- Derived quick-look panels with explicit snapshot/time titles
- XRD displayed as an isotropic-compression proxy
- plasmon/XRTS quick-look regime classification with explicit weighting/geometry
- Thomson-only transmission budget with explicit labeling
- spectroscopy Doppler/broadening quick look with explicit unit selection

### Still not production-ready in the strict sense

I am not calling the full scientific stack production-ready yet.

Reasons:

- XRD remains an isotropic-compression quick look, not a crystallographic model
- plasmon/XRTS remains an NRL-based quick-look regime panel, not a full scattering solver
- transmission remains Thomson-only in shipping behavior
- spectroscopy remains a quick-look Doppler/broadening panel
- moving-radius geometry is still reconstructed from zone-center data where explicit cell edges are unavailable
- shock tracking remains a continuity-constrained primary-front heuristic, even though the current crash/stability gate passed

## Remaining limitations

- moving-radius boundary/path reconstruction is still approximate when explicit cell edges are absent from HDF5
- secondary/reflected shock structures are still not a production front-segmentation model
- time traces at physical coordinates still use nearest-zone mapping where labeled
- the current validation is strong scripted offscreen GUI coverage, but it is not a substitute for long manual desktop interaction on the target workstation
