# Phase 4.1.2c Snapshot Reactivity and Performance Report

## Scope

This pass focused on snapshot-driven interaction only:

- global snapshot slider hardening
- lightweight Derived snapshot refresh
- explicit busy/update feedback
- verification that Derived snapshot-local panels really follow the active global snapshot
- Viewer colorbar regression check

No new physics models were added.

## What Was Causing the Lag

The main performance problem was architectural rather than numerical:

1. Every global snapshot change could fall through to a full Derived recompute.
2. The shell snapshot slider applied updates directly on every intermediate value.
3. Derived context updates were repopulating control widgets even for snapshot-only changes.
4. Time traces were being rebuilt even when only snapshot-local profiles/tables needed to move.
5. Users had weak feedback during refresh, so "still computing" and "stale result" looked the same.

That combination caused event storms on fast slider movement, especially on large moving-radius runs.

## What Changed

### Global Snapshot Interaction

In [main_app.py](C:/Users/dimab/Documents/Helios_parser/src/helios_app/main_app.py):

- the global slider now uses preview-during-drag plus coalesced apply
- only the latest requested snapshot survives
- slider release forces a single immediate apply for the final position
- spinbox/time-entry changes also use the same coalesced request path

### Lightweight Snapshot Refresh

In [analysis.py](C:/Users/dimab/Documents/Helios_parser/src/helios/services/derived/analysis.py) and [controller.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/controller.py):

- full-run analysis and snapshot-local refresh were split
- snapshot changes now reuse cached full-run results when filters/geometry/weighting are unchanged
- time traces are preserved from the cached full result
- only snapshot-local summaries, profiles, and tables are refreshed
- stale request results are rejected through the existing request/generation flow

### Derived Workspace Feedback

In [workspace.py](C:/Users/dimab/Documents/Helios_parser/src/helios_analysis/workspace.py):

- snapshot-only context updates no longer reset filter/weighting/profile controls
- the workspace shows explicit snapshot-update status text
- time-trace plots now show a moving cursor marker at the active global snapshot

### Viewer Colorbar Check

In [plots.py](C:/Users/dimab/Documents/Helios_parser/src/helios_viewer/plots.py):

- the field-map colorbar geometry was pinned to a stable visible width
- the Cu validation rerun confirmed that the colorbar bar itself is present and rendered after snapshot/roundtrip/theme changes

## Which Modules Use Lightweight Refresh vs Full Recompute

### Lightweight refresh on snapshot change

These now update snapshot-local products only:

- `XRD`
- `Plasmon`
- `Transmission`
- `Spectroscopy`

For these modules:

- profile plots refresh for the new global snapshot
- snapshot-local scalar summaries refresh
- titles/labels move to the new snapshot/time
- full time traces are reused
- time traces get an updated cursor marker instead of being recomputed

### Full recompute still required

These still trigger full all-time analysis:

- weighting changes
- geometry changes
- region/material filter changes
- density threshold changes
- zone clip changes
- any change that alters the effective selected-zone mask

Shock tracking remains a full-run product. Snapshot changes only move the snapshot cursor over the existing shock traces.

## Snapshot Propagation Results

Validation script:
[phase412c_snapshot_validation.json](C:/Users/dimab/Documents/Helios_parser/outputs/reports/phase412c_snapshot_validation.json)

Focused tests:

- [test_phase412c_snapshot.py](C:/Users/dimab/Documents/Helios_parser/tests/test_phase412c_snapshot.py)
- [test_app_phase412.py](C:/Users/dimab/Documents/Helios_parser/tests/test_app_phase412.py)
- [test_derived_phase412.py](C:/Users/dimab/Documents/Helios_parser/tests/test_derived_phase412.py)

Measured snapshot-sync results:

| Dataset | Snapshot change | Elapsed | Update kind | Result |
|---|---:|---:|---|---|
| `Cu_0166_stabilized.h5` | `0 -> 180` | `0.308 s` | `snapshot` | all Derived modules changed titles and scalars |
| `10ns+10Si+60Al+15Si+4.27TW_stabilized.h5` | `0 -> 360` | `0.264 s` | `snapshot` | all Derived modules changed titles and scalars |
| `50Al+10E+25CH+3.5TW_stabilized.h5` | `0 -> 175` | `0.262 s` | `snapshot` | all Derived modules changed titles and scalars |

Per-module outcome from the validation script:

- `XRD`: profile title changed, snapshot-local scalars changed, time cursor visible
- `Plasmon`: profile title changed, snapshot-local scalars changed, time cursor visible
- `Transmission`: profile title changed, snapshot-local scalars changed, time cursor visible
- `Spectroscopy`: profile title changed, snapshot-local scalars changed, time cursor visible

Shock traces were not recomputed on snapshot-only changes; both shock position and shock velocity plots updated their cursor markers.

## Busy-State Feedback

The user now gets explicit visible feedback during slider-driven refresh:

- global bar shows `Snapshot N | t = ... | updating...` while a request is pending
- Derived shows `Updating snapshot N @ ... ns...` during the lightweight refresh
- once applied, Derived returns to `Derived mode ready.`

This does not remove all latency on very large runs, but it makes accepted slider changes legible instead of silent.

## Viewer Colorbar Validation

Cu viewer roundtrip artifact:
[phase412c_cu_viewer_colorbar.png](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase412c_cu_viewer_colorbar.png)

Colorbar crop used for visual confirmation:
[phase412c_colorbar_live_crop.png](C:/Users/dimab/Documents/Helios_parser/outputs/screenshots/phase412c_colorbar_live_crop.png)

Validated states on `Cu_0166_stabilized.h5`:

- initial Viewer load
- Mouse Mode
- Viewer -> Derived -> Viewer roundtrip
- dark theme
- light theme

All retained:

- label: `Density [g/cm3]`
- valid finite levels: `0.0001171 -> 12.05`
- rendered mesh mode
- visible color span

## Tests Actually Run

- `python -m unittest discover -v -s tests -p "test_phase412c_snapshot.py"`
- `python -m unittest discover -v -s tests -p "test_app_phase412.py"`
- `python -m unittest discover -v -s tests -p "test_derived_phase412.py"`
- `python -m unittest discover -v -s tests -p "test_viewer_phase4.py"`
- `python scripts\\phase412c_snapshot_validation.py`
- `py_compile` on the touched modules

## Remaining Limitations

- Snapshot changes are now lightweight, but mask/filter/geometry/weighting changes still require full Derived recompute and can remain noticeable on the largest runs.
- The validation here is scripted offscreen Qt interaction, not a subjective desktop-feel benchmark on local interactive hardware.
- Moving-radius workflows remain approximate where boundaries are reconstructed from zone-center data rather than explicit cell edges.
- If a dataset changes only weakly with time, some snapshot-local plots may change subtly rather than dramatically; the active snapshot cursor is now the clearest visual cue in those cases.
