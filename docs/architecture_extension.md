# HELIOS Parse / View Architecture Reference

This is the concise code-facing architecture note for the current 0.9.0 tree.
Use it together with:

- [Documentation index](index.html)
- [Developer guide](developer-guide.html)
- [Developer testing guide](developer-testing.html)

## Top-level application structure

The application is intentionally split into layers:

- [`src/helios_parser/`](../src/helios_parser/): HELIOS log parser, schema normalization, HDF5 writing, lazy HDF5 reader
- [`src/helios/`](../src/helios/): shared runtime context, cache buckets, task utilities, units, and derived services
- [`src/helios_viewer/`](../src/helios_viewer/): production HDF5 viewer, plotting, settings, and viewer controller
- [`src/helios_analysis/`](../src/helios_analysis/): Derived / Analysis workspace and controller
- [`src/helios_app/`](../src/helios_app/): top-level shell, About dialog, session state, and shell routing

The shell lives in [`src/helios_app/main_app.py`](../src/helios_app/main_app.py) and owns the registered
top-level modes:

- `parser`
- `viewer`
- `derived`

Do not create a second parallel application shell for new derived features.

## Coordinate semantics

The central geometry rule remains:

- HELIOS coordinate columns are treated as **edge coordinates**

Current invariants:

- `edge.shape == n_zones + 1`
- `center.shape == n_zones`
- `center[i] = 0.5 * (edge[i] + edge[i + 1])`
- zone width is validated against adjacent edges

Use edges for:

- map extents
- moving-mesh geometry
- boundaries and overlays
- cylindrical shell factors

Use centers for:

- lineouts
- probes
- zone-centered profiles
- scalar zone reporting

Reader/runtime helpers already expose this surface. Do not invent another coordinate interpretation path inside new viewer or derived code.

## Parser and runtime surface

The canonical raw-data path is:

1. parser reads `.log`
2. HDF5 writer persists normalized grid/field/diagnostic state
3. reader lazily re-exposes that state to viewer and derived code

Important runtime helpers include:

- `get_static_coordinate(location="center" | "edge")`
- `get_dynamic_coordinate(location="center" | "edge")`
- `get_run_status()`
- `get_visar_support_metadata()`

Compatibility logic still supports older HDF5 files, but new code should request explicit semantics instead of relying on legacy aliases.

## Shared cache and runtime plumbing

The current reuse path is:

- viewer loads a run
- viewer publishes raw arrays and metadata into the shared raw-data cache
- derived `load_run_data()` consumes that shared cache before reopening HDF5

This keeps viewer and derived on one consistent raw-data plumbing path instead of duplicating load logic.

## Derived orchestration

The derived orchestration entry point is [`src/helios/services/derived/analysis.py`](../src/helios/services/derived/analysis.py).

Important current seams:

- `DerivedModuleContract`
- `AnalysisStateCache`
- shared selection / weighting helpers
- lazy time-plot loading keyed by active tab
- typed result models in `derived/models.py`

Current user-facing split:

- `Shock`: fast legacy quick-look path
- `WaveFront`: advanced multi-branch path
- `Preheat`: separate advanced target-conditioning path

Current module-local lazy tabs:

- XRD
- Plasmon
- Transmission
- Spectroscopy

Do not make WaveFront or Preheat eager on ordinary Shock refresh.

## Plasmon XRTS observable layers

Plasmon now has three explicit comparison levels:

- backend dielectric response
- minimal XRTS observable reconstruction
- material-specific article-native Al XRTS reconstruction

The backend response still comes from the selected dielectric model in
[`src/helios/services/derived/plasmon.py`](../src/helios/services/derived/plasmon.py).
The observable seams sit above that in:

- [`src/helios/services/derived/plasmon_xrts_observable.py`](../src/helios/services/derived/plasmon_xrts_observable.py)
- [`src/helios/services/derived/plasmon_xrts_material.py`](../src/helios/services/derived/plasmon_xrts_material.py)

Keep these responsibilities separate:

- backend: `chi(q, omega)`, `epsilon(q, omega)`, loss, free-electron DSF
- observable layer: free / elastic / bound-core bookkeeping, convolution, observable-level peak extraction, normalization/subtraction provenance

Current observable split:

- `xrts_observable`: minimal Chihara-like Al seam, suitable as a control against the dielectric-only comparison
- `xrts_article_native_al`: material-specific Al assembly with explicit Cromer-Mann elastic form-factor bookkeeping, explicit shell-thresholded bound/core diagnostics, and peak extraction on the inelastic branch after elastic subtraction

The article-native Al layer is still intentionally honest about its limits:

- ion structure factor remains the unity assumption
- no tabulated Al bound-free cross section is available in the repo
- the current benchmark window stays below the first Al L-shell onset, so bound/core inelastic remains explicitly zero in that window
- exact article-side normalization/background subtraction is still not recoverable from the current repo assets

This is a material-specific observable reconstruction layer, not a full article-native atomic forward model.
Do not hide that distinction in UI or reporting.

## WaveFront and Preheat

WaveFront now owns:

- branch tracking
- support/significance labeling
- interface-event summaries
- advanced branch filtering and direction-aware inspection

Preheat is intentionally separate from Shock and WaveFront:

- it consumes `wave_tracking` and optional `interface_events`
- it owns its own region-of-interest selection
- it owns its own target-entry and pre-shock window logic

The app keeps these as separate tabs because they answer different questions.

## Execution and stale-result discipline

Heavy derived work stays off the GUI thread.

Current rules:

- use the controller/task path
- use cooperative cancellation
- latest valid result wins
- stale or malformed result payloads must be rejected before UI apply

Snapshot browsing should remain lightweight after advanced results are already cached.

## UI state and settings flow

Viewer settings remain the canonical user settings surface.

Those settings propagate through the shell into the derived workspace. Examples:

- units
- theme mode
- default profile coordinate
- wheel guard

If a UI behavior spans viewer and derived, follow the settings/shell propagation path rather than adding a second persistence source.

## File-level caution points

Be especially careful in:

- [`src/helios_viewer/main_window.py`](../src/helios_viewer/main_window.py): viewer state, settings, focus, coordinate/UI semantics
- [`src/helios_analysis/workspace.py`](../src/helios_analysis/workspace.py): dense derived UI state, lazy advanced tabs, snapshot synchronization
- [`src/helios_analysis/controller.py`](../src/helios_analysis/controller.py): request keys, caching, stale-result handling
- [`src/helios/services/derived/analysis.py`](../src/helios/services/derived/analysis.py): orchestration, lazy advanced flow
- [`src/helios/services/derived/preheat.py`](../src/helios/services/derived/preheat.py): ROI-aware target-entry and budget logic
- [`src/helios/services/derived/shock_tracking.py`](../src/helios/services/derived/shock_tracking.py): branch and interface-event logic

## Non-negotiable engineering rules

- do not bypass the reader with ad hoc HDF5 logic in viewer or derived code
- do not put heavy derived work on the GUI thread
- do not duplicate module-local settings in global UI state
- do not silently fabricate missing scientific fields
- do not treat provisional advanced detections as stable tracked branches in default UI
