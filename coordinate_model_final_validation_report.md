# Coordinate / Status Final Validation Report

## Conservative rounded-edge fallback

The post-Prompt-B rounded-edge fallback is now strictly conservative:

- cumulative-width edge reconstruction is only used when the parsed edge column is non-monotonic and the preserved boundary plus widths reproduces the parsed geometry within an explicit file-precision tolerance
- monotonic parsed edges are no longer silently replaced just because they disagree with width-implied edges
- the width-consistency tolerance is explicit in code: `0.5%` of the largest local zone width, with a hard floor of `1e-12`
- materially conflicting monotonic parsed edges are preserved and surfaced as coordinate-validation issues instead of being silently corrected
- tiny negative cylindrical inner-edge values caused by floating-point noise are clipped to `0` only when they are inside an explicit near-zero tolerance; larger negative cylindrical edges still raise

Relevant code:

- `src/helios_parser/coordinates.py`
- `tests/test_coordinate_semantics.py`

## Run status / incompletion handling

The parser now distinguishes:

- `completed`
- `aborted`
- `truncated`
- `unknown`

The scientific dataset is built only from fully valid snapshots.

If the final indexed snapshot block is damaged or partially cut:

- the final damaged block is dropped
- the last fully valid snapshot becomes the effective final scientific state
- run-status metadata records that the tail was dropped and why
- earlier valid snapshots remain usable

Relevant code:

- `src/helios_parser/document.py`
- `src/helios_parser/model.py`
- `src/helios_parser/hdf5.py`
- `src/helios_parser/reader.py`
- `tests/test_run_status.py`

## Exact local cylindrical geometry corrections

Implemented exact local cylindrical correction:

- cylindrical `mass` weighting now uses explicit shell geometry from edge coordinates, `rho * (r_outer^2 - r_inner^2)`, instead of a planar `rho * dr` approximation

Still approximate and explicitly labeled as such:

- cylindrical LOS / path integration in plasmon, transmission, and spectroscopy remains slab-like rather than a full cylindrical shell transport integral

Relevant code:

- `src/helios/services/derived/selection.py`
- `src/helios/services/derived/plasmon.py`
- `src/helios/services/derived/transmission.py`
- `src/helios/services/derived/spectroscopy.py`
- `tests/test_derived_services_phase4.py`

## Derived assumption-validity warnings

Added conservative picosecond-drive warnings based on parsed laser power-table duration.

Current heuristic:

- runs with inferred drive duration `<= 1 ns` are flagged as ps/sub-ns quick-look cases

Affected modules:

- XRD
- plasmon / XRTS
- transmission
- spectroscopy

Relevant code:

- `src/helios/services/derived/common.py`
- `src/helios/services/derived/xrd.py`
- `src/helios/services/derived/plasmon.py`
- `src/helios/services/derived/transmission.py`
- `src/helios/services/derived/spectroscopy.py`
- `tests/test_derived_phase41.py`

## Compatibility behavior

Older HDF5 files remain loadable.

If explicit run-status metadata is absent:

- `HeliosRun.get_run_status()` returns `state = unknown`
- `source = legacy_hdf5`
- the file is not silently treated as completed

Relevant code:

- `src/helios_parser/reader.py`
- `tests/test_run_status.py`

## End-to-end validation summary

### `5Fe+4.9TW+light.log`

- geometry: `PLANAR`
- status: `unknown` (`footer_absent`)
- indexed snapshots: `8`
- valid snapshots: `8`
- last valid time: `6.6497e-10 s`
- static boundary preserved: `edge[0] = 0.0`
- first center: `5.0e-09 cm`
- strict monotonicity: `True`
- midpoint center invariant: satisfied exactly
- note surfaced: last valid snapshot time differs from the header-declared maximum simulation time; this is recorded, not treated as a parser failure

### `Cu1e17.log`

- geometry: `PLANAR`
- status: `completed`
- indexed snapshots: `61`
- valid snapshots: `61`
- last valid time: `3.0000e-11 s`
- static boundary preserved: `edge[0] = 0.0`
- first center: `1.25e-06 cm`
- strict monotonicity: `True`
- midpoint center invariant: satisfied exactly
- malformed exponent tokens parse successfully
- inferred laser pulse duration: `1.2107607e-10 s`
- derived ps/sub-ns warnings surfaced in: `xrd`, `plasmon`, `transmission`, `spectroscopy`

### `Cu1e17_cyl.log`

- geometry: `CYLINDRICAL`
- status: `completed`
- indexed snapshots: `71`
- valid snapshots: `71`
- last valid time: `3.4898e-11 s`
- cylindrical inner boundary preserved and non-negative: `edge[0] = 0.0`
- first center: `1.25e-06 cm`
- strict monotonicity: `True`
- midpoint center invariant: satisfied exactly
- inferred laser pulse duration: `1.2107607e-10 s`
- derived ps/sub-ns warnings surfaced in: `xrd`, `plasmon`, `transmission`, `spectroscopy`
- cylindrical approximation warnings surfaced in: `plasmon`, `transmission`, `spectroscopy`

### Deliberately truncated final-block case

Source base:

- `5Fe+4.9TW+light.log`

Synthetic damage:

- final block cut before the second table

Observed behavior:

- parser status: `truncated`
- indexed snapshots: `8`
- valid snapshots retained: `7`
- damaged final-block reason: `Snapshot at cycle 23891 is missing the second field table.`
- HDF5 roundtrip preserves truncated status and the reduced valid snapshot count

### Older HDF5 compatibility

Checked:

- `outputs/hdf5/Cu_0166_stabilized.h5`

Observed behavior:

- compatibility status: `unknown`
- status source: `legacy_hdf5`
- file remains readable without silently claiming completion or new-edge semantics it does not store explicitly

## Automated checks run

- `python -m py_compile src/helios_parser/coordinates.py src/helios_parser/model.py src/helios_parser/document.py src/helios_parser/parser.py src/helios_parser/hdf5.py src/helios_parser/reader.py src/helios/services/derived/common.py src/helios/services/derived/selection.py src/helios/services/derived/xrd.py src/helios/services/derived/plasmon.py src/helios/services/derived/transmission.py src/helios/services/derived/spectroscopy.py tests/test_coordinate_semantics.py tests/test_run_status.py tests/test_derived_phase41.py tests/test_derived_services_phase4.py`
- `python -m unittest discover -v -s tests -p "test_coordinate_semantics.py"`
- `python -m unittest discover -v -s tests -p "test_run_status.py"`
- `python -m unittest discover -v -s tests -p "test_derived_phase41.py"`
- `python -m unittest discover -v -s tests -p "test_derived_services_phase4.py"`
- `python -m unittest discover -v -s tests -p "test_reader.py"`
- `python -m unittest discover -v -s tests -p "test_validation.py"`
- `python -m unittest discover -v -s tests -p "test_viewer_phase34.py"`
- `python -m unittest discover -v -s tests -p "test_viewer_phase4.py"`

## Remaining caveats

- Some real HELIOS dynamic edge tables remain monotonic but materially inconsistent with the width-implied cumulative grid; these are now preserved and surfaced as coordinate-validation issues rather than silently corrected.
- Cylindrical LOS/path integration in plasmon, transmission, and spectroscopy is still approximate.
- Shock / VISAR was not implemented here; only status/readiness and coordinate/geometry correctness plumbing was hardened.
