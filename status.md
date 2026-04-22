# HELIOS Analyzer Status

## Current state

HELIOS Analyzer is a three-mode desktop application:

- Parser
- Viewer
- Derived / Analysis

The current implementation includes the full coordinate-model correction,
parser/HDF5 schema clarification, cylindrical viewer handling, damaged-file
robustness, derived quick-look warning surfaces, persistent asynchronous
derived execution, bounded caches, and derived-result stability guards.

## What is now explicit

### Coordinate semantics

- HELIOS coordinate columns are treated as edge coordinates
- row-0 boundary is preserved
- centers are midpoint-derived from edges
- viewer and derived paths no longer rely on ambiguous center/edge semantics

### Parser robustness

- malformed scientific notation such as `4.973-174` is normalized safely
- damaged/partial final blocks are dropped without discarding earlier valid snapshots
- run/file status is surfaced as `completed`, `aborted`, `truncated`, or `unknown`

### Geometry handling

- planar runs use `x`
- cylindrical runs use `radius` / `r`
- viewer map extents, moving-mesh geometry, and boundary placement are edge-based
- viewer lineouts/probes and zone-centered reporting are center-based

### Derived semantics

- exact local cylindrical correction exists for cylindrical mass weighting
- full cylindrical LOS/path integration is still approximate in some derived modules and is warned explicitly
- ps-scale and other weaker-assumption cases are surfaced through warning metadata instead of silent overclaiming

### Execution model

- derived work runs asynchronously off the GUI thread
- the task system uses a persistent worker execution model
- cooperative cancellation is used for superseded work
- latest-wins result delivery is enforced
- stale/cancelled/invalid derived results are rejected before UI apply

### Cache behavior

- cache buckets are bounded
- cache stats now expose size, capacity, hits, misses, evictions, clear counts, and last clear reason where available

## Current quick-look limits

- shock tracking remains a heuristic primary-front tracker
- XRD remains isotropic compression only
- plasmon/XRTS remains an NRL-based quick look
- transmission remains Thomson-only in shipping behavior
- spectroscopy remains a Doppler/broadening quick look
- full cylindrical transport/path integration is not yet implemented end to end

## Compatibility

- older HDF5 files still load through reader compatibility logic
- runtime coordinate getters default to center coordinates for backward safety
- legacy coordinate aliases still exist for compatibility but should not be used ambiguously in new code

## Documentation set

- [README.md](C:/Users/dimab/Documents/Helios_parser/README.md)
- [docs/index.html](C:/Users/dimab/Documents/Helios_parser/docs/index.html)
- [docs/future-development.html](C:/Users/dimab/Documents/Helios_parser/docs/future-development.html)
- [docs/architecture_extension.md](C:/Users/dimab/Documents/Helios_parser/docs/architecture_extension.md)
- [docs/maintenance.html](C:/Users/dimab/Documents/Helios_parser/docs/maintenance.html)

These now describe the implemented edge/center model, parser robustness,
derived validity warnings, execution model, and cache observability instead of
the older assumptions.
