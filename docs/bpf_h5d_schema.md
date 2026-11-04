# BPF Primary H5D/HDF5 Schema

Date: 2026-11-03 (schema 2.2)

## Source Precedence

The production parser treats `.bpf` as the primary full-data source when a BPF
file is passed directly or when a same-stem BPF companion exists next to the
input log. The text `.log` parser remains the source for setup metadata,
readable diagnostics, material/region tables, and fallback parsing when no BPF
is available. `.exo` remains optional validation/subset data and is not
required for full BPF parsing.

## Layout

The H5D file keeps the existing reader-compatible layout:

- `/grid`: static mesh, material IDs, region IDs, optional photon-energy
  boundaries, and the explicit coordinate grids `coordinate_center`,
  `coordinate_edge`, `dynamic_coordinate_center`, `dynamic_coordinate_edge`.
- `/time`: time and cycle arrays.
- `/fields/<field_name>`: the actual numeric datasets.
- `/field_metadata/<field_name>`: per-field metadata attributes.
- `/metadata`: global schema, provenance, source-file, run-status, and setup
  metadata.
- `/diagnostics`: log-derived diagnostics when available.

Schema version `2.1` adds explicit plasma/total pressure quantities and
corrects the legacy `pressure` alias to mean total pressure. This is a versioned
semantic correction for BPF-derived files. The existing `pressure_j_cm3` field
continues to mean plasma pressure.

Schema version `2.2` changes chunk geometry only. Field, dynamic-coordinate and
`laser_source_*` datasets span 32 snapshots per chunk instead of 16, matching the
writer's flush batch so one flush fills one chunk. Radiation spectral datasets
span one dump per chunk instead of 16, so reading a single dump no longer
inflates sixteen. Axis order and every stored value are unchanged, so this is a
minor bump and readers need no change: chunk layout is invisible through the
HDF5 API. Files written by 1.4.0 and earlier stay readable because the reader
requires the group set, not the version attribute.

Compression is optional and off by default, so a default conversion is
uncompressed. `helios_to_hdf5 --compression lzf` enables the HDF5 shuffle filter
as well, because on float64 fields the filter saves 35.7 % under LZF; under
`--compression gzip` it is left off, because there it costs 3 %. Both are layout
details that the HDF5 API hides, so a reader needs no change either way.

## Coordinate semantics

HELIOS prints a `Radius` column per snapshot with one row per zone **boundary**,
so a snapshot with `n_zones` zones has `n_zones + 1` radius rows. The converter
stores those rows in `grid/dynamic_coordinate_edge`, which therefore has shape
`(n_snapshots, n_zones + 1)`, and stores the zone centres in
`grid/dynamic_coordinate_center`, shape `(n_snapshots, n_zones)`.

`fields/radius` is the centre grid, not the edge grid. Every other field is
zone-centred, so this keeps `radius` broadcastable against them. A consumer that
needs boundaries reads `dynamic_coordinate_edge`; a consumer that subtracts
`radius` from `radius` shifted by one is comparing centres with centres and gets
the wrong zone width.

The static pair follows the same rule: `coordinate_center` and `coordinate_edge`
hold the initial grid, and `coordinate_edge` has one more entry than
`coordinate_center`.

This distinction is checked by `tests/test_log_truth.py`, which parses the
source log's tables directly and confirms that `dynamic_coordinate_edge`
reproduces the printed `Radius` column and that `dynamic_coordinate_center` is
the midpoint of adjacent edges.

## Field Metadata

Every new field dataset stores:

- `field_name`
- `source`
- `dimensions`
- `units` / `unit`
- `label`
- `status`
- optional `quantity` (for example `plasma_pressure` or `total_pressure`)
- optional `description`
- optional `plotting_hints`
- optional `alias_of`

The reader uses this metadata first. If an older log-derived file has no
`/field_metadata` group, the reader infers safe legacy defaults from dataset
shape and unit attrs.

## Axis Semantics

Known axis names include:

- `time`
- `zone`
- `node`
- `frequency`
- `frequency_edge`
- `boundary`
- `charge_state`
- `region`
- `summary_value`
- `bpf_record_value`

Examples:

- log-compatible hydro fields: `("time", "zone")`
- BPF spectral flux/loss fields: `("time", "frequency")`
- BPF boundary flux pair: `("time", "boundary")`
- BPF ionization fractions: `("time", "zone", "charge_state")`
- BPF node coordinates/interface quantities: `("time", "node")`

## Unknown BPF Records

Unvalidated BPF records are preserved under stable names such as
`bpf_record_03`. Their metadata status is `unknown_bpf_record`, units are left
blank, and labels remain neutral. The parser does not assign physics names or
units without validation.

`bpf_record_03` is a fixed 50-wide raw vector in the current Cu and Fe sample
runs; it is not treated as a zone-width vector. `bpf_record_29` is preserved as
raw unknown data because same-run LOG comparison showed it is not the LOG
`LaserSrc` field.

## Resolved BPF/LOG Compatibility Fields

Same-run Cu LOG/BPF comparison established these production aliases and derived
fields:

- LOG `Velocity` maps to `zone_outer_velocity_cm_s`, the right/outer BPF
  interface velocity for each zone. `fluid_velocity_cm_s` remains the
  EXO-compatible zone-centered average of adjacent interface velocities.
- `pressure_j_cm3` and `plasma_pressure_j_cm3` mean plasma pressure:
  `ion_pressure_j_cm3 + electron_pressure_j_cm3`.
- `total_pressure_j_cm3` means plasma plus radiation pressure. The legacy
  `pressure` field is a hard-link alias of `total_pressure_j_cm3`, matching the
  LOG parser and derived-analysis interpretation.
- LOG `RadNetHeat` maps to derived `radiation_net_heating_j_g_s =
  radiation_heating_j_g_s - radiation_cooling_j_g_s`.
- LOG `Prad` / `pressure_radiation` maps to
  `radiation_pressure_j_cm3`, derived from BPF record 14.
- LOG `Erad` / `radiation_energy` maps to `radiation_energy_j_g`,
  derived from BPF record 14 divided by mass density.
- LOG `LaserSrc` maps to `laser_source_j_g`. When an aligned LOG companion is
  available this field is copied from LOG because HELIOS integrates the
  cumulative source on internal timesteps. BPF-only files use sampled
  trapezoidal integration of `laser_deposition_j_g_s` as the fallback.

BPF record 14 is validated as a padded radiation-energy-density vector:
interior entries are zone-centered `J/cm3`, endpoint entries are boundary
padding, the interior equals `3 * pressure_radiation`, and
`radiation_energy_density_j_cm3 / density` matches LOG radiation energy.

Records 30, 31, and 32 are mapped to HydroPLOT opacity fields using bundled
HydroPLOT UI/manual evidence and sample-data behavior:

- `planck_mean_opacity_absorption_cm2_g`
- `planck_mean_opacity_emission_cm2_g`
- `rosseland_mean_opacity_cm2_g`

These opacity fields use axes `("time", "zone")`, unit `cm2/g`, and status
`mapped`, not `validated`, because no LOG column provides a direct numerical
cross-check.

## Input Routing

`write_hdf5()` now performs explicit file-type routing:

- `.bpf` input parses BPF directly and uses a same-stem `.log` companion for
  setup metadata when present.
- `.log` input uses a same-stem `.bpf` companion as the primary full-data source
  when present.
- `.log` without a BPF companion remains a log-only conversion path.
- `.bpf` without a LOG companion remains a BPF-only conversion path with
  minimal metadata inferred from the BPF layout.

The chosen source model is recorded in `/metadata/source_precedence`,
`/metadata/parse_mode`, and `/metadata/source_files`.

## Reader Behavior

`HeliosRun.list_fields()` enumerates the file’s actual field inventory.
`HeliosRun.get_field_metadata(name)` returns the stored or inferred metadata.
`HeliosRun.plotting_modes_for_field(name)` derives allowed high-level views
from axis semantics, so sparse LOG-only files and richer BPF-derived files can
coexist without a fixed universal field list.
