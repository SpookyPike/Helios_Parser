# BPF Primary H5D/HDF5 Schema

Date: 2026-05-10

## Source Precedence

The production parser treats `.bpf` as the primary full-data source when a BPF
file is passed directly or when a same-stem BPF companion exists next to the
input log. The text `.log` parser remains the source for setup metadata,
readable diagnostics, material/region tables, and fallback parsing when no BPF
is available. `.exo` remains optional validation/subset data and is not
required for full BPF parsing.

## Layout

The H5D file keeps the existing reader-compatible layout:

- `/grid`: static mesh, material IDs, region IDs, and optional photon-energy
  boundaries.
- `/time`: time and cycle arrays.
- `/fields/<field_name>`: the actual numeric datasets.
- `/field_metadata/<field_name>`: per-field metadata attributes.
- `/metadata`: global schema, provenance, source-file, run-status, and setup
  metadata.
- `/diagnostics`: log-derived diagnostics when available.

Schema version `2.0` adds self-describing field metadata while preserving the
legacy `/fields/<name>` dataset contract.

## Field Metadata

Every new field dataset stores:

- `field_name`
- `source`
- `dimensions`
- `units` / `unit`
- `label`
- `status`
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
- LOG `Pressure` maps to derived `pressure_j_cm3 =
  ion_pressure_j_cm3 + electron_pressure_j_cm3`.
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
