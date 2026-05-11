# BPF / EXO Full Field Map and Parser Plan

Date: 2026-05-10

Purpose: document every located field/record in the available HELIOS `.bpf`
and `.exo` files, define how to locate each item, and decide whether `.exo`
files are needed for future full parsing.

## Short Verdict

For full parsing, `.bpf` should become the primary data source. The `.exo` file
is not required to parse the full data payload because it contains only a
SPECT3D-oriented subset. It is useful as an optional validation source and as a
fallback subset importer when `.bpf` is absent.

The current `.log` parser should remain useful for human-readable setup
metadata, input tables, diagnostics text, and compatibility with runs that do
not have `.bpf`.

## BPF File Structure

The inspected `.bpf` files are little-endian Fortran unformatted streams:

```text
int32 payload_byte_count
payload
int32 payload_byte_count
```

File-level records:

| Absolute record | Meaning | Type |
| ---: | --- | --- |
| `0` | run date | ASCII |
| `1` | run wall-clock time | ASCII |
| `2...` | repeated snapshot blocks | mixed |
| final trailer | all-zero 240-byte record | `float64[30]` |

Snapshot layout:

```python
first_header = record[2]
n_nodes = int(first_header[4])
n_zones = n_nodes - 1
n_freq_bins = int(first_header[6])
snapshot_record_count = 69 + 2 * n_zones
snapshot_start(i) = 2 + i * snapshot_record_count
```

After the 69 fixed per-snapshot records, there are `n_zones` ionization blocks:

```python
flag_record_for_zone_z = snapshot_start + 69 + 2*z
fraction_record_for_zone_z = snapshot_start + 70 + 2*z
```

The charge-state fraction width is `Z + 1`, where `Z` is the material atomic
number. In the samples: Cu has 30 states (`q=0..29`), Fe has 27 states
(`q=0..26`).

## Fixed BPF Snapshot Record Map

Status meanings:

- `known`: validated by `.exo`, `.log`, unit integration, or direct physical
  consistency.
- `mapped`: field meaning is supported by HydroPLOT/manual evidence and sample
  behavior, but no direct LOG/EXO numerical column exists for full validation.
- `partial`: structure and some entries are understood, but not all elements
  are named.
- `unknown`: location/shape is known; physics label is not yet validated.

| Rel. record | Key | Type | Shape | Unit | Status | Location rule |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | `snapshot_header` | f64 | 30 | | known | `start+0` |
| 1 | `global_radiation_energy_summary` | f64 | 70 | mixed | partial | `start+1` |
| 2 | `global_source_summary` | f64 | `n_freq_bins` | mixed summary vector | partial | `start+2` |
| 3 | `bpf_record_03` | f64 | 50 | | unknown | `start+3` |
| 4 | `node_position_cm` | f64 | `n_nodes` | cm | known | `start+4` |
| 5 | `zone_mass` | f64 | `n_zones` | `g/cm**X` | known | `start+5` |
| 6 | `ion_temperature_eV` | f64 | `n_zones` | eV | known | `start+6` |
| 7 | `radiation_temperature_eV` | f64 | `n_zones` | eV | known | `start+7` |
| 8 | `mass_density_g_cm3` | f64 | `n_zones` | g/cm3 | known | `start+8` |
| 9 | `mean_charge` | f64 | `n_zones` | dimensionless | known | `start+9` |
| 10 | `ion_pressure_j_cm3` | f64 | `n_zones` | J/cm3 | known | `start+10` |
| 11 | `radiation_cooling_j_g_s` | f64 | `n_zones` | J/g/s | known | `start+11` |
| 12 | `radiation_heating_j_g_s` | f64 | `n_zones` | J/g/s | known | `start+12` |
| 13 | `ion_energy_j_g` | f64 | `n_zones` | J/g | known | `start+13` |
| 14 | `bpf_record_14` | f64 | `n_zones+2` | J/cm3 | known | `start+14` |
| 15 | `bpf_record_15` | f64 | `n_nodes` | | partial | `start+15` |
| 16 | `interface_velocity_cm_s` | f64 | `n_nodes` | cm/s | known | `start+16` |
| 17 | `laser_deposition_j_g_s` | f64 | `n_zones` | J/g/s | known | `start+17` |
| 18 | `bpf_record_18` | f64 | `n_nodes` | | partial | `start+18` |
| 19 | `bpf_record_19` | f64 | `n_nodes` | | partial | `start+19` |
| 20 | `artificial_viscosity_j_cm3` | f64 | `n_zones` | J/cm3 | known | `start+20` |
| 21 | `bpf_record_21` | f64 | `n_nodes` | | partial | `start+21` |
| 22 | `bpf_record_22` | f64 | `n_nodes` | | partial | `start+22` |
| 23 | `bpf_record_23` | f64 | `n_nodes` | | partial | `start+23` |
| 24 | `bpf_record_24` | f64 | `n_nodes` | | partial | `start+24` |
| 25 | `bpf_record_25` | f64 | `n_nodes` | | partial | `start+25` |
| 26 | `bpf_record_26` | f64 | `n_nodes` | | partial | `start+26` |
| 27 | `bpf_record_27` | f64 | `n_nodes` | | partial | `start+27` |
| 28 | `bpf_record_28` | f64 | `n_nodes` | | partial | `start+28` |
| 29 | `bpf_record_29` | f64 | `n_zones` | | unknown | `start+29` |
| 30 | `planck_mean_opacity_absorption_cm2_g` | f64 | `n_zones` | cm2/g | mapped | `start+30` |
| 31 | `planck_mean_opacity_emission_cm2_g` | f64 | `n_zones` | cm2/g | mapped | `start+31` |
| 32 | `rosseland_mean_opacity_cm2_g` | f64 | `n_zones` | cm2/g | mapped | `start+32` |
| 33 | `ion_density_cm3` | f64 | `n_zones` | 1/cm3 | known | `start+33` |
| 34 | `boundary_net_flux_pair_j_s_cm2` | f64 | 2 | J/s/cm2 | known | `start+34` |
| 35 | `bpf_record_35` | i32 | 1 | count | mapped | `start+35` |
| 36 | `bpf_record_36` | f64 | 2 | | partial | `start+36` |
| 37 | `region_net_cooling_rate_j_s_cm2` | f64 | 1 | J/s/cm2 | known | `start+37` |
| 38 | `region_ion_flux_boundary_j_s_cm2` | f64 | 1 | J/s/cm2 | known | `start+38` |
| 39 | `region_electron_flux_boundary_j_s_cm2` | f64 | 1 | J/s/cm2 | known | `start+39` |
| 40 | `bpf_record_40` | f64 | 1 | | partial | `start+40` |
| 41 | `electron_temperature_eV` | f64 | `n_zones` | eV | known | `start+41` |
| 42 | `electron_pressure_j_cm3` | f64 | `n_zones` | J/cm3 | known | `start+42` |
| 43 | `electron_density_cm3` | f64 | `n_zones` | 1/cm3 | known | `start+43` |
| 44 | `bpf_record_44` | f64 | `n_zones` | | unknown | `start+44` |
| 45 | `frequency_group_boundaries_eV` | f64 | `n_freq_bins+1` | eV | known | `start+45` |
| 46 | `radiation_source_escaping_rmax_j_s_cm2_eV` | f64 | `n_freq_bins` | J/s/cm2/eV | mapped | `start+46` |
| 47 | `radiation_source_escaping_rmin_j_s_cm2_eV` | f64 | `n_freq_bins` | J/s/cm2/eV | mapped | `start+47` |
| 48 | `radiation_loss_rmax_j_cm2_eV` | f64 | `n_freq_bins` | J/cm2/eV | known | `start+48` |
| 49 | `radiation_loss_rmin_j_cm2_eV` | f64 | `n_freq_bins` | J/cm2/eV | known | `start+49` |
| 50 | `radiation_net_flux_rmin_j_s_cm2_eV` | f64 | `n_freq_bins` | J/s/cm2/eV | known | `start+50` |
| 51 | `radiation_net_flux_rmax_j_s_cm2_eV` | f64 | `n_freq_bins` | J/s/cm2/eV | known | `start+51` |
| 52 | `external_rad_source_rmin` | f64 | `n_freq_bins` | | known | `start+52` |
| 53 | `external_rad_source_rmax` | f64 | `n_freq_bins` | | known | `start+53` |
| 54 | `bpf_record_54` | i32 | 1 | | partial | `start+54` |
| 55 | `bpf_record_55` | f64 | 1 | | partial | `start+55` |
| 56 | `bpf_record_56` | i32 | 1 | | partial | `start+56` |
| 57 | `region_index_by_zone` | i32 | `n_zones` | dimensionless | known | `start+57` |
| 58 | `bpf_record_58` | i32 | 1 | | partial | `start+58` |
| 59 | `bpf_record_59` | i32 | 1 | | partial | `start+59` |
| 60 | `bpf_record_60` | i32 | 1 | | partial | `start+60` |
| 61 | `bpf_record_61` | i32 | 1 | | partial | `start+61` |
| 62 | `ionization_fraction_width` | i32 | 1 | count | known | `start+62` |
| 63 | `bpf_record_63` | i32 | 1 | | partial | `start+63` |
| 64 | `atomic_number_z` | i32 | 1 | count | known | `start+64` |
| 65 | `bpf_record_65` | i32 | 1 | | partial | `start+65` |
| 66 | `bpf_record_66` | f64 | 1 | | partial | `start+66` |
| 67 | `bpf_record_67` | i32 | 1 | | partial | `start+67` |
| 68 | `ionization_fraction_width_repeat` | i32 | 1 | count | known | `start+68` |

## Production Derived LOG-Compatible Fields

Same-run Cu LOG/BPF validation added these derived BPF fields for log-compatible
reader aliases:

| Derived field | Alias | Evidence |
| --- | --- | --- |
| `zone_outer_velocity_cm_s` | `velocity` | Matches LOG `Velocity`; the centered `fluid_velocity_cm_s` remains EXO-compatible. |
| `pressure_j_cm3` | `pressure` | `ion_pressure_j_cm3 + electron_pressure_j_cm3` matches LOG `Pressure` within LOG text precision. |
| `radiation_net_heating_j_g_s` | `radiation_net_heating` | `radiation_heating_j_g_s - radiation_cooling_j_g_s` matches LOG `RadNetHeat`. |
| `radiation_pressure_j_cm3` | `pressure_radiation` | One third of BPF record 14 interior matches LOG radiation pressure. |
| `radiation_energy_j_g` | `radiation_energy` | BPF record 14 interior divided by mass density matches LOG radiation energy. |
| `laser_source_j_g` | `laser_source` | Uses aligned LOG `LaserSrc` when a companion LOG is present; BPF-only fallback is trapezoidal integration of `laser_deposition_j_g_s`. Fixed record 29 does not match LOG `LaserSrc`. |

Per-zone trailing records:

| Relative rule | Key | Type | Shape | Status |
| --- | --- | --- | --- | --- |
| `69 + 2*z` | `ionization_fraction_flags_by_zone[z]` | i32 | 1 | known structurally |
| `70 + 2*z` | `ionization_fractions_by_zone_charge[z, :]` | f64 | `Z+1` | known |

## EXO Variable Map

The sample `.exo` files are NetCDF classic files readable with
`scipy.io.netcdf_file`. They use big-endian NetCDF numeric storage, which the
NetCDF reader handles.

| EXO variable | Dimensions | Meaning |
| --- | --- | --- |
| `elem_num_map` | `num_elem` | element IDs |
| `node_num_map` | `num_nodes` | node IDs |
| `connect1` | `num_el_in_blk1 x num_nod_per_el1` | element-node connectivity |
| `coord` | `num_dim x num_nodes` | initial node coordinate |
| `coor_names` | `num_dim x len_string` | coordinate-name strings |
| `name_nod_var` | `num_nod_var x len_string` | node-variable names; sample: `NODE_X(t)` |
| `name_elem_var` | `num_elem_var x len_string` | element-variable names |
| `name_fd_var` | `num_fd_var x len_string` | frequency-dependent variable names |
| `vals_nod_var` | `time_step x num_nod_var x num_nodes` | dynamic node variables |
| `FREQ_GROUP_BOUNDARIES` | `num_freq_groups` | photon group boundaries, eV |
| `vals_elem_var1eb1` | `time_step x num_el_in_blk1` | `VOL_FRACTION_01` |
| `vals_elem_var2eb1` | `time_step x num_el_in_blk1` | `MASS_DENSITY`, g/cm3 |
| `vals_elem_var3eb1` | `time_step x num_el_in_blk1` | `ELEC_TEMPERATURE`, eV |
| `vals_elem_var4eb1` | `time_step x num_el_in_blk1` | `ION_TEMPERATURE`, eV |
| `vals_elem_var5eb1` | `time_step x num_el_in_blk1` | `FLUID_VELOCITY`, cm/s |
| `vals_fd_var1` | `time_step x num_freq_bins` | `EXTERNAL_RAD_SOURCE_RMIN` |
| `vals_fd_var2` | `time_step x num_freq_bins` | `EXTERNAL_RAD_SOURCE_RMAX` |
| `time_whole` | `time_step` | output times, seconds |

The EXO element variables are named by `name_elem_var`, not by the numbered
variable names themselves. In the two samples:

```text
VOL_FRACTION_01
MASS_DENSITY
ELEC_TEMPERATURE
ION_TEMPERATURE
FLUID_VELOCITY
```

## BPF vs EXO Coverage

Every EXO physics field found in the samples is either directly present in BPF
or derivable from BPF:

| EXO field | BPF location |
| --- | --- |
| `time_whole` | `snapshot_header[1]` |
| `NODE_X(t)` | `start+4` |
| `MASS_DENSITY` | `start+8` |
| `ELEC_TEMPERATURE` | `start+41` |
| `ION_TEMPERATURE` | `start+6` |
| `FLUID_VELOCITY` | average of `start+16` interface velocity |
| `FREQ_GROUP_BOUNDARIES` | `start+45` |
| `EXTERNAL_RAD_SOURCE_RMIN` | `start+52` |
| `EXTERNAL_RAD_SOURCE_RMAX` | `start+53` |

`VOL_FRACTION_01` is constant 1.0 in both EXO samples. I did not assign it to a
named BPF record because no independent nontrivial variation was available.
For parser output it can be represented as an EXO-derived/constant field when
needed, but it should not be claimed as a mapped BPF physics field yet.

BPF contains important fields that EXO does not expose:

- ionization fractions by zone and charge state
- mean charge
- electron and ion pressure
- ion density
- artificial viscosity
- laser deposition/source terms
- radiation heating/cooling
- spectral net radiation flux at both boundaries
- cumulative spectral radiation loss at both boundaries
- several additional raw HydroPLOT records not yet named

## Do We Need EXO for Full Parsing?

No. For future parser support, `.bpf` is sufficient as the full-data source for
the located payload. EXO should be optional.

Recommended policy:

1. If `.bpf` exists, parse `.bpf` as the primary high-fidelity time-series and
   spectral-data source.
2. If `.log` exists, parse `.log` as setup metadata, human-readable diagnostics,
   and a validation/comparison source.
3. If `.exo` exists, use it only as optional validation and/or as a subset
   fallback when `.bpf` is missing.
4. Do not require `.exo` for BPF import. Requiring it would prevent parsing
   valid HELIOS runs where only `.bpf` and `.log` were saved.
5. If `.hna`/`.hnb` plot NetCDF files become available, prefer them as the
   strongest source for final HydroPLOT field labels/units because Prism
   documentation says they contain the same plot data as BPF for non-CR runs.

## Future Parser Implementation Plan

Add a production module, for example `src/helios_parser/bpf.py`, with:

- lazy Fortran-record iterator
- layout inference from first snapshot header
- streaming snapshot reader that does not materialize the whole file by default
- field specs matching `BPF_FIXED_RECORD_SPECS`
- known-field aliases compatible with current HDF5 field names
- raw-field preservation for unknown records under `raw/bpf_record_XX`
- ionization fraction dataset under a dedicated group, for example
  `fields/ionization_fraction[time, zone, charge_state]`
- spectral radiation datasets under a dedicated group, for example
  `radiation/frequency_boundaries_eV`,
  `radiation/net_flux_rmin_j_s_cm2_eV`, and
  `radiation/net_flux_rmax_j_s_cm2_eV`

Minimum validation checks:

- Fortran record markers match.
- `n_nodes > 1`, `n_zones = n_nodes - 1`, `n_freq_bins > 0`.
- snapshot times are finite and nondecreasing.
- frequency boundaries are finite and strictly increasing.
- known zone arrays have shape `n_zones`.
- known node arrays have shape `n_nodes`.
- ionization fractions are finite, nonnegative within tolerance, and sum to 1
  by zone.
- spectral flux integration reproduces scalar BPF/log flux summaries when a log
  is available.

The existing `scripts/inspect_bpf.py` now contains the field specs and an
`extract_all_snapshot_records(...)` function that can be used as the prototype
for the production reader.
