# HELIOS BPF Format Investigation

Date: 2026-05-10

This note records what was verified from the `new_data` BPF samples and the
local Prism HELIOS 6.2.0 installation. It is intentionally conservative:
fields are named only where the BPF records were matched to companion EXO data,
the text log, or unambiguous structural evidence.

## Input Files

| Run | BPF size | Companion files inspected |
| --- | ---: | --- |
| `new_data/25Cu+1.87TW/25Cu+1.87TW.bpf` | 40,375,894 bytes | `.exo`, `.log`, `.rhc` |
| `new_data/5Fe+4.9TW+light/5Fe+4.9TW+light.bpf` | 2,192,378 bytes | `.exo`, `.rhc`; repository root `.log` |

Local Prism documentation consulted:

- `C:/Program Files (x86)/Prism/Helios_6.2.0/doc/html/files/output_files.html`
- `C:/Program Files (x86)/Prism/Helios_6.2.0/doc/html/setup/output_controls.html`
- `C:/Program Files (x86)/Prism/Helios_6.2.0/doc/html/hydroplot_user_interface/frequency_tab.html`

The HELIOS docs identify `.bpf` as the binary plot file used by HydroPLOT.
They also state that, for non-CR calculations, the plot NetCDF files `.hna`
and `.hnb` contain the same plotting data as the binary plot format. The `.exo`
file is a separate NetCDF file for SPECT3D workflows and contains only a subset
of the BPF plot data.

## Container Format

The BPF file is a little-endian Fortran sequential unformatted stream:

```text
int32 payload_byte_count
payload bytes
int32 payload_byte_count
```

Every record in both samples has matching leading and trailing byte counts.
The first two records are ASCII strings:

- record 0: run date, for example `08/13/11`
- record 1: run clock time, for example `12:49:04.5`

Record 2 is the first snapshot header and is 240 bytes, interpreted as 30
little-endian `float64` values.

## Snapshot Blocking

For both inspected files:

- `n_nodes = int(header[4])`
- `n_zones = n_nodes - 1`
- `n_freq_bins = int(header[6])`
- one snapshot block contains `69 + 2 * n_zones` records
- snapshot `i` starts at record `2 + i * (69 + 2 * n_zones)`
- both samples end with one 240-byte all-zero trailer record after the final
  complete snapshot block

| Run | Records | Zones | Nodes | Frequency bins | Snapshot records | Snapshots | Time span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `25Cu+1.87TW.bpf` | 152,272 | 50 | 51 | 200 | 169 | 901 | `1.0e-15` to `4.501613752286756e-09` s |
| `5Fe+4.9TW+light.bpf` | 8,555 | 500 | 501 | 200 | 1,069 | 8 | `1.0e-15` to `6.649669733334302e-10` s |

The header time (`header[1]`) matches the EXO `time_whole` variable exactly for
sampled snapshots. The header cycle (`header[2]`) matches the cycle values seen
in the text log for inspected records.

## Validated Field Map

Offsets below are relative to the start of one snapshot block.

| Relative record | Field | Type and shape | Validation |
| ---: | --- | --- | --- |
| `0` | snapshot header | `float64[30]` | time and cycle checked against EXO/log |
| `4` | node position, cm | `float64[n_nodes]` | exact match to EXO `vals_nod_var[:, 0, :]` |
| `5` | zone mass | `float64[n_zones]` | consistent with density and initial zone widths |
| `6` | ion temperature, eV | `float64[n_zones]` | exact match to EXO `vals_elem_var4eb1` |
| `8` | mass density, g/cc | `float64[n_zones]` | exact match to EXO `vals_elem_var2eb1` |
| `16` | interface velocity, cm/s | `float64[n_nodes]` | zone-centered average matches EXO fluid velocity |
| `33` | ion density, cm^-3, inferred | `float64[n_zones]` | order and initial values match log ion-density output |
| `41` | electron temperature, eV | `float64[n_zones]` | exact match to EXO `vals_elem_var3eb1` |
| `45` | frequency group boundaries, eV | `float64[n_freq_bins + 1]` | exact match to EXO `FREQ_GROUP_BOUNDARIES` |
| `46` | escaping radiation source at Rmax, spectral | `float64[n_freq_bins]`, `J/s/cm2/eV` | bin-width integral matches terminal boundary flux in log |
| `47` | escaping radiation source at Rmin, spectral outward magnitude | `float64[n_freq_bins]`, `J/s/cm2/eV` | bin-width integral matches magnitude of log Rmin net boundary flux |
| `48` | cumulative radiation loss at Rmax, spectral | `float64[n_freq_bins]`, `J/cm2/eV` | bin-width integral contributes to log radiation lost from grid |
| `49` | cumulative radiation loss at Rmin, spectral | `float64[n_freq_bins]`, `J/cm2/eV` | bin-width integral contributes to log radiation lost from grid |
| `50` | net radiation flux at Rmin, spectral, positive in +R direction | `float64[n_freq_bins]`, `J/s/cm2/eV` | bin-width integral matches signed log Rmin net boundary flux |
| `51` | net radiation flux at Rmax, spectral, positive in +R direction | `float64[n_freq_bins]`, `J/s/cm2/eV` | bin-width integral matches log terminal boundary flux |
| `52` | external radiation source at Rmin | `float64[n_freq_bins]` | exact match to EXO `vals_fd_var1` |
| `53` | external radiation source at Rmax | `float64[n_freq_bins]` | exact match to EXO `vals_fd_var2` |
| `57` | region index by zone | `int32[n_zones]` | all zones map to region `1` in both single-region samples |
| `69 + 2*z` | ionization fraction flag/count for zone `z` | `int32[1]` | structural match |
| `70 + 2*z` | ionization fractions for zone `z` | `float64[Z + 1]` | width is 30 for Cu, 27 for Fe; cold initial state is neutral-dominant |

The BPF stores interface velocities. The EXO file stores zone-centered fluid
velocity, which is recovered as:

```python
fluid_velocity = 0.5 * (interface_velocity[:-1] + interface_velocity[1:])
```

## Partially Identified Content

HydroPLOT strings in the installed Prism binaries show that BPF can contain
many more plot variables than the current text-log parser exposes, including
radiation energy density, Rosseland opacity, radiation flux and power-loss
quantities, laser deposition, magnetic quantities, plasma pressure, ion and
electron heating, ionization fractions, and populations.

Several records in the fixed 69-record snapshot prefix have dimensions and
value ranges consistent with these HydroPLOT quantities, but they are not yet
mapped with enough evidence to expose as named production fields. They should
remain raw/unknown until a `.hna` or `.hnb` plot NetCDF file, HydroPLOT source
metadata, or more varied BPF/EXO/log cross-validation confirms their names and
units.

## Extraction Strategy

Use `scripts/inspect_bpf.py` for current validated extraction:

```powershell
python scripts\inspect_bpf.py new_data --validate-exo
python scripts\inspect_bpf.py new_data\25Cu+1.87TW\25Cu+1.87TW.bpf --extract-snapshot 0 --output-npz outputs\cu_snapshot0_bpf_common.npz
```

The script:

- validates Fortran record markers
- infers mesh size, frequency-bin count, snapshot count, and trailer records
- extracts only validated common fields
- optionally validates against same-stem `.exo` files when SciPy is available
- can write one snapshot's validated arrays to compressed NumPy `.npz`

## Validation Run

Command executed:

```powershell
python scripts\inspect_bpf.py new_data --validate-exo
```

Result:

- `25Cu+1.87TW.bpf`: 25/25 EXO checks passed across snapshots 0, 450,
  and 900.
- `5Fe+4.9TW+light.bpf`: 25/25 EXO checks passed across snapshots 0, 4,
  and 7.
- For all sampled snapshots, BPF time, node position, mass density, electron
  temperature, ion temperature, derived zone-centered fluid velocity, external
  radiation source spectra, and frequency group boundaries matched EXO exactly.
- For sampled snapshots, photon-energy bin integrals of records 50 and 51
  reproduce the signed log boundary fluxes in `J/s/cm2`; therefore the
  frequency-resolved flux arrays have units `J/s/cm2/eV`.
- Photon-energy bin integrals of records 48 and 49 sum to the log cumulative
  radiation loss from grid in `J/cm2`; therefore those arrays have units
  `J/cm2/eV`.

## Recommended Parser Direction

1. Keep BPF ingestion experimental until more variables are mapped.
2. Add production support first for the validated fields above, because they
   cover dynamic mesh, density, temperatures, velocity, external radiation
   source spectra, region IDs, and ionization fractions.
3. Treat unknown records as raw records with relative offsets and dimensions,
   not named physics.
4. If `.hna` and `.hnb` examples become available, use them as the strongest
   source of field names and units because Prism documents them as equivalent
   plot data for non-CR calculations.
5. Add sanity checks before importing BPF data into stabilized HDF5:
   monotonic nonnegative time, positive node count, positive frequency bins,
   monotonic frequency boundaries, finite density/temperature/velocity arrays,
   and expected zone/node dimensions for every snapshot.

## Current Limitations

- Only two BPF samples were available.
- Both samples are single-region runs, so multi-region region-index behavior is
  not yet validated.
- No `.hna` or `.hnb` files were present, so many HydroPLOT-specific fields are
  still unmapped.
- EXO validation covers only the subset exported for SPECT3D, not the full BPF
  payload.
- The ionization-fraction structure is clear, but charge-state ordering should
  be confirmed against a CR/population output before user-facing naming.
- `scripts/inspect_bpf.py` is an investigation/prototyping tool and loads the
  BPF payload records into memory. A production importer should stream records
  snapshot by snapshot.
