# LOG, BPF, and EXO Field Comparison

Date: 2026-05-10

This report compares the data available in the sample HELIOS `.log`, `.bpf`,
and `.exo` files and records the BPF-only fields that were extracted and
plotted.

## File Roles

| Format | Role | Strength | Limitation |
| --- | --- | --- | --- |
| `.log` | Human-readable simulation setup, zone tables, and scalar diagnostics | Rich metadata, units printed in text, easy to audit | Lower precision text, bulky parsing, no full charge-state distributions, no frequency-resolved boundary radiation spectra |
| `.bpf` | HydroPLOT binary plot file | Full plot payload, high precision, includes radiation spectra and ionization fractions | Proprietary/undocumented layout; fields must be mapped by validation |
| `.exo` | NetCDF SPECT3D export | Self-describing NetCDF subset; excellent validation anchor | Contains only a subset of BPF/plot data |

Prism HELIOS documentation says the binary plot file can also be written as
`.hna`/`.hnb` NetCDF plot files for non-CR calculations; those are the closest
documented equivalents to `.bpf`. The `.exo` file is not equivalent to `.bpf`;
it is a SPECT3D-oriented subset.

## `.log` Contents Observed

The active repository parser extracts these snapshot fields from the logs:

| Field | Unit |
| --- | --- |
| radius | cm |
| zone_width | cm |
| density | g/cm3 |
| velocity | cm/s |
| temperature_radiation | eV |
| temperature_i | eV |
| temperature_e | eV |
| pressure_i | J/cm3 |
| pressure_e | J/cm3 |
| pressure_radiation | J/cm3 |
| compression | rho/rho0 |
| electron_density | 1/cm3 |
| mean_charge | dimensionless |
| artificial_viscosity | J/cm3 |
| ion_energy | J/g |
| electron_energy | J/g |
| ion_heat_capacity | J/g/eV |
| electron_heat_capacity | J/g/eV |
| radiation_energy | J/g |
| kinetic_energy | J/g |
| radiation_heating | J/g/s |
| radiation_cooling | J/g/s |
| radiation_net_heating | J/g/s |
| laser_source | J/g |
| laser_deposition | J/g/s |

The logs also contain:

- run setup: materials, regions, EOS/opacity table paths, geometry, time
  controls, laser power table, radiation source setup, and frequency-grid setup
- initial zone masses and region masses
- per-snapshot scalar diagnostics: radiation boundary fluxes, energy summary,
  energy exchange, and energy balance
- event messages, for example hydrodynamics-on messages in the Fe sample

## `.exo` Contents Observed

The sample EXO files contain:

| EXO variable | Meaning | Unit/shape |
| --- | --- | --- |
| `time_whole` | output times | seconds, `time_step` |
| `coord` | initial coordinate | cm, `num_nodes` |
| `vals_nod_var[:, 0, :]` | dynamic node coordinate `NODE_X(t)` | cm, `time_step x num_nodes` |
| `vals_elem_var1eb1` | volume fraction | dimensionless, `time_step x num_elem` |
| `vals_elem_var2eb1` | mass density | g/cm3, `time_step x num_elem` |
| `vals_elem_var3eb1` | electron temperature | eV, `time_step x num_elem` |
| `vals_elem_var4eb1` | ion temperature | eV, `time_step x num_elem` |
| `vals_elem_var5eb1` | fluid velocity | cm/s, `time_step x num_elem` |
| `FREQ_GROUP_BOUNDARIES` | photon group boundaries | eV, `num_freq_groups` |
| `vals_fd_var1` | external radiation source at Rmin | frequency-dependent, `time_step x num_freq_bins` |
| `vals_fd_var2` | external radiation source at Rmax | frequency-dependent, `time_step x num_freq_bins` |

These EXO fields are a useful validation subset. They do not include
charge-state fractions or boundary radiation flux spectra.

## `.bpf` Contents Verified

The BPF file contains all EXO-validated fields above, plus additional plot data.
The following fields have been mapped with enough evidence to extract:

| BPF field | Unit | Also in `.log`? | Also in `.exo`? | Notes |
| --- | --- | --- | --- | --- |
| dynamic node position | cm | log has zone radii, lower precision | yes | exact EXO match |
| zone mass | g/cm**X | yes, initial table | no | BPF stores per-zone mass |
| ion temperature | eV | yes | yes | exact EXO match |
| mass density | g/cm3 | yes | yes | exact EXO match |
| interface velocity | cm/s | log has zone velocity | no | zone average matches EXO fluid velocity |
| electron temperature | eV | yes | yes | exact EXO match |
| frequency group boundaries | eV | setup ranges only | yes | exact EXO match |
| external radiation source Rmin/Rmax | EXO frequency-dependent units | setup says none in samples | yes | zero in samples |
| region index by zone | dimensionless | setup only | no | single-region samples all `1` |
| ionization fractions by zone and charge state | dimensionless | no; log has only mean charge | no | weighted charge sum reproduces log mean charge |
| spectral net radiation flux at Rmin | J/s/cm2/eV | no; log has energy-integrated scalar | no | bin integral matches signed log Rmin flux |
| spectral net radiation flux at Rmax | J/s/cm2/eV | no; log has energy-integrated scalar | no | bin integral matches log terminal flux |
| cumulative spectral radiation loss at Rmin/Rmax | J/cm2/eV | no; log has energy-integrated total loss | no | bin integrals sum to log cumulative radiation loss |

## Unit Validation

Radiation records were validated by integrating over photon-energy bin width:

```python
integrated_flux = sum(spectral_flux[e] * delta_E[e])
```

The integrated Rmin/Rmax fluxes reproduce the log boundary flux diagnostics in
`J/s/cm2`; therefore the BPF spectral flux units are `J/s/cm2/eV`.

Similarly:

```python
integrated_loss = sum((loss_rmin[e] + loss_rmax[e]) * delta_E[e])
```

reproduces the log cumulative radiation lost from grid in `J/cm2`; therefore
the BPF spectral loss units are `J/cm2/eV`.

Full-log comparison results:

| Run | Quantity | Max relative difference |
| --- | --- | ---: |
| `25Cu+1.87TW` | Rmin spectral-flux integral vs log scalar | `4.73e-4` |
| `25Cu+1.87TW` | Rmax spectral-flux integral vs log scalar | `4.90e-4` |
| `25Cu+1.87TW` | spectral-loss integral vs log scalar | `4.75e-5` |
| `5Fe+4.9TW+light` | Rmin spectral-flux integral vs log scalar | `2.20e-4` |
| `5Fe+4.9TW+light` | Rmax spectral-flux integral vs log scalar | `1.89e-4` |
| `5Fe+4.9TW+light` | spectral-loss integral vs log scalar | `1.41e-5` |

The remaining difference is consistent with text-log rounding; BPF retains
higher precision.

## Extracted BPF-Only Data

Script:

```powershell
python scripts\plot_bpf_extra_fields.py new_data
```

Saved arrays:

- `outputs/bpf_extra_fields/25Cu_1_87TW_bpf_extra_fields.npz`
- `outputs/bpf_extra_fields/5Fe_4_9TW_light_bpf_extra_fields.npz`

Each `.npz` contains:

- `times_s`
- `zone_centers_cm`
- `ionization_fractions`
- `mean_charge`
- `dominant_charge`
- `charge_states`
- `photon_group_boundaries_eV`
- `radiation_flux_rmin_j_s_cm2_eV`
- `radiation_flux_rmax_j_s_cm2_eV`
- `radiation_loss_rmin_j_cm2_eV`
- `radiation_loss_rmax_j_cm2_eV`

## Generated Contour Figures

For each run, the script generated:

- final charge-state fraction vs zone and charge state
- selected-zone charge-state history vs time and charge state
- dominant charge state vs time and zone
- spectral net radiation flux at Rmin vs time and photon energy
- spectral net radiation flux at Rmax vs time and photon energy
- cumulative spectral radiation loss at Rmin vs time and photon energy
- cumulative spectral radiation loss at Rmax vs time and photon energy

The figures are saved in:

- `outputs/bpf_extra_fields/`

## Scientific Interpretation

The plotted charge-state fractions are dimensionless probabilities. Their
charge-state weighted average reproduces the log's `Mean Chg`, confirming the
charge-state axis ordering as `q = 0, 1, ..., Z`.

The radiation flux plots are frequency-resolved versions of scalar diagnostics
that appear in the log. The log tells how much net radiation flux crosses a
boundary after integration over photon energy; BPF preserves the photon-energy
distribution of that flux.

The radiation loss plots are cumulative spectral fluence-like quantities. Their
integral over photon energy gives the cumulative radiation energy lost from the
grid in `J/cm2`.

## Remaining Unknowns

The BPF still contains additional HydroPLOT records that are not yet exposed by
the extraction scripts. Several likely correspond to opacity, radiation energy
density, heating/cooling components, and other HydroPLOT views, but those names
should remain provisional until validated by `.hna/.hnb` plot NetCDF files or
more independent cross-checks.
