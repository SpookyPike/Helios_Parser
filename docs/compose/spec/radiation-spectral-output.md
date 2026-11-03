---
feature: radiation-spectral-output
status: in-progress
updated: 2026-11-03
source: feat/radiation-spectral-output
---

# Radiation spectral output

## Report

## [S1] Problem

The Exodus path reads 9 zone fields and drops everything spectral. A HELIOS run
with `Write freq-dependent radiation output = 1` writes 18 further datasets into
the same `.exo`: four node-centred spectral arrays on the `time_freq` axis, five
boundary and interface flux arrays on the `time_whole` axis, and nine integrated
scalars per hydro dump. All of them are discarded, and the resulting HDF5 carries
no marker saying whether radiation output was off or was thrown away.

The requirement document is `PARSER_RADIATION_SPEC.md` in the campaign repository
(`meV/hydro/docs/`). Four use cases drive it: radiation energy in the target (A),
boundary spectra (B), one integrated number per time (C), and a
spectrum-versus-time map (D).

## [S2] Design

### S2.1 Two groups, two time axes

```
radiation/  on time_freq   (n_freq_t rows)   node-centred spectral arrays
flux/       on time_whole  (n_hydro_t rows)  boundary and interface spectra
```

`time_freq` is a subset of `time_whole` by value, never by index. The two groups
stay separate so a consumer cannot read a boundary array against the spectral
axis. `radiation/hydro_index` records, per spectral dump, the `time_whole` row it
corresponds to.

Written schema, with units in dataset attributes as the existing schema does:

```
radiation/time                    (n_freq_t,)              s
radiation/cycle                   (n_freq_t,)
radiation/hydro_index             (n_freq_t,)              index into flux/time
radiation/group_edges             (n_freq_t, n_groups+1)   eV
radiation/group_centre            (n_freq_t, n_groups)     eV, geometric mean
radiation/energy_density          (n_freq_t, n_zones, n_groups)  J/cm3/group
radiation/heating_rate            (n_freq_t, n_zones, n_groups)  J/(g s group)
radiation/cooling_rate            (n_freq_t, n_zones, n_groups)  J/(g s group)
radiation/rosseland_opacity       (n_freq_t, n_zones, n_groups)  cm2/g
radiation/node_axis               (n_zones,)               cm

flux/time                         (n_hydro_t,)             s
flux/cycle                        (n_hydro_t,)
flux/group_edges                  (n_hydro_t, n_groups+1)  eV
flux/incident_rmax                (n_hydro_t, n_groups)    J/(cm2 s eV)
flux/incident_rmin                (n_hydro_t, n_groups)    J/(cm2 s eV)
flux/net_interface                (n_hydro_t, n_bd, n_groups)  J/(cm2 s eV)
flux/region_boundary_node_index   (n_regions,)             0-based node index
flux/power_loss_rmax              (n_hydro_t, n_groups)
flux/power_loss_rmin              (n_hydro_t, n_groups)
flux/power_loss_rmax_integrated   (n_hydro_t, n_groups)
flux/power_loss_rmin_integrated   (n_hydro_t, n_groups)

radiation/integrated/             one row per hydro dump
    total_energy                    (n_hydro_t,)
    loss_rmax                       (n_hydro_t,)
    loss_rmin                       (n_hydro_t,)
    total_flux_rmax                 (n_hydro_t,)          J/(cm2 s)
    total_flux_rmin                 (n_hydro_t,)          J/(cm2 s)
    loss_at_boundaries              (n_hydro_t,)
    loss_at_boundaries_time_integral(n_hydro_t,)
    net_cooling_by_region           (n_hydro_t, n_regions)
    interface_flux                  (n_hydro_t, n_bd)     J/(cm2 s)
    electron_to_radiation_integral  (n_hydro_t,)
```

Units are set where the container dataset name states them. The bookkeeping
scalars carry no unit in their names, so they get `unit = ""` rather than a
guess.

### S2.2 The node dimension is n_zones + 2, and the padding is empty

The four node-centred spectral arrays hold `n_zones + 2` rows while
`zone_boundaries` holds `n_zones + 1`. Measured on all three fixtures, every
dump: rows `0` and `n_zones + 1` are exactly zero, and the interior rows are the
zone centres in Rmin to Rmax order. Two independent identities pin that down:

- `sum over groups of energy_density == rad_energy_density[interior]`, exactly.
- `rad_energy_density[interior] == 3 * rad_pressure`, to 1e-6 relative.

So the reader drops the two padding rows and writes `n_zones` rows with
`node_axis` equal to the zone centres, which makes the spectral arrays broadcast
against every other zone field. A file whose padding rows are not all zero is not
that layout, and the reader raises instead of discarding data.

### S2.3 Behaviour

- Radiation output off, or an `.exo` written without it: `metadata/radiation_output
  = "off"`, no `radiation/` or `flux/` group. Absence stays distinguishable from a
  failed read.
- On: `metadata/radiation_output = "on"` plus `n_groups`, `n_freq_snapshots`,
  `n_boundaries` and `n_regions`, so a reader can size the arrays without opening
  them.
- The spectral axis is not resampled onto `time_whole`. A consumer that wants a
  spectrum at a hydro time interpolates, with its own documented rule.
- `helios_version` stays empty: the container still records no code version.
- Axis mismatches fail loudly. A spectral array whose first dimension is not
  `n_freq_t`, a flux array whose first dimension is not `n_hydro_t`, or a group
  edge count that disagrees with the array width raises `ExodusFormatError`.

### S2.4 Group integration convention

Two per-group families in the same container use opposite normalizations, and the
dataset names label both "per eV". Measured on all three fixtures, every dump:

| family | convention | measured identity | error if the other rule is used |
|---|---|---|---|
| `FdIntgPowerLoss*` | density, weight by `dE` | `sum(FdIntgPowerLossRmin * dE)` = `FreqIntgRadEnLossRmin`, 1e-16 | plain sum off by 1e15 |
| `net_interface` | bin content, plain sum | `sum(net_interface[:, 0, :])` = `-TotalFluxRmin`, exact | `dE` weighting off by 1e4 |

Trapezoid weights on the density family are off by 1.5 % on the Rmin boundary,
where the grid is coarse next to the 10 keV split. `integrate_over_groups` uses
the rectangle rule for that family, and the writer records the convention in a
`normalization` dataset attribute so a consumer does not have to know the
measurement.

The requirement document proposes trapezoid weights and cross-checks against
`TotalFluxRmin`. Neither holds against these containers: the incident flux arrays
are identically zero in all three fixtures, and the array that does reproduce
`TotalFluxRmin` needs a plain sum rather than any integral.

### S2.5 Interface ordering

`net_interface` and `interface_flux` hold `n_bd = n_regions + 1` rows ordered
Rmin to Rmax. Measured: `interface_flux[:, 0] == -TotalFluxRmin` and
`interface_flux[:, -1] == +TotalFluxRmax`, exactly. `n_bd` is 2 for a single-region
target and 4 for a three-region one.

`Indices at region interfaces` holds `n_regions` entries, not `n_bd`: the 1-based
node count at each region's upper boundary. `region_boundary_node_index` stores
them converted to 0-based node indices, so they index `grid/coordinate_edge`
directly.

### S2.6 Integration helpers

In `helios.services.derived.radiation`, exported from `helios.services.derived`:

- `integrate_over_groups(values, group_edges)`: `sum(values * diff(group_edges))`
  over the last axis.
- `spectrum_at(path, time_s, side="rmax"|"rmin", quantity="incident"|"net")`: the
  nearest `flux/` row, with the sample time returned alongside so a caller cannot
  mistake it for the requested time.
- `spectrum_map(path, side=..., quantity=...)`: the `(n_time, n_group)` array plus
  the time and energy axes, which is use case D.

`quantity="net"` reads the net flux at that boundary. It is the informative one:
the incident arrays are identically zero in the reference fixtures, so a figure
built from them is blank.

Each raises `ValueError` when the file carries no radiation output.

`evaluate_radiation(dataset, snapshot_index=...)` builds the bundles the
Radiation tab renders. It reads only `flux/` and the integrated scalars, never the
`(dump, zone, group)` spectral field, so moving the snapshot slider does not
re-read a 26 MB container. The tab gets one profile bundle per flux dump (Rmin and
Rmax in one frame) and two time bundles, `total_flux` and `loss`, which plot the
stored scalars rather than a re-integration of the spectra.

`scripts/plot_radiation.py` renders the same three figures to PNG for a report:
the spectrum at one dump, the spectrum-versus-time map, and one zone's spectral
energy density against time. It is the only way to see the node-centred spectral
field today.

### S2.7 Tests

`tests/test_radiation_output.py`, commit gate, on the 4 MB single-region fixture:

1. `time_freq` and `time_whole` differ in length and are not prefix-equal.
2. The spectral arrays use `n_freq_t` and the flux arrays use `n_hydro_t`; a
   deliberately transposed read raises.
3. `integrate_over_groups` reproduces the stored integrated scalar.
4. `n_bd` is 2 for one region and 4 for three, with the measured ordering
   identity.
5. A run without the flag yields no groups and `radiation_output = "off"`.
6. The spectral arrays are zone-centred: `sum(energy_density) == 3 *
   rad_pressure` at the spectral dumps.

The 26 MB six-spectral-dump fixture is a local-only milestone test (`@slow`,
skipped when absent) covering the spectrum-versus-time map and the axis asymmetry
across 31 hydro dumps.

### S2.8 Fixture manifest

`tests/fixture_manifest.json` gains `radiation_tiny_exodus`,
`radiation_stack_exodus` (both tracked, required by the new test entry) and
`radiation_cu_long_exodus` (local-only, 26 104 140 bytes, not in
`entry_requirements`). The three `.rhc` decks are tracked as provenance and are
declared too. The `.gitignore` beside the fixtures keeps the 26 MB file out of
history.

## [S3] Out of Scope

- The radiation-adjacent zone fields the requirement document lists in §2
  (`rad_energy_density`, `plnk_abs_opac`, `plnk_ems_opac`, `ross_opac`,
  `heating_term`, `cooling_term`, `radsink_term`, `elec_ion_log_lambda`,
  `ion_ion_log_lambda`). They are zone fields, not spectral ones, and mapping
  them is a separate change to `EXODUS_ZONE_FIELDS`.
- Interpolating between two group grids when regrouping moves the edges. The
  per-dump edges are written and the consumer decides.
- Collisional-radiative runs. `num_cr_freq_groups_for_time` is constant in these
  non-CR files and the CR grid is documented to change per time step.
- A Viewer surface for the spectra. The Derived tab covers the boundary spectra
  (use cases B and D). The Viewer's field model is `(time, zone)` and a spectral
  energy density is `(dump, zone, group)`, so showing it there needs a group
  selector and a new plot mode; `scripts/plot_radiation.py` renders it in the
  meantime.
- A side or quantity selector in the Radiation tab. The tab shows both boundaries
  at once, which covers the comparison the campaign asks for; a selector would add
  parameter plumbing for no new information.
- A `.bpf` route. HELIOS 12 writes no plot file when `Plot file type = 1`.

## Tasks

- [ ] T1: fixtures and manifest entries — acceptance: `check_fixtures` passes for the new entry and every declared hash matches (covers: S2.8)
- [ ] T2: read the spectral block in `exodus.py` — acceptance: `read_exodus` on the tiny fixture returns spectral shapes `(1, 100, 250)`, `node_axis` of 100 zone centres, and raises on a transposed array (covers: S2.1, S2.2, S2.3)
- [ ] T3: write `radiation/` and `flux/` groups in `hdf5.py` — acceptance: the converted tiny file has both groups with the schema shapes and units (covers: S2.1)
- [ ] T4: write the integrated group and the on/off metadata — acceptance: converted tiny file reports `radiation_output = "on"` with `n_groups`, `n_freq_snapshots`, `n_boundaries`, `n_regions`; `cu_v12_slice.exo` reports `"off"` with no groups (covers: S2.3)
- [ ] T5: derived helpers — acceptance: `integrate_over_groups` reproduces `radiation/integrated/loss_rmin` from `flux/power_loss_rmin_integrated`; `spectrum_at` returns the sample time (covers: S2.4, S2.6)
- [ ] T6: commit-gate tests — acceptance: `python tests/test_radiation_output.py` passes (covers: S2.7)
- [ ] T7: milestone test on the 26 MB fixture — acceptance: passes locally with the fixture present, skips without it (covers: S2.7)
- [ ] T7b: Radiation tab in Derived / Analysis — acceptance: the tab lists one profile per flux dump with Rmin and Rmax, shows the two time bundles, and reports a reason for a run without radiation (covers: S2.6)
- [ ] T8: docs, changelog, version 1.4.0 — acceptance: `pyproject.toml`, `helios_app/release.py`, README and `docs/packaging.md` agree on 1.4.0 (covers: S2.3)
- [ ] T9: release v1.4.0 — acceptance: tag pushed, GitHub release carries the source ZIP, Debian package, Linux portable tarball and checksums (covers: S2.3)

## Journey log