# BPF Deep Forensic Analysis

Date: 2026-05-11

This note records the final evidence pass over BPF records that were still
unknown after the production parser work. The analysis used the available Cu
and Fe sample BPF/LOG/EXO runs, HydroPLOT manual figures, shape behavior,
cross-run consistency, direct LOG comparisons, correlation analysis, and
derived-quantity checks.

Confidence levels:

- `validated`: direct numerical relationship to LOG/EXO or a conservation-like
  identity with stable cross-run behavior.
- `mapped`: strong documentary/manual support and stable sample behavior, but
  no direct LOG/EXO column.
- `partially_characterized`: axis/family/activation behavior is known, but a
  precise physics name or unit is not defensible.
- `unresolved`: shape and storage are known, but evidence is insufficient for
  interpretation.

## Newly Validated Record

| Record | Outcome | Evidence | Production handling |
| ---: | --- | --- | --- |
| 14 | Radiation energy density with boundary padding, `J/cm3` | Interior entries equal `3 * LOG pressure_radiation` in both Cu and Fe. Interior divided by mass density reproduces LOG `radiation_energy` with only LOG text precision residuals. First and last entries are zero boundary padding. | Raw padded vector remains `bpf_record_14`; derived zone fields are `radiation_energy_density_j_cm3`, `radiation_pressure_j_cm3`, and `radiation_energy_j_g`. LOG aliases `pressure_radiation` and `radiation_energy` point to the derived fields. |

## Partially Characterized Records

| Record | Shape | Outcome | Evidence | Confidence |
| ---: | --- | --- | --- | --- |
| 15 | `n_nodes` | Positive node auxiliary following the expanding right-going front | Correlates with positive interface velocity/front position in Cu and Fe. Maxima occur near the expanding outer boundary. No stable unit or LOG/EXO match was found. | partially characterized |
| 18 | `n_nodes` | Node/interface radiation-flux-like auxiliary | Right-boundary values correlate with boundary radiation flux; spatial support follows the radiation/front region. Cross-run scale differs, so a validated flux label would be misleading. | partially characterized |
| 19 | `n_nodes` | Constant node mask/weight-like vector | Exactly one at every node and snapshot in both samples. No manual or LOG/EXO semantic match found. | partially characterized |
| 21-28 | `n_nodes` | Inactive optional node channels | Exactly zero at every node and snapshot in both samples. These are preserved as inactive channels rather than discarded. | partially characterized |
| 35 | scalar | Node-count repeat | Equals `n_nodes` in both samples: 51 for Cu and 501 for Fe. | mapped |
| 36 | 2-vector | Inactive pair | All-zero pair in both samples. | partially characterized |
| 40 | scalar | Inactive scalar | All-zero scalar in both samples. | partially characterized |
| 54 | scalar | Control flag | Constant value 1 in both samples. | partially characterized |
| 55 | scalar | Inactive scalar | All-zero scalar in both samples. | partially characterized |
| 56 | scalar | Inactive integer scalar | All-zero scalar in both samples. | partially characterized |
| 58 | scalar | Control flag | Constant value 1 in both samples. | partially characterized |
| 59 | scalar | Control flag | Constant value 1 in both samples. | partially characterized |
| 60 | scalar | Inactive integer scalar | All-zero scalar in both samples. | partially characterized |
| 61 | scalar | Inactive integer scalar | All-zero scalar in both samples. | partially characterized |
| 63 | scalar | Inactive integer scalar | All-zero scalar in both samples. | partially characterized |
| 65 | scalar | Control flag | Constant value -1 in both samples. | partially characterized |
| 66 | scalar | Control flag | Constant value 1 in both samples. | partially characterized |
| 67 | scalar | Control flag | Constant value 1 in both samples. | partially characterized |

## Still Unresolved

| Record | Shape | Evidence checked | Reason not renamed |
| ---: | --- | --- | --- |
| 44 | `n_zones` | Compared against LOG pressure radiation, compression, electron energy, ion/electron heat capacities, radiation energy, kinetic energy, laser source, and all known BPF zone fields. Correlations were inconsistent across Cu and Fe. | No stable numerical, documentary, or physical identity was found. It remains `bpf_record_44` with status `unknown_bpf_record`. |

## Important Negative Results

- `bpf_record_29` is not LOG `LaserSrc`. Dense Cu and sparse Fe comparisons
  show that LOG `LaserSrc` is an internally integrated cumulative source; the
  raw BPF record is orders of magnitude smaller and remains unknown.
- `bpf_record_44` should not be called radiation pressure, radiation energy,
  electron energy, or a heat capacity. At least one sample can show a high
  correlation with an energy-like variable, but the relationship fails
  cross-run consistency.
- Records 21-28 are not discarded even though they are all zero in current
  samples. They remain stored as stable BPF records because other HELIOS
  configurations may activate them.
