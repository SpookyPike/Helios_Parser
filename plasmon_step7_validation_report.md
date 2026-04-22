# Plasmon Step 7 / 0.9.2.12 validation report

## 1. Ambient Al overlays against digitized published curves

Reference source: Gawne et al., Phys. Rev. B 109, L241112 (2024), Supplemental Fig. 1

The overlay below uses the staged service path with a moderate spectral grid so manual finite-T Lindhard branches remain computationally tractable in CI-like validation runs.

| q [A^-1] | model | backend | peak [eV] | ref peak [eV] | peak offset [eV] | RMSE | overlay |
|---:|---|---|---:|---:|---:|---:|---|
| 0.25 | rpa | classical_maxwellian | 15.820 | 16.000 | -0.180 | 0.4828 | `ambient_q0p25.png` |
| 0.25 | rpa_static_lfc | classical_maxwellian | 15.897 | 16.000 | -0.103 | 0.4579 | `ambient_q0p25.png` |
| 0.25 | lindhard | finite_t_lindhard | 30.000 | 16.000 | 14.000 | 0.4669 | `ambient_q0p25.png` |
| 0.25 | lindhard_static_lfc | finite_t_lindhard | 30.000 | 16.000 | 14.000 | 0.4643 | `ambient_q0p25.png` |
| 0.55 | rpa | classical_maxwellian | 15.864 | 16.000 | -0.136 | 0.4618 | `ambient_q0p55.png` |
| 0.55 | rpa_static_lfc | classical_maxwellian | 16.211 | 16.000 | 0.211 | 0.4874 | `ambient_q0p55.png` |
| 0.55 | lindhard | finite_t_lindhard | 19.926 | 16.000 | 3.926 | 0.4965 | `ambient_q0p55.png` |
| 0.55 | lindhard_static_lfc | finite_t_lindhard | 20.401 | 16.000 | 4.401 | 0.4966 | `ambient_q0p55.png` |
| 0.92 | rpa | classical_maxwellian | 15.977 | 16.000 | -0.023 | 0.4371 | `ambient_q0p92.png` |
| 0.92 | rpa_static_lfc | classical_maxwellian | 16.876 | 16.000 | 0.876 | 0.4843 | `ambient_q0p92.png` |
| 0.92 | lindhard | finite_t_lindhard | 20.169 | 16.000 | 4.169 | 0.5097 | `ambient_q0p92.png` |
| 0.92 | lindhard_static_lfc | finite_t_lindhard | 21.006 | 16.000 | 5.006 | 0.5568 | `ambient_q0p92.png` |
| 1.26 | rpa | classical_maxwellian | 16.135 | 17.000 | -0.865 | 0.4852 | `ambient_q1p26.png` |
| 1.26 | rpa_static_lfc | classical_maxwellian | 17.615 | 17.000 | 0.615 | 0.5037 | `ambient_q1p26.png` |
| 1.26 | lindhard | finite_t_lindhard | 23.611 | 17.000 | 6.611 | 0.4999 | `ambient_q1p26.png` |
| 1.26 | lindhard_static_lfc | finite_t_lindhard | 24.451 | 17.000 | 7.451 | 0.5037 | `ambient_q1p26.png` |

### Ambient overlay summary

| model | mean RMSE | mean peak offset [eV] |
|---|---:|---:|
| rpa | 0.4667 | -0.301 |
| rpa_static_lfc | 0.4833 | 0.400 |
| lindhard | 0.4932 | 7.177 |
| lindhard_static_lfc | 0.5054 | 7.715 |

## 2. Ambient vs compressed Al peak-shift trend

Compressed state uses rho=3.5 g/cm^3, Te=0.3 eV as in the compressed-Al literature context. The table checks whether the service preserves the expected higher plasmon energy for the denser state.

| q [A^-1] | model | ambient peak [eV] | compressed peak [eV] | compressed-ambient [eV] |
|---:|---|---:|---:|---:|
| 0.25 | rpa | 15.820 | 17.999 | 2.179 |
| 0.25 | lindhard | 30.000 | 2.323 | -27.677 |
| 0.55 | rpa | 15.864 | 18.054 | 2.190 |
| 0.55 | lindhard | 19.926 | 24.330 | 4.404 |
| 0.92 | rpa | 15.977 | 18.150 | 2.173 |
| 0.92 | lindhard | 20.169 | 23.026 | 2.856 |
| 1.26 | rpa | 16.135 | 18.298 | 2.163 |
| 1.26 | lindhard | 23.611 | 26.023 | 2.413 |

## 3. Warm Al linewidth / damping trend

Warm state uses rho=2.7 g/cm^3, Te=6 eV. This table checks whether linewidth broadening grows toward larger q where Landau/continuum damping should become more important.

| q [A^-1] | model | peak [eV] | FWHM [eV] |
|---:|---|---:|---:|
| 0.25 | rpa | 16.091 | 0.159 |
| 0.25 | lindhard | 16.178 | 0.200 |
| 0.55 | rpa | 17.254 | 0.229 |
| 0.55 | lindhard | 17.763 | 0.222 |
| 0.92 | rpa | 20.083 | 2.144 |
| 0.92 | lindhard | 21.273 | 0.220 |
| 1.26 | rpa | 22.871 | 6.696 |
| 1.26 | lindhard | 23.993 | 0.155 |

## 4. Real-hydrodynamic LOS semantics and zone filtering

| dataset | auto-best quicklook backend | auto summary | max|LOS-effective| | max|filtered-full LOS| | benchmark status @ small angle | benchmark status @ 20 deg |
|---|---|---|---:|---:|---|---|
| Cu_0166_stabilized.h5 | mixed | mermin:1, rpa:19 | 0.944025 | 0.035333 | invalid_for_benchmark | invalid_for_benchmark |
| Cu1e17_cyl_stabilized.h5 | mixed | mermin:3, rpa:77 | 0.998754 | 0.028386 | invalid_for_benchmark | invalid_for_benchmark |

## 5. Snapshot refresh, lazy service reuse, and cache buckets

- Snapshot refresh check: base snapshot=20, refreshed snapshot=40, reused shock object=True, profile delta=26.800000.
- Lindhard cache bucket check: identical repeated request produced time_series_hits=1, time_series_misses=1, spectra_equal=True.
- Auto-best mixed LOS check: backend=mixed, auto summary=`mermin:1, mermin_static_lfc:1`.

## 6. Interpretation

- Manual finite-T Lindhard branches are now wired into the same service/UI/cache stack as the older models, but the ambient-Al overlay shows they are not yet closer to the digitized low-q reference curves than the older staged RPA baseline.
- Auto Best therefore stays conservative and uses the strongest validated local classical branch per state/cluster instead of silently auto-promoting the still-experimental Lindhard backend.
- Real-hydrodynamic checks still show that LOS spectra do not collapse to one pre-averaged state and that left-panel zone exclusion changes the integrated spectrum.