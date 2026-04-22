# Article benchmark report for 50Al+10E+25CH+3.5TW

This report benchmarks all user-visible HELIOS plasmon models against the article-facing aluminium references encoded in the repository.

Global benchmark settings:
- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- probe energy: **8.307 keV**
- electron policy: **benchmark_valence_aware**
- peak-extraction FWHM: **0.20 eV**
- representative-shape FWHM: **3.50 eV**

## 1. State selection

### Cold Al at t = 0

- snapshot/time: **0 / 0.0000 ns**
- zone span: **1-1000**
- selected zones: **1000**
- weighted rho: **2.700 g/cm^3**
- weighted Te: **0.025 eV**
- weighted Ti: **0.025 eV**
- weighted ne: **7.7640e+09 cm^-3**
- weighted Zbar: **1.289e-13**
- Selection keeps only material 1 (Al), which excludes the epoxy and CH layers by construction.
- This matches the ambient cross-check intent rather than the full target stack.

### Driven dense Al slab near the article probe time

- snapshot/time: **630 / 6.3001 ns**
- zone span: **561-973**
- selected zones: **413**
- weighted rho: **4.196 g/cm^3**
- weighted Te: **0.483 eV**
- weighted Ti: **0.489 eV**
- weighted ne: **1.8000e+22 cm^-3**
- weighted Zbar: **0.1918**
- Probe-time target: **6.30 ns**; nearest hydro snapshot is **6.3001 ns**.
- Selection keeps only material 1 (Al) and then clips to the contiguous rho >= 3.75 g/cm^3 slab to remove the laser-facing low-density Al blowoff.
- Epoxy and CH are excluded through the Al-only material filter before the density clip.

## 2. Ambient ranking vs experiment

| model | status | backend | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---:|---:|---:|---:|
| Finite-T Lindhard | valid | finite_t_lindhard | 0.117 | 7 | 3.485 | 3.765 |
| Auto best per state | valid | classical_maxwellian | 0.033 | 7 | 4.502 | 5.365 |
| RPA + static LFC | valid | classical_maxwellian | 0.094 | 7 | 4.502 | 5.365 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | 0.031 | 7 | 5.482 | 6.023 |
| RPA | valid | classical_maxwellian | 0.098 | 7 | 6.480 | 7.505 |
| Quick look | not_applicable | classical_maxwellian | 0.023 | 7 | 6.528 | 7.560 |
| Finite-T Lindhard + Mermin + static LFC | invalid_for_benchmark | finite_t_lindhard | 0.030 | 0 | nan | nan |
| Finite-T Lindhard + Mermin | invalid_for_benchmark | finite_t_lindhard | 0.030 | 0 | nan | nan |
| Mermin + static LFC | invalid_for_benchmark | classical_maxwellian | 0.031 | 0 | nan | nan |
| Mermin | invalid_for_benchmark | classical_maxwellian | 0.034 | 0 | nan | nan |

## 3. Driven ranking vs experiment

| model | status | backend | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---:|---:|---:|---:|
| Auto best per state | valid | classical_maxwellian | 0.030 | 5 | 2.680 | 3.365 |
| RPA + static LFC | valid | classical_maxwellian | 1.852 | 5 | 2.680 | 3.365 |
| RPA | valid | classical_maxwellian | 1.842 | 5 | 4.109 | 5.318 |
| Quick look | not_applicable | classical_maxwellian | 0.022 | 5 | 5.085 | 6.405 |
| Finite-T Lindhard | valid | finite_t_lindhard | 2.703 | 5 | 6.955 | 7.610 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | 0.037 | 5 | 7.525 | 8.012 |
| Finite-T Lindhard + Mermin | invalid_for_benchmark | finite_t_lindhard | 0.025 | 0 | nan | nan |
| Finite-T Lindhard + Mermin + static LFC | invalid_for_benchmark | finite_t_lindhard | 0.025 | 0 | nan | nan |
| Mermin | invalid_for_benchmark | classical_maxwellian | 0.025 | 0 | nan | nan |
| Mermin + static LFC | invalid_for_benchmark | classical_maxwellian | 0.026 | 0 | nan | nan |

## 4. Key observations

- Ambient best-vs-experiment MAE: **Finite-T Lindhard = 3.485 eV**
- Driven best-vs-experiment MAE: **Auto best per state = 2.680 eV**
- Quick look remains a heuristic dispersion estimate: it yields a predicted peak proxy, but benchmark_status remains not_applicable because there is no benchmark-grade spectral fit behind that branch.
- Mermin-style branches that rely on a finite constant collision-rate closure are reported explicitly as invalid_for_benchmark when that closure does not resolve a finite non-negative ν for the selected LOS states.
- Runtime cost is dominated by classical RPA/LFC LOS-integrated spectrum construction on the driven dense slab; the finite-T Lindhard + static LFC branch is much cheaper in the current implementation.

## 5. Generated artifacts

- `outputs\validation_outputs\plasmon_article_cases_smoke_current\all_model_results.json`
- `outputs\validation_outputs\plasmon_article_cases_smoke_current\benchmark_points.csv`
- `outputs\validation_outputs\plasmon_article_cases_smoke_current\benchmark_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_smoke_current\ambient_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_smoke_current\driven_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_smoke_current\ambient_representative_spectra.png`
- `outputs\validation_outputs\plasmon_article_cases_smoke_current\driven_representative_spectra.png`