# Article benchmark report for 50Al+10E+25CH+3.5TW

This report benchmarks all user-visible HELIOS plasmon models against the article-facing aluminium references encoded in the repository.

Global benchmark settings:
- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- probe energy: **8.307 keV**
- electron policy: **Article Al benchmark (article_al_benchmark)**
- collision closure: **benchmark_dense**
- ambient reference provenance: **manual_digitization_v2**
- driven reference provenance: **manual_digitization_v2**
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
- weighted ne: **1.8066e+23 cm^-3**
- weighted Zbar: **3**
- raw vs effective ne: **7.7640e+09 -> 1.8066e+23 cm^-3**
- raw vs effective Zbar: **1.289e-13 -> 3**
- Selection keeps only material 1 (Al), which excludes the epoxy and CH layers by construction.
- This matches the ambient cross-check intent rather than the full target stack.
- Benchmark preset: al_ambient_article; requested electron policy: article_al_benchmark.

### Driven dense Al slab near the article probe time

- snapshot/time: **630 / 6.3001 ns**
- zone span: **561-973**
- selected zones: **413**
- weighted rho: **4.196 g/cm^3**
- weighted Te: **0.483 eV**
- weighted Ti: **0.489 eV**
- weighted ne: **2.8078e+23 cm^-3**
- weighted Zbar: **3**
- raw vs effective ne: **1.8000e+22 -> 2.8078e+23 cm^-3**
- raw vs effective Zbar: **0.1918 -> 3**
- Probe-time target: **6.30 ns**; nearest hydro snapshot is **6.3001 ns**.
- Selection keeps only material 1 (Al) and then clips to the contiguous rho >= 3.75 g/cm^3 slab to remove the laser-facing low-density Al blowoff.
- Epoxy and CH are excluded through the Al-only material filter before the density clip.
- Benchmark preset: al_driven_article; requested electron policy: article_al_benchmark.

## 2. Ambient ranking vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 0.129 | 7 | 3.485 | 3.765 |
| RPA + static LFC | valid | classical_maxwellian | benchmark_dense | 0.094 | 7 | 4.502 | 5.365 |
| Auto best per state | valid | classical_maxwellian | benchmark_dense | 0.036 | 7 | 4.503 | 5.366 |
| Mermin + static LFC | valid | classical_maxwellian | benchmark_dense | 0.106 | 7 | 4.503 | 5.366 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 0.121 | 7 | 4.647 | 5.348 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.031 | 7 | 5.482 | 6.023 |
| RPA | valid | classical_maxwellian | benchmark_dense | 0.094 | 7 | 6.480 | 7.505 |
| Mermin | valid | classical_maxwellian | benchmark_dense | 0.091 | 7 | 6.482 | 7.506 |
| Quick look | not_applicable | classical_maxwellian | benchmark_dense | 0.021 | 7 | 6.528 | 7.560 |
| Finite-T Lindhard + Mermin + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.034 | 7 | 7.215 | 8.866 |

## 3. Driven ranking vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| RPA + static LFC | valid | classical_maxwellian | benchmark_dense | 1.997 | 5 | 2.680 | 3.365 |
| Auto best per state | valid | classical_maxwellian | benchmark_dense | 0.030 | 5 | 2.730 | 3.430 |
| Mermin + static LFC | valid | classical_maxwellian | benchmark_dense | 2.156 | 5 | 2.730 | 3.430 |
| RPA | valid | classical_maxwellian | benchmark_dense | 1.980 | 5 | 4.109 | 5.318 |
| Mermin | valid | classical_maxwellian | benchmark_dense | 2.001 | 5 | 4.153 | 5.391 |
| Quick look | not_applicable | classical_maxwellian | benchmark_dense | 0.019 | 5 | 5.085 | 6.405 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 3.280 | 5 | 5.822 | 6.117 |
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 3.172 | 5 | 6.955 | 7.610 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | benchmark_dense | 3.082 | 5 | 7.525 | 8.012 |
| Finite-T Lindhard + Mermin + static LFC | valid | finite_t_lindhard | benchmark_dense | 3.181 | 5 | 7.852 | 8.609 |

## 4. Key observations

- Ambient best-vs-experiment MAE: **Finite-T Lindhard = 3.485 eV**
- Driven best-vs-experiment MAE: **RPA + static LFC = 2.680 eV**
- Quick look remains a heuristic dispersion estimate: it yields a predicted peak proxy, but benchmark_status remains not_applicable because there is no benchmark-grade spectral fit behind that branch.
- The benchmark_dense collision closure made the Mermin-family numerically executable on both Al benchmark states; those branches now need to be judged by agreement and runtime rather than by trivial invalidation.
- Runtime cost is dominated by classical RPA/LFC LOS-integrated spectrum construction on the driven dense slab; the finite-T Lindhard + static LFC branch is much cheaper in the current implementation.

## 5. Generated artifacts

- `outputs\validation_outputs\plasmon_article_cases_after_science_pass\all_model_results.json`
- `outputs\validation_outputs\plasmon_article_cases_after_science_pass\benchmark_points.csv`
- `outputs\validation_outputs\plasmon_article_cases_after_science_pass\benchmark_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_after_science_pass\ambient_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_after_science_pass\driven_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_after_science_pass\ambient_representative_spectra.png`
- `outputs\validation_outputs\plasmon_article_cases_after_science_pass\driven_representative_spectra.png`