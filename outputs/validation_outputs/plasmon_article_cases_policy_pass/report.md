# Article benchmark report for 50Al+10E+25CH+3.5TW

This report benchmarks all user-visible HELIOS plasmon models against the article-facing aluminium references encoded in the repository and now treats electron policy as an explicit comparison axis.

Global benchmark settings:
- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- probe energy: **8.307 keV**
- primary electron policy: **Article Al benchmark (article_al_benchmark)**
- benchmark policy set: **Raw HELIOS ne/zbar, Article Al benchmark, Benchmark valence-aware, Valence-locked benchmark**
- collision closure: **benchmark_dense**
- ambient reference provenance: **manual_digitization_v2**
- driven reference provenance: **manual_digitization_v2**
- peak-extraction FWHM: **0.20 eV**
- representative-shape FWHM: **3.50 eV**
- classical response cache stats: **{'finite_t_susceptibility_hits': 0, 'finite_t_susceptibility_misses': 0, 'finite_t_susceptibility_currsize': 0, 'finite_t_susceptibility_maxsize': 512}**

## A. State selection

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
- Primary benchmark preset: al_ambient_article; primary requested electron policy: article_al_benchmark.

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
- Primary benchmark preset: al_driven_article; primary requested electron policy: article_al_benchmark.

### Electron-policy input sanity

| case | electron policy | scope | input status | rho [g/cm^3] | n_e [cm^-3] | Z_eff from n_e | reason |
|---|---|---|---|---:|---:|---:|---|
| ambient_al_t0 | Raw HELIOS ne/zbar | general_purpose | invalid_input_policy | 2.700 | 7.7640e+09 | 1.29e-13 | Raw HELIOS electron state is not physically credible for the dense Al benchmark slab: rho=2.700 g/cm^3, ne=7.7640e+09 cm^-3, Z_eff=1.29e-13 (raw ne=7.7640e+09 cm^-3, raw Z_eff=1.29e-13). |
| ambient_al_t0 | Article Al benchmark | benchmark_only | credible | 2.700 | 1.8066e+23 | 3 | - |
| ambient_al_t0 | Benchmark valence-aware | benchmark_only | credible | 2.700 | 1.8066e+23 | 3 | - |
| ambient_al_t0 | Valence-locked benchmark | benchmark_only | credible | 2.700 | 1.8066e+23 | 3 | - |
| driven_al_dense_slab | Raw HELIOS ne/zbar | general_purpose | invalid_input_policy | 4.196 | 1.8000e+22 | 0.192 | Raw HELIOS electron state is not physically credible for the dense Al benchmark slab: rho=4.196 g/cm^3, ne=1.8000e+22 cm^-3, Z_eff=0.192 (raw ne=1.8000e+22 cm^-3, raw Z_eff=0.192). |
| driven_al_dense_slab | Article Al benchmark | benchmark_only | credible | 4.196 | 2.8078e+23 | 3 | - |
| driven_al_dense_slab | Benchmark valence-aware | benchmark_only | credible | 4.196 | 2.8078e+23 | 3 | - |
| driven_al_dense_slab | Valence-locked benchmark | benchmark_only | credible | 4.196 | 2.8078e+23 | 3 | - |
## B. Runtime engineering findings

| case | model | baseline mean [s] | current mean [s] | speedup x |
|---|---|---:|---:|---:|
| ambient_al_t0 | Quick look | 0.021 | 0.042 | 0.50 |
| ambient_al_t0 | RPA | 0.094 | 0.070 | 1.33 |
| ambient_al_t0 | Mermin | 0.091 | 0.064 | 1.42 |
| ambient_al_t0 | RPA + static LFC | 0.094 | 0.061 | 1.55 |
| ambient_al_t0 | Mermin + static LFC | 0.106 | 0.063 | 1.67 |
| ambient_al_t0 | Finite-T Lindhard | 0.129 | 0.059 | 2.18 |
| ambient_al_t0 | Finite-T Lindhard + Mermin | 0.121 | 0.063 | 1.93 |
| ambient_al_t0 | Finite-T Lindhard + static LFC | 0.031 | 0.061 | 0.50 |
| ambient_al_t0 | Finite-T Lindhard + Mermin + static LFC | 0.034 | 0.052 | 0.66 |
| ambient_al_t0 | Auto best per state | 0.036 | 0.058 | 0.63 |
| driven_al_dense_slab | Quick look | 0.019 | 0.027 | 0.68 |
| driven_al_dense_slab | RPA | 1.980 | 0.034 | 57.96 |
| driven_al_dense_slab | Mermin | 2.001 | 0.032 | 62.29 |
| driven_al_dense_slab | RPA + static LFC | 1.997 | 0.037 | 54.01 |
| driven_al_dense_slab | Mermin + static LFC | 2.156 | 0.034 | 62.84 |
| driven_al_dense_slab | Finite-T Lindhard | 3.172 | 0.033 | 94.72 |
| driven_al_dense_slab | Finite-T Lindhard + Mermin | 3.280 | 0.044 | 75.35 |
| driven_al_dense_slab | Finite-T Lindhard + static LFC | 3.082 | 0.050 | 61.86 |
| driven_al_dense_slab | Finite-T Lindhard + Mermin + static LFC | 3.181 | 0.054 | 58.81 |
| driven_al_dense_slab | Auto best per state | 0.030 | 0.050 | 0.60 |

## C. Electron-policy sensitivity

| case | model | raw input status | raw MAE [eV] | article MAE [eV] | benchmark-policy spread [eV] | raw-article delta [eV] |
|---|---|---|---:|---:|---:|---:|
| ambient_al_t0 | Finite-T Lindhard | invalid_input_policy | 11.181 | 11.181 | 0.000 | 0.000 |
| ambient_al_t0 | Finite-T Lindhard + Mermin | invalid_input_policy | 11.181 | 11.181 | 0.000 | 0.000 |
| ambient_al_t0 | Quick look | invalid_input_policy | 21.626 | 6.528 | 0.000 | 15.098 |
| ambient_al_t0 | Auto best per state | invalid_input_policy | nan | nan | nan | nan |
| ambient_al_t0 | Finite-T Lindhard + Mermin + static LFC | invalid_input_policy | nan | nan | nan | nan |
| ambient_al_t0 | Finite-T Lindhard + static LFC | invalid_input_policy | nan | nan | nan | nan |
| ambient_al_t0 | Mermin | invalid_input_policy | nan | nan | nan | nan |
| ambient_al_t0 | Mermin + static LFC | invalid_input_policy | nan | nan | nan | nan |
| ambient_al_t0 | RPA | invalid_input_policy | nan | nan | nan | nan |
| ambient_al_t0 | RPA + static LFC | invalid_input_policy | nan | nan | nan | nan |
| driven_al_dense_slab | Finite-T Lindhard | invalid_input_policy | 11.003 | 11.003 | 0.000 | 0.000 |
| driven_al_dense_slab | Finite-T Lindhard + Mermin | invalid_input_policy | 10.516 | 10.516 | 0.000 | 0.000 |
| driven_al_dense_slab | Quick look | invalid_input_policy | 19.007 | 5.085 | 0.000 | 13.923 |
| driven_al_dense_slab | Auto best per state | invalid_input_policy | nan | nan | nan | nan |
| driven_al_dense_slab | Finite-T Lindhard + Mermin + static LFC | invalid_input_policy | nan | nan | nan | nan |
| driven_al_dense_slab | Finite-T Lindhard + static LFC | invalid_input_policy | nan | nan | nan | nan |
| driven_al_dense_slab | Mermin | invalid_input_policy | nan | nan | nan | nan |
| driven_al_dense_slab | Mermin + static LFC | invalid_input_policy | nan | nan | nan | nan |
| driven_al_dense_slab | RPA | invalid_input_policy | nan | nan | nan | nan |
| driven_al_dense_slab | RPA + static LFC | invalid_input_policy | nan | nan | nan | nan |

## D. Primary-policy model ranking

### Ambient vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| Quick look | not_applicable | classical_maxwellian | benchmark_dense | 0.042 | 7 | 6.528 | 7.560 |
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 0.059 | 7 | 11.181 | 12.096 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 0.063 | 7 | 11.181 | 12.096 |
| Finite-T Lindhard + Mermin + static LFC | invalid_for_benchmark | finite_t_lindhard | benchmark_dense | 0.052 | 0 | nan | nan |
| Auto best per state | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.058 | 0 | nan | nan |
| RPA + static LFC | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.061 | 0 | nan | nan |
| Finite-T Lindhard + static LFC | invalid_for_benchmark | finite_t_lindhard | benchmark_dense | 0.061 | 0 | nan | nan |
| Mermin + static LFC | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.063 | 0 | nan | nan |
| Mermin | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.064 | 0 | nan | nan |
| RPA | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.070 | 0 | nan | nan |

### Driven vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| Quick look | not_applicable | classical_maxwellian | benchmark_dense | 0.027 | 5 | 5.085 | 6.405 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 0.044 | 5 | 10.516 | 10.855 |
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 0.033 | 5 | 11.003 | 11.303 |
| Mermin | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.032 | 0 | nan | nan |
| RPA | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.034 | 0 | nan | nan |
| Mermin + static LFC | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.034 | 0 | nan | nan |
| RPA + static LFC | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.037 | 0 | nan | nan |
| Finite-T Lindhard + static LFC | invalid_for_benchmark | finite_t_lindhard | benchmark_dense | 0.050 | 0 | nan | nan |
| Auto best per state | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.050 | 0 | nan | nan |
| Finite-T Lindhard + Mermin + static LFC | invalid_for_benchmark | finite_t_lindhard | benchmark_dense | 0.054 | 0 | nan | nan |

## E. Judged conclusions

- Best benchmark-grade ambient spectral branch: **Finite-T Lindhard** with MAE **11.181 eV**.
- Best benchmark-grade driven spectral branch: **Finite-T Lindhard + Mermin** with MAE **10.516 eV**.
- Raw HELIOS ne/zbar is retained only as a diagnostic sensitivity axis here. For the dense Al benchmark slabs it can produce physically absurd electron states and must not be interpreted as a fair model-family benchmark input.
- For the Al-only ambient/driven benchmark slabs, the three benchmark-only electron policies are effectively degenerate; the main electron-policy sensitivity is the jump from raw HELIOS to the benchmark-only Al mappings.
- Quick look can have a smaller peak MAE than the full spectral branches, but it remains a heuristic dispersion proxy and is not a benchmark-grade spectral model.
- The benchmark_dense collision closure keeps the Mermin family numerically usable, but user-facing prominence should still be judged by agreement, domain validity, and cost rather than mere run success.
- The remaining classical runtime hotspot is still the finite-temperature Maxwellian response path underneath the LOS-integrated classical branches; this pass reduced practical runtime further via response reuse, but did not change the underlying theory approximation.

## F. Generated artifacts

- `outputs\validation_outputs\plasmon_article_cases_policy_pass\all_model_results.json`
- `outputs\validation_outputs\plasmon_article_cases_policy_pass\benchmark_points.csv`
- `outputs\validation_outputs\plasmon_article_cases_policy_pass\benchmark_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_policy_pass\policy_state_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_policy_pass\policy_sensitivity_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_policy_pass\timing_delta_vs_previous_pass.csv`
- `outputs\validation_outputs\plasmon_article_cases_policy_pass\ambient_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_policy_pass\driven_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_policy_pass\ambient_representative_spectra.png`
- `outputs\validation_outputs\plasmon_article_cases_policy_pass\driven_representative_spectra.png`