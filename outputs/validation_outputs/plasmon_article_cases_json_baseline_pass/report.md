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
- classical response cache stats: **{'finite_t_susceptibility_hits': 471, 'finite_t_susceptibility_misses': 471, 'finite_t_susceptibility_currsize': 471, 'finite_t_susceptibility_maxsize': 512}**

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

| case | electron policy | scope | input status | headline role | rho [g/cm^3] | n_e [cm^-3] | Z_eff from n_e | baseline mode | JSON entry | reason |
|---|---|---|---|---|---:|---:|---:|---|---|---|
| ambient_al_t0 | Raw HELIOS ne/zbar | general_purpose | invalid_input_policy | diagnostic_only | 2.700 | 7.7640e+09 | 1.29e-13 | raw_fields | - | Raw HELIOS electron state is not physically credible for the dense Al benchmark slab: rho=2.700 g/cm^3, ne=7.7640e+09 cm^-3, Z_eff=1.29e-13 (raw ne=7.7640e+09 cm^-3, raw Z_eff=1.29e-13). |
| ambient_al_t0 | Article Al benchmark | benchmark_only | credible | headline_credible | 2.700 | 1.8066e+23 | 3 | cold_baseline_plus_increment | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal | - |
| ambient_al_t0 | Benchmark valence-aware | benchmark_only | credible | headline_credible | 2.700 | 1.8066e+23 | 3 | cold_baseline_plus_increment | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal | - |
| ambient_al_t0 | Valence-locked benchmark | benchmark_only | credible | headline_credible | 2.700 | 1.8066e+23 | 3 | cold_baseline_only | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal | - |
| driven_al_dense_slab | Raw HELIOS ne/zbar | general_purpose | invalid_input_policy | diagnostic_only | 4.196 | 1.8000e+22 | 0.192 | raw_fields | - | Raw HELIOS electron state is not physically credible for the dense Al benchmark slab: rho=4.196 g/cm^3, ne=1.8000e+22 cm^-3, Z_eff=0.192 (raw ne=1.8000e+22 cm^-3, raw Z_eff=0.192). |
| driven_al_dense_slab | Article Al benchmark | benchmark_only | credible | headline_credible | 4.196 | 2.8078e+23 | 3 | cold_baseline_plus_increment | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal | - |
| driven_al_dense_slab | Benchmark valence-aware | benchmark_only | credible | headline_credible | 4.196 | 2.8078e+23 | 3 | cold_baseline_plus_increment | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal | - |
| driven_al_dense_slab | Valence-locked benchmark | benchmark_only | credible | headline_credible | 4.196 | 2.8078e+23 | 3 | cold_baseline_only | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal | - |
## B. Runtime engineering findings

| case | model | baseline mean [s] | current mean [s] | speedup x |
|---|---|---:|---:|---:|
| ambient_al_t0 | Quick look | 0.021 | 0.018 | 1.16 |
| ambient_al_t0 | RPA | 0.094 | 0.097 | 0.96 |
| ambient_al_t0 | Mermin | 0.091 | 0.112 | 0.82 |
| ambient_al_t0 | RPA + static LFC | 0.094 | 0.056 | 1.66 |
| ambient_al_t0 | Mermin + static LFC | 0.106 | 0.037 | 2.87 |
| ambient_al_t0 | Finite-T Lindhard | 0.129 | 0.148 | 0.88 |
| ambient_al_t0 | Finite-T Lindhard + Mermin | 0.121 | 0.129 | 0.94 |
| ambient_al_t0 | Finite-T Lindhard + static LFC | 0.031 | 0.041 | 0.75 |
| ambient_al_t0 | Finite-T Lindhard + Mermin + static LFC | 0.034 | 0.048 | 0.70 |
| ambient_al_t0 | Auto best per state | 0.036 | 0.051 | 0.72 |
| driven_al_dense_slab | Quick look | 0.019 | 0.024 | 0.78 |
| driven_al_dense_slab | RPA | 1.980 | 2.499 | 0.79 |
| driven_al_dense_slab | Mermin | 2.001 | 3.037 | 0.66 |
| driven_al_dense_slab | RPA + static LFC | 1.997 | 0.075 | 26.56 |
| driven_al_dense_slab | Mermin + static LFC | 2.156 | 0.079 | 27.18 |
| driven_al_dense_slab | Finite-T Lindhard | 3.172 | 3.517 | 0.90 |
| driven_al_dense_slab | Finite-T Lindhard + Mermin | 3.280 | 3.464 | 0.95 |
| driven_al_dense_slab | Finite-T Lindhard + static LFC | 3.082 | 3.454 | 0.89 |
| driven_al_dense_slab | Finite-T Lindhard + Mermin + static LFC | 3.181 | 3.644 | 0.87 |
| driven_al_dense_slab | Auto best per state | 0.030 | 0.043 | 0.70 |

## C. Electron-policy sensitivity

| case | model | raw input status | raw MAE [eV] | article MAE [eV] | benchmark-policy spread [eV] | raw-article delta [eV] |
|---|---|---|---:|---:|---:|---:|
| ambient_al_t0 | Auto best per state | invalid_input_policy | nan | 4.503 | 0.000 | nan |
| ambient_al_t0 | Finite-T Lindhard | invalid_input_policy | 11.181 | 3.485 | 0.000 | 7.696 |
| ambient_al_t0 | Finite-T Lindhard + Mermin | invalid_input_policy | 11.181 | 4.647 | 0.000 | 6.534 |
| ambient_al_t0 | Finite-T Lindhard + Mermin + static LFC | invalid_input_policy | nan | 7.215 | 0.000 | nan |
| ambient_al_t0 | Finite-T Lindhard + static LFC | invalid_input_policy | nan | 5.482 | 0.000 | nan |
| ambient_al_t0 | Mermin | invalid_input_policy | nan | 6.482 | 0.000 | nan |
| ambient_al_t0 | Mermin + static LFC | invalid_input_policy | nan | 4.503 | 0.000 | nan |
| ambient_al_t0 | Quick look | invalid_input_policy | 21.626 | 6.528 | 0.000 | 15.098 |
| ambient_al_t0 | RPA | invalid_input_policy | nan | 6.480 | 0.000 | nan |
| ambient_al_t0 | RPA + static LFC | invalid_input_policy | nan | 4.502 | 0.000 | nan |
| driven_al_dense_slab | Auto best per state | invalid_input_policy | nan | 2.730 | 0.000 | nan |
| driven_al_dense_slab | Finite-T Lindhard | invalid_input_policy | 11.003 | 6.955 | 0.000 | 4.048 |
| driven_al_dense_slab | Finite-T Lindhard + Mermin | invalid_input_policy | 10.516 | 5.822 | 0.000 | 4.694 |
| driven_al_dense_slab | Finite-T Lindhard + Mermin + static LFC | invalid_input_policy | nan | 7.852 | 0.000 | nan |
| driven_al_dense_slab | Finite-T Lindhard + static LFC | invalid_input_policy | nan | 7.525 | 0.000 | nan |
| driven_al_dense_slab | Mermin | invalid_input_policy | nan | 4.153 | 0.000 | nan |
| driven_al_dense_slab | Mermin + static LFC | invalid_input_policy | nan | 2.730 | 0.000 | nan |
| driven_al_dense_slab | Quick look | invalid_input_policy | 19.007 | 5.085 | 0.000 | 13.923 |
| driven_al_dense_slab | RPA | invalid_input_policy | nan | 4.109 | 0.000 | nan |
| driven_al_dense_slab | RPA + static LFC | invalid_input_policy | nan | 2.680 | 0.000 | nan |

## D. Primary-policy model ranking

### Ambient vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 0.148 | 7 | 3.485 | 3.765 |
| RPA + static LFC | valid | classical_maxwellian | benchmark_dense | 0.056 | 7 | 4.502 | 5.365 |
| Mermin + static LFC | valid | classical_maxwellian | benchmark_dense | 0.037 | 7 | 4.503 | 5.366 |
| Auto best per state | valid | classical_maxwellian | benchmark_dense | 0.051 | 7 | 4.503 | 5.366 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 0.129 | 7 | 4.647 | 5.348 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.041 | 7 | 5.482 | 6.023 |
| RPA | valid | classical_maxwellian | benchmark_dense | 0.097 | 7 | 6.480 | 7.505 |
| Mermin | valid | classical_maxwellian | benchmark_dense | 0.112 | 7 | 6.482 | 7.506 |
| Quick look | not_applicable | classical_maxwellian | benchmark_dense | 0.018 | 7 | 6.528 | 7.560 |
| Finite-T Lindhard + Mermin + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.048 | 7 | 7.215 | 8.866 |

### Driven vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| RPA + static LFC | valid | classical_maxwellian | benchmark_dense | 0.075 | 5 | 2.680 | 3.365 |
| Auto best per state | valid | classical_maxwellian | benchmark_dense | 0.043 | 5 | 2.730 | 3.430 |
| Mermin + static LFC | valid | classical_maxwellian | benchmark_dense | 0.079 | 5 | 2.730 | 3.430 |
| RPA | valid | classical_maxwellian | benchmark_dense | 2.499 | 5 | 4.109 | 5.318 |
| Mermin | valid | classical_maxwellian | benchmark_dense | 3.037 | 5 | 4.153 | 5.391 |
| Quick look | not_applicable | classical_maxwellian | benchmark_dense | 0.024 | 5 | 5.085 | 6.405 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 3.464 | 5 | 5.822 | 6.117 |
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 3.517 | 5 | 6.955 | 7.610 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | benchmark_dense | 3.454 | 5 | 7.525 | 8.012 |
| Finite-T Lindhard + Mermin + static LFC | valid | finite_t_lindhard | benchmark_dense | 3.644 | 5 | 7.852 | 8.609 |

## E. Simulation-to-simulation reconciliation

| case | model | published branch | comparison kind | status | runtime mean [s] | MAE [eV] | RMSE [eV] | note |
|---|---|---|---|---|---:|---:|---:|---|
| ambient_al_t0 | RPA + static LFC | gawne | direct_family_proxy | valid | 0.056 | 2.439 | 2.808 | Closest published ambient calculation/reference branch in the current reference layer. |
| ambient_al_t0 | Mermin + static LFC | gawne | caveated_family_proxy | valid | 0.037 | 2.440 | 2.809 | Compared to the ambient Gawne branch with an extra collision closure that the reference branch does not include. |
| ambient_al_t0 | Auto best per state | - | no_family_matched_reference | valid | 0.051 | - | - | No dedicated ambient published branch of the same family is present in the current reference layer. |
| ambient_al_t0 | Finite-T Lindhard | - | no_family_matched_reference | valid | 0.148 | - | - | No dedicated ambient published branch of the same family is present in the current reference layer. |
| ambient_al_t0 | Finite-T Lindhard + Mermin | - | no_family_matched_reference | valid | 0.129 | - | - | No dedicated ambient published branch of the same family is present in the current reference layer. |
| ambient_al_t0 | Finite-T Lindhard + Mermin + static LFC | - | no_family_matched_reference | valid | 0.048 | - | - | No dedicated ambient published branch of the same family is present in the current reference layer. |
| ambient_al_t0 | Finite-T Lindhard + static LFC | - | no_family_matched_reference | valid | 0.041 | - | - | No dedicated ambient published branch of the same family is present in the current reference layer. |
| ambient_al_t0 | Mermin | - | no_family_matched_reference | valid | 0.112 | - | - | No dedicated ambient published branch of the same family is present in the current reference layer. |
| ambient_al_t0 | Quick look | - | not_benchmark_grade | not_applicable | 0.018 | - | - | Quick-look and other heuristic branches are not used for article theory reconciliation. |
| ambient_al_t0 | RPA | - | no_family_matched_reference | valid | 0.097 | - | - | No dedicated ambient published branch of the same family is present in the current reference layer. |
| driven_al_dense_slab | RPA + static LFC | lfc | direct_family_match | valid | 0.075 | 5.155 | 5.837 | Direct comparison against the published driven LFC branch. |
| driven_al_dense_slab | RPA | rpa | direct_family_match | valid | 2.499 | 8.977 | 10.259 | Direct comparison against the published driven RPA branch. |
| driven_al_dense_slab | Finite-T Lindhard + Mermin | tddft | conceptual_proxy | valid | 3.464 | 5.685 | 6.099 | Compared to the published TDDFT-like branch as the closest available quantum/finite-T reference, not as an exact family match. |
| driven_al_dense_slab | Finite-T Lindhard | tddft | conceptual_proxy | valid | 3.517 | 6.817 | 7.615 | Compared to the published TDDFT-like branch as the closest available quantum/finite-T reference, not as an exact family match. |
| driven_al_dense_slab | Finite-T Lindhard + static LFC | tddft | conceptual_proxy | valid | 3.454 | 7.387 | 7.995 | Compared to the published TDDFT-like branch as the closest available quantum/finite-T reference, not as an exact family match. |
| driven_al_dense_slab | Finite-T Lindhard + Mermin + static LFC | tddft | conceptual_proxy | valid | 3.644 | 7.714 | 8.604 | Compared to the published TDDFT-like branch as the closest available quantum/finite-T reference, not as an exact family match. |
| driven_al_dense_slab | Mermin + static LFC | lfc | caveated_family_proxy | valid | 0.079 | 5.278 | 5.936 | Compared to the published LFC branch with an additional collision closure caveat. |
| driven_al_dense_slab | Mermin | rpa | caveated_family_proxy | valid | 3.037 | 9.089 | 10.346 | Compared to the published RPA branch because no driven Mermin reference branch is available. |
| driven_al_dense_slab | Auto best per state | - | not_benchmark_grade | valid | 0.043 | - | - | Quick-look and auto-composite branches are not used as direct article theory matches. |
| driven_al_dense_slab | Quick look | - | not_benchmark_grade | not_applicable | 0.024 | - | - | Quick-look and auto-composite branches are not used as direct article theory matches. |

## F. Judged conclusions

- Best benchmark-grade ambient spectral branch: **Finite-T Lindhard** with MAE **3.485 eV**.
- Best benchmark-grade driven spectral branch: **RPA + static LFC** with MAE **2.680 eV**.
- Raw HELIOS ne/zbar is retained only as a diagnostic sensitivity axis here. For the dense Al benchmark slabs it can produce physically absurd electron states and must not be interpreted as a fair model-family benchmark input.
- For the Al-only ambient/driven benchmark slabs, the three benchmark-only electron policies are effectively degenerate; the main electron-policy sensitivity is the jump from raw HELIOS to the benchmark-only Al mappings.
- The root JSON cold-baseline table now drives the benchmark input policy explicitly. For the Al article cases that means elements.Al with cold_Zeff = 3 and driven_default = cold_plus_increment.
- Quick look can have a smaller peak MAE than the full spectral branches, but it remains a heuristic dispersion proxy and is not a benchmark-grade spectral model.
- The benchmark_dense collision closure keeps the Mermin family numerically usable, but user-facing prominence should still be judged by agreement, domain validity, and cost rather than mere run success.
- Ambient reconciliation is only direct for the static-LFC/Gawne-like branch in the current reference layer; there is no dedicated published ambient RPA branch here, so ambient RPA remains experiment-only rather than true branch-to-branch reconciliation.
- Driven reconciliation is now cleanly separated by family: RPA vs published RPA, static-LFC vs published LFC, and Lindhard-family vs TDDFT-like proxy. Remaining disagreement after the input-policy fix is therefore mostly a genuine model/state mismatch, not a broken benchmark input.
- The remaining classical runtime hotspot is still the finite-temperature Maxwellian response path underneath the LOS-integrated classical branches; this pass reduced practical runtime further via response reuse, but did not change the underlying theory approximation.

## G. Generated artifacts

- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\all_model_results.json`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\benchmark_points.csv`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\benchmark_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\policy_state_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\policy_sensitivity_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\reconciliation_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\timing_delta_vs_previous_pass.csv`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\ambient_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\driven_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\ambient_representative_spectra.png`
- `outputs\validation_outputs\plasmon_article_cases_json_baseline_pass\driven_representative_spectra.png`