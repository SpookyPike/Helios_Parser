# Article benchmark report for 50Al+10E+25CH+3.5TW

This pass focuses on the driven Al benchmark. The cold-baseline input problem is already fixed; the remaining work here is driven-state electron response and driven-state identity reconciliation before judging genuine family-to-family mismatch.

Global benchmark settings:
- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- probe energy: **8.307 keV**
- ambient headline policy: **Article Al benchmark (article_al_benchmark)**
- driven headline policy: **Article Al + driven increment (article_al_driven_increment)**
- driven response model: **Response-function ensemble average (experimental) (response_function_ensemble_experimental)**
- driven ensemble response mode: **epsilon_average_before_loss**
- benchmark policy set: **Raw HELIOS ne/zbar, Article Al benchmark, Article Al + driven increment, Benchmark valence-aware, Valence-locked benchmark**
- collision closure: **benchmark_dense**
- ambient reference provenance: **manual_digitization_v2**
- driven reference provenance: **manual_digitization_v2**
- representative convolution FWHM: **3.50 eV**
- ambient point-extraction FWHM: **0.20 eV**
- classical response cache stats: **{'finite_t_susceptibility_hits': 5655, 'finite_t_susceptibility_misses': 8295, 'finite_t_susceptibility_currsize': 512, 'finite_t_susceptibility_maxsize': 512}**

## A. State selection and target reconciliation

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
- Selection keeps only Al and excludes epoxy/CH by construction.
- Ambient headline policy: article_al_benchmark.

### Previous driven hydro dense-slab selection

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
- Nearest snapshot to the article probe time 6.30 ns with rho >= 3.75 g/cm^3.
- This is the selection used in the previous driven pass and serves as the baseline for before/after reconciliation.

### Best hydro plateau found near the article state

- snapshot/time: **647 / 6.4700 ns**
- zone span: **525-1000**
- selected zones: **476**
- weighted rho: **4.155 g/cm^3**
- weighted Te: **0.485 eV**
- weighted Ti: **0.491 eV**
- weighted ne: **2.7801e+23 cm^-3**
- weighted Zbar: **3**
- raw vs effective ne: **1.7071e+22 -> 2.7801e+23 cm^-3**
- raw vs effective Zbar: **0.1845 -> 3**
- Search swept nearby snapshots and density floors to minimize mismatch to the article-driven density/temperature window.
- This improves slab identity somewhat, but it is still a hydro state and not the article density-averaged fixed-temperature construction.

### Article-reconciled driven state

- snapshot/time: **-1 / 6.3000 ns**
- zone span: **1-4**
- selected zones: **4**
- weighted rho: **4.125 g/cm^3**
- weighted Te: **0.600 eV**
- weighted Ti: **0.600 eV**
- weighted ne: **3.0833e+23 cm^-3**
- weighted Zbar: **3.345**
- raw vs effective ne: **2.7620e+23 -> 3.0833e+23 cm^-3**
- raw vs effective Zbar: **3 -> 3.345**
- Built from 4 uniform Al states spanning 3.75-4.50 g/cm^3 at fixed Te = 0.60 eV, then combined at the response-function level before loss/spectrum extraction.
- Driven headline policy: article_al_driven_increment.

### Driven state comparison

| selection | kind | snapshot | time [ns] | rho [g/cm^3] | Te [eV] | Ti [eV] | n_i [cm^-3] | n_e [cm^-3] | Z_eff | path [um] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| previous_selected_state | hydro_dense_floor_3p75 | 630 | 6.3001 | 4.196 | 0.483 | 0.489 | 9.3593e+22 | 2.8078e+23 | 3.000 | 27.67 |
| new_selected_state | hydro_plateau_best_match | 647 | 6.4700 | 4.155 | 0.485 | 0.491 | 9.2669e+22 | 2.7801e+23 | 3.000 | 30.00 |
| target_article_state | article_density_average_fixed_temperature | -1 | 6.3000 | 4.125 | 0.600 | 0.600 | 9.2068e+22 | 3.0833e+23 | 3.345 | 30.00 |
## B. Driven electron-policy construction

| policy | status | headline role | cold baseline | driven increment | final Z_eff | effective n_e [cm^-3] | response model | response weighting | baseline mode | increment mode | JSON entry |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---|
| Article Al benchmark | credible | headline_credible | 3.000 | 0.000 | 3.000 | 2.7620e+23 | none | uniform | cold_baseline_plus_increment | raw_positive_increment_only | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal |
| Article Al + driven increment | credible | headline_credible | 3.000 | 0.345 | 3.345 | 3.0833e+23 | response_function_ensemble_experimental | uniform | cold_baseline_only | benchmark_driven_increment | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal |
| Benchmark valence-aware | credible | headline_credible | 3.000 | 0.000 | 3.000 | 2.7620e+23 | none | uniform | cold_baseline_plus_increment | raw_positive_increment_only | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal |
| Valence-locked benchmark | credible | headline_credible | 3.000 | 0.000 | 3.000 | 2.7620e+23 | none | uniform | cold_baseline_only | none | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal |

Raw HELIOS is kept as a diagnostic contrast on the hydro-selected slabs, but it is intentionally excluded from the synthetic article-reconciled state because that state is built from uniform reference states and has no raw hydro charge-state field to preserve.

## C. Policy sensitivity on the article-reconciled driven state

| model | article baseline MAE [eV] | driven increment MAE [eV] | benchmark valence-aware MAE [eV] | valence-locked MAE [eV] | credible-policy spread [eV] |
|---|---:|---:|---:|---:|---:|
| RPA + static LFC | 2.819 | 2.302 | 2.819 | 2.819 | 0.517 |
| Auto best per state | 2.824 | 2.306 | 2.824 | 2.824 | 0.517 |
| Mermin + static LFC | 2.824 | 2.306 | 2.824 | 2.824 | 0.517 |
| RPA | 4.431 | 3.713 | 4.431 | 4.431 | 0.719 |
| Mermin | 4.445 | 3.721 | 4.445 | 4.445 | 0.724 |
| Quantum hydrodynamic (experimental) | 5.800 | 6.687 | 5.800 | 5.800 | 0.886 |
| Finite-T Lindhard + Mermin | 6.548 | 7.448 | 6.548 | 6.548 | 0.900 |
| Finite-T Lindhard + Mermin + static LFC | 7.294 | 8.196 | 7.294 | 7.294 | 0.902 |
| Finite-T Lindhard | 5.776 | 6.732 | 5.776 | 5.776 | 0.956 |
| Finite-T Lindhard + static LFC | 6.591 | 7.646 | 6.591 | 6.591 | 1.055 |
| Quick look | nan | nan | nan | nan | nan |

## D. Direct driven branch-to-branch reconciliation

| our branch | published branch | policy | state selection | MAE before [eV] | MAE after [eV] | judged improvement |
|---|---|---|---|---:|---:|---|
| Finite-T Lindhard | tddft | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 6.817 | 6.594 | no material change |
| Finite-T Lindhard + Mermin | tddft | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 5.685 | 7.311 | worse |
| Finite-T Lindhard + Mermin + static LFC | tddft | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 7.714 | 8.059 | worse |
| Finite-T Lindhard + static LFC | tddft | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 7.387 | 7.508 | no material change |
| Mermin | rpa | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 9.089 | 8.355 | improved |
| Mermin + static LFC | lfc | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 5.278 | 4.556 | improved |
| Quantum hydrodynamic (experimental) | rpa | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | - | 2.271 | not_comparable |
| RPA | rpa | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 8.977 | 8.345 | improved |
| RPA + static LFC | lfc | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 5.155 | 4.551 | improved |

### Reconciliation decomposition

| branch | previous hydro slab [eV] | best hydro slab [eV] | article state + old policy [eV] | article state + driven increment [eV] |
|---|---:|---:|---:|---:|
| Finite-T Lindhard -> tddft | 5.862 | 5.761 | 5.638 | 6.594 |
| Finite-T Lindhard + Mermin -> tddft | 6.620 | 6.523 | 6.411 | 7.311 |
| Finite-T Lindhard + Mermin + static LFC -> tddft | 7.365 | 7.269 | 7.156 | 8.059 |
| Finite-T Lindhard + static LFC -> tddft | 6.664 | 6.575 | 6.454 | 7.508 |
| Mermin -> rpa | 9.374 | 9.481 | 9.382 | 8.355 |
| Mermin + static LFC -> lfc | 5.567 | 5.677 | 5.609 | 4.556 |
| Quantum hydrodynamic (experimental) -> rpa | 2.400 | 2.433 | 2.462 | 2.271 |
| RPA -> rpa | 9.351 | 9.460 | 9.369 | 8.345 |
| RPA + static LFC -> lfc | 5.547 | 5.667 | 5.599 | 4.551 |

## E. Headline practical ranking

### Ambient vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 0.187 | 7 | 3.485 | 3.765 |
| Quantum hydrodynamic (experimental) | valid | quantum_hydrodynamic | benchmark_dense | 0.075 | 7 | 3.950 | 5.865 |
| RPA + static LFC | valid | classical_maxwellian | benchmark_dense | 0.080 | 7 | 4.502 | 5.365 |
| Auto best per state | valid | classical_maxwellian | benchmark_dense | 0.078 | 7 | 4.503 | 5.366 |
| Mermin + static LFC | valid | classical_maxwellian | benchmark_dense | 0.100 | 7 | 4.503 | 5.366 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 0.175 | 7 | 4.647 | 5.348 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.065 | 7 | 5.482 | 6.023 |
| RPA | valid | classical_maxwellian | benchmark_dense | 0.129 | 7 | 6.480 | 7.505 |
| Mermin | valid | classical_maxwellian | benchmark_dense | 0.148 | 7 | 6.482 | 7.506 |
| Quick look | not_applicable | classical_maxwellian | benchmark_dense | 0.050 | 7 | 6.528 | 7.560 |
| Finite-T Lindhard + Mermin + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.076 | 7 | 7.215 | 8.866 |

### Driven article-reconciled state vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| RPA + static LFC | valid | classical_maxwellian | benchmark_dense | 0.006 | 5 | 2.302 | 2.885 |
| Auto best per state | valid | classical_maxwellian | benchmark_dense | 0.006 | 5 | 2.306 | 2.892 |
| Mermin + static LFC | valid | classical_maxwellian | benchmark_dense | 0.008 | 5 | 2.306 | 2.892 |
| RPA | valid | classical_maxwellian | benchmark_dense | 0.234 | 5 | 3.713 | 4.735 |
| Mermin | valid | classical_maxwellian | benchmark_dense | 0.239 | 5 | 3.721 | 4.748 |
| Quantum hydrodynamic (experimental) | valid | quantum_hydrodynamic | benchmark_dense | 0.005 | 5 | 6.687 | 7.780 |
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 0.362 | 5 | 6.732 | 7.136 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 0.354 | 5 | 7.448 | 8.301 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.004 | 5 | 7.646 | 7.997 |
| Finite-T Lindhard + Mermin + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.005 | 5 | 8.196 | 8.833 |
| Quick look | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.002 | 0 | nan | nan |

## F. Judged conclusions

- Best practical ambient benchmark branch remains **Finite-T Lindhard** with MAE **3.485 eV**.
- Best practical driven branch on the article-reconciled state is **RPA + static LFC** with MAE **2.302 eV**.
- Raw HELIOS remains diagnostic-only for article-facing Al. It is intentionally visible only as a contrast axis and is not part of the headline ranking.
- The new driven increment policy is explicit and modest. It keeps the JSON cold baseline as the floor, adds a bounded temperature/compression-driven increment for Al, and reports the baseline and increment contributions separately.
- The best hydro plateau is still colder than the article target. Tightening the slab selection helps, but it does not by itself close the driven branch-to-branch gap.
- Rebuilding the driven benchmark around the article density-average state is necessary for fair comparison. It removes a real state-identity mismatch that was previously inflating confusion about the classical branches.
- After the state and policy fixes, the strongest practical classical family match is still **RPA + static LFC**. It improves relative to the previous pass, but it remains materially below the published driven LFC branch.
- The Mermin family remains usable-with-caveats rather than headline-prominent. It now runs under the benchmark_dense closure, but it does not outperform the best classical parent strongly enough to justify primary prominence.
- The remaining driven mismatch is now mostly genuine model disagreement and/or missing driven-state electron-response physics, not benchmark plumbing. The next blocker is therefore a better justified driven electron increment / response model rather than another cache or UI fix.

## G. Generated artifacts

- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\all_model_results.json`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\benchmark_points.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\benchmark_summary.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\policy_state_summary.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\policy_sensitivity_summary.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\reconciliation_summary.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\driven_policy_table.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\driven_state_comparison.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\driven_branch_reconciliation_delta.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\ambient_dataset_overlay.png`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\driven_legacy_hydro_overlay.png`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\driven_best_hydro_overlay.png`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\driven_dataset_overlay.png`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\ambient_representative_spectra.png`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_function_ensemble_experimental\driven_representative_spectra.png`