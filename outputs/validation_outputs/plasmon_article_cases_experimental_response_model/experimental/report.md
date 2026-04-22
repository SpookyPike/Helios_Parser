# Article benchmark report for 50Al+10E+25CH+3.5TW

This pass focuses on the driven Al benchmark. The cold-baseline input problem is already fixed; the remaining work here is driven-state electron response and driven-state identity reconciliation before judging genuine family-to-family mismatch.

Global benchmark settings:
- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- probe energy: **8.307 keV**
- ambient headline policy: **Article Al benchmark (article_al_benchmark)**
- driven headline policy: **Article Al + driven increment (article_al_driven_increment)**
- driven response model: **Electron-column-weighted control (experimental) (electron_column_weighted_control)**
- benchmark policy set: **Raw HELIOS ne/zbar, Article Al benchmark, Article Al + driven increment, Benchmark valence-aware, Valence-locked benchmark**
- collision closure: **benchmark_dense**
- ambient reference provenance: **manual_digitization_v2**
- driven reference provenance: **manual_digitization_v2**
- representative convolution FWHM: **3.50 eV**
- ambient point-extraction FWHM: **0.20 eV**
- classical response cache stats: **{'finite_t_susceptibility_hits': 3417, 'finite_t_susceptibility_misses': 4953, 'finite_t_susceptibility_currsize': 512, 'finite_t_susceptibility_maxsize': 512}**

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
- Built from 4 uniform Al states spanning 3.75-4.50 g/cm^3 at fixed Te = 0.60 eV, then averaged at the spectrum level before peak extraction.
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
| Article Al + driven increment | credible | headline_credible | 3.000 | 0.345 | 3.345 | 3.0833e+23 | electron_column_weighted_control | effective_electron_column | cold_baseline_only | benchmark_driven_increment | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal |
| Benchmark valence-aware | credible | headline_credible | 3.000 | 0.000 | 3.000 | 2.7620e+23 | none | uniform | cold_baseline_plus_increment | raw_positive_increment_only | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal |
| Valence-locked benchmark | credible | headline_credible | 3.000 | 0.000 | 3.000 | 2.7620e+23 | none | uniform | cold_baseline_only | none | Al@elements.Al: cold_Zeff=3.000, driven_default=cold_plus_increment, class=simple_metal |

Raw HELIOS is kept as a diagnostic contrast on the hydro-selected slabs, but it is intentionally excluded from the synthetic article-reconciled state because that state is built from uniform reference states and has no raw hydro charge-state field to preserve.

## C. Policy sensitivity on the article-reconciled driven state

| model | article baseline MAE [eV] | driven increment MAE [eV] | benchmark valence-aware MAE [eV] | valence-locked MAE [eV] | credible-policy spread [eV] |
|---|---:|---:|---:|---:|---:|
| Auto best per state | 2.824 | 2.249 | 2.824 | 2.824 | 0.575 |
| Mermin + static LFC | 2.824 | 2.249 | 2.824 | 2.824 | 0.575 |
| RPA + static LFC | 2.819 | 2.241 | 2.819 | 2.819 | 0.578 |
| RPA | 4.431 | 3.652 | 4.431 | 4.431 | 0.780 |
| Mermin | 4.445 | 3.660 | 4.445 | 4.445 | 0.785 |
| Finite-T Lindhard + Mermin | 6.548 | 7.519 | 6.548 | 6.548 | 0.971 |
| Finite-T Lindhard + Mermin + static LFC | 7.294 | 8.267 | 7.294 | 7.294 | 0.973 |
| Finite-T Lindhard | 5.776 | 6.892 | 5.776 | 5.776 | 1.116 |
| Finite-T Lindhard + static LFC | 6.591 | 7.819 | 6.591 | 6.591 | 1.228 |
| Quick look | nan | nan | nan | nan | nan |

## D. Direct driven branch-to-branch reconciliation

| our branch | published branch | policy | state selection | MAE before [eV] | MAE after [eV] | judged improvement |
|---|---|---|---|---:|---:|---|
| Finite-T Lindhard | tddft | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 6.817 | 6.754 | no material change |
| Finite-T Lindhard + Mermin | tddft | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 5.685 | 7.381 | worse |
| Finite-T Lindhard + Mermin + static LFC | tddft | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 7.714 | 8.129 | worse |
| Finite-T Lindhard + static LFC | tddft | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 7.387 | 7.682 | worse |
| Mermin | rpa | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 9.089 | 8.235 | improved |
| Mermin + static LFC | lfc | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 5.278 | 4.441 | improved |
| RPA | rpa | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 8.977 | 8.197 | improved |
| RPA + static LFC | lfc | Article Al + driven increment | legacy_dense_slab -> article_density_average_fixed_temperature | 5.155 | 4.427 | improved |

### Reconciliation decomposition

| branch | previous hydro slab [eV] | best hydro slab [eV] | article state + old policy [eV] | article state + driven increment [eV] |
|---|---:|---:|---:|---:|
| Finite-T Lindhard -> tddft | 5.862 | 5.761 | 5.638 | 6.754 |
| Finite-T Lindhard + Mermin -> tddft | 6.620 | 6.523 | 6.411 | 7.381 |
| Finite-T Lindhard + Mermin + static LFC -> tddft | 7.365 | 7.269 | 7.156 | 8.129 |
| Finite-T Lindhard + static LFC -> tddft | 6.664 | 6.575 | 6.454 | 7.682 |
| Mermin -> rpa | 9.374 | 9.481 | 9.382 | 8.235 |
| Mermin + static LFC -> lfc | 5.567 | 5.677 | 5.609 | 4.441 |
| RPA -> rpa | 9.351 | 9.460 | 9.369 | 8.197 |
| RPA + static LFC -> lfc | 5.547 | 5.667 | 5.599 | 4.427 |

## E. Headline practical ranking

### Ambient vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 0.194 | 7 | 3.485 | 3.765 |
| RPA + static LFC | valid | classical_maxwellian | benchmark_dense | 0.094 | 7 | 4.502 | 5.365 |
| Auto best per state | valid | classical_maxwellian | benchmark_dense | 0.070 | 7 | 4.503 | 5.366 |
| Mermin + static LFC | valid | classical_maxwellian | benchmark_dense | 0.082 | 7 | 4.503 | 5.366 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 0.176 | 7 | 4.647 | 5.348 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.065 | 7 | 5.482 | 6.023 |
| RPA | valid | classical_maxwellian | benchmark_dense | 0.149 | 7 | 6.480 | 7.505 |
| Mermin | valid | classical_maxwellian | benchmark_dense | 0.147 | 7 | 6.482 | 7.506 |
| Quick look | not_applicable | classical_maxwellian | benchmark_dense | 0.054 | 7 | 6.528 | 7.560 |
| Finite-T Lindhard + Mermin + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.063 | 7 | 7.215 | 8.866 |

### Driven article-reconciled state vs experiment

| model | status | backend | collision | runtime mean [s] | valid points | MAE [eV] | RMSE [eV] |
|---|---|---|---|---:|---:|---:|---:|
| RPA + static LFC | valid | classical_maxwellian | benchmark_dense | 0.012 | 5 | 2.241 | 2.822 |
| Auto best per state | valid | classical_maxwellian | benchmark_dense | 0.007 | 5 | 2.249 | 2.833 |
| Mermin + static LFC | valid | classical_maxwellian | benchmark_dense | 0.014 | 5 | 2.249 | 2.833 |
| RPA | valid | classical_maxwellian | benchmark_dense | 0.346 | 5 | 3.652 | 4.637 |
| Mermin | valid | classical_maxwellian | benchmark_dense | 0.332 | 5 | 3.660 | 4.660 |
| Finite-T Lindhard | valid | finite_t_lindhard | benchmark_dense | 0.468 | 5 | 6.892 | 7.301 |
| Finite-T Lindhard + Mermin | valid | finite_t_lindhard | benchmark_dense | 0.409 | 5 | 7.519 | 8.345 |
| Finite-T Lindhard + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.005 | 5 | 7.819 | 8.179 |
| Finite-T Lindhard + Mermin + static LFC | valid | finite_t_lindhard | benchmark_dense | 0.005 | 5 | 8.267 | 8.883 |
| Quick look | invalid_for_benchmark | classical_maxwellian | benchmark_dense | 0.005 | 0 | nan | nan |

## F. Judged conclusions

- Best practical ambient benchmark branch remains **Finite-T Lindhard** with MAE **3.485 eV**.
- Best practical driven branch on the article-reconciled state is **RPA + static LFC** with MAE **2.241 eV**.
- Raw HELIOS remains diagnostic-only for article-facing Al. It is intentionally visible only as a contrast axis and is not part of the headline ranking.
- The new driven increment policy is explicit and modest. It keeps the JSON cold baseline as the floor, adds a bounded temperature/compression-driven increment for Al, and reports the baseline and increment contributions separately.
- The best hydro plateau is still colder than the article target. Tightening the slab selection helps, but it does not by itself close the driven branch-to-branch gap.
- Rebuilding the driven benchmark around the article density-average state is necessary for fair comparison. It removes a real state-identity mismatch that was previously inflating confusion about the classical branches.
- After the state and policy fixes, the strongest practical classical family match is still **RPA + static LFC**. It improves relative to the previous pass, but it remains materially below the published driven LFC branch.
- The Mermin family remains usable-with-caveats rather than headline-prominent. It now runs under the benchmark_dense closure, but it does not outperform the best classical parent strongly enough to justify primary prominence.
- The remaining driven mismatch is now mostly genuine model disagreement and/or missing driven-state electron-response physics, not benchmark plumbing. The next blocker is therefore a better justified driven electron increment / response model rather than another cache or UI fix.

## G. Generated artifacts

- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\all_model_results.json`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\benchmark_points.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\benchmark_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\policy_state_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\policy_sensitivity_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\reconciliation_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\driven_policy_table.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\driven_state_comparison.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\driven_branch_reconciliation_delta.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\ambient_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\driven_legacy_hydro_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\driven_best_hydro_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\driven_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\ambient_representative_spectra.png`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental\driven_representative_spectra.png`