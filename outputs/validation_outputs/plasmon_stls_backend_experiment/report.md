# Finite-T STLS backend experiment

- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- models: **RPA, RPA + static LFC, Quantum hydrodynamic, Finite-T STLS**
- driven response control: **Scalar driven increment control**
- backend integrity: **real self-consistent static STLS**
  - collisionless
  - finite-T Lindhard ideal kernel
  - explicit `G(q) -> chi(q,omega) -> S(q) -> G(q)` iteration
  - no dynamic `G(q,omega)` and no qSTLS/VS compressibility enforcement in this first baseline

## 1. STLS convergence

- benchmark-point convergence: **all converged**
- benchmark-point mean iterations: **28.00**
- benchmark-point mean relative residual: **4.470e-04**

| state | rho [g/cm^3] | Te [eV] | Z_eff | converged | iterations | rel residual | G(q_rep) | q_rep/kF |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| ambient_baseline | 2.700 | 0.025 | 3.000 | True | 35 | 4.267e-04 | 0.8174 | 0.7317 |
| driven_density_3.750 | 3.750 | 0.600 | 3.256 | True | 29 | 3.818e-04 | 0.5866 | 0.6382 |
| driven_density_4.000 | 4.000 | 0.600 | 3.317 | True | 28 | 4.855e-04 | 0.5439 | 0.6207 |
| driven_density_4.250 | 4.250 | 0.600 | 3.378 | True | 28 | 4.815e-04 | 0.5042 | 0.6046 |
| driven_density_4.500 | 4.500 | 0.600 | 3.428 | True | 28 | 4.868e-04 | 0.4844 | 0.5903 |

## 2. Headline benchmark comparison

### Ambient experiment-facing MAE

| model | backend | runtime mean [s] | experiment MAE [eV] | matched branch | matched MAE [eV] |
|---|---|---:|---:|---|---:|
| Finite-T STLS (experimental) | finite_t_stls | 0.226 | 2.307 | gawne | 2.418 |
| Quantum hydrodynamic (experimental) | quantum_hydrodynamic | 0.073 | 3.950 | gawne | 2.351 |
| RPA | classical_maxwellian | 0.115 | 6.480 | - | - |
| RPA + static LFC | classical_maxwellian | 0.064 | 4.502 | gawne | 2.439 |

### Driven experiment-facing MAE

| model | backend | runtime mean [s] | experiment MAE [eV] | matched branch | matched MAE [eV] |
|---|---|---:|---:|---|---:|
| Finite-T STLS (experimental) | finite_t_stls | 0.656 | 3.618 | lfc | 0.730 |
| Quantum hydrodynamic (experimental) | quantum_hydrodynamic | 0.005 | 6.691 | rpa | 2.276 |
| RPA | classical_maxwellian | 0.264 | 3.692 | rpa | 8.283 |
| RPA + static LFC | classical_maxwellian | 0.006 | 2.288 | lfc | 4.523 |

## 3. Driven branch-to-branch comparison

### RPA-like branch

| model | published branch | matched MAE [eV] | note |
|---|---|---:|---|
| RPA | rpa | 8.283 | direct_family_match |
| Quantum hydrodynamic (experimental) | rpa | 2.276 | collective_fluid_proxy |

### Correlation-sensitive / LFC branch

| model | published branch | matched MAE [eV] | note |
|---|---|---:|---|
| RPA + static LFC | lfc | 4.523 | direct_family_match |
| Finite-T STLS (experimental) | lfc | 0.730 | correlation_backend_proxy |

## 4. Judged comparison

- Same-reference driven LFC comparison: **Finite-T STLS vs RPA + static LFC = +3.792 eV** (material gain).
- Best driven RPA-like proxy remains **Quantum hydrodynamic (experimental)** at **2.276 eV**.
- Best driven LFC/correlation proxy is **Finite-T STLS (experimental)** at **0.730 eV**.
- Best ambient correlation-sensitive proxy is **Quantum hydrodynamic (experimental)** at **2.351 eV**.
- Strongest experiment-facing STLS delta against an existing control is **+3.073 eV** versus **Quantum hydrodynamic (experimental)** (material gain).

## 5. Final judgement

- **Real STLS implemented and gives material gain.**

## 6. Generated artifacts

- `outputs\validation_outputs\plasmon_stls_backend_experiment\benchmark_summary.csv`
- `outputs\validation_outputs\plasmon_stls_backend_experiment\control_vs_new_backend_delta.csv`
- `outputs\validation_outputs\plasmon_stls_backend_experiment\stls_convergence_summary.csv`
- `outputs\validation_outputs\plasmon_stls_backend_experiment\stls_state_diagnostics.csv`
- `outputs\validation_outputs\plasmon_stls_backend_experiment\experimental_model_provenance.csv`
- `outputs\validation_outputs\plasmon_stls_backend_experiment\all_model_results.json`
- `outputs\validation_outputs\plasmon_stls_backend_experiment\ambient_dataset_overlay.png`
- `outputs\validation_outputs\plasmon_stls_backend_experiment\driven_dataset_overlay.png`