# New backend experiment

- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- controls compared:
  - No driven response correction
  - Scalar driven increment control
  - Electron-column-weighted control (experimental)
  - Collision-shape broadening (experimental)
  - Response-function ensemble average (experimental)
- new backend: **Quantum hydrodynamic (experimental)**

## 1. Backend selection

- Selected backend: **quantum hydrodynamic (QHD)**.
- It is structurally different because it replaces the response object with a damped quantum-fluid dielectric rather than altering fields, weights, broadening, or ensemble mixing inside the old backend family.

## 2. Control preservation

- Scalar control equivalence against the frozen driven-increment baseline: **passed**.
- Equivalence rows written: **40**.

## 3. Ambient headline effects

- Maximum ambient delta between the new backend and its scalar-control variant: **0.000000 eV**.

| response model | ambient experiment MAE [eV] | delta vs scalar-control same backend [eV] |
|---|---:|---:|
| Collision-shape broadening (experimental) | 3.950 | +0.000 |
| Electron-column-weighted control (experimental) | 3.950 | +0.000 |
| No driven response correction | 3.950 | +0.000 |
| Response-function ensemble average (experimental) | 3.950 | +0.000 |
| Scalar driven increment control | 3.950 | +0.000 |

## 4. Driven headline effects

| response model | published branch | new backend matched MAE [eV] | delta vs scalar same-backend [eV] | delta vs best same-reference scalar-control branch [eV] | judgement |
|---|---|---:|---:|---:|---|
| Collision-shape broadening (experimental) | rpa | 2.275 | +0.001 | +6.008 | material gain |
| Electron-column-weighted control (experimental) | rpa | 2.301 | -0.025 | +5.982 | material gain |
| No driven response correction | rpa | 2.462 | -0.186 | +5.821 | material gain |
| Response-function ensemble average (experimental) | rpa | 2.271 | +0.006 | +6.013 | material gain |
| Scalar driven increment control | rpa | 2.276 | +0.000 | +6.007 | material gain |

## 5. Judgement

- Best driven new-backend variant: **Response-function ensemble average (experimental)** with matched MAE **2.271 eV** against **rpa**.
- Best same-reference scalar-control comparator remains **RPA** at **8.283 eV**.
- The new backend gives a **defensible material gain** over the strongest same-reference scalar-control branch.

## 6. Generated artifacts

- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\report.md`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\benchmark_summary.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_model_comparison.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\control_vs_new_backend_delta.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\experimental_model_provenance.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\backend_diagnostics.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\response_model_equivalence.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_new_backend_experiment\backend_selection_summary.md`
