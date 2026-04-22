# Higher-level plasmon model jump

- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- models compared:
  - No driven response correction
  - Scalar driven increment control
  - Electron-column-weighted control (experimental)
  - Collision-shape broadening (experimental)
  - Response-function ensemble average (experimental)

## 1. Branch selection

- Selected higher-level branch: **true ensemble-response formulation at the response-function level**.
- The new experimental path averages statewise dielectric response before loss/spectrum extraction.

## 2. Control preservation

- Scalar control equivalence against the frozen driven-increment baseline: **passed**.
- Equivalence rows written: **40**.

## 3. Ambient behaviour

- Maximum ambient experiment-MAE shift between scalar control and the higher-level branch: **0.000000 eV**.

## 4. Driven article-facing behaviour

- Best matched-branch shift from the higher-level branch: **Finite-T Lindhard** (tddft) with delta MAE = **+0.072 eV**.
- Mean classical-family delta vs scalar control: **-0.036 eV**.
- Mean Lindhard-family delta vs scalar control: **+0.030 eV**.

| case | branch | control matched MAE [eV] | weighted matched MAE [eV] | shape matched MAE [eV] | higher-level matched MAE [eV] | judgement |
|---|---|---:|---:|---:|---:|---|
| driven_al_article_state | Auto best per state -> - | - | - | - | - | marginal change |
| driven_al_article_state | Finite-T Lindhard -> tddft | 6.666 | 6.754 | 6.654 | 6.594 | marginal change |
| driven_al_article_state | Finite-T Lindhard + Mermin -> tddft | 7.305 | 7.381 | 7.299 | 7.311 | marginal change |
| driven_al_article_state | Finite-T Lindhard + Mermin + static LFC -> tddft | 8.052 | 8.129 | 8.047 | 8.059 | marginal change |
| driven_al_article_state | Finite-T Lindhard + static LFC -> tddft | 7.569 | 7.682 | 7.562 | 7.508 | marginal change |
| driven_al_article_state | Mermin -> rpa | 8.325 | 8.235 | 8.328 | 8.355 | marginal change |
| driven_al_article_state | Mermin + static LFC -> lfc | 4.532 | 4.441 | 4.536 | 4.556 | marginal change |
| driven_al_article_state | Quick look -> - | - | - | - | - | not_comparable |
| driven_al_article_state | RPA -> rpa | 8.283 | 8.197 | 8.288 | 8.345 | marginal change |
| driven_al_article_state | RPA + static LFC -> lfc | 4.523 | 4.427 | 4.527 | 4.551 | marginal change |

## 5. Judgement

- The higher-level branch gives only **marginal change**.
- The driven mismatch still appears to need a deeper physics upgrade than response-function ensemble mixing alone.

## 6. Generated artifacts

- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_higher_level_model_jump\report.md`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_higher_level_model_jump\benchmark_summary.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_higher_level_model_jump\response_model_comparison.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_higher_level_model_jump\control_vs_new_branch_delta.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_higher_level_model_jump\experimental_model_provenance.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_higher_level_model_jump\response_model_equivalence.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_higher_level_model_jump\response_ensemble_diagnostics.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_higher_level_model_jump\branch_selection_summary.md`
