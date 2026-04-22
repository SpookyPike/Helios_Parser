# Response-shape experimental model comparison

- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- models compared:
  - No driven response correction
  - Scalar driven increment control
  - Electron-column-weighted control (experimental)
  - Collision-shape broadening (experimental)

## 1. Control preservation

- Scalar control equivalence against the frozen driven-increment baseline: **passed**.
- Equivalence rows written: **40**.

## 2. Ambient behaviour

- Maximum ambient experiment-MAE shift between scalar control and the response-shape model: **0.000000 eV**.

## 3. Driven article-facing behaviour

- Best matched-branch shift from the response-shape model: **Finite-T Lindhard** (tddft) with delta MAE = **+0.012 eV**.
- Mean classical-family delta vs scalar control: **-0.004 eV**.
- Mean Lindhard-family delta vs scalar control: **+0.008 eV**.

| case | branch | control matched MAE [eV] | weighted matched MAE [eV] | shape matched MAE [eV] | judgement |
|---|---|---:|---:|---:|---|
| driven_al_article_state | Auto best per state -> - | - | - | - | marginal change |
| driven_al_article_state | Finite-T Lindhard -> tddft | 6.666 | 6.754 | 6.654 | marginal change |
| driven_al_article_state | Finite-T Lindhard + Mermin -> tddft | 7.305 | 7.381 | 7.299 | marginal change |
| driven_al_article_state | Finite-T Lindhard + Mermin + static LFC -> tddft | 8.052 | 8.129 | 8.047 | marginal change |
| driven_al_article_state | Finite-T Lindhard + static LFC -> tddft | 7.569 | 7.682 | 7.562 | marginal change |
| driven_al_article_state | Mermin -> rpa | 8.325 | 8.235 | 8.328 | marginal change |
| driven_al_article_state | Mermin + static LFC -> lfc | 4.532 | 4.441 | 4.536 | marginal change |
| driven_al_article_state | Quick look -> - | - | - | - | not_comparable |
| driven_al_article_state | RPA -> rpa | 8.283 | 8.197 | 8.288 | marginal change |
| driven_al_article_state | RPA + static LFC -> lfc | 4.523 | 4.427 | 4.527 | marginal change |

## 4. Judgement

- The first response-shape model gives only **marginal change**.
- The residual driven mismatch still points to a larger physics upgrade beyond simple experimental response modifiers.

## 5. Generated artifacts

- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_article_cases_response_shape_experiment\report.md`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_article_cases_response_shape_experiment\response_model_comparison.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_article_cases_response_shape_experiment\control_vs_experimental_delta.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_article_cases_response_shape_experiment\experimental_model_provenance.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_article_cases_response_shape_experiment\benchmark_summary.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_article_cases_response_shape_experiment\policy_state_summary.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_article_cases_response_shape_experiment\response_model_equivalence.csv`
- `C:\Users\dimab\Documents\Helios_parser\outputs\validation_outputs\plasmon_article_cases_response_shape_experiment\shape_modifier_state_diagnostics.csv`