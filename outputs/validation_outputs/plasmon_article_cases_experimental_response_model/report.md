# Experimental driven-response model comparison

- dataset: **50Al+10E+25CH+3.5TW_stabilized.h5**
- models compared:
  - No driven response correction
  - Scalar driven increment control
  - Electron-column-weighted control (experimental)

## 1. Control preservation

- Scalar control equivalence against the frozen driven-increment baseline: **passed**.
- Equivalence rows written: **40**.

## 2. Ambient behaviour

- Maximum ambient experiment-MAE shift between scalar control and the experimental model: **0.000000 eV**.
- This pass should not materially change the ambient benchmark because the driven-response layer is only active for the article-driven increment policy.

## 3. Driven article-facing behaviour

- Best matched-branch improvement from the experimental model: **RPA + static LFC** (lfc) with delta MAE = **+0.096 eV**.

| case | branch | policy | control matched MAE [eV] | experimental matched MAE [eV] | judgement |
|---|---|---|---:|---:|---|
| driven_al_article_state | Auto best per state -> - | article_al_driven_increment | - | - | marginal change |
| driven_al_article_state | Finite-T Lindhard -> tddft | article_al_driven_increment | 6.666 | 6.754 | marginal change |
| driven_al_article_state | Finite-T Lindhard + Mermin -> tddft | article_al_driven_increment | 7.305 | 7.381 | marginal change |
| driven_al_article_state | Finite-T Lindhard + Mermin + static LFC -> tddft | article_al_driven_increment | 8.052 | 8.129 | marginal change |
| driven_al_article_state | Finite-T Lindhard + static LFC -> tddft | article_al_driven_increment | 7.569 | 7.682 | marginal change |
| driven_al_article_state | Mermin -> rpa | article_al_driven_increment | 8.325 | 8.235 | marginal change |
| driven_al_article_state | Mermin + static LFC -> lfc | article_al_driven_increment | 4.532 | 4.441 | marginal change |
| driven_al_article_state | Quick look -> - | article_al_driven_increment | - | - | not_comparable |
| driven_al_article_state | RPA -> rpa | article_al_driven_increment | 8.283 | 8.197 | marginal change |
| driven_al_article_state | RPA + static LFC -> lfc | article_al_driven_increment | 4.523 | 4.427 | marginal change |

## 4. Judgement

- The first experimental non-scalar model gives only **marginal change**.
- This suggests the current framework still needs a larger response-level upgrade beyond simple ensemble weighting.

## 5. Generated artifacts

- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\report.md`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\response_model_comparison.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\benchmark_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\policy_state_summary.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\control_vs_experimental_delta.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\experimental_model_provenance.csv`
- `outputs\validation_outputs\plasmon_article_cases_experimental_response_model\response_model_equivalence.csv`