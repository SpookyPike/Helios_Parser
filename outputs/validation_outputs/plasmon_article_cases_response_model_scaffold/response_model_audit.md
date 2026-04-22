# Plasmon response-model scaffold audit

## 1. Provenance audit

Search scope:
- `src/helios/services/derived/reference_data/plasmon`
- `scripts`
- `_tmp_plasmon_fourth_pass`
- `docs`

Observed candidates:
- `_tmp_plasmon_fourth_pass\work_h3\scripts\validate_plasmon_step8_dispersion.py`
- `_tmp_plasmon_fourth_pass\work_h3\src\helios\services\derived\plasmon_lfc.py`
- `scripts\validate_plasmon_step8_dispersion.py`
- `src\helios\services\derived\reference_data\plasmon\ambient_al_dispersion_figs5.json`
- `src\helios\services\derived\reference_data\plasmon\driven_al_dispersion_article.json`

Judgement:
- No article-native driven calculation tables were found in the repo.
- The only structured article-facing branch inputs are the existing JSON-backed reference series in `src/helios/services/derived/reference_data/plasmon`.
- Validation scripts and archived temporary pass remnants exist, but they do not provide a cleaner native RPA/LFC/TDDFT calculation asset than the current manual-digitization reference layer.

## 2. Mandatory plasmon UI regression recovery note

- Probe energy editability regression root cause: stale result-sync was still allowed to overwrite actively edited plasmon request controls before the edit committed.
- Model-selection regression root cause: compare-all lived only as a boolean toggle; the actual comparison model set was no longer explicitly selectable.
- Study/plot-option regression root cause: dispersion routing no longer preferred width/comparison bundles, so peak/FWHM workflows were effectively hidden behind legacy fallback bundle ordering.
- Layout regression root cause: the plasmon sidebar still had a hard width clamp, which prevented allocating meaningful width to the graph panel.
- UI files changed: `src/helios_analysis/workspace.py`, `src/helios/services/derived/plasmon.py`, `src/helios/services/derived/plasmon_config.py`, `src/helios/services/derived/analysis.py`, `src/helios/services/derived/models.py`, `tests/test_plasmon_phase8.py`, `tests/test_plasmon_ui_phase2.py`.

## 3. Response-model scaffold

- New abstraction: `src/helios/services/derived/plasmon_driven_response.py`.
- Control models implemented:
  - `none`
  - `scalar_increment_control`
- The article-driven scalar policy now routes through the new driven-response abstraction instead of carrying its increment logic only as a special case inside the benchmark/electron-policy path.
- The JSON cold baseline remains the floor; raw HELIOS remains diagnostic-only for synthetic article states.

## 4. Response-evaluation path audit

- The synthetic driven article state is still constructed as an explicit density ensemble, not as a hidden surrogate slab.
- For each density node, the code builds a uniform Al state, evaluates plasmon response for that state, then averages the final spectrum on a common energy grid.
- Peak extraction happens after the spectrum average. The scaffold does not introduce a new pre-response surrogate collapse.

## 5. Control equivalence

- Equivalence rows written: **68**
- Rows within tolerance: **68 / 68**
- The scaffold target is exact reproduction of the frozen scalar-policy baseline before any non-scalar model work is attempted.

## 6. Phase 3 status

- No experimental non-scalar driven-response model was added in this pass.
- Reason: provenance search did not uncover article-native response assets, and the next honest upgrade should be a larger response-level model change rather than another benchmark-side surrogate tweak.

Generated alongside this audit:
- `C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/plasmon_article_cases_response_model_scaffold/response_model_equivalence.csv`
- `C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/plasmon_article_cases_response_model_scaffold/benchmark_summary.csv`
- `C:/Users/dimab/Documents/Helios_parser/outputs/validation_outputs/plasmon_article_cases_response_model_scaffold/policy_state_summary.csv`