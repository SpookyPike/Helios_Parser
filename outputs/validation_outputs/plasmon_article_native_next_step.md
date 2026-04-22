# Article-native observable next step

Current output directory: `outputs\validation_outputs\plasmon_article_native_observable_experiment`

- mean experiment-facing minimal-minus-article-native MAE delta: `0.000 eV`
- best single experiment-facing improvement: `0.000 eV`
- worst experiment-facing regression: `0.000 eV`

What was added in this pass:
- explicit Al elastic form-factor bookkeeping (neutral/core/screening split)
- explicit elastic subtraction before article-facing peak extraction
- shell-thresholded core bookkeeping tied to the current benchmark window

What still remains missing if the residual stays large:
- nontrivial S_ii(q) / ion-feature physics
- real bound-free Al inelastic scattering, not just shell-threshold bookkeeping
- article-native normalization, subtraction, and detector/background assumptions beyond the current recoverable level

If the residual is still large after this pass, another generic dielectric/backend tweak is no longer justified by the evidence. The next blocker is the missing material-specific atomic/ion observable layer itself.
