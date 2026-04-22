# XRTS observable next step

Current experiment output directory: `outputs\validation_outputs\plasmon_xrts_observable_experiment`

Interpretation of the first observable-layer pass:
- mean experiment-facing dielectric-minus-observable MAE delta across the primary cases/models: `-0.645 eV`
- best single experiment-facing improvement: `0.010 eV`
- worst experiment-facing regression: `-5.159 eV`

Judgement:
- this minimal observable layer does not close the residual
- ambient remains effectively unchanged
- driven RPA and RPA + static LFC improve only marginally
- driven finite-T STLS becomes substantially worse at the experiment-facing comparison level

What still remains missing if the residual stays large:
- article-native atomic/Chihara assumptions for Al
- material-specific bound/core inelastic term rather than the current explicit-zero narrow-window approximation
- better elastic/ion-feature modeling than a compact central proxy
- possibly article-specific normalization or subtraction conventions if the paper compared a processed observable rather than a raw convolved spectrum

Recommended next step if observable-layer improvement is only partial:
- keep QHD and finite-T STLS as backend controls
- add a material-specific Al XRTS decomposition layer with explicit atomic form factors / ion feature assumptions before attempting another dielectric backend
