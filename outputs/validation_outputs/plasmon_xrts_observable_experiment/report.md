# XRTS observable experiment

This pass compares the existing dielectric-only benchmark path against a material-specific minimal Al XRTS observable reconstruction built on the same backend responses.

## Summary

- compared modes: `Dielectric-only` vs `XRTS observable`
- backend subset: RPA, RPA + static LFC, Quantum hydrodynamic (experimental), Finite-T STLS (experimental)
- mean experiment-facing dielectric-minus-observable MAE delta: `-0.645 eV`
- mean matched-branch dielectric-minus-observable MAE delta: `-0.999 eV`
- best experiment-facing change: `driven_al_article_state` / `RPA + static LFC` = `0.010 eV` (marginal change)
- worst experiment-facing change: `driven_al_article_state` / `Finite-T STLS (experimental)` = `-5.159 eV` (worse)

## Judgement

The first material-specific observable layer does **not** close most of the residual.

- ambient comparisons stay effectively unchanged
- driven RPA and driven RPA + static LFC improve only marginally
- driven finite-T STLS degrades strongly once the observable is reconstructed and the inelastic branch is extracted after elastic-core exclusion
- therefore the remaining gap is not solved by a minimal free+elastic Chihara-like reconstruction alone

## Observable Interpretation

The observable layer does not change the backend dielectric itself. It changes the comparison level by adding explicit free/electric/bound bookkeeping and extracting the plasmon from the convolved inelastic branch instead of the raw backend DSF peak.

## Component Notes

- `ambient_al_t0` / `Finite-T STLS (experimental)`: free `0.652`, bound `0.000`, elastic `0.348`
- `ambient_al_t0` / `Quantum hydrodynamic (experimental)`: free `0.680`, bound `0.000`, elastic `0.320`
- `ambient_al_t0` / `RPA`: free `0.752`, bound `0.000`, elastic `0.248`
- `ambient_al_t0` / `RPA + static LFC`: free `0.735`, bound `0.000`, elastic `0.265`
- `driven_al_article_state` / `Finite-T STLS (experimental)`: free `1.000`, bound `0.000`, elastic `0.000`
- `driven_al_article_state` / `Quantum hydrodynamic (experimental)`: free `1.000`, bound `0.000`, elastic `0.000`
- `driven_al_article_state` / `RPA`: free `1.000`, bound `0.000`, elastic `0.000`
- `driven_al_article_state` / `RPA + static LFC`: free `1.000`, bound `0.000`, elastic `0.000`

## Convolution Sensitivity

- `Quantum hydrodynamic (experimental)` at FWHM `0.2 eV`: peak `0.000 eV`, width `0.269 eV`
- `Quantum hydrodynamic (experimental)` at FWHM `1.0 eV`: peak `0.000 eV`, width `1.016 eV`
- `Quantum hydrodynamic (experimental)` at FWHM `3.5 eV`: peak `27.181 eV`, width `4.591 eV`
- `Quantum hydrodynamic (experimental)` at FWHM `5.0 eV`: peak `27.171 eV`, width `5.847 eV`
- `Finite-T STLS (experimental)` at FWHM `0.2 eV`: peak `0.000 eV`, width `0.270 eV`
- `Finite-T STLS (experimental)` at FWHM `1.0 eV`: peak `0.000 eV`, width `1.018 eV`
- `Finite-T STLS (experimental)` at FWHM `3.5 eV`: peak `26.956 eV`, width `5.110 eV`
- `Finite-T STLS (experimental)` at FWHM `5.0 eV`: peak `26.755 eV`, width `6.645 eV`

