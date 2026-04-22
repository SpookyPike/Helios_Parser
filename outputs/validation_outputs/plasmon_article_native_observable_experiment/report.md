# Article-native Al XRTS observable experiment

This pass compares the current minimal XRTS observable against a more article-native Al observable assembly built on the same validated backend responses.

## Summary

- compared modes: `XRTS observable` vs `XRTS article-native Al`
- backend subset: RPA, RPA + static LFC, Quantum hydrodynamic (experimental), Finite-T STLS (experimental)
- mean experiment-facing minimal-minus-article-native MAE delta: `0.000 eV`
- best experiment-facing change: `ambient_al_t0` / `Quantum hydrodynamic (experimental)` = `0.000 eV` (marginal change)
- worst experiment-facing change: `ambient_al_t0` / `Finite-T STLS (experimental)` = `0.000 eV` (marginal change)

## Interpretation

The article-native layer changes the observable construction level, not the backend dielectric.

- free-electron inelastic response still comes from the backend DSF
- the elastic feature is now assembled from Al form-factor bookkeeping rather than the minimal proxy alone
- the comparison peak is taken from the inelastic branch after explicit elastic subtraction

## Article-native Component Breakdown

- `ambient_al_t0` / `Finite-T STLS (experimental)`: free `1.000`, bound `0.000`, elastic `0.000`
- `ambient_al_t0` / `Quantum hydrodynamic (experimental)`: free `1.000`, bound `0.000`, elastic `0.000`
- `ambient_al_t0` / `RPA`: free `1.000`, bound `0.000`, elastic `0.000`
- `ambient_al_t0` / `RPA + static LFC`: free `1.000`, bound `0.000`, elastic `0.000`
- `driven_al_article_state` / `Finite-T STLS (experimental)`: free `0.383`, bound `0.000`, elastic `0.617`
- `driven_al_article_state` / `Quantum hydrodynamic (experimental)`: free `0.394`, bound `0.000`, elastic `0.606`
- `driven_al_article_state` / `RPA`: free `0.463`, bound `0.000`, elastic `0.537`
- `driven_al_article_state` / `RPA + static LFC`: free `0.431`, bound `0.000`, elastic `0.569`

## Article-native Headline Rows

- `ambient_al_t0` / `Finite-T STLS (experimental)`: experiment MAE `2.307 eV`, matched-branch MAE `2.418 eV`, status `valid`
- `ambient_al_t0` / `Quantum hydrodynamic (experimental)`: experiment MAE `3.950 eV`, matched-branch MAE `2.351 eV`, status `valid`
- `ambient_al_t0` / `RPA`: experiment MAE `6.480 eV`, matched-branch MAE `nan eV`, status `valid`
- `ambient_al_t0` / `RPA + static LFC`: experiment MAE `4.502 eV`, matched-branch MAE `2.439 eV`, status `valid`
- `driven_al_article_state` / `Finite-T STLS (experimental)`: experiment MAE `3.640 eV`, matched-branch MAE `0.749 eV`, status `valid`
- `driven_al_article_state` / `Quantum hydrodynamic (experimental)`: experiment MAE `6.709 eV`, matched-branch MAE `2.282 eV`, status `valid`
- `driven_al_article_state` / `RPA`: experiment MAE `3.683 eV`, matched-branch MAE `8.264 eV`, status `valid`
- `driven_al_article_state` / `RPA + static LFC`: experiment MAE `2.277 eV`, matched-branch MAE `4.501 eV`, status `valid`

