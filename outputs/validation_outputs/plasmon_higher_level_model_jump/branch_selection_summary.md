# Higher-level plasmon branch selection

Selected branch: **Branch C — true ensemble-response formulation at the response-function level**.

Why this branch was selected:
- It changes the response object itself, not just scalar fields, weights, or post-hoc broadening.
- It is compatible with the current benchmark harness because state-resolved dielectric arrays already exist in the per-state benchmark results.
- It preserves all current controls exactly and can be run as an explicit experimental comparison path.
- It avoids inventing a fake new backend or another disguised scalar closure tweak.

Why the other branches were not chosen now:
- Branch A, different response backend: no genuinely distinct article-facing backend is recoverable from the repo today; the likely result would be a thin wrapper around the current families.
- Branch B, more explicit finite-density collision/dielectric treatment: the current residual gap is no longer dominated by closure tuning, and another collision-side change would likely become another local surrogate.
- Branch D, article-native ingestion/reproduction: the repo contains digitized reference series, but not article-native response tables or executable article-side calculation assets.

Exact structural change introduced by Branch C:
- The synthetic driven article ensemble can now be combined at the dielectric-response level.
- For the new experimental branch, statewise complex dielectric functions are averaged before computing the loss function and spectrum.
- This is different from the existing control paths, which preserve the current benchmark baseline and average final state spectra.
