# Next Heavy Step After QHD and STLS

## Selected Direction

Build a **material-specific / article-native reproduction path for the Al XRTS observable**.

This is the next heavy step because the repo now already has:

- a backend that captures the driven RPA-like branch well (`QHD`), and
- a backend that captures the driven LFC/correlation-sensitive branch well (`finite-T STLS`),

but neither backend becomes the experiment-facing winner.

That means the next leverage is no longer "find another generic free-electron dielectric".

## What Should Be Built Next

Implement a new heavy path that sits above the current free-electron backends and reproduces the **material-specific experimental/theoretical comparison layer** more faithfully.

Target capability:

1. keep `QHD` and `STLS` as explicit free-electron response options;
2. add a material-specific Al XRTS observable model around them;
3. make the comparison to the article-facing experimental observable explicit and reproducible.

## Why This Beats the Alternatives

### Why not qSTLS first

qSTLS is scientifically respectable, but it is not the highest-yield immediate step because:

- static STLS already closes the driven LFC-like calculation branch strongly;
- the common remaining residual is experiment-facing, not merely "the LFC-like branch is still off".

### Why not another generic many-body backend

Another generic many-body route is lower-yield right now because:

- the repo already shows that branch-matching success does not guarantee experiment-facing success;
- without a material-specific target layer, a heavier backend may simply repeat that pattern at larger cost.

### Why not only a hybrid interpretation layer

A hybrid interpretation layer is worth adding eventually, but it is not sufficient as the next heavy step because:

- it explains specialization among current backends,
- but it does not add the missing experiment-facing physics.

## Likely Code Areas

The next pass should primarily affect new modules, not destabilize the validated controls.

Likely additions:

- `src/helios/services/derived/plasmon_material_xrts.py`
- `src/helios/services/derived/plasmon_article_reproduction.py`
- `src/helios/services/derived/plasmon_material_provenance.py`

Likely integration points:

- `src/helios/services/derived/plasmon.py`
- `src/helios/services/derived/models.py`
- `scripts/benchmark_plasmon_article_cases.py`
- a new dedicated article-reproduction benchmark runner

## Required Inputs / Assumptions

The new heavy path should make explicit which of the following are known versus assumed:

- article-side decomposition of the total XRTS observable;
- free-electron versus non-free-electron contributions;
- any material-specific pseudopotential / form-factor assumptions;
- whether the published experiment comparison contains layers not represented by a pure free-electron dielectric alone.

If article-native assets can be recovered, use them directly.
If they cannot, then the new path must still document the material-specific assumptions explicitly instead of hiding them inside a generic backend label.

## Provenance Requirements

The next heavy path must report:

- free-electron backend used (`QHD`, `STLS`, or other control),
- material-specific layer enabled/disabled,
- any article-native table or decomposition asset used,
- whether the result is:
  - free-electron branch only, or
  - full experiment-facing reproduced observable.

## Success Criteria

Success should be judged by all of the following:

1. preserve current control paths exactly:
   - `noop`
   - `scalar_increment_control`
   - `QHD`
   - `finite-T STLS`
2. reduce driven experiment-facing MAE below the current headline winner without destroying branch-level honesty;
3. keep branch-to-branch comparisons explicit:
   - `QHD` for RPA-like,
   - `STLS` for LFC-like,
   - new path for total observable reproduction;
4. make material/article assumptions explicit in provenance.

## Failure Criteria

The next heavy pass should be judged a failure if it:

- simply wraps `QHD` or `STLS` with another surrogate correction;
- hides material-specific assumptions in hard-coded constants;
- improves experiment-facing MAE only by losing branch-level interpretability;
- cannot explain what part of the observable is free-electron response versus higher-level material/article modelling.

## Short Recommendation

Do **not** spend the next heavy pass on another generic dielectric backend.

Build a **material-specific / article-native Al reproduction layer** that uses the now-validated `QHD` and `STLS` backends as explicit free-electron controls and targets the remaining experiment-facing residual directly.
