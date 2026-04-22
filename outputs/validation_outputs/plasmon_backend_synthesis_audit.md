# Plasmon Backend Synthesis Audit

## Scope

This memo synthesizes the current backend evidence after the repo now contains both:

- a real quantum hydrodynamic backend, and
- a real finite-temperature self-consistent static STLS backend.

No new backend is implemented here. The goal is to classify what the remaining driven-Al experiment-facing residual is actually telling us.

## 1. Evidence Base

Primary artifacts:

- `outputs/validation_outputs/plasmon_new_backend_experiment/report.md`
- `outputs/validation_outputs/plasmon_new_backend_experiment/control_vs_new_backend_delta.csv`
- `outputs/validation_outputs/plasmon_stls_backend_experiment/report.md`
- `outputs/validation_outputs/plasmon_stls_backend_experiment/control_vs_new_backend_delta.csv`
- `outputs/validation_outputs/plasmon_stls_backend_experiment/stls_convergence_summary.csv`
- `outputs/validation_outputs/plasmon_stls_backend_experiment/stls_state_diagnostics.csv`

Fixed facts carried into this synthesis:

- benchmark plumbing is already clean;
- raw HELIOS is diagnostic-only for article-facing synthetic states;
- driven scalar control is preserved;
- weighting/broadening/ensemble-mixing surrogates are exhausted;
- QHD and STLS are both real backend changes, not surrogate wrappers.

## 2. Backend Decomposition

### 2.1 What QHD captures well

Observed strength:

- Driven article state, published RPA-like branch:
  - `RPA -> published RPA`: `8.283 eV`
  - `QHD -> published RPA`: `2.276 eV`

Interpretation:

- QHD captures collective-fluid dispersion physics that the classical Maxwellian `RPA` path misses.
- It improves the structure of `epsilon(q,omega)` in the collective branch, not just a scalar field or width.
- This makes QHD the best current proxy for the driven RPA-like calculation branch.

What this implies physically:

- The driven mismatch was never purely an `n_e` / `Z_eff` problem.
- Collective pressure / recoil / fluid-closure physics matters at the response level.

### 2.2 What STLS captures well

Observed strength:

- Driven article state, published LFC/correlation branch:
  - `RPA + static LFC -> published LFC`: `4.523 eV`
  - `Finite-T STLS -> published LFC`: `0.730 eV`

Ambient cross-check:

- Ambient `Gawne` proxy:
  - `QHD`: `2.351 eV`
  - `STLS`: `2.418 eV`
  - `RPA + static LFC`: `2.439 eV`

Interpretation:

- STLS captures self-consistent static exchange-correlation / local-field physics that the old `static_lfc` surrogate does not.
- It is the first backend in the repo that materially closes the correlation-sensitive branch against the published driven LFC-like calculation.

What this implies physically:

- Missing correlation physics was real.
- The old ESA-style `static_lfc` path was not sufficient as a many-body backend.

### 2.3 Where QHD fails

QHD is not the experiment-facing winner:

- Driven article state, experiment MAE:
  - `QHD`: `6.691 eV`
  - `RPA + static LFC`: `2.288 eV`
  - `STLS`: `3.618 eV`

Interpretation:

- QHD can repair the collective/RPA-like branch while still missing the physics that matters for the experiment-facing observable.
- That means "better collective dielectric" alone is not enough.

### 2.4 Where STLS fails

STLS is not the experiment-facing winner either:

- Driven article state, experiment MAE:
  - `STLS`: `3.618 eV`
  - `RPA + static LFC`: `2.288 eV`

Interpretation:

- Better static many-body correlation is not enough to explain the experiment-facing discrepancy.
- STLS strongly improves the published LFC-like branch, but that does not automatically propagate to best agreement with the experimental reference series.

### 2.5 What remains common to both

Common pattern:

- QHD materially improves the driven RPA-like calculation branch.
- STLS materially improves the driven LFC-like calculation branch.
- Neither becomes the best experiment-facing branch.

This is the most important decomposition result in the repo today.

It means single-backend thinking has already failed in a very specific way:

- the current generic free-electron backends can match distinct published theory branches,
- but the experiment-facing residual persists beyond either branch improvement.

## 3. Residual Classification

### 3.1 What the residual is unlikely to be

Not mainly unresolved state construction:

- state plumbing was already cleaned;
- both QHD and STLS were benchmarked on the same driven article-facing state construction;
- branch-specific gains persisted under that common state definition.

Not mainly simple collision/damping deficiency:

- surrogate broadening and shape modifiers gave only marginal changes;
- the residual after QHD/STLS is multi-eV and structurally branch-dependent.

Not mainly "one more generic UEG backend":

- QHD already covers the collective-fluid side;
- STLS already covers static correlation/local-field side;
- yet experiment-facing ranking still points elsewhere.

### 3.2 What the residual is most consistent with

The remaining experiment-facing residual is most consistent with:

1. **beyond-UEG / material-specific physics**, and
2. **article-side total-observable modelling differences**

rather than another missing generic dielectric correction inside the same UEG-style free-electron layer.

Why:

- STLS already makes the LFC-like published branch close enough that another generic correlation refinement is no longer the highest-leverage explanation.
- QHD already makes the RPA-like published branch close enough that another generic collective-fluid refinement is no longer the highest-leverage explanation.
- The residual that survives both is the gap between "free-electron branch match" and "experiment-facing total observable".

That is more naturally explained by physics such as:

- material-specific electron-ion / pseudopotential effects,
- article-side modelling assumptions beyond homogeneous-UEG response,
- explicit free/bound/elastic decomposition differences,
- or a more article-native reconstruction of how the reported experimental/theoretical comparison was formed.

## 4. Next-Step Class Selection

Selected class: **B. material-specific backend / article-native reproduction path**

Why B beats the alternatives:

### A. dynamic local-field / qSTLS-type extension

Rejected as the immediate next step.

Reason:

- it is physically respectable, but STLS already closes the correlation-sensitive calculation branch strongly;
- qSTLS would most likely refine the same free-electron correlation branch rather than explain why experiment-facing agreement still prefers a simpler surrogate branch.

In other words: qSTLS is still scientifically interesting, but it is no longer the highest-leverage next investment.

### B. material-specific backend / article-native reproduction path

Selected.

Reason:

- it is the best match to the current evidence pattern;
- it directly targets the residual that remains common after both QHD and STLS;
- it can use QHD and STLS as validated free-electron controls rather than discarding them.

### C. hybrid backend interpretation layer

Rejected as the next heavy step.

Reason:

- it is likely useful later for user-facing interpretation,
- but by itself it does not add the missing physics.

It would explain the split between branch winners, but not close the experiment-facing residual.

### D. another heavier many-body route

Rejected for now.

Reason:

- without a material-specific / article-native target layer, this risks repeating the same pattern at higher computational cost:
  better free-electron branch matching without resolving the experiment-facing mismatch.

### E. blocked / insufficient evidence

Rejected.

Reason:

- the evidence is already strong enough to rank the next step.

## 5. Recommendation

The next serious physics step is **not** another generic dielectric backend.

It is a **material-specific / article-native reproduction path for the Al XRTS observable**, where:

- QHD remains the best current RPA-like free-electron control,
- STLS remains the best current correlation/LFC-like free-electron control,
- and the new heavy implementation targets the missing layer between free-electron branch matching and experiment-facing comparison.

This is the strongest, highest-yield next move supported by the current repo evidence.
