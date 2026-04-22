# Driven Al framework audit

This audit decomposes the remaining driven-Al mismatch after the current benchmark-input and policy fixes.

## Setup decomposition means

| family bucket | hydro-window delta [eV] | synthetic-state delta [eV] | policy-increment delta [eV] | residual gap after setup [eV] |
|---|---:|---:|---:|---:|
| classical_family | 0.111 | -0.082 | -1.074 | 6.416 |
| lindhard_proxy_family | -0.096 | -0.117 | 0.984 | 7.398 |

Interpretation:
- Positive delta means that change worsened MAE; negative delta means it improved MAE.
- For the classical family, hydro-window tuning and synthetic-state construction are both small effects; the policy increment is the only setup change that materially improves agreement.
- For the Lindhard-family proxy branches, the policy increment actually worsens agreement, which means the remaining issue is not simply too-low electron density.

## Collision-closure effect inside the current framework

| parent -> child | published branch | MAE delta [eV] |
|---|---|---:|
| rpa -> mermin | rpa | 0.041 |
| rpa_static_lfc -> mermin_static_lfc | lfc | 0.009 |
| lindhard -> lindhard_mermin | tddft | 0.639 |
| lindhard_static_lfc -> lindhard_mermin_static_lfc | tddft | 0.483 |

## Judgement

- The remaining classical gap is not dominated by hydro slab search anymore; the best-hydro slab is slightly worse than the original driven slab for the direct classical branch matches.
- The synthetic article-state construction is not the main blocker either; it changes the direct classical MAEs only modestly once the cold-baseline policy is held fixed.
- The driven electron increment is real and helpful for the classical family, but it is a scalar Zeff correction. It improves RPA/RPA+static-LFC by about 0.7 eV, then saturates with several eV of residual mismatch still left.
- The Mermin closure is not the top blocker for the classical family: it changes the driven article-state MAE by only a few hundredths of an eV relative to its parent branches.
- The Lindhard-family proxy mismatch is not fixed by the increment and often gets worse, which points to a genuine family/response mismatch rather than benchmark plumbing.

## Path B conclusion

The current framework is effectively exhausted for article-facing driven Al. Further small scalar-electron-policy nudges are not worth developer time.

The smallest major model change that is still worth doing is:

1. Replace the scalar driven-Zeff increment with a response-level driven-electron model that produces state-dependent electron response across the density/temperature ensemble, not just a single adjusted ne/Zeff.
2. Average susceptibility/response over the reconciled driven state distribution with that upgraded electron model, rather than relying on the present low-dimensional Zeff shift inside the same classical/Lindhard closures.
3. Only after that revisit family-to-family disagreement against published RPA/LFC/TDDFT branches.
