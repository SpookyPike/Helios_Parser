# New plasmon backend selection

Selected backend: **quantum hydrodynamic (QHD)**.

Candidates considered:
- **Quantum hydrodynamic dielectric backend** (selected): changes the response object itself, keeps density-sensitive collective pressure and Bohm recoil explicit, and fits the current benchmark architecture without hidden article-specific hacks.
- **Denser many-body / static-structure backend** (not selected now): would likely need a trustworthy STLS/TDDFT-like implementation or external reference data that are not present in the repo.
- **Article-native calculation ingestion** (blocked): the repo still has digitized reference curves, but not article-native response tables or executable calculation assets.
- **Stronger dense-collision dielectric variant** (rejected now): too close to the current closure family and too likely to become another disguised surrogate instead of a real backend jump.

Why QHD was selected:
- It is genuinely distinct from the existing classical Maxwellian and finite-T Lindhard families.
- It changes the dielectric response itself rather than fields, weights, or final-spectrum post-processing.
- It remains compatible with the current per-state / LOS-integrated / article-benchmark harness.
- It does not require external libraries or article-native tables that the repo does not have.

Physics change introduced:
- The new backend evaluates a damped quantum-fluid dielectric
- `epsilon = 1 - omega_p^2 / (omega*(omega + i nu) - beta_eff^2 q^2 - omega_B^2)`
- with `beta_eff^2 = 3 v_th^2 + 3/5 v_F^2` and Bohm recoil retained explicitly.

Important limitation:
- This is still an experimental collective-fluid proxy, not a recovered article-native backend or a many-body TDDFT/STLS solver.
