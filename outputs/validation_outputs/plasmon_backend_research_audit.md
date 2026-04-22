# Plasmon Backend Research Audit

Date: 2026-03-28

Scope:
- research-first backend selection only
- no core physics code changes
- baseline controls and benchmark plumbing treated as fixed

Repo-grounded starting point:
- current classical/Lindhard family plus response-model surrogates is locally saturated
- `scalar_increment_control` is the only local correction that materially moved the driven classical family
- ensemble weighting, shape broadening, and response-level ensemble mixing changed results only marginally
- the experimental QHD backend gave a real gain for the driven RPA-like comparison, but it did not become the experiment-facing winner
- the current `static_lfc` path in code is explicitly an ESA-style surrogate, not a full published ESA/STLS-quality backend

## 1. Candidate backend families

### A. Finite-T self-consistent local-field-correction backend: STLS / VS / qSTLS family

What it changes:
- replaces `G(q)` from the current lightweight ESA-style surrogate with a self-consistent dielectric-formalism closure
- changes the interacting density response through
  `chi(q,omega) = chi0(q,omega) / [1 - v(q) * (1 - G(q,omega)) * chi0(q,omega)]`
- in the practical first step, this would likely mean a finite-T STLS or Vashishta-Singwi (VS) static/dynamic local-field-correction backend

Physics added:
- exchange-correlation effects beyond RPA
- self-consistency between response and correlations
- compressibility-sum-rule control in the VS family
- an explicit correlation kernel rather than a hand-built `G(q)` asymptote

Why it could fix the mismatch:
- the current driven residual is no longer dominated by state hygiene or scalar `n_e`
- QHD improved the RPA-like branch, which suggests collective-fluid pressure/recoil matters
- QHD did not win the experiment-facing/LFC-like branch, which points to missing exchange-correlation structure rather than missing broadening alone
- a real STLS/VS backend directly targets the correlation physics that separates RPA/QHD from LFC-like and experiment-facing behaviour

Known limitations:
- STLS-family methods remain UEG-based
- qSTLS is not a guaranteed breakthrough: recent work reports only marginal improvement over semi-classical STLS in dynamic properties and identifies a deficiency tied to evaluating the ideal response at noninteracting occupations
- it still will not reproduce full crystal/band-structure physics

Key external sources:
- Tolias et al., *Dynamic properties of the warm dense uniform electron gas with the qSTLS dielectric scheme* (2025): [arXiv:2411.04904](https://arxiv.org/abs/2411.04904)
- Kalkavouras, *Self-consistent dielectric formalism scheme for the paramagnetic electron gas under warm dense matter conditions* (2022): [KTH record](https://kth.diva-portal.org/smash/record.jsf?pid=diva2:1709484)
- Dornheim et al., *The static local field correction of the warm dense electron gas: an ab initio path integral Monte Carlo study* (2019): [arXiv:1907.08473](https://arxiv.org/abs/1907.08473)

### B. LR-TDDFT / Liouville-Lanczos / adiabatic-kernel TDDFT backend

What it changes:
- abandons UEG dielectric closures and computes the interacting density response from a real electronic-structure framework
- response is built from Kohn-Sham orbitals or Liouville-Lanczos propagation plus an XC kernel

Physics added:
- material-specific electronic structure
- finite-temperature XC-kernel effects
- in principle, bound-state and band-structure sensitivity

Why it could fix the mismatch:
- recent Al/WDM literature repeatedly uses TDDFT-class methods as the strongest benchmark or reference branch
- this is the most credible route if the remaining error comes from physics outside UEG-style dielectric closures

Known limitations:
- it is much heavier than the current architecture
- it requires a real DFT engine, pseudopotentials/PAW data, and finite-temperature electronic-structure workflows
- TD-OF-DFT is not a drop-in substitute; published work reports difficulty with bound contributions and plasmon dynamics unless dynamic kinetic-energy physics is added

Key external sources:
- Moldabekov et al., *Applying the Liouville-Lanczos Method of Time-Dependent Density-Functional Theory to Warm Dense Matter* (2025): [arXiv:2502.04921](https://arxiv.org/abs/2502.04921)
- Moldabekov et al., *Linear-response TDDFT approach to warm dense matter with adiabatic exchange-correlation kernels* (2023): [arXiv:2302.04822](https://arxiv.org/abs/2302.04822)
- Baczewski et al., *Dynamical structure factors of warm dense matter from time-dependent orbital-free and mixed TDDFT* (2024): [arXiv:2410.23599](https://arxiv.org/abs/2410.23599)
- Schwalbe et al., *First-principles modeling of plasmons in aluminum under ambient and extreme conditions* (2020): [arXiv:2009.12163](https://arxiv.org/abs/2009.12163)

### C. Ab initio dynamic-local-field-correction / PIMC-driven DSF reconstruction backend

What it changes:
- replaces model local-field closures with a dynamic `G(q,omega)` constrained by ab initio imaginary-time correlation data
- computes `S(q,omega)` by reconstructing the dynamic local-field correction rather than by choosing an analytic surrogate

Physics added:
- nonperturbative exchange-correlation physics
- frequency-dependent local-field correction
- direct connection to exact imaginary-time density correlation data

Why it could fix the mismatch:
- it targets exactly the dynamic correlation content that static-RPA/QHD/STLS-type closures approximate
- if the residual driven gap is mostly dynamic exchange-correlation physics, this family has the right knob

Known limitations:
- it is only honest if backed by tabulated PIMC/ITCF data or a reconstruction workflow
- without those assets, it collapses into another surrogate
- reconstruction and continuation are numerically delicate

Key external sources:
- Groth, Dornheim, Vorberger, *Ab Initio Path Integral Monte Carlo Approach to the Static and Dynamic Density Response of the Uniform Electron Gas* (2019): [arXiv:1903.11970](https://arxiv.org/abs/1903.11970)
- Hou et al., *Exchange-correlation effect in the charge response of a warm dense electron gas* (2022): [Phys. Rev. B 106, L081126](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.L081126)

### D. Average-atom / phase-shift / explicit-collision dielectric backend

What it changes:
- replaces simple collision closures with relaxation rates derived from average-atom electronic structure and transport cross sections
- can be combined with Mermin-style free-electron response or broader XRTS decomposition

Physics added:
- material/state-dependent collision physics
- explicit phase-shift transport cross sections
- better free-bound / free-free consistency than a flat collision prescription

Why it could fix the mismatch:
- if the remaining gap were mostly linewidth/damping or average-atom ionization physics, this backend would be relevant
- it is more physical than a flat constant-`nu` closure

Known limitations:
- the current residual does not look collision-dominated
- current evidence says local broadening changes very little once the main response family is fixed
- a serious average-atom backend would require a new atomic solver layer, not just a new `nu`

Key external sources:
- Johnson and Nilsen, *Average-atom treatment of relaxation time in X-ray Thomson scattering from warm-dense matter* (2016): [arXiv:1512.09155](https://arxiv.org/abs/1512.09155)
- Johnson et al., *Average-Atom Model for X-ray Scattering from Warm Dense Matter* (2012): [arXiv:1211.0227](https://arxiv.org/abs/1211.0227)

## 2. Mapping to the current mismatch

Why scalar correction helped classical branches:
- it moved the plasma frequency and collective energy scale, so some of the driven error clearly came from incorrect effective electron count
- but the gain saturated quickly, which shows that correcting `n_e` alone is not enough

Why weighting, shape modifiers, and response-level ensemble mixing did not help:
- those methods change how precomputed per-state spectra are combined
- their weak impact shows that the main problem is not final averaging
- the problem is the per-state response law itself

Why QHD helped the driven RPA-like comparison:
- QHD adds collective pressure and Bohm-recoil physics directly to the pole structure of the dielectric
- that is enough to move the RPA-like branch much closer to the published RPA-like curve

Why QHD did not become the experiment-facing winner:
- QHD still lacks explicit exchange-correlation kernel physics
- it remains a collective-fluid proxy, not a correlation-aware many-body dielectric
- the experiment-facing branch is better captured by correlation-sensitive models than by a fluid correction alone

Why the Lindhard family underperforms:
- finite-T Lindhard still uses an ideal-electron response as its foundation
- it lacks a strong exchange-correlation kernel
- recent qSTLS analysis further argues that the dielectric-formalism bottleneck is tied to evaluating the ideal response at noninteracting occupations
- in Al, this likely compounds with real-material physics beyond a bare UEG picture

Missing physics most likely responsible for the multi-eV residual:
- dynamic or at least self-consistent exchange-correlation local-field effects
- a more faithful correlation kernel than the current ESA-style surrogate
- only secondarily, improved collision physics
- possibly, beyond-UEG material specificity for the final accuracy frontier, but that points to TDDFT rather than another local correction

Why the RPA-like and LFC/experiment-facing branches split:
- QHD and RPA primarily correct collective dispersion
- the LFC/experiment-facing branch is more sensitive to exchange-correlation physics in `G(q,omega)` or its effective kernel representation
- therefore the next backend should target correlation structure, not another damping or weighting patch

## 3. Feasibility in this repo

Finite-T STLS / VS backend:
- status: implementable with moderate effort
- external data needed: no
- external libraries needed: no, not strictly
- heavy numerics: yes, but still compatible with the current statewise/cache-driven architecture
- architecture fit: strong
- main risk: convergence and performance, not conceptual mismatch

TDDFT / Liouville-Lanczos backend:
- status: blocked for the current repo
- external data needed: not necessarily, but external electronic-structure machinery is required
- external libraries needed: effectively yes
- heavy numerics: very high
- architecture fit: weak unless the repo becomes a wrapper over an external DFT engine

PIMC dynamic-LFC reconstruction backend:
- status: blocked
- external data needed: yes
- external libraries needed: likely yes, or at minimum external tabulated ITCF / `G(q,omega)` data
- heavy numerics: high
- architecture fit: moderate if the data exist, weak if they do not

Average-atom / phase-shift Mermin backend:
- status: implementable only with high effort
- external data needed: no
- external libraries needed: not strictly, but a serious average-atom solver is missing from the repo
- heavy numerics: medium to high
- architecture fit: moderate
- expected leverage on the current mismatch: lower than STLS/VS because current evidence does not point to damping as the dominant residual

## 4. Final selection

Selected backend:
- **finite-temperature self-consistent VS/STLS dielectric backend**

Why this is the best next step:
- it is the highest-leverage backend that is still honestly implementable inside this repo
- it directly targets the missing exchange-correlation physics that QHD does not capture
- it is genuinely different from the current backend family because `G(q)` would be solved from a self-consistent dielectric closure instead of inserted as a handcrafted ESA-style surrogate
- it is less likely than an average-atom collision backend to degenerate into another disguised damping fix
- it avoids the blocking prerequisites of TDDFT and PIMC-dynamic-LFC routes

Why I am not selecting TDDFT next:
- TDDFT is the more ultimate accuracy path, but it is not the next practical backend inside this codebase without importing an external DFT ecosystem

Why I am not selecting PIMC dynamic-LFC next:
- it is physically strong, but not honest without external tabulated many-body data or a new reconstruction layer

Why I am not selecting average-atom collisions next:
- the current evidence says the residual is not primarily collision-limited

## 5. Recommendation

Proceed next with:
- a **collisionless finite-T VS/STLS backend first**
- optional `qSTLS` or collision-coupled extensions only after the static/self-consistent backend is benchmarked

Do not start with:
- another collision closure
- another QHD variant
- another static-LFC surrogate
- direct TDDFT implementation inside this repo

## References

- Bespalov et al., *Experimental validation of electron correlation models in warm dense matter*: [arXiv:2509.10107](https://arxiv.org/abs/2509.10107)
- Moldabekov et al., *Applying the Liouville-Lanczos Method of TDDFT to Warm Dense Matter*: [arXiv:2502.04921](https://arxiv.org/abs/2502.04921)
- Moldabekov et al., *LR-TDDFT with adiabatic XC kernels for WDM*: [arXiv:2302.04822](https://arxiv.org/abs/2302.04822)
- Tolias et al., *Dynamic properties of the warm dense UEG with the qSTLS dielectric scheme*: [arXiv:2411.04904](https://arxiv.org/abs/2411.04904)
- Groth, Dornheim, Vorberger, *Ab Initio PIMC approach to static and dynamic density response*: [arXiv:1903.11970](https://arxiv.org/abs/1903.11970)
- Dornheim et al., *Static local field correction of the warm dense electron gas*: [arXiv:1907.08473](https://arxiv.org/abs/1907.08473)
- Hou et al., *Exchange-correlation effect in the charge response of a warm dense electron gas*: [Phys. Rev. B 106, L081126](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.L081126)
- Johnson and Nilsen, *Average-atom treatment of relaxation time in XRTS*: [arXiv:1512.09155](https://arxiv.org/abs/1512.09155)
