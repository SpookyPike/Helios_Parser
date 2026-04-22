# Plasmon Backend Next Step Plan

Selected next backend:
- **finite-temperature self-consistent VS/STLS dielectric backend**

Scope:
- implementation plan only
- no code in this pass

## 1. New modules

Planned new files:
- `src/helios/services/derived/plasmon_stls.py`
- optionally `src/helios/services/derived/plasmon_stls_cache.py` if cache separation becomes necessary

Planned responsibilities:
- compute finite-T STLS / VS descriptors from `(r_s, Theta, q-grid)`
- solve self-consistent static structure / local-field-correction closure
- provide `G(q)` and a backend-specific `epsilon(q,omega)`
- expose convergence diagnostics and provenance

## 2. Pipeline integration

Planned integration points:
- add one new model key in `plasmon_config.py`
- add backend dispatch in `plasmon.py`
- reuse the existing statewise spectrum path and LOS-integrated clustering path
- keep driven-response models orthogonal to the backend

Conceptual flow:
1. build the current state descriptors `n_e`, `T_e`, `r_s`, `Theta`, `q`
2. call the STLS/VS solver for `G(q)`
3. build
   `chi = chi0 / [1 - v(q) * (1 - G(q)) * chi0]`
4. compute `epsilon`, loss, DSF, peak metrics
5. pass outputs through the existing benchmark/report/export pipeline

## 3. Inputs required

Per state:
- electron density
- electron temperature
- `r_s`
- `Theta`
- scattering wave number `q`
- probe energy / energy grid

Optional later inputs:
- spin polarization
- collision model, only after the collisionless backend is benchmarked

## 4. Outputs and provenance

Required result fields:
- `backend = ft_stls_vs`
- `backend_summary`
- `lfc_closure = stls` or `vs`
- `lfc_dynamic_level = static_self_consistent`
- convergence iteration count
- convergence residual
- `G(q)` representative value at the benchmark `q`
- sum-rule / compressibility diagnostic if available

This provenance must stay separate from:
- electron policy
- driven-response model
- synthetic vs hydro state identity

## 5. Caching and performance expectations

Cache key should include:
- model/backend name
- `r_s`
- `Theta`
- `q-grid`
- any closure parameters

Performance expectations:
- solve `G(q)` once per clustered state or once per unique `(r_s, Theta, q-grid)` bucket
- reuse the existing clustering seam
- do not add benchmark-specific fast paths

## 6. Tests

Minimum required tests:
- backend executes on ambient and driven benchmark states
- `G(q) -> 0` limit recovers the RPA-like branch numerically in a controlled test
- convergence diagnostics are reported
- no synthetic/raw mixing regression
- ambient benchmark remains bounded and explicit
- benchmark provenance clearly distinguishes STLS/VS backend from ESA-style surrogate
- UI model selection exposes the backend cleanly if made user-visible

## 7. Benchmark plan

Benchmark against:
- `rpa`
- `rpa_static_lfc`
- `quantum_hydrodynamic`
- current driven-response controls

Headline comparisons:
- ambient `gawne` proxy
- driven published `RPA`
- driven published `LFC`
- experiment-facing MAE, but only as a secondary check

Primary success criterion:
- beat QHD on the driven correlation-sensitive branch, not just on the RPA-like collective branch

Secondary success criterion:
- improve on the current ESA-style `static_lfc` branch without relying on extra scalar tuning

## 8. Stop conditions

Abort the implementation if either becomes true:
- the solver only reproduces the existing ESA-style surrogate in disguise
- convergence/performance cost is so high that the backend cannot be used within the current cached statewise pipeline

If that happens, the next escalation should be:
- external TDDFT coupling, or
- tabulated many-body dynamic-LFC ingestion
