# Plasmon science pass technical report

## A. Software and state-flow findings

- Plasmon request/result matching had one remaining UI bug: photon-energy matching was reading mixed unit sources between the global display settings and the plasmon spinbox. That let preset-compatible results look stale even when every physical parameter matched. The workspace now persists a per-spin photon unit and uses it for serialization, matching, and result-sync.
- Electron-policy reporting was globally correct but subset-incorrect. Article-facing benchmark outputs could say that CH/epoxy were kept raw even when the active slab was Al-only. The service now filters electron-policy reporting to the active subset before building warnings and result metadata.
- The existing stability fixes from the previous pass remained valid: no destructive keyboard-tracking recompute, no stale control overwrite, and no hidden spectrum/dispersion cross-build.

## B. Runtime and performance findings

- The dominant driven hotspot remains spectrum construction, not LOS bookkeeping or cache dispatch. The new after-profile for driven RPA shows most cumulative time inside `plasmon_spectrum._plasma_dispersion_function_quadrature`, called through `finite_t_susceptibility -> epsilon_rpa`.
- The earlier quadrature/trapz hotspot improved materially. Baseline driven RPA was about 6.206 s mean; after this pass it is 1.980 s mean. Baseline driven RPA + static LFC was about 6.231 s mean; after this pass it is 1.997 s mean.
- The main remaining bottleneck is still repeated evaluation of the plasma-dispersion quadrature for clustered LOS states in the classical RPA/LFC branches.

## C. Benchmark methodology findings

- Article-facing Al benchmarking now uses the explicit `article_al_benchmark` electron policy rather than silently trusting raw HELIOS `ne/zbar`.
- Ambient case: snapshot 0, Al-only slab, zones 1-1000, weighted rho 2.700 g/cm^3, weighted Te 0.025 eV, effective Zbar locked to 3 for Al.
- Driven case: nearest article probe-time snapshot 630 at 6.3001 ns, Al-only dense slab clipped to rho >= 3.75 g/cm^3, zones 561-973, weighted rho 4.196 g/cm^3, weighted Te 0.483 eV, effective Zbar locked to 3 for Al.
- Reference data now come from explicit JSON files with provenance tags (`manual_digitization_v2`) instead of opaque in-code scaffold dicts.

## D. Model-by-model physical findings

- Ambient best agreement to experiment is Finite-T Lindhard: MAE 3.485 eV.
- Driven best agreement to experiment is RPA + static LFC: MAE 2.680 eV.
- `benchmark_dense` made the Mermin-family numerically executable. Driven Mermin is now valid with MAE 4.153 eV; driven Mermin + static LFC is valid with MAE 2.730 eV.
- That does not make the Mermin-family universally best. In the driven case, Mermin + static LFC is close to the best classical branch, but plain Mermin still trails RPA + static LFC. In the ambient case, Mermin closely tracks its classical parent and does not outperform Finite-T Lindhard.
- The largest remaining source of disagreement is model physics, not state-flow. Ambient disagreement still reflects tension between classical Maxwellian baselines and degenerate-Al behaviour. Driven disagreement still depends strongly on the Al electron mapping and the chosen closure family.

## E. Recommended next improvements

1. Reduce classical RPA/LFC driven cost by caching or tabulating the plasma-dispersion response at the clustered-state level rather than reevaluating the quadrature for each clustered spectrum.
2. Add a benchmark-only user switch for comparing `article_al_benchmark` against `benchmark_valence_aware` and `valence_locked`, so electron-mapping sensitivity is visible instead of implicit.
3. Push the reference layer from `manual_digitization_v2` to higher-quality digitization or extracted tabulated article data when available.
4. Decide whether Mermin-family should stay user-visible in article mode by default. It is now usable, but not uniformly strongest, and its closure assumptions still need explicit caveats.
