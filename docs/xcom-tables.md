# Cold-XCOM attenuation tables

HELIOS 1.2.1 replaces the coarse 500 eV fallback grid with adaptive, edge-aware
tables over **1–12 keV**. The tables describe neutral-atom photon interactions;
they are a cold reference rather than a warm-dense-matter opacity model.

## What changed

- Calculations use binary64 arithmetic and coefficients are serialized with up to
  17 significant digits. Upstream atomic data retain their original accuracy.
- Every absorption edge retains two rows at the same energy: the below-edge
  limit followed by the above-edge limit. Log-log interpolation uses the limits
  on the same side of the edge. An exact-edge query returns the first,
  below-edge row, preserving the existing API convention.
- Adaptive sampling refines smooth intervals until native checks at geometric
  quarter, midpoint, and three-quarter points satisfy a 0.01% refinement target.
  Independent validation uses 200 seeded holdouts per base material plus probes
  0.01 eV below and above every edge. The release requires sampled relative
  interpolation error below 0.02%; the regenerated tables measured below 0.01%.
- Extrapolation is disabled by default. Non-finite or non-positive query
  energies are rejected. An explicit legacy extrapolation option remains
  available, but extrapolated values are outside the validated table domain.
- The immutable table fingerprint is cached per backend instance to avoid
  reading and hashing the larger resource during every calculation.

The numerical tolerance is measured against the bundled XCOM calculations. It
does not establish physical accuracy, resolve near-edge chemical/solid-state
structure, or validate the application’s heuristic cold/plasma classification.

## Materials and quantities

The 12 base entries are Al, Cu, Fe, Si, Ti, Au, Be, C, CH, C2H4O, C22H10N2O5,
and SiO2. The glass alias shares SiO2 values. C2H4O is an assumed epoxy surrogate;
actual epoxy or glass composition must be supplied for quantitative comparisons.
The generated data contain 1,919 rows including the glass alias.

Each row stores mass coefficients in cm²/g for coherent scattering, incoherent
scattering, photoelectric absorption, nuclear-field and electron-field pair
production, total attenuation, and total attenuation without coherent scattering.
Pair-production coefficients are zero throughout this 1–12 keV table.

For a multilayer line of sight,

```text
areal_density_material = sum_zones(density_g_cm3 * path_length_cm)
tau(E) = sum_materials(mu_over_rho_material(E) * areal_density_material)
transmission(E) = exp(-tau(E))
```

The application’s Transmission backend uses total attenuation with coherent
scattering. Other channels are available in the table, but are not interchangeable
with energy-absorption coefficients. XCOM does not supply plasma ionization,
electron density, EOS parameters, or optical-laser opacity.

## Rebuild and independently verify

Generation needs Python 3.12+ and GNU Fortran. Application users do not need a
Fortran compiler to use the shipped JSON resources.

```bash
python3 x-com_fallback/build_xcom_fallback_from_xcom.py
python3 x-com_fallback/build_xcom_fallback_from_xcom.py --verify-only \
  --validation-report outputs/reports/xcom_table_validation.json
```

The generator extracts `XCOM.tar.gz` into a temporary directory, compiles there,
and queries the local program in batches of at most 64 energies. This also avoids
trying to execute a binary on a Windows-backed repository filesystem that cannot
retain executable permissions. Temporary source patches expand output precision
and correct legacy array declarations; the vendor archive is left untouched.
The fixed-width vendor database read formats remain unchanged.

The compile flags are `-std=legacy -O2 -fdefault-real-8 -fdefault-double-8`.
Metadata records archive and patched-source SHA-256 hashes, flags, generation
schema, exact-edge convention, and validation results. The generator produces:

- `x-com_fallback/xcom_fallback_1keV_12keV_extended.json`
- `x-com_fallback/xcom_fallback_1keV_12keV_extended.csv`
- an identical JSON copy under `src/helios/services/derived/resources/`

The historical filenames remain unchanged for compatibility. Regeneration is
byte-identical with the same source and toolchain. Restart the app after updating
resources so the in-memory table and its fingerprint use the new version.

## Regression checks

`tests/test_xcom_table_precision.py` checks NIST Cu anchors, both sides of Cu/Fe
edges, every edge pair, component sums, provenance, strict query defaults, the
standalone helper, bounded fingerprint work, and independent native holdouts.
The native check is skipped when GNU Fortran is unavailable; ordinary runtime
table regressions still run. The existing Transmission tests also verify that a
1,300-zone request resolves only the 12 unique material/energy combinations.

The old native-only Cu transmission fixture did not match its Cu composition and
areal density. Its references now use native XCOM coefficients for the actual
request. This fixture previously remained hidden by a skip when the optional
live wrapper was unavailable.

## Availability and scientific references

The new table ships in the v1.2.1 source, Debian, and Linux portable assets.
GitHub CI was skipped for this patch at the maintainer’s request. A new Windows
portable executable was not built; Windows users can use the source bundle or
build locally. The optional live XCOM wrapper remains a separate integration;
installing the updated table does not enable that missing wrapper.

- [NIST XCOM scope and limitations](https://physics.nist.gov/PhysRefData/Xcom/Text/intro.html)
- [NIST interpolation across atomic-shell boundaries and mixture weighting](https://physics.nist.gov/PhysRefData/Xcom/Text/chap3.html)
- [NIST Cu reference attenuation coefficients](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z29.html)
- [NIST downloadable XCOM 3.1 source/data](https://physics.nist.gov/PhysRefData/Xcom/Text/download.html)
