# Scientific package resources

The JSON resources in this directory are runtime inputs for installed and
frozen HELIOS builds. Their source copies remain in the repository for the
existing table-generation and release workflows; tests compare the copies so a
regeneration cannot silently leave the shipped data stale.

- `xcom_fallback_1keV_12keV_extended.json` is generated from the XCOM-derived
  table by `x-com_fallback/build_xcom_fallback_from_xcom.py`.
- `hed_helios_cold_electronic_baseline_core.json` is the benchmark cold-state
  electronic baseline used by plasmon electron-policy calculations.
- `../reference_data/plasmon/*.json` contains the article-facing dispersion
  reference data and is included alongside these package resources.

The XCOM lookup retains both edge limits. Exact-energy lookup selects the first,
below-edge row; interpolation must stay on one side of an edge.

The XCOM table uses binary64 calculations, adaptive log-log sampling, and
explicit below/above-edge pairs over 1–35 keV. Exact-edge queries retain the first
(below-edge) row convention. Extrapolation is disabled by default. The table
metadata records archive and patched-source hashes, compiler flags, and native
holdout errors. These quantify numerical interpolation, not physical accuracy.
The accompanying `xcom_atomic_weights.json` comes from the same archive's
`ATWTS.DAT`; composition identities include its fingerprint.
See `docs/xcom-tables.md` in the source tree for regeneration instructions.
