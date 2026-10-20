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

The XCOM lookup intentionally keeps the tabulated order and repeated energy
rows. Exact-energy lookup selects the first matching row; energy-edge handling
must preserve that convention.
