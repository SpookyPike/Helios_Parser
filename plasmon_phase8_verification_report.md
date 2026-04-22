# Plasmon Phase 8 verification (7/7 checks passed)

- [x] **quicklook backward-compatible** — ωpe=1.05 eV
- [x] **rpa stable spectrum** — points=801
- [x] **rpa_static_lfc finite branch** — G=0.00168
- [x] **FWHM broadens observed linewidth** — 0.1414 -> 4.017 eV
- [x] **invalid Mermin is flagged for benchmark use** — status=invalid_for_benchmark, fallback=1.000
- [x] **LOS reacts to deselection** — synthetic two-region validation
- [x] **cache buckets reuse identical request** — 162.30 ms -> 0.68 ms

## Bundled example sanity checks
- `5Fe+4.9TW+light_stabilized.h5`: requested=mermin_static_lfc, applied=mermin_static_lfc, benchmark=invalid_for_benchmark, full_exec=False, fallback=0.393, peak=n/a, FWHM=n/a, finite_fraction=n/a (empty spectrum), zones=500, clusters=262
- `Cu1e17_cyl_stabilized.h5`: requested=mermin_static_lfc, applied=mermin_static_lfc, benchmark=invalid_for_benchmark, full_exec=False, fallback=0.833, peak=n/a, FWHM=n/a, finite_fraction=n/a (empty spectrum), zones=400, clusters=6
- `Cu_0166_stabilized.h5`: requested=mermin_static_lfc, applied=mermin_static_lfc, benchmark=invalid_for_benchmark, full_exec=False, fallback=0.857, peak=n/a, FWHM=n/a, finite_fraction=n/a (empty spectrum), zones=300, clusters=7

## Notes
- Quicklook compatibility and RPA/Mermin/LFC limit checks were re-run from source code, not copied from earlier summaries.
- Bundled examples do not contain multiple region/material selections, so deselection sensitivity is validated on a synthetic multi-region dataset.