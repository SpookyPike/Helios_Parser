---
name: helios-cli
description: Drive the HELIOS Parse / View command line to convert HELIOS radiation-hydrodynamics output and to run quick diagnostics on it. Use when asked to convert a HELIOS .log, .bpf or .exo file to HDF5; to inspect what simulation a converted run holds; to get per-zone or per-snapshot statistics such as mean, percentiles or peak time for a field; to extract a profile or a time trace; to compare two conversions; to check a converted file; or to run the derived analysis (shock, wavefront, preheat, XRD, spectroscopy, plasmon, transmission) without the desktop application. Triggers include "inspect a HELIOS run", "zone statistics", "compare two conversions", "is this HDF5 file valid", "convert this deck".
---

# HELIOS command line

Two commands. `helios_to_hdf5` converts; `helios_diag` reads a converted run and
answers questions about it. Both write JSON to stdout and logs to stderr.

## The contract

Every `helios_diag` subcommand emits one JSON object with this shape. Treat it as
stable; it is versioned and test-covered.

```json
{
  "command": "stats",
  "tool_version": "2.1.0",
  "generated_at": "2026-11-04T18:20:00Z",
  "file": {
    "path": "run.h5", "bytes": 410672400,
    "format": "HELIOS log converted to HDF5",
    "schema_version": "2.2", "geometry": "planar", "geometry_raw": "planar",
    "source_file": "run.log", "helios_version": "6.2.0",
    "parse_mode": "log", "coordinate_model": "centre_primary",
    "run_status": {"state": "complete", "source": "log_footer"},
    "n_snapshots": 1501, "n_zones": 1200, "n_regions": 1, "n_materials": 1
  },
  "field": {"name": "density", "unit": "g/cm3", "label": "Mass dens",
            "label_origin": "recorded", "dimensions": ["time", "zone"]},
  "request": {"mode": "over_zones", "axis": "zone"},
  "accounting": {"n_values": 1801200, "n_nan": 0, "n_used": 1801200},
  "notes": [],
  "rows": []
}
```

`helios_to_hdf5 --json` emits one object per input instead, with `bytes`,
`n_snapshots`, `n_zones`, `compression`, `duration_ms`, `warnings` and `status`.

## Discovery, not memorisation

Do not guess flags or field names. In this order:

```bash
helios_diag --help                    # the subcommands
helios_diag stats --help              # that subcommand's flags
helios_diag inspect run.h5            # what fields exist, in what units
```

`inspect` is the map: it lists every field with its unit, label, dimensions and
shape, plus the deck setup and the laser pulse. Read it before asking for a field.

## Recipes

Convert a deck. A `.log` and a `.bpf` beside each other are read as one run; an
`.exo` carries everything.

```bash
helios_to_hdf5 run.log run.h5 --json
helios_to_hdf5 run.exo run.h5 --compression lzf --json   # lzf: ~35% smaller
helios_to_hdf5 runs/*.log --outdir out/ --json           # one object per input
```

What simulation is this, and what did the laser do?

```bash
helios_diag inspect run.h5 | jq '.physics.laser'
# wavelength, incidence angle in degrees, propagation direction, power table
```

Per-zone statistics, with the time each zone peaked.

```bash
helios_diag stats run.h5 -f density | jq '.rows[0:3]'
# zone, mean, min, max, p05, p50, p95, std, peak_time_s
helios_diag stats run.h5 -f density --at 1139        # one profile
helios_diag stats run.h5 -f density --over time      # one row per snapshot
helios_diag stats run.h5 -f density --by-region      # aggregate per region
```

The numbers behind a statistic.

```bash
helios_diag query run.h5 -f density --snapshot 1139   # zone,value
helios_diag query run.h5 -f density --zone 900        # snapshot,time_s,value
helios_diag query run.h5 -f density --time 1.1e-8     # nearest snapshot
```

Did my change alter the output?

```bash
helios_diag diff before.h5 after.h5          # exits 7 when a dataset differs
helios_diag diff a.h5 b.h5 --fields density  # narrow it down
```

Is this file sound?

```bash
helios_diag validate run.h5                  # exits 7 on a failed check
```

The derived analysis, headless.

```bash
helios_diag analyse run.h5 --module xrd
helios_diag analyse run.h5 --module shock,preheat --snapshot 1139
helios_diag analyse run.h5 --module plasmon --include-experimental   # hidden by default
```

`plasmon` and `transmission` are hidden in the production backend. Without
`--include-experimental` the command exits 2 and tells you; it never returns a payload of
nulls pretending to be a result.

## Reading the output

- **Read `notes` before the numbers.** Every degradation is written there: a
  missing schema version, a label inferred from a dataset name, a zone that never
  peaked, a requested time resolved to a different snapshot.
- **`null` is not zero.** `unit: null` means the file did not record the unit.
  `peak_time_s: null` means the zone is flat and has no peak. Never substitute a
  value.
- **`label_origin: "inferred"`** means the label came from the dataset name, not
  from recorded metadata. The number is still real; the name may be generic.
- **`accounting.n_nan`** is how many values were skipped. A mean over 1501 values
  with 40 skipped NaNs is not the same claim as a mean over 1501 values.
- **`request`** states the axis and the reduction. Quote it when you report a
  number, so the reader knows what it is a statistic of.

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 2 | usage error (bad flags) |
| 3 | input missing or unreadable |
| 4 | the input is not readable HELIOS output (`helios_to_hdf5` only) |
| 5 | the output already exists; pass `--overwrite` |
| 6 | the output could not be written |
| 7 | `diff` found a difference, or `validate` found a failed check |
| 130 | interrupted |

A batch conversion keeps going after a failure and returns the first non-zero
code, so one bad file does not hide the others.

## Do not

- Do not parse stderr as data. It carries logs; stdout carries the answer.
- Do not assume a field exists. Run `inspect` first.
- Do not treat `unit: null` or `peak_time_s: null` as zero.
- Do not read a statistic without its `request` block; `mean` over zones and
  `mean` over snapshots are different numbers.
- Do not pipe `--format table` output into a parser. JSON is the machine format;
  `table` and `csv` are for humans.