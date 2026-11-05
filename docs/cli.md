# Command line reference

Two commands. `helios_to_hdf5` converts HELIOS output to HDF5; `helios_diag` reads a
converted run and answers questions about it. Both write JSON to stdout and logs to
stderr, so an agent can consume the answer without parsing prose.

The same surface is described for agents in `skills/helios-cli/SKILL.md`.

## Converting

```text
helios_to_hdf5 INPUT OUTPUT
helios_to_hdf5 INPUT -o OUTPUT
helios_to_hdf5 INPUT... --outdir DIR
```

| flag | effect |
|---|---|
| `--compression {lzf,gzip}` | off by default. `lzf` is the fast option and enables the HDF5 shuffle filter; `gzip` is the smaller one and does not |
| `--overwrite` | replace an existing output |
| `--json` | one JSON object per input on stdout |
| `--progress` | one machine-readable line per stage on stderr |
| `--quiet` | warnings and errors only |
| `--verbose` | debug output, and errors are re-raised |
| `--version` | print the version and exit |

Input may be `.log`, `.bpf` or `.exo`. A `.log` and a `.bpf` beside each other are read
as one run: the log supplies the metadata, the BPF the bulk fields. An `.exo` carries
both. Companion files are found automatically.

A batch keeps going after a failure and returns the first non-zero code, so one bad
file does not hide the rest.

```bash
helios_to_hdf5 run.log run.h5 --json
helios_to_hdf5 runs/*.log --outdir out/ --json --compression lzf
```

`--compression lzf` measured 184.4 MB against 118.6 MB on the largest tracked deck, a
35.7 % saving for 2.6 % more write time. `gzip` reaches 102.2 MB at 12.4 s against
9.6 s. Neither changes a stored value.

## Diagnosing

Every subcommand accepts `--format {json,table,csv}` (default `json`) and `--out FILE`.

`inspect`, `stats`, `query` and `analyse` accept a raw deck as well as a converted run:
a `.log`, a `.bpf`, or an HDF5 `.exo`. Nothing has to be converted first, and the answers
are the same numbers the converted file holds. `validate` and `diff` still require a
converted run, because both check the writer contract rather than the physics.

| input | `inspect` | `stats`, `query` | `analyse` |
|---|---|---|---|
| converted `.h5` | metadata only | one field read | six core fields plus the optional set |
| raw `.log` | header plus the first snapshot | one full parse, then the field | one full parse, then the derived modules |
| raw `.bpf` | layout plus the first snapshot | one full parse | one full parse |
| HDF5 `.exo` | full read | full read | full read |

A raw deck carries no `schema_version`, no writer contract and no VISAR structure. The
output says so in `notes` instead of guessing, and a classic NetCDF-3 `.exo` is named as
the unsupported variant rather than failing as a broken HDF5 file.

### `helios_diag inspect RUN_OR_DECK`

What simulation is this file, and what does it hold. No field data is read; for a raw
deck the header and the first snapshot are parsed, which is 10 ms of work on the 530 MB
reference log plus one sequential scan to count the snapshots.

It reports the file provenance, the deck setup and the laser pulse decoded from
`metadata/input_parameters` (wavelength, incidence angle, propagation direction, power
table, frequency grid, time controls), every field with its unit, label, dimensions and
shape, and the VISAR readiness.

```bash
helios_diag inspect run.h5 | jq '.physics.laser'
```

### `helios_diag stats RUN -f FIELD`

| mode | one row per | columns |
|---|---|---|
| `--over zones` (default) | zone | zone, mean, min, max, p05, p50, p95, std, peak_time_s |
| `--over time` | snapshot | snapshot, time_s, mean, min, max, p05, p50, p95, std |
| `--at N` / `--at-time S` | zone | zone, value |
| `--global` | the whole field | every statistic |

`peak_time_s` is the time of that zone's own maximum, not a threshold crossing. For the shock breakout time with a configurable threshold, use `analyse --breakout-fraction`. A zone
whose spread is inside `--flat-atol` / `--flat-rtol` reports no peak, because `argmax`
on a constant array returns index 0 and that reads as a peak at t=0.

`--quantiles 5,50,95` sets the percentiles. `--region` and `--material` restrict the
zones; `--by-region` and `--by-material` emit one aggregate row per group instead.

```bash
helios_diag stats run.h5 -f density | jq '.rows[0:3]'
helios_diag stats run.h5 -f density --at 1139
helios_diag stats run.h5 -f density --over time
```

### `helios_diag query RUN -f FIELD`

The numbers behind a statistic. Each form uses the narrowest reader accessor, so a
profile costs one row read rather than the whole field.

```bash
helios_diag query run.h5 -f density --snapshot 1139   # zone,value
helios_diag query run.h5 -f density --zone 900        # snapshot,time_s,value
helios_diag query run.h5 -f density --time 1.1e-8     # nearest snapshot
```

### `helios_diag diff LEFT RIGHT`

Compares two conversions dataset by dataset. For each numeric dataset present in both:
`max_abs_diff`, `max_rel_diff`, `n_differing`, and `status`. Shape and presence
mismatches are reported structurally rather than raised. `--rtol`, `--atol` and
`--fields` tune it. Two datasets are in memory at a time, never two files.

This is the correctness half of an A/B measurement, and it is what would have caught
both the 2.0.1 fixture drift and the 2.0.2 shuffle change in one command.

### `helios_diag validate RUN`

Checks a converted file against the writer contract: the entries the reader needs, the
snapshot count, field metadata consistency, zero-extent families, the radiation marker,
and the schema 2.2 chunk layout. Checks that only apply to the modern layout report
`skip` on a pre-2.0 file, because such a file is a legitimate input.

### `helios_diag analyse RUN`

Runs the derived analysis without the desktop application.

```bash
helios_diag analyse run.h5 --module xrd
helios_diag analyse run.h5 --module shock,preheat --snapshot 1139
```

Known modules: `shock`, `wavefront`, `preheat`, `xrd`, `spectroscopy`, `plasmon`,
`transmission`. Arrays larger than `--max-array` elements are summarized rather than
dumped.

`plasmon` and `transmission` are hidden in the production backend. Asking for one without
`--include-experimental` exits 2 and says so, rather than returning a payload of nulls.
The flag enables them for that invocation only:

```bash
helios_diag analyse run.h5 --module plasmon --include-experimental
```

`--breakout-fraction F` sets where along its track the front counts as broken out, as a
fraction of the distance from the starting face: 0.5 is the midpoint, a value near 1 is
the rear face. The default is 0.5, so the reported time is the midpoint crossing; use
`--breakout-fraction 0.95` for the near-face reading. The value enters the derived cache
key, so changing it recomputes rather than reusing a cached shock result.

```bash
helios_diag analyse run.h5 --module shock --breakout-fraction 0.25   # earlier, quarter way
helios_diag analyse run.h5 --module shock --breakout-fraction 0.95   # near the rear face
```

When the front stops on a material interface and never reaches the threshold, the result
carries a `caution` warning naming the interface zone and the two regions. That is a
different diagnosis from "the shock did not arrive", which stays an `info` note.

`radiation` is production and needs no flag; it reports "this run carries no radiation
output" when the deck left the flag off.

One limit, stated in the output too: the loader always reads the six core fields, so
`--module` bounds the optional set rather than the core.

### `helios_plot_radiation RUN`

Boundary spectra and the spectral energy-density map, for a run that carries
frequency-resolved radiation. Needs the `plot` extra. A run with flux dumps but an empty
spectral axis skips the third figure and says why.

## The JSON envelope

```json
{
  "command": "stats",
  "tool_version": "2.2.1",
  "generated_at": "2026-11-04T18:20:00Z",
  "file": {"path": "...", "bytes": 0, "format": "...", "schema_version": "2.2",
           "geometry": "planar", "geometry_raw": "planar", "source_file": "...",
           "helios_version": "...", "parse_mode": "log",
           "coordinate_model": "centre_primary", "run_status": {},
           "n_snapshots": 0, "n_zones": 0, "n_regions": 0, "n_materials": 0},
  "field": {"name": "density", "unit": "g/cm3", "label": "Mass dens",
            "label_origin": "recorded", "dimensions": ["time", "zone"]},
  "request": {"mode": "over_zones", "axis": "zone"},
  "accounting": {"n_values": 0, "n_nan": 0, "n_used": 0},
  "notes": [],
  "rows": []
}
```

Resolution order is fixed: file attribute, then `metadata/`, then dataset attribute, then
`field_metadata/`, then inferred. `label_origin` records which step answered.

**A unit is never guessed.** HELIOS stores `laser_source/wavelength` as a bare `0.515`.
Units are emitted only where the repository itself already declares them; everything else
is `null` with a note naming the convention. A wrong unit on a laser wavelength is worse
than an absent one.

Non-finite values become `null`, and serialization runs with `allow_nan=False`, because
`NaN` is not valid JSON and strict parsers reject it.

Read `notes` before the numbers. Every degradation is stated there: a missing schema
version, a label inferred from a dataset name, a zone that never peaked, a requested time
resolved to a different snapshot.

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 2 | usage error |
| 3 | input missing or unreadable |
| 4 | not readable HELIOS output (`helios_to_hdf5`) |
| 5 | output exists, pass `--overwrite` |
| 6 | output could not be written |
| 7 | `diff` found a difference, or `validate` found a failed check |
| 130 | interrupted |

## Measured cost

On `10ns+100Al+60CH+6.1TW_stabilized.h5` (410 MB, 1501 snapshots by 1200 zones), pinned
to the P-cores with `taskset -c 0-11`, work time excluding interpreter startup:

| command | measured |
|---|---|
| `--version`, `--help` | about 30 ms, no numpy or h5py import |
| `inspect` | 8 ms |
| `stats --over zones` | 52 ms |
| `stats --over time` | 41 ms |
| `stats --at N` | 11 ms |
| `query --snapshot N` | 11 ms |
| `inspect` on the raw 530.7 MB `.log` | 0.5 s, one sequential scan to count snapshots |
| `inspect` on the raw 40 MB `.bpf` | 61 ms |
| `stats` on the raw 43 MB `.log` | 0.3 s, one full parse |

The reductions are the floor: 31 ms of the 52 is numpy, and the rest is opening the file
and building the envelope. Reading one field is 3 ms; reading one snapshot row is under
1 ms, which is why `--at` does not read the whole field.

## The agent skill

`skills/helios-cli/` holds one canonical `SKILL.md` plus two adapters generated from the
same body.

| harness | artifact | install |
|---|---|---|
| Claude Code | `SKILL.md` | `cp -r skills/helios-cli ~/.claude/skills/helios-cli` |
| MiMoCode | `SKILL.md` | `cp -r skills/helios-cli ~/.config/mimocode/skills/helios-cli` |
| OpenCode | `SKILL.md` | `cp -r skills/helios-cli .opencode/skill/helios-cli` |
| Codex | `AGENTS.fragment.md` | append it to your `AGENTS.md` |
| anything else | `README.md` | the same body without frontmatter |

`tests/test_cli_skill_consistency.py` keeps it honest: every command the skill tells an
agent to run must answer `--help`, every subcommand must appear in the skill, and the
envelope it quotes must match a real payload.

## Recipes

The repository ships a `justfile` for the common workflows: `just` lists them, `just core`
runs the commit gate, `just stats run.h5 density` runs a diagnostic.