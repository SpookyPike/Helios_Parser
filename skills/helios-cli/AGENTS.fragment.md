# HELIOS command line

This repository ships two commands. `helios_to_hdf5` converts HELIOS `.log` /
`.bpf` / `.exo` output to HDF5; `helios_diag` reads a converted run.

Both write JSON to stdout and logs to stderr. Every `helios_diag` subcommand
emits one object with `command`, `tool_version`, `generated_at`, `file`, and
`notes`; statistics add `field`, `request`, `accounting` and `rows`.

Work in this order:

1. `helios_diag --help`, then `helios_diag <sub> --help`. Do not guess flags.
2. `helios_diag inspect run.h5` — it lists every field with its unit, label and
   shape, plus the deck setup and the laser pulse. Read it before asking for a
   field.
3. Then `stats` for per-zone or per-snapshot statistics, `query` for the numbers
   behind them, `diff` to compare two conversions, `validate` to check a file,
   `analyse` for the derived modules.

Rules that matter:

- Read `notes` before the numbers. Every degradation is stated there.
- `null` is not zero. `unit: null` means the file recorded no unit;
  `peak_time_s: null` means the zone is flat and has no peak.
- `accounting.n_nan` is how many values were skipped.
- Quote the `request` block when reporting a number; `mean` over zones and
  `mean` over snapshots are different statistics.
- Exit codes: 0 ok, 2 usage, 3 unreadable, 4 not HELIOS output, 5 output exists,
  6 write failure, 7 a difference or a failed check, 130 interrupted.
- Do not parse stderr as data. Do not pipe `--format table` into a parser.

The full reference, with recipes, is `skills/helios-cli/SKILL.md`.
