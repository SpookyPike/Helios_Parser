# Recipes for the HELIOS Parse / View repository.
#
# Everything here wraps a command that already exists; nothing is invented. Run
# `just` with no arguments to list the recipes, `just --list` for the same, and
# `just <recipe> --help`-style flags pass through with `*ARGS`.
#
# The venv is the one this repository documents. Override it for one run with
#   just venv=/opt/science-python/bin/python core

venv := ".venv/bin/python"
python := venv
ruff := "ruff"
sample := "tests/fixtures/v12/cu_v12_prefix.log"
sample_h5 := "/tmp/helios-sample.h5"

# Print the recipes.
default:
    @just --list --unsorted

# ---------------------------------------------------------------- environment

# Create the virtual environment and install the project with the desktop extras.
setup:
    python3 -m venv .venv
    {{python}} -m pip install --upgrade pip
    {{python}} -m pip install -e ".[desktop]"

# ---------------------------------------------------------------- quality gates

# The commit gate: about 8 seconds, five entries, pinned to the P-cores.
core:
    taskset -c 0-11 {{python}} scripts/run_test_suite.py --profile core

# The milestone gate: every entry, with coverage against the 75 % floor.
full:
    taskset -c 0-11 {{python}} scripts/run_test_suite.py --profile full --coverage

# Build the native LOG table parser into src/helios_parser (needs cargo).
native:
    {{python}} scripts/build_native_table.py

# Verify the installed native parser without building it.
native-check:
    {{python}} scripts/build_native_table.py --check

# One entry by module name, e.g. `just one test_radiation_output.py`.
one entry:
    taskset -c 0-11 {{python}} scripts/run_test_suite.py --profile full --only {{entry}}

# List the entries each profile selects, without running anything.
entries profile="core":
    {{python}} scripts/run_test_suite.py --profile {{profile}} --list

# Ruff, which lives outside the venv on this machine.
lint:
    {{ruff}} check src tests scripts

# The Rust gate for native/table_parser: formatting plus clippy at deny level.
rust-lint:
    cargo fmt --manifest-path native/table_parser/Cargo.toml --check
    cargo clippy --manifest-path native/table_parser/Cargo.toml --all-targets -- -D warnings

# Rust unit tests for the kernel contract.
rust-test:
    cargo test --manifest-path native/table_parser/Cargo.toml

# What a commit has to pass: lint plus the core gate.
check: lint core

# ---------------------------------------------------------------- the command line

# Convert one input to HDF5. `just convert run.log run.h5`, or pass flags through.
convert input output *ARGS:
    {{python}} -m helios_cli {{input}} {{output}} {{ARGS}}

# What simulation is this file, and which fields does it hold.
inspect path:
    {{python}} -m helios_diag inspect {{path}}

# Per-zone statistics with peak times. Flags pass through: --over time, --at 1139, --global.
stats path field *ARGS:
    {{python}} -m helios_diag stats {{path}} -f {{field}} {{ARGS}}

# One profile or one time trace. `just query run.h5 density --snapshot 1139`
query path field *ARGS:
    {{python}} -m helios_diag query {{path}} -f {{field}} {{ARGS}}

# Compare two conversions; exits 7 when they differ beyond tolerance.
diff left right *ARGS:
    {{python}} -m helios_diag diff {{left}} {{right}} {{ARGS}}

# Check a converted file against the writer contract; exits 7 on a failed check.
validate path:
    {{python}} -m helios_diag validate {{path}}

# Run the derived modules headlessly. `just analyse run.h5 --module xrd`
analyse path *ARGS:
    {{python}} -m helios_diag analyse {{path}} {{ARGS}}

# Convert the tracked sample deck, so the recipes above have something to read.
sample-h5:
    {{python}} -m helios_cli {{sample}} {{sample_h5}} --overwrite --quiet

# ---------------------------------------------------------------- fixtures

# Regenerate the golden HDF5 fixtures from the source logs.
goldens:
    {{python}} scripts/regenerate_golden_fixtures.py

# Capture the screenshots the published guides reference.
screenshots:
    {{python}} scripts/capture_docs_assets.py

# ---------------------------------------------------------------- performance

# The measured performance suite.
perf:
    taskset -c 0-11 {{python}} scripts/run_performance_suite.py

# ---------------------------------------------------------------- release

# Print the version every packaging script reads.
version:
    @{{python}} -c "import sys; sys.path.insert(0, 'src'); import helios_release; print(helios_release.RELEASE_VERSION)"

# Every place the version has to be bumped, so a bump is not done from memory.
version-carriers:
    @grep -rn "2\.[0-9]\+\.[0-9]\+" README.md pyproject.toml src/helios_release.py docs/index.html docs/release-notes.html docs/user-guide.html docs/developer-guide.html docs/architecture_extension.md docs/packaging.md 2>/dev/null | grep -v "12\.0" | head -40

# The shareable source bundle.
bundle:
    {{python}} scripts/create_release_bundle.py

# Build the distributables. The Windows ZIP has to be built on Windows.
deb:
    {{python}} scripts/build_debian_package.py

# The Linux portable tarball.
tarball:
    {{python}} scripts/build_linux_portable_tarball.py

# What to run before tagging: lint, the full gate, and the version check.
preflight: lint full version

# ---------------------------------------------------------------- plotting

# Radiation figures for a run that carries frequency-resolved output.
plot-radiation path *ARGS:
    {{python}} scripts/plot_radiation.py {{path}} {{ARGS}}