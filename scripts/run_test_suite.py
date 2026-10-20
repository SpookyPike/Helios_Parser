from __future__ import annotations

"""Run HELIOS test modules in isolated processes and retain an outcome manifest."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = ROOT / "tests" / "fixture_manifest.json"
CORE_NAMES = {
    "smoke_test.py",
    "test_bpf_pipeline.py",
    "test_cli_errors.py",
    "test_test_runner.py",
}
TIMEOUT_SECONDS = 120
TIMEOUT_OVERRIDES = {
    "test_app_phase412.py": 300,
    "test_validation.py": 300,
    "test_viewer_payloads.py": 300,
    "test_wave6_derived.py": 300,
}
PRODUCTION_ENTRIES = {"test_v1_release_gate.py"}
SKIP_PATTERN = re.compile(r"\bskipped\s*=\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class TestEntry:
    name: str
    command: tuple[str, ...]
    kind: str


def discover_entries(root: Path = ROOT) -> list[TestEntry]:
    tests = root / "tests"
    entries = [TestEntry("smoke_test.py", (sys.executable, "tests/smoke_test.py"), "smoke")]
    for path in sorted(tests.glob("test_*.py")):
        name = path.name
        if name == "test_viewer_payloads.py":
            command = (sys.executable, "scripts/run_payload_audit.py")
            kind = "payload_wrapper"
        else:
            command = (
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                name,
            )
            kind = "unittest_module"
        entries.append(TestEntry(name, command, kind))
    return entries


def select_entries(entries: list[TestEntry], profile: str, only: list[str]) -> list[TestEntry]:
    available = {entry.name for entry in entries}
    unknown = sorted(set(only) - available)
    if unknown:
        raise ValueError("unknown --only entry(s): " + ", ".join(unknown))
    if only:
        wanted = set(only)
    elif profile == "core":
        wanted = CORE_NAMES
    else:
        wanted = available
    return [entry for entry in entries if entry.name in wanted]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_fixtures(
    entries: list[TestEntry],
    manifest_path: Path = FIXTURE_MANIFEST,
    root: Path = ROOT,
    checked: dict[str, str | None] | None = None,
) -> list[str]:
    """Return actionable missing/mismatched fixture errors for selected entries."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"fixture manifest unavailable: {manifest_path}: {exc}"]
    requirements: dict[str, list[str]] = manifest.get("entry_requirements", {})
    prefix_requirements: dict[str, list[str]] = manifest.get("entry_prefix_requirements", {})
    fixtures: dict[str, dict[str, Any]] = manifest.get("fixtures", {})
    errors: list[str] = []
    checked = checked if checked is not None else {}
    for entry in entries:
        fixture_ids = list(requirements.get(entry.name, []))
        for prefix, prefix_ids in prefix_requirements.items():
            if entry.name.startswith(prefix):
                fixture_ids.extend(prefix_ids)
        for fixture_id in dict.fromkeys(fixture_ids):
            if fixture_id in checked:
                if checked[fixture_id]:
                    errors.append(f"{entry.name}: {checked[fixture_id]}")
                continue
            record = fixtures.get(fixture_id)
            if not record:
                checked[fixture_id] = f"fixture manifest has no record {fixture_id!r}"
                errors.append(f"{entry.name}: {checked[fixture_id]}")
                continue
            path = root / record["path"]
            if not path.is_file():
                checked[fixture_id] = f"missing fixture {record['path']} ({record.get('provenance', 'unspecified provenance')})"
                errors.append(f"{entry.name}: {checked[fixture_id]}")
                continue
            expected_size = record.get("size_bytes")
            if expected_size is not None and path.stat().st_size != expected_size:
                checked[fixture_id] = f"fixture size mismatch for {record['path']}: expected {expected_size}, got {path.stat().st_size}"
                errors.append(f"{entry.name}: {checked[fixture_id]}")
                continue
            expected_hash = record.get("sha256")
            if expected_hash and _sha256(path) != expected_hash:
                checked[fixture_id] = f"fixture hash mismatch for {record['path']}"
                errors.append(f"{entry.name}: {checked[fixture_id]}")
                continue
            checked[fixture_id] = None
    return errors


def required_fixture_ids(entries: list[TestEntry], manifest_path: Path = FIXTURE_MANIFEST) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact = manifest.get("entry_requirements", {})
    prefixes = manifest.get("entry_prefix_requirements", {})
    result: set[str] = set()
    for entry in entries:
        result.update(exact.get(entry.name, []))
        for prefix, fixture_ids in prefixes.items():
            if entry.name.startswith(prefix):
                result.update(fixture_ids)
    return result


def materialize_test_tree(
    entries: list[TestEntry], manifest_path: Path = FIXTURE_MANIFEST, source_root: Path = ROOT
) -> tuple[tempfile.TemporaryDirectory[str], Path, set[str]]:
    """Copy the current source and only declared input fixtures to a disposable test root."""
    temporary = tempfile.TemporaryDirectory(prefix="helios-test-tree-")
    target = Path(temporary.name) / "workspace"
    target.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.egg-info", ".pytest_cache")
    for directory in ("src", "tests", "scripts", "new_data", "x-com_fallback", "packaging", "docs", "release_assets"):
        source = source_root / directory
        if source.is_dir():
            shutil.copytree(source, target / directory, ignore=ignore)
    for name in (".gitattributes", ".editorconfig", ".gitignore", "pyproject.toml", "README.md", "STARTUP_MANUAL.md"):
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, target / name)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures: dict[str, dict[str, Any]] = manifest.get("fixtures", {})
    fixture_ids = required_fixture_ids(entries, manifest_path)
    protected: set[str] = set()
    for fixture_id in sorted(fixture_ids):
        record = fixtures[fixture_id]
        relative = Path(record["path"])
        destination = target / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / relative, destination)
        protected.add(relative.as_posix())
        destination.chmod(destination.stat().st_mode & ~0o222)
    (target / "outputs").mkdir(exist_ok=True)
    return temporary, target, protected


def reset_isolated_outputs(root: Path, protected: set[str]) -> None:
    output_root = root / "outputs"
    if not output_root.exists():
        output_root.mkdir()
        return
    for path in sorted(output_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            relative = path.relative_to(root).as_posix()
            if relative not in protected:
                path.unlink()
        elif path.is_dir() and not any(child.exists() for child in path.iterdir()):
            path.rmdir()


def build_env(entry: TestEntry, isolated_dir: Path, root: Path = ROOT) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(root / "src"), str(root / "tests"), env.get("PYTHONPATH", "")) if value
    )
    env["HELIOS_ENABLE_EXPERIMENTAL"] = "" if entry.name in PRODUCTION_ENTRIES else "1"
    env["HELIOS_DEV_MODE"] = "" if entry.name in PRODUCTION_ENTRIES else env.get("HELIOS_DEV_MODE", "")
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        target = isolated_dir / name.removeprefix("XDG_").lower()
        target.mkdir(parents=True, exist_ok=True)
        env[name] = str(target)
    env["HELIOS_TEST_SETTINGS_DIR"] = str(isolated_dir / "settings")
    (isolated_dir / "settings").mkdir(exist_ok=True)
    (root / "outputs").mkdir(exist_ok=True)
    env["HELIOS_TEST_OUTPUT_DIR"] = str(root / "outputs")
    return env


def environment_summary() -> dict[str, str | None]:
    def version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "h5py": version("h5py"),
        "PySide6": version("PySide6"),
        "pyqtgraph": version("pyqtgraph"),
    }


def run_entry(entry: TestEntry, report_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    timeout = TIMEOUT_OVERRIDES.get(entry.name, TIMEOUT_SECONDS)
    log_path = report_dir / f"{entry.name}.log"
    started = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    with tempfile.TemporaryDirectory(prefix="helios-test-") as scratch:
        isolated_dir = Path(scratch)
        env = build_env(entry, isolated_dir, root)
        command = list(entry.command)
        if command[0] == sys.executable:
            command[0] = sys.executable
        popen_options: dict[str, Any] = {}
        if os.name == "posix":
            popen_options["start_new_session"] = True
        elif os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            **popen_options,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            partial_output = exc.output or ""
            if isinstance(partial_output, bytes):
                partial_output = partial_output.decode(errors="replace")
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    tail, _ = process.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    tail, _ = process.communicate()
                # Descendants can close the wrapper's pipes and outlive its
                # process. Remove any remaining group members before continuing.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            elif os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                tail, _ = process.communicate()
            else:
                process.kill()
                tail, _ = process.communicate()
            if isinstance(tail, bytes):
                tail = tail.decode(errors="replace")
            output = partial_output
            if tail and tail not in output:
                output += tail
            returncode = 124
            output += f"\nTIMEOUT after {timeout} seconds; terminated process tree\n"
        elapsed = time.perf_counter() - started
    log_path.write_text(output, encoding="utf-8")
    match = SKIP_PATTERN.search(output)
    return {
        "name": entry.name,
        "kind": entry.kind,
        "command": list(entry.command),
        "cwd": str(root),
        "log": str(log_path),
        "runtime_seconds": round(elapsed, 6),
        "return_code": returncode,
        "signal": -returncode if returncode is not None and returncode < 0 else None,
        "timed_out": timed_out,
        "timeout_seconds": timeout,
        "skipped": int(match.group(1)) if match else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run every selected HELIOS test module in a fresh process.")
    parser.add_argument("--list", action="store_true", help="List test entries and exit.")
    parser.add_argument("--profile", choices=("core", "full"), default="core")
    parser.add_argument("--only", action="append", default=[], help="Run a discovered entry; may be repeated.")
    parser.add_argument("--report-dir", type=Path, help="Directory for per-entry logs and the JSON outcome manifest.")
    parser.add_argument("--fixture-manifest", type=Path, default=FIXTURE_MANIFEST)
    args = parser.parse_args(argv)
    entries = discover_entries()
    try:
        selected = select_entries(entries, args.profile, args.only)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        for entry in selected:
            print(entry.name)
        return 0
    if not selected:
        parser.error("selection contains no test entries")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = args.report_dir or Path(tempfile.gettempdir()) / "helios-test-runs" / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    checked_fixtures: dict[str, str | None] = {}
    entry_fixture_errors = {
        entry.name: check_fixtures([entry], args.fixture_manifest, checked=checked_fixtures) for entry in selected
    }
    runnable = [entry for entry in selected if not entry_fixture_errors[entry.name]]
    if runnable:
        temporary, execution_root, protected = materialize_test_tree(runnable, args.fixture_manifest)
    else:
        temporary, execution_root, protected = None, ROOT, set()
    outcomes = []
    try:
        for entry in selected:
            if entry_fixture_errors[entry.name]:
                errors = entry_fixture_errors[entry.name]
                log_path = report_dir / f"{entry.name}.log"
                output = "\n".join(f"FIXTURE ERROR: {error}" for error in errors) + "\n"
                log_path.write_text(output, encoding="utf-8")
                for line in output.splitlines():
                    print(line, file=sys.stderr)
                outcomes.append({
                    "name": entry.name,
                    "kind": entry.kind,
                    "command": list(entry.command),
                    "cwd": str(execution_root),
                    "log": str(log_path),
                    "runtime_seconds": 0.0,
                    "return_code": 125,
                    "signal": None,
                    "timed_out": False,
                    "timeout_seconds": TIMEOUT_OVERRIDES.get(entry.name, TIMEOUT_SECONDS),
                    "skipped": 0,
                    "fixture_errors": errors,
                    "not_executed": True,
                })
                continue
            reset_isolated_outputs(execution_root, protected)
            print(f"=== {entry.name} ===", flush=True)
            outcome = run_entry(entry, report_dir, execution_root)
            outcomes.append(outcome)
            label = "TIMEOUT" if outcome["timed_out"] else ("PASS" if outcome["return_code"] == 0 else "FAIL")
            print(f"--- {entry.name}: {label}, {outcome['runtime_seconds']:.2f}s; log={outcome['log']}", flush=True)
    finally:
        if temporary is not None:
            temporary.cleanup()

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "selected": [entry.name for entry in selected],
        "environment": environment_summary(),
        "outcomes": outcomes,
        "passed": sum(row["return_code"] == 0 and not row["timed_out"] for row in outcomes),
        "failed": sum(row["return_code"] != 0 or row["timed_out"] for row in outcomes),
        "not_executed": [row["name"] for row in outcomes if row.get("not_executed")],
    }
    manifest_path = report_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0 if result["failed"] == 0 else (2 if result["not_executed"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
