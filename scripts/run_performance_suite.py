#!/usr/bin/env python3
"""Run reproducible, paired performance cases against two source trees.

The fixture manifest is JSON. Each case defines a command, named fixture paths,
numeric JSON metrics to compare, and sample counts. Commands receive
``{source}``, ``{python}``, and ``{fixture:<name>}`` substitutions and must
print one JSON object to stdout. Use ``paired`` for acceptance measurements;
``baseline`` and ``candidate`` are useful for collecting profiles separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_SAMPLES = 10
REGRESSION_LIMIT = 1.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(root: Path) -> dict[str, str | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    digest = hashlib.sha256()
    for relative in ("src", "scripts"):
        directory = root / relative
        if not directory.is_dir():
            continue
        for path in sorted(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(bytes.fromhex(_sha256(path)))
    return {"git_revision": revision, "source_sha256": digest.hexdigest()}


def _environment() -> dict[str, Any]:
    try:
        import numpy

        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    try:
        import h5py

        h5py_version = h5py.__version__
    except ImportError:
        h5py_version = None
    try:
        import PySide6

        qt_version = PySide6.__version__
    except ImportError:
        qt_version = None
    try:
        import psutil

        ram_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        ram_bytes = None
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemTotal:"):
                    ram_bytes = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            pass
    processor = platform.processor()
    if not processor:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace").splitlines():
                if line.startswith("model name"):
                    processor = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "helios_environment": {key: value for key, value in sorted(os.environ.items()) if key.startswith("HELIOS_")},
        "numpy": numpy_version,
        "h5py": h5py_version,
        "pyside6": qt_version,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported fixture manifest schema: {manifest.get('schema_version')!r}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture manifest must contain a non-empty cases list")
    names = [case.get("id") for case in cases]
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise ValueError("every case needs a unique non-empty id")
    for case in cases:
        if not isinstance(case.get("command"), list) or not case["command"]:
            raise ValueError(f"case {case['id']} needs a non-empty command array")
        if not isinstance(case.get("metrics"), dict) or not case["metrics"]:
            raise ValueError(f"case {case['id']} needs numeric JSON metrics")
        if any(direction not in ("lower", "higher") for direction in case["metrics"].values()):
            raise ValueError(f"case {case['id']} metric directions must be lower or higher")
        case.setdefault("warmups", 1)
        case.setdefault("samples", DEFAULT_SAMPLES)
        if int(case["warmups"]) < 0 or int(case["samples"]) < 1:
            raise ValueError(f"case {case['id']} has invalid warmup/sample counts")
        fixtures = case.get("fixtures", {})
        if not isinstance(fixtures, dict):
            raise ValueError(f"case {case['id']} fixtures must be an object")
        for key, record in fixtures.items():
            fixture_path = Path(record["path"]).expanduser().resolve()
            if not fixture_path.is_file():
                raise FileNotFoundError(f"case {case['id']} fixture {key} is missing: {fixture_path}")
            actual_hash = _sha256(fixture_path)
            expected_hash = record.get("sha256")
            if expected_hash and expected_hash != actual_hash:
                raise ValueError(f"case {case['id']} fixture {key} SHA-256 mismatch")
            record["path"] = str(fixture_path)
            record["sha256"] = actual_hash
    return manifest


def _metric_value(value: Any, dotted_path: str) -> float:
    if isinstance(value, dict) and dotted_path in value:
        value = value[dotted_path]
    else:
        for part in dotted_path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"missing metric {dotted_path!r}")
            value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {dotted_path!r} is not numeric")
    if not (float("-inf") < float(value) < float("inf")):
        raise ValueError(f"metric {dotted_path!r} is not finite")
    return float(value)


def _case_fixtures(case: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {key: {"path": value["path"], "sha256": value["sha256"]} for key, value in sorted(case.get("fixtures", {}).items())}


def _command(case: dict[str, Any], source: Path, output_path: Path | None = None) -> list[str]:
    substitutions = {
        "source": str(source.resolve()),
        "python": sys.executable,
        "runner": str(Path(__file__).resolve().parent),
    }
    if output_path is not None:
        substitutions["output"] = str(output_path)
    substitutions.update({f"fixture:{key}": value["path"] for key, value in case.get("fixtures", {}).items()})
    pattern = re.compile(r"\{([^{}]+)\}")
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token not in substitutions:
            raise ValueError(f"unknown command substitution {{{token}}} in case {case['id']}")
        return substitutions[token]
    return [pattern.sub(replace, str(part)) for part in case["command"]]


def _run_sample(case: dict[str, Any], source: Path) -> tuple[float, dict[str, float], str]:
    with tempfile.TemporaryDirectory(prefix="helios-perf-") as tmp:
        command = _command(case, source, Path(tmp) / "benchmark-output.json")
        start = time.perf_counter()
        environment = os.environ.copy()
        source_root = str(source.resolve())
        environment["HELIOS_PERF_SOURCE_ROOT"] = source_root
        environment["PYTHONPATH"] = str(source.resolve() / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        environment.update({str(key): str(value) for key, value in case.get("environment", {}).items()})
        result = subprocess.run(command, cwd=source, env=environment, text=True, capture_output=True, check=False)
        elapsed = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(
                f"case {case['id']} failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-4000:]}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"case {case['id']} stdout is not one JSON object: {exc}\n{result.stdout[-2000:]}") from exc
    metrics = {name: _metric_value(payload, name) for name in case["metrics"]}
    for check in case.get("checks", []):
        observed = _metric_value(payload, check["metric"])
        if observed > float(check["maximum"]):
            raise ValueError(
                f"case {case['id']} check {check['metric']}={observed} exceeds {check['maximum']}"
            )
    return elapsed, metrics, result.stdout


def _profile(manifest_path: Path, source: Path, side: str) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    source = source.resolve()
    cases: dict[str, Any] = {}
    for case in manifest["cases"]:
        for _ in range(int(case["warmups"])):
            _run_sample(case, source)
        samples = []
        for index in range(int(case["samples"])):
            elapsed, metrics, stdout = _run_sample(case, source)
            samples.append({"index": index, "process_elapsed_s": elapsed, "metrics": metrics, "stdout_json": json.loads(stdout)})
        cases[case["id"]] = {
            "fixtures": _case_fixtures(case),
            "settings": {"command": case["command"], "warmups": int(case["warmups"]), "samples": int(case["samples"]), "metrics": case["metrics"], "checks": case.get("checks", []), "equivalent_metrics": case.get("equivalent_metrics", {}), "environment": case.get("environment", {})},
            "samples": samples,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "side": side,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": _source_fingerprint(source),
        "environment": _environment(),
        "cases": cases,
    }


def _paired(manifest_path: Path, baseline_source: Path, candidate_source: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    baseline_source, candidate_source = baseline_source.resolve(), candidate_source.resolve()
    sides = {
        "baseline": {"schema_version": SCHEMA_VERSION, "side": "baseline", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": _source_fingerprint(baseline_source), "environment": _environment(), "paired": True, "pair_seed": seed, "cases": {}},
        "candidate": {"schema_version": SCHEMA_VERSION, "side": "candidate", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": _source_fingerprint(candidate_source), "environment": _environment(), "paired": True, "pair_seed": seed, "cases": {}},
    }
    rng = random.Random(seed)
    for case in manifest["cases"]:
        for side, source in (("baseline", baseline_source), ("candidate", candidate_source)):
            for _ in range(int(case["warmups"])):
                _run_sample(case, source)
        samples: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
        for index in range(int(case["samples"])):
            order = ["baseline", "candidate"]
            rng.shuffle(order)
            for side in order:
                source = baseline_source if side == "baseline" else candidate_source
                elapsed, metrics, stdout = _run_sample(case, source)
                samples[side].append({"index": index, "pair_order": order, "process_elapsed_s": elapsed, "metrics": metrics, "stdout_json": json.loads(stdout)})
        for side in ("baseline", "candidate"):
            sides[side]["cases"][case["id"]] = {
                "fixtures": _case_fixtures(case),
                "settings": {"command": case["command"], "warmups": int(case["warmups"]), "samples": int(case["samples"]), "metrics": case["metrics"], "checks": case.get("checks", []), "equivalent_metrics": case.get("equivalent_metrics", {}), "environment": case.get("environment", {})},
                "samples": samples[side],
            }
    return sides["baseline"], sides["candidate"]


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _median_bootstrap_interval(values: list[float], *, seed: int) -> list[float]:
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(2000)
    )
    return [_percentile(medians, 0.025), _percentile(medians, 0.975)]


def compare_profiles(baseline: dict[str, Any], candidate: dict[str, Any], *, minimum_samples: int = 10) -> dict[str, Any]:
    if baseline.get("schema_version") != SCHEMA_VERSION or candidate.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("profile schema mismatch")
    if baseline.get("environment") != candidate.get("environment"):
        raise ValueError("benchmark environments differ")
    if set(baseline.get("cases", {})) != set(candidate.get("cases", {})):
        raise ValueError("baseline/candidate case sets differ")
    paired = bool(baseline.get("paired") and candidate.get("paired"))
    if paired and baseline.get("pair_seed") != candidate.get("pair_seed"):
        raise ValueError("paired profile seeds differ")
    report_cases = {}
    failures: list[str] = []
    for case_id in sorted(baseline["cases"]):
        before, after = baseline["cases"][case_id], candidate["cases"][case_id]
        if before.get("fixtures") != after.get("fixtures") or before.get("settings") != after.get("settings"):
            raise ValueError(f"case {case_id} has incompatible fixtures or benchmark settings")
        old_samples, new_samples = before.get("samples", []), after.get("samples", [])
        if len(old_samples) < minimum_samples or len(new_samples) < minimum_samples:
            raise ValueError(f"case {case_id} has insufficient samples ({len(old_samples)}, {len(new_samples)}; need {minimum_samples})")
        if paired and [sample.get("index") for sample in old_samples] != [sample.get("index") for sample in new_samples]:
            raise ValueError(f"case {case_id} paired sample indices differ")
        if before["settings"].get("equivalent_metrics") and not paired:
            raise ValueError(f"case {case_id} numerical equivalence checks require paired profiles")
        metric_names = set(before["settings"]["metrics"])
        case_result: dict[str, Any] = {"samples": len(old_samples), "paired": paired, "metrics": {}}
        equivalent: dict[str, Any] = {}
        for metric, tolerance in before["settings"].get("equivalent_metrics", {}).items():
            absolute_tolerance = float(tolerance.get("absolute_tolerance", 0.0))
            relative_tolerance = float(tolerance.get("relative_tolerance", 0.0))
            residuals = []
            for old_sample, new_sample in zip(old_samples, new_samples):
                old_value = _metric_value(old_sample.get("stdout_json"), metric)
                new_value = _metric_value(new_sample.get("stdout_json"), metric)
                tolerance_value = max(absolute_tolerance, relative_tolerance * max(abs(old_value), abs(new_value)))
                residuals.append({"absolute_error": abs(old_value - new_value), "tolerance": tolerance_value})
            passed = all(item["absolute_error"] <= item["tolerance"] for item in residuals)
            if not passed:
                failures.append(f"{case_id}.{metric}: baseline/candidate equivalence tolerance exceeded")
            equivalent[metric] = {"passed": passed, "maximum_absolute_error": max(item["absolute_error"] for item in residuals), "maximum_tolerance": max(item["tolerance"] for item in residuals)}
        case_result["equivalent_metrics"] = equivalent
        for metric in sorted(metric_names):
            direction = before["settings"]["metrics"][metric]
            old = [_metric_value(sample.get("metrics"), metric) for sample in old_samples]
            new = [_metric_value(sample.get("metrics"), metric) for sample in new_samples]
            ratios = []
            for old_value, new_value in zip(old, new):
                denominator = old_value if direction == "lower" else new_value
                numerator = new_value if direction == "lower" else old_value
                if denominator == 0.0:
                    ratios.append(1.0 if numerator == 0.0 else sys.float_info.max)
                else:
                    ratios.append(numerator / denominator)
            if any(ratio < 0 or not (ratio < float("inf")) for ratio in ratios):
                raise ValueError(f"case {case_id} metric {metric} has invalid comparison values")
            median_ratio = statistics.median(ratios)
            ci = _median_bootstrap_interval(ratios, seed=seed_from_name(case_id, metric))
            if median_ratio > REGRESSION_LIMIT:
                failures.append(f"{case_id}.{metric}: median regression ratio {median_ratio:.3f} exceeds {REGRESSION_LIMIT:.2f}")
            if not paired:
                verdict = "unpaired_inconclusive"
            elif median_ratio < 1.0 and ci[1] < 1.0:
                verdict = "improved"
            elif median_ratio > 1.0 and ci[0] > 1.0:
                verdict = "regressed"
            else:
                verdict = "inconclusive"
            case_result["metrics"][metric] = {
                "direction": direction,
                "baseline_median": statistics.median(old),
                "candidate_median": statistics.median(new),
                "candidate_over_baseline_ratio": (
                    statistics.median(new) / statistics.median(old)
                    if statistics.median(old) != 0.0
                    else (1.0 if statistics.median(new) == 0.0 else None)
                ),
                "normalized_regression_ratio": median_ratio,
                "paired_median_ratio_bootstrap_95pct_ci": ci,
                "verdict": verdict,
            }
        report_cases[case_id] = case_result
    return {"schema_version": SCHEMA_VERSION, "paired": paired, "minimum_samples": minimum_samples, "passed": not failures, "failures": failures, "cases": report_cases}


def seed_from_name(case_id: str, metric: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{case_id}\0{metric}".encode()).digest()[:4], "big")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for side in ("baseline", "candidate"):
        command = sub.add_parser(side)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--fixtures", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    paired = sub.add_parser("paired", help="run alternating fresh-process baseline/candidate samples")
    paired.add_argument("--baseline-source", type=Path, required=True)
    paired.add_argument("--candidate-source", type=Path, required=True)
    paired.add_argument("--fixtures", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True, help="directory receiving baseline.json and candidate.json")
    paired.add_argument("--seed", type=int, default=1729)
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--minimum-samples", type=int, default=10)
    args = parser.parse_args()
    try:
        if args.operation in ("baseline", "candidate"):
            result = _profile(args.fixtures, args.source, args.operation)
            _write(args.output, result)
            print(json.dumps({"profile": str(args.output), "cases": sorted(result["cases"])}))
            return 0
        if args.operation == "paired":
            baseline, candidate = _paired(args.fixtures, args.baseline_source, args.candidate_source, args.seed)
            args.output.mkdir(parents=True, exist_ok=True)
            _write(args.output / "baseline.json", baseline)
            _write(args.output / "candidate.json", candidate)
            print(json.dumps({"baseline": str(args.output / 'baseline.json'), "candidate": str(args.output / 'candidate.json')}))
            return 0
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        result = compare_profiles(baseline, candidate, minimum_samples=args.minimum_samples)
        if args.output:
            _write(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"performance suite failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
