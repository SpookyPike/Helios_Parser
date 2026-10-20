from __future__ import annotations

import argparse
import hashlib
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from helios.cache import AnalyzerCacheSet, get_session_raw_data_cache, retire_run_revision
from helios.runtime import RunContext, RunRevision
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.common import load_run_data
from helios.cache import estimate_nbytes
from helios.cache import clear_session_raw_data_cache


def _rss_bytes() -> int | None:
    """Read current resident set size using a native OS interface."""

    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "darwin":
        try:
            import subprocess

            value = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
            ).strip()
            return int(value) * 1024
        except (OSError, ValueError):
            return None
    try:
        import psutil  # optional benchmark-only dependency

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity() -> dict[str, str]:
    source_paths = (
        Path("src/helios/cache.py"),
        Path("src/helios_analysis/controller.py"),
        Path("scripts/bench_cache_retention.py"),
    )
    digest = hashlib.sha256()
    for relative_path in source_paths:
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative_path).read_bytes())
        digest.update(b"\0")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {"git_revision": revision, "working_tree_source_sha256": digest.hexdigest()}


def _total_memory_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    return None


def benchmark(path: Path, cycles: int) -> dict[str, object]:
    if _rss_bytes() is None:
        raise RuntimeError("current RSS measurement is unavailable on this platform; install psutil for a supported backend")
    clear_session_raw_data_cache(reason="retention_benchmark_start")
    before = _rss_bytes()
    coordinator = get_session_raw_data_cache().coordinator
    cache_layers = AnalyzerCacheSet(coordinator=coordinator)
    dataset_cache = cache_layers.raw_data_cache.bucket("retention_benchmark_datasets", max_items=3)
    analysis_cache = cache_layers.derived_cache.bucket("retention_benchmark_analysis", max_items=3)
    samples: list[dict[str, int | float]] = []
    for cycle in range(int(cycles)):
        start = time.perf_counter()
        revision = RunRevision.create(path)
        dataset_key = (revision.cache_key, ())
        dataset = load_run_data(path, revision=revision, requested_fields=())
        context = RunContext(
            path=path.resolve(),
            revision=revision,
            summary=dict(dataset.summary),
            metadata=dict(dataset.metadata),
            fields=tuple(dataset.field_capabilities.available_fields),
            diagnostics=(),
            time_values=dataset.time_s,
            static_x_values=dataset.static_x_cm,
            zone_region_id=dataset.zone_region_id,
            zone_material_index=dataset.zone_material_index,
            has_dynamic_radius=dataset.radius_cm is not None,
        )
        analysis_result = compute_analysis_result(
            dataset,
            context,
            parameters=DerivedAnalysisParameters(),
            context_key=(revision.cache_key, int(dataset.time_s.size), int(dataset.static_x_cm.size)),
            requested_time_plot_modules=frozenset(),
        )
        elapsed = time.perf_counter() - start
        loaded_rss = _rss_bytes()
        active_bytes = estimate_nbytes(dataset)
        active_analysis_bytes = estimate_nbytes(analysis_result)
        dataset_cache[dataset_key] = dataset
        analysis_cache[analysis_result.context_key] = analysis_result
        retained_before_retirement = estimate_nbytes(tuple(dataset_cache.values()))
        retained_analysis_before_retirement = estimate_nbytes(tuple(analysis_cache.values()))
        del dataset
        del context
        del analysis_result
        retired_entries = retire_run_revision(revision)
        gc.collect()
        retired_rss = _rss_bytes()
        cache_stats = coordinator.stats()
        aggregate_bytes = cache_stats.bytes_used
        samples.append(
            {
                "load_seconds": elapsed,
                "active_dataset_bytes": active_bytes,
                "active_analysis_result_bytes": active_analysis_bytes,
                "retired_cache_entries": retired_entries,
                "retained_dataset_cache_bytes_before_retirement": retained_before_retirement,
                "retained_dataset_cache_bytes": estimate_nbytes(tuple(dataset_cache.values())),
                "retained_analysis_cache_bytes_before_retirement": retained_analysis_before_retirement,
                "retained_analysis_cache_bytes": estimate_nbytes(tuple(analysis_cache.values())),
                "loaded_rss_bytes": int(loaded_rss or 0),
                "post_retirement_rss_bytes": int(retired_rss or 0),
                "aggregate_cache_bytes": int(aggregate_bytes),
                "aggregate_evictions": int(cache_stats.evictions),
            }
        )
    first_eviction = next(
        (index for index, sample in enumerate(samples) if int(sample["aggregate_evictions"]) > 0),
        None,
    )
    post_saturation = samples[first_eviction:] if first_eviction is not None else []
    post_saturation_rss = [int(sample["post_retirement_rss_bytes"]) for sample in post_saturation]
    post_retirement_rss = [int(sample["post_retirement_rss_bytes"]) for sample in samples]
    post_retirement_cache_bytes = [int(sample["aggregate_cache_bytes"]) for sample in samples]
    if any(int(sample["retired_cache_entries"]) <= 0 for sample in samples):
        raise RuntimeError("revision retirement removed no entries; benchmark did not exercise retained run data")
    return {
        "path": str(path.resolve()),
        "source_size_bytes": path.stat().st_size,
        "source_sha256": _sha256(path),
        "source_identity": _source_identity(),
        "cycles": int(cycles),
        "environment": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "total_memory_bytes": _total_memory_bytes(),
            "python": sys.version,
        },
        "rss_backend": "procfs" if sys.platform.startswith("linux") else "ps",
        "rss_before_bytes": int(before or 0),
        "samples": samples,
        "median_load_seconds": statistics.median(float(sample["load_seconds"]) for sample in samples),
        "first_eviction_cycle": None if first_eviction is None else first_eviction + 1,
        "post_saturation_rss_range_bytes": max(post_saturation_rss) - min(post_saturation_rss) if post_saturation_rss else None,
        "post_retirement_rss_range_bytes": max(post_retirement_rss) - min(post_retirement_rss) if post_retirement_rss else None,
        "post_retirement_cache_bytes_range": max(post_retirement_cache_bytes) - min(post_retirement_cache_bytes) if post_retirement_cache_bytes else None,
        "aggregate_cache_limit_bytes": coordinator.max_bytes,
        "retained_dataset_cache_bytes_after_final_cycle": int(samples[-1]["retained_dataset_cache_bytes"]),
        "aggregate_cache_bytes_after_final_cycle": int(samples[-1]["aggregate_cache_bytes"]),
        "retirement_modeled": True,
        "interpretation": (
            "Each cycle calls the same revision-retirement API used when Derived switches or closes a run. Matching entries are removed across all registered cache owners before RSS is sampled. "
            "Active task/view references are outside this cache-only measurement, and RSS can remain high because native allocators retain pages."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure retained cache bytes and process RSS across run retirement cycles.")
    parser.add_argument("run", type=Path, help="Representative stabilized HDF5 run")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cycles < 2:
        parser.error("--cycles must be at least 2")
    try:
        report = benchmark(args.run, args.cycles)
    except Exception as exc:
        parser.exit(2, f"memory benchmark unavailable or failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
