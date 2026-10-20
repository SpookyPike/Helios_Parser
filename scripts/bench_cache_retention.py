from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from helios.cache import AnalyzerCacheSet, get_session_raw_data_cache
from helios.runtime import RunRevision
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


def benchmark(path: Path, cycles: int) -> dict[str, object]:
    if _rss_bytes() is None:
        raise RuntimeError("current RSS measurement is unavailable on this platform; install psutil for a supported backend")
    clear_session_raw_data_cache(reason="retention_benchmark_start")
    before = _rss_bytes()
    coordinator = get_session_raw_data_cache().coordinator
    cache_layers = AnalyzerCacheSet(coordinator=coordinator)
    dataset_cache = cache_layers.raw_data_cache.bucket("retention_benchmark_datasets", max_items=3)
    samples: list[dict[str, int | float]] = []
    for cycle in range(int(cycles)):
        start = time.perf_counter()
        revision = RunRevision.create(path)
        dataset_key = (revision.cache_key, ())
        dataset = load_run_data(path, revision=revision, requested_fields=())
        elapsed = time.perf_counter() - start
        loaded_rss = _rss_bytes()
        active_bytes = estimate_nbytes(dataset)
        dataset_cache[dataset_key] = dataset
        del dataset
        gc.collect()
        retired_rss = _rss_bytes()
        aggregate_bytes = coordinator.stats().bytes_used
        samples.append(
            {
                "load_seconds": elapsed,
                "active_dataset_bytes": active_bytes,
                "retained_dataset_cache_bytes": estimate_nbytes(tuple(dataset_cache.values())),
                "loaded_rss_bytes": int(loaded_rss or 0),
                "retired_rss_bytes": int(retired_rss or 0),
                "aggregate_cache_bytes": int(aggregate_bytes),
            }
        )
    retired_values = [int(sample["retired_rss_bytes"]) for sample in samples]
    tail_size = min(len(retired_values), max(5, len(retired_values) // 3))
    retired_tail = retired_values[-tail_size:]
    return {
        "path": str(path.resolve()),
        "source_size_bytes": path.stat().st_size,
        "cycles": int(cycles),
        "os": platform.platform(),
        "python": sys.version,
        "rss_backend": "procfs" if sys.platform.startswith("linux") else "ps",
        "rss_before_bytes": int(before or 0),
        "samples": samples,
        "median_load_seconds": statistics.median(float(sample["load_seconds"]) for sample in samples),
        "retired_rss_tail_range_bytes": max(retired_tail) - min(retired_tail) if retired_tail else 0,
        "aggregate_cache_limit_bytes": coordinator.max_bytes,
        "retained_dataset_cache_bytes_after_final_cycle": int(samples[-1]["retained_dataset_cache_bytes"]),
        "aggregate_cache_bytes_after_final_cycle": int(samples[-1]["aggregate_cache_bytes"]),
        "interpretation": "Three dataset entries remain cached to match DerivedController; RSS can remain high because native allocators retain pages.",
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
