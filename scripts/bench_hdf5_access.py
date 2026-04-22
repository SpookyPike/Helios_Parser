from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helios_parser import write_hdf5


def _measure_snapshot_reads(dataset: h5py.Dataset, snapshot_index: int, *, repeats: int = 32) -> dict[str, float]:
    times: list[float] = []
    row_bytes = int(dataset.shape[1]) * np.dtype(dataset.dtype).itemsize
    for _ in range(repeats):
        start = time.perf_counter()
        values = dataset[snapshot_index, :]
        _ = float(np.nansum(values[:1])) if values.size else 0.0
        times.append(time.perf_counter() - start)
    total_elapsed = sum(times)
    return {
        "median_latency_ms": statistics.median(times) * 1000.0,
        "mean_latency_ms": statistics.fmean(times) * 1000.0,
        "throughput_mb_s": ((row_bytes * repeats) / max(total_elapsed, 1.0e-12)) / (1024.0 * 1024.0),
    }


def _measure_zone_reads(dataset: h5py.Dataset, zone_index: int, *, repeats: int = 32) -> dict[str, float]:
    times: list[float] = []
    col_bytes = int(dataset.shape[0]) * np.dtype(dataset.dtype).itemsize
    for _ in range(repeats):
        start = time.perf_counter()
        values = dataset[:, zone_index]
        _ = float(np.nansum(values[:1])) if values.size else 0.0
        times.append(time.perf_counter() - start)
    total_elapsed = sum(times)
    return {
        "median_latency_ms": statistics.median(times) * 1000.0,
        "mean_latency_ms": statistics.fmean(times) * 1000.0,
        "throughput_mb_s": ((col_bytes * repeats) / max(total_elapsed, 1.0e-12)) / (1024.0 * 1024.0),
    }


def _measure_slider_scrub(
    dataset: h5py.Dataset,
    *,
    start_snapshot: int,
    count: int = 64,
    repeats: int = 4,
) -> dict[str, float]:
    count = max(1, min(int(count), int(dataset.shape[0]) - int(start_snapshot)))
    times: list[float] = []
    bytes_per_row = int(dataset.shape[1]) * np.dtype(dataset.dtype).itemsize
    for _ in range(repeats):
        start = time.perf_counter()
        checksum = 0.0
        for index in range(start_snapshot, start_snapshot + count):
            row = dataset[index, :]
            checksum += float(np.nansum(row[:1])) if row.size else 0.0
        _ = checksum
        times.append(time.perf_counter() - start)
    total_elapsed = sum(times)
    return {
        "count": int(count),
        "median_total_ms": statistics.median(times) * 1000.0,
        "mean_total_ms": statistics.fmean(times) * 1000.0,
        "throughput_mb_s": ((bytes_per_row * count * repeats) / max(total_elapsed, 1.0e-12)) / (1024.0 * 1024.0),
    }


def _benchmark_file(path: Path, *, field_name: str, snapshot_index: int | None = None, zone_index: int | None = None) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        dataset = handle["fields"][field_name]
        n_snapshots, n_zones = dataset.shape
        snapshot = int(np.clip(0 if snapshot_index is None else snapshot_index, 0, n_snapshots - 1))
        zone = int(np.clip(0 if zone_index is None else zone_index, 0, n_zones - 1))
        scrub_start = int(np.clip(snapshot, 0, max(0, n_snapshots - 1)))
        return {
            "path": str(path),
            "shape": [int(n_snapshots), int(n_zones)],
            "dtype": str(dataset.dtype),
            "chunks": None if dataset.chunks is None else [int(v) for v in dataset.chunks],
            "snapshot_access": _measure_snapshot_reads(dataset, snapshot),
            "zone_access": _measure_zone_reads(dataset, zone),
            "slider_scrub": _measure_slider_scrub(dataset, start_snapshot=scrub_start),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark HELIOS HDF5 access patterns.")
    parser.add_argument("--baseline-h5", type=Path, default=ROOT / "Cu_0166_stabilized.h5")
    parser.add_argument("--source-log", type=Path, default=ROOT / "Cu_0166.log")
    parser.add_argument("--field", default="density")
    parser.add_argument("--snapshot", type=int, default=None)
    parser.add_argument("--zone", type=int, default=None)
    args = parser.parse_args()

    report: dict[str, object] = {
        "baseline": _benchmark_file(args.baseline_h5, field_name=args.field, snapshot_index=args.snapshot, zone_index=args.zone)
    }
    if args.source_log.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = Path(tmpdir) / f"{args.source_log.stem}_chunked.h5"
            write_hdf5(args.source_log, candidate_path, overwrite=True)
            report["candidate"] = _benchmark_file(
                candidate_path,
                field_name=args.field,
                snapshot_index=args.snapshot,
                zone_index=args.zone,
            )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
