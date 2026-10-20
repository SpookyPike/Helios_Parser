from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

import h5py
import numpy as np

ROOT = Path(os.environ.get("HELIOS_PERF_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from helios_parser import write_hdf5


def _rss_bytes() -> int:
    if sys.platform.startswith("linux"):
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError as exc:
        raise RuntimeError("process RSS measurement requires /proc/self/statm or benchmark-only psutil") from exc


def benchmark(source: Path, output: Path) -> dict[str, object]:
    baseline_rss = _rss_bytes()
    peak_rss = [baseline_rss]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                peak_rss[0] = max(peak_rss[0], _rss_bytes())
            except (OSError, ValueError, IndexError):
                pass
            stop.wait(0.001)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    first_write_s: float | None = None
    started = time.perf_counter()

    def progress(event) -> None:
        nonlocal first_write_s
        if first_write_s is None and event.stage == "snapshots" and int(event.current) >= 1:
            first_write_s = time.perf_counter() - started

    try:
        write_hdf5(source, output, overwrite=True, progress_callback=progress)
    finally:
        elapsed_s = time.perf_counter() - started
        stop.set()
        sampler.join(timeout=1.0)
    with h5py.File(output, "r") as handle:
        snapshots = int(handle["time/time"].shape[0])
        zones = int(handle["grid/zone_id"].shape[0])
        fields = handle["fields"]
        checksums = {
            name: float(np.nansum(fields[name][...], dtype=np.float64))
            for name in ("density", "laser_source", "laser_deposition")
            if name in fields
        }
    return {
        "elapsed_s": elapsed_s,
        "time_to_first_write_s": first_write_s,
        "peak_rss_delta_bytes": max(0, int(peak_rss[0]) - baseline_rss),
        "output_snapshots": snapshots,
        "output_zones": zones,
        "field_checksums": checksums,
        "source_bytes": source.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure BPF-primary conversion with a LOG companion.")
    parser.add_argument("--source", type=Path, required=True, help="BPF; same-stem LOG is auto-discovered")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(benchmark(args.source, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
