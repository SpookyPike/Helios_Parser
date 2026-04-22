from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import psutil
except ImportError:  # pragma: no cover - benchmark fallback
    psutil = None

from helios_parser import HeliosParser, write_hdf5


def _sample_peak_rss(stop_event: threading.Event) -> dict[str, float | None]:
    process = psutil.Process() if psutil is not None else None
    baseline_rss = process.memory_info().rss if process is not None else None
    peak_rss = baseline_rss
    while not stop_event.is_set():
        if process is not None:
            peak_rss = max(int(peak_rss or 0), int(process.memory_info().rss))
        time.sleep(0.005)
    peak_value = None if peak_rss is None else float(peak_rss)
    baseline_value = None if baseline_rss is None else float(baseline_rss)
    return {
        "baseline_rss_bytes": baseline_value,
        "peak_rss_bytes": peak_value,
        "peak_rss_delta_bytes": None if peak_value is None or baseline_value is None else float(peak_value - baseline_value),
    }


def _measure_with_peak(fn):
    stop_event = threading.Event()
    peak_holder: dict[str, float | None] = {
        "baseline_rss_bytes": None,
        "peak_rss_bytes": None,
        "peak_rss_delta_bytes": None,
    }

    def _sampler() -> None:
        peak_holder.update(_sample_peak_rss(stop_event))

    thread = threading.Thread(target=_sampler, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        result = fn()
    finally:
        elapsed_s = time.perf_counter() - started
        stop_event.set()
        thread.join(timeout=1.0)
    return result, elapsed_s, dict(peak_holder)


def main() -> None:
    parser = HeliosParser()
    source = ROOT / "10ns+10Si+60Al+15Si+4.27TW.log"

    def _old_full_materialization():
        with parser.open_document(source) as document:
            header = document.inspect()
            iterator = document.iter_snapshots(header=header)
            first_snapshot = next(iterator)
            return {"first_time_s": float(first_snapshot.time)}

    def _new_streaming_first_snapshot():
        with parser.open_document(source) as document:
            header = document.inspect()
            iterator = document.iter_snapshots_streaming(header=header)
            first_snapshot = next(iterator)
            return {"first_time_s": float(first_snapshot.time)}

    def _full_parse_materialization():
        simulation = parser.parse(source)
        return {"snapshot_count": int(simulation.time["time"].size)}

    old_result, old_elapsed_s, old_rss = _measure_with_peak(_old_full_materialization)
    new_result, new_elapsed_s, new_rss = _measure_with_peak(_new_streaming_first_snapshot)
    full_result, full_elapsed_s, full_rss = _measure_with_peak(_full_parse_materialization)

    streaming_metrics: dict[str, float | None] = {"time_to_first_write_s": None}

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "streaming_bench.h5"

        def _write_streaming():
            first_snapshot_write_s: float | None = None

            def _progress(event) -> None:
                nonlocal first_snapshot_write_s
                if first_snapshot_write_s is None and event.stage == "snapshots" and int(event.current) >= 1:
                    first_snapshot_write_s = time.perf_counter()

            started = time.perf_counter()
            write_hdf5(source, output, overwrite=True, parser=parser, progress_callback=_progress)
            if first_snapshot_write_s is not None:
                streaming_metrics["time_to_first_write_s"] = first_snapshot_write_s - started
            return {}

        _, streaming_elapsed_s, streaming_rss = _measure_with_peak(_write_streaming)

    result = {
        "source": source.name,
        "old_full_materialization": {
            "time_to_first_snapshot_s": old_elapsed_s,
            **old_rss,
            **old_result,
        },
        "new_streaming_iterator": {
            "time_to_first_snapshot_s": new_elapsed_s,
            **new_rss,
            **new_result,
        },
        "full_parse_materialization": {
            "elapsed_s": full_elapsed_s,
            **full_rss,
            **full_result,
        },
        "streaming_hdf5_write": {
            "time_to_first_write_s": streaming_metrics["time_to_first_write_s"],
            "total_elapsed_s": streaming_elapsed_s,
            **streaming_rss,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    main()
