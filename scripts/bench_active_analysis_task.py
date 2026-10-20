from __future__ import annotations

"""Measure process RSS while the real asynchronous derived-task worker runs."""

import argparse
import hashlib
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PySide6 import QtCore

from helios.cache import estimate_nbytes
from helios.cancellation import CancellationToken
from helios.runtime import RunContext, RunRevision
from helios.services.derived import DerivedAnalysisParameters, compute_analysis_result, load_run_data
from helios.tasks import AnalysisTaskManager, AnalysisTaskResult


def rss_bytes() -> int:
    if sys.platform.startswith("linux"):
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    import psutil  # benchmark-only dependency on non-Linux platforms

    return int(psutil.Process().memory_info().rss)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(path: Path, sample_interval_ms: int) -> dict[str, object]:
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    baseline_rss = rss_bytes()
    peaks = [baseline_rss]
    stop_sampling = threading.Event()

    def sample() -> None:
        while not stop_sampling.wait(sample_interval_ms / 1000):
            try:
                peaks.append(rss_bytes())
            except (OSError, ValueError):
                pass

    sampler = threading.Thread(target=sample, name="rss-sampler", daemon=True)
    sampler.start()
    manager = AnalysisTaskManager()
    revision = RunRevision.create(path)
    result_holder: list[AnalysisTaskResult] = []
    errors: list[str] = []
    loop = QtCore.QEventLoop()
    manager.result_ready.connect(lambda result: (result_holder.append(result), loop.quit()))
    manager.task_failed.connect(lambda _task, message: (errors.append(message), loop.quit()))

    def compute() -> object:
        dataset = load_run_data(path, revision=revision, requested_fields=(
            "pressure_i", "pressure_e", "pressure_radiation", "total_pressure_j_cm3", "artificial_viscosity"
        ))
        context = RunContext(
            path=path.resolve(), revision=revision, summary=dict(dataset.summary), metadata=dict(dataset.metadata),
            fields=tuple(dataset.field_capabilities.available_fields), diagnostics=(),
            time_values=dataset.time_s, static_x_values=dataset.static_x_cm,
            zone_region_id=dataset.zone_region_id, zone_material_index=dataset.zone_material_index,
            has_dynamic_radius=dataset.radius_cm is not None,
        )
        return compute_analysis_result(
            dataset, context, parameters=DerivedAnalysisParameters(),
            context_key=(revision.cache_key, "active-task-memory"),
            requested_time_plot_modules=frozenset(), requested_modules=frozenset(),
            include_wavefront=False, progress_check=CancellationToken().check_cancelled,
        )

    started = time.perf_counter()
    manager.submit(context_key=(revision.cache_key,), fn=compute)
    timeout = QtCore.QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(300_000)
    loop.exec()
    elapsed = time.perf_counter() - started
    stop_sampling.set()
    sampler.join(timeout=2)
    peaks.append(rss_bytes())
    if errors:
        raise RuntimeError(errors[0])
    if not result_holder:
        raise TimeoutError("asynchronous analysis task did not finish within 300 seconds")
    result = result_holder[0].result
    peak_rss = max(peaks)
    report = {
        "path": str(path.resolve()), "source_size_bytes": path.stat().st_size,
        "source_sha256": sha256(path),
        "revision": revision.cache_key, "elapsed_seconds": elapsed,
        "rss_backend": "procfs", "sample_interval_ms": sample_interval_ms,
        "rss_before_bytes": baseline_rss, "peak_task_rss_bytes": peak_rss,
        "peak_task_rss_delta_bytes": peak_rss - baseline_rss,
        "rss_after_result_bytes": rss_bytes(),
        "result_retained_bytes": estimate_nbytes(result),
        "task_manager_stats": manager.stats().__dict__ if hasattr(manager.stats(), "__dict__") else {
            name: getattr(manager.stats(), name) for name in manager.stats().__slots__
        },
        "environment": {"os": platform.platform(), "python": sys.version, "numpy": np.__version__},
        "scope": "real AnalysisTaskManager worker: HDF5 dataset load plus shock-only derived calculation; RSS includes interpreter, Qt, HDF5, NumPy and task working allocations; result bytes are reported separately.",
    }
    result_holder.clear()
    del result
    gc.collect()
    report["rss_after_result_release_bytes"] = rss_bytes()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-interval-ms", type=int, default=5)
    args = parser.parse_args()
    report = run(args.run, args.sample_interval_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
