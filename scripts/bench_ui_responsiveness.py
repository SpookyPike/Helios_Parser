#!/usr/bin/env python3
"""Measure native viewer snapshot settling latency on a large HDF5 run.

Example: QT_QPA_PLATFORM=wayland PYTHONPATH=src .venv/bin/python \
    scripts/bench_ui_responsiveness.py --code-root . --run path/to/run.h5
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.code_root.resolve()
    sys.path.insert(0, str(root / "src"))

    from PySide6 import QtCore, QtWidgets
    import h5py
    import numpy as np
    from helios_viewer.main_window import HeliosViewerMainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["helios-ui-benchmark"])
    app.setOrganizationName("HeliosBenchmark")
    app.setApplicationName("HeliosBenchmark")
    window = HeliosViewerMainWindow()
    window.resize(1280, 900)
    window.show()
    window.load_file(str(args.run.resolve()))

    def pump_until(predicate, timeout_s: float = 90.0) -> bool:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
            if predicate():
                return True
            time.sleep(0.001)
        return False

    if not pump_until(lambda: window.run_payload is not None and window.current_field_payload is not None):
        raise TimeoutError("viewer did not finish loading run and default field")
    pump_until(lambda: window.field_map_widget.last_image_shape is not None, timeout_s=15.0)

    gaps: list[float] = []
    last = time.perf_counter()
    timer = QtCore.QTimer()
    timer.setInterval(1)

    def heartbeat() -> None:
        nonlocal last
        now = time.perf_counter()
        gaps.append((now - last) * 1000.0)
        last = now

    timer.timeout.connect(heartbeat)
    timer.start()
    max_index = int(window.snapshot_slider.maximum())
    if max_index < 1:
        raise RuntimeError(f"fixture has too few snapshots: max slider index={max_index}")
    latencies: list[float] = []
    correct_renders = 0
    tab_switches = 0
    max_pending_requests = 0
    pending_request_metric_available = hasattr(window.controller, "_pending_requests")
    start_render_count = int(window.lineout_plot.set_curves_call_count)

    for i in range(args.warmup + args.samples):
        target = (i * 37 + 11) % (max_index + 1)
        if i % 2 == 0:
            window.plot_tabs.setCurrentWidget(window.diagnostic_plot)
            tab_switches += 1
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        window.plot_tabs.setCurrentWidget(window.lineout_plot)
        tab_switches += 1
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        started = time.perf_counter()
        window.snapshot_slider.setValue(target)

        def settled() -> bool:
            nonlocal max_pending_requests
            pending = getattr(window.controller, "_pending_requests", None)
            if pending is not None:
                max_pending_requests = max(max_pending_requests, len(pending))
            return (
                window.active_snapshot_index() == target
                and f"snapshot {target} |" in window.lineout_plot.current_title
            )

        if not pump_until(settled, timeout_s=15.0):
            raise TimeoutError(f"snapshot {target} did not render correctly")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if i >= args.warmup:
            latencies.append(elapsed_ms)
            correct_renders += 1

    timer.stop()
    result = {
        "revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "code_root": str(root),
        "run": str(args.run.resolve()),
        "run_bytes": args.run.stat().st_size,
        "run_sha256": hashlib.sha256(args.run.read_bytes()).hexdigest(),
        "platform": os.environ.get("QT_QPA_PLATFORM", "default"),
        "system": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "available_memory_bytes": __import__("psutil").virtual_memory().available if importlib.util.find_spec("psutil") else None,
        "python": sys.version,
        "qt_version": QtCore.qVersion(),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "source_sha256": hashlib.sha256(b"\n".join(
            p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes()
            for p in sorted((root / "src").rglob("*.py"))
        )).hexdigest(),
        "display": {k: os.environ.get(k) for k in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE")},
        "samples": len(latencies),
        "warmup": args.warmup,
        "settling_latency_ms": {
            "p50": statistics.median(latencies),
            "p95_nearest_rank": percentile(latencies, 0.95),
            "max": max(latencies),
            "samples": latencies,
        },
        "event_loop_gap_ms": {
            "samples": len(gaps),
            "p50": statistics.median(gaps) if gaps else None,
            "p95_nearest_rank": percentile(gaps, 0.95) if gaps else None,
            "max": max(gaps) if gaps else None,
        },
        "correct_latest_snapshot_renders": correct_renders,
        "tab_switches": tab_switches,
        "pending_request_metric_available": pending_request_metric_available,
        "max_pending_requests_observed_while_waiting": max_pending_requests if pending_request_metric_available else None,
        "line_render_count_delta": int(window.lineout_plot.set_curves_call_count) - start_render_count,
        "final_snapshot_index": window.active_snapshot_index(),
        "final_title": window.lineout_plot.current_title,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    window.close()
    window.controller.shutdown()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
