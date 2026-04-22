from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6 import QtWidgets

from helios_viewer.plots import FieldMapWidget


def _benchmark(label: str, fn, *args, iterations: int = 25, warmup: int = 5):
    for _ in range(max(0, warmup)):
        fn(*args)
    samples_ms: list[float] = []
    for _ in range(max(1, iterations)):
        started = time.perf_counter()
        fn(*args)
        samples_ms.append((time.perf_counter() - started) * 1.0e3)
    return {
        "label": label,
        "iterations": int(iterations),
        "median_ms": float(statistics.median(samples_ms)),
        "mean_ms": float(statistics.fmean(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
    }


def _old_field_map_bounds(array: np.ndarray):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return (float(np.min(finite)), float(np.max(finite)))


def _new_field_map_bounds(array: np.ndarray):
    return FieldMapWidget._finite_min_max(array)


def _old_mask_cache_hit(mask: np.ndarray):
    return mask.copy()


def _new_mask_cache_hit(mask: np.ndarray):
    view = mask.view()
    view.setflags(write=False)
    return view


def _render_field_map(widget: FieldMapWidget, data: np.ndarray, coordinate: np.ndarray, time_values: np.ndarray) -> None:
    widget.set_field_map(
        data,
        coordinate,
        time_values,
        orientation="coord_x_time_y",
        title="Microbench density",
        x_label="Coordinate",
        y_label="Time",
        colorbar_label="Density",
        auto_levels=True,
    )


def main() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    field_array = np.linspace(-5.0, 12.0, num=1024 * 1024, dtype=np.float64).reshape(1024, 1024)
    field_array[0, 0] = np.nan
    field_array[1, 1] = np.inf
    field_array[2, 2] = -np.inf
    mask = np.random.default_rng(0).random(1_000_000) > 0.35

    widget = FieldMapWidget()
    try:
        coordinate = np.linspace(0.0, 10.0, num=field_array.shape[1], dtype=np.float64)
        time_values = np.linspace(0.0, 5.0, num=field_array.shape[0], dtype=np.float64)

        results = {
            "field_map_bounds_old": _benchmark("field_map_bounds_old", _old_field_map_bounds, field_array),
            "field_map_bounds_new": _benchmark("field_map_bounds_new", _new_field_map_bounds, field_array),
            "field_map_render_current": _benchmark(
                "field_map_render_current",
                _render_field_map,
                widget,
                field_array,
                coordinate,
                time_values,
                iterations=10,
                warmup=2,
            ),
            "mask_cache_hit_old_copy": _benchmark("mask_cache_hit_old_copy", _old_mask_cache_hit, mask),
            "mask_cache_hit_new_view": _benchmark("mask_cache_hit_new_view", _new_mask_cache_hit, mask),
        }
    finally:
        widget.close()
        app.processEvents()

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
