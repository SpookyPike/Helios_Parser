from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(os.environ.get("HELIOS_PERF_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from helios.services.derived.common import load_run_data
from helios.cache import clear_session_raw_data_cache
try:
    from helios.runtime import RunRevision
except ImportError:  # baseline revisions before explicit run tokens
    RunRevision = None


def _unique_array_bytes(value, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if isinstance(value, np.ndarray):
        root = value
        while isinstance(root.base, np.ndarray):
            root = root.base
        identity = id(root)
        if identity in seen:
            return 0
        seen.add(identity)
        return int(root.nbytes)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_unique_array_bytes(getattr(value, field.name), seen) for field in fields(value))
    if isinstance(value, dict):
        return sum(_unique_array_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_unique_array_bytes(item, seen) for item in value)
    return 0


def _rss_bytes() -> int:
    if sys.platform.startswith("linux"):
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError as exc:
        raise RuntimeError("peak RSS requires /proc/self/statm or benchmark-only psutil") from exc


def _required_checksums(dataset) -> dict[str, float]:
    names = (
        "time_s", "static_x_cm", "static_x_edge_cm", "zone_width_cm", "density_g_cm3",
        "velocity_cm_s", "temperature_e_ev", "temperature_i_ev", "electron_density_cm3",
        "mean_charge", "zone_region_id", "zone_material_index",
    )
    result = {}
    for name in names:
        values = np.asarray(getattr(dataset, name))
        result[name] = float(np.nansum(values, dtype=np.float64))
    return result


def benchmark(path: Path) -> dict[str, object]:
    clear_session_raw_data_cache(reason="demand_loading_benchmark_start")
    signature = inspect.signature(load_run_data)
    started_rss = _rss_bytes()
    peak = [started_rss]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], _rss_bytes())
            except (OSError, ValueError, IndexError):
                pass
            stop.wait(0.001)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        if "requested_fields" in signature.parameters:
            if RunRevision is None:
                raise RuntimeError("candidate loader exposes field requests but RunRevision is unavailable")
            dataset = load_run_data(path, revision=RunRevision.create(path), requested_fields=())
        else:
            dataset = load_run_data(path)
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        sampler.join(timeout=1.0)
    required = _required_checksums(dataset)
    optional_names = (
        "temperature_radiation_ev", "pressure_i_j_cm3", "pressure_e_j_cm3", "pressure_radiation_j_cm3",
        "pressure_plasma_j_cm3", "pressure_total_j_cm3", "artificial_viscosity_j_cm3", "ion_energy_j_g",
        "electron_energy_j_g", "radiation_energy_j_g", "kinetic_energy_j_g", "ion_heat_capacity_j_g_ev",
        "electron_heat_capacity_j_g_ev", "radiation_heating_j_g_s", "radiation_cooling_j_g_s",
        "radiation_sink_j_g_s", "radiation_net_heating_j_g_s", "laser_source_j_g", "laser_deposition_j_g_s",
    )
    arrays = [getattr(dataset, field.name) for field in fields(dataset) if isinstance(getattr(dataset, field.name), np.ndarray)]
    optional_bytes = _unique_array_bytes([getattr(dataset, name) for name in optional_names if isinstance(getattr(dataset, name, None), np.ndarray)])
    return {
        "elapsed_s": elapsed,
        "retained_array_bytes": _unique_array_bytes(dataset),
        "optional_array_bytes": int(optional_bytes),
        "peak_rss_delta_bytes": max(0, int(peak[0]) - started_rss),
        "array_count": len(arrays),
        "required_checksums": required,
        "loader_mode": "requested_fields_empty" if "requested_fields" in signature.parameters else "legacy_eager",
        "input_path": str(path.resolve()),
        "input_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure demand-driven derived loading and process RSS.")
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
