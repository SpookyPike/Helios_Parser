from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helios.cache import clear_session_raw_data_cache
from helios.instrumentation import reset_metrics, snapshot_metrics
from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.common import load_run_data, publish_field_payload, publish_open_run_payload, shared_raw_cache_stats
from helios_parser import HeliosRun
from helios_viewer.models import OpenRunPayload


def _context_from_run(path: Path) -> RunContext:
    with HeliosRun(path) as run:
        payload = OpenRunPayload(
            run_generation=0,
            path=path,
            summary=run.summary(),
            metadata=run.get_metadata(),
            fields=run.list_fields(),
            field_units={name: run.get_field_unit(name) for name in run.list_fields()},
            diagnostics=run.list_diagnostics(),
            diagnostic_units={name: run.get_diagnostic_unit(name) for name in run.list_diagnostics()},
            regions=run.get_regions(),
            materials=run.get_materials(),
            time=run.get_time(),
            time_unit=run.get_time_unit(),
            static_x=run.get_static_coordinate(location="center"),
            static_x_edges=run.get_static_coordinate(location="edge"),
            static_x_unit=run.get_grid_unit("x"),
            zone_region_id=run.get_grid("zone_region_id"),
            zone_material_index=run.get_grid("zone_material_index"),
            has_dynamic_radius="radius" in run.list_fields(),
            radius_unit=run.get_field_unit("radius") if "radius" in run.list_fields() else "",
        )
    return RunContext.from_payload(payload)


def main() -> int:
    path = ROOT / "Cu_0166_stabilized.h5"
    context = _context_from_run(path)

    clear_session_raw_data_cache(reason="wave5_benchmark_start")
    reset_metrics()
    start = time.perf_counter()
    cold_dataset = load_run_data(path)
    cold_load_s = time.perf_counter() - start

    start = time.perf_counter()
    warm_dataset = load_run_data(path)
    warm_load_s = time.perf_counter() - start

    start = time.perf_counter()
    compute_analysis_result(
        warm_dataset,
        context,
        parameters=DerivedAnalysisParameters(),
        context_key=("wave5", "warm"),
    )
    derived_compute_s = time.perf_counter() - start
    baseline_metrics = snapshot_metrics()

    clear_session_raw_data_cache(reason="wave5_viewer_publish")
    reset_metrics()
    with HeliosRun(path) as run:
        fields = run.list_fields()
        payload = OpenRunPayload(
            run_generation=0,
            path=path,
            summary=run.summary(),
            metadata=run.get_metadata(),
            fields=fields,
            field_units={name: run.get_field_unit(name) for name in fields},
            diagnostics=run.list_diagnostics(),
            diagnostic_units={name: run.get_diagnostic_unit(name) for name in run.list_diagnostics()},
            regions=run.get_regions(),
            materials=run.get_materials(),
            time=run.get_time(),
            time_unit=run.get_time_unit(),
            static_x=run.get_static_coordinate(location="center"),
            static_x_edges=run.get_static_coordinate(location="edge"),
            static_x_unit=run.get_grid_unit("x"),
            zone_region_id=run.get_grid("zone_region_id"),
            zone_material_index=run.get_grid("zone_material_index"),
            has_dynamic_radius="radius" in fields,
            radius_unit=run.get_field_unit("radius") if "radius" in fields else "",
        )
        publish_open_run_payload(
            path,
            summary=payload.summary,
            metadata=payload.metadata,
            regions=payload.regions,
            materials=payload.materials,
            fields=payload.fields,
            diagnostics=payload.diagnostics,
            time_values=payload.time,
            static_x_center=payload.static_x,
            static_x_edge=payload.static_x_edges,
            zone_region_id=payload.zone_region_id,
            zone_material_index=payload.zone_material_index,
            has_dynamic_radius=payload.has_dynamic_radius,
        )
        for field_name in (
            "density",
            "velocity",
            "temperature_e",
            "temperature_i",
            "temperature_radiation",
            "electron_density",
            "mean_charge",
            "zone_width",
            "radius",
        ):
            if field_name not in fields:
                continue
            data = run.get_dynamic_coordinate(location="center") if field_name == "radius" and run.has_dynamic_coordinate() else run.get_field(field_name)
            edge_data = run.get_dynamic_coordinate(location="edge") if field_name == "radius" and run.has_dynamic_coordinate() else None
            publish_field_payload(path, field_name=field_name, data=data, edge_data=edge_data)

    start = time.perf_counter()
    reused_dataset = load_run_data(path)
    viewer_warm_load_s = time.perf_counter() - start
    reuse_metrics = snapshot_metrics()

    print(
        json.dumps(
            {
                "path": str(path),
                "cold_load_s": cold_load_s,
                "warm_load_s": warm_load_s,
                "derived_compute_s": derived_compute_s,
                "viewer_published_warm_load_s": viewer_warm_load_s,
                "baseline_metrics": baseline_metrics,
                "reuse_metrics": reuse_metrics,
                "shared_raw_cache_stats": shared_raw_cache_stats(),
                "dataset_shapes": {
                    "density": tuple(int(v) for v in cold_dataset.density_g_cm3.shape),
                    "time": tuple(int(v) for v in cold_dataset.time_s.shape),
                },
                "reuse_same_density_object": reused_dataset.density_g_cm3 is not None,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
