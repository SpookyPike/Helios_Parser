from __future__ import annotations

from pathlib import Path
import json
import sys
import time

import numpy as np
from PySide6 import QtWidgets

ROOT = Path(__file__).resolve().parents[1]
HDF5_ROOT = ROOT
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result
from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry
from helios.services.derived.shock_tracking import track_shock_front
from helios.services.derived.spectroscopy import evaluate_spectroscopy
from helios.services.derived.transmission import evaluate_transmission
from helios.services.derived.xrd import estimate_xrd
from helios_analysis.controller import DerivedController


def _context_from_dataset(path: Path, dataset, *, snapshot_index: int | None = None) -> RunContext:
    return RunContext(
        path=path,
        summary=dict(dataset.summary),
        metadata=dict(dataset.metadata),
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(),
        static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(),
        zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
        has_dynamic_radius=dataset.radius_cm is not None,
        snapshot_index=min(88, max(0, dataset.time_s.size - 1)) if snapshot_index is None else int(snapshot_index),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="moving_radius" if dataset.radius_cm is not None else "zone",
        selected_region_ids=tuple(int(value) for value in np.asarray(dataset.regions["region_index"], dtype=np.int32)),
        selected_material_ids=tuple(int(abs(value)) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _wait_for(predicate, *, timeout_s: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout_s
    app = _ensure_app()
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("Timed out waiting for benchmark condition.")


def main() -> None:
    _ensure_app()
    path = HDF5_ROOT / "Cu_0166_stabilized.h5"
    dataset = load_run_data(path)
    context = _context_from_dataset(path, dataset)
    parameters = DerivedAnalysisParameters()

    started = time.perf_counter()
    full = compute_analysis_result(dataset, context, parameters=parameters, context_key=("bench", "full"))
    full_latency = time.perf_counter() - started

    started = time.perf_counter()
    partial = compute_analysis_result(
        dataset,
        context,
        parameters=parameters,
        context_key=("bench", "partial"),
        requested_time_plot_modules=frozenset(),
    )
    partial_latency = time.perf_counter() - started

    started = time.perf_counter()
    lazy_module_latency: dict[str, float] = {}
    lazy_loaded_modules: dict[str, list[str]] = {}
    for module_name in ("xrd", "plasmon", "transmission", "spectroscopy"):
        module_started = time.perf_counter()
        lazy_result = compute_analysis_result(
            dataset,
            context,
            parameters=parameters,
            context_key=("bench", module_name),
            requested_time_plot_modules=frozenset({module_name}),
            base_result=partial,
        )
        lazy_module_latency[module_name] = time.perf_counter() - module_started
        lazy_loaded_modules[module_name] = sorted(
            name
            for name in ("xrd", "plasmon", "transmission", "spectroscopy")
            if getattr(lazy_result, name).time_plots
        )

    geometry = build_analysis_geometry(
        dataset,
        context,
        observation_side=parameters.observation_side,
        line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
        line_of_sight_impact_parameter_cm=parameters.line_of_sight_impact_parameter_cm,
        profile_coordinate_mode=parameters.profile_coordinate_mode,
    )

    def _repeat_stats(fn) -> dict[str, object]:
        analysis_cache = AnalysisStateCache()
        fn(analysis_cache)
        stats_after_first = analysis_cache.stats()
        fn(analysis_cache)
        stats_after_second = analysis_cache.stats()
        return {
            "after_first": stats_after_first,
            "after_second": stats_after_second,
        }

    repeat_stats = {
        "shock": _repeat_stats(
            lambda cache: track_shock_front(
                dataset,
                context,
                parameters=parameters,
                geometry=geometry,
                analysis_cache=cache,
            )
        ),
        "xrd": _repeat_stats(
            lambda cache: estimate_xrd(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                photon_energy_kev=parameters.xrd_photon_energy_kev,
                initial_bragg_angle_deg=parameters.xrd_initial_bragg_angle_deg,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=True,
                analysis_cache=cache,
            )
        ),
        "plasmon": _repeat_stats(
            lambda cache: evaluate_plasmon_regime(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                photon_energy_kev=parameters.plasmon_photon_energy_kev,
                scattering_angle_deg=parameters.plasmon_scattering_angle_deg,
                adiabatic_index=parameters.plasmon_adiabatic_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=True,
                analysis_cache=cache,
            )
        ),
        "transmission": _repeat_stats(
            lambda cache: evaluate_transmission(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=True,
                analysis_cache=cache,
            )
        ),
        "spectroscopy": _repeat_stats(
            lambda cache: evaluate_spectroscopy(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                line_wavelength_nm=parameters.spectroscopy_line_wavelength_nm,
                parameters=parameters,
                geometry=geometry,
                include_time_plots=True,
                analysis_cache=cache,
            )
        ),
    }

    shared_cache = AnalysisStateCache()
    track_shock_front(
        dataset,
        context,
        parameters=parameters,
        geometry=geometry,
        analysis_cache=shared_cache,
    )
    estimate_xrd(
        dataset,
        context,
        snapshot_index=context.snapshot_index,
        photon_energy_kev=parameters.xrd_photon_energy_kev,
        initial_bragg_angle_deg=parameters.xrd_initial_bragg_angle_deg,
        parameters=parameters,
        geometry=geometry,
        include_time_plots=True,
        analysis_cache=shared_cache,
    )
    evaluate_plasmon_regime(
        dataset,
        context,
        snapshot_index=context.snapshot_index,
        photon_energy_kev=parameters.plasmon_photon_energy_kev,
        scattering_angle_deg=parameters.plasmon_scattering_angle_deg,
        adiabatic_index=parameters.plasmon_adiabatic_index,
        parameters=parameters,
        geometry=geometry,
        include_time_plots=True,
        analysis_cache=shared_cache,
    )
    evaluate_transmission(
        dataset,
        context,
        snapshot_index=context.snapshot_index,
        parameters=parameters,
        geometry=geometry,
        include_time_plots=True,
        analysis_cache=shared_cache,
    )
    evaluate_spectroscopy(
        dataset,
        context,
        snapshot_index=context.snapshot_index,
        line_wavelength_nm=parameters.spectroscopy_line_wavelength_nm,
        parameters=parameters,
        geometry=geometry,
        include_time_plots=True,
        analysis_cache=shared_cache,
    )

    controller = DerivedController()
    final_result = {"value": None}
    try:
        workspace = controller.widget()
        workspace.show()
        controller.analysis_ready.connect(lambda result: final_result.__setitem__("value", result))
        controller.set_active(True)
        controller.set_run_context(context)
        start_cancel = time.perf_counter()
        workspace.result_tabs.setCurrentIndex(workspace.result_tabs.indexOf(workspace.plasmon_tab))
        workspace.zone_upper_spin.setValue(max(10, min(200, context.n_zones)))
        workspace.zone_upper_spin.setValue(max(10, min(120, context.n_zones)))
        _wait_for(lambda: final_result["value"] is not None, timeout_s=20.0)
        cancellation_latency = time.perf_counter() - start_cancel
    finally:
        controller.shutdown()

    print(
        json.dumps(
            {
                "full_compute_s": full_latency,
                "partial_snapshot_only_s": partial_latency,
                "lazy_module_time_plots_s": lazy_module_latency,
                "cancellation_responsive_update_s": cancellation_latency,
                "full_loaded_modules": sorted(name for name in ("xrd", "plasmon", "transmission", "spectroscopy") if getattr(full, name).time_plots),
                "partial_loaded_modules": sorted(name for name in ("xrd", "plasmon", "transmission", "spectroscopy") if getattr(partial, name).time_plots),
                "lazy_loaded_modules": lazy_loaded_modules,
                "repeat_stats": repeat_stats,
                "shared_cache_full_pass_stats": shared_cache.stats(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
