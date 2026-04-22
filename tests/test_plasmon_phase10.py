from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters, compute_analysis_result, refresh_analysis_result_for_snapshot
from helios.services.derived.common import load_run_data
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry
from helios.services.derived.plasmon import evaluate_plasmon_regime
from test_plasmon_phase9 import _compute as _plasmon_compute
from test_plasmon_phase3 import _synthetic_dataset


def _context_from_dataset(path: Path, dataset, *, snapshot_index: int) -> RunContext:
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
        snapshot_index=int(snapshot_index),
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="zone",
        selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
        selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )



class PlasmonPhase10Tests(unittest.TestCase):
    def test_auto_best_reports_mixed_los_backend_summary(self) -> None:
        dataset, context = _synthetic_dataset()
        te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
        ne = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
        te[:, :3] = 0.3
        ne[:, :3] = 1.8e23
        te[:, 3:] = 30.0
        ne[:, 3:] = 1.0e21
        from dataclasses import replace
        dataset = replace(dataset, temperature_e_ev=te, electron_density_cm3=ne)
        result = _plasmon_compute(
            dataset,
            context,
            plasmon_model="auto_best",
            plasmon_execution_mode="quicklook",
            plasmon_integration_mode="los_integrated",
            plasmon_energy_window_ev=24.0,
            plasmon_energy_points=401,
            plasmon_instrument_fwhm_ev=0.4,
            plasmon_cluster_log_ne_tol=0.25,
            plasmon_cluster_log_te_tol=0.25,
            plasmon_cluster_z_tol=0.25,
        )
        self.assertEqual(result.requested_model_name, "auto_best")
        self.assertEqual(result.model_name, "auto_best")
        self.assertEqual(result.response_backend, "mixed")
        self.assertTrue(any(token in result.auto_model_summary for token in ("mermin", "rpa_static_lfc", "mermin_static_lfc")))

    def test_lindhard_snapshot_refresh_updates_snapshot_local_profiles(self) -> None:
        path = _test_bootstrap.example_data_path("Cu_0166_stabilized.h5")
        dataset = load_run_data(path)
        base_context = _context_from_dataset(path, dataset, snapshot_index=20)
        params = DerivedAnalysisParameters(
            plasmon_model="auto_best",
            plasmon_execution_mode="quicklook",
            plasmon_integration_mode="los_integrated",
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=301,
            plasmon_instrument_fwhm_ev=0.8,
        )
        base = compute_analysis_result(dataset, base_context, parameters=params, context_key=("plasmon10", "base"), requested_time_plot_modules=frozenset())
        updated_context = base_context.copy()
        updated_context.set_snapshot_index(40)
        refreshed = refresh_analysis_result_for_snapshot(
            dataset,
            updated_context,
            parameters=params,
            context_key=("plasmon10", "refresh"),
            base_result=base,
        )
        self.assertEqual(refreshed.snapshot_index, 40)
        self.assertEqual(len(refreshed.plasmon.time_plots), len(base.plasmon.time_plots))
        self.assertIs(refreshed.shock, base.shock)
        base_profile = np.asarray(base.plasmon.profile_plots[0].y_series[0], dtype=np.float64)
        refreshed_profile = np.asarray(refreshed.plasmon.profile_plots[0].y_series[0], dtype=np.float64)
        finite = np.isfinite(base_profile) & np.isfinite(refreshed_profile)
        self.assertTrue(np.any(finite))
        self.assertGreater(float(np.nanmax(np.abs(base_profile[finite] - refreshed_profile[finite]))), 0.0)

    def test_lindhard_cache_bucket_reuses_identical_request(self) -> None:
        path = _test_bootstrap.example_data_path("Cu_0166_stabilized.h5")
        dataset = load_run_data(path)
        context = _context_from_dataset(path, dataset, snapshot_index=20)
        cache = AnalysisStateCache()
        params = DerivedAnalysisParameters(
            plasmon_model="lindhard",
            plasmon_execution_mode="benchmark",
            plasmon_integration_mode="effective_state",
            plasmon_energy_window_ev=24.0,
            plasmon_energy_points=301,
            plasmon_instrument_fwhm_ev=0.5,
        )
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=params.observation_side,
            line_of_sight_angle_deg=params.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=params.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=params.profile_coordinate_mode,
        )
        first = evaluate_plasmon_regime(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=params.plasmon_photon_energy_kev,
            scattering_angle_deg=params.plasmon_scattering_angle_deg,
            adiabatic_index=params.plasmon_adiabatic_index,
            parameters=params,
            geometry=geometry,
            include_time_plots=False,
            analysis_cache=cache,
        )
        second = evaluate_plasmon_regime(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=params.plasmon_photon_energy_kev,
            scattering_angle_deg=params.plasmon_scattering_angle_deg,
            adiabatic_index=params.plasmon_adiabatic_index,
            parameters=params,
            geometry=geometry,
            include_time_plots=False,
            analysis_cache=cache,
        )
        stats = cache.stats()
        self.assertGreaterEqual(stats["time_series_hits"], 1)
        self.assertGreaterEqual(stats["time_series_misses"], 1)
        self.assertTrue(np.allclose(first.spectrum_intensity, second.spectrum_intensity, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
