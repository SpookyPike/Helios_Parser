from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon import (
    _build_rpa_spectrum_payload,
    _cluster_zone_plasmon_states,
    _effective_electron_fields,
    _extract_zone_plasmon_states,
    _plasmon_state_summary,
    _summary_values,
    evaluate_plasmon_regime,
)
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry
from test_plasmon_phase3 import _synthetic_dataset


class PlasmonPhase5Tests(unittest.TestCase):
    def _clustered_synthetic(self):
        dataset, context = _synthetic_dataset()
        te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
        ti = np.asarray(dataset.temperature_i_ev, dtype=np.float64).copy()
        ne = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
        z = np.asarray(dataset.mean_charge, dtype=np.float64).copy()
        te[:, :3] = 120.0
        ti[:, :3] = 80.0
        ne[:, :3] = 8.0e20
        z[:, :3] = 6.0
        te[:, 3:] = 240.0
        ti[:, 3:] = 120.0
        ne[:, 3:] = 3.5e20
        z[:, 3:] = 3.0
        dataset = replace(
            dataset,
            temperature_e_ev=te,
            temperature_i_ev=ti,
            electron_density_cm3=ne,
            mean_charge=z,
        )
        return dataset, context

    def _geometry(self, dataset, context, params):
        return build_analysis_geometry(
            dataset,
            context,
            observation_side=params.observation_side,
            line_of_sight_angle_deg=params.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=params.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=params.profile_coordinate_mode,
        )

    def test_los_integrated_rpa_matches_weighted_cluster_sum(self) -> None:
        dataset, context = self._clustered_synthetic()
        params = DerivedAnalysisParameters(
            plasmon_model="rpa",
            plasmon_integration_mode="los_integrated",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=30.0,
            plasmon_energy_points=501,
            plasmon_instrument_fwhm_ev=0.0,
            plasmon_normalization="none",
            plasmon_cluster_log_ne_tol=0.01,
            plasmon_cluster_log_te_tol=0.01,
            plasmon_cluster_z_tol=0.05,
        )
        geometry = self._geometry(dataset, context, params)
        result = evaluate_plasmon_regime(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=params.plasmon_photon_energy_kev,
            scattering_angle_deg=params.plasmon_scattering_angle_deg,
            adiabatic_index=params.plasmon_adiabatic_index,
            parameters=params,
            geometry=geometry,
            include_time_plots=False,
        )
        weighting_mode = "electron_column"
        electron_fields = _effective_electron_fields(dataset, params)
        current, _, mask, selection = _summary_values(
            dataset,
            context,
            params,
            geometry,
            snapshot_index=context.snapshot_index,
            weighting_mode=weighting_mode,
            electron_fields=electron_fields,
        )
        zone_states = _extract_zone_plasmon_states(
            dataset,
            snapshot_index=context.snapshot_index,
            mask=mask,
            selection=selection,
            weighting_mode=weighting_mode,
            geometry=geometry,
            electron_fields=electron_fields,
        )
        clustered, _ = _cluster_zone_plasmon_states(
            dataset,
            snapshot_index=context.snapshot_index,
            selection=selection,
            zone_states=zone_states,
            weighting_mode=weighting_mode,
            geometry=geometry,
            parameters=params,
            electron_fields=electron_fields,
        )
        raw_params = replace(params, plasmon_instrument_fwhm_ev=0.0, plasmon_normalization="none", plasmon_integration_mode="effective_state")
        manual = np.zeros_like(result.spectrum_energy_ev, dtype=np.float64)
        for idx in range(clustered["weight"].size):
            state = _plasmon_state_summary(
                float(clustered["te_ev"][idx]),
                float(clustered["ti_ev"][idx]),
                float(clustered["ne_cm3"][idx]),
                float(clustered["zbar"][idx]),
                float(clustered["ion_mass_mu"][idx]),
                params,
            )
            payload = _build_rpa_spectrum_payload(state, raw_params)
            manual += float(clustered["weight"][idx]) * np.asarray(payload["spectrum"], dtype=np.float64)
        self.assertEqual(result.integration_mode, "los_integrated")
        self.assertEqual(result.zone_count_used, 6)
        self.assertEqual(result.cluster_count_used, 2)
        self.assertTrue(np.allclose(result.spectrum_intensity, manual, rtol=1.0e-10, atol=1.0e-10))
        self.assertGreater(np.nanmax(result.spectrum_intensity), 0.0)
        self.assertGreater(float(current["scattering_wavevector_m_inv"]), 0.0)

    def test_los_integrated_cache_bucket_reuses_payload(self) -> None:
        dataset, context = self._clustered_synthetic()
        params = DerivedAnalysisParameters(
            plasmon_model="mermin",
            plasmon_integration_mode="los_integrated",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=30.0,
            plasmon_energy_points=501,
            plasmon_collision_model="manual_constant",
            plasmon_manual_collision_rate_s=8.0e14,
        )
        geometry = self._geometry(dataset, context, params)
        cache = AnalysisStateCache()
        first = evaluate_plasmon_regime(
            dataset, context,
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
            dataset, context,
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
        self.assertGreaterEqual(stats["time_series_misses"], 1)
        self.assertGreaterEqual(stats["time_series_hits"], 1)
        self.assertEqual(first.integration_mode, "los_integrated")
        self.assertTrue(np.allclose(first.spectrum_intensity, second.spectrum_intensity, equal_nan=True))

    def test_reference_example_los_integrated_is_finite_and_differs_from_effective_state(self) -> None:
        path = _test_bootstrap.example_data_path("5Fe+4.9TW+light_stabilized.h5")
        dataset = load_run_data(path)
        context = RunContext(
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
            snapshot_index=min(5, max(0, len(dataset.time_s) - 1)),
            map_coordinate="static_x",
            slice_coordinate="zone",
            selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
            selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
        )

        def compute(integration_mode: str):
            params = DerivedAnalysisParameters(
                plasmon_model="rpa",
                plasmon_integration_mode=integration_mode,
                plasmon_photon_energy_kev=8.0,
                plasmon_scattering_angle_deg=25.0,
                plasmon_energy_window_ev=60.0,
                plasmon_energy_points=801,
                plasmon_instrument_fwhm_ev=1.0,
            )
            geometry = self._geometry(dataset, context, params)
            return evaluate_plasmon_regime(
                dataset, context,
                snapshot_index=context.snapshot_index,
                photon_energy_kev=params.plasmon_photon_energy_kev,
                scattering_angle_deg=params.plasmon_scattering_angle_deg,
                adiabatic_index=params.plasmon_adiabatic_index,
                parameters=params,
                geometry=geometry,
                include_time_plots=False,
            )

        effective = compute("effective_state")
        integrated = compute("los_integrated")
        finite_fraction = np.count_nonzero(np.isfinite(integrated.spectrum_intensity)) / float(integrated.spectrum_intensity.size)
        self.assertGreater(finite_fraction, 0.98)
        self.assertEqual(integrated.integration_mode, "los_integrated")
        self.assertGreaterEqual(integrated.zone_count_used, integrated.cluster_count_used)
        self.assertGreater(integrated.zone_count_used, 0)
        self.assertGreater(np.nanmax(np.abs(integrated.spectrum_intensity - effective.spectrum_intensity)), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
