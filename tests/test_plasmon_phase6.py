from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.plasmon_spectrum import (
    energy_axis_ev,
    epsilon_mermin,
    epsilon_mermin_static_lfc,
    epsilon_rpa_static_lfc,
)
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry
from test_plasmon_phase3 import _synthetic_dataset


class PlasmonPhase6Tests(unittest.TestCase):
    def _compute(self, model: str = "mermin_static_lfc", **kwargs):
        dataset, context = _synthetic_dataset(te_ev=kwargs.pop("te_ev", 120.0), ne_cm3=kwargs.pop("ne_cm3", 8.0e20))
        parameters = DerivedAnalysisParameters(
            plasmon_model=model,
            plasmon_photon_energy_kev=kwargs.pop("photon_energy_kev", 0.5),
            plasmon_scattering_angle_deg=kwargs.pop("scattering_angle_deg", 1.0),
            plasmon_energy_window_ev=kwargs.pop("energy_window_ev", 40.0),
            plasmon_energy_points=kwargs.pop("energy_points", 801),
            plasmon_instrument_fwhm_ev=kwargs.pop("instrument_fwhm_ev", 0.5),
            plasmon_collision_model=kwargs.pop("collision_model", "manual_constant"),
            plasmon_manual_collision_rate_s=kwargs.pop("collision_rate_s", 1.5e15),
            plasmon_lfc_model=kwargs.pop("lfc_model", "esa_static"),
            plasmon_integration_mode=kwargs.pop("integration_mode", "effective_state"),
        )
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=parameters.observation_side,
            line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=parameters.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=parameters.profile_coordinate_mode,
        )
        return evaluate_plasmon_regime(
            dataset,
            context,
            snapshot_index=context.snapshot_index,
            photon_energy_kev=parameters.plasmon_photon_energy_kev,
            scattering_angle_deg=parameters.plasmon_scattering_angle_deg,
            adiabatic_index=parameters.plasmon_adiabatic_index,
            parameters=parameters,
            geometry=geometry,
            include_time_plots=False,
            analysis_cache=kwargs.pop("analysis_cache", None),
        )

    def test_epsilon_mermin_static_lfc_reduces_to_rpa_static_lfc_when_nu_zero(self) -> None:
        energy = energy_axis_ev(20.0, 401)
        _, eps_rpa_lfc, g_rpa, q_rpa = epsilon_rpa_static_lfc(
            energy,
            k_m_inv=1.8e8,
            te_ev=120.0,
            ne_cm3=8.0e20,
            imag_shift_ev=0.05,
            rs=1.55,
            theta=0.6,
        )
        _, eps_mermin_lfc, g_mermin, q_mermin = epsilon_mermin_static_lfc(
            energy,
            k_m_inv=1.8e8,
            te_ev=120.0,
            ne_cm3=8.0e20,
            collision_rate_s=0.0,
            imag_shift_ev=0.05,
            rs=1.55,
            theta=0.6,
        )
        self.assertAlmostEqual(g_rpa, g_mermin, places=12)
        self.assertAlmostEqual(q_rpa, q_mermin, places=12)
        self.assertTrue(np.allclose(eps_rpa_lfc, eps_mermin_lfc, equal_nan=True, rtol=1.0e-11, atol=1.0e-11))

    def test_epsilon_mermin_static_lfc_reduces_to_mermin_when_g_goes_to_zero(self) -> None:
        energy = energy_axis_ev(20.0, 401)
        _, eps_mermin = epsilon_mermin(
            energy,
            k_m_inv=1.0e4,
            te_ev=120.0,
            ne_cm3=8.0e20,
            collision_rate_s=1.0e15,
            imag_shift_ev=0.05,
        )
        with mock.patch("helios.services.derived.plasmon_spectrum.esa_static_local_field_correction", return_value=0.0):
            _, eps_mermin_lfc, g_value, _ = epsilon_mermin_static_lfc(
                energy,
                k_m_inv=1.0e4,
                te_ev=120.0,
                ne_cm3=8.0e20,
                collision_rate_s=1.0e15,
                imag_shift_ev=0.05,
                rs=1.55,
                theta=0.6,
            )
        self.assertEqual(g_value, 0.0)
        self.assertTrue(np.allclose(eps_mermin, eps_mermin_lfc, equal_nan=True, rtol=1.0e-11, atol=1.0e-11))

    def test_mermin_static_lfc_builds_finite_spectrum_and_differs_from_parent_branches(self) -> None:
        mermin = self._compute(model="mermin", photon_energy_kev=8.0, scattering_angle_deg=25.0)
        rpa_lfc = self._compute(model="rpa_static_lfc", photon_energy_kev=8.0, scattering_angle_deg=25.0)
        adv = self._compute(model="mermin_static_lfc", photon_energy_kev=8.0, scattering_angle_deg=25.0, collision_rate_s=1.5e15)
        self.assertEqual(adv.model_name, "mermin_static_lfc")
        self.assertGreater(adv.static_lfc_value, 0.0)
        self.assertGreater(adv.electron_collision_rate_s, 0.0)
        self.assertTrue(np.all(np.isfinite(adv.spectrum_intensity)))
        self.assertGreater(np.nanmax(np.abs(adv.spectrum_intensity - mermin.spectrum_intensity)), 1.0e-5)
        self.assertGreater(np.nanmax(np.abs(adv.spectrum_intensity - rpa_lfc.spectrum_intensity)), 1.0e-5)

    def test_mermin_static_lfc_cache_bucket_reuses_payload(self) -> None:
        cache = AnalysisStateCache()
        first = self._compute(model="mermin_static_lfc", collision_rate_s=9.0e14, analysis_cache=cache)
        second = self._compute(model="mermin_static_lfc", collision_rate_s=9.0e14, analysis_cache=cache)
        stats = cache.stats()
        self.assertGreaterEqual(stats["time_series_misses"], 1)
        self.assertGreaterEqual(stats["time_series_hits"], 1)
        self.assertTrue(np.allclose(first.spectrum_intensity, second.spectrum_intensity, equal_nan=True))

    def test_los_integrated_mermin_static_lfc_is_finite(self) -> None:
        result = self._compute(model="mermin_static_lfc", integration_mode="los_integrated", collision_rate_s=8.0e14)
        self.assertEqual(result.integration_mode, "los_integrated")
        self.assertGreaterEqual(result.zone_count_used, result.cluster_count_used)
        self.assertGreater(result.cluster_count_used, 0)
        finite_fraction = np.count_nonzero(np.isfinite(result.spectrum_intensity)) / float(result.spectrum_intensity.size)
        self.assertGreater(finite_fraction, 0.98)

    def test_reference_example_mermin_static_lfc_is_finite(self) -> None:
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
        params = DerivedAnalysisParameters(
            plasmon_model="mermin_static_lfc",
            plasmon_photon_energy_kev=8.0,
            plasmon_scattering_angle_deg=25.0,
            plasmon_energy_window_ev=60.0,
            plasmon_energy_points=801,
            plasmon_instrument_fwhm_ev=1.0,
            plasmon_collision_model="nrl_constant",
            plasmon_lfc_model="esa_static",
        )
        geometry = build_analysis_geometry(
            dataset,
            context,
            observation_side=params.observation_side,
            line_of_sight_angle_deg=params.line_of_sight_angle_deg,
            line_of_sight_impact_parameter_cm=params.line_of_sight_impact_parameter_cm,
            profile_coordinate_mode=params.profile_coordinate_mode,
        )
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
        self.assertIn(result.model_name, {"mermin_static_lfc", "rpa_static_lfc"})
        finite_fraction = np.count_nonzero(np.isfinite(result.spectrum_intensity)) / float(result.spectrum_intensity.size)
        self.assertGreater(finite_fraction, 0.98)
        self.assertGreater(result.static_lfc_value, 0.0)


if __name__ == "__main__":
    unittest.main()
