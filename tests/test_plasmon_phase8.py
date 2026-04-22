from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
try:
    from scipy.special import wofz
except ModuleNotFoundError:  # pragma: no cover - optional reference backend
    wofz = None

import _test_bootstrap  # noqa: F401

from helios.cancellation import AnalysisCancelled
from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
from helios.services.derived.models import DerivedRunData
import helios.services.derived.plasmon as plasmon_module
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.plasmon_config import plasmon_ui_capabilities
from helios.services.derived.plasmon_electron_policy import PLASMON_BENCHMARK_POLICY_COMPARISON
from helios.services.derived.plasmon_spectrum import (
    classical_response_cache_info,
    clear_classical_response_cache,
    epsilon_rpa,
    epsilon_rpa_static_lfc,
    finite_t_susceptibility,
    plasma_dispersion_function,
)
from helios.services.derived.plasmon_validation import q_to_angle_deg, uniform_al_dataset
from helios.services.derived.selection import build_analysis_geometry
from test_plasmon_phase3 import _synthetic_dataset


def _geometry(dataset, context, params):
    return build_analysis_geometry(
        dataset,
        context,
        observation_side=params.observation_side,
        line_of_sight_angle_deg=params.line_of_sight_angle_deg,
        line_of_sight_impact_parameter_cm=params.line_of_sight_impact_parameter_cm,
        profile_coordinate_mode=params.profile_coordinate_mode,
    )


def _compute(dataset, context, **kwargs):
    params = DerivedAnalysisParameters(**kwargs)
    geom = _geometry(dataset, context, params)
    return evaluate_plasmon_regime(
        dataset, context,
        snapshot_index=context.snapshot_index,
        photon_energy_kev=params.plasmon_photon_energy_kev,
        scattering_angle_deg=params.plasmon_scattering_angle_deg,
        adiabatic_index=params.plasmon_adiabatic_index,
        parameters=params,
        geometry=geom,
        include_time_plots=False,
    )


class PlasmonPhase8Tests(unittest.TestCase):
    def test_plasma_dispersion_matches_faddeeva_reference(self) -> None:
        if wofz is None:
            self.skipTest("scipy.special.wofz is not available in this environment.")
        zeta = np.asarray([0.2 + 0.1j, -0.7 + 0.3j, 1.5 + 0.05j], dtype=np.complex128)
        expected = 1j * np.sqrt(np.pi) * wofz(zeta)
        actual = plasma_dispersion_function(zeta)
        self.assertTrue(np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12))

    def test_quicklook_backward_compatible(self) -> None:
        dataset, context = _synthetic_dataset()
        base = _compute(dataset, context)
        explicit = _compute(dataset, context, plasmon_model='quicklook')
        self.assertEqual(base.model_name, 'quicklook')
        self.assertEqual(explicit.model_name, 'quicklook')
        self.assertAlmostEqual(base.plasma_frequency_ev, explicit.plasma_frequency_ev, places=12)
        self.assertAlmostEqual(base.k_lambda_debye, explicit.k_lambda_debye, places=12)
        self.assertEqual(base.regime_label, explicit.regime_label)

    def test_rpa_spectrum_is_finite_on_reference_example(self) -> None:
        path = _test_bootstrap.example_data_path("5Fe+4.9TW+light_stabilized.h5")
        dataset = load_run_data(path)
        context = RunContext(
            path=path,
            summary=dict(dataset.summary), metadata=dict(dataset.metadata),
            fields=('density','velocity','temperature_e','temperature_i','electron_density','mean_charge'),
            diagnostics=(), time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(), static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
            zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(), zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
            has_dynamic_radius=dataset.radius_cm is not None, snapshot_index=min(5, max(0, len(dataset.time_s)-1)), map_coordinate='static_x', slice_coordinate='zone',
            selected_region_ids=tuple(int(v) for v in np.unique(np.asarray(dataset.zone_region_id, dtype=np.int32))),
            selected_material_ids=tuple(int(v) for v in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
        )
        result = _compute(dataset, context, plasmon_model='rpa', plasmon_photon_energy_kev=7.5, plasmon_scattering_angle_deg=20.0, plasmon_energy_window_ev=80.0, plasmon_energy_points=1201)
        self.assertEqual(result.model_name, 'rpa')
        self.assertEqual(result.benchmark_status, 'valid')
        self.assertTrue(result.model_executed_fully)
        self.assertAlmostEqual(float(result.fallback_fraction), 0.0, places=12)
        self.assertGreater(np.count_nonzero(np.isfinite(result.spectrum_intensity))/float(result.spectrum_intensity.size), 0.98)

    def test_rpa_static_lfc_reduces_to_rpa_when_g_zero(self) -> None:
        energy = np.linspace(-10.0, 10.0, 401, dtype=np.float64)
        _, eps_rpa = epsilon_rpa(energy, k_m_inv=1.8e8, te_ev=120.0, ne_cm3=8.0e20, imag_shift_ev=0.05)
        with mock.patch('helios.services.derived.plasmon_spectrum.esa_static_local_field_correction', return_value=0.0):
            _, eps_lfc, g_value, _ = epsilon_rpa_static_lfc(energy, k_m_inv=1.8e8, te_ev=120.0, ne_cm3=8.0e20, imag_shift_ev=0.05, rs=1.55, theta=0.6)
        self.assertEqual(g_value, 0.0)
        self.assertTrue(np.allclose(eps_rpa, eps_lfc, equal_nan=True, rtol=1.0e-11, atol=1.0e-11))

    def test_instrument_fwhm_changes_observed_linewidth(self) -> None:
        dataset, context = _synthetic_dataset()
        sharp = _compute(dataset, context, plasmon_model='rpa', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=40.0, plasmon_energy_points=801, plasmon_instrument_fwhm_ev=0.0)
        broad = _compute(dataset, context, plasmon_model='rpa', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=40.0, plasmon_energy_points=801, plasmon_instrument_fwhm_ev=2.0)
        self.assertGreater(float(broad.peak_fwhm_ev), float(sharp.peak_fwhm_ev))

    def test_los_integrated_responds_to_region_and_material_deselection(self) -> None:
        dataset, context = _synthetic_dataset()
        te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
        ne = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
        te[:, :3] = 90.0
        te[:, 3:] = 260.0
        ne[:, :3] = 9.0e20
        ne[:, 3:] = 2.5e20
        dataset = replace(dataset, temperature_e_ev=te, electron_density_cm3=ne)
        full = _compute(dataset, context, plasmon_model='rpa', plasmon_integration_mode='los_integrated', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=501, plasmon_normalization='none', plasmon_instrument_fwhm_ev=0.0)
        region_context = replace(context, selected_region_ids=(1,), selected_material_ids=(1,))
        subset = _compute(dataset, region_context, plasmon_model='rpa', plasmon_integration_mode='los_integrated', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=501, plasmon_normalization='none', plasmon_instrument_fwhm_ev=0.0)
        self.assertGreater(np.nanmax(np.abs(full.spectrum_intensity - subset.spectrum_intensity)), 1.0e-8)

    def test_invalid_mermin_request_is_marked_invalid_for_benchmark(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model='mermin',
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
            plasmon_collision_model='manual_constant',
            plasmon_manual_collision_rate_s=-1.0,
        )
        self.assertEqual(result.model_name, 'mermin')
        self.assertEqual(result.benchmark_status, 'invalid_for_benchmark')
        self.assertFalse(result.model_executed_fully)
        self.assertAlmostEqual(float(result.fallback_fraction), 1.0, places=12)
        self.assertEqual(result.spectrum_energy_ev.size, 0)
        self.assertEqual(result.spectrum_intensity.size, 0)

    def test_invalid_lindhard_mermin_request_reports_collision_reason_and_backend(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model='lindhard_mermin',
            plasmon_integration_mode='los_integrated',
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
            plasmon_collision_model='manual_constant',
            plasmon_manual_collision_rate_s=-1.0,
        )
        self.assertEqual(result.model_name, 'lindhard_mermin')
        self.assertEqual(result.benchmark_status, 'invalid_for_benchmark')
        self.assertEqual(result.response_backend, 'finite_t_lindhard')
        joined = "\n".join(w.message for w in result.warnings)
        self.assertIn('collision rate', joined.lower())
        self.assertFalse(result.model_executed_fully)

    def test_valid_mermin_request_reports_full_execution(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model='mermin',
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
            plasmon_collision_model='manual_constant',
            plasmon_manual_collision_rate_s=1.5e15,
        )
        self.assertEqual(result.model_name, 'mermin')
        self.assertEqual(result.benchmark_status, 'valid')
        self.assertTrue(result.model_executed_fully)
        self.assertAlmostEqual(float(result.fallback_fraction), 0.0, places=12)
        self.assertGreater(result.spectrum_energy_ev.size, 0)

    def test_invalid_static_lfc_backend_is_marked_invalid_for_benchmark(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model='rpa_static_lfc',
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
            plasmon_lfc_model='none',
        )
        self.assertEqual(result.model_name, 'rpa_static_lfc')
        self.assertEqual(result.benchmark_status, 'invalid_for_benchmark')
        self.assertFalse(result.model_executed_fully)
        self.assertEqual(result.spectrum_energy_ev.size, 0)

    def test_benchmark_mode_refines_grid_and_peak_fit(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_execution_mode='benchmark',
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
        )
        self.assertEqual(result.execution_mode, 'benchmark')
        self.assertGreaterEqual(int(result.spectrum_points), 4001)
        self.assertEqual(result.peak_fit_method, 'local_quadratic')

    def test_plasmon_ui_capabilities_expose_dispersion_compare_views_without_result(self) -> None:
        capabilities = plasmon_ui_capabilities(
            model='rpa',
            execution_mode='benchmark',
            study_mode='dispersion',
            compare_models=True,
            compare_policies=False,
        )
        self.assertTrue(capabilities.advanced_model_requested)
        self.assertEqual(capabilities.primary_label, 'Peak shift / comparison')
        self.assertIn('dispersion_selected_model', tuple(option.key for option in capabilities.time_options))
        self.assertIn('dispersion_compare_models', tuple(option.key for option in capabilities.time_options))
        self.assertIn('dispersion_compare_width_models', tuple(option.key for option in capabilities.profile_options))

    def test_plasmon_ui_capabilities_disable_policy_compare_outside_benchmark(self) -> None:
        capabilities = plasmon_ui_capabilities(
            model='rpa',
            execution_mode='quicklook',
            study_mode='spectrum',
            compare_models=False,
            compare_policies=True,
        )
        self.assertFalse(capabilities.compare_policies_available)
        self.assertIn('benchmark-only', capabilities.compare_policies_reason.lower())
        self.assertNotIn('spectrum_compare_policies', tuple(option.key for option in capabilities.time_options))

    def test_los_integrated_is_not_equal_to_effective_state_for_bimodal_hydro_state(self) -> None:
        dataset, context = _synthetic_dataset()
        te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
        ne = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
        te[:, :3] = 40.0
        te[:, 3:] = 280.0
        ne[:, :3] = 1.5e20
        ne[:, 3:] = 8.5e20
        dataset = replace(dataset, temperature_e_ev=te, electron_density_cm3=ne)
        effective = _compute(dataset, context, plasmon_model='rpa', plasmon_execution_mode='benchmark', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=35.0, plasmon_energy_points=801, plasmon_normalization='none', plasmon_instrument_fwhm_ev=0.0, plasmon_integration_mode='effective_state')
        integrated = _compute(dataset, context, plasmon_model='rpa', plasmon_execution_mode='benchmark', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=35.0, plasmon_energy_points=801, plasmon_normalization='none', plasmon_instrument_fwhm_ev=0.0, plasmon_integration_mode='los_integrated')
        self.assertGreater(np.nanmax(np.abs(effective.spectrum_intensity - integrated.spectrum_intensity)), 1.0e-6)
        self.assertGreaterEqual(int(integrated.cluster_count_used), 2)

    def test_dispersion_scan_bundle_respects_k_axis_and_point_count(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_study_mode='dispersion',
            plasmon_scan_axis='k_angstrom_inv',
            plasmon_scan_start=0.4,
            plasmon_scan_stop=1.6,
            plasmon_scan_points=9,
            plasmon_photon_energy_kev=8.0,
            plasmon_scattering_angle_deg=20.0,
            plasmon_energy_window_ev=35.0,
            plasmon_energy_points=801,
        )
        bundle = next(bundle for bundle in result.profile_plots if bundle.key == 'dispersion_selected_model')
        self.assertEqual(bundle.x_values.size, 9)
        self.assertIn('[1/A]', bundle.x_label)
        self.assertEqual(result.study_mode, 'dispersion')
        self.assertEqual(result.scan_axis, 'k_angstrom_inv')

    def test_compare_all_models_adds_spectrum_comparison_bundle(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_compare_models=True,
            plasmon_study_mode='spectrum',
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
        )
        bundle = next(bundle for bundle in result.profile_plots if bundle.key == 'spectrum_compare_models')
        self.assertGreaterEqual(len(bundle.curve_names), 3)
        self.assertEqual(len(bundle.curve_names), len(bundle.y_series))
        self.assertTrue(result.compare_models)

    def test_compare_model_selection_limits_model_overlay_bundle(self) -> None:
        dataset, context = uniform_al_dataset(2.7, 0.3)
        result = _compute(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_execution_mode='benchmark',
            plasmon_compare_models=True,
            plasmon_compare_model_names=('rpa_static_lfc', 'lindhard'),
            plasmon_study_mode='spectrum',
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=q_to_angle_deg(1.28, 8.307),
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
            plasmon_lfc_model='esa_static',
        )
        bundle = next(bundle for bundle in result.profile_plots if bundle.key == 'spectrum_compare_models')
        self.assertEqual(bundle.curve_names, ('RPA + static LFC', 'Finite-T Lindhard'))
        self.assertEqual(result.comparison_models, ('rpa_static_lfc', 'lindhard'))

    def test_compare_all_models_adds_dispersion_width_comparison_bundle(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_compare_models=True,
            plasmon_study_mode='dispersion',
            plasmon_scan_axis='k_angstrom_inv',
            plasmon_scan_start=0.4,
            plasmon_scan_stop=1.2,
            plasmon_scan_points=7,
            plasmon_photon_energy_kev=8.0,
            plasmon_scattering_angle_deg=20.0,
            plasmon_energy_window_ev=35.0,
            plasmon_energy_points=801,
        )
        width_bundle = next(bundle for bundle in result.profile_plots if bundle.key == 'dispersion_compare_width_models')
        self.assertEqual(width_bundle.x_values.size, 7)
        self.assertGreaterEqual(len(width_bundle.curve_names), 3)
        self.assertEqual(len(width_bundle.curve_names), len(width_bundle.y_series))

    def test_compare_policies_adds_spectrum_policy_bundle(self) -> None:
        dataset, context = uniform_al_dataset(2.7, 0.3)
        result = _compute(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_execution_mode='benchmark',
            plasmon_compare_policies=True,
            plasmon_study_mode='spectrum',
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=q_to_angle_deg(1.28, 8.307),
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
        )
        bundle = next(bundle for bundle in result.profile_plots if bundle.key == 'spectrum_compare_policies')
        self.assertEqual(tuple(result.policy_comparison_policies), PLASMON_BENCHMARK_POLICY_COMPARISON)
        self.assertEqual(len(bundle.curve_names), len(PLASMON_BENCHMARK_POLICY_COMPARISON))
        self.assertTrue(result.compare_policies)

    def test_compare_policies_adds_dispersion_policy_bundle(self) -> None:
        dataset, context = uniform_al_dataset(2.7, 0.3)
        result = _compute(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_execution_mode='benchmark',
            plasmon_compare_policies=True,
            plasmon_study_mode='dispersion',
            plasmon_scan_axis='k_angstrom_inv',
            plasmon_scan_start=0.92,
            plasmon_scan_stop=1.60,
            plasmon_scan_points=7,
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=q_to_angle_deg(1.28, 8.307),
            plasmon_energy_window_ev=45.0,
            plasmon_energy_points=1201,
        )
        bundle = next(bundle for bundle in result.profile_plots if bundle.key == 'dispersion_compare_policies')
        self.assertEqual(bundle.x_values.size, 7)
        self.assertEqual(len(bundle.curve_names), len(PLASMON_BENCHMARK_POLICY_COMPARISON))
        self.assertEqual(tuple(result.policy_comparison_policies), PLASMON_BENCHMARK_POLICY_COMPARISON)

    def test_spectrum_mode_does_not_build_dispersion_scan_bundles(self) -> None:
        dataset, context = _synthetic_dataset()
        with (
            mock.patch.object(plasmon_module, "_build_dispersion_bundles", side_effect=AssertionError("dispersion bundles should not be built in spectrum mode")),
            mock.patch.object(plasmon_module, "_build_spectrum_compare_bundle", wraps=plasmon_module._build_spectrum_compare_bundle) as compare_mock,
        ):
            result = _compute(
                dataset,
                context,
                plasmon_model='rpa',
                plasmon_compare_models=True,
                plasmon_study_mode='spectrum',
                plasmon_photon_energy_kev=0.5,
                plasmon_scattering_angle_deg=1.0,
                plasmon_energy_window_ev=40.0,
                plasmon_energy_points=801,
            )
        compare_mock.assert_called_once()
        self.assertEqual(result.study_mode, 'spectrum')
        self.assertTrue(any(bundle.key == 'spectrum_compare_models' for bundle in result.profile_plots))

    def test_dispersion_mode_does_not_build_spectrum_compare_bundle(self) -> None:
        dataset, context = _synthetic_dataset()
        with (
            mock.patch.object(plasmon_module, "_build_spectrum_compare_bundle", side_effect=AssertionError("spectrum comparison should not be built in dispersion mode")),
            mock.patch.object(plasmon_module, "_build_dispersion_bundles", wraps=plasmon_module._build_dispersion_bundles) as dispersion_mock,
        ):
            result = _compute(
                dataset,
                context,
                plasmon_model='rpa',
                plasmon_compare_models=True,
                plasmon_study_mode='dispersion',
                plasmon_scan_axis='k_angstrom_inv',
                plasmon_scan_start=0.4,
                plasmon_scan_stop=1.2,
                plasmon_scan_points=7,
                plasmon_photon_energy_kev=8.0,
                plasmon_scattering_angle_deg=20.0,
                plasmon_energy_window_ev=35.0,
                plasmon_energy_points=801,
            )
        dispersion_mock.assert_called_once()
        self.assertEqual(result.study_mode, 'dispersion')
        self.assertTrue(any(bundle.key == 'dispersion_selected_model' for bundle in result.profile_plots))

    def test_plasmon_result_carries_runtime_breakdown_diagnostics(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model='rpa',
            plasmon_study_mode='spectrum',
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=801,
        )
        self.assertGreaterEqual(float(result.total_runtime_s), 0.0)
        self.assertGreaterEqual(float(result.spectrum_runtime_s), 0.0)
        self.assertGreaterEqual(float(result.comparison_runtime_s), 0.0)
        self.assertGreaterEqual(float(result.dispersion_runtime_s), 0.0)
        self.assertGreaterEqual(float(result.time_series_runtime_s), 0.0)

    def test_classical_response_cache_reuses_identical_state_calls(self) -> None:
        clear_classical_response_cache()
        energy = np.linspace(-20.0, 20.0, 801, dtype=np.float64)
        first = finite_t_susceptibility(
            energy,
            k_m_inv=1.8e10,
            te_ev=0.6,
            ne_cm3=1.8e23,
            imag_shift_ev=0.02,
        )
        after_first = classical_response_cache_info()
        second = finite_t_susceptibility(
            energy,
            k_m_inv=1.8e10,
            te_ev=0.6,
            ne_cm3=1.8e23,
            imag_shift_ev=0.02,
        )
        after_second = classical_response_cache_info()
        self.assertTrue(np.allclose(first, second, equal_nan=True))
        self.assertEqual(int(after_first["finite_t_susceptibility_misses"]), 1)
        self.assertGreaterEqual(int(after_second["finite_t_susceptibility_hits"]), 1)

    def test_progress_check_cancels_long_dispersion_scan_cooperatively(self) -> None:
        dataset, context = _synthetic_dataset()
        params = DerivedAnalysisParameters(
            plasmon_model='rpa',
            plasmon_study_mode='dispersion',
            plasmon_compare_models=True,
            plasmon_scan_axis='k_angstrom_inv',
            plasmon_scan_start=0.4,
            plasmon_scan_stop=1.6,
            plasmon_scan_points=21,
            plasmon_photon_energy_kev=8.0,
            plasmon_scattering_angle_deg=20.0,
            plasmon_energy_window_ev=35.0,
            plasmon_energy_points=801,
        )
        geom = _geometry(dataset, context, params)
        checks = {"count": 0}

        def _cancel_soon() -> None:
            checks["count"] += 1
            if checks["count"] >= 3:
                raise AnalysisCancelled("synthetic cancellation")

        with self.assertRaises(AnalysisCancelled):
            evaluate_plasmon_regime(
                dataset,
                context,
                snapshot_index=context.snapshot_index,
                photon_energy_kev=params.plasmon_photon_energy_kev,
                scattering_angle_deg=params.plasmon_scattering_angle_deg,
                adiabatic_index=params.plasmon_adiabatic_index,
                parameters=params,
                geometry=geom,
                include_time_plots=False,
                progress_check=_cancel_soon,
            )
        self.assertGreaterEqual(checks["count"], 3)

    def test_zone_index_filters_change_los_integrated_mix(self) -> None:
        dataset, context = _synthetic_dataset()
        te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
        te[:, :2] = 60.0
        te[:, 2:4] = 140.0
        te[:, 4:] = 260.0
        dataset = replace(dataset, temperature_e_ev=te)
        full = _compute(dataset, context, plasmon_model='rpa', plasmon_execution_mode='benchmark', plasmon_integration_mode='los_integrated', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=801, plasmon_normalization='none', plasmon_instrument_fwhm_ev=0.0)
        middle = _compute(dataset, context, plasmon_model='rpa', plasmon_execution_mode='benchmark', plasmon_integration_mode='los_integrated', plasmon_photon_energy_kev=0.5, plasmon_scattering_angle_deg=1.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=801, plasmon_normalization='none', plasmon_instrument_fwhm_ev=0.0, zone_index_lower=3, zone_index_upper=4)
        self.assertLess(int(middle.zone_count_used), int(full.zone_count_used))
        self.assertGreater(np.nanmax(np.abs(full.spectrum_intensity - middle.spectrum_intensity)), 1.0e-6)

    def test_benchmark_los_domain_policing_rejects_noncollective_edge_zones_until_filtered(self) -> None:
        dataset, context = _synthetic_dataset(te_ev=0.3, ne_cm3=1.8e23)
        density = np.asarray(dataset.density_g_cm3, dtype=np.float64).copy()
        temperature_e = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
        temperature_i = np.asarray(dataset.temperature_i_ev, dtype=np.float64).copy()
        electron_density = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
        base_density = float(np.asarray(dataset.density_g_cm3, dtype=np.float64)[0, 2])
        density[:, :2] = 0.01
        temperature_e[:, :2] = 200.0
        temperature_i[:, :2] = 200.0
        electron_density[:, :2] *= (0.01 / max(base_density, 1.0e-12))
        dataset = replace(dataset, density_g_cm3=density, temperature_e_ev=temperature_e, temperature_i_ev=temperature_i, electron_density_cm3=electron_density)
        full = _compute(dataset, context, plasmon_model='rpa', plasmon_execution_mode='benchmark', plasmon_integration_mode='los_integrated', plasmon_photon_energy_kev=8.31, plasmon_scattering_angle_deg=17.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=801)
        cut = _compute(dataset, context, plasmon_model='rpa', plasmon_execution_mode='benchmark', plasmon_integration_mode='los_integrated', plasmon_photon_energy_kev=8.31, plasmon_scattering_angle_deg=17.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=801, zone_index_lower=3, zone_index_upper=6)
        self.assertEqual(full.benchmark_status, 'invalid_for_benchmark')
        self.assertGreater(int(full.noncollective_zone_count), 0)
        self.assertEqual(cut.benchmark_status, 'valid')

    def test_benchmark_los_static_lfc_domain_policing_rejects_bad_zone_subset(self) -> None:
        dataset, context = _synthetic_dataset(te_ev=0.3, ne_cm3=1.8e23)
        te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
        te[:, :2] = 6000.0
        dataset = replace(dataset, temperature_e_ev=te)
        full = _compute(dataset, context, plasmon_model='rpa_static_lfc', plasmon_execution_mode='benchmark', plasmon_integration_mode='los_integrated', plasmon_lfc_model='esa_static', plasmon_photon_energy_kev=8.31, plasmon_scattering_angle_deg=17.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=801)
        cut = _compute(dataset, context, plasmon_model='rpa_static_lfc', plasmon_execution_mode='benchmark', plasmon_integration_mode='los_integrated', plasmon_lfc_model='esa_static', plasmon_photon_energy_kev=8.31, plasmon_scattering_angle_deg=17.0, plasmon_energy_window_ev=30.0, plasmon_energy_points=801, zone_index_lower=3, zone_index_upper=6)
        self.assertEqual(full.benchmark_status, 'invalid_for_benchmark')
        self.assertGreater(int(full.lfc_out_of_domain_zone_count), 0)
        self.assertEqual(cut.benchmark_status, 'valid')

    def test_docs_describe_applicability_limits_and_warnings(self) -> None:
        readme = Path('README.md').read_text(encoding='utf-8').lower()
        notes = Path('docs/release-notes.html').read_text(encoding='utf-8').lower()
        user = Path('docs/user-guide.html').read_text(encoding='utf-8').lower()
        combined = readme + '\n' + notes + '\n' + user
        self.assertIn('warning', combined)
        self.assertIn('degenerate', combined)


if __name__ == '__main__':
    unittest.main()
