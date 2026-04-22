from __future__ import annotations

from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.plasmon import evaluate_plasmon_regime
from helios.services.derived.plasmon_spectrum import (
    energy_axis_ev,
    epsilon_lindhard,
    epsilon_lindhard_mermin,
    epsilon_lindhard_mermin_static_lfc,
    epsilon_lindhard_static_lfc,
)
from helios.services.derived.selection import AnalysisStateCache, build_analysis_geometry
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
    analysis_cache = kwargs.pop("analysis_cache", None)
    params = DerivedAnalysisParameters(**kwargs)
    geom = _geometry(dataset, context, params)
    return evaluate_plasmon_regime(
        dataset,
        context,
        snapshot_index=context.snapshot_index,
        photon_energy_kev=params.plasmon_photon_energy_kev,
        scattering_angle_deg=params.plasmon_scattering_angle_deg,
        adiabatic_index=params.plasmon_adiabatic_index,
        parameters=params,
        geometry=geom,
        include_time_plots=False,
        analysis_cache=analysis_cache,
    )


class PlasmonPhase9Tests(unittest.TestCase):
    def test_lindhard_backend_builds_finite_spectrum(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model="lindhard",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=401,
        )
        self.assertEqual(result.model_name, "lindhard")
        self.assertEqual(result.response_backend, "finite_t_lindhard")
        self.assertGreater(result.spectrum_energy_ev.size, 0)
        finite_fraction = np.count_nonzero(np.isfinite(result.spectrum_intensity)) / float(result.spectrum_intensity.size)
        self.assertGreater(finite_fraction, 0.98)

    def test_lindhard_mermin_reduces_to_lindhard_when_nu_zero(self) -> None:
        energy = energy_axis_ev(20.0, 401)
        _, eps_l = epsilon_lindhard(energy, k_m_inv=1.0e8, te_ev=20.0, ne_cm3=1.0e22, imag_shift_ev=0.05)
        _, eps_m = epsilon_lindhard_mermin(energy, k_m_inv=1.0e8, te_ev=20.0, ne_cm3=1.0e22, collision_rate_s=0.0, imag_shift_ev=0.05)
        self.assertTrue(np.allclose(eps_l, eps_m, equal_nan=True, rtol=1.0e-9, atol=1.0e-9))

    def test_lindhard_mermin_static_lfc_reduces_to_lindhard_mermin_when_g_zero(self) -> None:
        energy = energy_axis_ev(20.0, 401)
        _, eps_base = epsilon_lindhard_mermin(energy, k_m_inv=1.0e8, te_ev=20.0, ne_cm3=1.0e22, collision_rate_s=1.0e15, imag_shift_ev=0.05)
        with mock.patch("helios.services.derived.plasmon_spectrum.esa_static_local_field_correction", return_value=0.0):
            _, eps_lfc, g_value, _ = epsilon_lindhard_mermin_static_lfc(
                energy,
                k_m_inv=1.0e8,
                te_ev=20.0,
                ne_cm3=1.0e22,
                collision_rate_s=1.0e15,
                imag_shift_ev=0.05,
                rs=2.0,
                theta=0.5,
            )
        self.assertEqual(g_value, 0.0)
        self.assertTrue(np.allclose(eps_base, eps_lfc, equal_nan=True, rtol=1.0e-9, atol=1.0e-9))

    def test_auto_best_chooses_strongest_validated_backend_on_effective_state(self) -> None:
        dataset, context = _synthetic_dataset()
        result = _compute(
            dataset,
            context,
            plasmon_model="auto_best",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=30.0,
            plasmon_energy_points=401,
            plasmon_lfc_model="esa_static",
            plasmon_collision_model="manual_constant",
            plasmon_manual_collision_rate_s=1.0e15,
        )
        self.assertEqual(result.model_name, "auto_best")
        self.assertEqual(result.response_backend, "classical_maxwellian")
        self.assertTrue(result.auto_model_summary.startswith(("mermin", "rpa")))

    def test_auto_best_los_can_mix_local_models(self) -> None:
        dataset, context = _synthetic_dataset()
        te = np.asarray(dataset.temperature_e_ev, dtype=np.float64).copy()
        ne = np.asarray(dataset.electron_density_cm3, dtype=np.float64).copy()
        te[:, :3] = 0.3
        ne[:, :3] = 1.8e23
        te[:, 3:] = 30.0
        ne[:, 3:] = 1.0e21
        dataset = replace(dataset, temperature_e_ev=te, electron_density_cm3=ne)
        result = _compute(
            dataset,
            context,
            plasmon_model="auto_best",
            plasmon_integration_mode="los_integrated",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=30.0,
            plasmon_energy_points=401,
            plasmon_lfc_model="esa_static",
            plasmon_collision_model="manual_constant",
            plasmon_manual_collision_rate_s=1.0e15,
        )
        self.assertEqual(result.model_name, "auto_best")
        self.assertTrue(any(token in result.auto_model_summary for token in ("mermin", "rpa_static_lfc", "mermin_static_lfc")))
        self.assertIn(",", result.auto_model_summary)

    def test_lindhard_cache_bucket_reuses_identical_request(self) -> None:
        dataset, context = _synthetic_dataset()
        cache = AnalysisStateCache()
        first = _compute(
            dataset,
            context,
            plasmon_model="lindhard",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=25.0,
            plasmon_energy_points=301,
            analysis_cache=cache,
        )
        second = _compute(
            dataset,
            context,
            plasmon_model="lindhard",
            plasmon_photon_energy_kev=0.5,
            plasmon_scattering_angle_deg=1.0,
            plasmon_energy_window_ev=25.0,
            plasmon_energy_points=301,
            analysis_cache=cache,
        )
        stats = cache.stats()
        self.assertGreaterEqual(stats["time_series_misses"], 1)
        self.assertGreaterEqual(stats["time_series_hits"], 1)
        self.assertTrue(np.allclose(first.spectrum_intensity, second.spectrum_intensity, equal_nan=True))

    def test_manual_lindhard_benchmark_remains_valid_where_classical_rpa_rejects_noncollective(self) -> None:
        dataset, context = _synthetic_dataset(te_ev=200.0, ne_cm3=5.0e20)
        rpa = _compute(
            dataset,
            context,
            plasmon_model="rpa",
            plasmon_execution_mode="benchmark",
            plasmon_photon_energy_kev=8.0,
            plasmon_scattering_angle_deg=40.0,
            plasmon_energy_window_ev=50.0,
            plasmon_energy_points=401,
        )
        lindhard = _compute(
            dataset,
            context,
            plasmon_model="lindhard",
            plasmon_execution_mode="benchmark",
            plasmon_photon_energy_kev=8.0,
            plasmon_scattering_angle_deg=40.0,
            plasmon_energy_window_ev=50.0,
            plasmon_energy_points=401,
        )
        self.assertEqual(rpa.benchmark_status, "invalid_for_benchmark")
        self.assertEqual(lindhard.benchmark_status, "valid")
        self.assertGreater(lindhard.spectrum_energy_ev.size, 0)


if __name__ == "__main__":
    unittest.main()
