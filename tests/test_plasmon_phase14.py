from __future__ import annotations

import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.services.derived.plasmon_lindhard import clear_finite_t_lindhard_cache, finite_t_lindhard_cache_info, finite_t_lindhard_susceptibility
from helios.services.derived.plasmon_validation import compute_plasmon, uniform_al_dataset


class PlasmonPhase14Tests(unittest.TestCase):
    def test_lindhard_benchmark_mode_uses_lighter_refined_grid(self) -> None:
        dataset, context = uniform_al_dataset(4.2, 0.5)
        result = compute_plasmon(
            dataset,
            context,
            plasmon_model='lindhard',
            plasmon_execution_mode='benchmark',
            plasmon_integration_mode='effective_state',
            plasmon_photon_energy_kev=8.307,
            plasmon_scattering_angle_deg=20.0,
            plasmon_energy_window_ev=40.0,
            plasmon_energy_points=401,
            plasmon_instrument_fwhm_ev=0.2,
            plasmon_electron_policy='valence_locked',
        )
        self.assertEqual(result.execution_mode, 'benchmark')
        self.assertGreaterEqual(int(result.spectrum_points), 1201)
        self.assertLess(int(result.spectrum_points), 4001)
        self.assertEqual(result.peak_fit_method, 'local_quadratic')

    def test_finite_t_lindhard_cache_hits_on_repeat_request(self) -> None:
        clear_finite_t_lindhard_cache()
        energy = np.linspace(-20.0, 20.0, 301, dtype=np.float64)
        first = finite_t_lindhard_susceptibility(
            energy,
            k_m_inv=1.0e10,
            te_ev=0.5,
            ne_cm3=2.8e23,
            imag_shift_ev=0.2,
            benchmark=True,
        )
        second = finite_t_lindhard_susceptibility(
            energy,
            k_m_inv=1.0e10,
            te_ev=0.5,
            ne_cm3=2.8e23,
            imag_shift_ev=0.2,
            benchmark=True,
        )
        info = finite_t_lindhard_cache_info()
        self.assertGreaterEqual(int(info['misses']), 1)
        self.assertGreaterEqual(int(info['hits']), 1)
        self.assertTrue(np.allclose(first, second, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
