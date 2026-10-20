from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
import numpy as np

from helios.services.derived.plasmon_spectrum import (
    _energy_axis_cache_key,
    _energy_axis_from_cache_key,
    _finite_t_susceptibility_cached,
    classical_response_cache_info,
    clear_classical_response_cache,
    finite_t_susceptibility,
)


class PlasmonEnergyGridCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_classical_response_cache()
        base = np.linspace(-2.0, 2.0, 65, dtype=np.float64)

        nearly_linear = base.copy()
        nearly_linear[27] += 5.0e-13

        t = np.linspace(-1.0, 1.0, 65, dtype=np.float64)
        quadratic = 2.0 * t + 0.1 * (t**2 - 1.0)

        unit = np.linspace(0.0, 1.0, 65, dtype=np.float64)
        logarithmic = -2.0 + 4.0 * (np.geomspace(1.0, 101.0, 65) - 1.0) / 100.0
        adaptive_unit = unit + 0.03 * np.sin(2.0 * np.pi * unit)
        adaptive = -2.0 + 4.0 * adaptive_unit

        self.axes = {
            "linear": base,
            "nearly_linear": nearly_linear,
            "quadratic": quadratic,
            "logarithmic": logarithmic,
            "adaptive": adaptive,
        }

    def test_keys_round_trip_exactly_and_distinguish_same_endpoint_grids(self) -> None:
        keys = {name: _energy_axis_cache_key(axis) for name, axis in self.axes.items()}

        self.assertEqual(len(set(keys.values())), len(self.axes))
        self.assertEqual(keys["linear"][0], "linspace")
        for name, axis in self.axes.items():
            with self.subTest(grid=name):
                self.assertTrue(
                    np.array_equal(_energy_axis_from_cache_key(keys[name]), axis)
                )

        for axis in self.axes.values():
            self.assertEqual(axis.size, 65)
            self.assertEqual(axis[0], -2.0)
            self.assertEqual(axis[-1], 2.0)

    def test_cache_keeps_each_grid_separate_and_matches_uncached_calculation(
        self,
    ) -> None:
        params = {
            "k_m_inv": 1.0e10,
            "te_ev": 120.0,
            "ne_cm3": 8.0e20,
            "imag_shift_ev": 0.01,
        }

        for name, axis in self.axes.items():
            with self.subTest(grid=name):
                key = _energy_axis_cache_key(axis)
                cached = finite_t_susceptibility(axis, **params)
                reference = _finite_t_susceptibility_cached.__wrapped__(key, **params)
                self.assertTrue(np.array_equal(cached, reference, equal_nan=True))

        info = classical_response_cache_info()
        self.assertEqual(info["finite_t_susceptibility_misses"], len(self.axes))
        self.assertEqual(info["finite_t_susceptibility_currsize"], len(self.axes))

        for axis in self.axes.values():
            finite_t_susceptibility(axis, **params)
        self.assertEqual(
            classical_response_cache_info()["finite_t_susceptibility_hits"],
            len(self.axes),
        )

    def test_rejects_non_vector_and_non_finite_energy_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            _energy_axis_cache_key(np.zeros((2, 2), dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "finite"):
            _energy_axis_cache_key(np.asarray([0.0, np.nan]))
        with self.assertRaisesRegex(ValueError, "finite"):
            _energy_axis_cache_key(np.asarray([0.0, np.inf]))


if __name__ == "__main__":
    unittest.main()
