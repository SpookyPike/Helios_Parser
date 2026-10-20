from __future__ import annotations

import unittest

import _test_bootstrap  # noqa: F401
import numpy as np
from helios.services.derived.preheat import _cumulative_energy_difference_budget, _rate_integration_budget


class PreheatEnergyBudgetTests(unittest.TestCase):
    def test_cumulative_energy_uses_window_difference(self) -> None:
        energy = np.asarray([[5.0], [15.0], [25.0]])
        cumulative, budget, coverage = _cumulative_energy_difference_budget(
            energy,
            reference_areal_mass=np.asarray([1.0]),
            target_mask=np.asarray([True]),
            time_indices=np.asarray([1, 2]),
        )
        np.testing.assert_array_equal(cumulative[[1, 2]], [0.0, 10.0])
        self.assertEqual(budget, 10.0)
        self.assertEqual(coverage, 1.0)

    def test_rate_integration_uses_reference_mass_and_nonuniform_dt(self) -> None:
        rate = np.asarray([[2.0], [4.0], [6.0]])
        cumulative, budget, coverage = _rate_integration_budget(
            rate,
            reference_areal_mass=np.asarray([1.0]),
            target_mask=np.asarray([True]),
            time_s=np.asarray([0.0, 1.0, 3.0]),
            time_indices=np.asarray([0, 1, 2]),
        )
        np.testing.assert_allclose(cumulative[[0, 1, 2]], [0.0, 3.0, 13.0])
        self.assertEqual(budget, 13.0)
        self.assertEqual(coverage, 1.0)

    def test_missing_zone_data_makes_budget_unavailable_with_coverage(self) -> None:
        rate = np.asarray([[1.0, np.nan], [2.0, 3.0]])
        cumulative, budget, coverage = _rate_integration_budget(
            rate,
            reference_areal_mass=np.asarray([1.0, 1.0]),
            target_mask=np.asarray([True, True]),
            time_s=np.asarray([0.0, 1.0]),
            time_indices=np.asarray([0, 1]),
        )
        self.assertIsNone(cumulative)
        self.assertIsNone(budget)
        self.assertEqual(coverage, 0.75)

    def test_source_integral_matches_cumulative_energy_difference(self) -> None:
        times = np.asarray([0.0, 1.0e-9, 2.0e-9])
        rate = np.full((3, 1), 1.0e10)
        cumulative, rate_budget, _ = _rate_integration_budget(
            rate,
            reference_areal_mass=np.asarray([1.0]),
            target_mask=np.asarray([True]),
            time_s=times,
            time_indices=np.asarray([0, 1, 2]),
        )
        _, energy_budget, _ = _cumulative_energy_difference_budget(
            np.asarray([[0.0], [10.0], [20.0]]),
            reference_areal_mass=np.asarray([1.0]),
            target_mask=np.asarray([True]),
            time_indices=np.asarray([0, 1, 2]),
        )
        self.assertEqual(rate_budget, 20.0)
        self.assertEqual(energy_budget, 20.0)
        self.assertEqual(cumulative[-1], 20.0)


if __name__ == "__main__":
    unittest.main()
