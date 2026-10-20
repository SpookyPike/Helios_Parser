from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import _test_bootstrap  # noqa: F401
import numpy as np
from helios.services.derived.preheat import _cumulative_energy_difference_budget, _rate_integration_budget
from helios.services.derived.preheat import evaluate_preheat
from test_derived_services_phase4 import _synthetic_dataset


class PreheatEnergyBudgetTests(unittest.TestCase):
    def _evaluate_residual(self, *, include_zero_sources: bool):
        dataset, _ = _synthetic_dataset()
        dataset.density_g_cm3[:] = 1.0
        dataset.ion_energy_j_g = np.repeat(np.arange(5)[:, None] * 1.0e7, 10, axis=1)
        dataset.electron_energy_j_g = dataset.ion_energy_j_g.copy()
        dataset.radiation_energy_j_g = dataset.ion_energy_j_g.copy()
        if include_zero_sources:
            dataset.laser_deposition_j_g_s = np.zeros_like(dataset.density_g_cm3)
            dataset.radiation_net_heating_j_g_s = np.zeros_like(dataset.density_g_cm3)
        branch = SimpleNamespace(branch_id="test", support_class="tracked", significance=1.0)
        target = {"region_id": 2, "zone_start": 5, "zone_stop": 10, "material_index": 2, "label": "Region 2"}
        selection = (target, target, target, None, 4.0e-9, "Region 1 -> 2", 5, "user_selected", ())
        with mock.patch("helios.services.derived.preheat._primary_compressive_branch", return_value=branch), mock.patch(
            "helios.services.derived.preheat._select_target_region", return_value=selection
        ):
            result = evaluate_preheat(dataset, wave_tracking=SimpleNamespace(), target_region_id=2)
        return next(row for row in result.budget_rows if row.key == "residual")

    def test_residual_is_unavailable_when_source_channels_are_missing(self) -> None:
        residual = self._evaluate_residual(include_zero_sources=False)
        self.assertFalse(residual.available)
        self.assertIsNone(residual.integrated_value)
        self.assertIsNone(residual.coverage_fraction)

    def test_explicit_zero_source_channels_produce_a_covered_residual(self) -> None:
        residual = self._evaluate_residual(include_zero_sources=True)
        self.assertTrue(residual.available)
        self.assertIsNotNone(residual.integrated_value)
        self.assertEqual(1.0, residual.coverage_fraction)

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
