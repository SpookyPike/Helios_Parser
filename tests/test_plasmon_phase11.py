from __future__ import annotations

import unittest

import numpy as np

import _test_bootstrap  # noqa: F401

from helios.services.derived.plasmon_reference_data import (
    GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5,
    USER_DRIVEN_AL_DISPERSION_REFERENCE,
)
from scripts.validate_plasmon_step8_dispersion import (
    _all_q_points,
    _metric_row,
    _series_points,
    _compute_model_grid,
)


class PlasmonPhase11Tests(unittest.TestCase):
    def test_reference_dispersion_inputs_expose_provenance_and_geometry(self) -> None:
        ambient = GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5
        driven = USER_DRIVEN_AL_DISPERSION_REFERENCE
        self.assertEqual(str(ambient["provenance"]["quality"]), "manual_digitization_v2")
        self.assertEqual(str(driven["provenance"]["quality"]), "manual_digitization_v2")
        self.assertAlmostEqual(float(ambient["geometry"]["xray_energy_kev"]), 8.307, places=3)
        self.assertAlmostEqual(float(driven["geometry"]["instrument_fwhm_ev"]), 3.5, places=6)
        self.assertIn("notes", dict(ambient["provenance"]))
        self.assertIn("notes", dict(driven["provenance"]))

    def test_reference_dispersion_series_are_monotonic(self) -> None:
        for dataset in (GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5, USER_DRIVEN_AL_DISPERSION_REFERENCE):
            for series in dict(dataset['series']).values():
                q = np.asarray(series['q_ang_inv'], dtype=np.float64)
                peak = np.asarray(series['peak_ev'], dtype=np.float64)
                self.assertTrue(np.all(np.diff(q) > 0.0))
                self.assertTrue(np.all(np.diff(peak) > 0.0))

    def test_extracted_experiment_shows_positive_driven_minus_ambient_shift(self) -> None:
        ambient_q, ambient_peak, _ = _series_points(GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5, 'experiment')
        driven_q, driven_peak, _ = _series_points(USER_DRIVEN_AL_DISPERSION_REFERENCE, 'experiment')
        self.assertTrue(np.allclose(ambient_q, driven_q, atol=0.02, rtol=0.0))
        self.assertTrue(np.all((driven_peak - ambient_peak) > 0.0))

    def test_rpa_family_scaffold_keeps_positive_compression_shift(self) -> None:
        ambient_grid = _compute_model_grid(2.70, 0.30, _all_q_points(GAWNE_2024_AMBIENT_AL_DISPERSION_FIGS5))
        driven_grid = _compute_model_grid(4.125, 0.60, _all_q_points(USER_DRIVEN_AL_DISPERSION_REFERENCE))
        q_shared = np.asarray([0.99, 1.28, 1.57, 2.57], dtype=np.float64)
        for model in ('rpa', 'rpa_static_lfc', 'auto_best'):
            ambient = np.asarray([float(ambient_grid[model][float(q)]['peak_ev']) for q in q_shared], dtype=np.float64)
            driven = np.asarray([float(driven_grid[model][float(q)]['peak_ev']) for q in q_shared], dtype=np.float64)
            self.assertTrue(np.all(np.isfinite(ambient)))
            self.assertTrue(np.all(np.isfinite(driven)))
            self.assertTrue(np.all((driven - ambient) > 0.0))

    def test_metric_row_reports_nan_when_model_has_no_finite_peaks(self) -> None:
        q = np.asarray([1.0, 1.5], dtype=np.float64)
        y = np.asarray([10.0, 12.0], dtype=np.float64)
        q_map = {1.0: {'peak_ev': float('nan')}, 1.5: {'peak_ev': float('nan')}}
        row = _metric_row('dummy', q_map, q, y)
        self.assertEqual(row['valid_points'], 0)
        self.assertTrue(np.isnan(float(row['mae_ev'])))


if __name__ == '__main__':
    unittest.main()
