from __future__ import annotations

import math
import unittest

import _test_bootstrap  # noqa: F401
import numpy as np

from helios.services.derived.shock_tracking import _least_squares_tanh_fit


def _reference_fit(x: np.ndarray, values: tuple[np.ndarray, ...], initial_xf: float, initial_width: float):
    """The pre-2.0 per-system lstsq search, kept as the correctness oracle."""

    valid_fields = [
        np.asarray(field, dtype=np.float64)
        for field in values
        if field.size == x.size and np.count_nonzero(np.isfinite(field)) >= 3
    ]
    if x.size < 4 or not valid_fields:
        return None
    finite_dx = np.diff(np.asarray(x, dtype=np.float64))
    positive_dx = finite_dx[np.isfinite(finite_dx) & (finite_dx > 0.0)]
    cell_width = float(np.nanmedian(positive_dx)) if positive_dx.size else max(float(initial_width), 1.0e-8)
    width_candidates = np.linspace(max(cell_width * 0.75, 1.0e-8), max(cell_width * 4.0, cell_width), 4)
    x_candidates = np.linspace(float(initial_xf - cell_width), float(initial_xf + cell_width), 5)
    best_error = math.inf
    best_xf = float(initial_xf)
    best_width = max(float(initial_width), cell_width)
    for x_f in x_candidates:
        for width in width_candidates:
            basis = np.column_stack(
                (np.ones(x.size, dtype=np.float64), np.tanh((x - float(x_f)) / max(float(width), 1.0e-12)))
            )
            total_error = 0.0
            for field in valid_fields:
                finite = np.isfinite(field) & np.all(np.isfinite(basis), axis=1)
                if np.count_nonzero(finite) < 3:
                    continue
                solution, *_ = np.linalg.lstsq(basis[finite], field[finite], rcond=None)
                residual = field[finite] - basis[finite] @ solution
                amplitude = max(float(np.nanstd(field[finite])), 1.0e-12)
                total_error += float(np.sqrt(np.mean((residual / amplitude) ** 2)))
            if total_error < best_error:
                best_error = total_error
                best_xf = float(x_f)
                best_width = float(width)
    if not np.isfinite(best_error):
        return None
    return best_xf, best_width


class LeastSquaresTanhFitTest(unittest.TestCase):
    def test_matches_the_per_system_reference(self) -> None:
        rng = np.random.default_rng(20261103)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        for _ in range(200):
            position = float(rng.uniform(0.5, 3.5))
            width = float(rng.uniform(0.2, 2.0))
            base = np.tanh((x - position) / width)
            fields = tuple(base * float(rng.uniform(0.1, 10.0)) + rng.normal(0.0, 0.02, x.size) for _ in range(3))
            fit = _least_squares_tanh_fit(x, fields, position, width)
            reference = _reference_fit(x, fields, position, width)
            self.assertIsNotNone(fit)
            self.assertIsNotNone(reference)
            assert fit is not None and reference is not None
            self.assertAlmostEqual(fit.position_cm, reference[0], delta=1.0e-12)
            self.assertAlmostEqual(fit.width_cm, reference[1], delta=1.0e-12)

    def test_degenerate_window_still_returns_a_fit(self) -> None:
        # A window where every candidate basis is near-collinear drives the
        # normal equations singular. The fallback must keep the fit alive.
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        fit = _least_squares_tanh_fit(x, (field,), 1.0, 1.0)
        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertTrue(math.isfinite(fit.position_cm))
        self.assertTrue(math.isfinite(fit.width_cm))
        self.assertTrue(math.isfinite(fit.fit_quality))

    def test_non_finite_coordinate_is_ignored(self) -> None:
        x = np.array([0.0, 1.0, np.nan, 3.0, 4.0], dtype=np.float64)
        field = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        fit = _least_squares_tanh_fit(x, (field,), 2.0, 1.0)
        reference = _reference_fit(x, (field,), 2.0, 1.0)
        self.assertIsNotNone(fit)
        self.assertIsNotNone(reference)
        assert fit is not None and reference is not None
        self.assertAlmostEqual(fit.position_cm, reference[0], delta=1.0e-12)
        self.assertAlmostEqual(fit.width_cm, reference[1], delta=1.0e-12)

    def test_too_few_points_returns_none(self) -> None:
        x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        field = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        self.assertIsNone(_least_squares_tanh_fit(x, (field,), 1.0, 1.0))

    def test_all_non_finite_field_returns_none(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        field = np.full(5, np.nan, dtype=np.float64)
        self.assertIsNone(_least_squares_tanh_fit(x, (field,), 2.0, 1.0))


if __name__ == "__main__":
    unittest.main()