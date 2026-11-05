"""The shock breakout threshold is a parameter, and its default is unchanged.

`track_shock_front` used to compare the tracked front against a fixed 0.5 from
the near side of the interface axis. That threshold is now
`DerivedAnalysisParameters.breakout_threshold_fraction`, so the default must
reproduce the old values exactly and the CLI must reject an out-of-range value as
a usage error rather than a traceback.
"""
from __future__ import annotations

import types
import unittest

import _test_bootstrap  # noqa: F401

from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.shock_tracking import breakout_threshold_index


class BreakoutThresholdTests(unittest.TestCase):
    def test_default_reproduces_the_previous_constants(self) -> None:
        # high_to_low was a literal 0.5.
        self.assertEqual(breakout_threshold_index(3, "high_to_low", 0.5), 0.5)
        # low_to_high was `n_interfaces - 1.5`.
        for n_interfaces in (1, 2, 3, 5):
            self.assertEqual(
                breakout_threshold_index(n_interfaces, "low_to_high", 0.5),
                float(max(0, n_interfaces - 1.5)),
            )

    def test_fraction_moves_the_threshold_monotonically(self) -> None:
        high = [breakout_threshold_index(4, "high_to_low", value) for value in (0.25, 0.5, 0.75)]
        low = [breakout_threshold_index(4, "low_to_high", value) for value in (0.25, 0.5, 0.75)]
        self.assertEqual(high, sorted(high))
        self.assertEqual(low, sorted(low, reverse=True))

    def test_out_of_range_fraction_is_rejected(self) -> None:
        for value in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                breakout_threshold_index(3, "high_to_low", value)

    def test_parameter_enters_the_shock_cache_key(self) -> None:
        default = DerivedAnalysisParameters()
        changed = DerivedAnalysisParameters(breakout_threshold_fraction=0.25)
        self.assertNotEqual(default.core_key(), changed.core_key())
        self.assertEqual(default.core_key(), DerivedAnalysisParameters().core_key())


class BreakoutFractionCliTests(unittest.TestCase):
    def _args(self, fraction: float) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            path="unused.log",
            module=["shock"],
            snapshot=None,
            include_experimental=False,
            breakout_fraction=fraction,
            max_array=64,
        )

    def test_out_of_range_fraction_is_a_usage_error(self) -> None:
        from helios_diag import analyse as analyse_module
        from helios_diag.envelope import UsageError

        for value in (0.0, 1.0, -0.25):
            with self.assertRaises(UsageError):
                analyse_module.run(self._args(value))


if __name__ == "__main__":
    unittest.main()