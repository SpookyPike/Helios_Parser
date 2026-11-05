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
import numpy as np

from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.shock_tracking import (
    _interpolate_threshold_crossing,
    breakout_threshold_index,
    stalled_interface,
)


class BreakoutThresholdTests(unittest.TestCase):
    def test_threshold_is_a_position_on_the_interface_axis(self) -> None:
        # 2.2.0 returned the fraction itself, which the tracker then compared
        # against a zone index, so every value meant "reach the rear-most cell".
        self.assertEqual(breakout_threshold_index(300, "high_to_low", 0.5), 149.5)
        self.assertEqual(breakout_threshold_index(300, "low_to_high", 0.5), 149.5)
        self.assertAlmostEqual(breakout_threshold_index(300, "high_to_low", 0.95), 14.95)
        self.assertAlmostEqual(breakout_threshold_index(300, "low_to_high", 0.95), 284.05)
        # A single-interface mesh has one reachable index.
        self.assertEqual(breakout_threshold_index(2, "high_to_low", 0.5), 0.5)

    def test_two_fractions_cross_at_two_different_times(self) -> None:
        # A monotone track from interface 10 down to 0 over ten snapshots.
        time_s = np.linspace(0.0, 1.0, 11)
        track = np.linspace(10.0, 0.0, 11)
        early = _interpolate_threshold_crossing(
            time_s, track, threshold=breakout_threshold_index(11, "high_to_low", 0.2), direction="high_to_low"
        )[1]
        late = _interpolate_threshold_crossing(
            time_s, track, threshold=breakout_threshold_index(11, "high_to_low", 0.9), direction="high_to_low"
        )[1]
        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        self.assertLess(early, late)

    def test_fraction_moves_the_threshold_monotonically(self) -> None:
        # A larger fraction is further along the track: for high_to_low that is a
        # lower interface index, for low_to_high a higher one.
        high = [breakout_threshold_index(4, "high_to_low", value) for value in (0.25, 0.5, 0.75)]
        low = [breakout_threshold_index(4, "low_to_high", value) for value in (0.25, 0.5, 0.75)]
        self.assertEqual(high, sorted(high, reverse=True))
        self.assertEqual(low, sorted(low))

    def test_out_of_range_fraction_is_rejected(self) -> None:
        for value in (0.0, 1.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                breakout_threshold_index(3, "high_to_low", value)

    def test_parameter_enters_the_shock_cache_key(self) -> None:
        default = DerivedAnalysisParameters()
        changed = DerivedAnalysisParameters(breakout_threshold_fraction=0.25)
        self.assertNotEqual(default.core_key(), changed.core_key())
        self.assertEqual(default.core_key(), DerivedAnalysisParameters().core_key())


class StalledInterfaceTests(unittest.TestCase):
    def test_track_that_stops_on_a_boundary_is_reported(self) -> None:
        track = np.array([20.0, 14.0, 10.0, 9.0, 9.0, 9.0])
        self.assertEqual(stalled_interface(track, [(1, 2, 10)]), (1, 2, 10))

    def test_track_that_stops_away_from_a_boundary_is_not(self) -> None:
        track = np.array([20.0, 14.0, 12.0, 11.0])
        self.assertIsNone(stalled_interface(track, [(1, 2, 10)]))

    def test_a_track_still_moving_near_a_boundary_is_not_a_stall(self) -> None:
        # Ends within one interface of the boundary but the tail is not flat.
        track = np.array([20.0, 16.0, 12.0, 10.8, 10.4, 10.1])
        self.assertIsNone(stalled_interface(track, [(1, 2, 10)]))

    def test_a_slowly_moving_track_near_a_boundary_is_not_a_stall(self) -> None:
        # Three samples spanning 0.6 interface units: moving, not parked.
        track = np.array([20.0, 16.0, 12.0, 10.9, 10.6, 10.3])
        self.assertIsNone(stalled_interface(track, [(1, 2, 10)]))

    def test_empty_track_is_not_a_stall(self) -> None:
        self.assertIsNone(stalled_interface(np.array([np.nan, np.nan]), [(1, 2, 10)]))


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