"""``helios_diag stats``: numbers, the flat-zone guard, and NaN accounting.

The reductions are compared against numpy computed independently in the test, so
a change in the fast path cannot silently change an answer.
"""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import _test_bootstrap  # noqa: F401
import numpy as np

from helios_diag import stats as stats_module
from helios_parser import HeliosRun, write_hdf5

ROOT = Path(__file__).resolve().parents[1]
LOG_SAMPLE = ROOT / "tests" / "fixtures" / "v12" / "cu_v12_prefix.log"


def _args(**overrides):
    base = dict(
        path=None,
        field="density",
        over=None,
        at=None,
        at_time=None,
        global_stats=False,
        quantiles="5,50,95",
        region=None,
        material=None,
        by_region=False,
        by_material=False,
        flat_atol=1e-12,
        flat_rtol=1e-9,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class StatsNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.run_h5 = Path(cls._tmpdir.name) / "stats.h5"
        write_hdf5(LOG_SAMPLE, cls.run_h5, overwrite=True)
        cls.handle = HeliosRun(cls.run_h5)
        cls.data = np.asarray(cls.handle.get_field("density"), dtype=np.float64)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.handle.close()
        cls._tmpdir.cleanup()

    def test_per_zone_statistics_match_numpy(self) -> None:
        result = stats_module.run(_args(path=str(self.run_h5)))
        rows = result.rows
        self.assertEqual(len(rows), self.data.shape[1])
        for zone in (0, 7, self.data.shape[1] - 1):
            column = self.data[:, zone]
            self.assertAlmostEqual(rows[zone]["mean"], float(np.nanmean(column)), places=12)
            self.assertAlmostEqual(rows[zone]["min"], float(np.nanmin(column)), places=12)
            self.assertAlmostEqual(rows[zone]["max"], float(np.nanmax(column)), places=12)
            for name, quantile in (("p05", 5), ("p50", 50), ("p95", 95)):
                self.assertAlmostEqual(rows[zone][name], float(np.nanpercentile(column, quantile)), places=12)

    def test_per_snapshot_statistics_match_numpy(self) -> None:
        result = stats_module.run(_args(path=str(self.run_h5), over="time"))
        rows = result.rows
        self.assertEqual(len(rows), self.data.shape[0])
        for snapshot in (0, self.data.shape[0] - 1):
            self.assertAlmostEqual(rows[snapshot]["mean"], float(np.nanmean(self.data[snapshot])), places=12)

    def test_the_profile_at_a_snapshot_matches_the_time_row(self) -> None:
        over_time = stats_module.run(_args(path=str(self.run_h5), over="time")).rows
        at_snapshot = stats_module.run(_args(path=str(self.run_h5), at=3))
        self.assertEqual(len(at_snapshot.rows), self.data.shape[1])
        self.assertAlmostEqual(float(np.mean([row["value"] for row in at_snapshot.rows])), over_time[3]["mean"], places=12)

    def test_global_statistics_match_numpy(self) -> None:
        result = stats_module.run(_args(path=str(self.run_h5), global_stats=True))
        row = result.rows[0]
        self.assertAlmostEqual(row["mean"], float(np.nanmean(self.data)), places=12)
        self.assertAlmostEqual(row["max"], float(np.nanmax(self.data)), places=12)
        self.assertEqual(row["p95"], float(np.nanpercentile(self.data, 95)))

    def test_the_accounting_adds_up(self) -> None:
        result = stats_module.run(_args(path=str(self.run_h5)))
        accounting = result.payload["accounting"]
        self.assertEqual(accounting["n_values"], int(self.data.size))
        self.assertEqual(accounting["n_used"] + accounting["n_nan"], accounting["n_values"])

    def test_the_requested_quantiles_become_the_columns(self) -> None:
        result = stats_module.run(_args(path=str(self.run_h5), quantiles="1,99"))
        self.assertIn("p01", result.rows[0])
        self.assertIn("p99", result.rows[0])
        self.assertNotIn("p95", result.rows[0])


class FlatZoneGuardTests(unittest.TestCase):
    """A constant zone must not report a peak at t=0."""

    def test_a_flat_zone_reports_no_peak(self) -> None:
        import numpy as np

        times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        low = np.asarray([1.0, 1.0], dtype=np.float64)
        high = np.asarray([1.0, 5.0], dtype=np.float64)
        peak_index = np.asarray([0, 1])
        peaks = stats_module._peak_times(peak_index, low, high, times, 1e-12, 1e-9)
        self.assertIsNone(peaks[0], "a constant zone has no peak")
        self.assertEqual(peaks[1], 1.0)

    def test_a_zone_with_real_spread_reports_its_peak(self) -> None:
        import numpy as np

        times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        peaks = stats_module._peak_times(
            np.asarray([2]), np.asarray([0.0]), np.asarray([3.0]), times, 1e-12, 1e-9
        )
        self.assertEqual(peaks[0], 2.0)


class NanHandlingTests(unittest.TestCase):
    def test_accounting_counts_skipped_values(self) -> None:
        import numpy as np

        block = np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float64)
        accounting = stats_module._accounting(block, clean=False)
        self.assertEqual(accounting, {"n_values": 4, "n_nan": 1, "n_used": 3})

    def test_a_clean_block_skips_the_second_pass(self) -> None:
        import numpy as np

        block = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        self.assertEqual(stats_module._accounting(block, clean=True)["n_nan"], 0)

    def test_the_two_reduction_paths_agree_on_clean_data(self) -> None:
        import numpy as np

        block = np.arange(12, dtype=np.float64).reshape(4, 3) + 1.0
        plain = stats_module._reduce(block, 0, [50.0], clean=True)
        aware = stats_module._reduce(block, 0, [50.0], clean=False)
        np.testing.assert_allclose(plain["mean"], aware["mean"])
        np.testing.assert_allclose(plain["quantiles"], aware["quantiles"])


if __name__ == "__main__":
    unittest.main()