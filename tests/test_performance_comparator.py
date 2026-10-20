from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_performance_suite import compare_profiles


def _profiles(before: list[float], after: list[float], *, metric: str = "latency_ms", direction: str = "lower"):
    settings = {"command": ["bench"], "warmups": 1, "samples": len(before), "metrics": {metric: direction}}
    baseline = {
        "schema_version": 1,
        "environment": {"python": "3.12"},
        "paired": True,
        "pair_seed": 3,
        "cases": {"case": {"fixtures": {"input": {"sha256": "abc"}}, "settings": settings,
                            "samples": [{"index": i, "metrics": {metric: value}} for i, value in enumerate(before)]}},
    }
    candidate = {
        "schema_version": 1,
        "environment": {"python": "3.12"},
        "paired": True,
        "pair_seed": 3,
        "cases": {"case": {"fixtures": {"input": {"sha256": "abc"}}, "settings": settings,
                            "samples": [{"index": i, "metrics": {metric: value}} for i, value in enumerate(after)]}},
    }
    return baseline, candidate


class PerformanceComparatorTests(unittest.TestCase):
    def test_rejects_incompatible_environments(self) -> None:
        baseline, candidate = _profiles([10] * 10, [9] * 10)
        candidate["environment"]["python"] = "3.13"
        with self.assertRaisesRegex(ValueError, "environments differ"):
            compare_profiles(baseline, candidate)

    def test_rejects_missing_metric(self) -> None:
        baseline, candidate = _profiles([10] * 10, [9] * 10)
        del candidate["cases"]["case"]["samples"][0]["metrics"]["latency_ms"]
        with self.assertRaisesRegex(ValueError, "missing metric"):
            compare_profiles(baseline, candidate)

    def test_rejects_insufficient_samples(self) -> None:
        baseline, candidate = _profiles([10] * 4, [9] * 4)
        with self.assertRaisesRegex(ValueError, "insufficient samples"):
            compare_profiles(baseline, candidate, minimum_samples=5)

    def test_reports_intentional_slowdown_as_failure(self) -> None:
        baseline, candidate = _profiles([10] * 10, [12] * 10)
        report = compare_profiles(baseline, candidate)
        self.assertFalse(report["passed"])
        metric = report["cases"]["case"]["metrics"]["latency_ms"]
        self.assertEqual("regressed", metric["verdict"])
        self.assertTrue(any("exceeds" in item for item in report["failures"]))

    def test_reports_memory_retention_regression(self) -> None:
        baseline, candidate = _profiles([100] * 10, [130] * 10, metric="retained_bytes")
        report = compare_profiles(baseline, candidate)
        self.assertFalse(report["passed"])
        self.assertIn("retained_bytes", report["cases"]["case"]["metrics"])

    def test_accepts_complete_memory_reduction_to_zero(self) -> None:
        baseline, candidate = _profiles([100] * 10, [0] * 10, metric="optional_array_bytes")
        report = compare_profiles(baseline, candidate)
        self.assertTrue(report["passed"])
        self.assertEqual("improved", report["cases"]["case"]["metrics"]["optional_array_bytes"]["verdict"])

    def test_inconclusive_unpaired_profiles_do_not_claim_improvement(self) -> None:
        baseline, candidate = _profiles([10] * 10, [9] * 10)
        baseline["paired"] = candidate["paired"] = False
        report = compare_profiles(baseline, candidate)
        self.assertEqual("unpaired_inconclusive", report["cases"]["case"]["metrics"]["latency_ms"]["verdict"])

    def test_checks_numerical_equivalence_within_declared_tolerance(self) -> None:
        baseline, candidate = _profiles([10] * 10, [9] * 10)
        for profile, shift in ((baseline, 0.0), (candidate, 1e-12)):
            case = profile["cases"]["case"]
            case["settings"]["equivalent_metrics"] = {"required_checksum": {"absolute_tolerance": 1e-10, "relative_tolerance": 0.0}}
            for sample in case["samples"]:
                sample["stdout_json"] = {"required_checksum": 42.0 + shift}
        report = compare_profiles(baseline, candidate)
        self.assertTrue(report["cases"]["case"]["equivalent_metrics"]["required_checksum"]["passed"])

    def test_numerical_mismatch_fails_the_comparison(self) -> None:
        baseline, candidate = _profiles([10] * 10, [9] * 10)
        for profile, checksum in ((baseline, 42.0), (candidate, 43.0)):
            case = profile["cases"]["case"]
            case["settings"]["equivalent_metrics"] = {"required_checksum": {"absolute_tolerance": 1e-10, "relative_tolerance": 0.0}}
            for sample in case["samples"]:
                sample["stdout_json"] = {"required_checksum": checksum}
        report = compare_profiles(baseline, candidate)
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
