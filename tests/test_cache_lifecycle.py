from __future__ import annotations

import unittest

import numpy as np

from helios.cache import AnalyzerCacheSet, CacheCoordinator, CacheLayer, estimate_nbytes


class CacheLifecycleTests(unittest.TestCase):
    def test_clear_preserves_bucket_identity_and_stats_across_reuse(self) -> None:
        layer = CacheLayer(default_bucket_size=4)
        bucket = layer.bucket("results")
        bucket["first"] = 1

        layer.clear(reason="test-clear")
        self.assertIs(bucket, layer.bucket("results"))
        self.assertEqual(0, len(bucket))
        self.assertEqual(1, bucket.stats().clears)
        self.assertEqual("test-clear", bucket.stats().last_clear_reason)

        bucket["second"] = 2
        layer.clear(reason="test-clear-again")
        self.assertIs(bucket, layer.bucket("results"))
        self.assertEqual(2, bucket.stats().clears)
        self.assertEqual(2, layer.stats().buckets["results"].clears)

    def test_clear_bucket_preserves_identity_and_remove_is_explicit(self) -> None:
        layer = CacheLayer()
        bucket = layer.bucket("fields")
        bucket["density"] = object()

        layer.clear_bucket("fields", reason="run-retired")
        self.assertIs(bucket, layer.bucket("fields"))
        self.assertEqual(0, len(bucket))

        layer.remove_bucket("fields", reason="owner-destroyed")
        self.assertNotIn("fields", layer.stats().buckets)
        self.assertIsNot(bucket, layer.bucket("fields"))

    def test_byte_budget_counts_shared_array_bases_once_and_evicts_lru(self) -> None:
        layer = CacheLayer()
        bucket = layer.bucket("arrays", max_bytes=64)
        values = np.arange(4, dtype=np.float64)
        bucket["base"] = values
        bucket["view"] = values.view()
        self.assertEqual(32, bucket.stats().bytes_used)

        bucket["other"] = np.arange(4, dtype=np.float64)
        self.assertEqual(64, bucket.stats().bytes_used)
        bucket["third"] = np.arange(2, dtype=np.float64)
        self.assertNotIn("base", bucket)
        self.assertLessEqual(bucket.stats().bytes_used, 64)
        self.assertEqual(64, bucket.stats().byte_capacity)

    def test_individually_oversized_cache_values_are_rejected(self) -> None:
        bucket = CacheLayer().bucket("small", max_bytes=16)
        with self.assertRaisesRegex(ValueError, "bucket limit is 16"):
            bucket["too-large"] = np.arange(3, dtype=np.float64)
        self.assertEqual(0, len(bucket))
        self.assertEqual(24, estimate_nbytes(np.arange(3, dtype=np.float64)))

    def test_cache_set_accepts_injected_per_layer_budgets(self) -> None:
        cache_set = AnalyzerCacheSet(raw_bucket_byte_limit=32, derived_bucket_byte_limit=48)
        raw_bucket = cache_set.raw_data_cache.bucket("raw")
        derived_bucket = cache_set.derived_cache.bucket("derived")
        self.assertEqual(32, raw_bucket.stats().byte_capacity)
        self.assertEqual(48, derived_bucket.stats().byte_capacity)

    def test_aggregate_budget_evicts_oldest_entry_across_cache_owners(self) -> None:
        coordinator = CacheCoordinator(max_bytes=12)
        viewer = CacheLayer(coordinator=coordinator).bucket("viewer")
        derived = CacheLayer(coordinator=coordinator).bucket("derived")

        viewer["old"] = np.arange(8, dtype=np.uint8)
        derived["new"] = np.arange(8, dtype=np.uint8)

        self.assertNotIn("old", viewer)
        self.assertIn("new", derived)
        stats = coordinator.stats()
        self.assertEqual(8, stats.bytes_used)
        self.assertEqual(12, stats.byte_capacity)
        self.assertEqual(1, stats.evictions)

    def test_aggregate_accounting_deduplicates_arrays_shared_between_owners(self) -> None:
        coordinator = CacheCoordinator(max_bytes=24)
        viewer = CacheLayer(coordinator=coordinator).bucket("viewer")
        derived = CacheLayer(coordinator=coordinator).bucket("derived")
        shared = np.arange(16, dtype=np.uint8)

        viewer[""] = shared
        derived[""] = {"": shared[4:]}

        self.assertEqual(shared.nbytes, coordinator.stats().bytes_used)
        self.assertEqual(0, coordinator.stats().evictions)

    def test_analyzer_cache_sets_share_the_session_coordinator(self) -> None:
        viewer = AnalyzerCacheSet()
        derived = AnalyzerCacheSet()
        viewer_bucket = viewer.raw_data_cache.bucket("viewer_probe")
        derived_bucket = derived.derived_cache.bucket("derived_probe")

        self.assertIs(viewer_bucket.coordinator, derived_bucket.coordinator)
        self.assertGreaterEqual(viewer.memory_stats().registered_buckets, 2)

    def test_aggregate_limit_can_be_injected_across_analyzer_cache_sets(self) -> None:
        coordinator = CacheCoordinator(max_bytes=12)
        viewer = AnalyzerCacheSet(coordinator=coordinator)
        derived = AnalyzerCacheSet(coordinator=coordinator)
        viewer_bucket = viewer.raw_data_cache.bucket("viewer")
        derived_bucket = derived.derived_cache.bucket("derived")

        viewer_bucket["old"] = np.arange(8, dtype=np.uint8)
        derived_bucket["new"] = np.arange(8, dtype=np.uint8)

        self.assertNotIn("old", viewer_bucket)
        self.assertIn("new", derived_bucket)
        self.assertLessEqual(viewer.memory_stats().bytes_used, 12)


if __name__ == "__main__":
    unittest.main()
