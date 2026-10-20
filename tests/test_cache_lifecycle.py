from __future__ import annotations

import unittest

import numpy as np

from helios.cache import CacheLayer, estimate_nbytes


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


if __name__ == "__main__":
    unittest.main()
