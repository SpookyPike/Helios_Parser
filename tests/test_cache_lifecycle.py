from __future__ import annotations

import unittest

from helios.cache import CacheLayer


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


if __name__ == "__main__":
    unittest.main()
