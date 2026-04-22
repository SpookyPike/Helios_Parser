"""Explicit cache layers for HELIOS Analyzer.

The application now distinguishes between three cache scopes:

- raw_data_cache: reader/parser level payload reuse
- derived_cache: derived-analysis products and intermediate results
- view_cache: display-layer conversions and render helpers

Each cache layer exposes named buckets with simple explicit invalidation and
lightweight observability via size/capacity/hit/miss/eviction counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CacheBucketStats:
    size: int
    capacity: int | None
    hits: int
    misses: int
    evictions: int
    clears: int
    last_clear_reason: str | None


@dataclass(frozen=True, slots=True)
class CacheLayerStats:
    bucket_count: int
    buckets: dict[str, CacheBucketStats]
    last_clear_reason: str | None


@dataclass(slots=True)
class BoundedCacheBucket(MutableMapping[Any, Any]):
    """Small LRU-style bucket used for long-lived cache layers.

    The cache stays intentionally simple: every bucket keeps insertion/access
    order and evicts the oldest entries once `max_items` is exceeded. This
    bounds session memory growth without changing the existing dict-like API
    used throughout the viewer and analysis layers.
    """

    max_items: int | None = None
    _items: OrderedDict[Any, Any] = field(default_factory=OrderedDict)
    _hits: int = 0
    _misses: int = 0
    _evictions: int = 0
    _clears: int = 0
    _last_clear_reason: str | None = None

    def __getitem__(self, key: Any) -> Any:
        value = self._items[key]
        self._hits += 1
        self._items.move_to_end(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        self._evict_if_needed()

    def __delitem__(self, key: Any) -> None:
        del self._items[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self._items:
            value = self._items[key]
            self._hits += 1
            self._items.move_to_end(key)
            return value
        self._misses += 1
        return default

    def clear(self, *, reason: str | None = None) -> None:
        self._items.clear()
        self._clears += 1
        self._last_clear_reason = None if reason is None else str(reason)

    def stats(self) -> CacheBucketStats:
        return CacheBucketStats(
            size=len(self._items),
            capacity=None if self.max_items is None else int(self.max_items),
            hits=int(self._hits),
            misses=int(self._misses),
            evictions=int(self._evictions),
            clears=int(self._clears),
            last_clear_reason=self._last_clear_reason,
        )

    def _evict_if_needed(self) -> None:
        if self.max_items is None or self.max_items <= 0:
            return
        while len(self._items) > int(self.max_items):
            self._items.popitem(last=False)
            self._evictions += 1


@dataclass(slots=True)
class CacheLayer:
    """Named cache buckets with explicit invalidation."""

    default_bucket_size: int | None = None
    _buckets: dict[str, BoundedCacheBucket] = field(default_factory=dict)
    _last_clear_reason: str | None = None

    def bucket(self, name: str, *, max_items: int | None = None) -> BoundedCacheBucket:
        """Return the named mutable cache bucket, creating it if needed."""

        key = str(name)
        if key not in self._buckets:
            self._buckets[key] = BoundedCacheBucket(max_items=self.default_bucket_size if max_items is None else max_items)
        elif max_items is not None:
            self._buckets[key].max_items = int(max_items)
            self._buckets[key]._evict_if_needed()
        return self._buckets[key]

    def clear_bucket(self, name: str, *, reason: str | None = None) -> None:
        """Remove a single bucket if it exists."""

        key = str(name)
        bucket = self._buckets.pop(key, None)
        if bucket is not None:
            bucket.clear(reason=reason or f"clear_bucket:{key}")
            self._last_clear_reason = reason or f"clear_bucket:{key}"

    def clear(self, *, reason: str | None = None) -> None:
        """Clear the entire cache layer."""

        self._last_clear_reason = None if reason is None else str(reason)
        for key, bucket in list(self._buckets.items()):
            bucket.clear(reason=reason or f"clear_layer:{key}")
        self._buckets.clear()

    def stats(self) -> CacheLayerStats:
        return CacheLayerStats(
            bucket_count=len(self._buckets),
            buckets={name: bucket.stats() for name, bucket in self._buckets.items()},
            last_clear_reason=self._last_clear_reason,
        )


@dataclass(slots=True)
class AnalyzerCacheSet:
    """Top-level cache groups used by the current platform."""

    # These are intentionally conservative defaults. Each layer stays bounded,
    # while hot-path buckets can still override per-bucket limits if needed.
    raw_data_cache: CacheLayer = field(default_factory=lambda: CacheLayer(default_bucket_size=4))
    derived_cache: CacheLayer = field(default_factory=lambda: CacheLayer(default_bucket_size=24))
    view_cache: CacheLayer = field(default_factory=lambda: CacheLayer(default_bucket_size=16))

    def stats(self) -> dict[str, CacheLayerStats]:
        return {
            "raw_data_cache": self.raw_data_cache.stats(),
            "derived_cache": self.derived_cache.stats(),
            "view_cache": self.view_cache.stats(),
        }


_SESSION_RAW_DATA_CACHE = CacheLayer(default_bucket_size=64)


def get_session_raw_data_cache() -> CacheLayer:
    """Return the process-local raw-data cache shared by viewer and derived.

    This cache stays intentionally narrow in scope: it only stores already
    loaded run metadata/arrays so viewer and derived paths can reuse the same
    in-memory data without changing ownership of their existing local caches.
    """

    return _SESSION_RAW_DATA_CACHE


def clear_session_raw_data_cache(*, reason: str | None = None) -> None:
    get_session_raw_data_cache().clear(reason=reason or "session_raw_cache_cleared")


def run_identity_for_path(path: str | Path) -> tuple[object, ...]:
    """Return a stable run identity for session-level raw-data reuse."""

    candidate = Path(path)
    resolved = candidate.resolve(strict=False)
    try:
        stat = candidate.stat()
    except OSError:
        return (str(resolved), None, None)
    return (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
