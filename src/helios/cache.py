"""Explicit cache layers for HELIOS Analyzer.

The application now distinguishes between three cache scopes:

- raw_data_cache: reader/parser level payload reuse
- derived_cache: derived-analysis products and intermediate results
- view_cache: display-layer conversions and render helpers

Each cache layer exposes named buckets with simple explicit invalidation and
lightweight observability via size/capacity/hit/miss/eviction counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CacheBucketStats:
    size: int
    capacity: int | None
    hits: int
    misses: int
    evictions: int
    clears: int
    last_clear_reason: str | None
    bytes_used: int = 0
    byte_capacity: int | None = None


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
    max_bytes: int | None = None
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
        if self.max_bytes is not None and int(self.max_bytes) > 0 and estimate_nbytes(value) > int(self.max_bytes):
            raise ValueError(f"Cache entry requires {estimate_nbytes(value)} bytes; bucket limit is {int(self.max_bytes)}.")
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
            bytes_used=estimate_nbytes(tuple(self._items.values())),
            byte_capacity=None if self.max_bytes is None else int(self.max_bytes),
        )

    def _evict_if_needed(self) -> None:
        while self._items and (
            (self.max_items is not None and self.max_items > 0 and len(self._items) > int(self.max_items))
            or (self.max_bytes is not None and self.max_bytes > 0 and estimate_nbytes(tuple(self._items.values())) > int(self.max_bytes))
        ):
            self._items.popitem(last=False)
            self._evictions += 1


def estimate_nbytes(value: Any) -> int:
    """Estimate retained NumPy/object bytes, deduplicating shared array bases."""

    arrays: set[int] = set()
    objects: set[int] = set()

    def visit(item: Any) -> int:
        if isinstance(item, np.ndarray):
            base = item
            while isinstance(getattr(base, "base", None), np.ndarray):
                base = base.base
            identity = id(base)
            if identity in arrays:
                return 0
            arrays.add(identity)
            return int(base.nbytes)
        identity = id(item)
        if identity in objects:
            return 0
        if isinstance(item, dict):
            objects.add(identity)
            return sum(visit(key) + visit(child) for key, child in item.items())
        if isinstance(item, (tuple, list, set, frozenset)):
            objects.add(identity)
            return sum(visit(child) for child in item)
        if is_dataclass(item) and not isinstance(item, type):
            objects.add(identity)
            return sum(visit(getattr(item, descriptor.name)) for descriptor in fields(item))
        if isinstance(item, str):
            return len(item.encode("utf-8"))
        if isinstance(item, (bytes, bytearray, memoryview)):
            return len(item)
        return 0

    return int(visit(value))


@dataclass(slots=True)
class CacheLayer:
    """Named cache buckets with explicit invalidation."""

    default_bucket_size: int | None = None
    default_bucket_byte_limit: int | None = None
    _buckets: dict[str, BoundedCacheBucket] = field(default_factory=dict)
    _last_clear_reason: str | None = None

    def bucket(self, name: str, *, max_items: int | None = None, max_bytes: int | None = None) -> BoundedCacheBucket:
        """Return the named mutable cache bucket, creating it if needed."""

        key = str(name)
        if key not in self._buckets:
            self._buckets[key] = BoundedCacheBucket(
                max_items=self.default_bucket_size if max_items is None else max_items,
                max_bytes=self.default_bucket_byte_limit if max_bytes is None else max_bytes,
            )
        elif max_items is not None:
            self._buckets[key].max_items = int(max_items)
            self._buckets[key]._evict_if_needed()
        if key in self._buckets and max_bytes is not None:
            self._buckets[key].max_bytes = int(max_bytes)
            self._buckets[key]._evict_if_needed()
        return self._buckets[key]

    def clear_bucket(self, name: str, *, reason: str | None = None) -> None:
        """Empty a single bucket while preserving its registered identity."""

        key = str(name)
        bucket = self._buckets.get(key)
        if bucket is not None:
            bucket.clear(reason=reason or f"clear_bucket:{key}")
            self._last_clear_reason = reason or f"clear_bucket:{key}"

    def remove_bucket(self, name: str, *, reason: str | None = None) -> None:
        """Unregister a bucket explicitly when its identity is no longer owned."""

        key = str(name)
        bucket = self._buckets.pop(key, None)
        if bucket is not None:
            bucket.clear(reason=reason or f"remove_bucket:{key}")
            self._last_clear_reason = reason or f"remove_bucket:{key}"

    def clear(self, *, reason: str | None = None) -> None:
        """Clear the entire cache layer."""

        self._last_clear_reason = None if reason is None else str(reason)
        for key, bucket in list(self._buckets.items()):
            bucket.clear(reason=reason or f"clear_layer:{key}")

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
    raw_bucket_byte_limit: int | None = None
    derived_bucket_byte_limit: int | None = None
    view_bucket_byte_limit: int | None = None

    def __post_init__(self) -> None:
        self.raw_data_cache.default_bucket_byte_limit = self.raw_bucket_byte_limit
        self.derived_cache.default_bucket_byte_limit = self.derived_bucket_byte_limit
        self.view_cache.default_bucket_byte_limit = self.view_bucket_byte_limit

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
