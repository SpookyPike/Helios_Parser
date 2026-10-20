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
import weakref
import threading
from contextlib import nullcontext

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


@dataclass(frozen=True, slots=True)
class CacheCoordinatorStats:
    registered_buckets: int
    bytes_used: int
    byte_capacity: int | None
    evictions: int


class CacheCoordinator:
    """Optional aggregate byte budget for cache buckets with shared ownership.

    The coordinator accounts for shared NumPy allocations once across all
    registered buckets and evicts the globally least-recently-used cache entry
    when its configured limit is exceeded. Objects held by active callers or
    tasks are outside this cache-only budget.
    """

    def __init__(self, max_bytes: int | None = None) -> None:
        self.max_bytes = None if max_bytes is None else max(0, int(max_bytes))
        self._buckets: weakref.WeakValueDictionary[int, BoundedCacheBucket] = weakref.WeakValueDictionary()
        self._bucket_refs: dict[int, weakref.ReferenceType[BoundedCacheBucket]] = {}
        self._entry_allocations: dict[int, dict[Any, dict[int, int]]] = {}
        self._allocation_refs: dict[int, int] = {}
        self._allocation_sizes: dict[int, int] = {}
        self._bytes_used = 0
        self._clock = 0
        self._evictions = 0
        # One lock protects the allocation ledger and every bucket registered
        # with this coordinator. Bucket operations acquire this lock before
        # touching their OrderedDict, so global eviction cannot race a writer.
        self._lock = threading.RLock()

    def register(self, bucket: BoundedCacheBucket) -> None:
        with self._lock:
            bucket_id = id(bucket)
            if bucket_id in self._buckets:
                return
            self._buckets[id(bucket)] = bucket
            self._bucket_refs[bucket_id] = weakref.ref(bucket, lambda _ref, key=bucket_id: self._discard_bucket_id(key))
            self._entry_allocations[bucket_id] = {}
            for key, value in tuple(bucket._items.items()):
                self.entry_updated(bucket, key, value)

    def unregister(self, bucket: BoundedCacheBucket) -> None:
        with self._lock:
            self._discard_bucket_id(id(bucket))

    def _discard_bucket_id(self, bucket_id: int) -> None:
        with self._lock:
            if bucket_id not in self._buckets:
                if bucket_id not in self._entry_allocations:
                    self._bucket_refs.pop(bucket_id, None)
                    return
            entries = self._entry_allocations.pop(bucket_id, {})
            for allocations in entries.values():
                self._release_allocations(allocations)
            self._bucket_refs.pop(bucket_id, None)
            self._buckets.pop(bucket_id, None)

    def _release_allocations(self, allocations: dict[int, int]) -> None:
        for allocation_id in allocations:
            refs = self._allocation_refs.get(allocation_id, 0) - 1
            if refs <= 0:
                self._allocation_refs.pop(allocation_id, None)
                self._bytes_used -= self._allocation_sizes.pop(allocation_id, 0)
            else:
                self._allocation_refs[allocation_id] = refs

    def entry_updated(self, bucket: BoundedCacheBucket, key: Any, value: Any) -> None:
        with self._lock:
            self.entry_removed(bucket, key)
            allocations = _array_allocation_sizes(value)
            self._entry_allocations.setdefault(id(bucket), {})[key] = allocations
            for allocation_id, size in allocations.items():
                refs = self._allocation_refs.get(allocation_id, 0)
                if refs == 0:
                    self._allocation_sizes[allocation_id] = size
                    self._bytes_used += size
                self._allocation_refs[allocation_id] = refs + 1

    def entry_removed(self, bucket: BoundedCacheBucket, key: Any) -> None:
        with self._lock:
            allocations = self._entry_allocations.get(id(bucket), {}).pop(key, None)
            if allocations is None:
                return
            self._release_allocations(allocations)

    def clear_bucket(self, bucket: BoundedCacheBucket) -> None:
        with self._lock:
            for key in tuple(self._entry_allocations.get(id(bucket), {})):
                self.entry_removed(bucket, key)

    def retire_revision(self, revision: Any) -> int:
        """Remove cache entries owned by one explicitly opened run revision.

        Revision-scoped keys are used by shared reader and derived caches;
        viewer payload caches carry the revision on the payload value. Scanning
        those small bounded indexes lets the application retire a run across
        all registered cache owners without clearing data for another open run.
        """

        with self._lock:
            revision_key = getattr(revision, "cache_key", revision)
            removed = 0
            for bucket in tuple(self._buckets.values()):
                for key, value in tuple(bucket._items.items()):
                    if _contains_revision(key, revision_key) or _value_uses_revision(value, revision):
                        del bucket[key]
                        removed += 1
            return removed

    def touch(self, bucket: BoundedCacheBucket, key: Any) -> None:
        with self._lock:
            self._clock += 1
            bucket._entry_age[key] = self._clock

    def set_limit(self, max_bytes: int | None) -> None:
        with self._lock:
            self.max_bytes = None if max_bytes is None else max(0, int(max_bytes))
            self.enforce()

    def bytes_used(self) -> int:
        with self._lock:
            return int(self._bytes_used)

    def stats(self) -> CacheCoordinatorStats:
        with self._lock:
            return CacheCoordinatorStats(
                registered_buckets=len(self._buckets),
                bytes_used=self._bytes_used,
                byte_capacity=self.max_bytes,
                evictions=int(self._evictions),
            )

    def enforce(self) -> None:
        with self._lock:
            if self.max_bytes is None or self.max_bytes <= 0:
                return
            while self._buckets and self.bytes_used() > self.max_bytes:
                oldest: tuple[int, BoundedCacheBucket, Any] | None = None
                for bucket in tuple(self._buckets.values()):
                    if not bucket._items:
                        continue
                    key = next(iter(bucket._items))
                    age = bucket._entry_age.get(key, 0)
                    if oldest is None or age < oldest[0]:
                        oldest = (age, bucket, key)
                if oldest is None:
                    return
                _, bucket, key = oldest
                bucket._evict_key(key)
                self._evictions += 1


DEFAULT_AGGREGATE_CACHE_BYTE_LIMIT = 512 * 1024 * 1024
_DEFAULT_CACHE_COORDINATOR = CacheCoordinator(max_bytes=DEFAULT_AGGREGATE_CACHE_BYTE_LIMIT)


@dataclass(eq=False)
class BoundedCacheBucket(MutableMapping[Any, Any]):
    """Small LRU-style bucket used for long-lived cache layers.

    The cache stays intentionally simple: every bucket keeps insertion/access
    order and evicts the oldest entries once `max_items` is exceeded. This
    bounds session memory growth without changing the existing dict-like API
    used throughout the viewer and analysis layers.
    """

    max_items: int | None = None
    max_bytes: int | None = None
    coordinator: CacheCoordinator | None = field(default=None, repr=False)
    _items: OrderedDict[Any, Any] = field(default_factory=OrderedDict)
    _entry_age: dict[Any, int] = field(default_factory=dict, repr=False)
    _hits: int = 0
    _misses: int = 0
    _evictions: int = 0
    _clears: int = 0
    _last_clear_reason: str | None = None

    def _guard(self):
        return self.coordinator._lock if self.coordinator is not None else nullcontext()

    def __getitem__(self, key: Any) -> Any:
        with self._guard():
            value = self._items[key]
            self._hits += 1
            self._items.move_to_end(key)
            if self.coordinator is not None:
                self.coordinator.touch(self, key)
            return value

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._guard():
            size = estimate_nbytes(value)
            if self.max_bytes is not None and int(self.max_bytes) > 0 and size > int(self.max_bytes):
                self.pop(key, None)
                return
            if self.coordinator is not None and self.coordinator.max_bytes is not None and self.coordinator.max_bytes > 0 and size > self.coordinator.max_bytes:
                self.pop(key, None)
                return
            if self.coordinator is not None:
                self.coordinator.register(self)
            self._items[key] = value
            self._items.move_to_end(key)
            if self.coordinator is not None:
                self.coordinator.entry_updated(self, key, value)
                self.coordinator.touch(self, key)
            self._evict_if_needed()
            if self.coordinator is not None:
                self.coordinator.enforce()

    def __delitem__(self, key: Any) -> None:
        with self._guard():
            if self.coordinator is not None:
                self.coordinator.entry_removed(self, key)
            del self._items[key]
            self._entry_age.pop(key, None)

    def __iter__(self) -> Iterator[Any]:
        # Mapping views (for example ``values()``) call ``__getitem__`` while
        # iterating. Reads update LRU order, so iterate over a stable key
        # snapshot rather than the mutable OrderedDict view.
        with self._guard():
            return iter(tuple(self._items))

    def __len__(self) -> int:
        with self._guard():
            return len(self._items)

    def get(self, key: Any, default: Any = None) -> Any:
        with self._guard():
            if key in self._items:
                value = self._items[key]
                self._hits += 1
                self._items.move_to_end(key)
                if self.coordinator is not None:
                    self.coordinator.touch(self, key)
                return value
            self._misses += 1
            return default

    def clear(self, *, reason: str | None = None) -> None:
        with self._guard():
            if self.coordinator is not None:
                self.coordinator.clear_bucket(self)
            self._items.clear()
            self._entry_age.clear()
            self._clears += 1
            self._last_clear_reason = None if reason is None else str(reason)

    def stats(self) -> CacheBucketStats:
        with self._guard():
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
            key = next(iter(self._items))
            self._evict_key(key)

    def _evict_key(self, key: Any) -> None:
        if key in self._items:
            if self.coordinator is not None:
                self.coordinator.entry_removed(self, key)
            del self._items[key]
            self._entry_age.pop(key, None)
            self._evictions += 1


def _array_allocation_sizes(value: Any) -> dict[int, int]:
    """Return unique NumPy base allocations retained by one cache entry."""

    arrays: dict[int, int] = {}
    objects: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, np.ndarray):
            base = item
            while isinstance(getattr(base, "base", None), np.ndarray):
                base = base.base
            arrays[id(base)] = int(base.nbytes)
            return
        identity = id(item)
        if identity in objects:
            return
        if isinstance(item, dict):
            objects.add(identity)
            for key, child in item.items():
                visit(key)
                visit(child)
            return
        if isinstance(item, (tuple, list, set, frozenset)):
            objects.add(identity)
            for child in item:
                visit(child)
            return
        if is_dataclass(item) and not isinstance(item, type):
            objects.add(identity)
            for descriptor in fields(item):
                visit(getattr(item, descriptor.name))

    visit(value)
    return arrays


def _contains_revision(value: Any, revision_key: Any) -> bool:
    if value is revision_key:
        return True
    if isinstance(value, tuple):
        if value == revision_key:
            return True
        return any(_contains_revision(item, revision_key) for item in value)
    if isinstance(value, list):
        return any(_contains_revision(item, revision_key) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_revision(key, revision_key) or _contains_revision(item, revision_key)
            for key, item in value.items()
        )
    return False


def _value_uses_revision(value: Any, revision: Any) -> bool:
    value_revision = getattr(value, "revision", None)
    return value_revision is not None and value_revision == revision


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
    coordinator: CacheCoordinator | None = None
    _buckets: dict[str, BoundedCacheBucket] = field(default_factory=dict)
    _last_clear_reason: str | None = None

    def __post_init__(self) -> None:
        if self.coordinator is None:
            self.coordinator = _DEFAULT_CACHE_COORDINATOR

    def bucket(self, name: str, *, max_items: int | None = None, max_bytes: int | None = None) -> BoundedCacheBucket:
        """Return the named mutable cache bucket, creating it if needed."""

        key = str(name)
        if key not in self._buckets:
            self._buckets[key] = BoundedCacheBucket(
                max_items=self.default_bucket_size if max_items is None else max_items,
                max_bytes=self.default_bucket_byte_limit if max_bytes is None else max_bytes,
                coordinator=self.coordinator,
            )
            if self.coordinator is not None:
                self.coordinator.register(self._buckets[key])
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
    aggregate_byte_limit: int | None = None
    coordinator: CacheCoordinator = field(default_factory=lambda: _DEFAULT_CACHE_COORDINATOR, repr=False)

    def __post_init__(self) -> None:
        for layer in (self.raw_data_cache, self.derived_cache, self.view_cache):
            previous_coordinator = layer.coordinator
            if previous_coordinator is not None and previous_coordinator is not self.coordinator:
                for bucket in layer._buckets.values():
                    previous_coordinator.unregister(bucket)
        self.raw_data_cache.coordinator = self.coordinator
        self.derived_cache.coordinator = self.coordinator
        self.view_cache.coordinator = self.coordinator
        for layer in (self.raw_data_cache, self.derived_cache, self.view_cache):
            for bucket in layer._buckets.values():
                bucket.coordinator = self.coordinator
                self.coordinator.register(bucket)
        self.raw_data_cache.default_bucket_byte_limit = self.raw_bucket_byte_limit
        self.derived_cache.default_bucket_byte_limit = self.derived_bucket_byte_limit
        self.view_cache.default_bucket_byte_limit = self.view_bucket_byte_limit
        if self.aggregate_byte_limit is not None:
            self.coordinator.set_limit(self.aggregate_byte_limit)

    def stats(self) -> dict[str, CacheLayerStats]:
        return {
            "raw_data_cache": self.raw_data_cache.stats(),
            "derived_cache": self.derived_cache.stats(),
            "view_cache": self.view_cache.stats(),
        }

    def memory_stats(self) -> CacheCoordinatorStats:
        return self.coordinator.stats()


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


def retire_run_revision(revision: Any) -> int:
    """Release cache-owned references for a retired ``RunRevision``."""

    coordinator = get_session_raw_data_cache().coordinator
    if coordinator is None:
        return 0
    return coordinator.retire_revision(revision)


def run_identity_for_path(path: str | Path) -> tuple[object, ...]:
    """Return a stable run identity for session-level raw-data reuse."""

    candidate = Path(path)
    resolved = candidate.resolve(strict=False)
    try:
        stat = candidate.stat()
    except OSError:
        return (str(resolved), None, None)
    return (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
