"""Lightweight timing/counter helpers for local performance visibility."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import threading
import time
from typing import Iterator


@dataclass(frozen=True, slots=True)
class TimerStat:
    count: int
    total_s: float
    last_s: float
    min_s: float
    max_s: float


class _TimerAccumulator:
    __slots__ = ("count", "total_s", "last_s", "min_s", "max_s")

    def __init__(self) -> None:
        self.count = 0
        self.total_s = 0.0
        self.last_s = 0.0
        self.min_s = float("inf")
        self.max_s = 0.0

    def update(self, elapsed_s: float) -> None:
        value = float(elapsed_s)
        self.count += 1
        self.total_s += value
        self.last_s = value
        self.min_s = min(self.min_s, value)
        self.max_s = max(self.max_s, value)

    def snapshot(self) -> TimerStat:
        minimum = 0.0 if self.count == 0 else float(self.min_s)
        return TimerStat(
            count=int(self.count),
            total_s=float(self.total_s),
            last_s=float(self.last_s),
            min_s=minimum,
            max_s=float(self.max_s),
        )


_LOCK = threading.Lock()
_TIMERS: dict[str, _TimerAccumulator] = {}
_COUNTERS: Counter[str] = Counter()


def record_duration(name: str, elapsed_s: float) -> None:
    key = str(name)
    with _LOCK:
        timer = _TIMERS.get(key)
        if timer is None:
            timer = _TimerAccumulator()
            _TIMERS[key] = timer
        timer.update(float(elapsed_s))


def increment_counter(name: str, amount: int = 1) -> None:
    with _LOCK:
        _COUNTERS[str(name)] += int(amount)


@contextmanager
def timed_block(
    name: str,
    *,
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
    details: str | None = None,
) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_s = time.perf_counter() - started
        record_duration(name, elapsed_s)
        if logger is not None and logger.isEnabledFor(level):
            suffix = "" if details in {None, ""} else f" ({details})"
            logger.log(level, "%s took %.3f ms%s", str(name), elapsed_s * 1.0e3, suffix)


def snapshot_metrics() -> dict[str, object]:
    with _LOCK:
        return {
            "timers": {name: timer.snapshot() for name, timer in _TIMERS.items()},
            "counters": dict(_COUNTERS),
        }


def format_metrics_summary() -> str:
    snapshot = snapshot_metrics()
    timer_lines = [
        f"{name}: count={stat.count} total={stat.total_s:.3f}s last={stat.last_s:.3f}s min={stat.min_s:.3f}s max={stat.max_s:.3f}s"
        for name, stat in sorted(snapshot["timers"].items())
    ]
    counter_lines = [f"{name}: {value}" for name, value in sorted(snapshot["counters"].items())]
    sections: list[str] = []
    if timer_lines:
        sections.append("Timers")
        sections.extend(timer_lines)
    if counter_lines:
        sections.append("Counters")
        sections.extend(counter_lines)
    return "\n".join(sections) if sections else "No instrumentation recorded."


def reset_metrics() -> None:
    with _LOCK:
        _TIMERS.clear()
        _COUNTERS.clear()
