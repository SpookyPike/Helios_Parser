"""Backend-safe cooperative cancellation primitives.

These helpers are intentionally Qt-free so long-running backend analysis code
can be imported and exercised in headless environments without pulling GUI
dependencies into the import graph.
"""

from __future__ import annotations

import threading


class AnalysisCancelled(RuntimeError):
    """Raised when a superseded background analysis exits cooperatively."""


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check_cancelled(self) -> None:
        if self._event.is_set():
            raise AnalysisCancelled("Analysis request was superseded by a newer request.")
