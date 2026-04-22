"""Background analysis task primitives for shipping non-blocking workflows.

This uses a small persistent execution model based on ``QThreadPool`` rather
than creating a fresh ``QThread`` per task. That keeps repeated derived
recomputes cheaper while preserving the existing latest-wins and cooperative
cancellation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Callable

from PySide6 import QtCore

from .cancellation import AnalysisCancelled, CancellationToken


@dataclass(slots=True)
class AnalysisTaskResult:
    """Result produced by a background analysis task."""

    task_id: str
    context_key: tuple[object, ...]
    result: Any


@dataclass(slots=True)
class AnalysisTaskHandle:
    """Caller-facing handle for a queued analysis task."""

    task_id: str
    context_key: tuple[object, ...]
    cancellation_token: CancellationToken


@dataclass(frozen=True, slots=True)
class TaskManagerStats:
    submitted: int
    completed: int
    cancelled: int
    failed: int
    active: int
    max_thread_count: int


class _TaskSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str, str)
    cancelled = QtCore.Signal(str)


class _TaskRunnable(QtCore.QRunnable):
    def __init__(
        self,
        task_id: str,
        context_key: tuple[object, ...],
        fn: Callable[[], Any],
        cancellation_token: CancellationToken,
        signals: _TaskSignals,
    ) -> None:
        super().__init__()
        self._task_id = task_id
        self._context_key = context_key
        self._fn = fn
        self._cancellation_token = cancellation_token
        self._signals = signals
        self.setAutoDelete(True)

    @QtCore.Slot()
    def run(self) -> None:
        if self._cancellation_token.is_cancelled():
            self._signals.cancelled.emit(self._task_id)
            return
        try:
            result = self._fn()
            if self._cancellation_token.is_cancelled():
                self._signals.cancelled.emit(self._task_id)
                return
        except AnalysisCancelled:
            self._signals.cancelled.emit(self._task_id)
            return
        except Exception as exc:  # pragma: no cover - defensive worker path
            self._signals.failed.emit(self._task_id, str(exc))
            return
        self._signals.finished.emit(AnalysisTaskResult(self._task_id, self._context_key, result))


class AnalysisTaskManager(QtCore.QObject):
    """Queue lightweight background tasks and drop stale results safely."""

    result_ready = QtCore.Signal(object)
    task_failed = QtCore.Signal(str, str)
    task_cancelled = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._active_tokens: dict[str, CancellationToken] = {}
        self._signals: dict[str, _TaskSignals] = {}
        self._submitted = 0
        self._completed = 0
        self._cancelled = 0
        self._failed = 0

    def submit(
        self,
        *,
        context_key: tuple[object, ...],
        fn: Callable[[], Any],
        cancellation_token: CancellationToken | None = None,
    ) -> AnalysisTaskHandle:
        task_id = uuid.uuid4().hex
        active_token = cancellation_token or CancellationToken()
        signals = _TaskSignals()
        signals.finished.connect(self._handle_finished)
        signals.failed.connect(self._handle_failed)
        signals.cancelled.connect(self._handle_cancelled)
        runnable = _TaskRunnable(task_id, context_key, fn, active_token, signals)
        self._active_tokens[task_id] = active_token
        self._signals[task_id] = signals
        self._submitted += 1
        self._pool.start(runnable)
        return AnalysisTaskHandle(task_id=task_id, context_key=context_key, cancellation_token=active_token)

    def stats(self) -> TaskManagerStats:
        return TaskManagerStats(
            submitted=int(self._submitted),
            completed=int(self._completed),
            cancelled=int(self._cancelled),
            failed=int(self._failed),
            active=int(len(self._active_tokens)),
            max_thread_count=int(self._pool.maxThreadCount()),
        )

    @QtCore.Slot(object)
    def _handle_finished(self, result: AnalysisTaskResult) -> None:
        self._active_tokens.pop(result.task_id, None)
        self._signals.pop(result.task_id, None)
        self._completed += 1
        self.result_ready.emit(result)

    @QtCore.Slot(str, str)
    def _handle_failed(self, task_id: str, message: str) -> None:
        self._active_tokens.pop(task_id, None)
        self._signals.pop(task_id, None)
        self._failed += 1
        self.task_failed.emit(task_id, message)

    @QtCore.Slot(str)
    def _handle_cancelled(self, task_id: str) -> None:
        self._active_tokens.pop(task_id, None)
        self._signals.pop(task_id, None)
        self._cancelled += 1
        self.task_cancelled.emit(task_id)

    def cancel(self, task_id: str | None) -> None:
        if not task_id:
            return
        token = self._active_tokens.get(task_id)
        if token is not None:
            token.cancel()

    def shutdown(self, timeout_ms: int = 2000) -> None:
        """Request cooperative stop and wait briefly for queued work to finish."""

        for token in list(self._active_tokens.values()):
            token.cancel()
        self._pool.waitForDone(int(timeout_ms))
        self._active_tokens.clear()
        self._signals.clear()
