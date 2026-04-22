"""Shared runtime scaffolding for HELIOS Analyzer.

This package is intentionally lightweight. It does not contain parser or viewer
science logic; it provides the platform-level primitives used by the shipping
shell, viewer, and Derived / Analysis mode to share state cleanly. Task imports
stay lazy here so backend-only users do not pull Qt into the import graph
unless they explicitly need the GUI task runner.
"""

from .cache import AnalyzerCacheSet, CacheLayer
from .runtime import RunContext

__all__ = [
    "AnalyzerCacheSet",
    "CacheLayer",
    "RunContext",
]


def __getattr__(name: str):
    """Lazy task imports so backend-only users do not need Qt installed."""

    if name in {"AnalysisTaskHandle", "AnalysisTaskManager", "AnalysisTaskResult"}:
        from .tasks import AnalysisTaskHandle, AnalysisTaskManager, AnalysisTaskResult

        mapping = {
            "AnalysisTaskHandle": AnalysisTaskHandle,
            "AnalysisTaskManager": AnalysisTaskManager,
            "AnalysisTaskResult": AnalysisTaskResult,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
