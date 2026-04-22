"""Platform-oriented helpers shared across HELIOS Analyzer layers.

This package intentionally stays lightweight. It contains infrastructure that
is not specific to parser science, viewer rendering, or derived formulas:

- dataset / artifact discovery
- Windows-safe archive inspection and extraction
- future-friendly platform helpers that multiple top-level modes can share
"""

from .archive_utils import ArchiveInspection, ArchiveMember, extract_archive, inspect_archive
from .registry import ArtifactRecord, DatasetRegistry, build_dataset_registry

__all__ = [
    "ArchiveInspection",
    "ArchiveMember",
    "ArtifactRecord",
    "DatasetRegistry",
    "build_dataset_registry",
    "extract_archive",
    "inspect_archive",
]
