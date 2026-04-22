from pathlib import Path

from .hdf5 import WriteProgress, write_hdf5
from .model import HeliosBlockIndex, HeliosHeader, HeliosPreview, RunStatusInfo, Simulation, Snapshot, SnapshotBlock, TextSpan
from .parser import HeliosParser
from .reader import HeliosRun, VisarBoundaryCandidate, VisarReadinessStatus, VisarSupportMetadata

_DEFAULT_PARSER = HeliosParser()


def inspect(path: str | Path, *, access_mode: str | None = None) -> HeliosHeader:
    return _DEFAULT_PARSER.inspect(path, access_mode=access_mode)


def preview(path: str | Path, *, access_mode: str | None = None) -> HeliosPreview:
    return _DEFAULT_PARSER.preview(path, access_mode=access_mode)


def parse(path: str | Path, *, access_mode: str | None = None) -> Simulation:
    return _DEFAULT_PARSER.parse(path, access_mode=access_mode)


__all__ = [
    "HeliosBlockIndex",
    "HeliosHeader",
    "HeliosParser",
    "HeliosPreview",
    "HeliosRun",
    "inspect",
    "parse",
    "preview",
    "RunStatusInfo",
    "Simulation",
    "Snapshot",
    "SnapshotBlock",
    "TextSpan",
    "VisarBoundaryCandidate",
    "VisarReadinessStatus",
    "VisarSupportMetadata",
    "WriteProgress",
    "write_hdf5",
]
