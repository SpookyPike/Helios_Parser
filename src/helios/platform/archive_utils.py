"""Windows-safe archive discovery helpers.

These helpers use only the Python standard library and avoid shell-specific
extraction assumptions. They are suitable for direct folder checkouts as well as
zip/tar.gz-distributed auxiliary bundles such as optional XCOM adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


ARCHIVE_ZIP = "zip"
ARCHIVE_TAR_GZ = "tar.gz"


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """Single archive entry summary."""

    path: str
    size: int
    is_dir: bool


@dataclass(frozen=True, slots=True)
class ArchiveInspection:
    """Structured archive inspection result."""

    path: Path
    archive_type: str
    member_count: int
    top_level_entries: tuple[str, ...]
    members: tuple[ArchiveMember, ...]


def archive_type_for_path(path: str | Path) -> str | None:
    """Return a normalized archive type for supported bundles."""

    resolved = Path(path)
    name = resolved.name.lower()
    if name.endswith(".zip"):
        return ARCHIVE_ZIP
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return ARCHIVE_TAR_GZ
    return None


def _top_level_entries(member_paths: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for member_path in member_paths:
        parts = PurePosixPath(member_path).parts
        if not parts:
            continue
        top = parts[0]
        if top not in seen:
            seen.append(top)
    return tuple(seen)


def inspect_archive(path: str | Path, *, max_members: int | None = None) -> ArchiveInspection:
    """Inspect a zip or tar.gz archive without extracting it."""

    resolved = Path(path)
    archive_type = archive_type_for_path(resolved)
    if archive_type is None:
        raise ValueError(f"Unsupported archive type for {resolved}")

    members: list[ArchiveMember] = []
    all_paths: list[str] = []

    if archive_type == ARCHIVE_ZIP:
        with zipfile.ZipFile(resolved) as bundle:
            infos = bundle.infolist()
            for index, info in enumerate(infos):
                normalized = str(PurePosixPath(info.filename))
                all_paths.append(normalized)
                if max_members is None or index < max_members:
                    members.append(
                        ArchiveMember(
                            path=normalized,
                            size=int(info.file_size),
                            is_dir=info.is_dir(),
                        )
                    )
            member_count = len(infos)
    else:
        with tarfile.open(resolved, "r:*") as bundle:
            infos = bundle.getmembers()
            for index, info in enumerate(infos):
                normalized = str(PurePosixPath(info.name))
                all_paths.append(normalized)
                if max_members is None or index < max_members:
                    members.append(
                        ArchiveMember(
                            path=normalized,
                            size=int(info.size),
                            is_dir=info.isdir(),
                        )
                    )
            member_count = len(infos)

    return ArchiveInspection(
        path=resolved,
        archive_type=archive_type,
        member_count=member_count,
        top_level_entries=_top_level_entries(all_paths),
        members=tuple(members),
    )


def _safe_target_path(destination: Path, member_name: str) -> Path:
    target = (destination / PurePosixPath(member_name)).resolve()
    destination_resolved = destination.resolve()
    if destination_resolved not in target.parents and target != destination_resolved:
        raise ValueError(f"Archive member escapes destination: {member_name}")
    return target


def extract_archive(path: str | Path, destination: str | Path) -> Path:
    """Extract a supported archive into ``destination`` safely."""

    resolved = Path(path)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    archive_type = archive_type_for_path(resolved)
    if archive_type is None:
        raise ValueError(f"Unsupported archive type for {resolved}")

    if archive_type == ARCHIVE_ZIP:
        with zipfile.ZipFile(resolved) as bundle:
            for info in bundle.infolist():
                target = _safe_target_path(destination_path, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info, "r") as source, target.open("wb") as output:
                    output.write(source.read())
        return destination_path

    with tarfile.open(resolved, "r:*") as bundle:
        for member in bundle.getmembers():
            target = _safe_target_path(destination_path, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with extracted, target.open("wb") as output:
                output.write(extracted.read())
    return destination_path
