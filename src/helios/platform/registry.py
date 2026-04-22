"""Dataset and artifact discovery for HELIOS Analyzer.

The registry is used for validation, documentation, and future backend
integration planning. It is intentionally descriptive rather than intrusive: it
inspects what is already present in the repository without changing parser or
viewer behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from helios.platform.archive_utils import archive_type_for_path, inspect_archive
from helios_parser import HeliosRun, inspect


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Single discovered repository artifact."""

    path: str
    filename: str
    artifact_type: str
    directly_usable: bool
    archive_type: str | None = None
    source_kind: str | None = None
    zones: int | None = None
    snapshots: int | None = None
    geometry: str | None = None
    materials: tuple[str, ...] = ()
    top_level_entries: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["materials"] = list(self.materials)
        payload["top_level_entries"] = list(self.top_level_entries)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class DatasetRegistry:
    """Structured repository artifact registry."""

    root: str
    records: tuple[ArtifactRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "records": [record.to_dict() for record in self.records],
        }


def _classify_hdf5(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if "stabilized" in name:
        return "stabilized_hdf5", "stabilized"
    if "phase" in name or "regression" in name:
        return "derived_hdf5_artifact", "generated"
    return "hdf5", "direct"


def _material_labels_from_regions(regions: dict[str, np.ndarray]) -> tuple[str, ...]:
    material_index = np.asarray(regions.get("material_index", np.empty(0)), dtype=np.int32)
    material_table = np.asarray(regions.get("material_table_index", np.empty(0)), dtype=np.int32)
    labels: list[str] = []
    for region_id, table_id in zip(material_index.tolist(), material_table.tolist()):
        label = f"region_material={region_id}, table_material={table_id}"
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _normalize_hdf5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _normalize_hdf5_value(value[()])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fallback_hdf5_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        grid_group = handle.get("grid")
        time_group = handle.get("time")
        fields_group = handle.get("fields")
        metadata_group = handle.get("metadata")
        n_zones = 0
        if grid_group is not None:
            for candidate_name in ("x", "zone_id", "zone_width"):
                if candidate_name in grid_group:
                    n_zones = int(grid_group[candidate_name].shape[0])
                    break
        n_snapshots = 0
        if time_group is not None:
            for candidate_name in ("time", "cycle", "time_step"):
                if candidate_name in time_group:
                    n_snapshots = int(time_group[candidate_name].shape[0])
                    break
        metadata: dict[str, Any] = {}
        geometry = ""
        if metadata_group is not None and "geometry" in metadata_group:
            geometry = str(_normalize_hdf5_value(metadata_group["geometry"][()]) or "")
            metadata["geometry"] = geometry
        summary = {
            "n_zones": n_zones,
            "n_snapshots": n_snapshots,
            "geometry": geometry,
            "available_fields": list(fields_group.keys()) if fields_group is not None else [],
        }
        return summary, metadata


def _inspect_log(path: Path) -> ArtifactRecord:
    header = inspect(path)
    materials: tuple[str, ...] = ()
    notes: list[str] = []
    regions = tuple(getattr(header, "spatial_regions", ()) or ())
    if regions:
        labels = []
        for region in regions:
            material_index = region.get("material_index")
            material_table_index = region.get("material_table_index")
            labels.append(f"region_material={material_index}, table_material={material_table_index}")
        materials = tuple(dict.fromkeys(labels))
    else:
        notes.append("header_without_spatial_regions")
    if getattr(header, "laser_source", None):
        notes.append("laser_source_present")
    if getattr(header, "photon_energy_grid", None):
        notes.append("photon_grid_present")
    return ArtifactRecord(
        path=str(path),
        filename=path.name,
        artifact_type="helios_log",
        directly_usable=True,
        source_kind="direct",
        zones=int(header.n_zones),
        snapshots=int(getattr(header, "n_snapshots", 0) or 0),
        geometry=str(getattr(header, "geometry", "") or ""),
        materials=materials,
        notes=tuple(notes),
    )


def _inspect_hdf5(path: Path) -> ArtifactRecord:
    artifact_type, source_kind = _classify_hdf5(path)
    notes: list[str] = []
    try:
        with HeliosRun(path) as run:
            try:
                summary = run.summary()
            except Exception:
                summary, metadata = _fallback_hdf5_summary(path)
                notes.append("legacy_hdf5_layout")
            else:
                try:
                    metadata = run.get_metadata()
                except Exception:
                    metadata = {}
            try:
                regions = run.get_regions()
            except Exception:
                regions = {}
    except Exception:
        summary, metadata = _fallback_hdf5_summary(path)
        regions = {}
        notes.append("legacy_hdf5_layout")

    if "radiation_sink" in summary.get("available_fields", ()):
        notes.append("newer_helios_field_radiation_sink")
    if metadata.get("laser_entry"):
        notes.append("laser_entry_metadata")
    if not regions:
        notes.append("missing_regions_group")
    return ArtifactRecord(
        path=str(path),
        filename=path.name,
        artifact_type=artifact_type,
        directly_usable=True,
        source_kind=source_kind,
        zones=int(summary.get("n_zones", 0)),
        snapshots=int(summary.get("n_snapshots", 0)),
        geometry=str(summary.get("geometry", metadata.get("geometry", "")) or ""),
        materials=_material_labels_from_regions(regions),
        notes=tuple(notes),
    )


def _inspect_archive_file(path: Path) -> ArtifactRecord:
    inspection = inspect_archive(path, max_members=12)
    notes: list[str] = []
    top_levels = inspection.top_level_entries
    joined = " ".join(top_levels).lower()
    if "xcom" in joined:
        notes.append("xcom_bundle")
    return ArtifactRecord(
        path=str(path),
        filename=path.name,
        artifact_type="archive",
        directly_usable=False,
        archive_type=inspection.archive_type,
        source_kind="archive",
        top_level_entries=inspection.top_level_entries,
        notes=tuple(notes),
    )


def _inspect_misc(path: Path) -> ArtifactRecord | None:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        artifact_type = "image_asset"
        notes: list[str] = []
        if "icon" in path.stem.lower():
            notes.append("icon_asset")
        if "three" in path.stem.lower() and "icon" in path.stem.lower():
            notes.append("icon_source_sheet")
        return ArtifactRecord(
            path=str(path),
            filename=path.name,
            artifact_type=artifact_type,
            directly_usable=False,
            source_kind="asset",
            notes=tuple(notes),
        )
    if suffix == ".pdf":
        return ArtifactRecord(
            path=str(path),
            filename=path.name,
            artifact_type="pdf_reference",
            directly_usable=False,
            source_kind="reference",
        )
    return None


def build_dataset_registry(root: str | Path, *, include_outputs: bool = True) -> DatasetRegistry:
    """Build a structured registry of logs, HDF5 files, archives, and assets."""

    root_path = Path(root)
    search_roots = [root_path]
    outputs_dir = root_path / "outputs"
    if include_outputs and outputs_dir.exists():
        search_roots.append(outputs_dir)

    seen: set[Path] = set()
    records: list[ArtifactRecord] = []

    for search_root in search_roots:
        for path in sorted(search_root.rglob("*")):
            if not path.is_file():
                continue
            if path in seen:
                continue
            seen.add(path)
            lower_name = path.name.lower()
            try:
                if lower_name.endswith(".log"):
                    records.append(_inspect_log(path))
                    continue
                if lower_name.endswith(".h5") or lower_name.endswith(".hdf5"):
                    records.append(_inspect_hdf5(path))
                    continue
                archive_type = archive_type_for_path(path)
                if archive_type is not None:
                    records.append(_inspect_archive_file(path))
                    continue
                misc = _inspect_misc(path)
                if misc is not None:
                    records.append(misc)
            except Exception as exc:
                records.append(
                    ArtifactRecord(
                        path=str(path),
                        filename=path.name,
                        artifact_type="inspection_error",
                        directly_usable=False,
                        source_kind="error",
                        notes=(str(exc),),
                    )
                )

    return DatasetRegistry(root=str(root_path), records=tuple(records))
