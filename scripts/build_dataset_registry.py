from __future__ import annotations

from pathlib import Path

from _validation_common import REPORT_ROOT, REPO_ROOT, build_registry, preferred_hdf5_records, write_markdown_table


def main() -> int:
    registry = build_registry()
    preferred = preferred_hdf5_records(registry)
    rows = []
    for record in registry.records:
        rows.append(
            (
                record.filename,
                record.artifact_type,
                "yes" if record.directly_usable else "no",
                record.archive_type or "",
                record.zones if record.zones is not None else "",
                record.snapshots if record.snapshots is not None else "",
                record.geometry or "",
                ", ".join(record.materials),
                ", ".join(record.notes),
            )
        )
    write_markdown_table(
        REPORT_ROOT / "dataset_registry.md",
        ("Filename", "Type", "Usable", "Archive", "Zones", "Snapshots", "Geometry", "Materials", "Notes"),
        rows,
    )
    preferred_rows = [(record.filename, record.path, record.zones or "", record.snapshots or "", record.geometry or "") for record in preferred]
    write_markdown_table(
        REPORT_ROOT / "preferred_hdf5_datasets.md",
        ("Dataset", "Path", "Zones", "Snapshots", "Geometry"),
        preferred_rows,
    )
    print(f"Wrote registry for {len(registry.records)} artifacts under {REPORT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
