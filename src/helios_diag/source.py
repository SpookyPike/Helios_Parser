"""Open a HELIOS run from either a converted HDF5 file or a raw deck.

`helios_diag` used to accept only a converted run. This module gives the commands
one entry point: `open_any(path)` returns a `HeliosRun` for a converted file and a
`RawRun` for a deck that was never converted, both behind the same accessor
surface.

Format detection is by suffix first and by magic bytes second, because a `.exo`
is an HDF5 container while a classic NetCDF-3 Exodus file is not and cannot be
read here.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
CLASSIC_NETCDF_MAGIC = b"CDF"

CONVERTED_SUFFIXES = {".h5", ".hdf5"}
LOG_SUFFIXES = {".log"}
BPF_SUFFIXES = {".bpf"}
EXODUS_SUFFIXES = {".exo"}


class UnsupportedInputError(ValueError):
    """The path is not a format this tool can read."""


def detect_input(path: str | Path) -> str:
    """Return ``hdf5``, ``log``, ``bpf``, ``exodus`` or ``netcdf3``.

    ``netcdf3`` is reported rather than raised so the caller can name the variant
    in its error message instead of leaking an HDF5 open failure.
    """
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    suffix = candidate.suffix.lower()
    if suffix in CONVERTED_SUFFIXES:
        return "hdf5"
    if suffix in LOG_SUFFIXES:
        return "log"
    if suffix in BPF_SUFFIXES:
        return "bpf"
    if suffix in EXODUS_SUFFIXES:
        with candidate.open("rb") as handle:
            head = handle.read(8)
        if head.startswith(CLASSIC_NETCDF_MAGIC):
            return "netcdf3"
        if head.startswith(HDF5_MAGIC):
            return "exodus"
        return "exodus"
    with candidate.open("rb") as handle:
        head = handle.read(8)
    if head.startswith(HDF5_MAGIC):
        return "hdf5"
    raise UnsupportedInputError(
        f"{candidate}: unrecognised input. Expected a converted .h5, a HELIOS .log, "
        "a .bpf, or an HDF5 Exodus II .exo."
    )


def is_raw(path: str | Path) -> bool:
    return detect_input(path) in {"log", "bpf", "exodus"}


# The six fields every derived module needs; `load_run_data` reads them whether
# or not the caller asked, so a raw open must stack them.
CORE_FIELDS = (
    "density",
    "velocity",
    "temperature_e",
    "temperature_i",
    "electron_density",
    "mean_charge",
    # `load_run_data` reads the width whenever the run lists it, so a filtered
    # raw open has to stack it or the derived assembly raises.
    "zone_width",
)


def open_any(path: str | Path, *, fields: Iterable[str] | None = None) -> Any:
    """Return a run view for a converted file or a raw deck.

    ``fields`` bounds the stacked field set for a raw deck, which is what keeps a
    single-field `stats` from stacking every column. A converted file is lazy and
    ignores it.
    """
    kind = detect_input(path)
    if kind == "hdf5":
        from helios_parser import HeliosRun

        return HeliosRun(path)
    if kind == "netcdf3":
        _raise_netcdf3(path)
    from helios_parser.bpf import LOG_COMPATIBLE_BPF_ALIASES
    from helios_parser.raw_run import RawRun

    if kind == "log":
        return RawRun(_simulation_from_log(path, fields=fields), "log")
    if kind == "bpf":
        return RawRun(
            _simulation_from_bpf(path, fields=fields),
            "bpf",
            alias_map={alias: canonical for canonical, alias in LOG_COMPATIBLE_BPF_ALIASES.items()},
        )
    return RawRun(_simulation_from_exodus(path, fields=fields), "exodus")


# --------------------------------------------------------------------- helpers


def _simulation_from_log(path: str | Path, *, fields: Iterable[str] | None = None):
    from helios_parser.parser import HeliosParser

    simulation = HeliosParser().parse(path, fields=fields)
    simulation.metadata.setdefault("format", "log")
    return simulation


def _simulation_from_bpf(path: str | Path, *, fields: Iterable[str] | None = None):
    """Mirror the writer's BPF path: fields, dynamic node coordinates, minimal grid.

    The grid derivation copies `_write_minimal_bpf_header_groups` on purpose. A
    second, different derivation would make the raw and converted runs disagree,
    which is exactly what `test_cli_raw_equivalence.py` exists to catch.
    """
    from helios_parser.bpf import LOG_COMPATIBLE_BPF_ALIASES, BpfFile, field_metadata_for_bpf_key
    from helios_parser.model import Simulation

    requested = None if fields is None else {str(name) for name in fields}
    # A requested LOG-compatible name (`density`) is not a BPF key
    # (`mass_density_g_cm3`), so stack the canonical too and alias it below.
    selected = requested
    if requested is not None:
        selected = set(requested)
        for canonical, alias in LOG_COMPATIBLE_BPF_ALIASES.items():
            if alias in requested:
                selected.add(canonical)
    with BpfFile(path) as bpf:
        layout = bpf.layout
        snapshots = list(bpf.iter_snapshots())
    if not snapshots:
        raise UnsupportedInputError(f"{Path(path).name}: the BPF carries no snapshots.")
    first = snapshots[0]
    field_names = [name for name in first.fields if selected is None or name in selected]
    n_snapshots = len(snapshots)
    stacked = {
        name: np.full((n_snapshots, layout.n_zones), np.nan, dtype=np.float64)
        for name in field_names
        if name != "node_position_cm"
    }
    dynamic_edge = np.full((n_snapshots, layout.n_nodes), np.nan, dtype=np.float64)
    dynamic_center = np.full((n_snapshots, layout.n_zones), np.nan, dtype=np.float64)
    time_s = np.empty(n_snapshots, dtype=np.float64)
    cycle = np.empty(n_snapshots, dtype=np.int64)
    for index, snapshot in enumerate(snapshots):
        time_s[index] = float(snapshot.time_s)
        cycle[index] = int(snapshot.cycle)
        node_position = np.asarray(snapshot.fields["node_position_cm"], dtype=np.float64)
        dynamic_edge[index, :] = node_position
        dynamic_center[index, :] = 0.5 * (node_position[:-1] + node_position[1:])
        for name in stacked:
            stacked[name][index, :] = np.asarray(snapshot.fields[name], dtype=np.float64)

    node_position = np.asarray(first.fields["node_position_cm"], dtype=np.float64)
    zone_width = np.diff(node_position)
    zone_center = 0.5 * (node_position[:-1] + node_position[1:])
    region_index_by_zone = np.asarray(
        first.fields.get("region_index_by_zone", np.ones(layout.n_zones)), dtype=np.int32
    )
    grid = {
        "coordinate_center": zone_center,
        "coordinate_edge": node_position,
        "x": zone_center,
        "zone_id": np.arange(1, layout.n_zones + 1, dtype=np.int32),
        "zone_width": zone_width,
        "zone_mass": np.asarray(first.fields["zone_mass"], dtype=np.float64),
        "zone_region_id": region_index_by_zone,
        "zone_material_index": region_index_by_zone,
        "dynamic_coordinate_center": dynamic_center,
        "dynamic_coordinate_edge": dynamic_edge,
    }
    grid_units = {
        "coordinate_center": "cm",
        "coordinate_edge": "cm",
        "x": "cm",
        "zone_id": "",
        "zone_width": "cm",
        "zone_mass": "g/cm**X",
        "zone_region_id": "",
        "zone_material_index": "",
        "dynamic_coordinate_center": "cm",
        "dynamic_coordinate_edge": "cm",
    }
    region_index = np.unique(region_index_by_zone).astype(np.int32)
    if region_index.size == 0:
        region_index = np.asarray([1], dtype=np.int32)
    regions = {
        "region_index": region_index,
        "min_zone_index": np.asarray([1], dtype=np.int32),
        "max_zone_index": np.asarray([layout.n_zones], dtype=np.int32),
        "material_index": region_index,
    }
    materials = {
        "index": np.unique(np.abs(region_index_by_zone)).astype(np.int32),
        "eos_model": np.asarray([""], dtype=object),
        "opacity_model": np.asarray([""], dtype=object),
    }
    field_units = {
        name: field_metadata_for_bpf_key(name).get("unit", "")
        for name in stacked
    }
    raw_field_map = {
        name: field_metadata_for_bpf_key(name).get("label", name)
        for name in stacked
    }
    # The writer hard-links a LOG-compatible alias for every BPF field it can, so
    # a converted BPF answers `density` as well as `mass_density_g_cm3`. Mirror
    # that here with the same array, not a copy, or the two runs disagree about
    # which fields exist.
    for canonical, alias in LOG_COMPATIBLE_BPF_ALIASES.items():
        if canonical in stacked and alias not in stacked:
            stacked[alias] = stacked[canonical]
            field_units[alias] = field_units[canonical]
            raw_field_map[alias] = raw_field_map[canonical]
    if requested is not None:
        # Drop the canonicals that were stacked only to satisfy an alias.
        keep = {name for name in requested if name in stacked}
        stacked = {name: stacked[name] for name in stacked if name in keep}
    all_field_names = [name for name in first.fields if name != "node_position_cm"]
    for canonical, alias in LOG_COMPATIBLE_BPF_ALIASES.items():
        if canonical in all_field_names and alias not in all_field_names:
            all_field_names.append(alias)
    metadata: dict[str, Any] = {
        "format": "bpf",
        "geometry": "planar",
        "n_zones": int(layout.n_zones),
        "n_snapshots": n_snapshots,
        "available_fields": all_field_names,
        "bpf_run_date": layout.run_date,
        "bpf_run_clock": layout.run_clock,
        "bpf_record_count": int(layout.record_count),
        "bpf_trailing_records": int(layout.trailing_records),
        "bpf_incomplete_final_snapshot": bool(layout.incomplete_final_snapshot),
    }
    return Simulation(
        source=Path(path),
        grid=grid,
        grid_units=grid_units,
        regions=regions,
        region_units={},
        materials=materials,
        material_units={},
        time={"time": time_s, "cycle": cycle, "time_step": np.full(n_snapshots, np.nan), "time_step_control": np.full(n_snapshots, "", dtype=object)},
        time_units={"time": "s", "cycle": "", "time_step": "s", "time_step_control": ""},
        fields=stacked,
        field_units=field_units,
        diagnostics={},
        input_parameters={},
        metadata=metadata,
        raw_field_map=raw_field_map,
    )


def _simulation_from_exodus(path: str | Path, *, fields: Iterable[str] | None = None):
    from helios_parser.exodus import read_exodus
    from helios_parser.model import Simulation

    run = read_exodus(path)
    selected = None if fields is None else {str(name) for name in fields}
    stacked = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in run.fields.items()
        if selected is None or name in selected
    }
    n_snapshots = int(run.n_snapshots)
    time_step = (
        np.asarray(run.time_step_s, dtype=np.float64)
        if run.time_step_s is not None
        else np.full(n_snapshots, np.nan)
    )
    grid = {
        "coordinate_center": np.asarray(run.coordinate_center_cm, dtype=np.float64),
        "coordinate_edge": np.asarray(run.coordinate_edge_cm, dtype=np.float64),
        "x": np.asarray(run.coordinate_center_cm, dtype=np.float64),
        "zone_id": np.arange(1, int(run.n_zones) + 1, dtype=np.int32),
        "zone_width": np.asarray(run.zone_width_cm, dtype=np.float64),
        "zone_region_id": np.ones(int(run.n_zones), dtype=np.int32),
        "zone_material_index": np.ones(int(run.n_zones), dtype=np.int32),
    }
    if run.zone_mass_g is not None:
        grid["zone_mass"] = np.asarray(run.zone_mass_g, dtype=np.float64)
    metadata: dict[str, Any] = {
        "format": "exodus",
        "geometry": run.geometry_kind or run.geometry,
        "n_zones": int(run.n_zones),
        "n_snapshots": n_snapshots,
        "available_fields": list(stacked),
        "temperature_model": run.temperature_model,
    }
    return Simulation(
        source=Path(path),
        grid=grid,
        grid_units={
            "coordinate_center": "cm",
            "coordinate_edge": "cm",
            "x": "cm",
            "zone_id": "",
            "zone_width": "cm",
            "zone_mass": "g/cm**X",
            "zone_region_id": "",
            "zone_material_index": "",
        },
        regions={
            "region_index": np.asarray([1], dtype=np.int32),
            "min_zone_index": np.asarray([1], dtype=np.int32),
            "max_zone_index": np.asarray([int(run.n_zones)], dtype=np.int32),
            "material_index": np.asarray([1], dtype=np.int32),
        },
        region_units={},
        materials={
            "index": np.asarray([1], dtype=np.int32),
            "eos_model": np.asarray([""], dtype=object),
            "opacity_model": np.asarray([""], dtype=object),
        },
        material_units={},
        time={
            "time": np.asarray(run.time_s, dtype=np.float64),
            "cycle": np.asarray(run.cycle, dtype=np.int64),
            "time_step": time_step,
            "time_step_control": np.full(n_snapshots, "", dtype=object),
        },
        time_units={"time": "s", "cycle": "", "time_step": "s", "time_step_control": ""},
        fields=stacked,
        field_units=dict(run.field_units),
        diagnostics={},
        input_parameters={},
        metadata=metadata,
        raw_field_map=dict(run.raw_field_map),
    )


# ------------------------------------------------------- header-only lookups


class RawHeaderView:
    """The accessor subset `helios_diag inspect` needs, without a full parse.

    `inspect` answers "what is this deck" from the header plus the first snapshot.
    A full parse would cost seconds on a large deck for the same answer, so this
    view is built from the header alone and never stacks fields.
    """

    def __init__(
        self,
        path: str | Path,
        fmt: str,
        *,
        header: Any = None,
        field_names: Iterable[str] = (),
        field_units: dict[str, str] | None = None,
        raw_field_map: dict[str, str] | None = None,
        geometry: str | None = None,
        n_zones: int = 0,
        n_snapshots: int = 0,
        n_regions: int = 0,
        n_materials: int = 0,
        alias_map: dict[str, str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.format = fmt
        self.header = header
        self.field_names = list(field_names)
        self.field_units = dict(field_units or {})
        self.raw_field_map = dict(raw_field_map or {})
        self.geometry = geometry
        self.n_zones = int(n_zones)
        self.n_snapshots = int(n_snapshots)
        self.n_regions = int(n_regions)
        self.n_materials = int(n_materials)
        self.alias_map = dict(alias_map or {})
        self.extra_metadata = dict(extra_metadata or {})
        self.attributes = dict(attributes or {})

    def list_fields(self) -> list[str]:
        return list(self.field_names)

    def get_field_metadata(self, field_name: str):
        from helios_parser.reader import FieldMetadata

        unit = self.field_units.get(field_name, "")
        alias_of = self.alias_map.get(field_name)
        quantity = ""
        if self.format == "bpf":
            from helios_parser.bpf import field_metadata_for_bpf_key

            quantity = str(field_metadata_for_bpf_key(alias_of or field_name).get("quantity") or "")
        return FieldMetadata(
            name=field_name,
            shape=(self.n_snapshots, self.n_zones),
            dtype="float64",
            source=self.format,
            dimensions=("time", "zone"),
            unit=unit,
            label=self.raw_field_map.get(field_name, field_name),
            status="validated" if unit else "legacy",
            alias_of=alias_of,
            quantity=quantity,
        )

    def get_metadata(self) -> dict[str, Any]:
        metadata = dict(self.extra_metadata)
        metadata["n_zones"] = self.n_zones
        metadata["n_snapshots"] = self.n_snapshots
        metadata["available_fields"] = list(self.field_names)
        metadata["raw_field_map"] = dict(self.raw_field_map)
        metadata.setdefault("parse_mode", f"raw_{self.format}")
        if self.header is not None:
            metadata["input_parameters"] = dict(getattr(self.header, "input_parameters", {}) or {})
        return metadata

    def get_file_attributes(self) -> dict[str, Any]:
        attributes = dict(self.attributes)
        attributes.setdefault("format", self.format)
        attributes.setdefault("geometry", self.geometry)
        attributes.setdefault("source_file", str(self.path))
        attributes.setdefault("parse_mode", f"raw_{self.format}")
        return attributes

    def get_run_status(self) -> dict[str, Any]:
        return {
            "state": "unknown",
            "source": f"raw_{self.format}",
            "footer_message": None,
            "footer_datetime": None,
            "intended_end_time_s": None,
            "last_valid_snapshot_time_s": None,
            "indexed_snapshot_count": int(self.n_snapshots),
            "valid_snapshot_count": int(self.n_snapshots),
            "dropped_partial_final_block": False,
            "damaged_final_block_reason": None,
            "notes": ("raw deck: no run status is recorded; convert it to get one",),
        }

    def check_visar_readiness(self):
        raise LookupError(
            "A raw HELIOS deck carries no VISAR structure; convert it first to inspect VISAR readiness."
        )

    def close(self) -> None:
        return None


def _raise_netcdf3(path: str | Path) -> None:
    raise UnsupportedInputError(
        f"{Path(path).name} is a classic NetCDF-3 Exodus file. This reader handles the "
        "NetCDF-4 (HDF5) Exodus files HELIOS 12 writes; convert the deck with "
        "`helios_to_hdf5` from a NetCDF-4 source, or re-export it from HELIOS."
    )


def raw_header(path: str | Path) -> RawHeaderView:
    """Header-only lookup for a raw deck: 10 ms of work on a 530 MB log."""
    kind = detect_input(path)
    if kind == "log":
        return _log_header_view(path)
    if kind == "bpf":
        return _bpf_header_view(path)
    if kind == "netcdf3":
        _raise_netcdf3(path)
    if kind == "exodus":
        from helios_parser.exodus import read_exodus

        run = read_exodus(path)
        return RawHeaderView(
            path,
            "exodus",
            field_names=list(run.fields),
            field_units=dict(run.field_units),
            raw_field_map=dict(run.raw_field_map),
            geometry=run.geometry_kind or run.geometry,
            n_zones=int(run.n_zones),
            n_snapshots=int(run.n_snapshots),
            n_regions=1,
            n_materials=1,
            extra_metadata={"temperature_model": run.temperature_model},
        )
    raise UnsupportedInputError(f"{path}: not a raw HELIOS deck.")


def _log_header_view(path: str | Path) -> RawHeaderView:
    from helios_parser.buffer import HeliosBuffer
    from helios_parser.indexing import StructuralIndexer
    from helios_parser.parser import DEFAULT_TOKENIZER, HeaderSemanticParser, SnapshotBlockSemanticParser

    source = Path(path)
    buffer = HeliosBuffer.from_path(source, access_mode="mmap")
    try:
        try:
            index = StructuralIndexer(DEFAULT_TOKENIZER).build(buffer)
        except ValueError as exc:
            raise UnsupportedInputError(f"{source.name}: {exc}") from exc
        header = HeaderSemanticParser().parse(source, buffer.slice(index.header_span))
        snapshot = None
        if index.snapshot_blocks:
            snapshot = SnapshotBlockSemanticParser().parse(
                buffer.slice(index.snapshot_blocks[0].span), header
            )
        return RawHeaderView(
            source,
            "log",
            header=header,
            field_names=list(snapshot.fields) if snapshot is not None else [],
            field_units=dict(snapshot.field_units) if snapshot is not None else {},
            raw_field_map=dict(snapshot.raw_field_map) if snapshot is not None else {},
            geometry=(header.metadata or {}).get("geometry"),
            n_zones=int(header.n_zones),
            n_snapshots=int(index.snapshot_count),
            n_regions=int(header.n_regions),
            n_materials=int(header.n_materials),
            extra_metadata={"helios_version": header.code_version},
        )
    finally:
        buffer.close()


def _bpf_header_view(path: str | Path) -> RawHeaderView:
    from helios_parser.bpf import LOG_COMPATIBLE_BPF_ALIASES, BpfFile, field_metadata_for_bpf_key

    with BpfFile(path) as bpf:
        layout = bpf.layout
        snapshot = bpf.extract_snapshot(0)
    field_names = [name for name in snapshot.fields if name != "node_position_cm"]
    field_units = {name: field_metadata_for_bpf_key(name).get("unit", "") for name in field_names}
    raw_field_map = {name: field_metadata_for_bpf_key(name).get("label", name) for name in field_names}
    alias_map: dict[str, str] = {}
    for canonical, alias in LOG_COMPATIBLE_BPF_ALIASES.items():
        if canonical in field_names and alias not in field_names:
            field_names.append(alias)
            field_units[alias] = field_units[canonical]
            raw_field_map[alias] = raw_field_map[canonical]
            alias_map[alias] = canonical
    return RawHeaderView(
        path,
        "bpf",
        field_names=field_names,
        field_units=field_units,
        raw_field_map=raw_field_map,
        alias_map=alias_map,
        geometry="planar",
        n_zones=int(layout.n_zones),
        n_snapshots=int(layout.n_snapshots),
        n_regions=1,
        n_materials=1,
        extra_metadata={
            "bpf_run_date": layout.run_date,
            "bpf_run_clock": layout.run_clock,
            "bpf_record_count": int(layout.record_count),
            "bpf_trailing_records": int(layout.trailing_records),
            "bpf_incomplete_final_snapshot": bool(layout.incomplete_final_snapshot),
        },
    )


class _RawRunRevision:
    """A cache key that is unique to one call.

    `load_run_data` keys its session cache on the file identity: resolved path,
    size and mtime. For a converted run that is the right key. For a raw deck it
    is not, because a caller can rewrite the deck, restore its size and mtime, and
    then reuse arrays parsed from the previous content. A key that never repeats
    removes the hazard outright; the cost is that a raw analysis is never served
    from the cache, which is what the CLI does anyway.
    """

    __slots__ = ("cache_key",)

    def __init__(self, path: Path, fmt: str) -> None:
        self.cache_key = ("raw_run", str(path), fmt, next(_RAW_RUN_SEQUENCE))


_RAW_RUN_SEQUENCE = itertools.count()


def load_raw_run_data(path: str | Path, *, requested_fields: Iterable[str] | None = None):
    """Build the derived-analysis dataset straight from a raw deck.

    `load_run_data` is written against the `HeliosRun` accessor surface; passing
    it a `RawRun` factory reuses its assembly, capability metadata and shape
    validation instead of duplicating them here.
    """
    from helios.services.derived.common import load_run_data

    if requested_fields is None:
        # The loader's None means "every field", so stack everything.
        run = open_any(path)
    else:
        run = open_any(path, fields=set(CORE_FIELDS) | {str(name) for name in requested_fields})
    revision = _RawRunRevision(run.path, run.format)
    return load_run_data(
        run.path,
        revision=revision,
        requested_fields=requested_fields,
        run_factory=lambda _source: run,
    )
