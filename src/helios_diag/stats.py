"""``helios_diag stats``: the quick diagnostic.

Three questions, one field at a time:

- ``--over zones``: what did each zone do across the whole run, and when did it
  peak. The peak time is the time breakout: on the reference run it separates
  zone 900 at 4.69e-9 s from zone 300 at 1.14e-8 s.
- ``--over time``: what did the field look like at each snapshot, summarised
  across zones.
- ``--at`` / ``--at-time``: the profile at one moment.
- ``--global``: every statistic over the whole field.

Two correctness rules that are easy to get wrong:

- ``argmax`` on a constant zone returns index 0, which reads as "peaked at
  t=0". A zone whose spread is inside the flat tolerance reports no peak.
- Fields are written with ``fillvalue=np.nan``, so every reduction is NaN-aware
  when the array actually contains NaN. When it does not, the plain reductions
  are used: measured 31.2 ms against 71.0 ms for the per-zone set.
"""

from __future__ import annotations

import argparse
from typing import Any

from . import envelope

MODE_OVER_ZONES = "over_zones"
MODE_OVER_TIME = "over_time"
MODE_AT = "at_snapshot"
MODE_GLOBAL = "global"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="Converted HDF5 run.")
    parser.add_argument("-f", "--field", required=True, help="Field to summarise, e.g. density.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--over", choices=("zones", "time"), help="Reduce over zones (default) or over snapshots.")
    mode.add_argument("--at", type=int, metavar="SNAPSHOT", help="Report the profile at this snapshot index.")
    mode.add_argument("--at-time", type=float, metavar="SECONDS", help="Report the profile at the nearest snapshot to this time.")
    mode.add_argument("--global", dest="global_stats", action="store_true", help="Report one row over the whole field.")
    parser.add_argument("--quantiles", default="5,50,95", help="Comma-separated percentiles. Default 5,50,95.")
    parser.add_argument("--region", type=int, help="Restrict to one region id.")
    parser.add_argument("--material", type=int, help="Restrict to one material id.")
    parser.add_argument("--by-region", action="store_true", help="One aggregate row per region.")
    parser.add_argument("--by-material", action="store_true", help="One aggregate row per material.")
    parser.add_argument("--flat-atol", type=float, default=1e-12, help="Absolute spread below which a zone counts as flat.")
    parser.add_argument("--flat-rtol", type=float, default=1e-9, help="Relative spread below which a zone counts as flat.")


def _parse_quantiles(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--quantiles needs at least one percentile")
    if any(value < 0 or value > 100 for value in values):
        raise ValueError("--quantiles values must lie between 0 and 100")
    return values


def _column_name(value: float) -> str:
    return f"p{int(value):02d}" if float(value).is_integer() else f"p{value:g}"


def _reduce(block, axis: int | None, quantiles: list[float], clean: bool) -> dict[str, Any]:
    """Reduce with plain numpy when the data is clean, NaN-aware when it is not."""
    import numpy as np  # local: keeps --help free of numpy

    if clean:
        return {
            "mean": np.mean(block, axis=axis),
            "min": np.min(block, axis=axis),
            "max": np.max(block, axis=axis),
            "std": np.std(block, axis=axis),
            "peak_index": np.argmax(block, axis=axis) if axis is not None else None,
            "quantiles": np.percentile(block, quantiles, axis=axis),
        }
    return {
        "mean": np.nanmean(block, axis=axis),
        "min": np.nanmin(block, axis=axis),
        "max": np.nanmax(block, axis=axis),
        "std": np.nanstd(block, axis=axis),
        "peak_index": np.nanargmax(block, axis=axis) if axis is not None else None,
        "quantiles": np.nanpercentile(block, quantiles, axis=axis),
    }


def _accounting(block, clean: bool) -> dict[str, int]:
    """Reuse the NaN probe the reduction already did instead of walking the array again."""
    import numpy as np  # local

    total = int(block.size)
    if clean or block.dtype.kind != "f":
        return {"n_values": total, "n_nan": 0, "n_used": total}
    nan = int(np.isnan(block).sum())
    return {"n_values": total, "n_nan": nan, "n_used": total - nan}


def _rows_from_columns(columns_data: dict[str, list]) -> list[dict[str, Any]]:
    """Transpose column lists into row dicts in one zip, not one dict per cell."""
    keys = list(columns_data)
    return [dict(zip(keys, values, strict=True)) for values in zip(*columns_data.values(), strict=True)]


def _peak_times(peak_index, low, high, times, flat_atol: float, flat_rtol: float) -> list[float | None]:
    """Peak time per zone, with flat zones reported as no peak rather than t=0."""
    import numpy as np  # local

    spread = np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64)
    magnitude = np.abs(np.asarray(high, dtype=np.float64))
    flat = spread <= np.maximum(flat_atol, flat_rtol * magnitude)
    out: list[float | None] = []
    for index, is_flat in zip(np.asarray(peak_index), flat, strict=True):
        out.append(None if bool(is_flat) else float(times[int(index)]))
    return out


def _zone_mask(run, region: int | None, material: int | None):
    import numpy as np  # local

    mask = None
    if region is not None:
        mask = run.get_region_mask(region)
    if material is not None:
        material_mask = run.get_material_mask(material)
        mask = material_mask if mask is None else (mask & material_mask)
    if mask is None:
        return None
    if not bool(np.any(mask)):
        raise ValueError(f"region {region} / material {material} selects no zones")
    return mask


def _row_for_zones(name, values, quantiles, columns) -> dict[str, Any]:
    row: dict[str, Any] = {"zone": int(name)}
    row.update(values)
    return row


def run(args: argparse.Namespace) -> envelope.Result:
    import numpy as np  # local: the whole command needs numpy, --help does not

    from helios_parser import HeliosRun

    quantiles = _parse_quantiles(args.quantiles)
    quantile_columns = [_column_name(value) for value in quantiles]
    run_handle = HeliosRun(args.path)
    try:
        file_block, file_notes = envelope.file_block(run_handle)
        field_block, field_notes = envelope.field_block(run_handle, args.field)
        notes = file_notes + field_notes
        times = np.asarray(run_handle.get_time(), dtype=np.float64)
        mask = _zone_mask(run_handle, args.region, args.material)

        mode = MODE_OVER_ZONES
        if args.over == "time":
            mode = MODE_OVER_TIME
        elif args.at is not None or args.at_time is not None:
            mode = MODE_AT
        elif args.global_stats:
            mode = MODE_GLOBAL

        if args.by_region or args.by_material:
            return _grouped(run_handle, args, mode, quantiles, quantile_columns, file_block, field_block, notes)

        if mode == MODE_AT:
            if args.at is not None:
                snapshot = int(args.at)
            else:
                snapshot = int(np.argmin(np.abs(times - float(args.at_time))))
                notes.append(f"--at-time {args.at_time:g} resolved to snapshot {snapshot} at t={times[snapshot]:g} s")
            profile = np.asarray(run_handle.get_snapshot_field(args.field, snapshot), dtype=np.float64)
            if mask is not None:
                profile = profile[mask]
            zones = np.nonzero(mask)[0] if mask is not None else np.arange(profile.size)
            profile_clean = not bool(np.isnan(profile).any())
            rows = _rows_from_columns({"zone": zones.tolist(), "value": profile.tolist()})
            payload = envelope.envelope(
                "stats",
                file=file_block,
                notes=notes,
                field=field_block,
                request={
                    "mode": MODE_AT,
                    "snapshot": snapshot,
                    "time_s": float(times[snapshot]),
                    "axis": "zone",
                    "reduction": "none, one value per zone",
                },
                accounting=_accounting(profile, profile_clean),
            )
            return envelope.Result(payload, rows, ("zone", "value"))

        data = np.asarray(run_handle.get_field(args.field), dtype=np.float64)
        if mask is not None:
            data = data[:, mask]
        clean = not bool(np.isnan(data).any())
        if not clean:
            notes.append("the field contains NaN; every statistic skips those values, see accounting")

        if mode == MODE_GLOBAL:
            flat = data.reshape(-1)
            stats = _reduce(flat, None, quantiles, clean)
            row: dict[str, Any] = {
                "mean": float(stats["mean"]),
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "std": float(stats["std"]),
            }
            for column, value in zip(quantile_columns, np.atleast_1d(stats["quantiles"]), strict=True):
                row[column] = float(value)
            payload = envelope.envelope(
                "stats",
                file=file_block,
                notes=notes,
                field=field_block,
                request={
                    "mode": MODE_GLOBAL,
                    "axis": "none",
                    "reduction": f"over all {data.shape[0]} snapshots and {data.shape[1]} zones",
                    "quantiles": quantiles,
                },
                accounting=_accounting(data, clean),
            )
            columns = ("mean", "min", "max", "std", *quantile_columns)
            return envelope.Result(payload, [row], columns)

        axis = 0 if mode == MODE_OVER_ZONES else 1
        stats = _reduce(data, axis, quantiles, clean)
        columns = (
            ("zone", "mean", "min", "max", *quantile_columns, "std", "peak_time_s")
            if mode == MODE_OVER_ZONES
            else ("snapshot", "time_s", "mean", "min", "max", *quantile_columns, "std")
        )

        # Column lists first, then one transpose: per-cell float() calls cost more
        # than the reductions do on a 1200-row table.
        column_data: dict[str, list] = {}
        if mode == MODE_OVER_ZONES:
            peaks = _peak_times(
                stats["peak_index"], stats["min"], stats["max"], times, args.flat_atol, args.flat_rtol
            )
            flat_count = sum(1 for value in peaks if value is None)
            if flat_count:
                notes.append(
                    f"{flat_count} zones are flat within tolerance and report no peak time; "
                    "argmax would have said t=0 for them"
                )
            column_data["zone"] = list(range(data.shape[1]))
            column_data["peak_time_s"] = peaks
            request = {
                "mode": MODE_OVER_ZONES,
                "axis": "zone",
                "reduction": f"over all {data.shape[0]} snapshots",
                "quantiles": quantiles,
            }
        else:
            column_data["snapshot"] = list(range(data.shape[0]))
            column_data["time_s"] = times.tolist()
            request = {
                "mode": MODE_OVER_TIME,
                "axis": "snapshot",
                "reduction": f"over all {data.shape[1]} zones",
                "quantiles": quantiles,
            }
        for key in ("mean", "min", "max"):
            column_data[key] = np.asarray(stats[key]).tolist()
        for column, values in zip(quantile_columns, stats["quantiles"], strict=True):
            column_data[column] = np.asarray(values).tolist()
        column_data["std"] = np.asarray(stats["std"]).tolist()
        # Keep the JSON key order identical to the declared column order, so a human
        # reading the payload sees the same table the csv and table writers produce.
        column_data = {key: column_data[key] for key in columns}
        rows = _rows_from_columns(column_data)

        payload = envelope.envelope(
            "stats",
            file=file_block,
            notes=notes,
            field=field_block,
            request=request,
            accounting=_accounting(data, clean),
        )
        return envelope.Result(payload, rows, columns)
    finally:
        run_handle.close()


def _grouped(run_handle, args, mode, quantiles, quantile_columns, file_block, field_block, notes):
    """Aggregate per region or per material, over the whole run.

    A campaign question is "how hot did region 2 get", not "what did each zone
    do", so each group collapses both axes into one row.
    """
    import numpy as np  # local

    if mode == MODE_AT:
        raise ValueError("--by-region and --by-material cannot be combined with --at or --at-time")

    data = np.asarray(run_handle.get_field(args.field), dtype=np.float64)
    clean = not bool(np.isnan(data).any())
    by_region = bool(args.by_region)
    key = "region" if by_region else "material"
    ids = run_handle.list_region_ids() if by_region else run_handle.list_material_ids()
    if not ids:
        raise ValueError("this run declares no regions or materials to group by")

    rows = []
    for identifier in ids:
        mask = run_handle.get_region_mask(int(identifier)) if by_region else run_handle.get_material_mask(int(identifier))
        if not bool(np.any(mask)):
            continue
        stats = _reduce(data[:, mask], None, quantiles, clean)
        row: dict[str, Any] = {
            key: int(identifier),
            "n_zones": int(mask.sum()),
            "mean": float(stats["mean"]),
            "min": float(stats["min"]),
            "max": float(stats["max"]),
            "std": float(stats["std"]),
        }
        for column, value in zip(quantile_columns, np.atleast_1d(stats["quantiles"]), strict=True):
            row[column] = float(value)
        rows.append(row)

    payload = envelope.envelope(
        "stats",
        file=file_block,
        notes=notes,
        field=field_block,
        request={
            "mode": mode,
            "axis": key,
            "reduction": f"over all {data.shape[0]} snapshots, aggregated per {key}",
            "quantiles": quantiles,
        },
        accounting=_accounting(data, clean),
    )
    return envelope.Result(payload, rows, (key, "n_zones", "mean", "min", "max", *quantile_columns, "std"))