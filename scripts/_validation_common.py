from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helios.platform.registry import ArtifactRecord, DatasetRegistry, build_dataset_registry
from helios.runtime import RunContext
from helios.services.derived import DerivedAnalysisParameters, compute_analysis_result, load_run_data


VALIDATION_ROOT = REPO_ROOT / "outputs" / "validation_outputs"
REPORT_ROOT = REPO_ROOT / "outputs" / "reports"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def canonical_dataset_name(filename: str) -> str:
    stem = Path(filename).stem
    for suffix in (
        "_stabilized",
        "_extended",
        "_architecture",
        "_phase31_autoopen",
        "_phase31_manual",
        "_phase3_shell",
        "_ui_small",
        "_ui_medium",
        "_ui_large",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _artifact_priority(record: ArtifactRecord) -> tuple[int, int]:
    priority = 0
    if record.artifact_type == "stabilized_hdf5":
        priority -= 30
    if "stabilized" in record.filename.lower():
        priority -= 20
    if "outputs\\hdf5" not in record.path.lower():
        priority -= 10
    if record.source_kind == "generated":
        priority += 20
    return (priority, len(record.filename))


def build_registry() -> DatasetRegistry:
    registry = build_dataset_registry(REPO_ROOT)
    save_json(REPORT_ROOT / "dataset_registry.json", registry.to_dict())
    return registry


def preferred_hdf5_records(registry: DatasetRegistry) -> list[ArtifactRecord]:
    groups: dict[str, list[ArtifactRecord]] = {}
    for record in registry.records:
        if record.artifact_type != "stabilized_hdf5":
            continue
        if not record.directly_usable:
            continue
        groups.setdefault(canonical_dataset_name(record.filename), []).append(record)
    preferred: list[ArtifactRecord] = []
    for _, records in sorted(groups.items()):
        preferred.append(sorted(records, key=_artifact_priority)[0])
    return preferred


def context_from_dataset(dataset, *, snapshot_index: int = 0) -> RunContext:
    default_coordinate = "moving_radius" if dataset.radius_cm is not None else "static_x"
    all_region_ids = tuple(int(value) for value in sorted(set(np.asarray(dataset.zone_region_id, dtype=np.int32).tolist())))
    all_material_ids = tuple(int(value) for value in sorted(set(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)).tolist())))
    return RunContext(
        path=dataset.path,
        summary=dict(dataset.summary),
        metadata=dict(dataset.metadata),
        fields=(),
        diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(),
        static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(),
        zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
        has_dynamic_radius=dataset.radius_cm is not None,
        snapshot_index=int(snapshot_index),
        map_coordinate=default_coordinate,
        slice_coordinate=default_coordinate,
        selected_region_ids=all_region_ids,
        selected_material_ids=all_material_ids,
    )


def load_dataset_for_record(record: ArtifactRecord):
    return load_run_data(Path(record.path))


def compute_result_for_record(
    record: ArtifactRecord,
    *,
    snapshot_index: int | None = None,
    parameters: DerivedAnalysisParameters | None = None,
):
    dataset = load_dataset_for_record(record)
    context = context_from_dataset(dataset, snapshot_index=0 if snapshot_index is None else snapshot_index)
    context.set_snapshot_index(0 if snapshot_index is None else snapshot_index)
    active_parameters = parameters or DerivedAnalysisParameters()
    context_key = (
        "offline-validation",
        str(record.path),
        int(context.snapshot_index),
        active_parameters.key(),
    )
    result = compute_analysis_result(
        dataset,
        context,
        parameters=active_parameters,
        context_key=context_key,
    )
    return dataset, context, result


def interface_zone_positions(dataset) -> np.ndarray:
    regions = np.asarray(dataset.zone_region_id, dtype=np.int32)
    if regions.size <= 1:
        return np.empty(0, dtype=np.float64)
    changes = np.where(np.diff(regions) != 0)[0]
    return changes.astype(np.float64) + 1.5


def region_boundaries_for_profile(dataset, coordinate_values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    region_ids = np.asarray(dataset.zone_region_id, dtype=np.int32)
    coordinate_values = np.asarray(coordinate_values, dtype=np.float64)
    if mask is not None:
        active = np.asarray(mask, dtype=bool)
    else:
        active = np.ones(region_ids.shape, dtype=bool)
    positions: list[float] = []
    for idx in np.where(np.diff(region_ids) != 0)[0]:
        if not (active[idx] or active[idx + 1]):
            continue
        left = coordinate_values[idx]
        right = coordinate_values[idx + 1]
        if np.isfinite(left) and np.isfinite(right):
            positions.append(float(0.5 * (left + right)))
    return np.asarray(positions, dtype=np.float64)


def save_figure(path: Path, figure: plt.Figure) -> None:
    ensure_dir(path.parent)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_bundle_grid(path: Path, title: str, bundles, *, ncols: int = 2) -> None:
    bundles = tuple(bundles)
    if not bundles:
        return
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(bundles) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.7 * nrows), squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    for axis in axes_flat[len(bundles) :]:
        axis.axis("off")
    for axis, bundle in zip(axes_flat, bundles):
        x_values = np.asarray(bundle.x_values, dtype=np.float64)
        curve_names = bundle.curve_names if bundle.curve_names else tuple(f"series {idx + 1}" for idx in range(len(bundle.y_series)))
        for name, series in zip(curve_names, bundle.y_series):
            axis.plot(x_values, np.asarray(series, dtype=np.float64), linewidth=1.4, label=str(name))
        for boundary in bundle.boundary_positions:
            axis.axvline(float(boundary), color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.7)
        axis.set_title(bundle.title)
        axis.set_xlabel(bundle.x_label)
        axis.set_ylabel(bundle.y_label)
        if len(curve_names) > 1:
            axis.legend(fontsize=8)
    figure.suptitle(title)
    save_figure(path, figure)


def write_markdown_table(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    ensure_dir(path.parent)
    lines = [
        "| " + " | ".join(str(value) for value in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
