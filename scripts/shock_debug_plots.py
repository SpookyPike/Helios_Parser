"""Generate offline shock-tracker debug plots for representative HELIOS runs.

This script is intentionally independent of the GUI. Phase 4.1 requires shock
logic to be inspected offline before it is trusted in Derived mode. The plots
produced here summarize:

- detector score vs time
- tracked position vs time
- tracked interface index vs time
- raw and smoothed shock velocity vs time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from helios.runtime import RunContext
from helios.services.derived.analysis import DerivedAnalysisParameters
from helios.services.derived.common import load_run_data
from helios.services.derived.selection import build_analysis_geometry
from helios.services.derived.shock_tracking import track_shock_front


DEFAULT_RUNS = (
    Path("outputs/hdf5/5Fe+4.9TW+light_stabilized.h5"),
    Path("outputs/hdf5/Cu_0166_stabilized.h5"),
    Path("outputs/hdf5/10ns+10Si+60Al+15Si+4.27TW_stabilized.h5"),
)


def _run_context_from_dataset(path: Path, dataset) -> RunContext:
    return RunContext(
        path=path,
        summary=dict(dataset.summary),
        metadata=dict(dataset.metadata),
        fields=("density", "velocity", "temperature_e", "temperature_i", "electron_density", "mean_charge"),
        diagnostics=(),
        time_values=np.asarray(dataset.time_s, dtype=np.float64).copy(),
        static_x_values=np.asarray(dataset.static_x_cm, dtype=np.float64).copy(),
        zone_region_id=np.asarray(dataset.zone_region_id, dtype=np.int32).copy(),
        zone_material_index=np.asarray(dataset.zone_material_index, dtype=np.int32).copy(),
        has_dynamic_radius=dataset.radius_cm is not None,
        snapshot_index=0,
        map_coordinate="moving_radius" if dataset.radius_cm is not None else "static_x",
        slice_coordinate="moving_radius" if dataset.radius_cm is not None else "zone",
        selected_region_ids=tuple(int(value) for value in np.asarray(dataset.regions["region_index"], dtype=np.int32)),
        selected_material_ids=tuple(int(abs(value)) for value in np.unique(np.abs(np.asarray(dataset.zone_material_index, dtype=np.int32)))),
    )


def generate_plot(path: Path, output_dir: Path) -> Path:
    dataset = load_run_data(path)
    context = _run_context_from_dataset(path, dataset)
    parameters = DerivedAnalysisParameters()
    geometry = build_analysis_geometry(
        dataset,
        context,
        observation_side=parameters.observation_side,
        line_of_sight_angle_deg=parameters.line_of_sight_angle_deg,
        profile_coordinate_mode=parameters.profile_coordinate_mode,
    )
    shock = track_shock_front(dataset, context, parameters=parameters, geometry=geometry)

    time_ns = np.asarray(shock.time_s, dtype=np.float64) * 1.0e9
    position_um = np.asarray(shock.position_cm, dtype=np.float64) * 1.0e4
    smoothed_position_um = np.asarray(shock.smoothed_position_cm, dtype=np.float64) * 1.0e4
    raw_index = np.asarray(shock.zone_index, dtype=np.float64)
    smooth_index = np.asarray(shock.smoothed_zone_index, dtype=np.float64)
    signed_velocity_km_s = np.asarray(shock.velocity_cm_s, dtype=np.float64) * 1.0e-5
    speed_km_s = np.asarray(shock.speed_magnitude_cm_s, dtype=np.float64) * 1.0e-5
    detector = np.asarray(shock.detector_score, dtype=np.float64)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(time_ns, detector, color="tab:red", lw=2)
    axes[0, 0].set_title("Detector score vs time")
    axes[0, 0].set_xlabel("Time [ns]")
    axes[0, 0].set_ylabel("Score")

    axes[0, 1].plot(time_ns, position_um, color="tab:blue", lw=1.5, alpha=0.55, label="raw")
    axes[0, 1].plot(time_ns, smoothed_position_um, color="tab:orange", lw=2.0, label="smoothed")
    axes[0, 1].set_title("Shock position vs time")
    axes[0, 1].set_xlabel("Time [ns]")
    axes[0, 1].set_ylabel("Position [um]")
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(time_ns, raw_index + 1.0, color="tab:green", lw=1.5, alpha=0.55, label="raw")
    axes[1, 0].plot(time_ns, smooth_index + 1.0, color="tab:purple", lw=2.0, label="smoothed")
    axes[1, 0].set_title("Tracked shock zone index vs time")
    axes[1, 0].set_xlabel("Time [ns]")
    axes[1, 0].set_ylabel("Zone / interface index")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(time_ns, signed_velocity_km_s, color="tab:cyan", lw=1.5, label="signed v")
    axes[1, 1].plot(time_ns, speed_km_s, color="tab:brown", lw=2.0, label="|v|")
    axes[1, 1].set_title("Shock velocity vs time")
    axes[1, 1].set_xlabel("Time [ns]")
    axes[1, 1].set_ylabel("Velocity [km/s]")
    axes[1, 1].legend(loc="best")

    for crossing in shock.interface_crossings:
        if crossing.crossing_time_s is None:
            continue
        crossing_ns = float(crossing.crossing_time_s) * 1.0e9
        axes[0, 1].axvline(crossing_ns, color="#64748b", lw=1.0, ls="--", alpha=0.7)
        axes[1, 0].axvline(crossing_ns, color="#64748b", lw=1.0, ls="--", alpha=0.7)

    figure.suptitle(f"{path.name} | {shock.method} | direction={shock.propagation_direction}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_shock_debug.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline shock debug plots for HELIOS runs.")
    parser.add_argument("paths", nargs="*", help="Optional stabilized HDF5 paths. Defaults to representative runs.")
    parser.add_argument("--output-dir", default="outputs/derived_debug", help="Output directory for PNG plots.")
    args = parser.parse_args()

    paths = tuple(Path(path) for path in args.paths) if args.paths else DEFAULT_RUNS
    output_dir = Path(args.output_dir)
    for path in paths:
        output = generate_plot(path, output_dir)
        print(output)


if __name__ == "__main__":
    main()
