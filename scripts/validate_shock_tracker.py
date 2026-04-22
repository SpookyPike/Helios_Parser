from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _validation_common import (
    VALIDATION_ROOT,
    build_registry,
    compute_result_for_record,
    interface_zone_positions,
    preferred_hdf5_records,
    save_json,
    save_figure,
)


OUTPUT_DIR = VALIDATION_ROOT / "shock_tracker"


def _shock_summary(result) -> dict[str, object]:
    shock = result.shock
    smoothed_zone = np.asarray(shock.smoothed_zone_index, dtype=np.float64)
    signed_velocity = np.asarray(shock.velocity_cm_s, dtype=np.float64)
    speed_mag = np.asarray(shock.speed_magnitude_cm_s, dtype=np.float64)
    diffs = np.diff(smoothed_zone[np.isfinite(smoothed_zone)])
    jump_count = int(np.sum(np.abs(diffs) > 50.0)) if diffs.size else 0
    sign_flips = int(np.sum(np.diff(np.signbit(signed_velocity[np.isfinite(signed_velocity)])) != 0)) if np.any(np.isfinite(signed_velocity)) else 0
    return {
        "propagation_direction": shock.propagation_direction,
        "activation_snapshot_index": shock.activation_snapshot_index,
        "breakout_time_s": shock.breakout_time_s,
        "interface_crossings": [
            {
                "interface_label": crossing.interface_label,
                "boundary_zone": crossing.boundary_zone,
                "crossing_snapshot": crossing.crossing_snapshot,
                "crossing_time_s": crossing.crossing_time_s,
                "crossing_position_cm": crossing.crossing_position_cm,
            }
            for crossing in shock.interface_crossings
        ],
        "jump_count_gt_50_zones": jump_count,
        "signed_velocity_sign_flips": sign_flips,
        "max_speed_cm_s": float(np.nanmax(speed_mag)) if np.any(np.isfinite(speed_mag)) else float("nan"),
    }


def _plot_record(record, dataset, result) -> None:
    shock = result.shock
    time_ns = np.asarray(dataset.time_s, dtype=np.float64) * 1.0e9
    density = np.asarray(dataset.density_g_cm3, dtype=np.float64)
    zone_indices = np.arange(1, density.shape[1] + 1, dtype=np.float64)
    detector = np.asarray(shock.detector_score, dtype=np.float64)
    smoothed_zone = np.asarray(shock.smoothed_zone_index, dtype=np.float64)
    raw_zone = np.asarray(shock.zone_index, dtype=np.float64)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    image = axes[0, 0].imshow(
        np.log10(np.maximum(density.T, 1.0e-30)),
        aspect="auto",
        origin="lower",
        extent=(float(time_ns[0]), float(time_ns[-1]), float(zone_indices[0]), float(zone_indices[-1])),
        cmap="turbo",
    )
    axes[0, 0].plot(time_ns, raw_zone, color="white", linewidth=1.0, alpha=0.55, label="raw track")
    axes[0, 0].plot(time_ns, smoothed_zone, color="cyan", linewidth=1.6, label="smoothed track")
    for interface_zone in interface_zone_positions(dataset):
        axes[0, 0].axhline(interface_zone, color="w", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[0, 0].set_title(f"{record.filename} | density map + shock track")
    axes[0, 0].set_xlabel("Time [ns]")
    axes[0, 0].set_ylabel("Zone index")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    figure.colorbar(image, ax=axes[0, 0], label="log10 density [g/cm3]")

    axes[0, 1].plot(time_ns, detector, color="#2563eb", linewidth=1.5)
    axes[0, 1].set_title("Shock detector score")
    axes[0, 1].set_xlabel("Time [ns]")
    axes[0, 1].set_ylabel("Score [arb]")

    axes[1, 0].plot(time_ns, raw_zone, color="#f59e0b", linewidth=1.0, alpha=0.6, label="raw zone")
    axes[1, 0].plot(time_ns, smoothed_zone, color="#0f766e", linewidth=1.6, label="smoothed zone")
    if shock.activation_snapshot_index is not None and 0 <= shock.activation_snapshot_index < time_ns.size:
        axes[1, 0].axvline(time_ns[int(shock.activation_snapshot_index)], color="#dc2626", linestyle=":", linewidth=1.2, label="activation")
    axes[1, 0].set_title(f"Tracked zone vs time ({shock.propagation_direction})")
    axes[1, 0].set_xlabel("Time [ns]")
    axes[1, 0].set_ylabel("Zone index")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].plot(time_ns, np.asarray(shock.speed_magnitude_cm_s) * 1.0e-5, color="#7c3aed", linewidth=1.6, label="|v|")
    axes[1, 1].plot(time_ns, np.asarray(shock.velocity_cm_s) * 1.0e-5, color="#dc2626", linewidth=1.1, alpha=0.85, label="signed v")
    axes[1, 1].set_title("Shock speed")
    axes[1, 1].set_xlabel("Time [ns]")
    axes[1, 1].set_ylabel("Velocity [km/s]")
    axes[1, 1].legend(loc="upper right", fontsize=8)

    save_figure(OUTPUT_DIR / f"{Path(record.filename).stem}_shock_validation.png", figure)


def main() -> int:
    registry = build_registry()
    results: dict[str, object] = {}
    for record in preferred_hdf5_records(registry):
        dataset, _context, result = compute_result_for_record(record)
        _plot_record(record, dataset, result)
        results[record.filename] = _shock_summary(result)
    save_json(OUTPUT_DIR / "summary.json", results)
    print(f"Validated shock tracker on {len(results)} datasets -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
