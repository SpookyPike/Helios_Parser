from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _unit(dataset: h5py.Dataset) -> str:
    return str(dataset.attrs.get("units") or dataset.attrs.get("unit") or "")


def _load(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        data = {
            "time": h5["time/time"][:],
            "x0": h5["grid/x"][:],
            "radius": h5["fields/radius"][:],
            "density": h5["fields/density"][:],
            "temperature_e": h5["fields/temperature_e"][:],
            "pressure": h5["fields/pressure"][:],
            "velocity": h5["fields/velocity"][:],
            "time_unit": _unit(h5["time/time"]),
            "x0_unit": _unit(h5["grid/x"]),
            "radius_unit": _unit(h5["fields/radius"]),
            "density_unit": _unit(h5["fields/density"]),
            "temperature_e_unit": _unit(h5["fields/temperature_e"]),
            "pressure_unit": _unit(h5["fields/pressure"]),
            "velocity_unit": _unit(h5["fields/velocity"]),
        }
    return data


def _pseudocolor(
    time: np.ndarray,
    x: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    mesh = ax.pcolormesh(x, time, values, shading="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _lineout_panels(
    x: np.ndarray,
    curves: list[tuple[np.ndarray, str, str]],
    *,
    title: str,
    xlabel: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(curves),
        ncols=1,
        figsize=(8.5, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    for ax, (values, label, ylabel) in zip(np.atleast_1d(axes), curves):
        ax.plot(x, values, linewidth=2.0)
        ax.set_ylabel(ylabel)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_file(input_path: Path, output_dir: Path, snapshot_index: int | None = None) -> list[Path]:
    data = _load(input_path)
    time = data["time"]
    x0 = data["x0"]
    radius = data["radius"]

    if snapshot_index is None:
        snapshot_index = len(time) // 2

    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    time_label = f"time ({data['time_unit']})" if data["time_unit"] else "time"
    x0_label = f"initial x ({data['x0_unit']})" if data["x0_unit"] else "initial x"
    radius_label = (
        f"zone position at snapshot ({data['radius_unit']})"
        if data["radius_unit"]
        else "zone position at snapshot"
    )

    contours = [
        ("density", "Density", data["density_unit"], data["density"]),
        ("temperature_e", "Electron Temperature", data["temperature_e_unit"], data["temperature_e"]),
        ("velocity", "Velocity", data["velocity_unit"], data["velocity"]),
    ]
    for key, label, unit, values in contours:
        output = output_dir / f"{stem}_{key}_contour.png"
        _pseudocolor(
            time,
            x0,
            values,
            title=f"{stem}: {label} vs time and initial x",
            xlabel=x0_label,
            ylabel=time_label,
            colorbar_label=f"{label} ({unit})" if unit else label,
            output=output,
        )
        outputs.append(output)

    lineout_output = output_dir / f"{stem}_lineout_snapshot_{snapshot_index:04d}.png"
    lineout_x = radius[snapshot_index]
    curves = [
        (
            data["density"][snapshot_index],
            "Density lineout",
            f"density ({data['density_unit']})" if data["density_unit"] else "density",
        ),
        (
            data["temperature_e"][snapshot_index],
            "Electron temperature lineout",
            (
                f"temperature_e ({data['temperature_e_unit']})"
                if data["temperature_e_unit"]
                else "temperature_e"
            ),
        ),
        (
            data["pressure"][snapshot_index],
            "Pressure lineout",
            f"pressure ({data['pressure_unit']})" if data["pressure_unit"] else "pressure",
        ),
    ]
    _lineout_panels(
        lineout_x,
        curves,
        title=f"{stem}: density / temperature_e / pressure at snapshot {snapshot_index}",
        xlabel=radius_label,
        output=lineout_output,
    )
    outputs.append(lineout_output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demonstration plots from HELIOS HDF5 output.")
    parser.add_argument("input", nargs="+", help="Input HDF5 files.")
    parser.add_argument("--output-dir", default="outputs/plots", help="Directory for saved plots.")
    parser.add_argument(
        "--snapshot-index",
        type=int,
        default=None,
        help="Snapshot index for the 1D lineout. Defaults to the middle snapshot.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for input_name in args.input:
        for plot_path in plot_file(Path(input_name), output_dir, snapshot_index=args.snapshot_index):
            print(plot_path)


if __name__ == "__main__":
    main()
