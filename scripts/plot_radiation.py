"""Plot the boundary spectra a frequency-resolved HELIOS run carries.

WHY THIS EXISTS.  The converted HDF5 holds the radiation field, but nothing in
the app draws it yet: the Viewer plots a 2-D (time, zone) field and a spectrum is
3-D (dump, zone, group).  Until a Viewer surface exists, this script answers the
two questions the campaign actually asks of the data:

    python scripts/plot_radiation.py run.h5 --out outputs/radiation
    python scripts/plot_radiation.py run.h5 --side rmin --snapshot 3

It writes two figures:

- `spectrum_<side>.png`: the spectrum at one dump, photon energy against flux,
  with the sample time in the title so the requested time is never confused with
  the sampled one.
- `spectrum_map_<side>.png`: photon energy against time, which is the contour the
  requirement document calls use case D.

The data comes from `flux/`, i.e. the `time_whole` axis.  The node-centred
spectral field on `radiation/` has its own axis and its own figure
(`energy_density_map.png`): photon energy against time for one zone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _script_bootstrap import ROOT  # noqa: E402

from helios.services.derived.radiation import (  # noqa: E402
    SIDE_DATASETS,
    spectrum_at,
    spectrum_map,
)

DEFAULT_OUTPUT = ROOT / "outputs" / "radiation"


def _unit(handle: h5py.File, path: str) -> str:
    return str(handle[path].attrs.get("unit", "") or "")


def _time_edges(times: np.ndarray) -> np.ndarray:
    """Cell edges for a pcolormesh over a non-uniform time axis."""
    if times.size == 1:
        half = max(abs(float(times[0])) * 1e-3, 1e-15) / 2.0
        return np.asarray([times[0] - half, times[0] + half])
    midpoints = 0.5 * (times[1:] + times[:-1])
    return np.concatenate(([times[0] - (midpoints[0] - times[0])], midpoints, [times[-1] + (times[-1] - midpoints[-1])]))


def _grid_moves(edges: np.ndarray) -> bool:
    """True when the per-dump photon grid is not identical across dumps.

    Regrouping moves the edges, and a `pcolormesh` has one y axis, so a map built
    from `edges[0]` would draw every later dump at the first dump's energies.
    """
    return edges.shape[0] > 1 and not np.array_equal(edges, np.broadcast_to(edges[:1], edges.shape))


def _spectral_panel(
    figure,
    axes,
    times: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    *,
    cmap: str,
    title: str,
    x_label: str,
    colour_label: str,
) -> None:
    """Draw a photon-energy map, or a per-dump waterfall when the grid moves."""
    magnitude = np.abs(values)
    positive = magnitude[magnitude > 0]
    floor = float(positive.min()) if positive.size else 1.0
    scaled = np.log10(np.maximum(magnitude, floor))
    if _grid_moves(edges):
        colours = plt.get_cmap(cmap)(np.linspace(0.15, 0.92, values.shape[0]))
        for row in range(values.shape[0]):
            centre = np.sqrt(edges[row, :-1] * edges[row, 1:])
            axes.plot(centre, scaled[row], color=colours[row], lw=1.1, label=f"{times[row]:.4g}")
        axes.set_xscale("log")
        axes.set_xlabel("photon energy [eV]")
        axes.set_ylabel(colour_label)
        axes.set_title(f"{title.split(',')[0]} (per-dump grid, one line per dump)")
        axes.legend(fontsize=6, title=x_label, ncol=2)
        return
    mesh = axes.pcolormesh(
        _time_edges(times),
        edges[0],
        scaled.T,
        shading="flat",
        cmap=cmap,
    )
    axes.set_yscale("log")
    axes.set_xlabel(x_label)
    axes.set_ylabel("photon energy [eV]")
    axes.set_title(title)
    colour = figure.colorbar(mesh, ax=axes)
    colour.set_label(colour_label)


def _dataset_for(side: str, quantity: str) -> str:
    return "net_interface" if quantity == "net" else SIDE_DATASETS[side]


def _plot_spectrum(target: Path, path: Path, side: str, quantity: str, time_s: float) -> float:
    sample_time, edges, spectrum = spectrum_at(path, time_s, side=side, quantity=quantity)
    with h5py.File(path, "r") as handle:
        flux_unit = _unit(handle, f"flux/{_dataset_for(side, quantity)}")
        time_unit = _unit(handle, "flux/time")
    centre = np.sqrt(edges[:-1] * edges[1:])
    figure, axes = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    try:
        axes.plot(centre, spectrum, lw=1.2, color="#1f5fa8")
        axes.set_xscale("log")
        axes.set_yscale("symlog", linthresh=max(float(np.abs(spectrum).max()) * 1e-9, 1e-30))
        axes.set_xlabel("photon energy [eV]")
        axes.set_ylabel(f"{quantity} flux at {side} [{flux_unit}]")
        axes.set_title(f"{side} {quantity} spectrum, sampled at {sample_time:.4g} {time_unit or 's'}")
        axes.grid(True, which="both", alpha=0.25, lw=0.5)
        # A plain group sum, not a width-weighted integral: the boundary spectra
        # are per-group bin contents (the sum reproduces TotalFluxRmin/Rmax
        # exactly), so weighting by dE would overstate the total by ~1e4.
        axes.text(
            0.02,
            0.96,
            f"group sum: {float(np.sum(spectrum)):.4g}",
            transform=axes.transAxes,
            va="top",
            fontsize=8,
            color="#444444",
        )
        figure.tight_layout()
        figure.savefig(target)
    finally:
        plt.close(figure)
    return sample_time


def _plot_spectrum_map(target: Path, path: Path, side: str, quantity: str) -> None:
    time_s, edges, values = spectrum_map(path, side=side, quantity=quantity)
    with h5py.File(path, "r") as handle:
        flux_unit = _unit(handle, f"flux/{_dataset_for(side, quantity)}")
        time_unit = _unit(handle, "flux/time")
    # The flux spans many decades, so the colour axis is the log of the magnitude.
    figure, axes = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    try:
        _spectral_panel(
            figure,
            axes,
            time_s,
            edges,
            values,
            cmap="magma",
            title=f"{side} {quantity} spectrum versus time [{flux_unit}], log10 magnitude",
            x_label=f"time [{time_unit or 's'}]",
            colour_label="log10 |flux|",
        )
        figure.tight_layout()
        figure.savefig(target)
    finally:
        plt.close(figure)


def _plot_energy_density_map(target: Path, path: Path, zone: int) -> None:
    with h5py.File(path, "r") as handle:
        if "radiation" not in handle:
            return
        values = np.asarray(handle["radiation/energy_density"][()], dtype=np.float64)
        times = np.asarray(handle["radiation/time"][()], dtype=np.float64)
        edges = np.asarray(handle["radiation/group_edges"][()], dtype=np.float64)
        unit = _unit(handle, "radiation/energy_density")
        time_unit = _unit(handle, "radiation/time")
    zone = int(np.clip(zone, 0, values.shape[1] - 1))
    figure, axes = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    try:
        _spectral_panel(
            figure,
            axes,
            times,
            edges,
            values[:, zone, :],
            cmap="viridis",
            title=f"zone {zone + 1} spectral energy density [{unit}], log10",
            x_label=f"time [{time_unit or 's'}]",
            colour_label=f"log10 [{unit}]",
        )
        figure.tight_layout()
        figure.savefig(target)
    finally:
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot the radiation output of a converted HELIOS run.")
    parser.add_argument("run", type=Path, help="stabilized HDF5 produced from a frequency-resolved .exo")
    parser.add_argument("--side", choices=sorted(SIDE_DATASETS), default="rmax")
    parser.add_argument(
        "--quantity",
        choices=("net", "incident"),
        default="net",
        help="net boundary flux (default; the incident arrays are zero in the reference runs)",
    )
    parser.add_argument("--snapshot", type=int, default=-1, help="flux row for the single-spectrum figure")
    parser.add_argument("--time", type=float, default=None, help="sample time in seconds instead of a row index")
    parser.add_argument("--zone", type=int, default=0, help="1-based zone for the energy-density figure")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    path = Path(args.run)
    if not path.is_file():
        parser.error(f"{path} does not exist")
    args.out.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as handle:
        if "flux" not in handle:
            parser.error(
                f"{path} carries no radiation output (metadata/radiation_output is not 'on'); "
                "convert a run whose deck set `Write freq-dependent radiation output = 1`."
            )
        times = np.asarray(handle["flux/time"][()], dtype=np.float64)
    if args.time is not None:
        requested_time = float(args.time)
    else:
        index = args.snapshot if args.snapshot >= 0 else times.size - 1
        if not 0 <= index < times.size:
            parser.error(f"snapshot {index} is outside 0..{times.size - 1}")
        requested_time = float(times[index])

    stem = f"{args.side}_{args.quantity}"
    spectrum_path = args.out / f"spectrum_{stem}.png"
    sample_time = _plot_spectrum(spectrum_path, path, args.side, args.quantity, requested_time)
    _plot_spectrum_map(args.out / f"spectrum_map_{stem}.png", path, args.side, args.quantity)
    _plot_energy_density_map(args.out / "energy_density_map.png", path, max(0, args.zone - 1))

    print(f"sampled {stem} at {sample_time:.6g} s")
    print(f"wrote {spectrum_path}")
    print(f"wrote {args.out / f'spectrum_map_{stem}.png'}")
    print(f"wrote {args.out / 'energy_density_map.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())